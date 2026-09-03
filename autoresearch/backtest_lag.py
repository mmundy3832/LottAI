"""
backtest_lag.py - Design B backtest for lag_analysis, fixed models over two windows.

Window 1 (test split): train Config A and Config B ONCE on pick3_combined.csv
  rows before the held-out test split (train+val, rows [0, _VAL_END) = 13,236
  draws, through 2024-04-03). Predict every draw in the test split
  (2024-04-04 .. 2026-02-13, 2,336 draws) -- the same split prepare_v3.py's
  own get_train_val_test() defines and the experiments were scored on.

Window 2 (live): train Config A and Config B ONCE on the full historical file
  (all 15,572 rows, through 2026-02-13). Predict every draw in
  pick3all_live.csv (2026-02-14 .. 2026-08-26, 662 rows as of this run), with
  features computed rolling forward through the live sequence -- each live
  draw's features use only historical rows plus already-observed live draws
  before it, never later ones.

For each window, prediction records are written in the schema lag_analysis.py
consumes (predicted_draw_date, draw_slot, top20_combos, consensus_combos,
actual_combo) to backtest_predictions_{test,live}.jsonl, and the same
lag-analysis algorithm from lag_analysis.py (lags 0..5, exact hit / >=2-of-3
positional match / per-position digit coverage, 2000-permutation null,
seed 42) is run separately on each window's records, writing
backtest_lag_output_{test,live}.json.

Deviations from predict_now.py's exact per-draw logic (see final report for
full disclosure):
  1. predict_now._run_ensemble is single-prediction-row only (it hardcodes
     shape (1, 10) for the probability arrays). Here models are fit ONCE per
     config/window and predict_proba is called in a batch over every target
     row in that window. This is numerically identical to calling
     _run_ensemble's per-row logic in a loop -- CalibratedClassifierCV
     calibrates only on the training fold (never sees the predict input) and
     predict_proba is row-independent -- so this is a batching/perf change,
     not a change in model behavior. Verified: predict_ensemble_batch()
     builds the classifier objects with the exact same constructor
     arguments as _run_ensemble's if/elif block.
  2. FEAT_A == FEAT_B (both ["basic","gaps","positional","temporal",
     "momentum"]), so the feature matrix is built ONCE per window and reused
     for both configs, instead of twice as predict_now.py does. Same values,
     saves one full feature-build pass.
  3. prepare_v3._feat_momentum's last feature ("consecutive-draw digit
     overlap") reads df.iloc[idx]'s OWN actual d1/d2/d3 -- the answer for the
     draw being predicted, not data strictly before it. This is a pre-existing
     bug in prepare_v3.py (not modified here, per instructions). It is
     reproduced as-is because the goal is to replicate the production
     predictor's exact behavior, and is flagged in the final report. Every
     other feature in FEAT_A/FEAT_B is strictly causal: _feat_basic uses
     idx-1, _feat_gaps/_feat_positional look backward from idx, and
     _feat_temporal only uses the target row's own date/day-of-week/month,
     which are known in advance (not leakage).

Do not modify prepare_v3.py, predict_now.py, or lag_analysis.py -- this file
only imports from them / reimplements their algorithms.
"""
import os, sys, json, time
import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from prepare_v3 import load_data, build_features, _VAL_END
import predict_now as pn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

NUM_COMBOS = pn.NUM_COMBOS
FEAT_A, FEAT_B = pn.FEAT_A, pn.FEAT_B
MODELS_A, MODELS_B = pn.MODELS_A, pn.MODELS_B
W_A, W_B = pn.W_A, pn.W_B
CONSENSUS_K = 50
INTERACT_HEAD, INTERACT_TAIL = 8, 6

assert FEAT_A == FEAT_B, "FEAT_A/FEAT_B diverged -- single feature build is no longer valid"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Production-parity correction for the leaked momentum feature.
#
# prepare_v3._feat_momentum (lines ~390-398) computes, for target row idx:
#     prev_digits = {df.d1[idx-1], df.d2[idx-1], df.d3[idx-1]}
#     curr_digits = {df.d1[idx],   df.d2[idx],   df.d3[idx]}
#     overlap = len(prev_digits & curr_digits)
# reading the target row's own actual digits -- not available at prediction
# time. predict_now.py (lines 228-233) never lets those real digits reach
# build_features: the row being predicted is fed in as a placeholder with
# d1=d2=d3=0. So in production curr_digits is always {0}, and
# overlap = 1.0 if 0 in prev_digits else 0.0.
#
# This reproduces that placeholder arithmetic for target rows only (test_idx /
# live_idx). Training rows are untouched -- in production those are real,
# already-observed historical draws, not placeholders, so prepare_v3's
# computation for them is not a leak. Verified against predict_now's actual
# placeholder construction on 5 sample rows (3 test-window, 2 live-window):
# corrected column == ground truth exactly, and no other column changes.
# ---------------------------------------------------------------------------

from prepare_v3 import FEATURE_DIMS as _FEATURE_DIMS


def _overlap_col_index(feature_sets):
    assert "momentum" in feature_sets
    offset = 0
    for fs in feature_sets:
        if fs == "momentum":
            break
        offset += _FEATURE_DIMS[fs]
    return offset + _FEATURE_DIMS["momentum"] - 1


def apply_production_overlap_correction(df, indices, X_raw, feature_sets):
    """Overwrite the momentum block's last column for target rows to match
    predict_now.py's production placeholder behavior. See module comment above."""
    col = _overlap_col_index(feature_sets)
    X_corr = X_raw.copy()
    d1 = df["d1"].values
    d2 = df["d2"].values
    d3 = df["d3"].values
    for row_i, idx in enumerate(indices):
        if idx > 0:
            prev_digits = {int(d1[idx - 1]), int(d2[idx - 1]), int(d3[idx - 1])}
            overlap = 1.0 if 0 in prev_digits else 0.0
        else:
            overlap = 0.0
        X_corr[row_i, col] = overlap
    return X_corr


# ---------------------------------------------------------------------------
# Batched ensemble -- same model construction as predict_now._run_ensemble,
# fit once, predict_proba over many rows.
# ---------------------------------------------------------------------------

def _build_clf(m):
    mtype = m["type"]
    if mtype == "et":
        clf = ExtraTreesClassifier(
            n_estimators=m["n_estimators"], max_depth=m["max_depth"],
            min_samples_leaf=m.get("min_samples_leaf", 1),
            min_samples_split=m.get("min_samples_split", 2),
            max_features=m.get("max_features", "sqrt"),
            random_state=42, n_jobs=8)
    elif mtype == "xgb":
        clf = xgb.XGBClassifier(
            objective="multi:softprob", num_class=10,
            n_estimators=m["n_estimators"], max_depth=m["max_depth"],
            learning_rate=m["learning_rate"], subsample=m["subsample"],
            colsample_bytree=m["colsample_bytree"],
            reg_alpha=m["reg_alpha"], reg_lambda=m["reg_lambda"],
            eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=8)
    elif mtype == "lgb":
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=10,
            n_estimators=m["n_estimators"], max_depth=m["max_depth"],
            learning_rate=m["learning_rate"], subsample=m["subsample"],
            colsample_bytree=m["colsample_bytree"],
            reg_alpha=m["reg_alpha"], reg_lambda=m["reg_lambda"],
            random_state=42, n_jobs=8, verbose=-1)
    else:
        raise ValueError(f"unknown model type {mtype}")
    if m.get("calibrate"):
        clf = CalibratedClassifierCV(
            clf, method=m.get("calibrate_method", "isotonic"),
            cv=m.get("calibrate_cv", 5))
    return clf


def fit_ensemble(X_train, y_d1, y_d2, y_d3, models_cfg, label=""):
    """Fit one classifier per (model_cfg, digit_position). Returns fitted[d_idx] = [(weight, clf), ...]."""
    fitted = []
    for d_idx, y_d in enumerate([y_d1, y_d2, y_d3]):
        row = []
        for m in models_cfg:
            t0 = time.time()
            clf = _build_clf(m)
            clf.fit(X_train, y_d)
            row.append((m["weight"], clf))
            log(f"  {label} d{d_idx+1} {m['type']:4s} fit: {time.time()-t0:6.1f}s "
                f"(n_train={X_train.shape[0]}, n_feat={X_train.shape[1]})")
        fitted.append(row)
    return fitted


def predict_ensemble_batch(fitted, X_pred, models_cfg):
    """Batch predict_proba for all rows in X_pred. Returns list of 3 (N,10) arrays."""
    total_w = sum(m["weight"] for m in models_cfg)
    N = X_pred.shape[0]
    digit_probs = []
    for d_idx in range(3):
        wp = np.zeros((N, 10))
        for weight, clf in fitted[d_idx]:
            proba = clf.predict_proba(X_pred)
            fp = np.zeros((N, 10))
            for ci, cls in enumerate(clf.classes_):
                fp[:, int(cls)] = proba[:, ci]
            wp += (weight / total_w) * fp
        digit_probs.append(wp)
    return digit_probs


def to_prob_matrix_batch(digit_probs):
    """digit_probs: list of 3 (N,10) arrays -> (N,1000) prob matrix, row-normalized."""
    N = digit_probs[0].shape[0]
    pm = np.zeros((N, NUM_COMBOS))
    for combo in range(NUM_COMBOS):
        d1, d2, d3 = pn.combo_to_digits(combo)
        pm[:, combo] = digit_probs[0][:, d1] * digit_probs[1][:, d2] * digit_probs[2][:, d3]
    pm = np.clip(pm, 0, None)
    row_sums = pm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pm /= row_sums
    return pm


def build_window_records(dates, slots, actual_combos, pm_a, pm_b):
    """Combine A/B into ensemble pm (same weighted log-product as predict_now),
    derive top20/consensus per row, build ledger-schema records."""
    N = pm_a.shape[0]
    W_TOT = W_A + W_B
    log_pm = (W_A * np.log(pm_a + 1e-20) + W_B * np.log(pm_b + 1e-20)) / W_TOT
    pm = np.exp(log_pm)
    pm = np.clip(pm, 0, None)
    pm /= pm.sum(axis=1, keepdims=True)

    records = []
    for i in range(N):
        sorted_combos = np.argsort(-pm[i])
        top_a = set(np.argsort(-pm_a[i])[:CONSENSUS_K])
        top_b = set(np.argsort(-pm_b[i])[:CONSENSUS_K])
        consensus_set = top_a & top_b
        consensus_combos = sorted(consensus_set, key=lambda c: -pm[i, c])

        date_str = dates[i]
        slot = int(slots[i])
        label, _, _, display = pn.DRAW_SLOTS[slot]
        day = pd.Timestamp(date_str).strftime("%A")

        records.append({
            "prediction_id": f"{date_str}_{slot}",
            "predicted_draw_date": date_str,
            "draw_slot": slot,
            "draw_label": label,
            "draw_display": display,
            "draw_day": day,
            "consensus_combos": [f"{c:03d}" for c in consensus_combos],
            "consensus_count": len(consensus_combos),
            "top20_combos": [f"{sorted_combos[k]:03d}" for k in range(20)],
            "actual_combo": actual_combos[i],
        })
    return records


# ---------------------------------------------------------------------------
# Live-window data loading (pick3all_live.csv -- combined single file)
# ---------------------------------------------------------------------------

_LIVE_ALL_PATH = os.path.join(_DIR, "..", "data", "pick3all_live.csv")
_GAME_TO_SLOT = {"Pick 3 Morning": 0, "Pick 3 Day": 1, "Pick 3 Evening": 2, "Pick 3 Night": 3}


def load_live_all():
    raw = pd.read_csv(_LIVE_ALL_PATH, header=None,
        names=["game", "month", "day", "year", "d1", "d2", "d3", "sum_col", "trailing"],
        dtype=str)
    raw = raw.dropna(subset=["year"]).reset_index(drop=True)
    raw["date"] = pd.to_datetime(
        raw["year"].str.strip() + "-" + raw["month"].str.strip() + "-" + raw["day"].str.strip(),
        format="%Y-%m-%d")
    raw["draw_time"] = raw["game"].map(_GAME_TO_SLOT)
    if raw["draw_time"].isna().any():
        bad = sorted(raw[raw["draw_time"].isna()]["game"].unique().tolist())
        raise ValueError(f"Unknown game labels in pick3all_live.csv: {bad}")
    for c in ["d1", "d2", "d3"]:
        raw[c] = raw[c].astype(int)
    raw["combo"] = raw["d1"].apply(str) + raw["d2"].apply(str) + raw["d3"].apply(str)
    raw["combo_int"] = raw["d1"] * 100 + raw["d2"] * 10 + raw["d3"]
    raw["day_of_week"] = raw["date"].dt.dayofweek
    raw["month"] = raw["date"].dt.month
    raw["year"] = raw["date"].dt.year
    raw["draw_time"] = raw["draw_time"].astype(int)
    raw = raw.sort_values(["date", "draw_time"]).reset_index(drop=True)
    # dedupe safety: keep first if a (date, draw_time) pair is repeated
    before = len(raw)
    raw = raw.drop_duplicates(subset=["date", "draw_time"], keep="first").reset_index(drop=True)
    if len(raw) != before:
        log(f"WARNING: dropped {before - len(raw)} duplicate (date, draw_time) rows from pick3all_live.csv")
    return raw[["date", "d1", "d2", "d3", "combo", "combo_int", "day_of_week", "month", "year", "draw_time"]]


# ---------------------------------------------------------------------------
# Window 1: held-out test split
# ---------------------------------------------------------------------------

def run_window_test(causal=False):
    log("=== WINDOW 1: test split (train+val -> test) ===")
    hist_df = load_data()
    train_end = _VAL_END  # 13,236
    n_total = len(hist_df)
    n_test = n_total - train_end
    log(f"total rows={n_total}, train+val rows=[0,{train_end}) ({train_end} draws), "
        f"test rows=[{train_end},{n_total}) ({n_test} draws)")

    train_idx = list(range(0, train_end))
    test_idx = list(range(train_end, n_total))

    t0 = time.time()
    X_train_raw = build_features(hist_df, train_idx, FEAT_A)
    X_test_raw = build_features(hist_df, test_idx, FEAT_A)
    log(f"feature build: {time.time()-t0:.1f}s  train={X_train_raw.shape} test={X_test_raw.shape}")

    if causal:
        X_test_raw = apply_production_overlap_correction(hist_df, test_idx, X_test_raw, FEAT_A)
        log("causal mode: corrected momentum-overlap column on test target rows "
            "(train rows untouched)")

    X_train_a = pn._apply_custom_interact(X_train_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=True)
    X_test_a = pn._apply_custom_interact(X_test_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=True)
    X_train_b = pn._apply_custom_interact(X_train_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=False)
    X_test_b = pn._apply_custom_interact(X_test_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=False)

    y_d1 = hist_df["d1"].values[train_idx].astype(int)
    y_d2 = hist_df["d2"].values[train_idx].astype(int)
    y_d3 = hist_df["d3"].values[train_idx].astype(int)

    t0 = time.time()
    fitted_a = fit_ensemble(X_train_a, y_d1, y_d2, y_d3, MODELS_A, label="A/test")
    train_time_a = time.time() - t0
    log(f"Config A TOTAL train time (test window): {train_time_a:.1f}s")

    t0 = time.time()
    fitted_b = fit_ensemble(X_train_b, y_d1, y_d2, y_d3, MODELS_B, label="B/test")
    train_time_b = time.time() - t0
    log(f"Config B TOTAL train time (test window): {train_time_b:.1f}s")

    dp_a = predict_ensemble_batch(fitted_a, X_test_a, MODELS_A)
    dp_b = predict_ensemble_batch(fitted_b, X_test_b, MODELS_B)
    pm_a = to_prob_matrix_batch(dp_a)
    pm_b = to_prob_matrix_batch(dp_b)

    dates = hist_df["date"].dt.strftime("%Y-%m-%d").values[test_idx]
    slots = hist_df["draw_time"].values[test_idx]
    actual = [f"{hist_df['d1'].values[i]}{hist_df['d2'].values[i]}{hist_df['d3'].values[i]}" for i in test_idx]

    records = build_window_records(dates, slots, actual, pm_a, pm_b)
    fname = "backtest_predictions_test_causal.jsonl" if causal else "backtest_predictions_test.jsonl"
    out_path = os.path.join(_DIR, fname)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {len(records)} records to {out_path}")
    return records, train_time_a, train_time_b


# ---------------------------------------------------------------------------
# Window 2: live window
# ---------------------------------------------------------------------------

def run_window_live(causal=False):
    log("=== WINDOW 2: live window (full history -> live) ===")
    hist_df = load_data()
    n_hist = len(hist_df)
    live_df = load_live_all()
    n_live = len(live_df)
    log(f"hist rows={n_hist} (through {hist_df['date'].iloc[-1].date()}), live rows={n_live} "
        f"({live_df['date'].iloc[0].date()} .. {live_df['date'].iloc[-1].date()})")

    combined_df = pd.concat([hist_df, live_df], ignore_index=True)
    train_idx = list(range(0, n_hist))
    live_idx = list(range(n_hist, n_hist + n_live))

    t0 = time.time()
    X_train_raw = build_features(combined_df, train_idx, FEAT_A)
    X_live_raw = build_features(combined_df, live_idx, FEAT_A)
    log(f"feature build: {time.time()-t0:.1f}s  train={X_train_raw.shape} live={X_live_raw.shape}")

    if causal:
        X_live_raw = apply_production_overlap_correction(combined_df, live_idx, X_live_raw, FEAT_A)
        log("causal mode: corrected momentum-overlap column on live target rows "
            "(train rows untouched)")

    X_train_a = pn._apply_custom_interact(X_train_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=True)
    X_live_a = pn._apply_custom_interact(X_live_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=True)
    X_train_b = pn._apply_custom_interact(X_train_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=False)
    X_live_b = pn._apply_custom_interact(X_live_raw, INTERACT_HEAD, INTERACT_TAIL, include_ratios=False)

    y_d1 = combined_df["d1"].values[train_idx].astype(int)
    y_d2 = combined_df["d2"].values[train_idx].astype(int)
    y_d3 = combined_df["d3"].values[train_idx].astype(int)

    t0 = time.time()
    fitted_a = fit_ensemble(X_train_a, y_d1, y_d2, y_d3, MODELS_A, label="A/live")
    train_time_a = time.time() - t0
    log(f"Config A TOTAL train time (live window): {train_time_a:.1f}s")

    t0 = time.time()
    fitted_b = fit_ensemble(X_train_b, y_d1, y_d2, y_d3, MODELS_B, label="B/live")
    train_time_b = time.time() - t0
    log(f"Config B TOTAL train time (live window): {train_time_b:.1f}s")

    dp_a = predict_ensemble_batch(fitted_a, X_live_a, MODELS_A)
    dp_b = predict_ensemble_batch(fitted_b, X_live_b, MODELS_B)
    pm_a = to_prob_matrix_batch(dp_a)
    pm_b = to_prob_matrix_batch(dp_b)

    dates = combined_df["date"].dt.strftime("%Y-%m-%d").values[live_idx]
    slots = combined_df["draw_time"].values[live_idx]
    actual = [f"{combined_df['d1'].values[i]}{combined_df['d2'].values[i]}{combined_df['d3'].values[i]}" for i in live_idx]

    records = build_window_records(dates, slots, actual, pm_a, pm_b)
    fname = "backtest_predictions_live_causal.jsonl" if causal else "backtest_predictions_live.jsonl"
    out_path = os.path.join(_DIR, fname)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {len(records)} records to {out_path}")
    return records, train_time_a, train_time_b


# ---------------------------------------------------------------------------
# Lag analysis (adapted from lag_analysis.py -- same algorithm, parameterized
# over an in-memory record list and output path instead of the hardcoded
# ledger path, per the "copy the functions" instruction).
# ---------------------------------------------------------------------------

def run_lag_analysis(records, out_path, window_name, n_perm=2000, seed=42, max_lag=5, combo_space=1000):
    total_records = len(records)
    scored = [r for r in records if r.get("actual_combo")]
    scored.sort(key=lambda r: (r["predicted_draw_date"], r["draw_slot"]))

    keys = [(r["predicted_draw_date"], r["draw_slot"]) for r in scored]
    assert len(keys) == len(set(keys)), f"duplicate (date, slot) in {window_name} records"

    n = len(scored)
    top20 = [r["top20_combos"] for r in scored]
    consensus = [r["consensus_combos"] for r in scored]
    actual = [r["actual_combo"] for r in scored]
    actual_digits = [tuple(int(c) for c in a) for a in actual]
    consensus_sizes = [len(c) for c in consensus]

    hit_top20 = np.zeros((n, n), dtype=bool)
    hit_cons = np.zeros((n, n), dtype=bool)
    maxpos_top20 = np.zeros((n, n), dtype=np.int8)
    maxpos_cons = np.zeros((n, n), dtype=np.int8)
    cov_top20 = np.zeros((n, n, 3), dtype=bool)
    cov_cons = np.zeros((n, n, 3), dtype=bool)

    top20_digits = [[tuple(int(ch) for ch in c) for c in combos] for combos in top20]
    cons_digits = [[tuple(int(ch) for ch in c) for c in combos] for combos in consensus]

    t0 = time.time()
    for i in range(n):
        t20_strs = set(top20[i])
        cons_strs = set(consensus[i])
        t20d = top20_digits[i]
        cnsd = cons_digits[i]
        for j in range(n):
            a_str = actual[j]
            a_d = actual_digits[j]

            hit_top20[i, j] = a_str in t20_strs
            hit_cons[i, j] = a_str in cons_strs

            best20 = 0
            cov20 = [False, False, False]
            for c in t20d:
                m = 0
                for p in range(3):
                    if c[p] == a_d[p]:
                        m += 1
                        cov20[p] = True
                if m > best20:
                    best20 = m
            maxpos_top20[i, j] = best20
            cov_top20[i, j] = cov20

            bestc = 0
            covc = [False, False, False]
            for c in cnsd:
                m = 0
                for p in range(3):
                    if c[p] == a_d[p]:
                        m += 1
                        covc[p] = True
                if m > bestc:
                    bestc = m
            maxpos_cons[i, j] = bestc
            cov_cons[i, j] = covc
    log(f"[{window_name}] pairwise matrix build (n={n}): {time.time()-t0:.1f}s")

    STAT_SOURCES = {
        "top20": dict(hit=hit_top20, maxpos=maxpos_top20, cov=cov_top20),
        "consensus": dict(hit=hit_cons, maxpos=maxpos_cons, cov=cov_cons),
    }

    def compute_stats(perm):
        out = {}
        for k in range(max_lag + 1):
            n_k = n - k
            rows = np.arange(n_k)
            cols = perm[k:n]
            for setname, src in STAT_SOURCES.items():
                hit = src["hit"][rows, cols]
                mp = src["maxpos"][rows, cols]
                cov = src["cov"][rows, cols, :]
                out[(k, setname, "exact_hit_rate")] = hit.mean()
                out[(k, setname, "pos_ge2_rate")] = (mp >= 2).mean()
                out[(k, setname, "mean_max_pos")] = mp.mean()
                for p in range(3):
                    out[(k, setname, f"cov_pos{p}")] = cov[:, p].mean()
        return out

    identity = np.arange(n)
    observed = compute_stats(identity)

    t0 = time.time()
    rng = np.random.RandomState(seed)
    null_samples = {key: np.empty(n_perm) for key in observed}
    for t in range(n_perm):
        perm = rng.permutation(n)
        stats = compute_stats(perm)
        for key, val in stats.items():
            null_samples[key][t] = val
    log(f"[{window_name}] {n_perm} permutations: {time.time()-t0:.1f}s")

    results = []
    for key, obs_val in observed.items():
        k, setname, stat = key
        nulls = null_samples[key]
        null_mean = nulls.mean()
        null_p95 = np.percentile(nulls, 95)
        p_value = (1 + np.sum(nulls >= obs_val)) / (n_perm + 1)
        results.append({
            "lag": k, "n": n - k, "set": setname, "stat": stat,
            "observed": float(obs_val), "null_mean": float(null_mean),
            "null_p95": float(null_p95), "p_value": float(p_value),
        })

    n_cells = len(results)
    n_sig = sum(1 for r in results if r["p_value"] < 0.05)
    top20_baseline = 20 / combo_space
    mean_cons_size = float(np.mean(consensus_sizes))
    cons_baseline = mean_cons_size / combo_space

    summary = {
        "window": window_name,
        "total_records": total_records,
        "scored_records": n,
        "combo_space": combo_space,
        "consensus_size": {
            "min": int(min(consensus_sizes)), "median": float(np.median(consensus_sizes)),
            "max": int(max(consensus_sizes)), "mean": mean_cons_size,
        },
        "top20_size": 20,
        "chance_baseline_top20_exact": top20_baseline,
        "chance_baseline_consensus_exact_approx": cons_baseline,
        "n_perm": n_perm, "seed": seed,
        "n_cells_tested": n_cells, "n_cells_p_lt_05": n_sig,
        "cells": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{window_name}] lag analysis -> {out_path}: n={n} cells={n_cells} p<.05={n_sig}")
    return summary


def main():
    t_start = time.time()

    records_test, tt_a_test, tt_b_test = run_window_test()
    run_lag_analysis(records_test, os.path.join(_DIR, "backtest_lag_output_test.json"), "test")

    records_live, tt_a_live, tt_b_live = run_window_live()
    run_lag_analysis(records_live, os.path.join(_DIR, "backtest_lag_output_live.json"), "live")

    log(f"ALL DONE. total wall time: {time.time()-t_start:.1f}s")
    log(f"train_time_a_test={tt_a_test:.1f}s train_time_b_test={tt_b_test:.1f}s "
        f"train_time_a_live={tt_a_live:.1f}s train_time_b_live={tt_b_live:.1f}s")


def main_causal():
    """Same Design B backtest, momentum-overlap leak neutralized (matches
    predict_now.py's placeholder behavior for target rows). Writes distinct
    output files -- does not touch the earlier contaminated outputs."""
    t_start = time.time()
    results_dir = os.path.join(_DIR, "results")

    records_test, tt_a_test, tt_b_test = run_window_test(causal=True)
    run_lag_analysis(records_test, os.path.join(results_dir, "backtest_lag_output_test_causal.json"), "test_causal")

    records_live, tt_a_live, tt_b_live = run_window_live(causal=True)
    run_lag_analysis(records_live, os.path.join(results_dir, "backtest_lag_output_live_causal.json"), "live_causal")

    log(f"ALL DONE (causal). total wall time: {time.time()-t_start:.1f}s")
    log(f"train_time_a_test={tt_a_test:.1f}s train_time_b_test={tt_b_test:.1f}s "
        f"train_time_a_live={tt_a_live:.1f}s train_time_b_live={tt_b_live:.1f}s")


if __name__ == "__main__":
    if "--causal" in sys.argv:
        main_causal()
    else:
        main()
