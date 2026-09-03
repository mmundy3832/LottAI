"""
prereg_early_hit.py - Preregistered test: Early-Hit Hypothesis
(plans/PREREG_EARLY_HIT.md), registered 2026-08-31 before this script ran.

Trains Config A (exp#1566) and Config B (exp#1572), blended exactly as
predict_now.py does, ONCE on all draws through 2026-06-05 (historical
data/pick3_combined.csv PLUS live rows from data/pick3all_live.csv up to and
including 2026-06-05). Predicts every live draw strictly after 2026-06-05
(held-out set, expected n ~280), with the momentum-overlap feature in its
production placeholder form on the held-out target rows (the causal
correction from backtest_lag.py).

Everything except the train/predict date split is reused, not reimplemented:
imports load_live_all, apply_production_overlap_correction, fit_ensemble,
predict_ensemble_batch, to_prob_matrix_batch, build_window_records, and log
from backtest_lag.py (which itself reuses prepare_v3.py / predict_now.py --
none of those three files are modified here). The pairwise hit / positional-
match matrix construction below mirrors backtest_lag.run_lag_analysis's
matrix section exactly; it is copied rather than called because that
function does not expose the matrices, does not support negative lags, and
does not compute a pooled statistic across multiple lags -- all three are
needed for the PRIMARY test in the prereg spec.

Primary test: S = pooled rate, over all (prediction for draw t, actual draw
t+k) pairs with k in {+1, +2} together, of best-in-top20 positional match
>= 2 of 3 digits. Null: 2000 permutations of the actual-combo sequence, seed
42, same statistic. One-sided empirical p = Pr[null >= observed].

Secondary: exact top20 hit rate at k=+1 and k=+2 separately; the full lag
0..5 table (top20 exact, consensus exact, both >=2-of-3 metrics); the same
four stats at k=-1 and k=-2 (late-hit control).

Outputs:
  autoresearch/results/prereg_predictions.jsonl   -- per-draw prediction records
  autoresearch/results/prereg_early_hit_output.json -- primary + secondary results
"""
import os, sys, json, time
import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_RESULTS_DIR = os.path.join(_DIR, "results")

from prepare_v3 import load_data, build_features
import predict_now as pn
import backtest_lag as bl

FEAT_A, FEAT_B = pn.FEAT_A, pn.FEAT_B
MODELS_A, MODELS_B = pn.MODELS_A, pn.MODELS_B
assert FEAT_A == FEAT_B, "FEAT_A/FEAT_B diverged -- single feature build is no longer valid"

log = bl.log

CUTOFF_DATE = pd.Timestamp("2026-06-05")  # train on all draws <= this date
N_PERM = 2000
SEED = 42
MAX_LAG = 5                # secondary "full lag 0..5 table"
CONTROL_KS = (-1, -2)      # secondary late-hit control
PRIMARY_KS = (1, 2)        # primary pooled statistic window
COMBO_SPACE = 1000


# ---------------------------------------------------------------------------
# Train/predict split (the one new element vs. backtest_lag.py's windows)
# ---------------------------------------------------------------------------

def build_train_predict_split():
    hist_df = load_data()
    live_df = bl.load_live_all()
    combined_df = pd.concat([hist_df, live_df], ignore_index=True)

    dates = combined_df["date"]
    train_mask = dates <= CUTOFF_DATE
    predict_mask = dates > CUTOFF_DATE
    train_idx = combined_df.index[train_mask].tolist()
    predict_idx = combined_df.index[predict_mask].tolist()

    n_hist = len(hist_df)
    n_live = len(live_df)
    n_live_train = sum(1 for i in train_idx if i >= n_hist)
    log(f"combined rows={len(combined_df)} (hist={n_hist} through "
        f"{hist_df['date'].iloc[-1].date()}, live={n_live} "
        f"{live_df['date'].iloc[0].date()}..{live_df['date'].iloc[-1].date()})")
    log(f"train rows (date <= {CUTOFF_DATE.date()}): {len(train_idx)} "
        f"(hist={n_hist}, live={n_live_train})")
    log(f"predict rows (date > {CUTOFF_DATE.date()}): {len(predict_idx)}")

    return combined_df, train_idx, predict_idx, n_hist, n_live_train


# ---------------------------------------------------------------------------
# Fit once on train, predict every held-out row (reuses backtest_lag.py's
# batched ensemble fit/predict and record-building functions)
# ---------------------------------------------------------------------------

def run_predictions():
    combined_df, train_idx, predict_idx, n_hist, n_live_train = build_train_predict_split()

    t0 = time.time()
    X_train_raw = build_features(combined_df, train_idx, FEAT_A)
    X_pred_raw = build_features(combined_df, predict_idx, FEAT_A)
    log(f"feature build: {time.time()-t0:.1f}s train={X_train_raw.shape} predict={X_pred_raw.shape}")

    X_pred_raw = bl.apply_production_overlap_correction(combined_df, predict_idx, X_pred_raw, FEAT_A)
    log("causal mode: corrected momentum-overlap column on held-out target rows (train rows untouched)")

    X_train_a = pn._apply_custom_interact(X_train_raw, bl.INTERACT_HEAD, bl.INTERACT_TAIL, include_ratios=True)
    X_pred_a = pn._apply_custom_interact(X_pred_raw, bl.INTERACT_HEAD, bl.INTERACT_TAIL, include_ratios=True)
    X_train_b = pn._apply_custom_interact(X_train_raw, bl.INTERACT_HEAD, bl.INTERACT_TAIL, include_ratios=False)
    X_pred_b = pn._apply_custom_interact(X_pred_raw, bl.INTERACT_HEAD, bl.INTERACT_TAIL, include_ratios=False)

    y_d1 = combined_df["d1"].values[train_idx].astype(int)
    y_d2 = combined_df["d2"].values[train_idx].astype(int)
    y_d3 = combined_df["d3"].values[train_idx].astype(int)

    t0 = time.time()
    fitted_a = bl.fit_ensemble(X_train_a, y_d1, y_d2, y_d3, MODELS_A, label="A/prereg")
    train_time_a = time.time() - t0
    log(f"Config A TOTAL train time: {train_time_a:.1f}s")

    t0 = time.time()
    fitted_b = bl.fit_ensemble(X_train_b, y_d1, y_d2, y_d3, MODELS_B, label="B/prereg")
    train_time_b = time.time() - t0
    log(f"Config B TOTAL train time: {train_time_b:.1f}s")

    dp_a = bl.predict_ensemble_batch(fitted_a, X_pred_a, MODELS_A)
    dp_b = bl.predict_ensemble_batch(fitted_b, X_pred_b, MODELS_B)
    pm_a = bl.to_prob_matrix_batch(dp_a)
    pm_b = bl.to_prob_matrix_batch(dp_b)

    dates = combined_df["date"].dt.strftime("%Y-%m-%d").values[predict_idx]
    slots = combined_df["draw_time"].values[predict_idx]
    actual = [f"{combined_df['d1'].values[i]}{combined_df['d2'].values[i]}{combined_df['d3'].values[i]}"
              for i in predict_idx]

    records = bl.build_window_records(dates, slots, actual, pm_a, pm_b)
    out_path = os.path.join(_RESULTS_DIR, "prereg_predictions.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {len(records)} records to {out_path}")

    meta = {
        "n_train_rows": len(train_idx),
        "n_hist_rows": n_hist,
        "n_live_train_rows": n_live_train,
        "n_holdout_draws": len(predict_idx),
        "train_time_a_s": train_time_a,
        "train_time_b_s": train_time_b,
        "holdout_date_min": str(dates.min()),
        "holdout_date_max": str(dates.max()),
    }
    return records, meta


# ---------------------------------------------------------------------------
# Pairwise hit / positional-match matrices (mirrors backtest_lag.
# run_lag_analysis's matrix section; copied because that function doesn't
# expose the matrices, doesn't support negative lags, and doesn't compute a
# pooled multi-lag statistic).
# ---------------------------------------------------------------------------

def build_pairwise_matrices(records):
    scored = [r for r in records if r.get("actual_combo")]
    scored.sort(key=lambda r: (r["predicted_draw_date"], r["draw_slot"]))
    keys = [(r["predicted_draw_date"], r["draw_slot"]) for r in scored]
    assert len(keys) == len(set(keys)), "duplicate (date, slot) in held-out records"

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
            for c in t20d:
                m = (c[0] == a_d[0]) + (c[1] == a_d[1]) + (c[2] == a_d[2])
                if m > best20:
                    best20 = m
            maxpos_top20[i, j] = best20

            bestc = 0
            for c in cnsd:
                m = (c[0] == a_d[0]) + (c[1] == a_d[1]) + (c[2] == a_d[2])
                if m > bestc:
                    bestc = m
            maxpos_cons[i, j] = bestc
    log(f"pairwise matrix build (n={n}): {time.time()-t0:.1f}s")

    return dict(n=n, hit_top20=hit_top20, hit_cons=hit_cons,
                maxpos_top20=maxpos_top20, maxpos_cons=maxpos_cons,
                consensus_sizes=consensus_sizes)


# ---------------------------------------------------------------------------
# Lag statistics, general (positive or negative k), and the primary pooled
# statistic. Same permutation (2000 draws, seed 42, np.random.RandomState)
# generates both the primary null and every secondary-table null cell, one
# permutation per trial, matching how the existing harness reuses a single
# permuted actual-sequence across every lag in one trial.
# ---------------------------------------------------------------------------

def _lag_rows_cols(n, k, perm):
    if k >= 0:
        rows = np.arange(0, n - k)
    else:
        rows = np.arange(-k, n)
    cols = perm[rows + k]
    return rows, cols


def stat_cell(mats, k, setname, stat, perm):
    n = mats["n"]
    rows, cols = _lag_rows_cols(n, k, perm)
    hit_src = mats["hit_top20"] if setname == "top20" else mats["hit_cons"]
    mp_src = mats["maxpos_top20"] if setname == "top20" else mats["maxpos_cons"]
    if stat == "exact_hit_rate":
        return hit_src[rows, cols].mean()
    elif stat == "pos_ge2_rate":
        return (mp_src[rows, cols] >= 2).mean()
    raise ValueError(stat)


def pooled_pos_ge2_rate(mats, ks, perm):
    """PRIMARY statistic: pooled top20 >=2-of-3 rate over multiple k, combined
    numerator/denominator across k (not averaged per-k)."""
    n = mats["n"]
    hits, total = 0, 0
    for k in ks:
        rows, cols = _lag_rows_cols(n, k, perm)
        mp = mats["maxpos_top20"][rows, cols]
        hits += int((mp >= 2).sum())
        total += len(rows)
    return hits / total


def run_analysis(mats):
    n = mats["n"]
    identity = np.arange(n)

    obs_primary = pooled_pos_ge2_rate(mats, PRIMARY_KS, identity)

    all_ks = sorted(set(range(0, MAX_LAG + 1)) | set(CONTROL_KS))
    set_names = ["top20", "consensus"]
    stat_names = ["exact_hit_rate", "pos_ge2_rate"]

    obs_cells = {}
    for k in all_ks:
        for setname in set_names:
            for stat in stat_names:
                obs_cells[(k, setname, stat)] = stat_cell(mats, k, setname, stat, identity)

    t0 = time.time()
    rng = np.random.RandomState(SEED)
    null_primary = np.empty(N_PERM)
    null_cells = {key: np.empty(N_PERM) for key in obs_cells}
    for t in range(N_PERM):
        perm = rng.permutation(n)
        null_primary[t] = pooled_pos_ge2_rate(mats, PRIMARY_KS, perm)
        for k in all_ks:
            for setname in set_names:
                for stat in stat_names:
                    null_cells[(k, setname, stat)][t] = stat_cell(mats, k, setname, stat, perm)
    log(f"{N_PERM} permutations (seed={SEED}): {time.time()-t0:.1f}s")

    def summarize(obs, nulls):
        return dict(
            observed=float(obs),
            null_mean=float(nulls.mean()),
            null_p95=float(np.percentile(nulls, 95)),
            p_value_one_sided=float((1 + np.sum(nulls >= obs)) / (N_PERM + 1)),
        )

    primary_result = summarize(obs_primary, null_primary)
    n_pairs_pooled = sum(n - abs(k) for k in PRIMARY_KS)
    primary_result.update(dict(
        statistic="pooled top20 best-in-top20 >=2-of-3 positional match rate, k in {+1,+2}",
        ks=list(PRIMARY_KS), n_perm=N_PERM, seed=SEED, n_pairs_pooled=n_pairs_pooled,
    ))
    primary_result["significant_p_lt_05"] = primary_result["p_value_one_sided"] < 0.05

    cells = []
    for (k, setname, stat), obs in obs_cells.items():
        res = summarize(obs, null_cells[(k, setname, stat)])
        res.update(dict(lag=k, set=setname, stat=stat, n_pairs=n - abs(k)))
        cells.append(res)
    cells.sort(key=lambda c: (c["lag"], c["set"], c["stat"]))

    return primary_result, cells


# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    records, meta = run_predictions()
    mats = build_pairwise_matrices(records)
    primary_result, cells = run_analysis(mats)

    exact_top20_k1 = next(c for c in cells if c["lag"] == 1 and c["set"] == "top20" and c["stat"] == "exact_hit_rate")
    exact_top20_k2 = next(c for c in cells if c["lag"] == 2 and c["set"] == "top20" and c["stat"] == "exact_hit_rate")
    full_lag_table = [c for c in cells if 0 <= c["lag"] <= MAX_LAG]
    negative_control = [c for c in cells if c["lag"] < 0]

    mean_cons_size = float(np.mean(mats["consensus_sizes"]))

    total_runtime = time.time() - t_start

    output = {
        "test_name": "PREREG_EARLY_HIT",
        "spec_file": "plans/PREREG_EARLY_HIT.md",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train": {
            "cutoff_date": str(CUTOFF_DATE.date()),
            "n_train_rows": meta["n_train_rows"],
            "n_hist_rows": meta["n_hist_rows"],
            "n_live_train_rows": meta["n_live_train_rows"],
        },
        "holdout": {
            "n_draws": meta["n_holdout_draws"],
            "n_scored": mats["n"],
            "date_min": meta["holdout_date_min"],
            "date_max": meta["holdout_date_max"],
        },
        "combo_space": COMBO_SPACE,
        "top20_size": 20,
        "chance_baseline_top20_exact": 20 / COMBO_SPACE,
        "consensus_size": {
            "min": int(min(mats["consensus_sizes"])),
            "median": float(np.median(mats["consensus_sizes"])),
            "max": int(max(mats["consensus_sizes"])),
            "mean": mean_cons_size,
        },
        "chance_baseline_consensus_exact_approx": mean_cons_size / COMBO_SPACE,
        "primary": primary_result,
        "secondary": {
            "exact_top20_hit_k1": exact_top20_k1,
            "exact_top20_hit_k2": exact_top20_k2,
            "full_lag_0_5_table": full_lag_table,
            "negative_control_k_minus1_minus2": negative_control,
        },
        "runtime_seconds": {
            "train_time_a_s": meta["train_time_a_s"],
            "train_time_b_s": meta["train_time_b_s"],
            "total_s": total_runtime,
        },
        "deviations": [
            "No per-draw retraining (as specified): models fit once on all "
            "train rows and predict_proba is called in batch over every "
            "held-out row, per the prereg's stated deviation from "
            "predict_now.py's live per-draw retraining."
        ],
    }

    out_path = os.path.join(_RESULTS_DIR, "prereg_early_hit_output.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"wrote {out_path}")

    log(f"PRIMARY: observed={primary_result['observed']:.4f} "
        f"null_mean={primary_result['null_mean']:.4f} "
        f"p={primary_result['p_value_one_sided']:.4f} "
        f"sig={primary_result['significant_p_lt_05']}")
    log(f"TOTAL RUNTIME: {total_runtime:.1f}s")


if __name__ == "__main__":
    main()
