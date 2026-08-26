"""
predict_now.py - Predict the next upcoming Pick 3 draw.

Ensemble of top 2 experiments by live_optimal_ev:
  Config A: exp#1566  live_ev=$19.32
  Config B: exp#1572  live_ev=$16.78

Importable: call run_prediction() to get structured results.
Standalone: python predict_now.py  (prints formatted output)
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

from prepare_v3 import load_data, build_features

NUM_COMBOS = 1000
def combo_to_digits(c): return (c // 100, (c // 10) % 10, c % 10)

# ---------------------------------------------------------------------------
# Draw schedule
# ---------------------------------------------------------------------------

_CT = ZoneInfo("America/Chicago")

DRAW_SLOTS = {
    0: ("Morning",  10,  0, "10:00 AM CT"),
    1: ("Day",      12, 27, "12:27 PM CT"),
    2: ("Evening",  18,  0,  "6:00 PM CT"),
    3: ("Night",    22, 12, "10:12 PM CT"),
}

def next_draw():
    """Return (date, draw_time_idx, label, display_time) for the next upcoming draw."""
    now = datetime.now(tz=_CT)
    for day_offset in range(7):
        candidate = now.date() + timedelta(days=day_offset)
        if candidate.weekday() == 6:   # Sunday — no draws
            continue
        for slot_idx, (label, hour, minute, display) in DRAW_SLOTS.items():
            draw_dt = datetime(candidate.year, candidate.month, candidate.day,
                               hour, minute, tzinfo=_CT)
            if draw_dt > now:
                return candidate, slot_idx, label, display
    raise RuntimeError("Could not find a next draw within 7 days")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(_DIR, "..", "data")
_LIVE_PATHS = [
    os.path.join(_DATA_DIR, "pick3morning_live.csv"),
    os.path.join(_DATA_DIR, "pick3day_live.csv"),
    os.path.join(_DATA_DIR, "pick3evening_live.csv"),
    os.path.join(_DATA_DIR, "pick3night_live.csv"),
]

def _load_live():
    parts = []
    for p in _LIVE_PATHS:
        raw = pd.read_csv(p, header=None,
            names=["game","month","day","year","d1","d2","d3","sum_col","trailing"],
            dtype=str)
        raw = raw.dropna(subset=["year"]).reset_index(drop=True)
        raw["date"] = pd.to_datetime(
            raw["year"].str.strip() + "-" + raw["month"].str.strip() + "-" + raw["day"].str.strip(),
            format="%Y-%m-%d")
        for c in ["d1","d2","d3"]:
            raw[c] = raw[c].astype(int)
        raw["combo"]       = raw["d1"].apply(str) + raw["d2"].apply(str) + raw["d3"].apply(str)
        raw["combo_int"]   = raw["d1"]*100 + raw["d2"]*10 + raw["d3"]
        raw["day_of_week"] = raw["date"].dt.dayofweek
        raw["month"]       = raw["date"].dt.month
        raw["year"]        = raw["date"].dt.year
        parts.append(raw[["date","d1","d2","d3","combo","combo_int",
                           "day_of_week","month","year"]])
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature pipeline helpers
# ---------------------------------------------------------------------------

def _build_split(combined_df, all_indices, predict_idx, feature_sets):
    X_all  = build_features(combined_df, all_indices,   feature_sets)
    X_pred = build_features(combined_df, [predict_idx], feature_sets)
    return X_all, X_pred


def _apply_custom_interact(X, n_head, n_tail, include_ratios):
    nh = min(n_head, X.shape[1])
    nt = min(n_tail, X.shape[1])
    f1, f2 = X[:, :nh], X[:, -nt:]
    parts = []
    for i in range(nh):
        for j in range(nt):
            parts.append(f1[:, i] * f2[:, j])
            if include_ratios:
                parts.append(f1[:, i] / (f2[:, j] + 1e-6))
    ci = np.column_stack(parts) if parts else np.zeros((X.shape[0], 1))
    ci = np.nan_to_num(ci, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.hstack([X, ci])


def _run_ensemble(X_all, X_pred, models_cfg, y_d1, y_d2, y_d3):
    total_w = sum(m["weight"] for m in models_cfg)
    digit_probs = []
    for d_idx, y_d in enumerate([y_d1, y_d2, y_d3]):
        wp = np.zeros((1, 10))
        for m in models_cfg:
            mtype = m["type"]
            if mtype == "et":
                clf = ExtraTreesClassifier(
                    n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                    min_samples_leaf=m.get("min_samples_leaf", 1),
                    min_samples_split=m.get("min_samples_split", 2),
                    max_features=m.get("max_features", "sqrt"),
                    random_state=42, n_jobs=-1)
            elif mtype == "xgb":
                clf = xgb.XGBClassifier(
                    objective="multi:softprob", num_class=10,
                    n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                    learning_rate=m["learning_rate"], subsample=m["subsample"],
                    colsample_bytree=m["colsample_bytree"],
                    reg_alpha=m["reg_alpha"], reg_lambda=m["reg_lambda"],
                    eval_metric="mlogloss", random_state=42, verbosity=0)
            elif mtype == "lgb":
                clf = lgb.LGBMClassifier(
                    objective="multiclass", num_class=10,
                    n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                    learning_rate=m["learning_rate"], subsample=m["subsample"],
                    colsample_bytree=m["colsample_bytree"],
                    reg_alpha=m["reg_alpha"], reg_lambda=m["reg_lambda"],
                    random_state=42, n_jobs=-1, verbose=-1)
            if m.get("calibrate"):
                clf = CalibratedClassifierCV(
                    clf, method=m.get("calibrate_method", "isotonic"),
                    cv=m.get("calibrate_cv", 5))
            clf.fit(X_all, y_d)
            proba = clf.predict_proba(X_pred)
            fp = np.zeros((1, 10))
            for ci, cls in enumerate(clf.classes_):
                fp[:, int(cls)] = proba[:, ci]
            wp += (m["weight"] / total_w) * fp
        digit_probs.append(wp)
        top3 = np.argsort(-wp[0])[:3]
        print(f"  d{d_idx+1} top-3: {list(top3)}  "
              f"probs: {[f'{wp[0,x]:.4f}' for x in top3]}")
    return digit_probs


def _to_prob_matrix(digit_probs):
    pm = np.zeros(NUM_COMBOS)
    for combo in range(NUM_COMBOS):
        d1, d2, d3 = combo_to_digits(combo)
        pm[combo] = digit_probs[0][0,d1] * digit_probs[1][0,d2] * digit_probs[2][0,d3]
    pm = np.clip(pm, 0, None)
    pm /= pm.sum()
    return pm


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

FEAT_A = ["basic", "gaps", "positional", "temporal", "momentum"]
MODELS_A = [
    {"type":"et",  "weight":0.3806883324481559,
     "n_estimators":91, "max_depth":15, "min_samples_leaf":3,
     "min_samples_split":6, "max_features":0.6,
     "calibrate":True, "calibrate_method":"isotonic", "calibrate_cv":5},
    {"type":"xgb", "weight":0.5081199233593982,
     "n_estimators":300, "max_depth":6, "learning_rate":0.05,
     "subsample":0.8, "colsample_bytree":0.7,
     "reg_alpha":0.0, "reg_lambda":1.0, "calibrate":False},
    {"type":"lgb", "weight":0.1845253046820533,
     "n_estimators":300, "max_depth":6, "learning_rate":0.05,
     "subsample":0.8, "colsample_bytree":0.7,
     "reg_alpha":3.8177079241940146, "reg_lambda":1.0, "calibrate":False},
]

FEAT_B = ["basic", "gaps", "positional", "temporal", "momentum"]
MODELS_B = [
    {"type":"et",  "weight":0.3806883324481559,
     "n_estimators":91, "max_depth":13, "min_samples_leaf":3,
     "min_samples_split":6, "max_features":0.6,
     "calibrate":True, "calibrate_method":"isotonic", "calibrate_cv":5},
    {"type":"xgb", "weight":0.5081199233593982,
     "n_estimators":104, "max_depth":6, "learning_rate":0.05,
     "subsample":0.8, "colsample_bytree":0.7,
     "reg_alpha":2.4796802528167055, "reg_lambda":1.0, "calibrate":False},
    {"type":"lgb", "weight":0.1845253046820533,
     "n_estimators":300, "max_depth":6, "learning_rate":0.05,
     "subsample":0.8, "colsample_bytree":0.7,
     "reg_alpha":3.8177079241940146, "reg_lambda":1.0, "calibrate":False},
]

W_A, W_B = 19.324786, 16.777778


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_prediction():
    """Run the full prediction pipeline. Returns a structured result dict."""
    draw_date, draw_slot, draw_label, draw_display = next_draw()
    draw_ts  = pd.Timestamp(draw_date)
    draw_day = draw_ts.strftime("%A")

    hist_df = load_data()
    live_df = _load_live()

    placeholder = pd.DataFrame([{
        "date": draw_ts, "d1": 0, "d2": 0, "d3": 0,
        "combo": "000", "combo_int": 0,
        "day_of_week": draw_ts.dayofweek,
        "month": draw_ts.month, "year": draw_ts.year,
    }])

    combined_df = pd.concat([hist_df, live_df, placeholder], ignore_index=True)
    n_history   = len(combined_df) - 1
    predict_idx = n_history
    all_indices = list(range(n_history))

    y_d1 = combined_df["d1"].values[:n_history].astype(int)
    y_d2 = combined_df["d2"].values[:n_history].astype(int)
    y_d3 = combined_df["d3"].values[:n_history].astype(int)

    last_known = combined_df["date"].iloc[n_history - 1].strftime("%Y-%m-%d")
    print(f"Training on {n_history} draws  (through {last_known})")
    print(f"Predicting:  {draw_date.strftime('%B %-d %Y')} — {draw_day} {draw_label} ({draw_display})")
    print()

    print("=== Config A: exp#1566 (live_ev=$19.32) ===")
    X_all_a, X_pred_a = _build_split(combined_df, all_indices, predict_idx, FEAT_A)
    X_all_a  = _apply_custom_interact(X_all_a,  8, 6, include_ratios=True)
    X_pred_a = _apply_custom_interact(X_pred_a, 8, 6, include_ratios=True)
    dp_a = _run_ensemble(X_all_a, X_pred_a, MODELS_A, y_d1, y_d2, y_d3)
    pm_a = _to_prob_matrix(dp_a)
    print()

    print("=== Config B: exp#1572 (live_ev=$16.78) ===")
    X_all_b, X_pred_b = _build_split(combined_df, all_indices, predict_idx, FEAT_B)
    X_all_b  = _apply_custom_interact(X_all_b,  8, 6, include_ratios=False)
    X_pred_b = _apply_custom_interact(X_pred_b, 8, 6, include_ratios=False)
    dp_b = _run_ensemble(X_all_b, X_pred_b, MODELS_B, y_d1, y_d2, y_d3)
    pm_b = _to_prob_matrix(dp_b)
    print()

    # Weighted product ensemble
    W_TOT = W_A + W_B
    log_pm = (W_A * np.log(pm_a + 1e-20) + W_B * np.log(pm_b + 1e-20)) / W_TOT
    pm = np.exp(log_pm)
    pm = np.clip(pm, 0, None)
    pm /= pm.sum()

    digit_ensemble = [(W_A * dp_a[d] + W_B * dp_b[d]) / W_TOT for d in range(3)]
    sorted_combos  = np.argsort(-pm)

    # Consensus plays: combos that both models rank in their top-K independently.
    # These are the high-conviction plays where the models genuinely agree.
    CONSENSUS_K = 50   # look at each model's top 50 for overlap
    top_a = set(np.argsort(-pm_a)[:CONSENSUS_K])
    top_b = set(np.argsort(-pm_b)[:CONSENSUS_K])
    consensus_set  = top_a & top_b
    # Rank consensus combos by the ensemble probability
    consensus_combos = sorted(consensus_set, key=lambda c: -pm[c])

    # EV for the consensus plays
    if consensus_combos:
        consensus_ev = sum(pm[c] for c in consensus_combos) * 500 - len(consensus_combos)
    else:
        consensus_ev = 0.0

    # Digit summary strings for analysis module
    digit_summary = []
    for d in range(3):
        probs = digit_ensemble[d][0]
        top5  = np.argsort(-probs)[:5]
        digit_summary.append(", ".join(f"{int(x)} ({100*probs[x]:.1f}%)" for x in top5))

    return {
        "draw_date":       draw_date,
        "draw_slot":       draw_slot,
        "draw_label":      draw_label,
        "draw_display":    draw_display,
        "draw_day":        draw_day,
        "pm":              pm,
        "pm_a":            pm_a,
        "pm_b":            pm_b,
        "sorted_combos":   sorted_combos,
        "digit_ensemble":  digit_ensemble,
        "digit_summary":   digit_summary,
        "consensus_combos":consensus_combos,
        "consensus_ev":    consensus_ev,
    }


def print_results(r):
    pm              = r["pm"]
    pm_a            = r["pm_a"]
    pm_b            = r["pm_b"]
    sorted_combos   = r["sorted_combos"]
    digit_ens       = r["digit_ensemble"]
    consensus_combos= r["consensus_combos"]
    consensus_ev    = r["consensus_ev"]

    header = (f"{r['draw_date'].strftime('%B %-d %Y')} — "
              f"{r['draw_day']} {r['draw_label']} ({r['draw_display']})")
    print("=" * 60)
    print(f"  PREDICTION: {header}")
    print("=" * 60)
    print()

    print("--- Per-digit breakdown ---")
    for d, name in enumerate(["Position 1", "Position 2", "Position 3"]):
        probs = digit_ens[d][0]
        top5  = np.argsort(-probs)[:5]
        print(f"  {name}: {list(top5)}  (probs: {[f'{probs[x]:.4f}' for x in top5]})")
    print()

    # Consensus: combos both models ranked in their independent top-50
    n = len(consensus_combos)
    print(f"--- Consensus plays ({n} combos both models agree on) ---")
    if consensus_combos:
        print(f"  {'Combo':>5}  {'p(ens)':>8}  {'p(A)':>8}  {'p(B)':>8}  "
              f"{'rnk(A)':>7}  {'rnk(B)':>7}")
        sorted_a = list(np.argsort(-pm_a))
        sorted_b = list(np.argsort(-pm_b))
        for combo in consensus_combos:
            d1, d2, d3 = combo_to_digits(combo)
            rank_a = sorted_a.index(combo) + 1
            rank_b = sorted_b.index(combo) + 1
            print(f"  {combo:03d}    {pm[combo]:.6f}  {pm_a[combo]:.6f}  {pm_b[combo]:.6f}"
                  f"  #{rank_a:>5}   #{rank_b:>5}   ({d1}-{d2}-{d3})")
        print(f"  EV if you buy all {n}: ${consensus_ev:+.3f}")
    else:
        print("  (no consensus — models disagree, consider skipping this draw)")
    print()

    # Full top-20 for reference
    print("--- Top 20 by ensemble probability (reference) ---")
    print(f"  {'Rank':>4}  {'Combo':>5}  {'Prob':>8}  {'Consensus':>9}")
    consensus_set = set(consensus_combos)
    for rank, combo in enumerate(sorted_combos[:20], 1):
        d1, d2, d3 = combo_to_digits(combo)
        flag = "  *** " if combo in consensus_set else ""
        print(f"  {rank:>4}  {combo:03d}    {pm[combo]:.6f}{flag}")


if __name__ == "__main__":
    result = run_prediction()
    print_results(result)
