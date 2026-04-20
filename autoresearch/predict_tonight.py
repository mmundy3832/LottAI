"""
predict_tonight.py - Predict tonight's Pick 3 Evening draw.

Trains the top live-generalizing ensemble on ALL available data
(historical + live draws through April 11 2026), then predicts
draw #3943 (April 12 2026 Evening).

Strategy: product ensemble of the two best distinct architectures
found by live_eval.py -- weighted by their live_optimal_ev scores.
"""

import sys, os, warnings, json
warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd

import prepare
from prepare import (load_data, build_features, evaluate_predictions,
                     NUM_COMBOS, combo_to_digits, FEATURE_DIMS)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel as SFM
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Load all available data + add tonight placeholder
# ---------------------------------------------------------------------------

_LIVE_PATH = os.path.join(_DIR, "..", "pick3evening_live.csv")

full_df = load_data()  # 3893 historical draws through Feb 13 2026

live_raw = pd.read_csv(_LIVE_PATH, header=None,
    names=["game","month","day","year","d1","d2","d3","sum_col","trailing"], dtype=str)
live_raw = live_raw.dropna(subset=["year"]).reset_index(drop=True)
live_raw["date"] = pd.to_datetime(
    live_raw["year"].str.strip() + "-" + live_raw["month"].str.strip() + "-" + live_raw["day"].str.strip(),
    format="%Y-%m-%d")
for c in ["d1","d2","d3"]:
    live_raw[c] = live_raw[c].astype(int)
live_raw["combo"]     = live_raw["d1"].apply(str) + live_raw["d2"].apply(str) + live_raw["d3"].apply(str)
live_raw["combo_int"] = live_raw["d1"]*100 + live_raw["d2"]*10 + live_raw["d3"]
live_raw["day_of_week"] = live_raw["date"].dt.dayofweek
live_raw["month"]     = live_raw["date"].dt.month
live_raw["year"]      = live_raw["date"].dt.year
live_raw = live_raw[["date","d1","d2","d3","combo","combo_int","day_of_week","month","year"]].copy()

# Tonight placeholder (temporal features need the date; digit values unused for features)
tonight_row = pd.DataFrame([{
    "date": pd.Timestamp("2026-04-12"),
    "d1": 0, "d2": 0, "d3": 0,
    "combo": "000", "combo_int": 0,
    "day_of_week": 6,   # Sunday
    "month": 4,
    "year": 2026,
}])

combined_df  = pd.concat([full_df, live_raw, tonight_row], ignore_index=True)
n_history    = len(combined_df) - 1      # 3942 known draws
predict_idx  = n_history                 # index 3942 = tonight
all_indices  = list(range(n_history))    # 0..3941 -- all known draws

y_all    = combined_df["combo_int"].values[:n_history]
y_all_d1 = y_all // 100
y_all_d2 = (y_all // 10) % 10
y_all_d3 = y_all % 10

print(f"Training on {n_history} draws  (through {combined_df['date'].iloc[n_history-1].strftime('%Y-%m-%d')})")
print(f"Predicting:  draw #{n_history+1}  ({combined_df['date'].iloc[predict_idx].strftime('%Y-%m-%d %A')})")
print()


# ---------------------------------------------------------------------------
# Helper: build + transform features for all history + tonight
# ---------------------------------------------------------------------------

def build_split(feature_sets):
    """Return X_all (n_history, F) and X_tonight (1, F)."""
    X_all     = build_features(combined_df, all_indices,    feature_sets)
    X_tonight = build_features(combined_df, [predict_idx],  feature_sets)
    return X_all, X_tonight


def apply_custom_interact(X, n_head, n_tail, include_ratios):
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


# ---------------------------------------------------------------------------
# Config A: exp#767 -- top live scorer ($28.82, buy 12)
#   features: basic, recency, gaps, temporal, momentum (no positional)
#   pipeline: custom_interact(9,5,ratios) -> select(1.47x mean, d1, 200est, depth14)
#   models:   3x LGB depth=3 (shallow)
# ---------------------------------------------------------------------------

def run_config_a():
    print("=== Config A (exp#767): 5-feature, interact->select, 3x LGB depth=3 ===")
    feature_sets = ["basic", "recency", "gaps", "temporal", "momentum"]
    X_all_raw, X_ton_raw = build_split(feature_sets)

    # custom_interact
    X_all_ci  = apply_custom_interact(X_all_raw, n_head=9, n_tail=5, include_ratios=True)
    X_ton_ci  = apply_custom_interact(X_ton_raw, n_head=9, n_tail=5, include_ratios=True)

    # feature selection (fit on all history, target d1)
    _sel_et = ExtraTreesClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
    _sel_et.fit(X_all_ci, y_all_d1)
    _selector = SFM(_sel_et, prefit=True, threshold="1.4704978126202712*mean")
    X_all_sel = _selector.transform(X_all_ci)
    X_ton_sel = _selector.transform(X_ton_ci)
    print(f"  Features after select: {X_all_sel.shape[1]}")

    # 3x LGB per digit
    configs = [
        dict(n_estimators=157, max_depth=3, learning_rate=0.0918659558251424,
             subsample=0.5385712959287365, colsample_bytree=0.7,
             reg_alpha=0.954075885221076, reg_lambda=0.11911368506991703,
             calibrate=False, weight=0.2738010133572043),
        dict(n_estimators=300, max_depth=3, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.4236841111386129,
             reg_alpha=2.0665963127424742, reg_lambda=1.0,
             calibrate=True, weight=0.2917179041570309),
        dict(n_estimators=244, max_depth=3, learning_rate=0.04801448575710961,
             subsample=0.8614624578010438, colsample_bytree=0.49040221862411426,
             reg_alpha=0.0, reg_lambda=0.8986918177871955,
             calibrate=False, weight=0.18796029116563417),
    ]
    total_w = sum(c["weight"] for c in configs)

    digit_probs = []  # 3 arrays of shape (1, 10)
    for d_idx, y_d in enumerate([y_all_d1, y_all_d2, y_all_d3]):
        weighted_prob = np.zeros((1, 10))
        for cfg in configs:
            clf = lgb.LGBMClassifier(
                objective="multiclass", num_class=10,
                n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"], subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                reg_alpha=cfg["reg_alpha"], reg_lambda=cfg["reg_lambda"],
                random_state=42, n_jobs=-1, verbose=-1)
            if cfg["calibrate"]:
                clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
            clf.fit(X_all_sel, y_d)
            proba = clf.predict_proba(X_ton_sel)
            fp = np.zeros((1, 10))
            for ci, cls in enumerate(clf.classes_):
                fp[:, int(cls)] = proba[:, ci]
            weighted_prob += (cfg["weight"] / total_w) * fp
        digit_probs.append(weighted_prob)
        top3 = np.argsort(-weighted_prob[0])[:3]
        print(f"  d{d_idx+1} top-3: {list(top3)}  probs: {[f'{weighted_prob[0,x]:.4f}' for x in top3]}")

    prob_matrix = np.zeros((1, NUM_COMBOS))
    for combo in range(NUM_COMBOS):
        d1, d2, d3 = combo_to_digits(combo)
        prob_matrix[0, combo] = digit_probs[0][0, d1] * digit_probs[1][0, d2] * digit_probs[2][0, d3]
    prob_matrix = np.clip(prob_matrix, 0, None)
    prob_matrix /= prob_matrix.sum()

    print()
    return prob_matrix, digit_probs


# ---------------------------------------------------------------------------
# Config B: exp#499 -- #4 live scorer ($26.82, buy 14)
#   features: all 6 (including positional)
#   pipeline: custom_interact(9,5,ratios) -> select(1.47x,d1,200est,depth14) -> scale(robust)
#   models:   LGB(depth=3) + LGB(depth=6) + LGB(depth=3)
# ---------------------------------------------------------------------------

def run_config_b():
    print("=== Config B (exp#499): 6-feature, interact->select->scale, 3x LGB ===")
    feature_sets = ["basic", "recency", "gaps", "positional", "temporal", "momentum"]
    X_all_raw, X_ton_raw = build_split(feature_sets)

    # custom_interact
    X_all_ci  = apply_custom_interact(X_all_raw, n_head=9, n_tail=5, include_ratios=True)
    X_ton_ci  = apply_custom_interact(X_ton_raw, n_head=9, n_tail=5, include_ratios=True)

    # feature selection
    _sel_et = ExtraTreesClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
    _sel_et.fit(X_all_ci, y_all_d1)
    _selector = SFM(_sel_et, prefit=True, threshold="1.4704978126202712*mean")
    X_all_sel = _selector.transform(X_all_ci)
    X_ton_sel = _selector.transform(X_ton_ci)

    # robust scale
    _scaler = RobustScaler()
    X_all_sc = _scaler.fit_transform(X_all_sel)
    X_ton_sc = _scaler.transform(X_ton_sel)
    print(f"  Features after select+scale: {X_all_sc.shape[1]}")

    configs = [
        dict(n_estimators=244, max_depth=3, learning_rate=0.0918659558251424,
             subsample=0.5385712959287365, colsample_bytree=0.7,
             reg_alpha=0.954075885221076, reg_lambda=0.11911368506991703,
             calibrate=False, weight=0.2738010133572043),
        dict(n_estimators=300, max_depth=6, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.4236841111386129,
             reg_alpha=2.0665963127424742, reg_lambda=1.0,
             calibrate=False, weight=0.2917179041570309),
        dict(n_estimators=244, max_depth=3, learning_rate=0.04801448575710961,
             subsample=0.8614624578010438, colsample_bytree=0.49040221862411426,
             reg_alpha=0.0, reg_lambda=0.8986918177871955,
             calibrate=False, weight=0.18796029116563417),
    ]
    total_w = sum(c["weight"] for c in configs)

    digit_probs = []
    for d_idx, y_d in enumerate([y_all_d1, y_all_d2, y_all_d3]):
        weighted_prob = np.zeros((1, 10))
        for cfg in configs:
            clf = lgb.LGBMClassifier(
                objective="multiclass", num_class=10,
                n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"], subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                reg_alpha=cfg["reg_alpha"], reg_lambda=cfg["reg_lambda"],
                random_state=42, n_jobs=-1, verbose=-1)
            clf.fit(X_all_sc, y_d)
            proba = clf.predict_proba(X_ton_sc)
            fp = np.zeros((1, 10))
            for ci, cls in enumerate(clf.classes_):
                fp[:, int(cls)] = proba[:, ci]
            weighted_prob += (cfg["weight"] / total_w) * fp
        digit_probs.append(weighted_prob)
        top3 = np.argsort(-weighted_prob[0])[:3]
        print(f"  d{d_idx+1} top-3: {list(top3)}  probs: {[f'{weighted_prob[0,x]:.4f}' for x in top3]}")

    prob_matrix = np.zeros((1, NUM_COMBOS))
    for combo in range(NUM_COMBOS):
        d1, d2, d3 = combo_to_digits(combo)
        prob_matrix[0, combo] = digit_probs[0][0, d1] * digit_probs[1][0, d2] * digit_probs[2][0, d3]
    prob_matrix = np.clip(prob_matrix, 0, None)
    prob_matrix /= prob_matrix.sum()

    print()
    return prob_matrix, digit_probs


# ---------------------------------------------------------------------------
# Run both configs, combine via product ensemble
# ---------------------------------------------------------------------------

pm_a, dp_a = run_config_a()
pm_b, dp_b = run_config_b()

# Product ensemble (multiply, renormalize) -- same method as consensus.py
# Weight by live_optimal_ev: A=28.816, B=26.816
w_a = 28.816
w_b = 26.816
total_w = w_a + w_b

# Weighted product in log space for numerical stability
log_pm = (w_a * np.log(pm_a + 1e-20) + w_b * np.log(pm_b + 1e-20)) / total_w
pm_ensemble = np.exp(log_pm)
pm_ensemble = np.clip(pm_ensemble, 0, None)
pm_ensemble /= pm_ensemble.sum()

# Per-digit ensemble (for reference)
digit_ensemble = []
for d in range(3):
    combined = (w_a * dp_a[d] + w_b * dp_b[d]) / total_w
    digit_ensemble.append(combined)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print("=" * 60)
print("  TONIGHT'S PREDICTIONS  (April 12 2026 -- Sunday Evening)")
print("=" * 60)
print()

sorted_combos = np.argsort(-pm_ensemble[0])

print("--- Per-digit breakdown ---")
pos_names = ["Position 1", "Position 2", "Position 3"]
for d, name in enumerate(pos_names):
    probs = digit_ensemble[d][0]
    top5  = np.argsort(-probs)[:5]
    print(f"  {name}: {list(top5)}  (probs: {[f'{probs[x]:.4f}' for x in top5]})")
print()

print("--- Top combos (ensemble, buy these straight) ---")
print(f"  {'Rank':>4}  {'Combo':>5}  {'Prob':>8}  {'d1 d2 d3'}")
print(f"  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*8}")
for rank, combo in enumerate(sorted_combos[:20], 1):
    d1, d2, d3 = combo_to_digits(combo)
    prob = pm_ensemble[0, combo]
    print(f"  {rank:>4}  {combo:05d}  {prob:.6f}  {d1}  {d2}  {d3}")

print()

# EV analysis -- how many should you buy?
print("--- Expected value by ticket count ---")
print(f"  {'Buy':>4}  {'Hit%':>6}  {'EV/draw':>8}  combos")
cumulative_prob = 0.0
for k in [1, 2, 3, 5, 7, 10, 12, 15, 20]:
    cumulative_prob = pm_ensemble[0, sorted_combos[:k]].sum()
    ev = cumulative_prob * 500 - k
    top_combos = [f"{sorted_combos[i]:03d}" for i in range(min(k, 5))]
    suffix = "..." if k > 5 else ""
    print(f"  {k:>4}  {100*cumulative_prob:>5.2f}%  ${ev:>+7.3f}  {' '.join(top_combos)}{suffix}")

print()

# Best K recommendation
best_ev = float("-inf")
best_k  = 1
for k in range(1, 21):
    prob_k = pm_ensemble[0, sorted_combos[:k]].sum()
    ev_k   = prob_k * 500 - k
    if ev_k > best_ev:
        best_ev = ev_k
        best_k  = k

print(f"--- RECOMMENDATION ---")
print(f"  Buy top {best_k} tickets  (model-optimal K)")
print(f"  Projected EV per draw: ${best_ev:+.3f}")
print()
print(f"  PLAYS (straight, $1 each):")
for i in range(best_k):
    combo = sorted_combos[i]
    d1, d2, d3 = combo_to_digits(combo)
    prob = pm_ensemble[0, combo]
    print(f"    {i+1:>2}. {combo:03d}   ({d1}-{d2}-{d3})   p={prob:.5f}")
