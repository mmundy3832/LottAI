"""
digit_signal.py - Analyze per-digit predictive signal strength.

Uses the top 3 configs by optimal_ev from experiments_v3.jsonl.
All 3 share identical pipeline structure (custom_interact -> select) with 3 LGB models.
Trains on train split, evaluates on TEST split (held-out).

Front pair: (d1, d2) -- 100 combos, $50 payout
Back pair:  (d2, d3) -- 100 combos, $50 payout

Texas Pick 3:
  Straight: 1000 combos, $500 payout, $1/ticket
  Front pair: 100 combos, $50 payout, $1/ticket
  Back pair:  100 combos, $50 payout, $1/ticket
"""

import sys
import os
import json
import numpy as np

# Add autoresearch directory to path so we can import prepare.py
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from prepare import (
    load_cached_features,
    get_train_val_test,
    NUM_COMBOS,
    NUM_DIGITS,
    combo_to_digits,
    digits_to_combo,
)

import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel


# ---------------------------------------------------------------------------
# Top 3 configs (by optimal_ev=7.495 from experiments_v3.jsonl)
# Exp IDs: 635, 637, 642 -- all share same pipeline, minor model param diffs
# We use one representative config for efficiency (exp 635) and two variants.
# ---------------------------------------------------------------------------

TOP_CONFIGS = [
    {
        "name": "Exp-635 (top, cv=10 on model1)",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "ci_n_head": 9,
        "ci_n_tail": 5,
        "select_threshold": 1.4704978126202712,
        "select_n_estimators": 200,
        "select_max_depth": 14,
        "models": [
            {"type": "lgb", "weight": 0.2738010133572043, "calibrate": False,
             "n_estimators": 171, "max_depth": 3, "learning_rate": 0.0918659558251424,
             "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.954075885221076, "reg_lambda": 0.11911368506991703},
            {"type": "lgb", "weight": 0.2917179041570309, "calibrate": True, "calibrate_cv": 10,
             "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.4236841111386129,
             "reg_alpha": 0.5530684279543798, "reg_lambda": 1.0},
            {"type": "lgb", "weight": 0.18796029116563417, "calibrate": False,
             "n_estimators": 244, "max_depth": 3, "learning_rate": 0.04801448575710961,
             "subsample": 0.976089341782093, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 0.11911368506991703},
        ],
        "weights": [0.363382, 0.387161, 0.249457],
    },
    {
        "name": "Exp-637 (variant: model0 n_est=157)",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "ci_n_head": 9,
        "ci_n_tail": 5,
        "select_threshold": 1.4704978126202712,
        "select_n_estimators": 200,
        "select_max_depth": 14,
        "models": [
            {"type": "lgb", "weight": 0.2738010133572043, "calibrate": False,
             "n_estimators": 157, "max_depth": 3, "learning_rate": 0.0918659558251424,
             "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.954075885221076, "reg_lambda": 0.11911368506991703},
            {"type": "lgb", "weight": 0.2917179041570309, "calibrate": True, "calibrate_cv": 5,
             "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.4236841111386129,
             "reg_alpha": 0.5530684279543798, "reg_lambda": 1.0},
            {"type": "lgb", "weight": 0.18796029116563417, "calibrate": False,
             "n_estimators": 244, "max_depth": 3, "learning_rate": 0.04801448575710961,
             "subsample": 0.976089341782093, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 0.11911368506991703},
        ],
        "weights": [0.363382, 0.387161, 0.249457],
    },
    {
        "name": "Exp-640 (variant: model2 lr=0.03824)",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "ci_n_head": 9,
        "ci_n_tail": 5,
        "select_threshold": 1.4704978126202712,
        "select_n_estimators": 200,
        "select_max_depth": 14,
        "models": [
            {"type": "lgb", "weight": 0.2738010133572043, "calibrate": False,
             "n_estimators": 171, "max_depth": 3, "learning_rate": 0.0918659558251424,
             "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.954075885221076, "reg_lambda": 0.11911368506991703},
            {"type": "lgb", "weight": 0.2917179041570309, "calibrate": True, "calibrate_cv": 5,
             "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.4236841111386129,
             "reg_alpha": 0.5530684279543798, "reg_lambda": 1.0},
            {"type": "lgb", "weight": 0.18796029116563417, "calibrate": False,
             "n_estimators": 244, "max_depth": 3, "learning_rate": 0.03823927040392093,
             "subsample": 0.976089341782093, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 0.11911368506991703},
        ],
        "weights": [0.363382, 0.387161, 0.249457],
    },
]


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------

train_df, val_df, test_df = get_train_val_test()

y_train_d1 = train_df["d1"].values
y_train_d2 = train_df["d2"].values
y_train_d3 = train_df["d3"].values

y_test_d1 = test_df["d1"].values
y_test_d2 = test_df["d2"].values
y_test_d3 = test_df["d3"].values
y_test_combo = test_df["combo_int"].values

n_train = len(train_df)
n_test = len(test_df)

print(f"Train: {n_train} draws | Test: {n_test} draws")
print(f"Test date range: {test_df['date'].iloc[0].strftime('%Y-%m-%d')} to "
      f"{test_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print()


# ---------------------------------------------------------------------------
# Feature pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(cfg, split):
    """Apply custom_interact -> select pipeline, return transformed X."""
    X_raw = load_cached_features(split, cfg["feature_sets"])

    n_head = cfg["ci_n_head"]
    n_tail = cfg["ci_n_tail"]
    ci_n_head = min(n_head, X_raw.shape[1])
    ci_n_tail = min(n_tail, X_raw.shape[1])

    f1 = X_raw[:, :ci_n_head]
    f2 = X_raw[:, -ci_n_tail:]
    parts = []
    for i in range(ci_n_head):
        for j in range(ci_n_tail):
            parts.append(f1[:, i] * f2[:, j])
            parts.append(f1[:, i] / (f2[:, j] + 1e-6))

    ci = np.column_stack(parts) if parts else np.zeros((X_raw.shape[0], 1))
    ci = np.nan_to_num(ci, nan=0.0, posinf=1e6, neginf=-1e6)
    X_ci = np.hstack([X_raw, ci])

    return X_ci


def apply_selector(selector, X):
    return selector.transform(X)


# ---------------------------------------------------------------------------
# Per-digit metrics
# ---------------------------------------------------------------------------

def digit_metrics(probs, y_true):
    """
    probs: (N, 10) probability array
    y_true: (N,) int array of true digit values 0-9

    Returns dict with top-1/3/5 hit rates and mean_rank.
    Baseline: random = 1/10 each, mean_rank = 5.5.
    """
    N = len(y_true)
    top1 = top3 = top5 = 0
    rank_sum = 0

    for i in range(N):
        p = probs[i]
        sorted_idx = np.argsort(-p)  # descending
        true_d = int(y_true[i])

        rank_0based = int(np.where(sorted_idx == true_d)[0][0])
        rank_1based = rank_0based + 1
        rank_sum += rank_1based

        if rank_0based < 1:
            top1 += 1
        if rank_0based < 3:
            top3 += 1
        if rank_0based < 5:
            top5 += 1

    return {
        "top1_hit": top1 / N,
        "top3_hit": top3 / N,
        "top5_hit": top5 / N,
        "mean_rank": rank_sum / N,  # random baseline = 5.5
    }


# ---------------------------------------------------------------------------
# Pair EV (front pair or back pair)
# 100 possible pairs (0-9 x 0-9), payout $50, $1/ticket
# ---------------------------------------------------------------------------

PAIR_PAYOUT = 50.0

def pair_ev(probs_a, probs_b, y_a, y_b):
    """
    probs_a: (N, 10) for digit A
    probs_b: (N, 10) for digit B
    y_a, y_b: (N,) true values

    Returns dict with optimal_k, optimal_ev, and top-k ev table.
    """
    N = len(y_a)
    pair_hits_by_k = {k: 0 for k in range(1, 21)}
    pair_ranks = []

    for i in range(N):
        # Score all 100 pairs
        pair_scores = np.outer(probs_a[i], probs_b[i])  # (10, 10)
        # Flatten with index: pair (a, b) -> a*10 + b
        flat = pair_scores.ravel()  # shape (100,)
        sorted_pairs = np.argsort(-flat)  # descending

        true_pair_idx = int(y_a[i]) * 10 + int(y_b[i])
        rank_0based = int(np.where(sorted_pairs == true_pair_idx)[0][0])
        pair_ranks.append(rank_0based + 1)

        for k in range(1, 21):
            if rank_0based < k:
                pair_hits_by_k[k] += 1

    mean_rank = float(np.mean(pair_ranks))  # baseline = 50.5 for 100 pairs

    ev_by_k = {}
    for k in range(1, 21):
        hit_rate = pair_hits_by_k[k] / N
        ev = hit_rate * PAIR_PAYOUT - k  # spend $k, win $50 if hit
        ev_by_k[k] = ev

    optimal_k = max(ev_by_k, key=ev_by_k.get)
    optimal_ev = ev_by_k[optimal_k]

    return {
        "mean_rank": mean_rank,
        "optimal_k": optimal_k,
        "optimal_ev": optimal_ev,
        "ev_by_k": ev_by_k,
        "hit_rate_by_k": {k: pair_hits_by_k[k] / N for k in range(1, 21)},
    }


# ---------------------------------------------------------------------------
# Straight EV on test set (for comparison)
# 1000 combos, $500 payout, $1/ticket
# ---------------------------------------------------------------------------

STRAIGHT_PAYOUT = 500.0

def straight_ev(digit_probs_combined, y_test_combo):
    """Compute straight bet EV on test set using combined digit probs."""
    N = len(y_test_combo)

    # Build prob matrix (N, 1000)
    prob_matrix = np.zeros((N, NUM_COMBOS))
    for c in range(NUM_COMBOS):
        d1, d2, d3 = combo_to_digits(c)
        prob_matrix[:, c] = (
            digit_probs_combined[0][:, d1] *
            digit_probs_combined[1][:, d2] *
            digit_probs_combined[2][:, d3]
        )

    # Normalize
    prob_matrix = np.clip(prob_matrix, 0, None)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-15] = 1.0
    prob_matrix = prob_matrix / row_sums

    # Compute top-K hit rates and EV for K=1..20
    hits_by_k = {k: 0 for k in range(1, 21)}
    for i in range(N):
        sorted_combos = np.argsort(-prob_matrix[i])
        true_combo = int(y_test_combo[i])
        rank_0based = int(np.where(sorted_combos == true_combo)[0][0])
        for k in range(1, 21):
            if rank_0based < k:
                hits_by_k[k] += 1

    ev_by_k = {}
    for k in range(1, 21):
        hit_rate = hits_by_k[k] / N
        ev = hit_rate * STRAIGHT_PAYOUT - k
        ev_by_k[k] = ev

    optimal_k = max(ev_by_k, key=ev_by_k.get)
    optimal_ev = ev_by_k[optimal_k]
    exact_hit = hits_by_k[1] / N

    return {
        "exact_hit_rate": exact_hit,
        "ev_per_dollar_top1": exact_hit * STRAIGHT_PAYOUT - 1.0,
        "optimal_k": optimal_k,
        "optimal_ev": optimal_ev,
        "ev_by_k": ev_by_k,
    }


# ---------------------------------------------------------------------------
# Train and evaluate one config
# ---------------------------------------------------------------------------

def run_config(cfg):
    print(f"\n{'='*70}")
    print(f"CONFIG: {cfg['name']}")
    print(f"{'='*70}")

    # --- Build feature pipeline for train ---
    print("  Building train features...")
    X_train_ci = build_pipeline(cfg, "train")

    # --- Feature selection (fit on train) ---
    print("  Fitting feature selector (ET, d1 target)...")
    sel_et = ExtraTreesClassifier(
        n_estimators=cfg["select_n_estimators"],
        max_depth=cfg["select_max_depth"],
        random_state=42, n_jobs=-1
    )
    sel_et.fit(X_train_ci, y_train_d1)
    selector = SelectFromModel(
        sel_et, prefit=True,
        threshold=f"{cfg['select_threshold']}*mean"
    )
    X_train = selector.transform(X_train_ci)
    print(f"  Features after selection: {X_train.shape[1]}")

    # --- Build feature pipeline for test ---
    print("  Building test features...")
    X_test_ci = build_pipeline(cfg, "test")
    X_test = selector.transform(X_test_ci)

    # --- Train per-digit models ---
    digit_targets_train = [y_train_d1, y_train_d2, y_train_d3]
    digit_targets_test = [y_test_d1, y_test_d2, y_test_d3]
    digit_names = ["d1", "d2", "d3"]

    model_probs_list = []  # one entry per model-spec, each is list of 3 (N_test, 10) arrays

    for m_idx, m_cfg in enumerate(cfg["models"]):
        print(f"  Training model {m_idx} (lgb, n_est={m_cfg['n_estimators']}, "
              f"calibrate={m_cfg['calibrate']})...")
        digit_probs = []
        for d_idx, y_d in enumerate(digit_targets_train):
            base = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=10,
                n_estimators=m_cfg["n_estimators"],
                max_depth=m_cfg["max_depth"],
                learning_rate=m_cfg["learning_rate"],
                subsample=m_cfg["subsample"],
                colsample_bytree=m_cfg["colsample_bytree"],
                reg_alpha=m_cfg["reg_alpha"],
                reg_lambda=m_cfg["reg_lambda"],
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            if m_cfg["calibrate"]:
                cv = m_cfg.get("calibrate_cv", 5)
                clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv)
            else:
                clf = base

            clf.fit(X_train, y_d)
            proba = clf.predict_proba(X_test)

            # Map to fixed (N, 10) array
            fp = np.zeros((n_test, 10))
            for ci, cls in enumerate(clf.classes_):
                fp[:, int(cls)] = proba[:, ci]
            digit_probs.append(fp)

        model_probs_list.append(digit_probs)

    # --- Weighted ensemble ---
    weights = cfg["weights"]
    # Normalize weights to sum to 1
    w_sum = sum(weights[:len(cfg["models"])])
    ws = [w / w_sum for w in weights[:len(cfg["models"])]]

    digit_probs_combined = []
    for d in range(3):
        combined = sum(ws[m] * model_probs_list[m][d] for m in range(len(cfg["models"])))
        digit_probs_combined.append(combined)

    # --- Per-digit metrics ---
    print("\n  --- Per-Digit Signal on TEST set ---")
    print(f"  {'Digit':<8} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'MeanRank':>10}  "
          f"(random: 10.0%, 30.0%, 50.0%, rank=5.5)")
    print(f"  {'-'*60}")

    digit_results = {}
    for d_idx in range(3):
        m = digit_metrics(digit_probs_combined[d_idx], digit_targets_test[d_idx])
        digit_results[digit_names[d_idx]] = m
        lift_top1 = m["top1_hit"] / 0.10
        print(f"  {digit_names[d_idx]:<8} "
              f"{m['top1_hit']:>7.2%} "
              f"{m['top3_hit']:>7.2%} "
              f"{m['top5_hit']:>7.2%} "
              f"{m['mean_rank']:>10.3f}  "
              f"(top1 lift: {lift_top1:.2f}x)")

    # --- Front pair (d1, d2) ---
    fp_result = pair_ev(
        digit_probs_combined[0], digit_probs_combined[1],
        y_test_d1, y_test_d2
    )

    # --- Back pair (d2, d3) ---
    bp_result = pair_ev(
        digit_probs_combined[1], digit_probs_combined[2],
        y_test_d2, y_test_d3
    )

    # --- Straight bet EV ---
    print("  Computing straight bet EV on test set...")
    st_result = straight_ev(digit_probs_combined, y_test_combo)

    # --- Summary ---
    print(f"\n  --- Pair & Straight EV on TEST set ---")
    print(f"  {'Bet Type':<16} {'MeanRank':>10} {'Opt-K':>7} {'Opt-EV':>10} {'EV/$ ratio':>12}")
    print(f"  {'-'*58}")

    # Front pair: EV/$ = optimal_ev / optimal_k (per dollar spent)
    fp_ev_per_dollar = fp_result["optimal_ev"] / max(fp_result["optimal_k"], 1)
    bp_ev_per_dollar = bp_result["optimal_ev"] / max(bp_result["optimal_k"], 1)
    st_ev_per_dollar = st_result["optimal_ev"] / max(st_result["optimal_k"], 1)

    fp_baseline_rank = 50.5  # 100 pairs, random
    bp_baseline_rank = 50.5
    st_baseline_rank = 500.5  # 1000 combos, random

    print(f"  {'FrontPair (d1d2)':<16} "
          f"{fp_result['mean_rank']:>9.1f} "
          f"  (vs baseline {fp_baseline_rank:.1f})")
    print(f"  {'  opt_k':<16} {'':<10} {fp_result['optimal_k']:>7} "
          f"{fp_result['optimal_ev']:>10.3f} {fp_ev_per_dollar:>12.3f}")

    print(f"  {'BackPair (d2d3)':<16} "
          f"{bp_result['mean_rank']:>9.1f} "
          f"  (vs baseline {bp_baseline_rank:.1f})")
    print(f"  {'  opt_k':<16} {'':<10} {bp_result['optimal_k']:>7} "
          f"{bp_result['optimal_ev']:>10.3f} {bp_ev_per_dollar:>12.3f}")

    print(f"  {'Straight':<16} "
          f"{'':<10} "
          f"{st_result['optimal_k']:>7} "
          f"{st_result['optimal_ev']:>10.3f} "
          f"{st_ev_per_dollar:>12.3f}")
    print(f"    (straight exact hit: {st_result['exact_hit_rate']:.4%}, "
          f"EV/$ top-1: {st_result['ev_per_dollar_top1']:.3f})")

    # --- EV comparison: pair vs straight normalized to $1 per draw spent ---
    # Front pair top-1: spend $1, win $50 if hit
    # Straight top-1: spend $1, win $500 if hit
    fp_top1_ev = fp_result["hit_rate_by_k"][1] * PAIR_PAYOUT - 1.0
    bp_top1_ev = bp_result["hit_rate_by_k"][1] * PAIR_PAYOUT - 1.0
    st_top1_ev = st_result["ev_per_dollar_top1"]

    print(f"\n  --- Single-ticket EV (top-1 pick, $1 bet) ---")
    print(f"  Front pair top-1: {fp_result['hit_rate_by_k'][1]:.3%} * $50 - $1 = ${fp_top1_ev:+.3f}")
    print(f"  Back  pair top-1: {bp_result['hit_rate_by_k'][1]:.3%} * $50 - $1 = ${bp_top1_ev:+.3f}")
    print(f"  Straight  top-1: {st_result['exact_hit_rate']:.3%} * $500 - $1 = ${st_top1_ev:+.3f}")

    # --- Front pair EV table (top 5 K) ---
    print(f"\n  --- Front Pair EV by K ---")
    for k in range(1, 11):
        hr = fp_result["hit_rate_by_k"][k]
        ev = fp_result["ev_by_k"][k]
        marker = " <-- optimal" if k == fp_result["optimal_k"] else ""
        print(f"  K={k:2d}: hit={hr:.3%}  EV=${ev:+.3f}{marker}")

    print(f"\n  --- Back Pair EV by K ---")
    for k in range(1, 11):
        hr = bp_result["hit_rate_by_k"][k]
        ev = bp_result["ev_by_k"][k]
        marker = " <-- optimal" if k == bp_result["optimal_k"] else ""
        print(f"  K={k:2d}: hit={hr:.3%}  EV=${ev:+.3f}{marker}")

    return {
        "config_name": cfg["name"],
        "digit_results": digit_results,
        "front_pair": fp_result,
        "back_pair": bp_result,
        "straight": st_result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = []

    for cfg in TOP_CONFIGS:
        result = run_config(cfg)
        all_results.append(result)

    # --- Aggregate summary across configs ---
    print(f"\n\n{'='*70}")
    print("  AGGREGATE SUMMARY (averaged across 3 top configs)")
    print(f"{'='*70}")

    digit_names = ["d1", "d2", "d3"]
    for d_name in digit_names:
        top1_vals = [r["digit_results"][d_name]["top1_hit"] for r in all_results]
        top3_vals = [r["digit_results"][d_name]["top3_hit"] for r in all_results]
        top5_vals = [r["digit_results"][d_name]["top5_hit"] for r in all_results]
        mr_vals = [r["digit_results"][d_name]["mean_rank"] for r in all_results]
        print(f"  {d_name}: avg_top1={np.mean(top1_vals):.2%} "
              f"avg_top3={np.mean(top3_vals):.2%} "
              f"avg_top5={np.mean(top5_vals):.2%} "
              f"avg_rank={np.mean(mr_vals):.3f}  "
              f"(random: top1=10%, top3=30%, top5=50%, rank=5.5)")

    # --- Which digit position has strongest signal? ---
    d1_top1 = np.mean([r["digit_results"]["d1"]["top1_hit"] for r in all_results])
    d2_top1 = np.mean([r["digit_results"]["d2"]["top1_hit"] for r in all_results])
    d3_top1 = np.mean([r["digit_results"]["d3"]["top1_hit"] for r in all_results])
    d1_rank = np.mean([r["digit_results"]["d1"]["mean_rank"] for r in all_results])
    d2_rank = np.mean([r["digit_results"]["d2"]["mean_rank"] for r in all_results])
    d3_rank = np.mean([r["digit_results"]["d3"]["mean_rank"] for r in all_results])

    top1_vals_all = {"d1": d1_top1, "d2": d2_top1, "d3": d3_top1}
    best_digit = max(top1_vals_all, key=top1_vals_all.get)
    rank_vals_all = {"d1": d1_rank, "d2": d2_rank, "d3": d3_rank}
    best_rank_digit = min(rank_vals_all, key=rank_vals_all.get)  # lower rank = better

    print(f"\n  Strongest signal by top-1 hit rate: {best_digit} ({top1_vals_all[best_digit]:.2%})")
    print(f"  Strongest signal by mean rank: {best_rank_digit} ({rank_vals_all[best_rank_digit]:.3f})")

    # --- Pair comparison ---
    fp_opt_evs = [r["front_pair"]["optimal_ev"] for r in all_results]
    bp_opt_evs = [r["back_pair"]["optimal_ev"] for r in all_results]
    st_opt_evs = [r["straight"]["optimal_ev"] for r in all_results]
    fp_mr = [r["front_pair"]["mean_rank"] for r in all_results]
    bp_mr = [r["back_pair"]["mean_rank"] for r in all_results]

    print(f"\n  Avg front pair optimal_ev: ${np.mean(fp_opt_evs):.3f}  "
          f"(mean_rank {np.mean(fp_mr):.1f} vs baseline 50.5)")
    print(f"  Avg back  pair optimal_ev: ${np.mean(bp_opt_evs):.3f}  "
          f"(mean_rank {np.mean(bp_mr):.1f} vs baseline 50.5)")
    print(f"  Avg straight   optimal_ev: ${np.mean(st_opt_evs):.3f}  "
          f"(1000 combos, $500 payout)")

    # --- EV per dollar comparison ---
    # Normalize: pair bets $1/ticket with $50 payout; straight $1 with $500 payout
    # To compare fairly: EV per dollar on top-1 bet
    fp_top1_evs = [
        r["front_pair"]["hit_rate_by_k"][1] * PAIR_PAYOUT - 1.0
        for r in all_results
    ]
    bp_top1_evs = [
        r["back_pair"]["hit_rate_by_k"][1] * PAIR_PAYOUT - 1.0
        for r in all_results
    ]
    st_top1_evs = [r["straight"]["ev_per_dollar_top1"] for r in all_results]

    print(f"\n  Top-1 single-ticket EV (per $1 spent):")
    print(f"    Front pair: ${np.mean(fp_top1_evs):+.3f}")
    print(f"    Back  pair: ${np.mean(bp_top1_evs):+.3f}")
    print(f"    Straight:   ${np.mean(st_top1_evs):+.3f}")

    winner = "Front pair" if np.mean(fp_top1_evs) > np.mean(bp_top1_evs) else "Back pair"
    best_pair_ev = max(np.mean(fp_top1_evs), np.mean(bp_top1_evs))
    straight_ev_val = np.mean(st_top1_evs)

    if best_pair_ev > straight_ev_val:
        print(f"\n  CONCLUSION: {winner} has BETTER single-ticket EV than straight bet")
        print(f"    Pair: ${best_pair_ev:+.3f} vs Straight: ${straight_ev_val:+.3f} per $1")
    else:
        print(f"\n  CONCLUSION: Straight bet has BETTER single-ticket EV than pairs")
        print(f"    Straight: ${straight_ev_val:+.3f} vs best pair: ${best_pair_ev:+.3f} per $1")

    print(f"\n  Note: All EVs on held-out TEST set ({n_test} draws). "
          f"Positive EV does not guarantee future profits.")
    print(f"{'='*70}")
    print("DONE")
