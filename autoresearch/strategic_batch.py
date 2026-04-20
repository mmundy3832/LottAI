"""
strategic_batch.py - 10 hand-crafted experiments for Pick 3 Evening prediction.

Each experiment targets a different strategy informed by prior research findings.
All use the same boilerplate/evaluation framework from prepare.py.

Usage:
    python -u strategic_batch.py
"""

import sys
import os
import json
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import entropy as scipy_entropy

import prepare
from prepare import (
    load_data, get_train_val_test, build_features,
    evaluate_predictions, NUM_COMBOS, combo_to_digits,
    digits_to_combo, FEATURE_DIMS, load_cached_features,
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# -- Pre-loaded data ----------------------------------------------------------
train_df, val_df, test_df = get_train_val_test()
full_df = load_data()
train_indices = list(range(0, len(train_df)))
val_indices = list(range(len(train_df), len(train_df) + len(val_df)))
y_train = train_df["combo_int"].values
y_val = val_df["combo_int"].values
n_val = len(val_df)
n_train = len(train_df)

# Per-digit labels
y_train_d1 = y_train // 100
y_train_d2 = (y_train // 10) % 10
y_train_d3 = y_train % 10
y_val_d1 = y_val // 100
y_val_d2 = (y_val // 10) % 10
y_val_d3 = y_val % 10

# Results output directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===============================================================================
#  EXPERIMENT FUNCTIONS
#  Each returns a prob_matrix of shape (n_val, 1000).
#  On failure, returns uniform baseline.
# ===============================================================================

def uniform_baseline():
    """Return uniform probability matrix as fallback."""
    return np.full((n_val, NUM_COMBOS), 1.0 / NUM_COMBOS)


def normalize_prob_matrix(pm):
    """Ensure each row sums to 1, clamp negatives to 0."""
    pm = np.clip(pm, 0, None)
    row_sums = pm.sum(axis=1, keepdims=True)
    # Avoid division by zero - if a row is all zeros, make uniform
    zero_rows = (row_sums < 1e-15).flatten()
    pm[zero_rows] = 1.0 / NUM_COMBOS
    row_sums[zero_rows.reshape(-1, 1)] = 1.0
    pm = pm / row_sums
    return pm


# -----------------------------------------------------------------------------
# Exp 1: Per-Digit Random Forest (Independent Digits)
# -----------------------------------------------------------------------------
def exp1_per_digit_rf():
    """Train 3 separate RF classifiers for d1, d2, d3.
    Combine: P(combo) = P(d1) * P(d2) * P(d3).
    """
    X_train = load_cached_features("train", ["basic", "recency", "gaps", "positional", "temporal", "momentum"])
    X_val = load_cached_features("val", ["basic", "recency", "gaps", "positional", "temporal", "momentum"])

    digit_targets = [y_train_d1, y_train_d2, y_train_d3]
    digit_probs_val = []

    for d_idx, y_d in enumerate(digit_targets):
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_d)
        # predict_proba may not have all 10 classes if some are missing
        proba = clf.predict_proba(X_val)
        # Map to full 10-class probability
        full_proba = np.zeros((n_val, 10))
        for ci, cls_label in enumerate(clf.classes_):
            full_proba[:, cls_label] = proba[:, ci]
        digit_probs_val.append(full_proba)

    # Combine: P(combo=d1*100+d2*10+d3) = P(d1) * P(d2) * P(d3)
    prob_matrix = np.zeros((n_val, NUM_COMBOS))
    for combo in range(NUM_COMBOS):
        d1, d2, d3 = combo_to_digits(combo)
        prob_matrix[:, combo] = (
            digit_probs_val[0][:, d1] *
            digit_probs_val[1][:, d2] *
            digit_probs_val[2][:, d3]
        )

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 2: Per-Digit Gradient Boosting
# -----------------------------------------------------------------------------
def exp2_per_digit_gb():
    """Same per-digit approach but with GradientBoosting.
    Tuned hyperparams: 100 trees, max_depth=4, learning_rate=0.1.
    """
    X_train = load_cached_features("train", ["basic", "recency", "gaps", "positional", "temporal", "momentum"])
    X_val = load_cached_features("val", ["basic", "recency", "gaps", "positional", "temporal", "momentum"])

    digit_targets = [y_train_d1, y_train_d2, y_train_d3]
    digit_probs_val = []

    for d_idx, y_d in enumerate(digit_targets):
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )
        clf.fit(X_train, y_d)
        proba = clf.predict_proba(X_val)
        full_proba = np.zeros((n_val, 10))
        for ci, cls_label in enumerate(clf.classes_):
            full_proba[:, cls_label] = proba[:, ci]
        digit_probs_val.append(full_proba)

    prob_matrix = np.zeros((n_val, NUM_COMBOS))
    for combo in range(NUM_COMBOS):
        d1, d2, d3 = combo_to_digits(combo)
        prob_matrix[:, combo] = (
            digit_probs_val[0][:, d1] *
            digit_probs_val[1][:, d2] *
            digit_probs_val[2][:, d3]
        )

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 3: KNN in Feature Space
# -----------------------------------------------------------------------------
def exp3_knn_feature_space():
    """Find K nearest historical draws by feature similarity.
    Weight their outcomes. K=20, distance-weighted.
    """
    X_train = load_cached_features("train", ["basic", "recency", "gaps", "positional", "momentum"])
    X_val = load_cached_features("val", ["basic", "recency", "gaps", "positional", "momentum"])

    # Standardize features for distance computation
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    K = 20

    # For each validation draw, find K nearest training draws by feature distance
    # and weight their combo outcomes by inverse distance
    prob_matrix = np.zeros((n_val, NUM_COMBOS))

    for i in range(n_val):
        # Compute distances to all training draws
        dists = np.linalg.norm(X_train_s - X_val_s[i], axis=1)
        # Get K nearest
        nearest_idx = np.argpartition(dists, K)[:K]
        nearest_dists = dists[nearest_idx]
        # Inverse distance weights (avoid division by zero)
        weights = 1.0 / (nearest_dists + 1e-10)
        weights /= weights.sum()
        # Distribute weight to the combos that occurred
        for j, idx in enumerate(nearest_idx):
            combo = y_train[idx]
            prob_matrix[i, combo] += weights[j]

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 4: Markov Chain (Transition Probabilities)
# -----------------------------------------------------------------------------
def exp4_markov_chain():
    """Build per-digit transition matrices: P(d_next | d_prev).
    Apply to last draw's digits. Tests sequential dependency.
    """
    # Build transition matrices from training data
    # For each position, count transitions d_prev -> d_next
    transition = np.zeros((3, 10, 10))  # [position, prev_digit, next_digit]

    for i in range(1, n_train):
        prev_d1, prev_d2, prev_d3 = combo_to_digits(y_train[i - 1])
        curr_d1, curr_d2, curr_d3 = combo_to_digits(y_train[i])
        transition[0, prev_d1, curr_d1] += 1
        transition[1, prev_d2, curr_d2] += 1
        transition[2, prev_d3, curr_d3] += 1

    # Normalize to probabilities (add Laplace smoothing)
    alpha = 1.0  # Laplace smoothing
    for pos in range(3):
        for prev in range(10):
            row_sum = transition[pos, prev].sum() + 10 * alpha
            transition[pos, prev] = (transition[pos, prev] + alpha) / row_sum

    # For each validation draw, use the PREVIOUS draw's digits to predict
    # We need the draw just before each val draw
    # The last training draw is the "previous" for val[0]
    # For val[i>0], the previous draw is val[i-1]

    # Get all draws in order: last train draw + all val draws
    all_combos = np.concatenate([[y_train[-1]], y_val])

    prob_matrix = np.zeros((n_val, NUM_COMBOS))

    for i in range(n_val):
        prev_combo = all_combos[i]  # previous draw
        prev_d1, prev_d2, prev_d3 = combo_to_digits(prev_combo)

        # P(combo) = P(d1|prev_d1) * P(d2|prev_d2) * P(d3|prev_d3)
        for combo in range(NUM_COMBOS):
            d1, d2, d3 = combo_to_digits(combo)
            prob_matrix[i, combo] = (
                transition[0, prev_d1, d1] *
                transition[1, prev_d2, d2] *
                transition[2, prev_d3, d3]
            )

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 5: Bayesian Updating with Multiple Feature Sets
# -----------------------------------------------------------------------------
def exp5_bayesian_update():
    """Start with uniform prior. Sequentially update with evidence from
    each feature set using per-digit Naive Bayes classifiers.
    Multiplicative Bayesian update.
    """
    feature_groups = ["recency", "gaps", "positional", "momentum"]

    # Start with uniform prior over all combos
    prob_matrix = np.ones((n_val, NUM_COMBOS))

    for fs_name in feature_groups:
        X_train_fs = load_cached_features("train", [fs_name])
        X_val_fs = load_cached_features("val", [fs_name])

        # Train per-digit NB on this feature set
        digit_probs = []
        for y_d in [y_train_d1, y_train_d2, y_train_d3]:
            clf = GaussianNB()
            clf.fit(X_train_fs, y_d)
            proba = clf.predict_proba(X_val_fs)
            full_proba = np.zeros((n_val, 10))
            for ci, cls_label in enumerate(clf.classes_):
                full_proba[:, cls_label] = proba[:, ci]
            # Add small floor to avoid zeroing out
            full_proba = np.clip(full_proba, 1e-6, None)
            digit_probs.append(full_proba)

        # Compute likelihood for each combo from this feature set
        likelihood = np.zeros((n_val, NUM_COMBOS))
        for combo in range(NUM_COMBOS):
            d1, d2, d3 = combo_to_digits(combo)
            likelihood[:, combo] = (
                digit_probs[0][:, d1] *
                digit_probs[1][:, d2] *
                digit_probs[2][:, d3]
            )

        # Bayesian update: posterior proportional to prior * likelihood
        prob_matrix *= likelihood

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 6: Constraint Intersection (Binary Properties)
# -----------------------------------------------------------------------------
def exp6_constraint_intersection():
    """Train binary classifiers for draw properties.
    Intersect constraints to redistribute probability.
    """
    X_train_all = load_cached_features("train", ["basic", "recency", "gaps", "positional", "temporal", "momentum"])
    X_val_all = load_cached_features("val", ["basic", "recency", "gaps", "positional", "temporal", "momentum"])

    # Define binary properties of each combo
    def combo_properties(combo_int):
        d1, d2, d3 = combo_to_digits(combo_int)
        digits = [d1, d2, d3]
        return {
            "sum_high": int(sum(digits) >= 14),  # sum >= 14
            "has_repeat": int(len(set(digits)) < 3),
            "d1_high": int(d1 >= 5),
            "d2_high": int(d2 >= 5),
            "d3_high": int(d3 >= 5),
            "ascending": int(d1 <= d2 <= d3),
        }

    # Build property labels for training data
    property_names = ["sum_high", "has_repeat", "d1_high", "d2_high", "d3_high", "ascending"]
    y_props_train = {}
    for prop in property_names:
        y_props_train[prop] = np.array([combo_properties(c)[prop] for c in y_train])

    # Train a classifier for each property
    prop_probs_val = {}
    for prop in property_names:
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        clf.fit(X_train_all, y_props_train[prop])
        # P(property=1 | features)
        proba = clf.predict_proba(X_val_all)
        # Handle case where classifier only saw one class
        if len(clf.classes_) == 2:
            idx_1 = list(clf.classes_).index(1)
            prop_probs_val[prop] = proba[:, idx_1]
        else:
            # Only one class seen - use 0.5 as default
            prop_probs_val[prop] = np.full(n_val, 0.5)

    # Pre-compute properties for all 1000 combos
    combo_props = {}
    for combo in range(NUM_COMBOS):
        combo_props[combo] = combo_properties(combo)

    # For each val draw, score combos based on how well they match predicted properties
    prob_matrix = np.ones((n_val, NUM_COMBOS))

    for i in range(n_val):
        for combo in range(NUM_COMBOS):
            props = combo_props[combo]
            for prop in property_names:
                p_yes = prop_probs_val[prop][i]
                if props[prop] == 1:
                    prob_matrix[i, combo] *= p_yes
                else:
                    prob_matrix[i, combo] *= (1.0 - p_yes)

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 7: Recency-Weighted Frequency
# -----------------------------------------------------------------------------
def exp7_recency_weighted_freq():
    """Weight historical combo frequencies by recency (exponential decay).
    Recent draws matter more. Per-digit approach with decay.
    """
    # Use per-digit recency weighting for better coverage
    # (direct combo frequency is too sparse: 1000 combos, ~2725 draws)
    decay_rate = 0.005  # exponential decay parameter

    # Get all draws preceding each validation draw
    # For val[i], we can use all training data + val[0..i-1]
    all_prior_combos = np.concatenate([y_train, y_val])

    prob_matrix = np.zeros((n_val, NUM_COMBOS))

    for i in range(n_val):
        # All draws before this validation draw
        n_prior = n_train + i
        prior_combos = all_prior_combos[:n_prior]

        # Per-digit recency-weighted frequencies
        digit_probs = []
        for pos in range(3):
            if pos == 0:
                prior_digits = prior_combos // 100
            elif pos == 1:
                prior_digits = (prior_combos // 10) % 10
            else:
                prior_digits = prior_combos % 10

            # Weights: most recent draw has weight 1, older draws decay
            weights = np.exp(-decay_rate * np.arange(n_prior - 1, -1, -1))
            # Accumulate weighted counts per digit
            digit_counts = np.zeros(10)
            for d in range(10):
                digit_counts[d] = weights[prior_digits == d].sum()
            digit_counts /= digit_counts.sum()
            digit_probs.append(digit_counts)

        # Combine: P(combo) = P(d1) * P(d2) * P(d3)
        for combo in range(NUM_COMBOS):
            d1, d2, d3 = combo_to_digits(combo)
            prob_matrix[i, combo] = (
                digit_probs[0][d1] *
                digit_probs[1][d2] *
                digit_probs[2][d3]
            )

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 8: Ensemble of Diverse Weak Learners
# -----------------------------------------------------------------------------
def exp8_ensemble(exp1_pm, exp3_pm, exp4_pm, exp7_pm):
    """Combine experiments 1, 3, 4, 7 by averaging their probability distributions.
    Diversity + consensus.
    """
    prob_matrix = (exp1_pm + exp3_pm + exp4_pm + exp7_pm) / 4.0
    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 9: Conditional Frequency (Given Previous Draw)
# -----------------------------------------------------------------------------
def exp9_conditional_frequency():
    """Bin previous draws by digit sum and repeat pattern.
    Compute conditional per-digit frequencies within each bin.
    """
    def draw_bin(combo_int):
        """Classify a draw into a bin based on its properties."""
        d1, d2, d3 = combo_to_digits(combo_int)
        digit_sum = d1 + d2 + d3
        has_repeat = len(set([d1, d2, d3])) < 3
        # Bin by sum range (low/mid/high) x repeat (yes/no) = 6 bins
        if digit_sum <= 9:
            sum_bin = 0
        elif digit_sum <= 18:
            sum_bin = 1
        else:
            sum_bin = 2
        return sum_bin * 2 + int(has_repeat)

    # Build conditional per-digit frequency tables from training data
    # For each bin, track per-position digit frequencies of the NEXT draw
    n_bins = 6
    # digit_freq[bin][pos][digit] = count
    digit_freq = np.ones((n_bins, 3, 10))  # start with 1 (Laplace smoothing)

    for i in range(1, n_train):
        prev_bin = draw_bin(y_train[i - 1])
        curr_d1, curr_d2, curr_d3 = combo_to_digits(y_train[i])
        digit_freq[prev_bin, 0, curr_d1] += 1
        digit_freq[prev_bin, 1, curr_d2] += 1
        digit_freq[prev_bin, 2, curr_d3] += 1

    # Normalize to probabilities
    for b in range(n_bins):
        for pos in range(3):
            digit_freq[b, pos] /= digit_freq[b, pos].sum()

    # Apply to validation set
    all_combos = np.concatenate([[y_train[-1]], y_val])

    prob_matrix = np.zeros((n_val, NUM_COMBOS))

    for i in range(n_val):
        prev_combo = all_combos[i]
        prev_bin = draw_bin(prev_combo)

        for combo in range(NUM_COMBOS):
            d1, d2, d3 = combo_to_digits(combo)
            prob_matrix[i, combo] = (
                digit_freq[prev_bin, 0, d1] *
                digit_freq[prev_bin, 1, d2] *
                digit_freq[prev_bin, 2, d3]
            )

    return normalize_prob_matrix(prob_matrix)


# -----------------------------------------------------------------------------
# Exp 10: Anomaly-Based Exclusion (Gap-Based Redistribution)
# -----------------------------------------------------------------------------
def exp10_anomaly_exclusion():
    """Compute each digit's 'overdue' score using gap analysis per position.
    Digits overdue get probability boost, recently-drawn digits get reduced.
    Gambler's fallacy as a model.
    """
    all_combos = np.concatenate([y_train, y_val])

    prob_matrix = np.zeros((n_val, NUM_COMBOS))

    for i in range(n_val):
        n_prior = n_train + i
        prior_combos = all_combos[:n_prior]

        # For each position, compute gap since each digit last appeared
        digit_scores = []
        for pos in range(3):
            if pos == 0:
                prior_digits = prior_combos // 100
            elif pos == 1:
                prior_digits = (prior_combos // 10) % 10
            else:
                prior_digits = prior_combos % 10

            gaps = np.zeros(10)
            for d in range(10):
                # Find most recent occurrence
                occurrences = np.where(prior_digits == d)[0]
                if len(occurrences) > 0:
                    gap = n_prior - 1 - occurrences[-1]
                else:
                    gap = n_prior  # never appeared
                gaps[d] = gap

            # Expected gap for uniform: ~10 draws between appearances
            expected_gap = 10.0
            # Score = gap / expected_gap (>1 means overdue, <1 means recent)
            scores = gaps / expected_gap
            # Convert to probabilities: overdue digits get boosted
            # Use softmax-like transformation
            scores = np.exp(scores)
            scores /= scores.sum()
            digit_scores.append(scores)

        # Combine: P(combo) = P(d1) * P(d2) * P(d3)
        for combo in range(NUM_COMBOS):
            d1, d2, d3 = combo_to_digits(combo)
            prob_matrix[i, combo] = (
                digit_scores[0][d1] *
                digit_scores[1][d2] *
                digit_scores[2][d3]
            )

    return normalize_prob_matrix(prob_matrix)


# ===============================================================================
#  MAIN EXECUTION
# ===============================================================================

def run_experiment(name, func, *args):
    """Run a single experiment, return results dict or None on failure."""
    print(f"\n{'-' * 70}")
    print(f"  Running: {name}")
    print(f"{'-' * 70}")

    t0 = time.time()
    try:
        prob_matrix = func(*args)
        elapsed = time.time() - t0

        # Validate output
        assert prob_matrix.shape == (n_val, NUM_COMBOS), \
            f"Shape mismatch: {prob_matrix.shape} != ({n_val}, {NUM_COMBOS})"
        assert not np.any(np.isnan(prob_matrix)), "NaN in prob_matrix"
        assert not np.any(np.isinf(prob_matrix)), "Inf in prob_matrix"

        results = evaluate_predictions(prob_matrix, y_val)
        results["experiment"] = name
        results["execution_time_s"] = round(elapsed, 2)
        results["status"] = "success"

        rank_imp = results["vs_baseline"]["rank_improvement"]
        print(f"  DONE in {elapsed:.1f}s | mean_rank={results['mean_rank']:.2f} "
              f"| rank_improvement={rank_imp:+.2f}")

        return results, prob_matrix

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED in {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return {
            "experiment": name,
            "status": "error",
            "error": str(e),
            "execution_time_s": round(elapsed, 2),
        }, None


def main():
    print("=" * 70)
    print("  STRATEGIC BATCH: 10 Hand-Crafted Experiments")
    print("  Pick 3 Evening | Validation set: {} draws".format(n_val))
    print("  Baseline mean_rank: 500.5 (uniform)")
    print("=" * 70)

    all_results = []
    prob_matrices = {}  # Store for ensemble

    # -- Experiments 1-7 (independent) ---------------------------------------
    experiments = [
        ("Exp 1: Per-Digit Random Forest", exp1_per_digit_rf),
        ("Exp 2: Per-Digit Gradient Boosting", exp2_per_digit_gb),
        ("Exp 3: KNN Feature Space", exp3_knn_feature_space),
        ("Exp 4: Markov Chain", exp4_markov_chain),
        ("Exp 5: Bayesian Update", exp5_bayesian_update),
        ("Exp 6: Constraint Intersection", exp6_constraint_intersection),
        ("Exp 7: Recency-Weighted Frequency", exp7_recency_weighted_freq),
    ]

    for name, func in experiments:
        result, pm = run_experiment(name, func)
        all_results.append(result)
        if pm is not None:
            prob_matrices[name] = pm

    # -- Experiment 8: Ensemble (depends on 1, 3, 4, 7) ---------------------
    exp1_key = "Exp 1: Per-Digit Random Forest"
    exp3_key = "Exp 3: KNN Feature Space"
    exp4_key = "Exp 4: Markov Chain"
    exp7_key = "Exp 7: Recency-Weighted Frequency"

    if all(k in prob_matrices for k in [exp1_key, exp3_key, exp4_key, exp7_key]):
        result, pm = run_experiment(
            "Exp 8: Ensemble (1+3+4+7)", exp8_ensemble,
            prob_matrices[exp1_key],
            prob_matrices[exp3_key],
            prob_matrices[exp4_key],
            prob_matrices[exp7_key],
        )
        all_results.append(result)
    else:
        missing = [k for k in [exp1_key, exp3_key, exp4_key, exp7_key] if k not in prob_matrices]
        print(f"\n  Exp 8: SKIPPED - missing components: {missing}")
        all_results.append({
            "experiment": "Exp 8: Ensemble (1+3+4+7)",
            "status": "skipped",
            "error": f"Missing: {missing}",
        })

    # -- Experiments 9-10 (independent) --------------------------------------
    experiments_late = [
        ("Exp 9: Conditional Frequency", exp9_conditional_frequency),
        ("Exp 10: Anomaly-Based Exclusion", exp10_anomaly_exclusion),
    ]

    for name, func in experiments_late:
        result, pm = run_experiment(name, func)
        all_results.append(result)

    # ===========================================================================
    #  RESULTS SUMMARY
    # ===========================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Experiment':<40s} {'Status':<10s} {'Mean Rank':>10s} {'Rank Imp':>10s} {'Time':>8s}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
    print(f"  {'BASELINE (uniform)':<40s} {'---':<10s} {'500.50':>10s} {'---':>10s} {'---':>8s}")

    # Sort successful experiments by mean_rank
    successful = [r for r in all_results if r.get("status") == "success"]
    successful.sort(key=lambda r: r["mean_rank"])

    for r in successful:
        rank_imp = r["vs_baseline"]["rank_improvement"]
        marker = " ***" if rank_imp > 20 else " **" if rank_imp > 10 else " *" if rank_imp > 0 else ""
        print(f"  {r['experiment']:<40s} {'OK':<10s} {r['mean_rank']:>10.2f} "
              f"{rank_imp:>+10.2f} {r['execution_time_s']:>7.1f}s{marker}")

    # Show failed/skipped
    failed = [r for r in all_results if r.get("status") != "success"]
    for r in failed:
        print(f"  {r['experiment']:<40s} {r['status'].upper():<10s} {'---':>10s} {'---':>10s} {'---':>8s}")

    # Save full results
    output_file = os.path.join(RESULTS_DIR, "strategic_batch_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_file}")

    # Key stats
    if successful:
        best = successful[0]
        best_imp = best["vs_baseline"]["rank_improvement"]
        print(f"\n  BEST: {best['experiment']} (mean_rank={best['mean_rank']:.2f}, improvement={best_imp:+.2f})")
        if best_imp > 20:
            print(f"  >>> SIGNIFICANT: Best experiment beats baseline by {best_imp:.1f} ranks!")
        elif best_imp > 0:
            print(f"  >>> Marginal improvement of {best_imp:.1f} ranks over baseline.")
        else:
            print(f"  >>> No experiment beat baseline. RNG appears close to random.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
