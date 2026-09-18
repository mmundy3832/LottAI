"""
Multivariate Hawkes Process Model for Lotto Texas Cross-Excitation Analysis
============================================================================

Models each of 54 lottery numbers as a dimension in a multivariate Hawkes process:
    lambda_i(t) = mu_i + sum_j sum_{t_k < t} alpha_ij * exp(-beta_ij * (t - t_k))

Primary approach: tick library (HawkesExpKern with L1 regularization)
Fallback: Discrete-time logistic regression with exponential decay weights

Outputs:
  - results/hawkes_excitation_matrix.json
  - results/hawkes_top_pairs.json
  - results/hawkes_predictions.json
  - results/hawkes_excitation_heatmap.png
  - results/hawkes_network.png
  - results/hawkes_self_excitation_vs_runs.png

Run: .venv/Scripts/python.exe -u hawkes_model.py
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # logistic sigmoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lottotexas.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
NUM_NUMBERS = 54          # Lotto Texas: pick 6 from 1-54
PICKS_PER_DRAW = 6
HOLDOUT_DRAWS = 400       # last 400 draws for evaluation
LOOKBACK_DRAWS = 3        # for discrete-time fallback: check t+1, t+2, t+3
DECAY_BETA_DEFAULT = 1.0  # default exponential decay rate for discrete model
L1_C = 0.1                # inverse regularization strength for logistic regression
TOP_K = 20                # top-K excitation / inhibition pairs to report

def log(msg):
    """Timestamped logging to stdout."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    """Load Lotto Texas CSV (no header).
    Columns: GameName, Month, Day, Year, Num1, Num2, Num3, Num4, Num5, Num6
    Returns DataFrame with date index and columns num1..num6.
    """
    log("Loading data from " + CSV_PATH)
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        names=['GameName', 'Month', 'Day', 'Year',
               'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    )
    df['Date'] = pd.to_datetime(
        df[['Year', 'Month', 'Day']].rename(
            columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}
        )
    )
    df = df.sort_values('Date').reset_index(drop=True)
    log(f"Loaded {len(df)} draws from {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")

    # Filter out pre-April 2006 draws: game changed from pick-6-from-50 to pick-6-from-54
    pre_filter_count = len(df)
    cutoff_date = pd.Timestamp('2006-04-01')
    df = df[df['Date'] >= cutoff_date].reset_index(drop=True)
    filtered_out = pre_filter_count - len(df)
    log(f"Filtered out {filtered_out} pre-2006 draws (before {cutoff_date.date()}), {len(df)} draws remaining")
    return df


def build_binary_matrix(df):
    """Build (n_draws x 54) binary matrix: 1 if number appeared in that draw."""
    n = len(df)
    mat = np.zeros((n, NUM_NUMBERS), dtype=np.int8)
    for idx in range(n):
        for col in ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']:
            num = int(df.iloc[idx][col])
            if 1 <= num <= NUM_NUMBERS:
                mat[idx, num - 1] = 1
    return mat


def build_timestamps(df):
    """Convert draw dates to float timestamps (days since first draw).
    Returns array of shape (n_draws,).
    """
    dates = df['Date'].values
    t0 = dates[0]
    # Convert to days as float
    deltas = (dates - t0) / np.timedelta64(1, 'D')
    return deltas.astype(np.float64)


# ---------------------------------------------------------------------------
# Approach 1: tick-based Hawkes (Primary)
# ---------------------------------------------------------------------------
def try_tick_hawkes(binary_matrix, timestamps, train_end_idx):
    """Attempt to fit using tick library. Returns excitation matrix or None."""
    try:
        from tick.hawkes import HawkesExpKern
        log("tick library found. Attempting tick-based Hawkes process fit...")
    except ImportError:
        log("WARNING: tick library not available. Falling back to discrete-time approach.")
        return None

    try:
        # Build event lists: for each number, list of timestamps where it appeared
        train_mat = binary_matrix[:train_end_idx]
        train_ts = timestamps[:train_end_idx]

        events = []
        for j in range(NUM_NUMBERS):
            event_times = train_ts[train_mat[:, j] == 1]
            events.append(event_times.astype(np.float64))

        # Fit Hawkes with exponential kernel + L1 regularization
        # Use a reasonable decay (beta=1 means ~1 day half-life equivalent)
        decay = 0.3  # slower decay to capture multi-draw effects
        log(f"Fitting HawkesExpKern with decay={decay}, C=1e-2 (L1 penalized)...")
        log(f"  {NUM_NUMBERS} dimensions, {train_end_idx} training draws")

        hawkes = HawkesExpKern(
            decays=decay,
            penalty='l1',
            C=1e-2,
            max_iter=500,
            tol=1e-5,
            verbose=False
        )
        hawkes.fit(events, end_times=train_ts[train_end_idx - 1] + 1.0)

        # Extract adjacency (excitation) matrix
        alpha = hawkes.adjacency.copy()  # shape (54, 54)
        baseline = hawkes.baseline.copy()  # shape (54,)

        log(f"tick Hawkes fit complete. Adjacency matrix shape: {alpha.shape}")
        log(f"  Non-zero excitation entries: {np.count_nonzero(alpha)} / {alpha.size}")
        log(f"  Max excitation: {alpha.max():.6f}, Mean baseline: {baseline.mean():.6f}")

        return {
            'alpha': alpha,
            'baseline': baseline,
            'method': 'tick_HawkesExpKern',
            'decay': decay,
        }

    except Exception as e:
        log(f"WARNING: tick Hawkes fitting failed: {e}")
        log("Falling back to discrete-time approach.")
        return None


# ---------------------------------------------------------------------------
# Approach 2: Discrete-time Logistic Regression Fallback
# ---------------------------------------------------------------------------
def discrete_time_hawkes(binary_matrix, timestamps, train_end_idx):
    """
    Discrete-time approximation of multivariate Hawkes process.

    For each target number i, fit a logistic regression:
        P(number_i appears at draw t) = logistic(w_0 + sum_j w_j * x_j(t))

    where x_j(t) = sum_{k in lookback} binary_j(t-k) * exp(-beta * k)

    This captures: does number j appearing recently predict number i appearing?
    L1 regularization promotes sparsity (most pairs should be zero).
    """
    log("=" * 70)
    log("DISCRETE-TIME HAWKES PROCESS (Logistic Regression Fallback)")
    log("=" * 70)

    n_train = train_end_idx
    train_mat = binary_matrix[:n_train]

    # Build exponential-decay features for each number
    # For draw t, feature_j = sum_{k=1}^{lookback} binary_j(t-k) * exp(-beta * k)
    log(f"Building exponential-decay features (lookback={LOOKBACK_DRAWS}, beta={DECAY_BETA_DEFAULT})...")

    # We'll use a wider lookback for richer features
    max_lookback = 10  # look back up to 10 draws
    decay_weights = np.exp(-DECAY_BETA_DEFAULT * np.arange(1, max_lookback + 1))  # shape (max_lookback,)

    # Precompute feature matrix: (n_train, 54) where each entry is the
    # weighted sum of recent appearances
    n_usable = n_train - max_lookback  # draws where we have full lookback
    X_all = np.zeros((n_usable, NUM_NUMBERS), dtype=np.float64)

    for lag in range(1, max_lookback + 1):
        w = decay_weights[lag - 1]
        X_all += w * train_mat[max_lookback - lag: n_train - lag, :]

    Y_all = train_mat[max_lookback:n_train, :]  # targets

    log(f"Feature matrix shape: {X_all.shape}, Target matrix shape: {Y_all.shape}")

    # Fit one logistic regression per target number
    alpha_matrix = np.zeros((NUM_NUMBERS, NUM_NUMBERS), dtype=np.float64)
    baseline = np.zeros(NUM_NUMBERS, dtype=np.float64)

    log("Fitting 54 logistic regression models (L1-penalized)...")
    t_start = time.time()

    for i in range(NUM_NUMBERS):
        if (i + 1) % 10 == 0 or i == 0:
            log(f"  Fitting number {i + 1}/54...")

        y = Y_all[:, i]

        # Check for degenerate cases
        if y.sum() < 5 or (1 - y).sum() < 5:
            log(f"  WARNING: Number {i+1} has too few positive/negative samples, skipping.")
            baseline[i] = np.log(y.mean() / (1 - y.mean() + 1e-10))
            continue

        model = LogisticRegression(
            penalty='l1',
            C=L1_C,
            solver='saga',
            max_iter=5000,
            tol=1e-5,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_all, y)

        alpha_matrix[i, :] = model.coef_[0]
        baseline[i] = model.intercept_[0]

    elapsed = time.time() - t_start
    log(f"Fitting complete in {elapsed:.1f}s")
    log(f"  Non-zero coefficients: {np.count_nonzero(alpha_matrix)} / {alpha_matrix.size}")
    log(f"  Sparsity: {1 - np.count_nonzero(alpha_matrix) / alpha_matrix.size:.1%}")

    return {
        'alpha': alpha_matrix,
        'baseline': baseline,
        'method': 'discrete_logistic_L1',
        'max_lookback': max_lookback,
        'decay_beta': DECAY_BETA_DEFAULT,
        'l1_C': L1_C,
    }


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------
def extract_top_pairs(alpha, top_k=TOP_K):
    """Extract top-K excitation and inhibition pairs from the alpha matrix."""
    # Flatten and get indices
    n = alpha.shape[0]

    # Excitation: largest positive values
    flat = alpha.flatten()
    top_excite_idx = np.argsort(flat)[::-1][:top_k]
    top_excite = []
    for idx in top_excite_idx:
        i, j = divmod(idx, n)
        val = float(alpha[i, j])
        if val > 0:
            top_excite.append({
                'source': int(j + 1),
                'target': int(i + 1),
                'strength': round(val, 6),
                'interpretation': f"Number {j+1} appearing boosts number {i+1}'s future rate"
            })

    # Inhibition: most negative values
    top_inhib_idx = np.argsort(flat)[:top_k]
    top_inhib = []
    for idx in top_inhib_idx:
        i, j = divmod(idx, n)
        val = float(alpha[i, j])
        if val < 0:
            top_inhib.append({
                'source': int(j + 1),
                'target': int(i + 1),
                'strength': round(val, 6),
                'interpretation': f"Number {j+1} appearing suppresses number {i+1}'s future rate"
            })

    return top_excite, top_inhib


def compute_self_excitation(alpha):
    """Extract diagonal of alpha matrix (self-excitation values)."""
    diag = np.diag(alpha)
    results = {}
    for i in range(len(diag)):
        results[int(i + 1)] = round(float(diag[i]), 6)
    return results


def compute_runs_test_zscores(binary_matrix):
    """
    Compute Wald-Wolfowitz runs test z-score for each number.
    Positive z = more runs than expected (over-dispersed / anti-clustering)
    Negative z = fewer runs than expected (clustering)
    """
    z_scores = {}
    for i in range(NUM_NUMBERS):
        seq = binary_matrix[:, i]
        n1 = seq.sum()
        n0 = len(seq) - n1

        if n1 == 0 or n0 == 0:
            z_scores[int(i + 1)] = 0.0
            continue

        # Count runs
        runs = 1
        for t in range(1, len(seq)):
            if seq[t] != seq[t - 1]:
                runs += 1

        # Expected runs and variance under null
        n = n0 + n1
        expected = 1 + (2 * n0 * n1) / n
        if n <= 1:
            z_scores[int(i + 1)] = 0.0
            continue
        var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
        if var <= 0:
            z_scores[int(i + 1)] = 0.0
            continue

        z = (runs - expected) / np.sqrt(var)
        z_scores[int(i + 1)] = round(float(z), 4)

    return z_scores


def detect_communities(alpha, threshold_quantile=0.95):
    """
    Build excitation graph from alpha matrix and detect communities.
    Only include edges above the threshold_quantile of positive weights.
    """
    import networkx as nx

    # Build directed graph with positive excitation edges
    G = nx.DiGraph()
    for i in range(NUM_NUMBERS):
        G.add_node(i + 1)

    positive_weights = alpha[alpha > 0]
    if len(positive_weights) == 0:
        log("  No positive excitation weights found for community detection.")
        return {}, G

    threshold = np.quantile(positive_weights, threshold_quantile)
    log(f"  Community detection edge threshold (q={threshold_quantile}): {threshold:.6f}")

    edge_count = 0
    for i in range(NUM_NUMBERS):
        for j in range(NUM_NUMBERS):
            if alpha[i, j] > threshold:
                G.add_edge(j + 1, i + 1, weight=float(alpha[i, j]))
                edge_count += 1

    log(f"  Excitation graph: {G.number_of_nodes()} nodes, {edge_count} edges")

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # Use greedy modularity community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G_undirected)
        community_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                community_map[int(node)] = int(idx)
        log(f"  Found {len(set(community_map.values()))} communities")
    except Exception as e:
        log(f"  Community detection failed: {e}")
        community_map = {i + 1: 0 for i in range(NUM_NUMBERS)}

    return community_map, G


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(binary_matrix, timestamps, alpha, baseline_vec, model_info,
                   train_end_idx):
    """
    Evaluate Hawkes model vs homogeneous Poisson baseline on held-out data.
    Returns evaluation metrics dict.
    """
    log("=" * 70)
    log("MODEL EVALUATION")
    log("=" * 70)

    n_total = len(binary_matrix)
    test_mat = binary_matrix[train_end_idx:]
    n_test = len(test_mat)
    method = model_info['method']

    log(f"Training draws: {train_end_idx}, Test draws: {n_test}")

    # --- Baseline: homogeneous rate (marginal probability from training) ---
    train_mat = binary_matrix[:train_end_idx]
    marginal_prob = train_mat.mean(axis=0)  # shape (54,)
    # Clip for numerical stability
    marginal_prob = np.clip(marginal_prob, 1e-6, 1 - 1e-6)

    # --- Hawkes model predictions on test set ---
    if method == 'discrete_logistic_L1':
        max_lookback = model_info['max_lookback']
        decay_beta = model_info['decay_beta']
        decay_weights = np.exp(-decay_beta * np.arange(1, max_lookback + 1))

        # For each test draw, compute features from preceding draws
        # (which may include some training draws for the first few test draws)
        hawkes_probs = np.zeros((n_test, NUM_NUMBERS), dtype=np.float64)

        for t_idx in range(n_test):
            abs_idx = train_end_idx + t_idx
            features = np.zeros(NUM_NUMBERS, dtype=np.float64)
            for lag in range(1, max_lookback + 1):
                src_idx = abs_idx - lag
                if src_idx >= 0:
                    features += decay_weights[lag - 1] * binary_matrix[src_idx, :]

            # Logistic: P(y_i=1) = sigmoid(baseline_i + alpha_i . features)
            logits = baseline_vec + alpha @ features  # this is wrong dimension-wise
            # alpha is (54, 54), features is (54,) -> result is (54,)
            # But we need alpha[i, :] . features for each i
            # Actually alpha @ features gives exactly that: row i dot features
            # Wait -- no, for discrete_logistic: each row i of alpha was fit as coef for target i
            # So alpha[i, :] are the weights for predicting number i from all feature_j
            # alpha @ features gives [alpha[0,:].features, alpha[1,:].features, ..., alpha[53,:].features]
            # which is correct
            logits_correct = baseline_vec + alpha.dot(features)
            hawkes_probs[t_idx, :] = expit(logits_correct)

    else:
        # tick-based: approximate with similar logic
        # For tick, alpha[i,j] = excitation from j to i, baseline is mu_i
        # We approximate intensity as baseline + sum of recent excitations
        decay = model_info.get('decay', 1.0)
        max_lookback = 10
        decay_weights_tick = np.exp(-decay * np.arange(1, max_lookback + 1))

        hawkes_probs = np.zeros((n_test, NUM_NUMBERS), dtype=np.float64)
        for t_idx in range(n_test):
            abs_idx = train_end_idx + t_idx
            excitation = np.zeros(NUM_NUMBERS, dtype=np.float64)
            for lag in range(1, max_lookback + 1):
                src_idx = abs_idx - lag
                if src_idx >= 0:
                    excitation += decay_weights_tick[lag - 1] * alpha.dot(binary_matrix[src_idx, :])

            intensity = baseline_vec + excitation
            # Convert intensity to probability (clip between 0 and 1)
            # For a Hawkes process, intensity is a rate, not probability
            # Approximate: P(at least one event in interval) approx 1 - exp(-lambda * dt)
            # Use average inter-draw interval
            avg_dt = float(np.mean(np.diff(timestamps[:train_end_idx])))
            prob = 1 - np.exp(-intensity * avg_dt)
            hawkes_probs[t_idx, :] = np.clip(prob, 1e-6, 1 - 1e-6)

    hawkes_probs = np.clip(hawkes_probs, 1e-6, 1 - 1e-6)

    # --- Log-likelihood comparison ---
    # Baseline log-likelihood
    baseline_ll = 0.0
    hawkes_ll = 0.0
    for t_idx in range(n_test):
        for j in range(NUM_NUMBERS):
            y = test_mat[t_idx, j]
            # Baseline
            p_base = marginal_prob[j]
            baseline_ll += y * np.log(p_base) + (1 - y) * np.log(1 - p_base)
            # Hawkes
            p_hawkes = hawkes_probs[t_idx, j]
            hawkes_ll += y * np.log(p_hawkes) + (1 - y) * np.log(1 - p_hawkes)

    # Per-observation log-likelihood
    n_obs = n_test * NUM_NUMBERS
    baseline_ll_avg = baseline_ll / n_obs
    hawkes_ll_avg = hawkes_ll / n_obs

    # Log-likelihood ratio
    ll_ratio = 2 * (hawkes_ll - baseline_ll)
    # Degrees of freedom: number of non-zero parameters in alpha
    n_params = int(np.count_nonzero(alpha)) + NUM_NUMBERS  # alpha entries + baselines
    # Approximate p-value from chi2 distribution
    if ll_ratio > 0:
        p_value = float(stats.chi2.sf(ll_ratio, df=max(n_params, 1)))
    else:
        p_value = 1.0

    log(f"  Baseline (Poisson) avg log-likelihood: {baseline_ll_avg:.6f}")
    log(f"  Hawkes model avg log-likelihood:       {hawkes_ll_avg:.6f}")
    log(f"  Log-likelihood ratio statistic:        {ll_ratio:.2f}")
    log(f"  Approx p-value (chi2, df={n_params}):  {p_value:.4e}")
    log(f"  Improvement: {((hawkes_ll_avg - baseline_ll_avg) / abs(baseline_ll_avg)) * 100:.4f}%")

    # --- Brier score comparison ---
    baseline_brier = np.mean((test_mat - marginal_prob[np.newaxis, :]) ** 2)
    hawkes_brier = np.mean((test_mat - hawkes_probs) ** 2)
    log(f"  Baseline Brier score: {baseline_brier:.6f}")
    log(f"  Hawkes Brier score:   {hawkes_brier:.6f}")

    # --- QQ plot of rescaled inter-event times (goodness-of-fit) ---
    log("  Computing rescaled inter-event times for QQ plot...")
    rescaled_times = []
    for j in range(NUM_NUMBERS):
        event_indices = np.where(binary_matrix[train_end_idx:, j] == 1)[0]
        if len(event_indices) < 2:
            continue
        # Rescaled inter-event times: integral of intensity between events
        # Approximate with cumulative probability
        cum_probs = np.cumsum(hawkes_probs[:, j])
        for k in range(1, len(event_indices)):
            t1, t2 = event_indices[k - 1], event_indices[k]
            if t2 > t1:
                rescaled_interval = cum_probs[t2] - cum_probs[t1]
                if rescaled_interval > 0:
                    rescaled_times.append(rescaled_interval)

    rescaled_times = np.array(rescaled_times)

    eval_results = {
        'baseline_avg_ll': round(float(baseline_ll_avg), 6),
        'hawkes_avg_ll': round(float(hawkes_ll_avg), 6),
        'll_ratio': round(float(ll_ratio), 2),
        'll_ratio_pvalue': float(p_value),
        'n_test_draws': n_test,
        'n_params': n_params,
        'baseline_brier': round(float(baseline_brier), 6),
        'hawkes_brier': round(float(hawkes_brier), 6),
        'improvement_pct': round(float(((hawkes_ll_avg - baseline_ll_avg) / abs(baseline_ll_avg)) * 100), 4),
    }

    return eval_results, hawkes_probs, rescaled_times


# ---------------------------------------------------------------------------
# Prediction Generation
# ---------------------------------------------------------------------------
def generate_predictions(binary_matrix, alpha, baseline_vec, model_info):
    """
    Generate delta_hawkes(number, draw) = log-odds adjustment for each number
    based on recent excitation history, for the most recent draws.
    """
    log("Generating Hawkes predictions (log-odds adjustments)...")

    method = model_info['method']
    n_total = len(binary_matrix)

    if method == 'discrete_logistic_L1':
        max_lookback = model_info['max_lookback']
        decay_beta = model_info['decay_beta']
        decay_weights = np.exp(-decay_beta * np.arange(1, max_lookback + 1))
    else:
        decay = model_info.get('decay', 1.0)
        max_lookback = 10
        decay_weights = np.exp(-decay * np.arange(1, max_lookback + 1))

    # Compute predictions for each of the last 50 draws plus "next draw"
    predictions = {}

    # "Next draw" prediction: using the most recent data
    features = np.zeros(NUM_NUMBERS, dtype=np.float64)
    for lag in range(1, max_lookback + 1):
        src_idx = n_total - lag
        if src_idx >= 0:
            features += decay_weights[lag - 1] * binary_matrix[src_idx, :]

    if method == 'discrete_logistic_L1':
        logits = baseline_vec + alpha.dot(features)
        probs = expit(logits)
        # Marginal baseline: average appearance rate
        marginal = binary_matrix.mean(axis=0)
        marginal_logits = np.log(marginal / (1 - marginal + 1e-10))
        delta_logodds = logits - marginal_logits
    else:
        intensity = baseline_vec + alpha.dot(features)
        probs = np.clip(intensity, 1e-6, 1.0)
        marginal = binary_matrix.mean(axis=0)
        delta_logodds = np.log(probs / (marginal + 1e-10) + 1e-10)

    next_draw_pred = {}
    for i in range(NUM_NUMBERS):
        next_draw_pred[str(i + 1)] = {
            'probability': round(float(probs[i]), 6),
            'delta_logodds': round(float(delta_logodds[i]), 6),
        }

    # Sort by probability descending
    sorted_nums = sorted(next_draw_pred.items(),
                         key=lambda x: x[1]['probability'], reverse=True)

    predictions['next_draw'] = {
        'all_numbers': next_draw_pred,
        'top_10': [{'number': int(k), **v} for k, v in sorted_nums[:10]],
        'bottom_10': [{'number': int(k), **v} for k, v in sorted_nums[-10:]],
    }

    # Historical predictions for last 50 draws (for backtesting)
    historical = []
    for draw_idx in range(max(0, n_total - 50), n_total):
        features_hist = np.zeros(NUM_NUMBERS, dtype=np.float64)
        for lag in range(1, max_lookback + 1):
            src_idx = draw_idx - lag
            if src_idx >= 0:
                features_hist += decay_weights[lag - 1] * binary_matrix[src_idx, :]

        if method == 'discrete_logistic_L1':
            logits_hist = baseline_vec + alpha.dot(features_hist)
            probs_hist = expit(logits_hist)
        else:
            probs_hist = np.clip(baseline_vec + alpha.dot(features_hist), 1e-6, 1.0)

        # Actual numbers drawn
        actual = list(np.where(binary_matrix[draw_idx] == 1)[0] + 1)

        draw_pred = {
            'draw_index': int(draw_idx),
            'actual_numbers': [int(x) for x in actual],
            'top_6_predicted': [int(x + 1) for x in np.argsort(probs_hist)[::-1][:6]],
            'hits': int(len(set(actual) & set(int(x + 1) for x in np.argsort(probs_hist)[::-1][:6]))),
        }
        historical.append(draw_pred)

    predictions['historical_backtest'] = historical

    # Compute average hits
    avg_hits = np.mean([h['hits'] for h in historical])
    # Expected by chance: 6 * 6/54 = 0.667
    expected_hits = PICKS_PER_DRAW * PICKS_PER_DRAW / NUM_NUMBERS
    log(f"  Backtest (last 50 draws): avg hits = {avg_hits:.3f} vs expected {expected_hits:.3f}")

    predictions['backtest_summary'] = {
        'avg_hits': round(float(avg_hits), 3),
        'expected_by_chance': round(float(expected_hits), 3),
        'n_draws': len(historical),
    }

    return predictions


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_excitation_heatmap(alpha, save_path):
    """Plot 54x54 excitation matrix as heatmap."""
    log("Generating excitation heatmap...")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Use diverging colormap (blue=inhibition, red=excitation)
    vmax = max(abs(alpha.max()), abs(alpha.min())) or 1.0
    im = ax.imshow(
        alpha,
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        aspect='equal',
        interpolation='nearest'
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Excitation / Inhibition Strength', fontsize=12)

    ax.set_xlabel('Source Number (j)', fontsize=12)
    ax.set_ylabel('Target Number (i)', fontsize=12)
    ax.set_title('Hawkes Process Cross-Excitation Matrix\n'
                 r'$\alpha_{ij}$: effect of number j on future rate of number i',
                 fontsize=14)

    # Tick labels every 5 numbers
    tick_positions = list(range(0, NUM_NUMBERS, 5))
    tick_labels = [str(p + 1) for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    # Add grid
    ax.set_xticks(np.arange(-0.5, NUM_NUMBERS, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, NUM_NUMBERS, 5), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)

    # Stats annotation
    n_nonzero = np.count_nonzero(alpha)
    n_positive = np.count_nonzero(alpha > 0)
    n_negative = np.count_nonzero(alpha < 0)
    stats_text = (f"Non-zero: {n_nonzero}/{alpha.size} ({n_nonzero/alpha.size:.1%})\n"
                  f"Excitatory: {n_positive}  |  Inhibitory: {n_negative}")
    ax.text(0.02, -0.08, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


def plot_network(alpha, community_map, save_path, top_edges=100):
    """Plot excitation network graph with community coloring."""
    import networkx as nx
    log("Generating excitation network plot...")

    # Build directed graph with strongest edges
    flat = alpha.flatten()
    # Get top positive edges
    sorted_idx = np.argsort(flat)[::-1]
    G = nx.DiGraph()
    for i in range(1, NUM_NUMBERS + 1):
        G.add_node(i)

    edge_count = 0
    for idx in sorted_idx:
        if edge_count >= top_edges:
            break
        i, j = divmod(idx, NUM_NUMBERS)
        val = alpha[i, j]
        if val > 0:
            G.add_edge(j + 1, i + 1, weight=float(val))
            edge_count += 1

    fig, ax = plt.subplots(figsize=(14, 14))

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Community colors
    n_communities = max(community_map.values()) + 1 if community_map else 1
    cmap = plt.cm.Set3
    node_colors = [cmap(community_map.get(n, 0) / max(n_communities, 1))
                   for n in G.nodes()]

    # Node sizes based on self-excitation
    diag = np.diag(alpha)
    node_sizes = []
    for n in G.nodes():
        se = abs(diag[n - 1])
        node_sizes.append(300 + 2000 * se / (abs(diag).max() + 1e-10))

    # Edge widths based on weight
    edges = G.edges(data=True)
    if edges:
        edge_weights = [d['weight'] for _, _, d in edges]
        max_w = max(edge_weights) if edge_weights else 1.0
        edge_widths = [0.5 + 3.0 * w / max_w for w in edge_weights]
        edge_alphas = [0.3 + 0.7 * w / max_w for w in edge_weights]
    else:
        edge_widths = []
        edge_alphas = []

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='black',
                           linewidths=0.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold')

    # Draw edges with varying alpha
    for (u, v, d), width, a in zip(edges, edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            width=width,
            alpha=a,
            edge_color='darkred',
            arrows=True,
            arrowsize=10,
            connectionstyle='arc3,rad=0.1'
        )

    ax.set_title(f'Hawkes Cross-Excitation Network\n'
                 f'Top {edge_count} excitation edges, {n_communities} communities',
                 fontsize=14)
    ax.axis('off')

    # Legend for communities
    if n_communities > 1:
        legend_elements = []
        for c in range(min(n_communities, 10)):
            members = [k for k, v in community_map.items() if v == c]
            label = f"Community {c} ({len(members)} numbers)"
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap(c / max(n_communities, 1)),
                           markersize=10, label=label)
            )
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


def plot_self_excitation_vs_runs(self_excitation, runs_zscores, save_path):
    """
    Scatter plot comparing self-excitation values to runs test z-scores.
    These should correlate: negative z (clustering) <-> positive self-excitation.
    """
    log("Generating self-excitation vs runs test comparison plot...")

    numbers = sorted(self_excitation.keys())
    se_vals = [self_excitation[n] for n in numbers]
    rz_vals = [runs_zscores.get(n, 0.0) for n in numbers]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel 1: Scatter plot ---
    ax = axes[0]
    colors = ['red' if rz < -1.96 else 'blue' if rz > 1.96 else 'gray' for rz in rz_vals]
    ax.scatter(se_vals, rz_vals, c=colors, s=50, alpha=0.7, edgecolors='black', linewidths=0.5)

    # Annotate notable points
    for i, n in enumerate(numbers):
        if abs(rz_vals[i]) > 1.96 or abs(se_vals[i]) > np.percentile(np.abs(se_vals), 90):
            ax.annotate(str(n), (se_vals[i], rz_vals[i]),
                        fontsize=7, ha='center', va='bottom')

    # Correlation
    if len(se_vals) > 2:
        r, p = stats.pearsonr(se_vals, rz_vals)
        ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\np = {p:.4f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96, color='red', linestyle=':', alpha=0.3, label='z = -1.96 (clustering)')
    ax.axhline(y=1.96, color='blue', linestyle=':', alpha=0.3, label='z = +1.96 (dispersed)')

    ax.set_xlabel('Hawkes Self-Excitation Coefficient', fontsize=12)
    ax.set_ylabel('Runs Test Z-Score', fontsize=12)
    ax.set_title('Self-Excitation vs Clustering Tendency', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')

    # --- Panel 2: Side-by-side bar chart for key numbers ---
    ax2 = axes[1]
    # Show the numbers mentioned in the prompt: 8,9,10 (clustering) and 2,4,5,14,16,32 (dispersed)
    highlight_nums = [2, 4, 5, 8, 9, 10, 14, 16, 32]
    se_highlight = [self_excitation.get(n, 0) for n in highlight_nums]
    rz_highlight = [runs_zscores.get(n, 0) for n in highlight_nums]

    x = np.arange(len(highlight_nums))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, se_highlight, width, label='Self-Excitation',
                    color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width / 2, rz_highlight, width, label='Runs Z-Score',
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in highlight_nums])
    ax2.set_xlabel('Lottery Number', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Key Numbers: Self-Excitation vs Runs Z-Score', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Color code the x-axis labels
    for i, n in enumerate(highlight_nums):
        if n in [8, 9, 10]:
            ax2.get_xticklabels()[i].set_color('red')
            ax2.get_xticklabels()[i].set_fontweight('bold')
        elif n in [2, 4, 5, 14, 16, 32]:
            ax2.get_xticklabels()[i].set_color('blue')
            ax2.get_xticklabels()[i].set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


def plot_qq(rescaled_times, save_path):
    """QQ plot of rescaled inter-event times against exponential distribution."""
    if len(rescaled_times) < 10:
        log("  Too few rescaled times for QQ plot, skipping.")
        return

    log("Generating QQ plot of rescaled inter-event times...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Sort rescaled times
    sorted_times = np.sort(rescaled_times)
    n = len(sorted_times)

    # Theoretical quantiles from Exp(1)
    theoretical = stats.expon.ppf((np.arange(1, n + 1) - 0.5) / n)

    ax.scatter(theoretical, sorted_times, s=2, alpha=0.3, color='steelblue')

    # Reference line
    max_val = max(theoretical.max(), sorted_times.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect fit')

    ax.set_xlabel('Theoretical Quantiles (Exp(1))', fontsize=12)
    ax.set_ylabel('Rescaled Inter-Event Times', fontsize=12)
    ax.set_title('QQ Plot: Goodness-of-Fit for Hawkes Model\n'
                 '(Points on line = good fit)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, np.percentile(theoretical, 99))
    ax.set_ylim(0, np.percentile(sorted_times, 99))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("MULTIVARIATE HAWKES PROCESS MODEL FOR LOTTO TEXAS")
    log("=" * 70)
    log(f"Configuration: {NUM_NUMBERS} numbers, holdout={HOLDOUT_DRAWS} draws, "
        f"lookback={LOOKBACK_DRAWS}, L1 C={L1_C}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------------------------------------
    df = load_data()
    binary_matrix = build_binary_matrix(df)
    timestamps = build_timestamps(df)

    n_total = len(df)
    train_end_idx = n_total - HOLDOUT_DRAWS
    log(f"Train/test split: {train_end_idx} train, {HOLDOUT_DRAWS} test")
    log(f"Binary matrix shape: {binary_matrix.shape}")
    log(f"Mean appearances per draw: {binary_matrix.sum(axis=1).mean():.1f}")
    log(f"Mean appearance rate per number: {binary_matrix.mean(axis=0).mean():.4f}")

    # -----------------------------------------------------------------------
    # 2. Fit Hawkes model (try tick first, fall back to discrete)
    # -----------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("FITTING HAWKES PROCESS MODEL")
    log("=" * 70)

    result = try_tick_hawkes(binary_matrix, timestamps, train_end_idx)

    if result is None:
        result = discrete_time_hawkes(binary_matrix, timestamps, train_end_idx)

    alpha = result['alpha']
    baseline_vec = result['baseline']
    method = result['method']

    log(f"\nFinal method used: {method}")

    # -----------------------------------------------------------------------
    # 3. Extract features from the excitation matrix
    # -----------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("EXCITATION MATRIX ANALYSIS")
    log("=" * 70)

    # 3a. Top excitation and inhibition pairs
    top_excite, top_inhib = extract_top_pairs(alpha)

    log(f"\nTop-{TOP_K} EXCITATION pairs (j -> i means j boosts i):")
    for rank, pair in enumerate(top_excite[:TOP_K], 1):
        log(f"  {rank:2d}. Number {pair['source']:2d} -> Number {pair['target']:2d}  "
            f"strength={pair['strength']:+.6f}")

    log(f"\nTop-{TOP_K} INHIBITION pairs (j -> i means j suppresses i):")
    for rank, pair in enumerate(top_inhib[:TOP_K], 1):
        log(f"  {rank:2d}. Number {pair['source']:2d} -> Number {pair['target']:2d}  "
            f"strength={pair['strength']:+.6f}")

    # 3b. Self-excitation diagonal
    self_excitation = compute_self_excitation(alpha)
    log("\nSelf-excitation values (diagonal of alpha):")
    sorted_se = sorted(self_excitation.items(), key=lambda x: x[1], reverse=True)
    log("  Strongest self-exciting (clustering):")
    for num, val in sorted_se[:10]:
        log(f"    Number {num:2d}: {val:+.6f}")
    log("  Strongest self-inhibiting (anti-clustering):")
    for num, val in sorted_se[-10:]:
        log(f"    Number {num:2d}: {val:+.6f}")

    # 3c. Runs test z-scores for comparison
    runs_zscores = compute_runs_test_zscores(binary_matrix)

    # 3d. Community detection
    log("\nCommunity detection on excitation graph...")
    community_map, G = detect_communities(alpha, threshold_quantile=0.95)

    # Print communities
    if community_map:
        communities_grouped = {}
        for num, comm in sorted(community_map.items()):
            if comm not in communities_grouped:
                communities_grouped[comm] = []
            communities_grouped[comm].append(num)
        for comm_id, members in sorted(communities_grouped.items()):
            log(f"  Community {comm_id}: {members}")

    # -----------------------------------------------------------------------
    # 4. Evaluation
    # -----------------------------------------------------------------------
    eval_results, hawkes_probs, rescaled_times = evaluate_model(
        binary_matrix, timestamps, alpha, baseline_vec, result, train_end_idx
    )

    # -----------------------------------------------------------------------
    # 5. Generate predictions
    # -----------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("GENERATING PREDICTIONS")
    log("=" * 70)
    predictions = generate_predictions(binary_matrix, alpha, baseline_vec, result)

    log("\nNext draw top-10 most likely numbers:")
    for entry in predictions['next_draw']['top_10']:
        log(f"  Number {entry['number']:2d}: P={entry['probability']:.4f}, "
            f"delta_logodds={entry['delta_logodds']:+.4f}")

    log("\nNext draw bottom-10 least likely numbers:")
    for entry in predictions['next_draw']['bottom_10']:
        log(f"  Number {entry['number']:2d}: P={entry['probability']:.4f}, "
            f"delta_logodds={entry['delta_logodds']:+.4f}")

    # -----------------------------------------------------------------------
    # 6. Save results
    # -----------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("SAVING RESULTS")
    log("=" * 70)

    # Excitation matrix
    excitation_output = {
        'method': method,
        'model_params': {k: v for k, v in result.items() if k not in ('alpha', 'baseline')},
        'alpha_matrix': alpha.tolist(),
        'baseline': baseline_vec.tolist(),
        'self_excitation': self_excitation,
        'communities': community_map,
        'non_zero_count': int(np.count_nonzero(alpha)),
        'sparsity': round(float(1 - np.count_nonzero(alpha) / alpha.size), 4),
        'matrix_stats': {
            'max': round(float(alpha.max()), 6),
            'min': round(float(alpha.min()), 6),
            'mean': round(float(alpha.mean()), 6),
            'std': round(float(alpha.std()), 6),
            'positive_count': int(np.count_nonzero(alpha > 0)),
            'negative_count': int(np.count_nonzero(alpha < 0)),
        }
    }
    excitation_path = os.path.join(RESULTS_DIR, 'hawkes_excitation_matrix.json')
    with open(excitation_path, 'w') as f:
        json.dump(excitation_output, f, indent=2)
    log(f"  Saved excitation matrix: {excitation_path}")

    # Top pairs
    top_pairs_output = {
        'method': method,
        'top_excitation_pairs': top_excite[:TOP_K],
        'top_inhibition_pairs': top_inhib[:TOP_K],
        'self_excitation': self_excitation,
        'runs_test_zscores': runs_zscores,
        'evaluation': eval_results,
    }
    top_pairs_path = os.path.join(RESULTS_DIR, 'hawkes_top_pairs.json')
    with open(top_pairs_path, 'w') as f:
        json.dump(top_pairs_output, f, indent=2)
    log(f"  Saved top pairs: {top_pairs_path}")

    # Predictions
    predictions_path = os.path.join(RESULTS_DIR, 'hawkes_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    log(f"  Saved predictions: {predictions_path}")

    # -----------------------------------------------------------------------
    # 7. Generate plots
    # -----------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("GENERATING PLOTS")
    log("=" * 70)

    plot_excitation_heatmap(
        alpha,
        os.path.join(RESULTS_DIR, 'hawkes_excitation_heatmap.png')
    )

    plot_network(
        alpha, community_map,
        os.path.join(RESULTS_DIR, 'hawkes_network.png')
    )

    plot_self_excitation_vs_runs(
        self_excitation, runs_zscores,
        os.path.join(RESULTS_DIR, 'hawkes_self_excitation_vs_runs.png')
    )

    plot_qq(
        rescaled_times,
        os.path.join(RESULTS_DIR, 'hawkes_qq_plot.png')
    )

    # -----------------------------------------------------------------------
    # 8. Summary
    # -----------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Method: {method}")
    log(f"Data: {n_total} draws, {train_end_idx} train / {HOLDOUT_DRAWS} test")
    log(f"Excitation matrix: {alpha.shape}, sparsity={1 - np.count_nonzero(alpha) / alpha.size:.1%}")
    log(f"  Positive (excitatory) entries: {np.count_nonzero(alpha > 0)}")
    log(f"  Negative (inhibitory) entries: {np.count_nonzero(alpha < 0)}")
    log(f"Evaluation:")
    log(f"  Baseline avg LL: {eval_results['baseline_avg_ll']:.6f}")
    log(f"  Hawkes avg LL:   {eval_results['hawkes_avg_ll']:.6f}")
    log(f"  Improvement:     {eval_results['improvement_pct']:.4f}%")
    log(f"  Brier (baseline): {eval_results['baseline_brier']:.6f}")
    log(f"  Brier (Hawkes):   {eval_results['hawkes_brier']:.6f}")
    log(f"Backtest (last 50): avg hits={predictions['backtest_summary']['avg_hits']:.3f} "
        f"vs chance={predictions['backtest_summary']['expected_by_chance']:.3f}")
    log("")
    log("Output files:")
    log(f"  {os.path.join(RESULTS_DIR, 'hawkes_excitation_matrix.json')}")
    log(f"  {os.path.join(RESULTS_DIR, 'hawkes_top_pairs.json')}")
    log(f"  {os.path.join(RESULTS_DIR, 'hawkes_predictions.json')}")
    log(f"  {os.path.join(RESULTS_DIR, 'hawkes_excitation_heatmap.png')}")
    log(f"  {os.path.join(RESULTS_DIR, 'hawkes_network.png')}")
    log(f"  {os.path.join(RESULTS_DIR, 'hawkes_self_excitation_vs_runs.png')}")
    log(f"  {os.path.join(RESULTS_DIR, 'hawkes_qq_plot.png')}")
    log("")
    log("Hawkes model pipeline complete.")


if __name__ == '__main__':
    main()
