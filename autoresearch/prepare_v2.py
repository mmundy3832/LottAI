"""
prepare_v2.py - Data loading and feature engineering for Pick 3 All Draw Times
(Morning/Day/Evening/Night).

This file is LOCKED during autoresearch - the AI agent never modifies it.
It provides all data and pre-computed features for experiment scripts.

Data: Pick 3 All Draw Times (15,572 draws across 4 daily draw times,
Sept 9 2013 - Feb 13 2026)
Digits: 0-9 in three positions
Combination space: 1,000 (000-999)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chisquare, entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_COMBOS = 1000
NUM_DIGITS = 10
NUM_POSITIONS = 3
PICK3_PAYOUT = 500  # $500 for a $1 straight bet

_DATA_FILES = [
    (os.path.join(os.path.dirname(__file__), "..", "pick3morning.csv"), 0),
    (os.path.join(os.path.dirname(__file__), "..", "pick3day.csv"),     1),
    (os.path.join(os.path.dirname(__file__), "..", "pick3evening.csv"), 2),
    (os.path.join(os.path.dirname(__file__), "..", "pick3night.csv"),   3),
]

# ---------------------------------------------------------------------------
# Box bet constants (precomputed once at module load)
# ---------------------------------------------------------------------------
# Group all 1000 straight combos by their unordered digit-set (box).
# key = tuple(sorted([d1, d2, d3])), value = list of combo indices
from collections import defaultdict as _defaultdict

_BOX_MAP = _defaultdict(list)
for _c in range(1000):
    _d1, _d2, _d3 = _c // 100, (_c // 10) % 10, _c % 10
    _BOX_MAP[tuple(sorted([_d1, _d2, _d3]))].append(_c)
_BOX_MAP = dict(_BOX_MAP)  # freeze

# Payout for each box key ($0 for triples -- can't box them in Texas Pick 3)
_BOX_PAYOUT = {}
for _key, _combos in _BOX_MAP.items():
    n = len(_combos)
    _BOX_PAYOUT[_key] = 0 if n == 1 else (160 if n == 3 else 80)

# 0-based rank of each box key in sorted(_BOX_MAP) -- for mean_rank baseline
_BOX_KEYS = list(_BOX_MAP.keys())       # 220 unique boxes
_COMBO_TO_BOX = {}
for _key, _combos in _BOX_MAP.items():
    for _c in _combos:
        _COMBO_TO_BOX[_c] = _key

# Split boundaries (fixed, time-ordered, no shuffling)
# 4 x 2725 = 10900  (same calendar date boundary as the evening-only system)
# 4 x 3309 = 13236  (same calendar date boundary for val end)
_TRAIN_END = 10900      # first ~70%: rows 0..10899
_VAL_END   = 13236      # next ~15%:  rows 10900..13235
#                         final ~15%: rows 13236..15571


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _load_raw():
    """Load all 4 raw CSVs and return a combined, sorted clean DataFrame."""
    dfs = []
    for filepath, draw_time_val in _DATA_FILES:
        df = pd.read_csv(
            filepath,
            header=None,
            names=["game", "month", "day", "year", "d1", "d2", "d3", "sum_col", "trailing"],
            dtype=str,
        )

        # Drop any fully-empty trailing rows
        df = df.dropna(subset=["year"]).reset_index(drop=True)

        # Build date column
        df["date"] = pd.to_datetime(
            df["year"].str.strip() + "-" +
            df["month"].str.strip() + "-" +
            df["day"].str.strip(),
            format="%Y-%m-%d",
        )

        # Digit columns as ints
        df["d1"] = df["d1"].astype(int)
        df["d2"] = df["d2"].astype(int)
        df["d3"] = df["d3"].astype(int)

        # Combo as 3-char string: "062"
        df["combo"] = df["d1"].apply(lambda x: str(x)) + \
                      df["d2"].apply(lambda x: str(x)) + \
                      df["d3"].apply(lambda x: str(x))

        # Combo as integer 0-999
        df["combo_int"] = df["d1"] * 100 + df["d2"] * 10 + df["d3"]

        # Temporal helpers
        df["day_of_week"] = df["date"].dt.dayofweek   # Monday=0 .. Sunday=6
        df["month"] = df["date"].dt.month              # 1-12
        df["year"] = df["date"].dt.year                # integer year

        # draw_time column (0=morning, 1=day, 2=evening, 3=night)
        df["draw_time"] = draw_time_val

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Sort by (date, draw_time) ascending
    combined = combined.sort_values(["date", "draw_time"]).reset_index(drop=True)

    # Keep only the columns we need, in order
    # draw_time is kept for future use but NOT included in FEATURE_SETS
    combined = combined[["date", "draw_time", "d1", "d2", "d3", "combo", "combo_int",
                         "day_of_week", "month", "year"]].copy()

    return combined


# Cache the loaded DataFrame at module level
_DF_CACHE = None

def load_data():
    """Return the full Pick 3 All Draw Times DataFrame (cached)."""
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = _load_raw()
    return _DF_CACHE.copy()


def get_train_val_test():
    """Returns (train_df, val_df, test_df) - time-ordered, no leakage."""
    df = load_data()
    train_df = df.iloc[:_TRAIN_END].reset_index(drop=True)
    val_df = df.iloc[_TRAIN_END:_VAL_END].reset_index(drop=True)
    test_df = df.iloc[_VAL_END:].reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Conversion Helpers
# ---------------------------------------------------------------------------

def combo_to_digits(c):
    """Convert combo int (e.g. 62) to (d1, d2, d3) tuple: (0, 6, 2)."""
    c = int(c)
    d1 = c // 100
    d2 = (c % 100) // 10
    d3 = c % 10
    return (d1, d2, d3)


def digits_to_combo(d1, d2, d3):
    """Convert digits (0, 6, 2) to combo int: 62."""
    return int(d1) * 100 + int(d2) * 10 + int(d3)


# ---------------------------------------------------------------------------
# Feature Set Functions
# ---------------------------------------------------------------------------
# Each takes the full DataFrame and a row index, returns a 1-D numpy array.
# Edge cases: when history is shorter than the window, use what's available.

def _feat_basic(df, idx):
    """Basic properties of the PREVIOUS draw (11 features).

    Uses idx-1 to avoid data leakage - we cannot know the current draw's
    digits when predicting it. For idx=0, returns zeros.
    """
    if idx == 0:
        return np.zeros(11, dtype=np.float64)

    row = df.iloc[idx - 1]
    d1, d2, d3 = row["d1"], row["d2"], row["d3"]
    digits = [d1, d2, d3]

    digit_sum = d1 + d2 + d3
    is_repeating = float(len(set(digits)) < 3)        # any digit appears twice or more
    is_triple = float(len(set(digits)) == 1)           # all three the same

    # Sequential: consecutive digits in any order (e.g. 123, 321, 345, 987)
    sorted_d = sorted(digits)
    is_sequential = float(
        sorted_d[1] - sorted_d[0] == 1 and sorted_d[2] - sorted_d[1] == 1
    )

    odd_count = sum(1 for d in digits if d % 2 == 1)
    even_count = 3 - odd_count
    high_count = sum(1 for d in digits if d >= 5)
    low_count = 3 - high_count

    return np.array([
        d1, d2, d3, digit_sum,
        is_repeating, is_triple, is_sequential,
        odd_count, even_count, high_count, low_count
    ], dtype=np.float64)


def _feat_recency(df, idx):
    """
    Recency features:
    - Per digit 0-9, per position (3 positions): frequency in last N draws
      for N in [10, 25, 50, 100]. That's 10*3*4 = 120 features.
    - Gap since this combo last appeared (capped at 1000). 1 feature.
    Total: 121 features.
    """
    windows = [10, 25, 50, 100]
    features = []

    for window in windows:
        start = max(0, idx - window)
        history = df.iloc[start:idx]
        n = len(history)
        if n == 0:
            features.extend([0.0] * (NUM_DIGITS * NUM_POSITIONS))
            continue
        for pos_col in ["d1", "d2", "d3"]:
            counts = history[pos_col].value_counts()
            for digit in range(NUM_DIGITS):
                freq = counts.get(digit, 0) / n
                features.append(freq)

    # Gap since this combo last appeared
    combo_int = df.iloc[idx]["combo_int"]
    gap = 1000  # default if never seen before
    for lookback in range(1, min(idx + 1, 1001)):
        if df.iloc[idx - lookback]["combo_int"] == combo_int:
            gap = lookback
            break
    features.append(float(gap))

    return np.array(features, dtype=np.float64)


def _feat_gaps(df, idx):
    """
    Gap features:
    - For each digit 0-9: draws since it last appeared in ANY position (10 features)
    - For each position x digit (3*10=30): draws since that digit in that position (30 features)
    Total: 40 features. Gaps capped at idx (or 1000 if never seen).
    """
    features = []

    # Per-digit gap (any position)
    for digit in range(NUM_DIGITS):
        gap = min(idx, 1000)  # default
        for lookback in range(1, min(idx + 1, 1001)):
            row = df.iloc[idx - lookback]
            if digit in (row["d1"], row["d2"], row["d3"]):
                gap = lookback
                break
        features.append(float(gap))

    # Per-position per-digit gap
    for pos_col in ["d1", "d2", "d3"]:
        for digit in range(NUM_DIGITS):
            gap = min(idx, 1000)  # default
            for lookback in range(1, min(idx + 1, 1001)):
                if df.iloc[idx - lookback][pos_col] == digit:
                    gap = lookback
                    break
            features.append(float(gap))

    return np.array(features, dtype=np.float64)


def _feat_positional(df, idx):
    """
    Positional features over last 50 draws:
    - Per position, per digit: frequency (3*10 = 30 features)
    - Digit correlation between position pairs over last 50 draws (3 pairs)
    Total: 33 features.
    """
    window = 50
    start = max(0, idx - window)
    history = df.iloc[start:idx]
    n = len(history)

    features = []

    # Per-position digit frequency
    for pos_col in ["d1", "d2", "d3"]:
        counts = history[pos_col].value_counts() if n > 0 else pd.Series(dtype=int)
        for digit in range(NUM_DIGITS):
            freq = counts.get(digit, 0) / n if n > 0 else 0.0
            features.append(freq)

    # Correlation between position pairs
    pos_cols = ["d1", "d2", "d3"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        if n >= 2:
            corr = history[pos_cols[i]].corr(history[pos_cols[j]])
            if pd.isna(corr):
                corr = 0.0
        else:
            corr = 0.0
        features.append(corr)

    return np.array(features, dtype=np.float64)


def _feat_temporal(df, idx):
    """
    Temporal features:
    - day_of_week one-hot (7 features)
    - month one-hot (12 features)
    - year_normalized 0-1 (1 feature)
    Total: 20 features.
    """
    row = df.iloc[idx]
    features = []

    # Day of week one-hot
    dow = row["day_of_week"]
    dow_oh = [0.0] * 7
    dow_oh[dow] = 1.0
    features.extend(dow_oh)

    # Month one-hot
    m = row["month"]
    m_oh = [0.0] * 12
    m_oh[m - 1] = 1.0
    features.extend(m_oh)

    # Year normalized -- use local load_data() (v2, all draw times)
    full_df = load_data()
    min_year = full_df["year"].min()
    max_year = full_df["year"].max()
    year_range = max_year - min_year
    if year_range == 0:
        year_range = 1
    year_norm = (row["year"] - min_year) / year_range
    features.append(year_norm)

    return np.array(features, dtype=np.float64)


def _feat_momentum(df, idx):
    """
    Momentum features:
    - Rolling chi-squared per position (last 50 draws vs uniform) - 3 features
    - Rolling entropy per position (last 50 draws) - 3 features
    - Consecutive-draw digit overlap count - 1 feature
    Total: 7 features.
    """
    window = 50
    start = max(0, idx - window)
    history = df.iloc[start:idx]
    n = len(history)

    features = []

    # Chi-squared and entropy per position
    for pos_col in ["d1", "d2", "d3"]:
        if n >= 5:
            observed = np.zeros(NUM_DIGITS)
            for digit in range(NUM_DIGITS):
                observed[digit] = (history[pos_col] == digit).sum()
            expected = np.full(NUM_DIGITS, n / NUM_DIGITS)
            chi2, _ = chisquare(observed, expected)
            features.append(chi2)

            # Entropy
            probs = observed / n
            probs = probs[probs > 0]  # avoid log(0)
            ent = scipy_entropy(probs, base=2)
            features.append(ent)
        else:
            features.append(0.0)  # chi2
            features.append(0.0)  # entropy

    # Consecutive-draw digit overlap
    if idx > 0:
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]
        prev_digits = {prev["d1"], prev["d2"], prev["d3"]}
        curr_digits = {curr["d1"], curr["d2"], curr["d3"]}
        overlap = len(prev_digits & curr_digits)
    else:
        overlap = 0
    features.append(float(overlap))

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature Set Registry
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "basic": _feat_basic,
    "recency": _feat_recency,
    "gaps": _feat_gaps,
    "positional": _feat_positional,
    "temporal": _feat_temporal,
    "momentum": _feat_momentum,
}

# Feature dimensionality for each set (for documentation/validation)
FEATURE_DIMS = {
    "basic": 11,
    "recency": 121,
    "gaps": 40,
    "positional": 33,
    "temporal": 20,
    "momentum": 7,
}


def build_features(df, indices, feature_sets=None):
    """
    Build feature matrix for given indices using requested feature sets.

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame (from load_data()).
    indices : array-like of int
        Row indices into df to build features for.
    feature_sets : list of str or None
        Which feature sets to include. If None, use all.

    Returns
    -------
    np.ndarray of shape (len(indices), total_feature_dim)
    """
    if feature_sets is None:
        feature_sets = list(FEATURE_SETS.keys())

    # Validate requested sets
    for fs in feature_sets:
        if fs not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {fs}. "
                             f"Available: {list(FEATURE_SETS.keys())}")

    rows = []
    for idx in indices:
        parts = []
        for fs in feature_sets:
            feat = FEATURE_SETS[fs](df, idx)
            parts.append(feat)
        rows.append(np.concatenate(parts))

    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(prob_matrix, actual_combos):
    """
    Evaluate prediction quality against actual outcomes.

    Parameters
    ----------
    prob_matrix : np.ndarray, shape (N, 1000)
        Predicted probability distribution over all 1000 combos per draw.
    actual_combos : np.ndarray, shape (N,)
        Actual combo_int values (0-999) for each draw.

    Returns
    -------
    dict with evaluation metrics.
    """
    N = len(actual_combos)
    assert prob_matrix.shape == (N, NUM_COMBOS), \
        f"prob_matrix shape {prob_matrix.shape} != ({N}, {NUM_COMBOS})"
    actual_combos = np.asarray(actual_combos, dtype=int)

    # --- Brier Score ---
    # Brier = mean over draws of sum_c (p_c - I(c==actual))^2
    brier_scores = []
    for i in range(N):
        target = np.zeros(NUM_COMBOS)
        target[actual_combos[i]] = 1.0
        brier = np.mean((prob_matrix[i] - target) ** 2)
        brier_scores.append(brier)
    mean_brier = float(np.mean(brier_scores))

    # --- Rank of true combo ---
    # Rank 1 = highest probability combo
    ranks = []
    for i in range(N):
        probs = prob_matrix[i]
        # argsort descending; rank = position of actual combo + 1
        sorted_indices = np.argsort(-probs)
        rank = int(np.where(sorted_indices == actual_combos[i])[0][0]) + 1
        ranks.append(rank)
    mean_rank = float(np.mean(ranks))

    # --- Log-likelihood ---
    log_liks = []
    for i in range(N):
        p = prob_matrix[i, actual_combos[i]]
        p = max(p, 1e-15)  # avoid log(0)
        log_liks.append(np.log(p))
    mean_log_lik = float(np.mean(log_liks))

    # --- Top-K hit rates ---
    # Track k=1..20 for optimal_ev, plus coarser buckets for reporting
    _OPT_KS = list(range(1, 21))
    top_k_hits = {k: 0 for k in _OPT_KS}
    top_k_hits.update({50: 0, 100: 0})
    exact_hits = 0
    for i in range(N):
        sorted_indices = np.argsort(-prob_matrix[i])
        top_1 = sorted_indices[0]
        if top_1 == actual_combos[i]:
            exact_hits += 1
        actual = actual_combos[i]
        # find rank of actual (0-based); mark all k above it
        rank_of_actual = int(np.where(sorted_indices == actual)[0][0])
        for k in top_k_hits:
            if rank_of_actual < k:
                top_k_hits[k] += 1

    top_5_hit   = top_k_hits[5]   / N
    top_6_hit   = top_k_hits[6]   / N
    top_10_hit  = top_k_hits[10]  / N
    top_50_hit  = top_k_hits[50]  / N
    top_100_hit = top_k_hits[100] / N
    exact_hit   = exact_hits / N

    # --- Optimal EV across K=1..20 ---
    # Buy top-K tickets: spend $K, win $500 if hit. EV = hit_rate*500 - K
    opt_ev_by_k = {k: (top_k_hits[k] / N) * PICK3_PAYOUT - k for k in _OPT_KS}
    optimal_k  = max(opt_ev_by_k, key=opt_ev_by_k.get)
    optimal_ev = opt_ev_by_k[optimal_k]

    # --- Box evaluation ---
    # Rank the 220 unique unordered digit-sets by summed combo probability.
    # For each draw, find where the actual box lands in that ranking.
    # EV(buy top-K boxes) = avg_payout_if_hit / N - K  (payout: $80 6-way, $160 3-way, $0 triple)
    _box_opt_ks = list(range(1, 21))
    box_rank_sum      = 0
    box_winnings_by_k = {k: 0.0 for k in _box_opt_ks}

    for i in range(N):
        probs = prob_matrix[i]
        # Sum straight-combo probabilities for each box key
        box_probs = {key: float(probs[combos].sum())
                     for key, combos in _BOX_MAP.items()}
        # Rank boxes descending
        sorted_box_keys = sorted(box_probs, key=box_probs.__getitem__, reverse=True)
        actual_box  = _COMBO_TO_BOX[int(actual_combos[i])]
        box_rank_0  = sorted_box_keys.index(actual_box)   # 0-based
        box_rank_sum += box_rank_0 + 1                    # 1-based for mean_rank
        payout = _BOX_PAYOUT[actual_box]
        for k in _box_opt_ks:
            if box_rank_0 < k:
                box_winnings_by_k[k] += payout

    box_mean_rank = box_rank_sum / N              # baseline = 110.5 (220 boxes)
    box_top_k_hit  = {k: (box_winnings_by_k[k] / N / max(_BOX_PAYOUT.values()))
                       for k in _box_opt_ks}      # rough hit-rate proxy (vs $80 max)
    box_opt_ev_by_k = {k: box_winnings_by_k[k] / N - k for k in _box_opt_ks}
    box_optimal_k   = max(box_opt_ev_by_k, key=box_opt_ev_by_k.get)
    box_optimal_ev  = box_opt_ev_by_k[box_optimal_k]

    # --- Expected Value ---
    # If we bet $1 on our top-1 prediction each draw:
    # EV = (hit_rate * payout) - 1
    ev_per_dollar = (exact_hit * PICK3_PAYOUT) - 1.0

    # --- Baseline comparison (uniform 1/1000) ---
    uniform_brier_target = np.zeros(NUM_COMBOS)
    # For uniform: Brier = mean((1/1000 - I(c==actual))^2)
    # = (999 * (1/1000)^2 + 1 * (1 - 1/1000)^2) / 1000
    uniform_p = 1.0 / NUM_COMBOS
    uniform_brier = (999 * uniform_p**2 + (1 - uniform_p)**2) / NUM_COMBOS
    uniform_rank = 500.5
    uniform_log_lik = np.log(uniform_p)
    uniform_top10 = 10 / NUM_COMBOS
    uniform_top5  = 5  / NUM_COMBOS
    uniform_top6  = 6  / NUM_COMBOS
    uniform_top50 = 50 / NUM_COMBOS
    uniform_top100 = 100 / NUM_COMBOS
    uniform_exact = 1 / NUM_COMBOS
    uniform_ev = (uniform_exact * PICK3_PAYOUT) - 1.0

    # Break-even EV for specific top-K straight bets (spend $K, win $500 if hit)
    top5_ev  = (top_5_hit  * PICK3_PAYOUT) - 5.0
    top6_ev  = (top_6_hit  * PICK3_PAYOUT) - 6.0

    vs_baseline = {
        "brier_improvement": uniform_brier - mean_brier,
        "rank_improvement": uniform_rank - mean_rank,
        "log_lik_improvement": mean_log_lik - uniform_log_lik,
        "top_5_lift":  top_5_hit  / uniform_top5  if uniform_top5  > 0 else 0,
        "top_6_lift":  top_6_hit  / uniform_top6  if uniform_top6  > 0 else 0,
        "top_10_lift": top_10_hit / uniform_top10 if uniform_top10 > 0 else 0,
        "top_50_lift": top_50_hit / uniform_top50 if uniform_top50 > 0 else 0,
        "top_100_lift": top_100_hit / uniform_top100 if uniform_top100 > 0 else 0,
        "ev_improvement": ev_per_dollar - uniform_ev,
    }

    return {
        "brier_score": mean_brier,
        "mean_rank": mean_rank,
        "log_likelihood": mean_log_lik,
        "top_5_hit": top_5_hit,
        "top_6_hit": top_6_hit,
        "top_10_hit": top_10_hit,
        "top_50_hit": top_50_hit,
        "top_100_hit": top_100_hit,
        "exact_hit": exact_hit,
        "expected_value": ev_per_dollar,
        "top5_ev": top5_ev,
        "top6_ev": top6_ev,
        "optimal_k": optimal_k,
        "optimal_ev": optimal_ev,
        "opt_ev_by_k": opt_ev_by_k,
        "box_mean_rank": box_mean_rank,
        "box_optimal_k": box_optimal_k,
        "box_optimal_ev": box_optimal_ev,
        "box_opt_ev_by_k": box_opt_ev_by_k,
        "vs_baseline": vs_baseline,
    }


# ---------------------------------------------------------------------------
# Feature Pre-computation & Caching
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".feature_cache_v2")

def precompute_all_features():
    """Pre-compute all feature sets for train, val, and test indices, save to .npy files.

    Only recomputes if cache is missing. Delete .feature_cache_v2/ to force rebuild.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)

    df = load_data()
    train_indices = list(range(0, _TRAIN_END))
    val_indices   = list(range(_TRAIN_END, _VAL_END))
    test_indices  = list(range(_VAL_END, len(df)))

    for split_name, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        for fs_name in FEATURE_SETS:
            cache_file = os.path.join(_CACHE_DIR, f"{split_name}_{fs_name}.npy")
            if os.path.exists(cache_file):
                continue
            print(f"  Pre-computing {split_name}/{fs_name} ({len(indices)} rows x {FEATURE_DIMS[fs_name]} features)...")
            X = build_features(df, indices, [fs_name])
            np.save(cache_file, X)
            print(f"    Saved {cache_file}")

    print("  Feature cache ready.")


def load_cached_features(split, feature_sets=None):
    """Load pre-computed features from cache.

    Parameters
    ----------
    split : str
        'train', 'val', or 'test'
    feature_sets : list of str or None
        Which feature sets to load. None = all.

    Returns
    -------
    np.ndarray of shape (N, total_dims)
    """
    if feature_sets is None:
        feature_sets = list(FEATURE_SETS.keys())

    parts = []
    for fs_name in feature_sets:
        cache_file = os.path.join(_CACHE_DIR, f"{split}_{fs_name}.npy")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file not found: {cache_file}. "
                f"Run precompute_all_features() first."
            )
        parts.append(np.load(cache_file))

    return np.hstack(parts)


# ---------------------------------------------------------------------------
# Import-time summary
# ---------------------------------------------------------------------------

def _print_summary():
    """Print data summary on import."""
    df = load_data()
    train, val, test = get_train_val_test()

    total_features = sum(FEATURE_DIMS.values())

    print("=" * 65)
    print("  Pick 3 All Draw Times (Morning/Day/Evening/Night) - Autoresearch Data Module")
    print("=" * 65)
    print(f"  Total draws:       {len(df):,}  (4 draw times combined)")
    print(f"  Date range:        {df['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Combination space: {NUM_COMBOS}")
    print(f"  -" * 32)
    print(f"  Train set:  {len(train):>5,} draws  "
          f"({train['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{train['date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  Val set:    {len(val):>5,} draws  "
          f"({val['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{val['date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  Test set:   {len(test):>5,} draws  "
          f"({test['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{test['date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  -" * 32)
    print(f"  Feature sets ({len(FEATURE_SETS)}):")
    for name, dim in FEATURE_DIMS.items():
        print(f"    {name:15s} {dim:>4d} features")
    print(f"    {'TOTAL':15s} {total_features:>4d} features")
    print("=" * 65)


_print_summary()
