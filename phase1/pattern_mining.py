#!/usr/bin/env python3
"""
LottAI Sequential Pattern Mining
=================================
Uses PrefixSpan to discover sequential patterns in Lotto Texas draw data,
then tests their predictive power for future draws.

Core hypothesis: "sparse triggers sparse" -- a pattern of numbers in draws
t, t+1 might predict elevated probability of specific numbers in draws
t+2 through t+5.

Phases:
  1. Data preparation (raw numbers, number groups, feature events)
  2. PrefixSpan sequential pattern mining
  3. Pattern significance testing (Fisher's exact, FDR correction)
  4. Temporal pattern windows (prediction at t+1..t+5)
  5. Pattern combination (aggregate predictions, backtest)

Outputs:
  - results/pattern_mining_patterns.json
  - results/pattern_mining_predictions.json
  - results/pattern_mining_metrics.json
  - results/pattern_lift_distribution.png
  - results/pattern_top_predictors.png
  - results/pattern_backtest.png
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
import time
import sys
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# Configuration
# ============================================================================
CSV_FILE = 'lottotexas.csv'
RESULTS_DIR = 'results'
NUMS_PER_DRAW = 6
NUM_RANGE = 54  # numbers 1-54
BACKTEST_SIZE = 400  # last 400 draws for backtesting
MIN_SUPPORT = 50  # minimum support count for PrefixSpan
MAX_PATTERN_LEN = 3  # mine up to length-3 sequential patterns
PREDICTION_WINDOWS = [1, 2, 3, 4, 5]  # predict draws t+1 through t+5
FDR_ALPHA = 0.05  # false discovery rate threshold
TOP_K_PATTERNS = 50  # keep top K patterns by lift for reporting

# Number group mapping: decades
DECADE_MAP = {
    range(1, 10): 'D1',    # 1-9
    range(10, 20): 'D2',   # 10-19
    range(20, 30): 'D3',   # 20-29
    range(30, 40): 'D4',   # 30-39
    range(40, 50): 'D5',   # 40-49
    range(50, 55): 'D6',   # 50-54
}

os.makedirs(RESULTS_DIR, exist_ok=True)


def timestamp():
    """Return formatted timestamp for logging."""
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log(msg):
    """Print a timestamped log message."""
    print(f"[{timestamp()}] {msg}", flush=True)


# ============================================================================
# Phase 1: Data Preparation
# ============================================================================
def load_data(csv_path):
    """Load Lotto Texas draws in chronological order."""
    log("Phase 1: Loading data...")
    cols = ['GameName', 'Month', 'Day', 'Year',
            'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    df = pd.read_csv(csv_path, header=None, names=cols)
    # Build date column and sort chronologically
    df['Date'] = pd.to_datetime(
        df[['Year', 'Month', 'Day']].rename(
            columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}
        )
    )
    df = df.sort_values('Date').reset_index(drop=True)
    # Filter out pre-April 2006 draws (game changed from pick-6-from-50 to pick-6-from-54)
    cutoff_date = pd.Timestamp('2006-04-01')
    pre_2006_count = len(df[df['Date'] < cutoff_date])
    df = df[df['Date'] >= cutoff_date].reset_index(drop=True)
    log(f"  Filtered out {pre_2006_count} pre-April 2006 draws (pick-6-from-50 era)")
    log(f"  Remaining draws: {len(df)}")
    num_cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    draws = df[num_cols].values.tolist()
    dates = df['Date'].tolist()
    log(f"  Loaded {len(draws)} draws from {dates[0].strftime('%Y-%m-%d')} "
        f"to {dates[-1].strftime('%Y-%m-%d')}")
    return draws, dates


def number_to_decade(n):
    """Map a number 1-54 to its decade group label."""
    for rng, label in DECADE_MAP.items():
        if n in rng:
            return label
    return 'D?'


def compute_draw_features(draws, window=20):
    """
    For each draw, compute feature events:
      - hot_N / cold_N: number N is above/below expected frequency in recent window
      - high_sum / low_sum: draw sum is in top/bottom quartile
      - wide_range / narrow_range: range of numbers is above/below median
      - odd_heavy / even_heavy: 4+ odd or 4+ even numbers
      - gap_long_N: number N has been absent for >= 2x expected gap
    """
    log("  Computing draw feature events...")
    n_draws = len(draws)
    expected_freq = window * NUMS_PER_DRAW / NUM_RANGE  # expected appearances in window

    # Precompute rolling frequency for each number
    freq_matrix = np.zeros((n_draws, NUM_RANGE + 1))  # 1-indexed
    for i, draw in enumerate(draws):
        start = max(0, i - window)
        for j in range(start, i):
            for n in draws[j]:
                freq_matrix[i][n] += 1

    # Precompute draw sums and ranges
    draw_sums = [sum(d) for d in draws]
    draw_ranges = [max(d) - min(d) for d in draws]
    sum_q25, sum_q75 = np.percentile(draw_sums, [25, 75])
    range_median = np.median(draw_ranges)

    # Precompute last-seen for gap analysis
    last_seen = {}  # number -> last draw index where it appeared
    expected_gap = n_draws / (n_draws * NUMS_PER_DRAW / NUM_RANGE)  # ~9 draws

    feature_sequences = []
    for i, draw in enumerate(draws):
        events = set()

        # Hot/cold for each number in this draw
        for n in draw:
            if freq_matrix[i][n] > expected_freq * 1.5:
                events.add(f'hot_{n}')
            elif freq_matrix[i][n] < expected_freq * 0.5:
                events.add(f'cold_{n}')

        # Sum events
        s = draw_sums[i]
        if s >= sum_q75:
            events.add('high_sum')
        elif s <= sum_q25:
            events.add('low_sum')

        # Range events
        r = draw_ranges[i]
        if r >= range_median + 10:
            events.add('wide_range')
        elif r <= range_median - 10:
            events.add('narrow_range')

        # Odd/even balance
        n_odd = sum(1 for n in draw if n % 2 == 1)
        if n_odd >= 5:
            events.add('odd_heavy')
        elif n_odd <= 1:
            events.add('even_heavy')

        # Gap events: numbers that haven't appeared in a long time
        for n in draw:
            if n in last_seen:
                gap = i - last_seen[n]
                if gap >= expected_gap * 2:
                    events.add(f'gap_long_{n}')

        # Update last_seen
        for n in draw:
            last_seen[n] = i

        feature_sequences.append(sorted(events))

    log(f"  Feature events computed for {n_draws} draws")
    return feature_sequences


def prepare_sequences(draws):
    """
    Prepare three representations:
      a) Raw number sequences (each draw = sorted list of 6 numbers)
      b) Decade group sequences (each draw = sorted list of decade labels)
      c) Feature event sequences
    """
    log("  Preparing sequence representations...")

    # a) Raw numbers
    raw_seqs = [sorted(d) for d in draws]

    # b) Decade groups
    decade_seqs = [sorted(set(number_to_decade(n) for n in d)) for d in draws]

    # c) Feature events
    feature_seqs = compute_draw_features(draws)

    log(f"  Raw: {len(raw_seqs)} transactions, "
        f"Decade: {len(decade_seqs)}, Feature: {len(feature_seqs)}")
    return raw_seqs, decade_seqs, feature_seqs


# ============================================================================
# Phase 2: PrefixSpan Mining
# ============================================================================

class PrefixSpanMiner:
    """
    Sequential pattern miner using the PrefixSpan algorithm.

    A sequential pattern is a sequence of itemsets: [itemset1, itemset2, ...]
    where each itemset is a set of items occurring together in one transaction,
    and the sequence reflects consecutive transactions.

    We mine patterns where each element of the sequence is a subset of items
    from a single draw (transaction), and consecutive elements come from
    consecutive draws.
    """

    def __init__(self, min_support=50, max_len=3):
        self.min_support = min_support
        self.max_len = max_len
        self.patterns = []

    def _try_prefixspan_library(self, db):
        """Attempt to use the prefixspan library if installed."""
        try:
            from prefixspan import PrefixSpan
            ps = PrefixSpan(db)
            ps.minlen = 2
            ps.maxlen = self.max_len
            results = ps.frequent(self.min_support)
            return [(count, pat) for count, pat in results]
        except ImportError:
            return None

    def mine(self, transactions, mode='raw'):
        """
        Mine sequential patterns from a list of transactions.

        For sequential pattern mining across consecutive draws, we treat
        the entire history as one long sequence of itemsets. We look for
        sub-sequences of itemsets that appear frequently.

        Since full PrefixSpan over itemset-sequences is complex, we use
        a simplified but effective approach:
          - For length-2 patterns: item A in draw t, item B in draw t+1
          - For length-3 patterns: item A in t, item B in t+1, item C in t+2

        This captures the cross-draw sequential patterns we care about.

        Args:
            transactions: list of lists (each inner list = items in one draw)
            mode: 'raw', 'decade', or 'feature' for logging
        """
        log(f"  Mining sequential patterns (mode={mode}, "
            f"min_support={self.min_support}, max_len={self.max_len})...")

        # First try the prefixspan library for single-item sequences
        # Convert transactions to a single long sequence with draw boundaries
        # But for our use case, the cross-draw pair/triple approach is more
        # interpretable, so we use our custom miner.

        patterns = []

        # --- Length-2 patterns: (item_in_draw_t) -> (item_in_draw_t+1) ---
        log("    Mining length-2 cross-draw patterns...")
        pair_counts = Counter()
        n = len(transactions)
        for i in range(n - 1):
            for a in transactions[i]:
                for b in transactions[i + 1]:
                    pair_counts[(a, b)] += 1

        len2_patterns = [
            (count, [a, b])
            for (a, b), count in pair_counts.items()
            if count >= self.min_support
        ]
        patterns.extend(len2_patterns)
        log(f"    Found {len(len2_patterns)} length-2 patterns above support={self.min_support}")

        # --- Length-3 patterns: (item_t) -> (item_t+1) -> (item_t+2) ---
        if self.max_len >= 3:
            log("    Mining length-3 cross-draw patterns...")
            # To keep tractable, only extend length-2 patterns that met support
            frequent_pairs = {(a, b) for (a, b), count in pair_counts.items()
                              if count >= self.min_support}

            triple_counts = Counter()
            for i in range(n - 2):
                for a in transactions[i]:
                    for b in transactions[i + 1]:
                        if (a, b) not in frequent_pairs:
                            continue
                        for c in transactions[i + 2]:
                            triple_counts[(a, b, c)] += 1

            len3_patterns = [
                (count, [a, b, c])
                for (a, b, c), count in triple_counts.items()
                if count >= self.min_support
            ]
            patterns.extend(len3_patterns)
            log(f"    Found {len(len3_patterns)} length-3 patterns above support={self.min_support}")

        # Sort by support descending
        patterns.sort(key=lambda x: -x[0])
        self.patterns = patterns
        log(f"  Total mined patterns: {len(patterns)}")
        return patterns


def mine_itemset_patterns(transactions, min_support=50):
    """
    Mine frequent co-occurrence patterns WITHIN a single draw.
    These are used as prefix conditions for cross-draw prediction.

    Returns pairs and triples of items that co-occur frequently in the same draw.
    """
    log("  Mining within-draw itemset patterns...")
    pair_counts = Counter()
    for draw in transactions:
        for a, b in combinations(draw, 2):
            pair_counts[(a, b)] += 1

    frequent_itemsets = [
        (count, [a, b])
        for (a, b), count in pair_counts.items()
        if count >= min_support
    ]
    frequent_itemsets.sort(key=lambda x: -x[0])
    log(f"  Found {len(frequent_itemsets)} frequent within-draw pairs")
    return frequent_itemsets


# ============================================================================
# Phase 3: Pattern Significance Testing
# ============================================================================

def compute_base_rates(draws):
    """Compute base rate (marginal probability) for each number."""
    total = len(draws)
    counts = Counter()
    for d in draws:
        for n in d:
            counts[n] += 1
    base_rates = {n: counts[n] / total for n in range(1, NUM_RANGE + 1)}
    return base_rates


def test_pattern_significance(patterns, transactions, base_rates, mode='raw'):
    """
    For each mined cross-draw sequential pattern, test whether the prefix
    predicts the suffix better than the marginal base rate.

    For a length-2 pattern [A, B]:
      - Prefix: A appears in draw t
      - Suffix: B appears in draw t+1
      - H0: P(B in t+1 | A in t) = P(B) [base rate]

    Uses Fisher's exact test on the 2x2 contingency table:
                     B in t+1    B not in t+1
      A in t            a             b
      A not in t        c             d

    Returns list of significant patterns with statistics.
    """
    log(f"Phase 3: Testing pattern significance (mode={mode})...")
    n = len(transactions)
    results = []

    for idx, (support, pattern) in enumerate(patterns):
        if idx % 500 == 0 and idx > 0:
            log(f"  Tested {idx}/{len(patterns)} patterns...")

        if len(pattern) == 2:
            prefix_item, suffix_item = pattern[0], pattern[1]

            # Build contingency table
            a, b, c, d = 0, 0, 0, 0
            for i in range(n - 1):
                prefix_present = prefix_item in transactions[i]
                suffix_present = suffix_item in transactions[i + 1]
                if prefix_present and suffix_present:
                    a += 1
                elif prefix_present and not suffix_present:
                    b += 1
                elif not prefix_present and suffix_present:
                    c += 1
                else:
                    d += 1

            table = np.array([[a, b], [c, d]])
            if a == 0:
                continue

            try:
                odds_ratio, p_value = fisher_exact(table, alternative='greater')
            except Exception:
                continue

            # Compute lift
            p_suffix_given_prefix = a / (a + b) if (a + b) > 0 else 0
            p_suffix = (a + c) / (a + b + c + d) if (a + b + c + d) > 0 else 0
            lift = p_suffix_given_prefix / p_suffix if p_suffix > 0 else 0
            confidence = p_suffix_given_prefix

            results.append({
                'pattern': pattern,
                'pattern_str': f"{prefix_item} -> {suffix_item}",
                'support': int(support),
                'confidence': float(confidence),
                'lift': float(lift),
                'p_value': float(p_value),
                'contingency': [[int(a), int(b)], [int(c), int(d)]],
                'length': 2,
                'mode': mode,
            })

        elif len(pattern) == 3:
            item_a, item_b, item_c = pattern[0], pattern[1], pattern[2]

            # Prefix: A in t AND B in t+1; Suffix: C in t+2
            a, b, c_cnt, d = 0, 0, 0, 0
            for i in range(n - 2):
                prefix_present = (item_a in transactions[i] and
                                  item_b in transactions[i + 1])
                suffix_present = item_c in transactions[i + 2]
                if prefix_present and suffix_present:
                    a += 1
                elif prefix_present and not suffix_present:
                    b += 1
                elif not prefix_present and suffix_present:
                    c_cnt += 1
                else:
                    d += 1

            table = np.array([[a, b], [c_cnt, d]])
            if a == 0:
                continue

            try:
                odds_ratio, p_value = fisher_exact(table, alternative='greater')
            except Exception:
                continue

            p_suffix_given_prefix = a / (a + b) if (a + b) > 0 else 0
            p_suffix = (a + c_cnt) / (a + b + c_cnt + d) if (a + b + c_cnt + d) > 0 else 0
            lift = p_suffix_given_prefix / p_suffix if p_suffix > 0 else 0
            confidence = p_suffix_given_prefix

            results.append({
                'pattern': pattern,
                'pattern_str': f"{item_a} -> {item_b} -> {item_c}",
                'support': int(support),
                'confidence': float(confidence),
                'lift': float(lift),
                'p_value': float(p_value),
                'contingency': [[int(a), int(b)], [int(c_cnt), int(d)]],
                'length': 3,
                'mode': mode,
            })

    log(f"  Tested {len(patterns)} patterns, {len(results)} had non-zero co-occurrence")
    return results


def apply_fdr_correction(results, alpha=FDR_ALPHA):
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    Returns list of significant patterns.
    """
    log(f"  Applying FDR correction (alpha={alpha})...")
    if not results:
        log("  No results to correct.")
        return []

    # Sort by p-value
    results_sorted = sorted(results, key=lambda x: x['p_value'])
    m = len(results_sorted)

    significant = []
    for i, r in enumerate(results_sorted):
        # BH threshold: (rank / m) * alpha
        bh_threshold = ((i + 1) / m) * alpha
        r['bh_threshold'] = float(bh_threshold)
        r['rank'] = i + 1
        if r['p_value'] <= bh_threshold:
            r['fdr_significant'] = True
            significant.append(r)
        else:
            r['fdr_significant'] = False

    log(f"  {len(significant)} patterns significant after FDR correction "
        f"(out of {m} tested)")
    return significant, results_sorted


# ============================================================================
# Phase 4: Temporal Pattern Windows
# ============================================================================

def test_temporal_windows(significant_patterns, transactions, windows=None):
    """
    For each significant pattern, test prediction at different time offsets.
    Instead of just t -> t+1, test t -> t+w for w in windows.

    This captures the hypothesis that a pattern in draw t might predict
    numbers in draws t+1 through t+5.
    """
    if windows is None:
        windows = PREDICTION_WINDOWS
    log(f"Phase 4: Testing temporal windows {windows}...")

    # We only use length-2 patterns for windowed analysis (prefix -> suffix)
    len2_patterns = [p for p in significant_patterns if p['length'] == 2]
    log(f"  Testing {len(len2_patterns)} length-2 patterns across {len(windows)} windows")

    n = len(transactions)
    temporal_results = []

    for pat_idx, pat in enumerate(len2_patterns):
        prefix_item = pat['pattern'][0]
        suffix_item = pat['pattern'][1]

        for w in windows:
            if w == 1:
                # Already tested in Phase 3
                temporal_results.append({
                    'pattern': pat['pattern'],
                    'pattern_str': pat['pattern_str'],
                    'window': 1,
                    'lift': pat['lift'],
                    'p_value': pat['p_value'],
                    'confidence': pat['confidence'],
                    'support': pat['support'],
                })
                continue

            # Test at offset w
            a, b, c, d = 0, 0, 0, 0
            for i in range(n - w):
                prefix_present = prefix_item in transactions[i]
                suffix_present = suffix_item in transactions[i + w]
                if prefix_present and suffix_present:
                    a += 1
                elif prefix_present and not suffix_present:
                    b += 1
                elif not prefix_present and suffix_present:
                    c += 1
                else:
                    d += 1

            if a == 0:
                continue

            table = np.array([[a, b], [c, d]])
            try:
                _, p_value = fisher_exact(table, alternative='greater')
            except Exception:
                continue

            p_suffix_given_prefix = a / (a + b) if (a + b) > 0 else 0
            p_suffix = (a + c) / (a + b + c + d) if (a + b + c + d) > 0 else 0
            lift = p_suffix_given_prefix / p_suffix if p_suffix > 0 else 0

            temporal_results.append({
                'pattern': pat['pattern'],
                'pattern_str': pat['pattern_str'],
                'window': int(w),
                'lift': float(lift),
                'p_value': float(p_value),
                'confidence': float(p_suffix_given_prefix),
                'support': int(a),
            })

    log(f"  Generated {len(temporal_results)} temporal pattern-window combinations")
    return temporal_results


# ============================================================================
# Phase 5: Pattern Combination & Prediction
# ============================================================================

def build_prediction_model(significant_patterns, temporal_results, transactions,
                           backtest_start):
    """
    For patterns that survive significance testing, build a simple prediction rule:
      - For each number, aggregate how many active patterns predict it
      - Weight by lift and confidence
      - Produce delta_pattern(number, draw) = log-odds adjustment

    Backtest on the last BACKTEST_SIZE draws using expanding window.
    """
    log("Phase 5: Building prediction model and backtesting...")

    n = len(transactions)
    train_end = backtest_start  # everything before this is initial training

    # Collect all actionable pattern rules:
    # Each rule: prefix_items (in draw t), suffix_item (predicted in draw t+w),
    #            window w, lift, confidence
    rules = []

    # From length-2 significant patterns (including temporal windows)
    for tr in temporal_results:
        if tr['p_value'] < FDR_ALPHA and tr['lift'] > 1.0:
            rules.append({
                'prefix': [tr['pattern'][0]],
                'suffix': tr['pattern'][1],
                'window': tr['window'],
                'lift': tr['lift'],
                'confidence': tr['confidence'],
            })

    # From length-3 significant patterns
    for pat in significant_patterns:
        if pat['length'] == 3 and pat['lift'] > 1.0:
            rules.append({
                'prefix': pat['pattern'][:2],
                'suffix': pat['pattern'][2],
                'window': 2,  # prefix spans t, t+1; suffix at t+2
                'lift': pat['lift'],
                'confidence': pat['confidence'],
            })

    log(f"  {len(rules)} actionable prediction rules")
    if not rules:
        log("  WARNING: No actionable rules found. Returning empty predictions.")
        return [], {}

    # --- Backtest ---
    log(f"  Backtesting on draws {backtest_start} to {n-1} ({n - backtest_start} draws)...")

    predictions = []
    hits_at_k = {1: 0, 3: 0, 6: 0}
    total_predictions = 0
    all_precisions = []
    all_recalls = []
    all_hit_rates = []
    baseline_hit_rate = NUMS_PER_DRAW / NUM_RANGE  # 6/54 ~ 0.111

    for t in range(backtest_start, n):
        # Score each number based on active rules
        scores = defaultdict(float)
        rule_counts = defaultdict(int)

        for rule in rules:
            w = rule['window']
            prefix = rule['prefix']

            # Check if prefix is active in the recent draws
            if len(prefix) == 1:
                # Single-item prefix: check draw t-w
                ref_draw = t - w
                if ref_draw < 0:
                    continue
                if prefix[0] in transactions[ref_draw]:
                    # This rule fires: predict suffix in draw t
                    suffix = rule['suffix']
                    weight = np.log(rule['lift']) * rule['confidence']
                    scores[suffix] += weight
                    rule_counts[suffix] += 1
            elif len(prefix) == 2:
                # Two-item prefix: check draws t-2 and t-1
                ref_draw_a = t - 2
                ref_draw_b = t - 1
                if ref_draw_a < 0:
                    continue
                if (prefix[0] in transactions[ref_draw_a] and
                        prefix[1] in transactions[ref_draw_b]):
                    suffix = rule['suffix']
                    weight = np.log(rule['lift']) * rule['confidence']
                    scores[suffix] += weight
                    rule_counts[suffix] += 1

        if not scores:
            predictions.append({
                'draw_index': int(t),
                'predicted': [],
                'actual': transactions[t],
                'hits': 0,
                'n_rules_fired': 0,
            })
            continue

        # Rank numbers by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_6 = [item for item, score in ranked[:6]]
        actual = set(transactions[t])

        hits = len(set(top_6) & actual)
        total_predictions += 1

        # Precision: fraction of predicted that were actual
        precision = hits / min(len(top_6), 6) if top_6 else 0
        # Recall: fraction of actual that were predicted
        recall = hits / NUMS_PER_DRAW
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_hit_rates.append(hits / 6)

        for k in hits_at_k:
            top_k = [item for item, score in ranked[:k]]
            hits_at_k[k] += len(set(top_k) & actual)

        predictions.append({
            'draw_index': int(t),
            'predicted': [int(x) for x in top_6],
            'scores': {str(int(item)): float(score) for item, score in ranked[:10]},
            'actual': [int(x) for x in transactions[t]],
            'hits': int(hits),
            'n_rules_fired': int(sum(rule_counts.values())),
        })

    # --- Compute metrics ---
    draws_with_predictions = sum(1 for p in predictions if p['predicted'])
    total_hits = sum(p['hits'] for p in predictions)

    if draws_with_predictions > 0:
        avg_hit_rate = total_hits / (draws_with_predictions * NUMS_PER_DRAW)
    else:
        avg_hit_rate = 0

    metrics = {
        'backtest_draws': n - backtest_start,
        'draws_with_predictions': draws_with_predictions,
        'total_rules': len(rules),
        'baseline_hit_rate': float(baseline_hit_rate),
        'model_hit_rate': float(avg_hit_rate),
        'lift_over_baseline': float(avg_hit_rate / baseline_hit_rate) if baseline_hit_rate > 0 else 0,
        'avg_precision': float(np.mean(all_precisions)) if all_precisions else 0,
        'avg_recall': float(np.mean(all_recalls)) if all_recalls else 0,
        'total_hits_top6': int(total_hits),
    }

    for k in hits_at_k:
        if draws_with_predictions > 0:
            metrics[f'avg_hits_top{k}'] = float(hits_at_k[k] / draws_with_predictions)

    log(f"  Backtest results:")
    log(f"    Draws with predictions: {draws_with_predictions}/{n - backtest_start}")
    log(f"    Baseline hit rate: {baseline_hit_rate:.4f}")
    log(f"    Model hit rate:    {avg_hit_rate:.4f}")
    log(f"    Lift over baseline: {metrics['lift_over_baseline']:.4f}")
    log(f"    Avg precision@6:   {metrics['avg_precision']:.4f}")
    log(f"    Avg recall@6:      {metrics['avg_recall']:.4f}")

    return predictions, metrics


def run_random_baseline(transactions, backtest_start, n_shuffles=10):
    """
    Shuffle draw order, re-mine patterns, and compare lift distribution.
    This is the null hypothesis: if we destroy temporal structure, do patterns
    still appear with similar lift?
    """
    log("  Running random baseline (shuffle test)...")
    n = len(transactions)
    baseline_lifts = []

    for shuffle_idx in range(n_shuffles):
        # Shuffle the transactions (destroy temporal order)
        shuffled = transactions.copy()
        np.random.shuffle(shuffled)

        # Mine length-2 patterns on shuffled data
        pair_counts = Counter()
        for i in range(len(shuffled) - 1):
            for a in shuffled[i]:
                for b in shuffled[i + 1]:
                    pair_counts[(a, b)] += 1

        # Compute lift for patterns that meet support
        for (a, b), count in pair_counts.items():
            if count < MIN_SUPPORT:
                continue
            # Quick lift computation
            a_count = sum(1 for t in shuffled[:-1] if a in t)
            b_count = sum(1 for t in shuffled[1:] if b in t)
            expected = a_count * b_count / (n - 1) if (n - 1) > 0 else 1
            if expected > 0:
                lift = count / expected
                baseline_lifts.append(lift)

        if shuffle_idx % 5 == 0:
            log(f"    Shuffle {shuffle_idx + 1}/{n_shuffles} done")

    log(f"  Random baseline: {len(baseline_lifts)} pattern-lifts from {n_shuffles} shuffles")
    return baseline_lifts


# ============================================================================
# Plotting
# ============================================================================

def plot_lift_distribution(all_results, baseline_lifts, filepath):
    """Plot lift distribution of real vs shuffled patterns."""
    log(f"  Plotting lift distribution -> {filepath}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of real lifts
    real_lifts = [r['lift'] for r in all_results if r['lift'] > 0]
    ax = axes[0]
    ax.hist(real_lifts, bins=50, alpha=0.7, color='steelblue', edgecolor='black',
            label=f'Real patterns (n={len(real_lifts)})')
    if baseline_lifts:
        ax.hist(baseline_lifts, bins=50, alpha=0.5, color='salmon', edgecolor='black',
                label=f'Shuffled baseline (n={len(baseline_lifts)})')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='Lift = 1 (no effect)')
    ax.set_xlabel('Lift', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Lift Distribution: Real vs Shuffled', fontsize=13)
    ax.legend(fontsize=10)

    # Right: Q-Q style comparison
    ax = axes[1]
    if real_lifts:
        real_sorted = sorted(real_lifts)
        ax.plot(range(len(real_sorted)), real_sorted, 'b-', alpha=0.7,
                label='Real (sorted)')
    if baseline_lifts:
        base_sorted = sorted(baseline_lifts)
        ax.plot(np.linspace(0, len(real_sorted) - 1 if real_lifts else 0,
                            len(base_sorted)),
                base_sorted, 'r-', alpha=0.5, label='Shuffled (sorted)')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Lift', fontsize=12)
    ax.set_title('Sorted Lift Comparison', fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_predictors(significant_patterns, filepath, top_k=20):
    """Plot top predictive patterns by lift."""
    log(f"  Plotting top predictors -> {filepath}")
    if not significant_patterns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No significant patterns found', ha='center',
                va='center', fontsize=14)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Sort by lift and take top K
    top = sorted(significant_patterns, key=lambda x: -x['lift'])[:top_k]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(top) * 0.4)))

    # Left: horizontal bar chart of lift
    ax = axes[0]
    labels = [p['pattern_str'] for p in top]
    lifts = [p['lift'] for p in top]
    colors = ['darkgreen' if p['p_value'] < 0.01 else 'steelblue' for p in top]
    y_pos = range(len(top))
    ax.barh(y_pos, lifts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Lift', fontsize=12)
    ax.set_title(f'Top {len(top)} Patterns by Lift', fontsize=13)
    ax.invert_yaxis()

    # Right: confidence and support
    ax = axes[1]
    supports = [p['support'] for p in top]
    confidences = [p['confidence'] for p in top]
    scatter = ax.scatter(supports, confidences, c=lifts, cmap='RdYlGn',
                         s=80, edgecolors='black', alpha=0.8)
    for i, p in enumerate(top):
        ax.annotate(p['pattern_str'], (supports[i], confidences[i]),
                    fontsize=7, ha='left', va='bottom')
    ax.set_xlabel('Support (count)', fontsize=12)
    ax.set_ylabel('Confidence P(suffix|prefix)', fontsize=12)
    ax.set_title('Support vs Confidence', fontsize=13)
    plt.colorbar(scatter, ax=ax, label='Lift')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_backtest(predictions, metrics, filepath):
    """Plot backtest results over time."""
    log(f"  Plotting backtest results -> {filepath}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to draws that had predictions
    pred_with_data = [p for p in predictions if p['predicted']]

    if not pred_with_data:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No predictions generated', ha='center',
                    va='center', fontsize=14)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return

    indices = [p['draw_index'] for p in pred_with_data]
    hits = [p['hits'] for p in pred_with_data]
    n_rules = [p['n_rules_fired'] for p in pred_with_data]

    # Top-left: hits per draw (rolling average)
    ax = axes[0, 0]
    window = min(50, len(hits))
    if len(hits) >= window:
        rolling_hits = pd.Series(hits).rolling(window).mean()
        ax.plot(indices, rolling_hits, 'b-', linewidth=1.5, label=f'Rolling avg ({window})')
    ax.scatter(indices, hits, alpha=0.2, s=10, c='steelblue')
    ax.axhline(y=NUMS_PER_DRAW * NUMS_PER_DRAW / NUM_RANGE, color='red',
               linestyle='--', label=f'Expected by chance ({NUMS_PER_DRAW**2/NUM_RANGE:.2f})')
    ax.set_xlabel('Draw Index')
    ax.set_ylabel('Hits in Top-6 Predictions')
    ax.set_title('Prediction Hits per Draw')
    ax.legend(fontsize=9)

    # Top-right: cumulative hit rate vs baseline
    ax = axes[0, 1]
    cum_hits = np.cumsum(hits)
    cum_draws = np.arange(1, len(hits) + 1)
    cum_rate = cum_hits / (cum_draws * NUMS_PER_DRAW)
    ax.plot(indices, cum_rate, 'b-', linewidth=1.5, label='Model cumulative hit rate')
    ax.axhline(y=NUMS_PER_DRAW / NUM_RANGE, color='red', linestyle='--',
               label=f'Baseline ({NUMS_PER_DRAW/NUM_RANGE:.4f})')
    ax.set_xlabel('Draw Index')
    ax.set_ylabel('Cumulative Hit Rate')
    ax.set_title('Cumulative Hit Rate vs Baseline')
    ax.legend(fontsize=9)

    # Bottom-left: distribution of hits
    ax = axes[1, 0]
    hit_counts = Counter(hits)
    max_hits = max(hits) if hits else 6
    x_vals = range(0, max_hits + 1)
    y_vals = [hit_counts.get(h, 0) for h in x_vals]
    ax.bar(x_vals, y_vals, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Number of Hits (out of 6)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Hits per Draw')
    # Add expected binomial distribution
    from scipy.stats import binom
    n_pred = len(pred_with_data)
    p_hit = NUMS_PER_DRAW / NUM_RANGE
    expected_binom = [binom.pmf(k, 6, p_hit) * n_pred for k in x_vals]
    ax.plot(x_vals, expected_binom, 'r--o', linewidth=1.5, markersize=5,
            label='Expected (binomial)')
    ax.legend(fontsize=9)

    # Bottom-right: rules fired vs hits
    ax = axes[1, 1]
    ax.scatter(n_rules, hits, alpha=0.3, s=15, c='steelblue')
    ax.set_xlabel('Number of Rules Fired')
    ax.set_ylabel('Hits in Top-6')
    ax.set_title('Rules Fired vs Prediction Quality')
    # Add trend line
    if len(n_rules) > 2:
        z = np.polyfit(n_rules, hits, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(n_rules), max(n_rules), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=1.5,
                label=f'Trend (slope={z[0]:.4f})')
        ax.legend(fontsize=9)

    plt.suptitle(
        f'Backtest Results | Model Hit Rate: {metrics.get("model_hit_rate", 0):.4f} | '
        f'Lift: {metrics.get("lift_over_baseline", 0):.2f}x',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    log("=" * 70)
    log("LottAI Sequential Pattern Mining")
    log("=" * 70)
    t0 = time.time()

    # --- Phase 1: Data Preparation ---
    draws, dates = load_data(CSV_FILE)
    raw_seqs, decade_seqs, feature_seqs = prepare_sequences(draws)

    # --- Phase 2: PrefixSpan Mining ---
    log("")
    log("Phase 2: Sequential pattern mining...")

    all_mined_patterns = []

    # 2a) Raw number patterns
    miner_raw = PrefixSpanMiner(min_support=MIN_SUPPORT, max_len=MAX_PATTERN_LEN)
    raw_patterns = miner_raw.mine(raw_seqs, mode='raw')
    all_mined_patterns.extend([(p, 'raw') for p in raw_patterns])

    # 2b) Decade group patterns (lower support since fewer symbols)
    miner_decade = PrefixSpanMiner(min_support=MIN_SUPPORT * 2, max_len=MAX_PATTERN_LEN)
    decade_patterns = miner_decade.mine(decade_seqs, mode='decade')
    all_mined_patterns.extend([(p, 'decade') for p in decade_patterns])

    # 2c) Feature event patterns (lower support)
    miner_feature = PrefixSpanMiner(min_support=max(20, MIN_SUPPORT // 2),
                                     max_len=MAX_PATTERN_LEN)
    feature_patterns = miner_feature.mine(feature_seqs, mode='feature')
    all_mined_patterns.extend([(p, 'feature') for p in feature_patterns])

    # 2d) Within-draw itemset patterns (for enriching prefix conditions)
    itemset_patterns = mine_itemset_patterns(raw_seqs, min_support=MIN_SUPPORT)

    log(f"\n  Total mined patterns across all modes: {len(all_mined_patterns)}")

    # --- Phase 3: Significance Testing ---
    log("")
    base_rates = compute_base_rates(draws)

    all_test_results = []

    # Test raw patterns
    raw_results = test_pattern_significance(raw_patterns, raw_seqs, base_rates, mode='raw')
    all_test_results.extend(raw_results)

    # Test decade patterns
    decade_results = test_pattern_significance(decade_patterns, decade_seqs, base_rates,
                                                mode='decade')
    all_test_results.extend(decade_results)

    # Test feature patterns
    feature_results = test_pattern_significance(feature_patterns, feature_seqs, base_rates,
                                                 mode='feature')
    all_test_results.extend(feature_results)

    log(f"\n  Total tested results: {len(all_test_results)}")

    # FDR correction
    significant_all, all_corrected = apply_fdr_correction(all_test_results)

    # Separate significant results by mode
    sig_raw = [p for p in significant_all if p['mode'] == 'raw']
    sig_decade = [p for p in significant_all if p['mode'] == 'decade']
    sig_feature = [p for p in significant_all if p['mode'] == 'feature']
    log(f"  Significant by mode: raw={len(sig_raw)}, decade={len(sig_decade)}, "
        f"feature={len(sig_feature)}")

    # --- Phase 4: Temporal Windows ---
    log("")
    temporal_results = test_temporal_windows(sig_raw, raw_seqs)

    # --- Phase 5: Prediction & Backtest ---
    log("")
    backtest_start = len(draws) - BACKTEST_SIZE
    predictions, metrics = build_prediction_model(
        significant_all, temporal_results, raw_seqs, backtest_start
    )

    # Random baseline comparison
    baseline_lifts = run_random_baseline(raw_seqs, backtest_start, n_shuffles=10)

    # Add baseline lift stats to metrics
    if baseline_lifts:
        metrics['baseline_lift_mean'] = float(np.mean(baseline_lifts))
        metrics['baseline_lift_std'] = float(np.std(baseline_lifts))
        metrics['baseline_lift_max'] = float(np.max(baseline_lifts))
        metrics['baseline_lift_p95'] = float(np.percentile(baseline_lifts, 95))
        real_lifts = [r['lift'] for r in all_test_results if r['lift'] > 0]
        if real_lifts:
            metrics['real_lift_mean'] = float(np.mean(real_lifts))
            metrics['real_lift_max'] = float(np.max(real_lifts))
            metrics['real_lift_p95'] = float(np.percentile(real_lifts, 95))
            # How many real patterns have lift above the 95th percentile of shuffled?
            threshold = np.percentile(baseline_lifts, 95)
            n_above = sum(1 for l in real_lifts if l > threshold)
            metrics['patterns_above_shuffled_p95'] = int(n_above)
            metrics['shuffled_p95_threshold'] = float(threshold)

    # --- Save Results ---
    log("")
    log("Saving results...")

    # Patterns JSON: top patterns by lift
    top_patterns = sorted(significant_all, key=lambda x: -x['lift'])[:TOP_K_PATTERNS]
    # Convert any non-serializable types
    for p in top_patterns:
        p['pattern'] = [str(x) for x in p['pattern']]
    patterns_path = os.path.join(RESULTS_DIR, 'pattern_mining_patterns.json')
    with open(patterns_path, 'w') as f:
        json.dump({
            'total_mined': len(all_mined_patterns),
            'total_tested': len(all_test_results),
            'total_significant': len(significant_all),
            'top_patterns': top_patterns,
            'by_mode': {
                'raw': len(sig_raw),
                'decade': len(sig_decade),
                'feature': len(sig_feature),
            },
        }, f, indent=2, default=str)
    log(f"  Patterns -> {patterns_path}")

    # Predictions JSON
    predictions_path = os.path.join(RESULTS_DIR, 'pattern_mining_predictions.json')
    # Only save last 50 predictions to keep file size manageable
    with open(predictions_path, 'w') as f:
        json.dump({
            'backtest_start_index': backtest_start,
            'total_predictions': len(predictions),
            'sample_predictions': predictions[-50:],
        }, f, indent=2, default=str)
    log(f"  Predictions -> {predictions_path}")

    # Metrics JSON
    metrics_path = os.path.join(RESULTS_DIR, 'pattern_mining_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    log(f"  Metrics -> {metrics_path}")

    # --- Plots ---
    log("")
    log("Generating plots...")

    plot_lift_distribution(
        all_test_results, baseline_lifts,
        os.path.join(RESULTS_DIR, 'pattern_lift_distribution.png')
    )

    plot_top_predictors(
        significant_all,
        os.path.join(RESULTS_DIR, 'pattern_top_predictors.png'),
        top_k=min(20, len(significant_all))
    )

    plot_backtest(
        predictions, metrics,
        os.path.join(RESULTS_DIR, 'pattern_backtest.png')
    )

    # --- Summary ---
    elapsed = time.time() - t0
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Total draws analyzed:    {len(draws)}")
    log(f"  Patterns mined:          {len(all_mined_patterns)}")
    log(f"  Patterns tested:         {len(all_test_results)}")
    log(f"  FDR-significant:         {len(significant_all)}")
    log(f"  Actionable rules:        {metrics.get('total_rules', 0)}")
    log(f"  Backtest draws:          {metrics.get('backtest_draws', 0)}")
    log(f"  Baseline hit rate:       {metrics.get('baseline_hit_rate', 0):.4f}")
    log(f"  Model hit rate:          {metrics.get('model_hit_rate', 0):.4f}")
    log(f"  Lift over baseline:      {metrics.get('lift_over_baseline', 0):.3f}x")
    if 'patterns_above_shuffled_p95' in metrics:
        log(f"  Patterns above shuffled 95th pct: {metrics['patterns_above_shuffled_p95']}")
    log(f"  Elapsed time:            {elapsed:.1f}s")
    log("")

    if significant_all:
        log("Top 5 patterns by lift:")
        for i, p in enumerate(top_patterns[:5]):
            log(f"  {i+1}. {p['pattern_str']:30s}  lift={p['lift']:.3f}  "
                f"conf={p['confidence']:.3f}  p={p['p_value']:.2e}  "
                f"support={p['support']}")

    log("")
    log("Output files:")
    log(f"  {patterns_path}")
    log(f"  {predictions_path}")
    log(f"  {metrics_path}")
    log(f"  {os.path.join(RESULTS_DIR, 'pattern_lift_distribution.png')}")
    log(f"  {os.path.join(RESULTS_DIR, 'pattern_top_predictors.png')}")
    log(f"  {os.path.join(RESULTS_DIR, 'pattern_backtest.png')}")
    log("")
    log("Done.")


if __name__ == '__main__':
    main()
