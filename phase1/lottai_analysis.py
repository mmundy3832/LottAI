#!/usr/bin/env python3
"""
LottAI Deep Analysis
====================
Comprehensive statistical analysis of Lotto Texas draw data.
Searches for patterns, biases, and structure in the draw history.

Outputs:
  - Console: progress + highlighted findings
  - results/ directory: plots (PNG) and summary (JSON)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy as scipy_entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations
from math import comb, factorial
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
import os
import json
import time

# ============================================================================
# Configuration
# ============================================================================
CSV_FILE = 'lottotexas.csv'
RESULTS_DIR = 'results'
NUMS_PER_DRAW = 6
SIGNIFICANCE = 0.01       # primary threshold for flagging
# TDA_PERMUTATIONS = 200    # number of random datasets for TDA permutation test
TDA_PERMUTATIONS = 50     # 50 is enough for valid null distribution; runs 4x faster
os.makedirs(RESULTS_DIR, exist_ok=True)

findings = []
start_time = time.time()

def log_finding(category, description, p_value=None, effect_size=None, details=None):
    finding = {
        'category': category,
        'description': description,
        'p_value': float(p_value) if p_value is not None else None,
        'effect_size': float(effect_size) if effect_size is not None else None,
        'details': details,
    }
    findings.append(finding)
    if p_value is not None and p_value < 0.001:
        marker = "!!!"
    elif p_value is not None and p_value < 0.01:
        marker = "!! "
    elif p_value is not None and p_value < 0.05:
        marker = "!  "
    else:
        marker = "** "
    print(f"\n{marker} FINDING [{category}]: {description}")
    if p_value is not None:
        print(f"    p-value: {p_value:.8f}")
    if effect_size is not None:
        print(f"    effect size: {effect_size:.4f}")

def section(title):
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"  {title}  [{elapsed:.0f}s elapsed]")
    print(f"{'='*80}\n")

def save_plot(fig, name):
    path = f'{RESULTS_DIR}/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot saved: {path}]")


# ============================================================================
# DATA LOADING
# ============================================================================
section("DATA LOADING")

df_raw = pd.read_csv(CSV_FILE, header=None,
                     names=['game', 'month', 'day', 'year', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6'])
df_raw['date'] = pd.to_datetime(df_raw[['year', 'month', 'day']])
df_raw = df_raw.sort_values('date').reset_index(drop=True)

# Filter to post-April-2006 data only.
# The game changed from pick-6-from-50 to pick-6-from-54 in April 2006.
# Pre-2006 data uses a different number range and pollutes the analysis.
n_total_raw = len(df_raw)
cutoff_date = pd.Timestamp('2006-04-01')
df_raw = df_raw[df_raw['date'] >= cutoff_date].reset_index(drop=True)
n_filtered_out = n_total_raw - len(df_raw)
print(f"Loaded {n_total_raw} total draws from CSV.")
print(f"Filtered out {n_filtered_out} pre-April-2006 draws (pick-6-from-50 era).")
print(f"Keeping {len(df_raw)} draws from {cutoff_date.date()} onward (pick-6-from-54 era).\n")

ncols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
draws = df_raw[ncols].values.astype(int)
dates = df_raw['date']
n_draws = len(draws)

# Determine number range from data
min_num = draws.min()
max_num = draws.max()
num_range = max_num  # assuming 1-based
all_numbers = draws.flatten()

print(f"Total draws: {n_draws}")
print(f"Date range: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
print(f"Number range: {min_num} to {max_num}")
print(f"Numbers per draw: {NUMS_PER_DRAW}")
print(f"Combination space: C({num_range},{NUMS_PER_DRAW}) = "
      f"{comb(num_range, NUMS_PER_DRAW):,}")
print(f"Sample coverage: {n_draws / comb(num_range, NUMS_PER_DRAW) * 100:.6f}%")

# Day of week
df_raw['dow'] = df_raw['date'].dt.dayofweek  # 0=Mon, 6=Sun
df_raw['dow_name'] = df_raw['date'].dt.day_name()
df_raw['month_num'] = df_raw['date'].dt.month
df_raw['year_num'] = df_raw['date'].dt.year

# Sorted draws (order-independent representation)
draws_sorted = np.sort(draws, axis=1)

# Binary indicator matrix: each draw as a 54-dim binary vector
binary_matrix = np.zeros((n_draws, num_range), dtype=int)
for i, draw in enumerate(draws):
    for n in draw:
        binary_matrix[i, n - 1] = 1


# ============================================================================
# 1. INDIVIDUAL NUMBER FREQUENCY
# ============================================================================
section("1. INDIVIDUAL NUMBER FREQUENCY ANALYSIS")

expected_freq = (NUMS_PER_DRAW * n_draws) / num_range
expected_std = np.sqrt(n_draws * (NUMS_PER_DRAW / num_range) * (1 - NUMS_PER_DRAW / num_range))

freq_counter = Counter(all_numbers)
frequencies = np.array([freq_counter.get(i, 0) for i in range(1, num_range + 1)])

print(f"Expected frequency per number: {expected_freq:.1f} +/- {expected_std:.1f} (1 sigma)")
print(f"Actual mean: {frequencies.mean():.1f}, std: {frequencies.std():.1f}")

# Top/bottom
sorted_idx = np.argsort(-frequencies)
print(f"\nMost frequent:")
for i in range(5):
    n = sorted_idx[i] + 1
    f = frequencies[sorted_idx[i]]
    z = (f - expected_freq) / expected_std
    print(f"  Number {n:2d}: {f:3d} times (z={z:+.2f}, {(f/expected_freq - 1)*100:+.1f}%)")
print(f"Least frequent:")
for i in range(5):
    n = sorted_idx[-(i+1)] + 1
    f = frequencies[sorted_idx[-(i+1)]]
    z = (f - expected_freq) / expected_std
    print(f"  Number {n:2d}: {f:3d} times (z={z:+.2f}, {(f/expected_freq - 1)*100:+.1f}%)")

# Chi-squared
chi2, p_chi = stats.chisquare(frequencies)
print(f"\nChi-squared test for uniformity: chi2={chi2:.2f}, df={num_range-1}, p={p_chi:.6f}")
if p_chi < SIGNIFICANCE:
    log_finding("Frequency", f"Number distribution is NOT uniform (chi2={chi2:.2f}, p={p_chi:.6f})", p_value=p_chi, effect_size=chi2/n_draws)
else:
    print(f"  => Consistent with uniform distribution")

# Z-scores for individual numbers
z_scores = (frequencies - expected_freq) / expected_std
outlier_numbers = np.where(np.abs(z_scores) > 2.5)[0] + 1
if len(outlier_numbers) > 0:
    for n in outlier_numbers:
        log_finding("Frequency Outlier",
                    f"Number {n} has z-score {z_scores[n-1]:+.2f} ({frequencies[n-1]} times vs expected {expected_freq:.0f})",
                    p_value=2 * (1 - stats.norm.cdf(abs(z_scores[n-1]))),
                    effect_size=z_scores[n-1])

# Plot
fig, ax = plt.subplots(figsize=(16, 6))
colors = ['crimson' if abs(z) > 2 else 'steelblue' for z in z_scores]
ax.bar(range(1, num_range+1), frequencies, color=colors, edgecolor='black', linewidth=0.3)
ax.axhline(expected_freq, color='red', ls='--', lw=1.5, label=f'Expected ({expected_freq:.0f})')
ax.axhline(expected_freq + 2*expected_std, color='orange', ls=':', alpha=0.7, label='±2σ')
ax.axhline(expected_freq - 2*expected_std, color='orange', ls=':', alpha=0.7)
ax.set_xlabel('Number')
ax.set_ylabel('Frequency')
ax.set_title(f'Lotto Texas Number Frequencies (n={n_draws} draws) — Red = |z| > 2')
ax.legend()
ax.set_xticks(range(1, num_range+1, 2))
save_plot(fig, '01_number_frequencies.png')


# ============================================================================
# 2. POSITION-DEPENDENT FREQUENCY (Draw Order Bias)
# ============================================================================
section("2. POSITION-DEPENDENT FREQUENCY (Draw Order Bias)")

print("Testing if position within a draw affects which numbers appear.")
print("If the machine has mechanical bias, the first ball drawn may differ from the last.\n")

position_freqs = np.zeros((NUMS_PER_DRAW, num_range), dtype=int)
for pos in range(NUMS_PER_DRAW):
    for n in draws[:, pos]:
        position_freqs[pos, n - 1] += 1

# Chi-squared test per position
for pos in range(NUMS_PER_DRAW):
    chi2_pos, p_pos = stats.chisquare(position_freqs[pos])
    print(f"  Position {pos+1}: chi2={chi2_pos:.1f}, p={p_pos:.4f}", end="")
    if p_pos < SIGNIFICANCE:
        print(f"  <--- SIGNIFICANT")
        log_finding("Position Bias",
                    f"Position {pos+1} shows non-uniform number distribution (chi2={chi2_pos:.1f}, p={p_pos:.6f})",
                    p_value=p_pos, effect_size=chi2_pos/(n_draws))
    else:
        print()

# Position mean values (if biased, early positions might favor lower/higher numbers)
pos_means = [draws[:, pos].mean() for pos in range(NUMS_PER_DRAW)]
print(f"\nMean number by draw position: {['%.1f' % m for m in pos_means]}")
print(f"Expected mean (uniform 1-{num_range}): {(num_range + 1) / 2:.1f}")

# Kruskal-Wallis test: do positions have different distributions?
kw_stat, kw_p = stats.kruskal(*[draws[:, pos] for pos in range(NUMS_PER_DRAW)])
print(f"\nKruskal-Wallis test across positions: H={kw_stat:.2f}, p={kw_p:.6f}")
if kw_p < SIGNIFICANCE:
    log_finding("Position Bias",
                f"Number distributions differ significantly across draw positions (H={kw_stat:.2f}, p={kw_p:.6f})",
                p_value=kw_p)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for pos, ax in enumerate(axes.flat):
    ax.bar(range(1, num_range+1), position_freqs[pos], color='steelblue', edgecolor='black', linewidth=0.2)
    ax.axhline(n_draws / num_range, color='red', ls='--', lw=1)
    ax.set_title(f'Position {pos+1} (mean={pos_means[pos]:.1f})')
    ax.set_xlabel('Number')
    ax.set_ylabel('Count')
fig.suptitle('Number Frequency by Draw Position', fontsize=14)
plt.tight_layout()
save_plot(fig, '02_position_frequencies.png')


# ============================================================================
# 3. GAP ANALYSIS (Time Between Appearances)
# ============================================================================
section("3. GAP ANALYSIS")

print("For each number, measuring the gap (in draws) between consecutive appearances.")
print("Under true randomness, gaps follow a geometric distribution with p = 6/54.\n")

p_appear = NUMS_PER_DRAW / num_range  # probability of appearing in any draw
expected_mean_gap = 1.0 / p_appear

# Compute gaps for each number
gap_data = {}
gap_pvalues = []
for num in range(1, num_range + 1):
    if num % 10 == 0:
        print(f"  Processing number {num}/{num_range}...", flush=True)
    appearances = np.where(binary_matrix[:, num-1] == 1)[0]
    if len(appearances) > 1:
        gaps = np.diff(appearances)
        gap_data[num] = gaps

        # Test against geometric distribution
        # Geometric dist: P(gap=k) = (1-p)^(k-1) * p
        # Mean = 1/p, Variance = (1-p)/p^2
        mean_gap = gaps.mean()
        # KS test against expected distribution
        # We use exponential approximation for large gaps
        ks_stat, ks_p = stats.kstest(gaps, 'expon', args=(0, expected_mean_gap))
        gap_pvalues.append((num, ks_stat, ks_p, mean_gap, len(gaps)))

# Sort by p-value
gap_pvalues.sort(key=lambda x: x[2])
bonferroni_gap = SIGNIFICANCE / num_range

print(f"Expected mean gap: {expected_mean_gap:.1f} draws")
print(f"Bonferroni-corrected threshold: {bonferroni_gap:.6f}\n")
print(f"Numbers with most unusual gap distributions:")
for num, ks, p, mean_g, n_gaps in gap_pvalues[:10]:
    sig = " ***" if p < bonferroni_gap else ""
    print(f"  Number {num:2d}: mean gap={mean_g:.1f}, KS={ks:.3f}, p={p:.6f}, n_gaps={n_gaps}{sig}")
    if p < bonferroni_gap:
        log_finding("Gap Distribution",
                    f"Number {num} has unusual gap distribution (mean={mean_g:.1f} vs expected {expected_mean_gap:.1f}, KS p={p:.6f})",
                    p_value=p, effect_size=mean_g / expected_mean_gap)

# Current gaps (how long since each number last appeared) - useful for "hot"/"cold"
current_gaps = []
for num in range(1, num_range + 1):
    last_app = np.where(binary_matrix[:, num-1] == 1)[0]
    if len(last_app) > 0:
        current_gaps.append((num, n_draws - 1 - last_app[-1]))
    else:
        current_gaps.append((num, n_draws))
current_gaps.sort(key=lambda x: -x[1])
print(f"\nColdest numbers (longest since last appearance):")
for num, gap in current_gaps[:10]:
    print(f"  Number {num:2d}: {gap} draws since last appearance")
print(f"Hottest numbers (most recent appearance):")
for num, gap in current_gaps[-5:]:
    print(f"  Number {num:2d}: {gap} draws since last appearance")


# ============================================================================
# 4. SUM DISTRIBUTION
# ============================================================================
section("4. SUM OF DRAWN NUMBERS")

draw_sums = draws_sorted.sum(axis=1)
print(f"Mean sum: {draw_sums.mean():.1f}")
print(f"Expected mean (uniform): {NUMS_PER_DRAW * (num_range + 1) / 2:.1f}")
print(f"Std: {draw_sums.std():.1f}")
print(f"Min: {draw_sums.min()}, Max: {draw_sums.max()}")

# Normality test on sums (CLT says should be approximately normal)
shapiro_stat, shapiro_p = stats.shapiro(draw_sums[:5000])  # shapiro limited to 5000
print(f"Shapiro-Wilk normality test: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")

# Compare to theoretical distribution via simulation
sim_sums = []
rng = np.random.default_rng(42)
for _ in range(100000):
    sim_draw = rng.choice(range(1, num_range + 1), size=NUMS_PER_DRAW, replace=False)
    sim_sums.append(sim_draw.sum())
sim_sums = np.array(sim_sums)
ks_stat, ks_p = stats.ks_2samp(draw_sums, sim_sums)
print(f"KS test vs simulated uniform draws: D={ks_stat:.4f}, p={ks_p:.6f}")
if ks_p < SIGNIFICANCE:
    log_finding("Sum Distribution",
                f"Sum distribution differs from expected (KS D={ks_stat:.4f}, p={ks_p:.6f})",
                p_value=ks_p, effect_size=ks_stat)

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(draw_sums, bins=50, density=True, alpha=0.7, label='Actual draws', color='steelblue', edgecolor='black')
ax.hist(sim_sums, bins=50, density=True, alpha=0.4, label='Simulated uniform', color='orange')
ax.set_xlabel('Sum of 6 numbers')
ax.set_ylabel('Density')
ax.set_title('Distribution of Draw Sums: Actual vs Expected')
ax.legend()
save_plot(fig, '03_sum_distribution.png')


# ============================================================================
# 5. ODD/EVEN AND HIGH/LOW PATTERNS
# ============================================================================
section("5. ODD/EVEN AND HIGH/LOW PATTERNS")

odd_counts = np.sum(draws_sorted % 2 == 1, axis=1)
even_counts = NUMS_PER_DRAW - odd_counts
mid = num_range // 2
high_counts = np.sum(draws_sorted > mid, axis=1)
low_counts = NUMS_PER_DRAW - high_counts

# Expected distribution of odd count: Hypergeometric
# 27 odd numbers, 27 even numbers, drawing 6
n_odd_in_range = sum(1 for i in range(1, num_range+1) if i % 2 == 1)
n_even_in_range = num_range - n_odd_in_range

print(f"Odd numbers in 1-{num_range}: {n_odd_in_range}, Even: {n_even_in_range}")
print(f"\nOdd count distribution:")
odd_counter = Counter(odd_counts)
for k in sorted(odd_counter.keys()):
    expected = stats.hypergeom.pmf(k, num_range, n_odd_in_range, NUMS_PER_DRAW) * n_draws
    actual = odd_counter[k]
    print(f"  {k} odd: {actual:4d} times (expected {expected:.0f})")

# Chi-squared test
observed_odd = np.array([odd_counter.get(k, 0) for k in range(NUMS_PER_DRAW + 1)])
expected_odd = np.array([stats.hypergeom.pmf(k, num_range, n_odd_in_range, NUMS_PER_DRAW) * n_draws
                         for k in range(NUMS_PER_DRAW + 1)])
# Remove zero-expected bins
mask = expected_odd > 1
chi2_odd, p_odd = stats.chisquare(observed_odd[mask], expected_odd[mask])
print(f"\nChi-squared (odd count): chi2={chi2_odd:.2f}, p={p_odd:.4f}")
if p_odd < SIGNIFICANCE:
    log_finding("Odd/Even", f"Odd/even ratio deviates from expected (p={p_odd:.6f})", p_value=p_odd)

print(f"\nHigh/Low split at {mid}:")
hl_counter = Counter(high_counts)
for k in sorted(hl_counter.keys()):
    n_high = sum(1 for i in range(1, num_range+1) if i > mid)
    expected = stats.hypergeom.pmf(k, num_range, n_high, NUMS_PER_DRAW) * n_draws
    print(f"  {k} high: {hl_counter[k]:4d} times (expected {expected:.0f})")


# ============================================================================
# 6. CONSECUTIVE NUMBER ANALYSIS
# ============================================================================
section("6. CONSECUTIVE NUMBER ANALYSIS")

consec_counts = []
for draw in draws_sorted:
    diffs = np.diff(draw)
    n_consec = np.sum(diffs == 1)
    consec_counts.append(n_consec)
consec_counts = np.array(consec_counts)
consec_counter = Counter(consec_counts)

print(f"Draws with consecutive numbers:")
for k in sorted(consec_counter.keys()):
    print(f"  {k} consecutive pair(s): {consec_counter[k]:4d} ({consec_counter[k]/n_draws*100:.1f}%)")

# Simulate expected
sim_consec = []
for _ in range(100000):
    d = np.sort(rng.choice(range(1, num_range+1), size=NUMS_PER_DRAW, replace=False))
    sim_consec.append(np.sum(np.diff(d) == 1))
sim_consec = np.array(sim_consec)
ks_stat, ks_p = stats.ks_2samp(consec_counts.astype(float), sim_consec.astype(float))
print(f"\nKS test vs simulated: D={ks_stat:.4f}, p={ks_p:.6f}")
if ks_p < SIGNIFICANCE:
    log_finding("Consecutive", f"Consecutive number pattern deviates from expected (p={ks_p:.6f})", p_value=ks_p)

print(f"Actual mean consecutive pairs: {consec_counts.mean():.3f}")
print(f"Simulated mean: {sim_consec.mean():.3f}")


# ============================================================================
# 7. DAY-OF-WEEK AND MONTHLY EFFECTS
# ============================================================================
section("7. TEMPORAL EFFECTS (Day-of-Week, Month, Year)")

# Day of week
print("Day-of-week draw frequency:")
dow_counts = df_raw['dow_name'].value_counts()
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
    if day in dow_counts:
        print(f"  {day}: {dow_counts[day]} draws")

# Do number frequencies differ by day of week?
draw_days = df_raw['dow'].values
unique_days = np.unique(draw_days)
if len(unique_days) > 1:
    # For each day, compute mean number
    day_means = {}
    for d in unique_days:
        mask = draw_days == d
        day_draws = draws[mask]
        day_means[d] = day_draws.mean()
        day_freqs = np.zeros(num_range)
        for num in day_draws.flatten():
            day_freqs[num-1] += 1
        # Normalize
        day_freqs = day_freqs / day_freqs.sum() * NUMS_PER_DRAW

    # Kruskal-Wallis on draw sums by day
    groups_by_day = [draw_sums[draw_days == d] for d in unique_days]
    groups_by_day = [g for g in groups_by_day if len(g) > 5]
    if len(groups_by_day) > 1:
        kw_stat, kw_p = stats.kruskal(*groups_by_day)
        day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        print(f"\nDraw sum by day-of-week (Kruskal-Wallis): H={kw_stat:.2f}, p={kw_p:.4f}")
        for d in unique_days:
            mask = draw_days == d
            print(f"  {day_names[d]}: mean sum = {draw_sums[mask].mean():.1f}, n={mask.sum()}")
        if kw_p < SIGNIFICANCE:
            log_finding("Day-of-Week", f"Draw sums differ by day of week (p={kw_p:.6f})", p_value=kw_p)

# Monthly effects
print(f"\nMonthly draw counts and mean sums:")
for m in range(1, 13):
    mask = df_raw['month_num'].values == m
    if mask.sum() > 0:
        print(f"  Month {m:2d}: {mask.sum():3d} draws, mean sum = {draw_sums[mask].mean():.1f}")

# Year-over-year trend
print(f"\nYearly mean number frequency (detecting equipment drift):")
years = sorted(df_raw['year_num'].unique())
yearly_chi2 = []
for y in years:
    mask = df_raw['year_num'].values == y
    y_draws = draws[mask]
    y_freq = np.zeros(num_range)
    for num in y_draws.flatten():
        y_freq[num-1] += 1
    expected_y = y_freq.sum() / num_range
    if expected_y > 0:
        chi2_y, p_y = stats.chisquare(y_freq)
        yearly_chi2.append((y, chi2_y, p_y, mask.sum()))
        sig = " *" if p_y < 0.05 else ""
        print(f"  {y}: {mask.sum():3d} draws, chi2={chi2_y:.1f}, p={p_y:.4f}{sig}")

# Test for trend in frequencies over time using sliding window
section("7b. FREQUENCY DRIFT (Sliding Window)")
window_size = 200
step = 50
drift_results = []
for start in range(0, n_draws - window_size, step):
    end = start + window_size
    window_draws = draws[start:end]
    w_freq = np.zeros(num_range)
    for num in window_draws.flatten():
        w_freq[num-1] += 1
    expected_w = w_freq.sum() / num_range
    chi2_w, p_w = stats.chisquare(w_freq)
    drift_results.append((start, end, dates.iloc[start].date(), dates.iloc[end-1].date(), chi2_w, p_w))

# Plot drift
fig, ax = plt.subplots(figsize=(14, 5))
x_vals = [r[0] for r in drift_results]
chi2_vals = [r[4] for r in drift_results]
p_vals = [r[5] for r in drift_results]
ax.plot(x_vals, chi2_vals, 'b-', alpha=0.7)
ax.axhline(stats.chi2.ppf(0.95, num_range - 1), color='orange', ls='--', label='95% threshold')
ax.axhline(stats.chi2.ppf(0.99, num_range - 1), color='red', ls='--', label='99% threshold')
ax.set_xlabel('Draw index')
ax.set_ylabel('Chi-squared statistic')
ax.set_title(f'Frequency Uniformity Over Time (sliding window={window_size})')
ax.legend()
save_plot(fig, '04_frequency_drift.png')

# Find windows with significant non-uniformity
sig_windows = [(s, e, d1, d2, c, p) for s, e, d1, d2, c, p in drift_results if p < 0.01]
if sig_windows:
    print(f"\nTime periods with significant non-uniformity (p<0.01):")
    for s, e, d1, d2, c, p in sig_windows[:5]:
        print(f"  Draws {s}-{e} ({d1} to {d2}): chi2={c:.1f}, p={p:.4f}")
    log_finding("Frequency Drift",
                f"Found {len(sig_windows)} time windows with significant frequency non-uniformity",
                p_value=min(p for _, _, _, _, _, p in sig_windows),
                details=f"{len(sig_windows)} windows out of {len(drift_results)} tested")


# ============================================================================
# 8. PAIRWISE CO-OCCURRENCE
# ============================================================================
section("8. PAIRWISE CO-OCCURRENCE ANALYSIS")

print("Testing whether certain number pairs appear together more/less than expected.")
print(f"Total pairs: C({num_range},2) = {num_range*(num_range-1)//2}")

# Expected co-occurrence probability
# P(both i and j in draw) = C(N-2, k-2) / C(N, k)
p_pair = comb(num_range - 2, NUMS_PER_DRAW - 2) / comb(num_range, NUMS_PER_DRAW)
expected_pair = p_pair * n_draws
print(f"Expected co-occurrence per pair: {expected_pair:.2f}")
bonferroni_pairs = SIGNIFICANCE / comb(num_range, 2)
print(f"Bonferroni-corrected threshold: {bonferroni_pairs:.8f}")

# Compute co-occurrence matrix
cooccur = binary_matrix.T @ binary_matrix  # shape: (54, 54)
np.fill_diagonal(cooccur, 0)

# Test each pair
pair_results = []
for i in range(num_range):
    for j in range(i+1, num_range):
        observed = cooccur[i, j]
        # Binomial test
        p_binom = stats.binom_test(observed, n_draws, p_pair) if hasattr(stats, 'binom_test') else \
                  stats.binomtest(observed, n_draws, p_pair).pvalue
        z = (observed - expected_pair) / np.sqrt(n_draws * p_pair * (1 - p_pair))
        pair_results.append((i+1, j+1, observed, z, p_binom))

pair_results.sort(key=lambda x: x[4])  # sort by p-value

print(f"\nMost over-represented pairs:")
for n1, n2, obs, z, p in pair_results[:10]:
    sig = " ***" if p < bonferroni_pairs else ""
    print(f"  ({n1:2d}, {n2:2d}): {obs:3d} times (z={z:+.2f}, p={p:.6f}){sig}")
    if p < bonferroni_pairs:
        log_finding("Co-occurrence",
                    f"Pair ({n1},{n2}) appears {obs} times (expected {expected_pair:.0f}, z={z:+.2f})",
                    p_value=p, effect_size=z)

print(f"\nMost under-represented pairs:")
pair_results_under = sorted(pair_results, key=lambda x: -x[4])
for n1, n2, obs, z, p in [(r[0],r[1],r[2],r[3],r[4]) for r in sorted(pair_results, key=lambda x: x[3])[:10]]:
    sig = " ***" if stats.binomtest(obs, n_draws, p_pair).pvalue < bonferroni_pairs else ""
    print(f"  ({n1:2d}, {n2:2d}): {obs:3d} times (z={z:+.2f}){sig}")

# Co-occurrence heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cooccur, cmap='RdBu_r', center=expected_pair, ax=ax,
            xticklabels=5, yticklabels=5)
ax.set_title('Number Co-occurrence Matrix (red=over, blue=under expected)')
ax.set_xlabel('Number')
ax.set_ylabel('Number')
save_plot(fig, '05_cooccurrence_heatmap.png')

# How many pairs are significant after Bonferroni?
n_sig_pairs = sum(1 for _, _, _, _, p in pair_results if p < bonferroni_pairs)
print(f"\nPairs significant after Bonferroni correction: {n_sig_pairs} / {len(pair_results)}")
if n_sig_pairs > 0:
    log_finding("Co-occurrence",
                f"{n_sig_pairs} pairs significant after Bonferroni correction (threshold={bonferroni_pairs:.8f})",
                details=[{"pair": (r[0], r[1]), "count": r[2], "z": round(r[3], 2)} for r in pair_results if r[4] < bonferroni_pairs])


# ============================================================================
# 9. SEQUENTIAL DRAW CORRELATION
# ============================================================================
section("9. SEQUENTIAL DRAW CORRELATION")

print("Testing if consecutive draws are correlated.")
print("Do numbers from draw N predict anything about draw N+1?\n")

# Method 1: Jaccard similarity between consecutive draws
jaccard_consecutive = []
for i in range(n_draws - 1):
    intersection = np.sum(binary_matrix[i] & binary_matrix[i+1])
    union = np.sum(binary_matrix[i] | binary_matrix[i+1])
    jaccard_consecutive.append(intersection / union if union > 0 else 0)
jaccard_consecutive = np.array(jaccard_consecutive)

# Expected Jaccard for random draws of 6 from 54
# E[|intersection|] = 6*6/54 = 0.667, E[|union|] = 6+6-0.667 = 11.333
# E[Jaccard] ~ 0.667/11.333 ~ 0.059
sim_jaccard = []
for _ in range(50000):
    d1 = set(rng.choice(range(1, num_range+1), size=NUMS_PER_DRAW, replace=False))
    d2 = set(rng.choice(range(1, num_range+1), size=NUMS_PER_DRAW, replace=False))
    inter = len(d1 & d2)
    union = len(d1 | d2)
    sim_jaccard.append(inter / union if union > 0 else 0)
sim_jaccard = np.array(sim_jaccard)

print(f"Mean Jaccard similarity (consecutive draws): {jaccard_consecutive.mean():.4f}")
print(f"Expected (random):                           {sim_jaccard.mean():.4f}")
ks_stat, ks_p = stats.ks_2samp(jaccard_consecutive, sim_jaccard)
print(f"KS test: D={ks_stat:.4f}, p={ks_p:.6f}")
if ks_p < SIGNIFICANCE:
    log_finding("Sequential Correlation",
                f"Consecutive draws show unusual similarity (Jaccard: actual={jaccard_consecutive.mean():.4f} vs expected={sim_jaccard.mean():.4f})",
                p_value=ks_p)

# Method 2: Number repeat rate
repeats = []
for i in range(n_draws - 1):
    s1 = set(draws[i])
    s2 = set(draws[i+1])
    repeats.append(len(s1 & s2))
repeats = np.array(repeats)
repeat_counter = Counter(repeats)
print(f"\nNumbers repeated in consecutive draws:")
for k in sorted(repeat_counter.keys()):
    exp = stats.hypergeom.pmf(k, num_range, NUMS_PER_DRAW, NUMS_PER_DRAW) * len(repeats)
    print(f"  {k} repeats: {repeat_counter[k]:4d} times (expected {exp:.0f})")

# Method 3: Autocorrelation of each number's appearance
print(f"\nAutocorrelation of individual numbers (lag 1):")
autocorr_vals = []
for num in range(1, num_range + 1):
    if num % 10 == 0:
        print(f"  Processing number {num}/{num_range}...", flush=True)
    series = binary_matrix[:, num-1].astype(float)
    series_centered = series - series.mean()
    if series_centered.std() > 0:
        ac1 = np.corrcoef(series_centered[:-1], series_centered[1:])[0, 1]
        autocorr_vals.append((num, ac1))
autocorr_vals.sort(key=lambda x: -abs(x[1]))

print(f"  Highest autocorrelation:")
for num, ac in autocorr_vals[:5]:
    # Test significance: under null, ac ~ N(0, 1/sqrt(n))
    z_ac = ac * np.sqrt(n_draws)
    p_ac = 2 * (1 - stats.norm.cdf(abs(z_ac)))
    sig = " ***" if p_ac < SIGNIFICANCE / num_range else ""
    print(f"    Number {num:2d}: r={ac:+.4f} (z={z_ac:+.2f}, p={p_ac:.6f}){sig}")
    if p_ac < SIGNIFICANCE / num_range:
        log_finding("Autocorrelation",
                    f"Number {num} shows significant lag-1 autocorrelation (r={ac:+.4f})",
                    p_value=p_ac, effect_size=ac)


# ============================================================================
# 10. SPECTRAL ANALYSIS (FFT)
# ============================================================================
section("10. SPECTRAL ANALYSIS (FFT)")

print("Looking for periodic patterns in number appearances.\n")

# FFT on each number's binary time series
significant_freqs = []
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# Pick 9 numbers to plot (most frequent, least frequent, and some in between)
plot_numbers = list(sorted_idx[:3] + 1) + list(sorted_idx[num_range//2-1:num_range//2+2] + 1) + list(sorted_idx[-3:] + 1)

for idx, (num, ax) in enumerate(zip(plot_numbers, axes.flat)):
    series = binary_matrix[:, num-1].astype(float)
    series_detrended = series - series.mean()

    N = len(series_detrended)
    yf = np.abs(fft(series_detrended)[:N//2])
    xf = fftfreq(N, d=1)[:N//2]

    # Skip DC component
    yf = yf[1:]
    xf = xf[1:]

    # Find dominant frequency
    peak_idx = np.argmax(yf)
    peak_freq = xf[peak_idx]
    peak_power = yf[peak_idx]

    ax.plot(xf, yf, 'b-', alpha=0.7, linewidth=0.5)
    ax.axvline(peak_freq, color='red', ls='--', alpha=0.5)
    ax.set_title(f'Number {num} (peak period: {1/peak_freq:.1f} draws)')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')

fig.suptitle('FFT Spectral Analysis of Number Appearances', fontsize=14)
plt.tight_layout()
save_plot(fig, '06_spectral_analysis.png')

# Aggregate spectral analysis: look for shared periodicities
print("Dominant periods across all numbers:")
period_counter = Counter()
for num in range(1, num_range + 1):
    if num % 10 == 0:
        print(f"  Processing number {num}/{num_range}...", flush=True)
    series = binary_matrix[:, num-1].astype(float) - binary_matrix[:, num-1].mean()
    N = len(series)
    yf = np.abs(fft(series)[:N//2])[1:]
    xf = fftfreq(N, d=1)[:N//2][1:]

    # Top 3 peaks
    top3 = np.argsort(-yf)[:3]
    for pi in top3:
        period = int(round(1.0 / xf[pi]))
        if 2 < period < n_draws // 2:
            period_counter[period] += 1

print("  Most common dominant periods (shared across numbers):")
for period, count in period_counter.most_common(10):
    print(f"    Period ~{period} draws: dominant in {count}/{num_range} numbers")


# ============================================================================
# 11. MUTUAL INFORMATION ANALYSIS
# ============================================================================
section("11. MUTUAL INFORMATION BETWEEN DRAWS")

print("Measuring information shared between consecutive draws.\n")

# Discretize: for each draw, compute a simple hash/signature
# Use sum, odd-count, and max as features
def draw_features(draw):
    s = sum(draw)
    odd = sum(1 for x in draw if x % 2 == 1)
    mx = max(draw)
    mn = min(draw)
    return s, odd, mx, mn

features_current = np.array([draw_features(draws[i]) for i in range(n_draws - 1)])
features_next = np.array([draw_features(draws[i+1]) for i in range(n_draws - 1)])

# Compute mutual information for each feature pair
from sklearn.metrics import mutual_info_score

feature_names = ['Sum', 'OddCount', 'Max', 'Min']
for fidx, fname in enumerate(feature_names):
    # Bin continuous features
    f_curr = pd.qcut(features_current[:, fidx], q=10, labels=False, duplicates='drop')
    f_next = pd.qcut(features_next[:, fidx], q=10, labels=False, duplicates='drop')
    mi = mutual_info_score(f_curr, f_next)

    # Compare to shuffled baseline
    mi_shuffled = []
    for shuf_i in range(1000):
        if (shuf_i + 1) % 200 == 0:
            print(f"    {fname} shuffle {shuf_i+1}/1000...", flush=True)
        shuffled = rng.permutation(f_next)
        mi_shuffled.append(mutual_info_score(f_curr, shuffled))
    mi_shuffled = np.array(mi_shuffled)
    z_mi = (mi - mi_shuffled.mean()) / mi_shuffled.std() if mi_shuffled.std() > 0 else 0
    p_mi = np.mean(mi_shuffled >= mi)

    print(f"  {fname}: MI={mi:.4f}, shuffled mean={mi_shuffled.mean():.4f}, z={z_mi:.2f}, p={p_mi:.4f}")
    if p_mi < SIGNIFICANCE:
        log_finding("Mutual Information",
                    f"{fname} of consecutive draws shows significant mutual information (MI={mi:.4f}, p={p_mi:.4f})",
                    p_value=p_mi, effect_size=mi)


# ============================================================================
# 12. RUNS TEST
# ============================================================================
section("12. RUNS TEST FOR RANDOMNESS")

print("Testing if each number's appearance/absence sequence has random runs.\n")

runs_results = []
for num in range(1, num_range + 1):
    if num % 10 == 0:
        print(f"  Processing number {num}/{num_range}...", flush=True)
    series = binary_matrix[:, num-1]
    n1 = series.sum()
    n0 = len(series) - n1
    if n1 == 0 or n0 == 0:
        continue

    # Count runs
    runs = 1
    for i in range(1, len(series)):
        if series[i] != series[i-1]:
            runs += 1

    # Expected runs under randomness
    expected_runs = 1 + (2 * n0 * n1) / (n0 + n1)
    var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1)**2 * (n0 + n1 - 1))
    z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
    runs_results.append((num, runs, expected_runs, z_runs, p_runs))

runs_results.sort(key=lambda x: x[4])
bonferroni_runs = SIGNIFICANCE / num_range
print(f"Numbers with most non-random run patterns (Bonferroni threshold: {bonferroni_runs:.6f}):")
for num, runs, exp, z, p in runs_results[:10]:
    sig = " ***" if p < bonferroni_runs else ""
    print(f"  Number {num:2d}: {runs} runs (expected {exp:.0f}, z={z:+.2f}, p={p:.6f}){sig}")
    if p < bonferroni_runs:
        log_finding("Runs Test",
                    f"Number {num} has non-random run pattern ({runs} runs, expected {exp:.0f}, z={z:+.2f})",
                    p_value=p, effect_size=z)


# ============================================================================
# 13. TDA / PERSISTENT HOMOLOGY
# ============================================================================
section("13. TOPOLOGICAL DATA ANALYSIS (Persistent Homology)")

# Subsample for TDA if dataset is large (distance matrix is O(n^2))
TDA_MAX_DRAWS = 2500
tda_binary_matrix = binary_matrix
tda_n_draws = n_draws
if n_draws > TDA_MAX_DRAWS:
    print(f"Dataset has {n_draws} draws (>{TDA_MAX_DRAWS}). Subsampling to 2000 for TDA only.")
    tda_rng = np.random.default_rng(seed=42)
    tda_indices = tda_rng.choice(n_draws, size=2000, replace=False)
    tda_indices.sort()
    tda_binary_matrix = binary_matrix[tda_indices]
    tda_n_draws = len(tda_binary_matrix)
    print(f"  Subsampled to {tda_n_draws} draws (seed=42 for reproducibility).")
else:
    print(f"Dataset has {n_draws} draws (<={TDA_MAX_DRAWS}). Using all draws for TDA.")

print(f"\nRepresenting each draw as a {num_range}-dim binary vector.")
print(f"Computing Jaccard distance matrix ({tda_n_draws} x {tda_n_draws})...")

# Jaccard distance (on TDA subset)
tda_dist_matrix = squareform(pdist(tda_binary_matrix, metric='jaccard'))
print(f"Distance matrix computed. Shape: {tda_dist_matrix.shape}")
print(f"Mean Jaccard distance: {tda_dist_matrix[np.triu_indices_from(tda_dist_matrix, k=1)].mean():.4f}")

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude

    print(f"\nRunning Vietoris-Rips persistence (H0 and H1)...")

    vr = VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0, 1], max_edge_length=1.0)
    diagrams = vr.fit_transform(tda_dist_matrix[np.newaxis, :, :])

    # Extract H0 and H1
    diag = diagrams[0]
    h0 = diag[diag[:, 2] == 0][:, :2]
    h1 = diag[diag[:, 2] == 1][:, :2]

    h0_persistence = h0[:, 1] - h0[:, 0]
    h1_persistence = h1[:, 1] - h1[:, 0]

    print(f"\nH0 (connected components): {len(h0)} features")
    print(f"  Max persistence: {h0_persistence.max():.4f}" if len(h0) > 0 else "  No features")
    print(f"  Mean persistence: {h0_persistence.mean():.4f}" if len(h0) > 0 else "")

    print(f"\nH1 (loops/holes): {len(h1)} features")
    if len(h1) > 0:
        print(f"  Max persistence: {h1_persistence.max():.4f}")
        print(f"  Mean persistence: {h1_persistence.mean():.4f}")
        print(f"  Top 5 most persistent H1 features:")
        h1_sorted = h1[np.argsort(-(h1[:, 1] - h1[:, 0]))]
        for i, feat in enumerate(h1_sorted[:5]):
            print(f"    birth={feat[0]:.4f}, death={feat[1]:.4f}, persistence={feat[1]-feat[0]:.4f}")
    else:
        print("  No H1 features found")

    # Persistence entropy
    pe = PersistenceEntropy()
    entropies = pe.fit_transform(diagrams)
    print(f"\nPersistence entropy: H0={entropies[0, 0]:.4f}, H1={entropies[0, 1]:.4f}")

    # Permutation test
    print(f"\nRunning permutation test ({TDA_PERMUTATIONS} random datasets)...")
    real_max_h0 = h0_persistence.max() if len(h0) > 0 else 0
    real_max_h1 = h1_persistence.max() if len(h1) > 0 else 0
    real_total_h1 = h1_persistence.sum() if len(h1) > 0 else 0
    real_entropy_h0 = entropies[0, 0]
    real_entropy_h1 = entropies[0, 1]

    perm_max_h0 = []
    perm_max_h1 = []
    perm_total_h1 = []
    perm_entropy_h0 = []
    perm_entropy_h1 = []

    for perm_i in range(TDA_PERMUTATIONS):
        print(f"  Permutation {perm_i+1}/{TDA_PERMUTATIONS}...", flush=True)

        # Generate random draws (same size as TDA dataset)
        rand_binary = np.zeros((tda_n_draws, num_range), dtype=int)
        for row in range(tda_n_draws):
            chosen = rng.choice(num_range, size=NUMS_PER_DRAW, replace=False)
            rand_binary[row, chosen] = 1

        rand_dist = squareform(pdist(rand_binary, metric='jaccard'))
        rand_diag = vr.transform(rand_dist[np.newaxis, :, :])
        rd = rand_diag[0]
        rh0 = rd[rd[:, 2] == 0][:, :2]
        rh1 = rd[rd[:, 2] == 1][:, :2]

        rh0_pers = rh0[:, 1] - rh0[:, 0] if len(rh0) > 0 else np.array([0])
        rh1_pers = rh1[:, 1] - rh1[:, 0] if len(rh1) > 0 else np.array([0])

        perm_max_h0.append(rh0_pers.max() if len(rh0_pers) > 0 else 0)
        perm_max_h1.append(rh1_pers.max() if len(rh1_pers) > 0 else 0)
        perm_total_h1.append(rh1_pers.sum() if len(rh1_pers) > 0 else 0)

        rand_entropy = pe.transform(rand_diag)
        perm_entropy_h0.append(rand_entropy[0, 0])
        perm_entropy_h1.append(rand_entropy[0, 1])

    perm_max_h0 = np.array(perm_max_h0)
    perm_max_h1 = np.array(perm_max_h1)
    perm_total_h1 = np.array(perm_total_h1)
    perm_entropy_h0 = np.array(perm_entropy_h0)
    perm_entropy_h1 = np.array(perm_entropy_h1)

    p_max_h0 = np.mean(perm_max_h0 >= real_max_h0)
    p_max_h1 = np.mean(perm_max_h1 >= real_max_h1)
    p_total_h1 = np.mean(perm_total_h1 >= real_total_h1)
    p_ent_h0 = np.mean(np.abs(perm_entropy_h0 - perm_entropy_h0.mean()) >=
                        np.abs(real_entropy_h0 - perm_entropy_h0.mean()))
    p_ent_h1 = np.mean(np.abs(perm_entropy_h1 - perm_entropy_h1.mean()) >=
                        np.abs(real_entropy_h1 - perm_entropy_h1.mean()))

    print(f"\nPermutation test results:")
    print(f"  Max H0 persistence: real={real_max_h0:.4f}, "
          f"perm mean={perm_max_h0.mean():.4f}, p={p_max_h0:.4f}")
    print(f"  Max H1 persistence: real={real_max_h1:.4f}, "
          f"perm mean={perm_max_h1.mean():.4f}, p={p_max_h1:.4f}")
    print(f"  Total H1 persistence: real={real_total_h1:.4f}, "
          f"perm mean={perm_total_h1.mean():.4f}, p={p_total_h1:.4f}")
    print(f"  H0 entropy: real={real_entropy_h0:.4f}, "
          f"perm mean={perm_entropy_h0.mean():.4f}, p={p_ent_h0:.4f}")
    print(f"  H1 entropy: real={real_entropy_h1:.4f}, "
          f"perm mean={perm_entropy_h1.mean():.4f}, p={p_ent_h1:.4f}")

    for name, real_val, p_val in [
        ("Max H0 persistence", real_max_h0, p_max_h0),
        ("Max H1 persistence", real_max_h1, p_max_h1),
        ("Total H1 persistence", real_total_h1, p_total_h1),
        ("H0 entropy", real_entropy_h0, p_ent_h0),
        ("H1 entropy", real_entropy_h1, p_ent_h1),
    ]:
        if p_val < SIGNIFICANCE:
            log_finding("TDA",
                        f"{name} is significantly different from random (real={real_val:.4f}, p={p_val:.4f})",
                        p_value=p_val, effect_size=real_val)

    # Plot persistence diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    if len(h0) > 0:
        ax1.scatter(h0[:, 0], h0[:, 1], s=15, alpha=0.5, label='H0')
    if len(h1) > 0:
        ax1.scatter(h1[:, 0], h1[:, 1], s=15, alpha=0.5, label='H1', color='red')
    lim = max(h0[:, 1].max() if len(h0) > 0 else 1, h1[:, 1].max() if len(h1) > 0 else 1)
    ax1.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax1.set_xlabel('Birth')
    ax1.set_ylabel('Death')
    ax1.set_title('Persistence Diagram (Real Data)')
    ax1.legend()

    # Compare max persistence distributions
    ax2.hist(perm_max_h1, bins=30, alpha=0.7, label='Random (H1 max)', color='gray')
    ax2.axvline(real_max_h1, color='red', lw=2, ls='--', label=f'Real data (p={p_max_h1:.3f})')
    ax2.set_xlabel('Max H1 Persistence')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Permutation Test: Max H1 Persistence (n={TDA_PERMUTATIONS})')
    ax2.legend()
    save_plot(fig, '07_tda_persistence.png')

except Exception as e:
    print(f"TDA analysis failed: {e}")
    print("Continuing with remaining analyses...")


# ============================================================================
# 14. CO-OCCURRENCE NETWORK ANALYSIS
# ============================================================================
section("14. CO-OCCURRENCE NETWORK ANALYSIS")

# Build graph where edge weight = co-occurrence z-score
G = nx.Graph()
for i in range(1, num_range + 1):
    G.add_node(i, frequency=frequencies[i-1])

# Add edges for pairs with |z| > 1.5
for n1, n2, obs, z, p in pair_results:
    if abs(z) > 1.5:
        G.add_edge(n1, n2, weight=z, abs_weight=abs(z), co_occurrence=obs)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (|z| > 1.5)")

# Community detection
if G.number_of_edges() > 0:
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G, weight='abs_weight'))
    print(f"\nDetected {len(communities)} communities:")
    for i, comm in enumerate(communities):
        print(f"  Community {i+1}: {sorted(comm)}")
    modularity = nx.algorithms.community.modularity(G, communities, weight='abs_weight')
    print(f"Modularity: {modularity:.4f}")

    # Is modularity significant? Compare to random graph
    random_mods = []
    for net_i in range(500):
        if (net_i + 1) % 50 == 0:
            print(f"  Network permutation {net_i+1}/500...", flush=True)
        G_rand = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
        # Assign random weights
        for u, v in G_rand.edges():
            G_rand[u][v]['abs_weight'] = rng.random()
        try:
            rand_comms = list(greedy_modularity_communities(G_rand, weight='abs_weight'))
            random_mods.append(nx.algorithms.community.modularity(G_rand, rand_comms, weight='abs_weight'))
        except:
            pass
    if random_mods:
        random_mods = np.array(random_mods)
        p_mod = np.mean(random_mods >= modularity)
        z_mod = (modularity - random_mods.mean()) / random_mods.std() if random_mods.std() > 0 else 0
        print(f"  vs random: mean={random_mods.mean():.4f}, p={p_mod:.4f}, z={z_mod:.2f}")
        if p_mod < SIGNIFICANCE:
            log_finding("Network",
                        f"Co-occurrence network has significant community structure (modularity={modularity:.4f}, p={p_mod:.4f})",
                        p_value=p_mod, effect_size=modularity)

    # Degree centrality
    deg_cent = nx.degree_centrality(G)
    top_central = sorted(deg_cent.items(), key=lambda x: -x[1])[:10]
    print(f"\nMost connected numbers (degree centrality):")
    for num, cent in top_central:
        print(f"  Number {num:2d}: centrality={cent:.3f}")

# Network visualization
if G.number_of_edges() > 0:
    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=2, seed=42, weight='abs_weight')
    node_sizes = [frequencies[n-1] * 5 for n in G.nodes()]
    node_colors = [frequencies[n-1] for n in G.nodes()]

    # Separate positive and negative edges
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color='red', alpha=0.3, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color='blue', alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           cmap='YlOrRd', alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title('Co-occurrence Network (red=over, blue=under expected, node size=frequency)')
    ax.axis('off')
    save_plot(fig, '08_cooccurrence_network.png')


# ============================================================================
# 15. PCA & DIMENSIONALITY REDUCTION
# ============================================================================
section("15. DIMENSIONALITY REDUCTION (PCA & t-SNE)")

# PCA on binary matrix
pca = PCA(n_components=10)
pca_result = pca.fit_transform(binary_matrix)
print(f"PCA explained variance ratios (top 10):")
for i, var in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)")
print(f"Total explained by 10 PCs: {pca.explained_variance_ratio_[:10].sum()*100:.1f}%")

# Expected for random binary vectors: all PCs should explain ~equal variance
expected_var = 1.0 / min(num_range, n_draws)
print(f"\nExpected variance per PC (random): ~{expected_var:.4f}")

# Is PC1 significantly larger than expected?
# Use Marchenko-Pastur distribution for comparison
gamma = n_draws / num_range  # aspect ratio
mp_upper = (1 + 1/np.sqrt(gamma))**2 / num_range
actual_pc1 = pca.explained_variance_ratio_[0]
print(f"Marchenko-Pastur upper edge: {mp_upper:.4f}")
print(f"Actual PC1: {actual_pc1:.4f}")
if actual_pc1 > mp_upper * 1.5:
    log_finding("PCA",
                f"PC1 variance ({actual_pc1:.4f}) exceeds Marchenko-Pastur bound ({mp_upper:.4f}), suggesting structure",
                effect_size=actual_pc1 / mp_upper)

# t-SNE (uses full dataset, not TDA subsample)
print("\nComputing full Jaccard distance matrix for t-SNE...")
dist_matrix_full = squareform(pdist(binary_matrix, metric='jaccard'))
print("Computing t-SNE (2D)...")
# tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='precomputed')
tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='precomputed', init='random')
tsne_result = tsne.fit_transform(dist_matrix_full)

# Color by time
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sc1 = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=range(n_draws), cmap='viridis', s=5, alpha=0.5)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('PCA of Draws (colored by time)')
plt.colorbar(sc1, ax=axes[0], label='Draw index')

sc2 = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=range(n_draws), cmap='viridis', s=5, alpha=0.5)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].set_title('t-SNE of Draws (colored by time)')
plt.colorbar(sc2, ax=axes[1], label='Draw index')
save_plot(fig, '09_dimensionality_reduction.png')


# ============================================================================
# 16. ENTROPY ANALYSIS
# ============================================================================
section("16. ENTROPY ANALYSIS")

# Shannon entropy of each draw (based on number values)
# For a uniform selection of 6 from 54, theoretical max entropy is log2(C(54,6))
draw_entropies = []
for draw in draws_sorted:
    # Treat draw as a probability distribution over 6 values
    probs = draw / draw.sum()
    h = scipy_entropy(probs, base=2)
    draw_entropies.append(h)
draw_entropies = np.array(draw_entropies)

print(f"Mean draw entropy: {draw_entropies.mean():.4f}")
print(f"Std: {draw_entropies.std():.4f}")

# Is entropy changing over time? (Linear regression)
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(range(n_draws), draw_entropies)
print(f"\nEntropy trend over time: slope={slope:.6f}, r={r_value:.4f}, p={p_value:.4f}")
if p_value < SIGNIFICANCE:
    log_finding("Entropy Trend",
                f"Draw entropy shows significant time trend (slope={slope:.6f}, p={p_value:.6f})",
                p_value=p_value, effect_size=r_value)

# Sliding window entropy
window_entropies = []
for start in range(0, n_draws - window_size, step):
    end = start + window_size
    window_binary = binary_matrix[start:end]
    # Column-wise frequency -> probability -> entropy
    col_probs = window_binary.mean(axis=0)
    col_probs = col_probs[col_probs > 0]  # remove zeros
    h = scipy_entropy(col_probs, base=2)
    window_entropies.append((start, h))

fig, ax = plt.subplots(figsize=(14, 5))
we_x = [w[0] for w in window_entropies]
we_y = [w[1] for w in window_entropies]
ax.plot(we_x, we_y, 'b-', alpha=0.7)
ax.set_xlabel('Draw index')
ax.set_ylabel('Entropy (bits)')
ax.set_title(f'Distribution Entropy Over Time (window={window_size})')
save_plot(fig, '10_entropy_over_time.png')


# ============================================================================
# 17. LAST DIGIT ANALYSIS
# ============================================================================
section("17. LAST DIGIT DISTRIBUTION")

last_digits = all_numbers % 10
ld_counter = Counter(last_digits)
ld_freq = np.array([ld_counter.get(d, 0) for d in range(10)])

# Expected: not uniform because numbers 1-54 don't have uniform last digits
# Count how many numbers in 1-54 have each last digit
expected_ld = np.zeros(10)
for n in range(1, num_range + 1):
    expected_ld[n % 10] += 1
expected_ld = expected_ld / expected_ld.sum() * len(all_numbers)

print("Last digit distribution:")
for d in range(10):
    print(f"  Digit {d}: {ld_freq[d]:5d} (expected {expected_ld[d]:.0f})")

chi2_ld, p_ld = stats.chisquare(ld_freq, expected_ld)
print(f"\nChi-squared test: chi2={chi2_ld:.2f}, p={p_ld:.6f}")
if p_ld < SIGNIFICANCE:
    log_finding("Last Digit", f"Last digit distribution anomalous (chi2={chi2_ld:.2f}, p={p_ld:.6f})", p_value=p_ld)


# ============================================================================
# 18. NUMBER SPREAD ANALYSIS
# ============================================================================
section("18. NUMBER SPREAD (Range & Spacing)")

draw_ranges = draws_sorted[:, -1] - draws_sorted[:, 0]
draw_spacings = np.diff(draws_sorted, axis=1)  # gaps between consecutive sorted numbers
mean_spacings = draw_spacings.mean(axis=1)

print(f"Draw range (max - min):")
print(f"  Mean: {draw_ranges.mean():.1f}, Std: {draw_ranges.std():.1f}")
print(f"  Min: {draw_ranges.min()}, Max: {draw_ranges.max()}")

print(f"\nMean spacing between consecutive sorted numbers:")
print(f"  Mean: {mean_spacings.mean():.2f}, Std: {mean_spacings.std():.2f}")

# Compare to simulation
sim_ranges = []
sim_spacings = []
for _ in range(100000):
    d = np.sort(rng.choice(range(1, num_range+1), size=NUMS_PER_DRAW, replace=False))
    sim_ranges.append(d[-1] - d[0])
    sim_spacings.append(np.diff(d).mean())
sim_ranges = np.array(sim_ranges)
sim_spacings = np.array(sim_spacings)

ks_range, p_range = stats.ks_2samp(draw_ranges.astype(float), sim_ranges.astype(float))
print(f"\nKS test (range): D={ks_range:.4f}, p={p_range:.6f}")
if p_range < SIGNIFICANCE:
    log_finding("Number Spread",
                f"Draw range distribution differs from expected (actual mean={draw_ranges.mean():.1f}, sim={sim_ranges.mean():.1f}, p={p_range:.6f})",
                p_value=p_range)


# ============================================================================
# SUMMARY
# ============================================================================
section("SUMMARY OF ALL FINDINGS")

elapsed_total = time.time() - start_time
print(f"Analysis completed in {elapsed_total:.0f} seconds.\n")

if not findings:
    print("No statistically significant findings detected.")
    print("The data appears consistent with a uniform random process.")
else:
    print(f"Total findings: {len(findings)}\n")

    # Group by category
    categories = {}
    for f in findings:
        cat = f['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(f)

    for cat, cat_findings in sorted(categories.items()):
        print(f"\n--- {cat} ({len(cat_findings)} findings) ---")
        for f in cat_findings:
            p_str = f"p={f['p_value']:.6f}" if f['p_value'] is not None else "no p-value"
            print(f"  {f['description']} ({p_str})")

    # Multiple testing correction (Benjamini-Hochberg)
    print(f"\n\n--- MULTIPLE TESTING CORRECTION (Benjamini-Hochberg) ---")
    pvalues_with_idx = [(i, f['p_value']) for i, f in enumerate(findings) if f['p_value'] is not None]
    if pvalues_with_idx:
        pvalues_with_idx.sort(key=lambda x: x[1])
        m = len(pvalues_with_idx)
        bh_significant = []
        for rank, (idx, p) in enumerate(pvalues_with_idx, 1):
            bh_threshold = rank / m * 0.05  # FDR = 5%
            if p <= bh_threshold:
                bh_significant.append(findings[idx])

        print(f"\nFindings surviving BH correction (FDR=5%): {len(bh_significant)} / {len(pvalues_with_idx)}")
        for f in bh_significant:
            print(f"  [{f['category']}] {f['description']} (p={f['p_value']:.6f})")

        if not bh_significant:
            print("  None. All findings may be due to multiple testing.")

# Save findings to JSON
findings_json = {
    'analysis_date': str(pd.Timestamp.now()),
    'n_draws': n_draws,
    'date_range': f"{dates.iloc[0].date()} to {dates.iloc[-1].date()}",
    'elapsed_seconds': elapsed_total,
    'n_findings': len(findings),
    'findings': findings,
}

with open(f'{RESULTS_DIR}/findings.json', 'w') as f:
    json.dump(findings_json, f, indent=2, default=str)
print(f"\nDetailed findings saved to {RESULTS_DIR}/findings.json")
print(f"Plots saved to {RESULTS_DIR}/ directory")
print(f"\n{'='*80}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*80}")
