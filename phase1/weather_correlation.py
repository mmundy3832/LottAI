#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weather-Lottery Correlation Analysis for Lotto Texas
=====================================================
Comprehensive analysis of whether Austin weather conditions on draw dates
correlate with lottery outcomes. Covers correlation, mutual information,
logistic regression, OLS, Granger causality, and random forest approaches.

Run with: .venv/Scripts/python.exe -u weather_correlation.py
"""

import os
import sys
import json
import time
import warnings
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

LOTTERY_CSV = os.path.join(BASE_DIR, 'lottotexas.csv')
WEATHER_CSV = os.path.join(BASE_DIR, 'austin_weather_2006_2024.csv')

LOG_FILE = os.path.join(RESULTS_DIR, 'weather_correlation.log')
FINDINGS_FILE = os.path.join(RESULTS_DIR, 'weather_correlation_findings.json')

N_NUMBERS = 54  # Lotto Texas draws from 1..54
N_DRAWN = 6     # 6 numbers per draw
MI_PERMUTATIONS = 200
GRANGER_MAX_LAG = 3

# Significance thresholds
ALPHA = 0.05
FDR_Q = 0.05

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger('weather_corr')
logger.setLevel(logging.DEBUG)

# File handler
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def log(msg, level='info'):
    """Log a message to both file and stdout."""
    getattr(logger, level)(msg)


# ---------------------------------------------------------------------------
# Utility: FDR correction helper
# ---------------------------------------------------------------------------
def fdr_correct(pvalues, method='fdr_bh', alpha=FDR_Q):
    """Apply FDR or Bonferroni correction. Returns (reject, corrected_p)."""
    pvals = np.asarray(pvalues, dtype=float)
    # Handle NaNs by replacing with 1.0
    mask = np.isfinite(pvals)
    if mask.sum() == 0:
        return np.zeros(len(pvals), dtype=bool), np.ones(len(pvals))
    reject = np.zeros(len(pvals), dtype=bool)
    corrected = np.ones(len(pvals))
    if mask.sum() > 0:
        r, cp, _, _ = multipletests(pvals[mask], alpha=alpha, method=method)
        reject[mask] = r
        corrected[mask] = cp
    return reject, corrected


def safe_json(obj):
    """Make an object JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


# ============================================================================
# PHASE 0: Data Loading & Joining
# ============================================================================
def phase0_load_and_join():
    log("=" * 70)
    log("PHASE 0: Data Loading & Joining")
    log("=" * 70)

    # Load lottery data (no header)
    lotto_cols = ['game', 'month', 'day', 'year',
                  'num1', 'num2', 'num3', 'num4', 'num5', 'num6']
    df_lotto = pd.read_csv(LOTTERY_CSV, header=None, names=lotto_cols)
    log(f"Loaded lottery CSV: {len(df_lotto)} rows")

    # Construct date
    df_lotto['date'] = pd.to_datetime(
        df_lotto[['year', 'month', 'day']].rename(
            columns={'year': 'year', 'month': 'month', 'day': 'day'}
        )
    )
    df_lotto = df_lotto.drop(columns=['game', 'month', 'day', 'year'])

    # Sort the drawn numbers so num1 <= num2 <= ... <= num6
    num_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
    sorted_nums = np.sort(df_lotto[num_cols].values, axis=1)
    for i, c in enumerate(num_cols):
        df_lotto[c] = sorted_nums[:, i]

    log(f"Lottery date range: {df_lotto['date'].min()} to {df_lotto['date'].max()}")

    # Load weather data (has header)
    df_weather = pd.read_csv(WEATHER_CSV, parse_dates=['date'])
    log(f"Loaded weather CSV: {len(df_weather)} rows")
    log(f"Weather date range: {df_weather['date'].min()} to {df_weather['date'].max()}")

    # Inner join on date
    df = pd.merge(df_lotto, df_weather, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    log(f"Joined dataset: {len(df)} draws with matching weather data")

    if len(df) == 0:
        log("ERROR: No matching dates between lottery and weather data!", level='error')
        sys.exit(1)

    return df, df_weather


# ============================================================================
# PHASE 1: Derived Lottery Features
# ============================================================================
def phase1_lottery_features(df):
    log("")
    log("=" * 70)
    log("PHASE 1: Derived Lottery Features")
    log("=" * 70)

    num_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
    nums = df[num_cols].values

    # Aggregate features
    df['draw_sum'] = nums.sum(axis=1)
    df['draw_mean'] = nums.mean(axis=1)
    df['draw_std'] = nums.std(axis=1, ddof=1)
    df['draw_range'] = nums.max(axis=1) - nums.min(axis=1)

    # Odd count and high count
    df['odd_count'] = np.sum(nums % 2 == 1, axis=1)
    df['high_count'] = np.sum(nums >= 28, axis=1)
    df['odd_ratio'] = df['odd_count'] / N_DRAWN
    df['high_ratio'] = df['high_count'] / N_DRAWN

    # Consecutive pairs: count how many pairs of adjacent drawn numbers differ by 1
    consecutive = 0
    consec_list = []
    for row in nums:
        sorted_row = np.sort(row)
        pairs = np.sum(np.diff(sorted_row) == 1)
        consec_list.append(pairs)
    df['consecutive_pairs'] = consec_list

    # Binary indicator columns for each number 1..54
    for k in range(1, N_NUMBERS + 1):
        df[f'number_present_{k}'] = np.any(nums == k, axis=1).astype(int)

    log(f"Created aggregate features: draw_sum, draw_mean, draw_std, draw_range, "
        f"odd_count, high_count, odd_ratio, high_ratio, consecutive_pairs")
    log(f"Created 54 binary indicator columns (number_present_1..54)")
    log(f"draw_sum range: {df['draw_sum'].min()} to {df['draw_sum'].max()}, "
        f"mean={df['draw_sum'].mean():.1f}")

    return df


# ============================================================================
# PHASE 2: Weather Feature Preparation
# ============================================================================
def phase2_weather_features(df):
    log("")
    log("=" * 70)
    log("PHASE 2: Weather Feature Preparation")
    log("=" * 70)

    # Use mean values as representatives
    df['temp'] = df['temp_mean_f']
    df['apparent_temp'] = df['apparent_temp_mean_f']
    df['pressure'] = df['pressure_msl_mean_hpa']
    df['humidity'] = df['humidity_mean_pct']
    df['wind_speed'] = df['wind_speed_mean_mph']
    df['precipitation'] = df['precipitation_mm']

    # Derived range features
    df['temp_range'] = df['temp_max_f'] - df['temp_min_f']
    df['pressure_range'] = df['pressure_msl_max_hpa'] - df['pressure_msl_min_hpa']
    df['humidity_range'] = df['humidity_max_pct'] - df['humidity_min_pct']

    # Binary features
    df['is_rainy'] = (df['precipitation'] > 0.5).astype(int)
    pressure_median = df['pressure'].median()
    df['is_high_pressure'] = (df['pressure'] > pressure_median).astype(int)

    # Pressure delta (1-day change) - requires sorted by date
    df = df.sort_values('date').reset_index(drop=True)
    # Compute pressure change from previous draw (not previous calendar day)
    df['pressure_delta_1d'] = df['pressure'].diff()
    # Fill first value with 0
    df['pressure_delta_1d'] = df['pressure_delta_1d'].fillna(0)

    weather_features = [
        'temp', 'apparent_temp', 'pressure', 'humidity',
        'wind_speed', 'precipitation', 'temp_range',
        'pressure_range', 'humidity_range', 'is_rainy',
        'is_high_pressure', 'pressure_delta_1d'
    ]

    log(f"Weather features ({len(weather_features)}): {weather_features}")
    for feat in weather_features:
        log(f"  {feat}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}, "
            f"range=[{df[feat].min():.2f}, {df[feat].max():.2f}]")

    return df, weather_features


# ============================================================================
# PHASE 3: Seasonal Detrending
# ============================================================================
def phase3_seasonal_detrending(df, df_weather_full, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 3: Seasonal Detrending")
    log("=" * 70)

    # Compute day-of-year smoothed means from FULL weather dataset
    df_weather_full = df_weather_full.copy()
    df_weather_full['doy'] = df_weather_full['date'].dt.dayofyear

    # Map raw weather column names to our feature names
    raw_to_feature = {
        'temp': 'temp_mean_f',
        'apparent_temp': 'apparent_temp_mean_f',
        'pressure': 'pressure_msl_mean_hpa',
        'humidity': 'humidity_mean_pct',
        'wind_speed': 'wind_speed_mean_mph',
        'precipitation': 'precipitation_mm',
    }

    # Continuous weather features to detrend
    continuous_weather = ['temp', 'apparent_temp', 'pressure', 'humidity',
                          'wind_speed', 'precipitation', 'temp_range',
                          'pressure_range', 'humidity_range', 'pressure_delta_1d']

    # Compute derived features on full weather data for seasonal baseline
    df_weather_full['temp_range_full'] = df_weather_full['temp_max_f'] - df_weather_full['temp_min_f']
    df_weather_full['pressure_range_full'] = df_weather_full['pressure_msl_max_hpa'] - df_weather_full['pressure_msl_min_hpa']
    df_weather_full['humidity_range_full'] = df_weather_full['humidity_max_pct'] - df_weather_full['humidity_min_pct']
    df_weather_full['pressure_delta_full'] = df_weather_full['pressure_msl_mean_hpa'].diff().fillna(0)

    full_feature_map = {
        'temp': 'temp_mean_f',
        'apparent_temp': 'apparent_temp_mean_f',
        'pressure': 'pressure_msl_mean_hpa',
        'humidity': 'humidity_mean_pct',
        'wind_speed': 'wind_speed_mean_mph',
        'precipitation': 'precipitation_mm',
        'temp_range': 'temp_range_full',
        'pressure_range': 'pressure_range_full',
        'humidity_range': 'humidity_range_full',
        'pressure_delta_1d': 'pressure_delta_full',
    }

    # Compute smoothed day-of-year means (circular smoothing with window ~15 days)
    doy_means = {}
    for feat, raw_col in full_feature_map.items():
        if raw_col not in df_weather_full.columns:
            continue
        doy_avg = df_weather_full.groupby('doy')[raw_col].mean()
        # Circular smooth: tile 3x, smooth, take middle
        vals = np.concatenate([doy_avg.values] * 3)
        window = 15
        kernel = np.ones(window) / window
        smoothed_full = np.convolve(vals, kernel, mode='same')
        n = len(doy_avg)
        smoothed = smoothed_full[n:2*n]
        doy_means[feat] = dict(zip(doy_avg.index, smoothed))

    # Store original values for diagnostic plot
    detrend_diagnostics = {}

    # Detrend weather features on the joined draw data
    df['doy'] = df['date'].dt.dayofyear
    for feat in continuous_weather:
        if feat in doy_means:
            original = df[feat].copy()
            seasonal = df['doy'].map(doy_means[feat])
            # Handle any NaN from leap year day 366
            seasonal = seasonal.fillna(df[feat].mean())
            df[f'{feat}_detrended'] = df[feat] - seasonal
            if feat == 'temp':
                detrend_diagnostics['temp_original'] = original.values
                detrend_diagnostics['temp_seasonal'] = seasonal.values
                detrend_diagnostics['temp_detrended'] = df['temp_detrended'].values
                detrend_diagnostics['dates'] = df['date'].values

    log(f"Detrended {len(continuous_weather)} continuous weather features using day-of-year smoothed means")

    # Check for linear trend in lottery features
    aggregate_lottery = ['draw_sum', 'draw_mean', 'draw_std', 'draw_range',
                         'odd_ratio', 'high_ratio', 'consecutive_pairs']
    trend_results = {}
    x_idx = np.arange(len(df))
    for feat in aggregate_lottery:
        slope, intercept, r_val, p_val, std_err = stats.linregress(x_idx, df[feat].values)
        trend_results[feat] = {
            'slope': slope, 'p_value': p_val, 'r_squared': r_val**2
        }
        if p_val < 0.05:
            log(f"  Significant linear trend in {feat}: slope={slope:.6f}, p={p_val:.4f} -> detrending")
            df[f'{feat}_detrended'] = df[feat] - (slope * x_idx + intercept) + df[feat].mean()
        else:
            df[f'{feat}_detrended'] = df[feat]
            log(f"  No significant trend in {feat}: p={p_val:.4f}")

    # Add post_aug2021 binary control variable
    # August 2021: Texas Lottery changed from 6/54 machine configuration
    cutoff = pd.Timestamp('2021-08-01')
    df['post_aug2021'] = (df['date'] >= cutoff).astype(int)
    n_post = df['post_aug2021'].sum()
    log(f"post_aug2021 control variable: {n_post} draws after cutoff, "
        f"{len(df) - n_post} before")

    # Add sin/cos seasonal controls
    df['sin_doy'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    return df, detrend_diagnostics, trend_results


# ============================================================================
# PHASE 4: Correlation Analysis
# ============================================================================
def phase4_correlation(df, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 4: Correlation Analysis")
    log("=" * 70)

    aggregate_lottery = ['draw_sum', 'draw_mean', 'draw_std', 'draw_range',
                         'odd_ratio', 'high_ratio', 'consecutive_pairs',
                         'odd_count', 'high_count']

    # Use detrended weather features where available
    weather_cols = []
    for f in weather_features:
        if f'{f}_detrended' in df.columns:
            weather_cols.append(f'{f}_detrended')
        else:
            weather_cols.append(f)

    # --- Pearson correlations: weather vs aggregate lottery ---
    log("Computing Pearson correlations (weather vs aggregate lottery)...")
    pearson_results = []
    pearson_matrix = np.zeros((len(weather_features), len(aggregate_lottery)))
    pearson_pvals = np.zeros_like(pearson_matrix)

    for i, wf in enumerate(weather_cols):
        for j, lf in enumerate(aggregate_lottery):
            lf_col = f'{lf}_detrended' if f'{lf}_detrended' in df.columns else lf
            r, p = stats.pearsonr(df[wf].values, df[lf_col].values)
            pearson_matrix[i, j] = r
            pearson_pvals[i, j] = p
            pearson_results.append({
                'weather': weather_features[i],
                'lottery': lf,
                'r': r, 'p': p, 'test': 'pearson'
            })

    # --- Point-biserial: weather vs each number's presence ---
    log("Computing point-biserial correlations (weather vs 54 number indicators)...")
    pb_results = []
    pb_matrix = np.zeros((N_NUMBERS, len(weather_features)))
    pb_pvals = np.zeros_like(pb_matrix)

    for k in range(1, N_NUMBERS + 1):
        y = df[f'number_present_{k}'].values
        for j, wf in enumerate(weather_cols):
            x = df[wf].values
            r, p = stats.pointbiserialr(y, x)
            pb_matrix[k-1, j] = r
            pb_pvals[k-1, j] = p
            pb_results.append({
                'number': k,
                'weather': weather_features[j],
                'r': r, 'p': p, 'test': 'point_biserial'
            })

    # --- Spearman rank correlations (robustness check) ---
    log("Computing Spearman rank correlations (weather vs aggregate lottery)...")
    spearman_results = []
    spearman_matrix = np.zeros((len(weather_features), len(aggregate_lottery)))
    spearman_pvals = np.zeros_like(spearman_matrix)

    for i, wf in enumerate(weather_cols):
        for j, lf in enumerate(aggregate_lottery):
            lf_col = f'{lf}_detrended' if f'{lf}_detrended' in df.columns else lf
            rho, p = stats.spearmanr(df[wf].values, df[lf_col].values)
            spearman_matrix[i, j] = rho
            spearman_pvals[i, j] = p
            spearman_results.append({
                'weather': weather_features[i],
                'lottery': lf,
                'rho': rho, 'p': p, 'test': 'spearman'
            })

    # --- Multiple testing correction ---
    all_pvals = (
        [r['p'] for r in pearson_results] +
        [r['p'] for r in pb_results] +
        [r['p'] for r in spearman_results]
    )
    n_pearson = len(pearson_results)
    n_pb = len(pb_results)
    n_spearman = len(spearman_results)
    n_total = len(all_pvals)

    log(f"Total correlation tests: {n_total} "
        f"(Pearson={n_pearson}, Point-biserial={n_pb}, Spearman={n_spearman})")

    raw_sig = sum(1 for p in all_pvals if p < ALPHA)
    fdr_reject, fdr_pvals = fdr_correct(all_pvals, method='fdr_bh')
    bonf_reject, bonf_pvals = fdr_correct(all_pvals, method='bonferroni')

    fdr_sig = fdr_reject.sum()
    bonf_sig = bonf_reject.sum()

    log(f"Raw significant (p<0.05): {raw_sig}/{n_total}")
    log(f"FDR-surviving (q<0.05): {fdr_sig}/{n_total}")
    log(f"Bonferroni-surviving: {bonf_sig}/{n_total}")

    # Expected by chance
    expected_raw = n_total * 0.05
    log(f"Expected by chance at alpha=0.05: ~{expected_raw:.0f}")

    # Top 10 strongest Pearson correlations
    pearson_sorted = sorted(pearson_results, key=lambda x: abs(x['r']), reverse=True)
    log("Top 10 Pearson correlations (by |r|):")
    for entry in pearson_sorted[:10]:
        log(f"  {entry['weather']:25s} vs {entry['lottery']:20s}: "
            f"r={entry['r']:+.4f}, p={entry['p']:.4e}")

    results = {
        'n_tests_total': n_total,
        'n_pearson': n_pearson,
        'n_point_biserial': n_pb,
        'n_spearman': n_spearman,
        'raw_significant': int(raw_sig),
        'fdr_significant': int(fdr_sig),
        'bonferroni_significant': int(bonf_sig),
        'expected_by_chance': float(expected_raw),
        'top10_pearson': [
            {'weather': e['weather'], 'lottery': e['lottery'],
             'r': float(e['r']), 'p': float(e['p'])}
            for e in pearson_sorted[:10]
        ],
    }

    return (results, pearson_matrix, pearson_pvals, pb_matrix, pb_pvals,
            spearman_matrix, spearman_pvals,
            weather_features, aggregate_lottery)


# ============================================================================
# PHASE 5: Mutual Information
# ============================================================================
def phase5_mutual_information(df, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 5: Mutual Information Analysis")
    log("=" * 70)

    aggregate_lottery = ['draw_sum', 'draw_mean', 'draw_std', 'draw_range',
                         'odd_ratio', 'high_ratio', 'consecutive_pairs']

    # Prepare weather matrix (use detrended where available)
    weather_cols = []
    for f in weather_features:
        if f'{f}_detrended' in df.columns:
            weather_cols.append(f'{f}_detrended')
        else:
            weather_cols.append(f)

    X_weather = df[weather_cols].values

    # Handle any NaN/inf
    X_weather = np.nan_to_num(X_weather, nan=0.0, posinf=0.0, neginf=0.0)

    # --- MI for aggregate lottery features (regression) ---
    log("Computing MI for aggregate lottery features...")
    mi_agg = np.zeros((len(aggregate_lottery), len(weather_features)))
    for j, lf in enumerate(aggregate_lottery):
        lf_col = f'{lf}_detrended' if f'{lf}_detrended' in df.columns else lf
        y = df[lf_col].values
        mi_vals = mutual_info_regression(X_weather, y, random_state=42, n_neighbors=5)
        mi_agg[j, :] = mi_vals

    log(f"MI aggregate matrix shape: {mi_agg.shape}")

    # --- MI for individual number presence (classification) ---
    log("Computing MI for individual number indicators (54 numbers)...")
    mi_nums = np.zeros((N_NUMBERS, len(weather_features)))
    for k in range(1, N_NUMBERS + 1):
        y = df[f'number_present_{k}'].values
        mi_vals = mutual_info_classif(X_weather, y, random_state=42, n_neighbors=5)
        mi_nums[k-1, :] = mi_vals

    log(f"MI numbers matrix shape: {mi_nums.shape}")

    # --- Permutation test for MI significance ---
    log(f"Running {MI_PERMUTATIONS} permutation surrogates for MI p-values...")
    t_start = time.time()

    # For aggregates
    mi_agg_pvals = np.ones_like(mi_agg)
    for j, lf in enumerate(aggregate_lottery):
        lf_col = f'{lf}_detrended' if f'{lf}_detrended' in df.columns else lf
        y = df[lf_col].values
        null_counts = np.zeros(len(weather_features))
        for perm_i in range(MI_PERMUTATIONS):
            y_perm = np.random.permutation(y)
            mi_perm = mutual_info_regression(X_weather, y_perm, random_state=perm_i, n_neighbors=5)
            null_counts += (mi_perm >= mi_agg[j, :]).astype(float)
        mi_agg_pvals[j, :] = (null_counts + 1) / (MI_PERMUTATIONS + 1)

    elapsed_agg = time.time() - t_start
    log(f"  Aggregate MI permutation test done in {elapsed_agg:.1f}s")

    # For number indicators (sample a subset for speed, or do all 54)
    mi_nums_pvals = np.ones_like(mi_nums)
    t_start2 = time.time()
    for k in range(1, N_NUMBERS + 1):
        y = df[f'number_present_{k}'].values
        null_counts = np.zeros(len(weather_features))
        for perm_i in range(MI_PERMUTATIONS):
            y_perm = np.random.permutation(y)
            mi_perm = mutual_info_classif(X_weather, y_perm, random_state=perm_i, n_neighbors=5)
            null_counts += (mi_perm >= mi_nums[k-1, :]).astype(float)
        mi_nums_pvals[k-1, :] = (null_counts + 1) / (MI_PERMUTATIONS + 1)
        if k % 10 == 0:
            log(f"  MI permutation: completed number {k}/54")

    elapsed_nums = time.time() - t_start2
    log(f"  Number MI permutation test done in {elapsed_nums:.1f}s")

    # FDR correction on all MI p-values
    all_mi_pvals = np.concatenate([mi_agg_pvals.ravel(), mi_nums_pvals.ravel()])
    fdr_reject_mi, _ = fdr_correct(all_mi_pvals, method='fdr_bh')
    raw_sig_mi = (all_mi_pvals < ALPHA).sum()
    fdr_sig_mi = fdr_reject_mi.sum()

    log(f"MI tests total: {len(all_mi_pvals)}")
    log(f"MI raw significant (p<0.05): {raw_sig_mi}")
    log(f"MI FDR-surviving: {fdr_sig_mi}")

    # Top MI values (aggregate)
    top_mi = []
    for j, lf in enumerate(aggregate_lottery):
        for i, wf in enumerate(weather_features):
            top_mi.append({
                'weather': wf, 'lottery': lf,
                'mi': float(mi_agg[j, i]),
                'p': float(mi_agg_pvals[j, i])
            })
    top_mi.sort(key=lambda x: x['mi'], reverse=True)
    log("Top 10 MI values (aggregate):")
    for entry in top_mi[:10]:
        log(f"  {entry['weather']:25s} vs {entry['lottery']:20s}: "
            f"MI={entry['mi']:.4f}, p={entry['p']:.3f}")

    results = {
        'n_tests': len(all_mi_pvals),
        'raw_significant': int(raw_sig_mi),
        'fdr_significant': int(fdr_sig_mi),
        'top10_mi_aggregate': top_mi[:10],
        'total_time_seconds': elapsed_agg + elapsed_nums,
    }

    return results, mi_agg, mi_nums, mi_agg_pvals, mi_nums_pvals, aggregate_lottery


# ============================================================================
# PHASE 6: Logistic Regression Per Number
# ============================================================================
def phase6_logistic_regression(df, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 6: Logistic Regression Per Number")
    log("=" * 70)

    # Weather features (detrended where available)
    weather_cols = []
    for f in weather_features:
        if f'{f}_detrended' in df.columns:
            weather_cols.append(f'{f}_detrended')
        else:
            weather_cols.append(f)

    # Control variables
    control_cols = ['sin_doy', 'cos_doy', 'post_aug2021']

    # Standardize weather features
    scaler = StandardScaler()
    X_weather_scaled = scaler.fit_transform(df[weather_cols].values)
    X_controls = df[control_cols].values
    X_full = np.column_stack([X_weather_scaled, X_controls])
    X_controls_only = X_controls.copy()

    # Add intercept for statsmodels
    X_full_sm = sm.add_constant(X_full)
    X_ctrl_sm = sm.add_constant(X_controls_only)

    lr_results = []
    omnibus_pvals = []
    l1_selected_features = {}

    log(f"Fitting logistic regression for each of {N_NUMBERS} numbers...")

    for k in range(1, N_NUMBERS + 1):
        y = df[f'number_present_{k}'].values

        # Skip if zero variance
        if y.sum() == 0 or y.sum() == len(y):
            lr_results.append({
                'number': k, 'lr_pval': 1.0, 'aic_full': np.nan,
                'aic_ctrl': np.nan, 'pseudo_r2': 0.0
            })
            omnibus_pvals.append(1.0)
            continue

        try:
            # Full model
            model_full = sm.Logit(y, X_full_sm)
            res_full = model_full.fit(disp=0, maxiter=100, method='bfgs')

            # Controls-only model
            model_ctrl = sm.Logit(y, X_ctrl_sm)
            res_ctrl = model_ctrl.fit(disp=0, maxiter=100, method='bfgs')

            # Likelihood ratio test
            lr_stat = -2 * (res_ctrl.llf - res_full.llf)
            df_diff = X_full_sm.shape[1] - X_ctrl_sm.shape[1]
            lr_pval = stats.chi2.sf(lr_stat, df_diff)

            lr_results.append({
                'number': k,
                'lr_pval': float(lr_pval),
                'lr_stat': float(lr_stat),
                'aic_full': float(res_full.aic),
                'aic_ctrl': float(res_ctrl.aic),
                'pseudo_r2': float(res_full.prsquared),
            })
            omnibus_pvals.append(lr_pval)

        except Exception as e:
            log(f"  Number {k}: convergence issue ({str(e)[:60]})", level='warning')
            lr_results.append({
                'number': k, 'lr_pval': 1.0,
                'aic_full': np.nan, 'aic_ctrl': np.nan, 'pseudo_r2': 0.0
            })
            omnibus_pvals.append(1.0)

        if k % 10 == 0:
            log(f"  Completed {k}/{N_NUMBERS} logistic regressions")

    # FDR correction on omnibus tests
    fdr_reject_lr, fdr_pvals_lr = fdr_correct(omnibus_pvals, method='fdr_bh')
    raw_sig_lr = sum(1 for p in omnibus_pvals if p < ALPHA)
    fdr_sig_lr = fdr_reject_lr.sum()

    log(f"Omnibus LR tests: {N_NUMBERS}")
    log(f"Raw significant (p<0.05): {raw_sig_lr}")
    log(f"FDR-surviving: {fdr_sig_lr}")
    log(f"Expected by chance: ~{N_NUMBERS * 0.05:.1f}")

    # L1-regularized logistic regression for variable selection
    log("Fitting L1-regularized logistic regression for variable selection...")
    from sklearn.linear_model import LogisticRegression as SkLogit
    feature_names = weather_features + control_cols

    for k in range(1, N_NUMBERS + 1):
        y = df[f'number_present_{k}'].values
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue
        try:
            l1_model = SkLogit(penalty='l1', solver='saga', C=1.0,
                               max_iter=2000, random_state=42)
            l1_model.fit(X_full, y)
            nonzero = np.where(np.abs(l1_model.coef_[0]) > 1e-4)[0]
            if len(nonzero) > 0:
                selected = [feature_names[idx] for idx in nonzero if idx < len(feature_names)]
                if selected:
                    l1_selected_features[k] = selected
        except Exception:
            pass

    # Count which weather features are most often selected
    feature_selection_counts = {}
    for k, feats in l1_selected_features.items():
        for f in feats:
            feature_selection_counts[f] = feature_selection_counts.get(f, 0) + 1

    if feature_selection_counts:
        log("L1 feature selection frequency (across 54 number models):")
        for f, cnt in sorted(feature_selection_counts.items(), key=lambda x: -x[1])[:10]:
            log(f"  {f}: selected in {cnt} models")

    # Top significant numbers
    sig_numbers = [(r['number'], r['lr_pval']) for r in lr_results if r['lr_pval'] < 0.05]
    if sig_numbers:
        sig_numbers.sort(key=lambda x: x[1])
        log(f"Numbers with significant omnibus LR test (raw p<0.05):")
        for num, p in sig_numbers[:10]:
            log(f"  Number {num}: p={p:.4e}")

    results = {
        'n_tests': N_NUMBERS,
        'raw_significant': int(raw_sig_lr),
        'fdr_significant': int(fdr_sig_lr),
        'expected_by_chance': N_NUMBERS * 0.05,
        'top_significant': [
            {'number': r['number'], 'lr_pval': r['lr_pval'], 'pseudo_r2': r['pseudo_r2']}
            for r in sorted(lr_results, key=lambda x: x['lr_pval'])[:10]
        ],
        'l1_feature_selection_counts': feature_selection_counts,
    }

    return results


# ============================================================================
# PHASE 7: OLS Regression on Aggregates
# ============================================================================
def phase7_ols_regression(df, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 7: OLS Regression on Aggregates")
    log("=" * 70)

    weather_cols = []
    for f in weather_features:
        if f'{f}_detrended' in df.columns:
            weather_cols.append(f'{f}_detrended')
        else:
            weather_cols.append(f)

    control_cols = ['sin_doy', 'cos_doy', 'post_aug2021']
    targets = ['draw_sum', 'draw_range', 'odd_ratio', 'high_ratio']

    scaler = StandardScaler()
    X_weather_scaled = scaler.fit_transform(df[weather_cols].values)
    X_controls = df[control_cols].values
    X_full = np.column_stack([X_weather_scaled, X_controls])
    X_full_sm = sm.add_constant(X_full)

    feature_names = ['const'] + [f'w_{wf}' for wf in weather_features] + control_cols

    ols_results = {}

    for target in targets:
        target_col = f'{target}_detrended' if f'{target}_detrended' in df.columns else target
        y = df[target_col].values

        try:
            # OLS with HAC standard errors (Newey-West)
            model = sm.OLS(y, X_full_sm)
            # Use HAC covariance (robust to heteroskedasticity and autocorrelation)
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})

            log(f"\n  OLS: {target}")
            log(f"    R-squared: {res.rsquared:.6f}")
            log(f"    Adj R-squared: {res.rsquared_adj:.6f}")
            log(f"    F-statistic: {res.fvalue:.4f}, p={res.f_pvalue:.4e}")

            # Significant predictors
            sig_idx = np.where(res.pvalues < 0.05)[0]
            sig_vars = []
            for idx in sig_idx:
                if idx < len(feature_names):
                    name = feature_names[idx]
                    sig_vars.append({
                        'name': name,
                        'coef': float(res.params[idx]),
                        'pval': float(res.pvalues[idx]),
                        'tstat': float(res.tvalues[idx])
                    })

            if sig_vars:
                log(f"    Significant predictors (HAC p<0.05):")
                for sv in sig_vars:
                    log(f"      {sv['name']}: coef={sv['coef']:.4f}, t={sv['tstat']:.2f}, p={sv['pval']:.4e}")

            ols_results[target] = {
                'r_squared': float(res.rsquared),
                'adj_r_squared': float(res.rsquared_adj),
                'f_statistic': float(res.fvalue),
                'f_pvalue': float(res.f_pvalue),
                'significant_predictors': sig_vars,
                'n_obs': int(res.nobs),
            }

        except Exception as e:
            log(f"  OLS for {target} failed: {e}", level='warning')
            ols_results[target] = {'error': str(e)}

    return ols_results


# ============================================================================
# PHASE 8: Granger Causality
# ============================================================================
def phase8_granger_causality(df, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 8: Granger Causality & Cross-Correlation")
    log("=" * 70)

    # Work with draw-by-draw time series (already sorted by date)
    weather_cols = []
    for f in weather_features:
        if f'{f}_detrended' in df.columns:
            weather_cols.append(f'{f}_detrended')
        else:
            weather_cols.append(f)

    targets = ['draw_sum', 'draw_range', 'odd_ratio', 'high_ratio']
    granger_results = {}
    xcorr_results = {}

    for target in targets:
        target_col = f'{target}_detrended' if f'{target}_detrended' in df.columns else target
        y = df[target_col].values

        granger_results[target] = {}
        xcorr_results[target] = {}

        for i, wf in enumerate(weather_features):
            wf_col = weather_cols[i]
            x = df[wf_col].values

            # --- Granger causality ---
            try:
                data_2col = np.column_stack([y, x])
                # Remove any rows with NaN
                mask = np.all(np.isfinite(data_2col), axis=1)
                data_2col = data_2col[mask]

                if len(data_2col) > GRANGER_MAX_LAG + 10:
                    gc_test = grangercausalitytests(data_2col,
                                                     maxlag=GRANGER_MAX_LAG,
                                                     verbose=False)
                    # Extract min p-value across lags (F-test)
                    min_p = min(gc_test[lag][0]['ssr_ftest'][1]
                                for lag in range(1, GRANGER_MAX_LAG + 1))
                    best_lag = min(range(1, GRANGER_MAX_LAG + 1),
                                   key=lambda lag: gc_test[lag][0]['ssr_ftest'][1])
                    granger_results[target][wf] = {
                        'min_p': float(min_p),
                        'best_lag': int(best_lag),
                    }
                else:
                    granger_results[target][wf] = {'min_p': 1.0, 'best_lag': 0}
            except Exception as e:
                granger_results[target][wf] = {'min_p': 1.0, 'best_lag': 0, 'error': str(e)[:50]}

            # --- Cross-correlation at lags -5 to +5 ---
            x_std = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-10)
            y_std = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-10)
            n = len(y_std)
            ccf_vals = {}
            for lag in range(-5, 6):
                if lag >= 0:
                    ccf_vals[lag] = float(np.mean(y_std[lag:] * x_std[:n-lag])) if lag < n else 0.0
                else:
                    alag = abs(lag)
                    ccf_vals[lag] = float(np.mean(y_std[:n-alag] * x_std[alag:])) if alag < n else 0.0
            xcorr_results[target][wf] = ccf_vals

    # Summarize Granger results
    all_granger_pvals = []
    granger_entries = []
    for target in targets:
        for wf in weather_features:
            p = granger_results[target][wf]['min_p']
            all_granger_pvals.append(p)
            granger_entries.append((target, wf, p,
                                    granger_results[target][wf].get('best_lag', 0)))

    raw_sig_gc = sum(1 for p in all_granger_pvals if p < ALPHA)
    fdr_reject_gc, _ = fdr_correct(all_granger_pvals, method='fdr_bh')
    fdr_sig_gc = fdr_reject_gc.sum()
    n_gc = len(all_granger_pvals)

    log(f"Granger causality tests: {n_gc}")
    log(f"Raw significant (p<0.05): {raw_sig_gc}")
    log(f"FDR-surviving: {fdr_sig_gc}")
    log(f"Expected by chance: ~{n_gc * 0.05:.1f}")

    # Top Granger results
    granger_entries.sort(key=lambda x: x[2])
    log("Top 10 Granger causality results:")
    for target, wf, p, lag in granger_entries[:10]:
        log(f"  {wf:25s} -> {target:15s}: p={p:.4e}, best_lag={lag}")

    results = {
        'n_tests': n_gc,
        'raw_significant': int(raw_sig_gc),
        'fdr_significant': int(fdr_sig_gc),
        'expected_by_chance': n_gc * 0.05,
        'top10': [
            {'weather': wf, 'target': target, 'p': float(p), 'best_lag': int(lag)}
            for target, wf, p, lag in granger_entries[:10]
        ],
        'granger_details': safe_json(granger_results),
    }

    return results, xcorr_results


# ============================================================================
# PHASE 9: Random Forest Predictability
# ============================================================================
def phase9_random_forest(df, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 9: Random Forest Predictability")
    log("=" * 70)

    weather_cols = []
    for f in weather_features:
        if f'{f}_detrended' in df.columns:
            weather_cols.append(f'{f}_detrended')
        else:
            weather_cols.append(f)

    control_cols = ['sin_doy', 'cos_doy', 'post_aug2021']
    X = df[weather_cols + control_cols].values
    X = np.nan_to_num(X, nan=0.0)

    # --- RF Regression on draw_sum ---
    log("RF Regression: draw_sum")
    target_col = 'draw_sum_detrended' if 'draw_sum_detrended' in df.columns else 'draw_sum'
    y_sum = df[target_col].values

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=8,
                                    min_samples_leaf=10, random_state=42,
                                    n_jobs=-1)
    cv_scores = cross_val_score(rf_reg, X, y_sum, cv=5, scoring='r2')
    mean_r2 = cv_scores.mean()
    std_r2 = cv_scores.std()
    log(f"  5-fold CV R-squared: {mean_r2:.4f} +/- {std_r2:.4f}")

    # Feature importances from full fit
    rf_reg.fit(X, y_sum)
    feat_names = weather_features + control_cols
    importances_reg = dict(zip(feat_names, rf_reg.feature_importances_.tolist()))
    log("  Feature importances (top 5):")
    for f, imp in sorted(importances_reg.items(), key=lambda x: -x[1])[:5]:
        log(f"    {f}: {imp:.4f}")

    # --- RF Classification per number ---
    log("RF Classification: per-number AUC")
    auc_scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in range(1, N_NUMBERS + 1):
        y = df[f'number_present_{k}'].values
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            auc_scores[k] = 0.5
            continue

        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                         min_samples_leaf=10, random_state=42,
                                         n_jobs=-1)
        aucs = []
        try:
            for train_idx, test_idx in skf.split(X, y):
                rf_clf.fit(X[train_idx], y[train_idx])
                y_prob = rf_clf.predict_proba(X[test_idx])[:, 1]
                auc = roc_auc_score(y[test_idx], y_prob)
                aucs.append(auc)
            auc_scores[k] = float(np.mean(aucs))
        except Exception:
            auc_scores[k] = 0.5

    auc_vals = list(auc_scores.values())
    log(f"  AUC distribution across 54 numbers:")
    log(f"    Mean: {np.mean(auc_vals):.4f}")
    log(f"    Std:  {np.std(auc_vals):.4f}")
    log(f"    Min:  {np.min(auc_vals):.4f}")
    log(f"    Max:  {np.max(auc_vals):.4f}")
    log(f"    Null expected: ~0.5000")

    # How many above 0.55?
    n_above_55 = sum(1 for a in auc_vals if a > 0.55)
    n_above_53 = sum(1 for a in auc_vals if a > 0.53)
    log(f"    Numbers with AUC > 0.55: {n_above_55}")
    log(f"    Numbers with AUC > 0.53: {n_above_53}")

    results = {
        'rf_regression_draw_sum': {
            'cv_r2_mean': float(mean_r2),
            'cv_r2_std': float(std_r2),
            'feature_importances': importances_reg,
        },
        'rf_classification_per_number': {
            'auc_mean': float(np.mean(auc_vals)),
            'auc_std': float(np.std(auc_vals)),
            'auc_min': float(np.min(auc_vals)),
            'auc_max': float(np.max(auc_vals)),
            'n_above_055': int(n_above_55),
            'n_above_053': int(n_above_53),
            'all_aucs': {int(k): float(v) for k, v in auc_scores.items()},
        },
    }

    return results, auc_scores


# ============================================================================
# PHASE 10: Effect Size
# ============================================================================
def phase10_effect_size(df, weather_features):
    log("")
    log("=" * 70)
    log("PHASE 10: Effect Size Analysis")
    log("=" * 70)

    aggregate_lottery = ['draw_sum', 'draw_mean', 'draw_std', 'draw_range',
                         'odd_ratio', 'high_ratio']

    # --- Cohen's d for binary weather splits ---
    splits = {
        'hot_vs_cold': ('temp', df['temp'].median()),
        'high_vs_low_pressure': ('pressure', df['pressure'].median()),
        'rainy_vs_dry': ('is_rainy', 0.5),
        'humid_vs_dry': ('humidity', df['humidity'].median()),
        'windy_vs_calm': ('wind_speed', df['wind_speed'].median()),
    }

    cohens_d_results = {}
    for split_name, (feat, threshold) in splits.items():
        group_high = df[df[feat] > threshold]
        group_low = df[df[feat] <= threshold]

        cohens_d_results[split_name] = {
            'n_high': len(group_high),
            'n_low': len(group_low),
            'features': {}
        }

        for lf in aggregate_lottery:
            lf_col = f'{lf}_detrended' if f'{lf}_detrended' in df.columns else lf
            x1 = group_high[lf_col].values
            x2 = group_low[lf_col].values

            # Cohen's d
            n1, n2 = len(x1), len(x2)
            if n1 < 2 or n2 < 2:
                continue
            s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            if pooled_std < 1e-10:
                d = 0.0
            else:
                d = (x1.mean() - x2.mean()) / pooled_std

            cohens_d_results[split_name]['features'][lf] = {
                'cohens_d': float(d),
                'mean_high': float(x1.mean()),
                'mean_low': float(x2.mean()),
                'diff': float(x1.mean() - x2.mean()),
            }

    log("Cohen's d effect sizes (binary weather splits):")
    for split_name, data in cohens_d_results.items():
        log(f"\n  {split_name} (n_high={data['n_high']}, n_low={data['n_low']}):")
        for lf, stats_d in data['features'].items():
            d = stats_d['cohens_d']
            label = 'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'
            log(f"    {lf:15s}: d={d:+.4f} ({label}), diff={stats_d['diff']:+.2f}")

    # --- Conditional frequency tables for top deviating numbers ---
    log("\nConditional frequency analysis for individual numbers:")

    # Focus on pressure split (most physically plausible)
    high_p = df[df['is_high_pressure'] == 1]
    low_p = df[df['is_high_pressure'] == 0]

    freq_deviations = []
    for k in range(1, N_NUMBERS + 1):
        freq_high = high_p[f'number_present_{k}'].mean()
        freq_low = low_p[f'number_present_{k}'].mean()
        expected = N_DRAWN / N_NUMBERS  # ~0.1111
        deviation = freq_high - freq_low
        freq_deviations.append({
            'number': k,
            'freq_high_pressure': float(freq_high),
            'freq_low_pressure': float(freq_low),
            'deviation': float(deviation),
            'abs_deviation': float(abs(deviation)),
        })

    freq_deviations.sort(key=lambda x: -x['abs_deviation'])
    log("Top 10 numbers by frequency deviation (high vs low pressure):")
    for entry in freq_deviations[:10]:
        log(f"  Number {entry['number']:2d}: high_p={entry['freq_high_pressure']:.4f}, "
            f"low_p={entry['freq_low_pressure']:.4f}, "
            f"diff={entry['deviation']:+.4f}")

    # --- "Extra correct balls per draw" calculation ---
    # If weather COULD predict, how many extra balls could you expect?
    # Maximum theoretical benefit from the strongest correlation
    log("\n'Extra correct balls per draw' calculation:")
    # Base rate per number: 6/54 = 0.1111
    base_rate = N_DRAWN / N_NUMBERS

    # Find strongest conditional frequency deviation across all weather splits
    max_extra = 0.0
    for split_name, (feat, threshold) in splits.items():
        group_high = df[df[feat] > threshold]
        group_low = df[df[feat] <= threshold]
        for k in range(1, N_NUMBERS + 1):
            f_h = group_high[f'number_present_{k}'].mean()
            f_l = group_low[f'number_present_{k}'].mean()
            # If you could pick the right group, your extra rate per number = |f_h - f_l| / 2
            extra = abs(f_h - f_l) / 2
            if extra > max_extra:
                max_extra = extra

    # Extra correct balls per draw = sum over 54 numbers of extra rate
    # But realistically you only pick 6, so use top-6 deviations
    extra_per_draw = max_extra * N_DRAWN  # Upper bound from single best split
    log(f"  Base rate per number: {base_rate:.4f}")
    log(f"  Strongest single-number conditional deviation: {max_extra:.4f}")
    log(f"  Upper-bound extra correct balls per draw (all 6 picks): {extra_per_draw:.4f}")
    log(f"  This is {extra_per_draw/N_DRAWN*100:.2f}% improvement over random")

    results = {
        'cohens_d': safe_json(cohens_d_results),
        'top_frequency_deviations_pressure': freq_deviations[:10],
        'extra_correct_balls': {
            'base_rate_per_number': float(base_rate),
            'max_conditional_deviation': float(max_extra),
            'upper_bound_extra_per_draw': float(extra_per_draw),
            'pct_improvement': float(extra_per_draw / N_DRAWN * 100),
        },
    }

    return results


# ============================================================================
# PHASE 11: Output - Plots & JSON
# ============================================================================
def phase11_plots(df, weather_features, aggregate_lottery,
                  pearson_matrix, pearson_pvals,
                  pb_matrix, pb_pvals,
                  mi_agg, mi_nums,
                  auc_scores, detrend_diagnostics):
    log("")
    log("=" * 70)
    log("PHASE 11: Generating Plots")
    log("=" * 70)

    sns.set_style('whitegrid')

    # --- Plot 1: Pearson correlation heatmap ---
    log("Generating weather_corr_heatmap.png...")
    fig, ax = plt.subplots(figsize=(14, 8))
    mask_sig = pearson_pvals >= 0.05
    im = ax.imshow(pearson_matrix, cmap='RdBu_r', vmin=-0.15, vmax=0.15, aspect='auto')
    ax.set_xticks(range(len(aggregate_lottery)))
    ax.set_xticklabels(aggregate_lottery, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(weather_features)))
    ax.set_yticklabels(weather_features, fontsize=9)
    ax.set_title('Pearson Correlation: Weather vs Lottery Aggregates\n(X = not significant at p<0.05)',
                 fontsize=12)
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)

    # Mark non-significant cells
    for i in range(pearson_matrix.shape[0]):
        for j in range(pearson_matrix.shape[1]):
            if mask_sig[i, j]:
                ax.text(j, i, 'X', ha='center', va='center', fontsize=8, color='gray')
            else:
                ax.text(j, i, f'{pearson_matrix[i,j]:.3f}', ha='center', va='center',
                        fontsize=7, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'weather_corr_heatmap.png'), dpi=150)
    plt.close(fig)

    # --- Plot 2: Point-biserial heatmap (54 numbers x 12 weather vars) ---
    log("Generating weather_pointbiserial_heatmap.png...")
    fig, ax = plt.subplots(figsize=(14, 18))
    im = ax.imshow(pb_matrix, cmap='RdBu_r', vmin=-0.1, vmax=0.1, aspect='auto')
    ax.set_xticks(range(len(weather_features)))
    ax.set_xticklabels(weather_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(N_NUMBERS))
    ax.set_yticklabels([str(k) for k in range(1, N_NUMBERS + 1)], fontsize=7)
    ax.set_ylabel('Lottery Number')
    ax.set_title('Point-Biserial Correlation: Weather vs Individual Number Presence',
                 fontsize=12)
    plt.colorbar(im, ax=ax, label='Point-biserial r', shrink=0.6)

    # Mark significant cells with a dot
    for i in range(pb_matrix.shape[0]):
        for j in range(pb_matrix.shape[1]):
            if pb_pvals[i, j] < 0.05:
                ax.plot(j, i, 'k.', markersize=3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'weather_pointbiserial_heatmap.png'), dpi=150)
    plt.close(fig)

    # --- Plot 3: MI heatmap ---
    log("Generating weather_mi_heatmap.png...")
    mi_combined = np.vstack([mi_agg, mi_nums])
    row_labels = list(aggregate_lottery) + [f'num_{k}' for k in range(1, N_NUMBERS + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [1, 1]})

    # Aggregate MI
    ax = axes[0]
    im0 = ax.imshow(mi_agg, cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xticks(range(len(weather_features)))
    ax.set_xticklabels(weather_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(aggregate_lottery)))
    ax.set_yticklabels(aggregate_lottery, fontsize=9)
    ax.set_title('MI: Weather vs Aggregates', fontsize=11)
    plt.colorbar(im0, ax=ax, label='MI (nats)', shrink=0.6)

    for i in range(mi_agg.shape[0]):
        for j in range(mi_agg.shape[1]):
            ax.text(j, i, f'{mi_agg[i,j]:.3f}', ha='center', va='center', fontsize=7)

    # Number MI
    ax = axes[1]
    im1 = ax.imshow(mi_nums, cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xticks(range(len(weather_features)))
    ax.set_xticklabels(weather_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(0, N_NUMBERS, 5))
    ax.set_yticklabels([str(k) for k in range(1, N_NUMBERS + 1, 5)], fontsize=8)
    ax.set_ylabel('Lottery Number')
    ax.set_title('MI: Weather vs Number Presence', fontsize=11)
    plt.colorbar(im1, ax=ax, label='MI (nats)', shrink=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'weather_mi_heatmap.png'), dpi=150)
    plt.close(fig)

    # --- Plot 4: Scatter plots (top 4 weather vars vs draw_sum) ---
    log("Generating weather_draw_sum_scatter.png...")
    weather_cols = []
    for f in weather_features:
        if f'{f}_detrended' in df.columns:
            weather_cols.append(f'{f}_detrended')
        else:
            weather_cols.append(f)

    target_col = 'draw_sum_detrended' if 'draw_sum_detrended' in df.columns else 'draw_sum'
    y = df[target_col].values

    # Find top 4 by |Pearson r|
    corrs_abs = []
    for i, wc in enumerate(weather_cols):
        r, _ = stats.pearsonr(df[wc].values, y)
        corrs_abs.append((i, abs(r), r))
    corrs_abs.sort(key=lambda x: -x[1])
    top4 = corrs_abs[:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, (i, abs_r, r) in enumerate(top4):
        ax = axes[ax_idx // 2][ax_idx % 2]
        wc = weather_cols[i]
        wf_name = weather_features[i]
        x = df[wc].values
        ax.scatter(x, y, alpha=0.3, s=10, c='steelblue')
        # Add regression line
        slope, intercept, _, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
        ax.set_xlabel(wf_name, fontsize=10)
        ax.set_ylabel('draw_sum (detrended)', fontsize=10)
        ax.set_title(f'{wf_name} vs draw_sum (r={r:.4f})', fontsize=10)

    plt.suptitle('Top 4 Weather-Lottery Correlations', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'weather_draw_sum_scatter.png'), dpi=150)
    plt.close(fig)

    # --- Plot 5: ML AUC distribution ---
    log("Generating weather_ml_auc_distribution.png...")
    auc_vals = [auc_scores[k] for k in range(1, N_NUMBERS + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(auc_vals, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Null (AUC=0.50)')
    ax.axvline(np.mean(auc_vals), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(auc_vals):.4f}')
    ax.set_xlabel('AUC')
    ax.set_ylabel('Count')
    ax.set_title('RF Classification AUC Distribution (54 Numbers)')
    ax.legend()

    # Bar chart sorted
    ax = axes[1]
    sorted_aucs = sorted(auc_scores.items(), key=lambda x: -x[1])
    numbers = [str(k) for k, _ in sorted_aucs]
    vals = [v for _, v in sorted_aucs]
    colors = ['salmon' if v > 0.55 else 'steelblue' if v > 0.5 else 'lightblue'
              for v in vals]
    ax.bar(range(N_NUMBERS), vals, color=colors, edgecolor='gray', linewidth=0.5)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(range(0, N_NUMBERS, 5))
    ax.set_xticklabels([numbers[i] for i in range(0, N_NUMBERS, 5)], fontsize=7)
    ax.set_xlabel('Number (sorted by AUC)')
    ax.set_ylabel('AUC')
    ax.set_title('Per-Number AUC (sorted)')

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'weather_ml_auc_distribution.png'), dpi=150)
    plt.close(fig)

    # --- Plot 6: Seasonal detrending diagnostic ---
    log("Generating weather_seasonal_detrending.png...")
    if 'temp_original' in detrend_diagnostics:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        dates = detrend_diagnostics['dates']
        n_pts = len(dates)

        ax = axes[0]
        ax.plot(range(n_pts), detrend_diagnostics['temp_original'], 'b-', alpha=0.5, linewidth=0.5)
        ax.set_ylabel('Temperature (F)')
        ax.set_title('Original Temperature on Draw Dates')

        ax = axes[1]
        ax.plot(range(n_pts), detrend_diagnostics['temp_seasonal'], 'orange', linewidth=1)
        ax.set_ylabel('Seasonal Mean (F)')
        ax.set_title('Day-of-Year Smoothed Seasonal Mean')

        ax = axes[2]
        ax.plot(range(n_pts), detrend_diagnostics['temp_detrended'], 'g-', alpha=0.5, linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_ylabel('Detrended Temp (F)')
        ax.set_title('Detrended Temperature (Original - Seasonal)')
        ax.set_xlabel('Draw Index (chronological)')

        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'weather_seasonal_detrending.png'), dpi=150)
        plt.close(fig)

    log("All plots saved to results/ directory.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    t_total_start = time.time()
    log("=" * 70)
    log("WEATHER-LOTTERY CORRELATION ANALYSIS")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # Collect all findings for JSON output
    findings = {}

    # ---- PHASE 0 ----
    t0 = time.time()
    df, df_weather_full = phase0_load_and_join()
    findings['phase0'] = {
        'n_matched_draws': len(df),
        'date_range': [str(df['date'].min()), str(df['date'].max())],
        'elapsed_seconds': time.time() - t0,
    }

    # ---- PHASE 1 ----
    t0 = time.time()
    df = phase1_lottery_features(df)
    findings['phase1'] = {
        'draw_sum_stats': {
            'mean': float(df['draw_sum'].mean()),
            'std': float(df['draw_sum'].std()),
            'min': int(df['draw_sum'].min()),
            'max': int(df['draw_sum'].max()),
        },
        'elapsed_seconds': time.time() - t0,
    }

    # ---- PHASE 2 ----
    t0 = time.time()
    df, weather_features = phase2_weather_features(df)
    findings['phase2'] = {
        'weather_features': weather_features,
        'n_weather_features': len(weather_features),
        'elapsed_seconds': time.time() - t0,
    }

    # ---- PHASE 3 ----
    t0 = time.time()
    df, detrend_diagnostics, trend_results = phase3_seasonal_detrending(
        df, df_weather_full, weather_features)
    findings['phase3'] = {
        'trend_tests': safe_json(trend_results),
        'n_post_aug2021': int(df['post_aug2021'].sum()),
        'elapsed_seconds': time.time() - t0,
    }

    # ---- PHASE 4 ----
    t0 = time.time()
    aggregate_lottery = ['draw_sum', 'draw_mean', 'draw_std', 'draw_range',
                         'odd_ratio', 'high_ratio', 'consecutive_pairs',
                         'odd_count', 'high_count']
    (corr_results, pearson_matrix, pearson_pvals, pb_matrix, pb_pvals,
     spearman_matrix, spearman_pvals,
     wf_used, agg_used) = phase4_correlation(df, weather_features)
    corr_results['elapsed_seconds'] = time.time() - t0
    findings['phase4_correlation'] = corr_results

    # ---- PHASE 5 ----
    t0 = time.time()
    mi_results, mi_agg, mi_nums, mi_agg_pvals, mi_nums_pvals, mi_agg_lottery = \
        phase5_mutual_information(df, weather_features)
    mi_results['elapsed_seconds'] = time.time() - t0
    findings['phase5_mutual_information'] = mi_results

    # ---- PHASE 6 ----
    t0 = time.time()
    lr_results = phase6_logistic_regression(df, weather_features)
    lr_results['elapsed_seconds'] = time.time() - t0
    findings['phase6_logistic_regression'] = lr_results

    # ---- PHASE 7 ----
    t0 = time.time()
    ols_results = phase7_ols_regression(df, weather_features)
    findings['phase7_ols_regression'] = safe_json(ols_results)
    findings['phase7_ols_regression']['elapsed_seconds'] = time.time() - t0

    # ---- PHASE 8 ----
    t0 = time.time()
    gc_results, xcorr_results = phase8_granger_causality(df, weather_features)
    gc_results['elapsed_seconds'] = time.time() - t0
    findings['phase8_granger_causality'] = gc_results

    # ---- PHASE 9 ----
    t0 = time.time()
    rf_results, auc_scores = phase9_random_forest(df, weather_features)
    rf_results['elapsed_seconds'] = time.time() - t0
    findings['phase9_random_forest'] = rf_results

    # ---- PHASE 10 ----
    t0 = time.time()
    es_results = phase10_effect_size(df, weather_features)
    es_results['elapsed_seconds'] = time.time() - t0
    findings['phase10_effect_size'] = es_results

    # ---- PHASE 11: Plots ----
    t0 = time.time()
    phase11_plots(df, weather_features, aggregate_lottery,
                  pearson_matrix, pearson_pvals,
                  pb_matrix, pb_pvals,
                  mi_agg, mi_nums,
                  auc_scores, detrend_diagnostics)
    findings['phase11_plots'] = {
        'files_generated': [
            'weather_corr_heatmap.png',
            'weather_pointbiserial_heatmap.png',
            'weather_mi_heatmap.png',
            'weather_draw_sum_scatter.png',
            'weather_ml_auc_distribution.png',
            'weather_seasonal_detrending.png',
        ],
        'elapsed_seconds': time.time() - t0,
    }

    # ---- Final Summary ----
    total_time = time.time() - t_total_start
    findings['summary'] = {
        'total_elapsed_seconds': total_time,
        'n_draws_analyzed': len(df),
        'conclusion': _generate_conclusion(findings),
    }

    # Write findings JSON
    with open(FINDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(safe_json(findings), f, indent=2, default=str)

    log("")
    log("=" * 70)
    log("ANALYSIS COMPLETE")
    log(f"Total time: {total_time:.1f} seconds")
    log(f"Results written to: {FINDINGS_FILE}")
    log(f"Plots saved to: {RESULTS_DIR}/")
    log("=" * 70)

    # Print conclusion
    log("")
    log("CONCLUSION:")
    log(findings['summary']['conclusion'])


def _generate_conclusion(findings):
    """Generate a human-readable conclusion from the findings."""
    lines = []

    # Correlation summary
    p4 = findings.get('phase4_correlation', {})
    lines.append(f"Correlation Analysis: {p4.get('raw_significant', '?')} of "
                 f"{p4.get('n_tests_total', '?')} tests raw significant (p<0.05), "
                 f"expected ~{p4.get('expected_by_chance', '?'):.0f} by chance. "
                 f"FDR-surviving: {p4.get('fdr_significant', '?')}, "
                 f"Bonferroni-surviving: {p4.get('bonferroni_significant', '?')}.")

    # MI summary
    p5 = findings.get('phase5_mutual_information', {})
    lines.append(f"Mutual Information: {p5.get('raw_significant', '?')} raw significant, "
                 f"{p5.get('fdr_significant', '?')} FDR-surviving.")

    # Logistic regression
    p6 = findings.get('phase6_logistic_regression', {})
    lines.append(f"Logistic Regression (per number): {p6.get('raw_significant', '?')} of 54 "
                 f"omnibus tests significant (expected ~{p6.get('expected_by_chance', '?'):.1f}).")

    # RF
    p9 = findings.get('phase9_random_forest', {})
    rf_reg = p9.get('rf_regression_draw_sum', {})
    rf_clf = p9.get('rf_classification_per_number', {})
    lines.append(f"Random Forest: draw_sum CV R2={rf_reg.get('cv_r2_mean', '?'):.4f}, "
                 f"per-number AUC mean={rf_clf.get('auc_mean', '?'):.4f} "
                 f"(null=0.5000).")

    # Effect size
    p10 = findings.get('phase10_effect_size', {})
    extra = p10.get('extra_correct_balls', {})
    lines.append(f"Effect Size: Upper-bound extra correct balls per draw = "
                 f"{extra.get('upper_bound_extra_per_draw', '?'):.4f} "
                 f"({extra.get('pct_improvement', '?'):.2f}% improvement).")

    # Overall verdict
    bonf_sig = p4.get('bonferroni_significant', 0)
    fdr_sig = p4.get('fdr_significant', 0)
    rf_r2 = rf_reg.get('cv_r2_mean', 0)

    if bonf_sig == 0 and fdr_sig <= 5 and rf_r2 < 0.01:
        verdict = ("VERDICT: No meaningful weather-lottery correlation detected. "
                   "Results are consistent with the lottery being independent of "
                   "Austin weather conditions. Any raw significant results are "
                   "consistent with multiple-testing false positives.")
    elif fdr_sig > 5 or rf_r2 > 0.02:
        verdict = ("VERDICT: Some weak statistical signals detected, but effect sizes "
                   "are negligible. Any patterns found are too small to be practically "
                   "useful for prediction.")
    else:
        verdict = ("VERDICT: Marginal signals at best. The lottery appears to operate "
                   "independently of weather conditions within detectable limits.")

    lines.append(verdict)
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
