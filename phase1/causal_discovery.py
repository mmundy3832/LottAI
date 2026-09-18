"""
Causal Discovery Analysis for Lotto Texas Data
===============================================

Implements Transfer Entropy and Granger Causality to detect temporal dependencies
between lottery numbers. Transfer Entropy measures the amount of information flow
from one time series to another, detecting causal relationships.

Author: LottAI Project
Date: 2026-02-25
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from itertools import combinations
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import pyinform as pi
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = r"D:\projects\LottAI\lottotexas.csv"
OUTPUT_FILE = r"D:\projects\LottAI\results\causal_discovery_findings.json"
MIN_DATE = "2006-04-01"
MAX_LAG = 10
NUM_SURROGATES = 100
FDR_THRESHOLD = 0.05
TOP_N = 20

def load_lottery_data(filepath: str, min_date: str) -> pd.DataFrame:
    """Load and filter lottery data to post-2006 draws."""
    print(f"Loading lottery data from {filepath}...")

    # Read CSV, skip header row
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[1:]  # Skip header row

    # Parse columns: Game,Month,Day,Year,N1,N2,N3,N4,N5,N6
    df.columns = ['Game', 'Month', 'Day', 'Year', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']

    # Create date column
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Filter to post-2006
    df = df[df['Date'] >= min_date].copy()

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    # Extract numbers
    number_cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
    for col in number_cols:
        df[col] = df[col].astype(int)

    print(f"Loaded {len(df)} draws from {df['Date'].min()} to {df['Date'].max()}")

    return df

def create_binary_time_series(df: pd.DataFrame, num_numbers: int = 54) -> np.ndarray:
    """
    Create binary time series for each number.
    Returns: (T, N) array where T=draws, N=54 numbers
    Each cell is 1 if number drawn, 0 otherwise
    """
    print(f"Creating binary time series for {num_numbers} numbers...")

    T = len(df)
    binary_series = np.zeros((T, num_numbers), dtype=int)

    number_cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']

    for i, row in df.iterrows():
        for col in number_cols:
            num = row[col]
            if 1 <= num <= num_numbers:
                binary_series[i, num - 1] = 1  # 0-indexed

    print(f"Created binary series: {binary_series.shape}")
    return binary_series

def compute_transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
    """
    Compute Transfer Entropy from source to target at given lag.
    TE(X→Y) measures how much information X provides about future Y
    beyond what Y's own history provides.
    """
    # PyInform expects discrete values
    # We need to create lagged versions

    # Create history and future arrays
    k = 1  # history length

    # Target history: Y_{t-k:t-1}
    # Target future: Y_t
    # Source history: X_{t-lag-k:t-lag-1}

    if len(source) <= lag + k or len(target) <= lag + k:
        return 0.0

    # Simple approach: use pyinform's transfer_entropy
    # It expects: transfer_entropy(source, target, k=history_length)
    # But we need to account for lag

    # Shift source by lag
    source_lagged = source[:-lag] if lag > 0 else source
    target_aligned = target[lag:]

    # Trim to same length
    min_len = min(len(source_lagged), len(target_aligned))
    source_lagged = source_lagged[-min_len:]
    target_aligned = target_aligned[-min_len:]

    if min_len < 10:  # Need minimum data
        return 0.0

    try:
        # PyInform's transfer entropy
        # te(source, target, k) where k is history length
        te = pi.transfer_entropy(source_lagged, target_aligned, k=1)
        return te
    except Exception as e:
        # If computation fails, return 0
        return 0.0

def permutation_test(source: np.ndarray, target: np.ndarray, lag: int,
                     te_observed: float, n_surrogates: int = 100) -> float:
    """
    Compute p-value using permutation test.
    Shuffle source series and recompute TE to create null distribution.
    """
    te_null = []

    for _ in range(n_surrogates):
        # Shuffle source to destroy temporal structure
        source_shuffled = np.random.permutation(source)
        te_surrogate = compute_transfer_entropy(source_shuffled, target, lag)
        te_null.append(te_surrogate)

    # P-value: fraction of surrogates >= observed
    p_value = np.sum(np.array(te_null) >= te_observed) / n_surrogates

    return p_value

def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Apply Benjamini-Hochberg FDR correction.
    Returns: (adjusted_p_values, threshold)
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH procedure
    adjusted_p = np.zeros(m)
    for i in range(m):
        adjusted_p[sorted_indices[i]] = min(1.0, sorted_p[i] * m / (i + 1))

    # Find threshold
    threshold = alpha
    for i in range(m):
        if sorted_p[i] > alpha * (i + 1) / m:
            threshold = sorted_p[i - 1] if i > 0 else 0
            break

    return adjusted_p, threshold

def calculate_excitation_strength(binary_series: np.ndarray, source_idx: int,
                                   target_idx: int, lag: int) -> float:
    """
    Calculate how much source number increases probability of target at given lag.
    Excitation strength = P(target|source at lag) / P(target|not source at lag)
    """
    source = binary_series[:, source_idx]
    target = binary_series[:, target_idx]

    if lag >= len(source):
        return 1.0

    # Align series with lag
    source_lagged = source[:-lag] if lag > 0 else source
    target_aligned = target[lag:]

    min_len = min(len(source_lagged), len(target_aligned))
    source_lagged = source_lagged[-min_len:]
    target_aligned = target_aligned[-min_len:]

    # P(target | source)
    source_drawn = source_lagged == 1
    if np.sum(source_drawn) == 0:
        return 1.0
    p_target_given_source = np.sum(target_aligned[source_drawn]) / np.sum(source_drawn)

    # P(target | not source)
    source_not_drawn = source_lagged == 0
    if np.sum(source_not_drawn) == 0:
        return 1.0
    p_target_given_not_source = np.sum(target_aligned[source_not_drawn]) / np.sum(source_not_drawn)

    # Avoid division by zero
    if p_target_given_not_source == 0:
        return 1.0

    excitation = p_target_given_source / p_target_given_not_source

    return excitation

def analyze_transfer_entropy(binary_series: np.ndarray, max_lag: int = 10,
                             n_surrogates: int = 100) -> Dict:
    """
    Perform comprehensive Transfer Entropy analysis.
    Tests all number pairs at all lags.
    """
    print(f"\nStarting Transfer Entropy analysis...")
    print(f"Testing {54 * 54 * max_lag} = {54 * 54 * max_lag} hypotheses")
    print(f"Using {n_surrogates} permutation surrogates per test")

    num_numbers = binary_series.shape[1]
    results = []
    test_count = 0
    total_tests = num_numbers * num_numbers * max_lag

    # Store all p-values for FDR correction
    all_p_values = []
    all_results_temp = []

    start_time = datetime.now()

    for source_num in range(1, num_numbers + 1):
        for target_num in range(1, num_numbers + 1):
            if source_num == target_num:
                continue  # Skip self-loops

            source_idx = source_num - 1
            target_idx = target_num - 1

            for lag in range(1, max_lag + 1):
                test_count += 1

                # Progress reporting
                if test_count % 500 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = test_count / elapsed if elapsed > 0 else 0
                    eta = (total_tests - test_count) / rate if rate > 0 else 0
                    print(f"Progress: {test_count}/{total_tests} tests ({100*test_count/total_tests:.1f}%) "
                          f"- Rate: {rate:.1f} tests/sec - ETA: {eta/60:.1f} min")

                # Compute Transfer Entropy
                te_value = compute_transfer_entropy(
                    binary_series[:, source_idx],
                    binary_series[:, target_idx],
                    lag
                )

                # Skip if TE is 0 (saves time on permutation test)
                if te_value == 0:
                    all_p_values.append(1.0)
                    all_results_temp.append({
                        'source': source_num,
                        'target': target_num,
                        'lag': lag,
                        'te_value': te_value,
                        'p_value': 1.0
                    })
                    continue

                # Permutation test for significance
                p_value = permutation_test(
                    binary_series[:, source_idx],
                    binary_series[:, target_idx],
                    lag,
                    te_value,
                    n_surrogates
                )

                all_p_values.append(p_value)
                all_results_temp.append({
                    'source': source_num,
                    'target': target_num,
                    'lag': lag,
                    'te_value': te_value,
                    'p_value': p_value
                })

    print(f"\nCompleted {test_count} tests in {(datetime.now() - start_time).total_seconds():.1f} seconds")

    # Apply FDR correction
    print("Applying Benjamini-Hochberg FDR correction...")
    all_p_values = np.array(all_p_values)
    adjusted_p_values, fdr_threshold = benjamini_hochberg_correction(all_p_values, FDR_THRESHOLD)

    # Add adjusted p-values and filter significant
    significant_results = []
    for i, result in enumerate(all_results_temp):
        result['fdr_p_value'] = adjusted_p_values[i]

        if result['fdr_p_value'] < FDR_THRESHOLD and result['te_value'] > 0:
            # Calculate excitation strength
            excitation = calculate_excitation_strength(
                binary_series,
                result['source'] - 1,
                result['target'] - 1,
                result['lag']
            )
            result['excitation_strength'] = float(excitation)

            # Interpretation
            pct_increase = (excitation - 1.0) * 100
            if excitation > 1.0:
                result['interpretation'] = (
                    f"Number {result['source']} increases prob of {result['target']} "
                    f"by {pct_increase:.1f}% at lag {result['lag']}"
                )
            elif excitation < 1.0:
                pct_decrease = (1.0 - excitation) * 100
                result['interpretation'] = (
                    f"Number {result['source']} decreases prob of {result['target']} "
                    f"by {pct_decrease:.1f}% at lag {result['lag']}"
                )
            else:
                result['interpretation'] = (
                    f"Number {result['source']} has no effect on {result['target']} "
                    f"at lag {result['lag']}"
                )

            # Convert numpy types to Python native types
            result['source'] = int(result['source'])
            result['target'] = int(result['target'])
            result['lag'] = int(result['lag'])
            result['te_value'] = float(result['te_value'])
            result['p_value'] = float(result['p_value'])
            result['fdr_p_value'] = float(result['fdr_p_value'])

            significant_results.append(result)

    # Sort by TE value (descending)
    significant_results.sort(key=lambda x: x['te_value'], reverse=True)

    print(f"Found {len(significant_results)} significant causal relationships (FDR < {FDR_THRESHOLD})")

    return {
        'significant_pairs': significant_results[:TOP_N],
        'total_tests': test_count,
        'fdr_threshold': FDR_THRESHOLD,
        'significant_count': len(significant_results),
        'all_significant': significant_results  # Keep all for further analysis
    }

def analyze_granger_causality(binary_series: np.ndarray, te_results: List[Dict]) -> Dict:
    """
    Validate top Transfer Entropy findings with Granger Causality.
    Only test the top significant TE pairs to save time.
    """
    print("\nValidating with Granger Causality tests...")

    granger_results = []

    # Test top 10 TE findings
    top_te = te_results[:min(10, len(te_results))]

    for result in top_te:
        source_idx = result['source'] - 1
        target_idx = result['target'] - 1
        max_lag = result['lag']

        source = binary_series[:, source_idx]
        target = binary_series[:, target_idx]

        # Granger causality needs at least some variation
        if np.sum(source) < 5 or np.sum(target) < 5:
            continue

        try:
            # Create dataframe for Granger test
            data = pd.DataFrame({
                'target': target,
                'source': source
            })

            # Test up to max_lag
            gc_result = grangercausalitytests(data[['target', 'source']],
                                              maxlag=max_lag,
                                              verbose=False)

            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, max_lag + 1):
                # Use F-test p-value
                p_value = gc_result[lag][0]['ssr_ftest'][1]
                p_values[lag] = p_value

            # Find minimum p-value
            min_lag = min(p_values.keys(), key=lambda k: p_values[k])
            min_p = p_values[min_lag]

            granger_results.append({
                'source': int(result['source']),
                'target': int(result['target']),
                'te_lag': int(result['lag']),
                'te_value': float(result['te_value']),
                'granger_best_lag': int(min_lag),
                'granger_p_value': float(min_p),
                'granger_significant': bool(min_p < 0.05),
                'agreement': bool(min_p < 0.05)  # Both TE and Granger agree
            })

        except Exception as e:
            print(f"Granger test failed for {result['source']}→{result['target']}: {e}")
            continue

    print(f"Completed {len(granger_results)} Granger causality tests")

    return {
        'validated_pairs': granger_results,
        'total_validated': len(granger_results),
        'agreement_count': sum(1 for r in granger_results if r['agreement'])
    }

def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("Causal Discovery Analysis - Lotto Texas")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")

    # Create results directory
    results_dir = Path(OUTPUT_FILE).parent
    results_dir.mkdir(exist_ok=True)

    # Load data
    df = load_lottery_data(DATA_FILE, MIN_DATE)

    # Create binary time series
    binary_series = create_binary_time_series(df)

    # Analyze Transfer Entropy
    te_results = analyze_transfer_entropy(binary_series, MAX_LAG, NUM_SURROGATES)

    # Validate with Granger Causality
    granger_results = analyze_granger_causality(
        binary_series,
        te_results['all_significant']
    )

    # Compile final results
    final_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'data_file': DATA_FILE,
            'min_date': MIN_DATE,
            'num_draws': len(df),
            'max_lag': MAX_LAG,
            'num_surrogates': NUM_SURROGATES,
            'fdr_threshold': FDR_THRESHOLD
        },
        'transfer_entropy': te_results,
        'granger_causality': granger_results
    }

    # Remove 'all_significant' before saving (too large)
    final_results['transfer_entropy'].pop('all_significant', None)

    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nKey findings:")
    print(f"- Total tests performed: {te_results['total_tests']}")
    print(f"- Significant causal relationships: {te_results['significant_count']}")
    print(f"- FDR threshold: {FDR_THRESHOLD}")

    if te_results['significant_count'] > 0:
        print(f"\nTop 5 causal relationships:")
        for i, result in enumerate(te_results['significant_pairs'][:5], 1):
            print(f"{i}. {result['interpretation']}")
            print(f"   TE={result['te_value']:.4f}, p={result['p_value']:.4f}, "
                  f"FDR-p={result['fdr_p_value']:.4f}")
    else:
        print("\nNo significant causal relationships found.")
        print("This suggests lottery numbers are drawn independently (as expected).")

    print(f"\nEnd time: {datetime.now()}")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
