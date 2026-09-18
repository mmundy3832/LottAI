"""
Change Point Detection Analysis for Lotto Texas Data
Uses ruptures library with PELT algorithm to detect regime changes in lottery draws
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from scipy.stats import chisquare, entropy
from collections import Counter
import ruptures as rpt
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def load_lottery_data(csv_path: str) -> pd.DataFrame:
    """Load and filter lottery data to post-2006 draws only"""
    print("Loading lottery data...")

    # Load CSV (no header in file)
    # Format: "Lotto Texas", Month, Day, Year, N1, N2, N3, N4, N5, N6
    df = pd.read_csv(csv_path, header=None)

    # Set column names
    df.columns = ['Game', 'Month', 'Day', 'Year', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']

    # Create date column from Month/Day/Year
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Drop the separate date columns
    df = df.drop(columns=['Game', 'Month', 'Day', 'Year'])

    # Filter to post-2006 data (April 1, 2006 or later)
    df = df[df['Date'] >= '2006-04-01'].copy()
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} draws from {df['Date'].min()} to {df['Date'].max()}")
    return df


def compute_chi_squared(numbers: List[int], n_bins: int = 54) -> Tuple[float, float]:
    """
    Compute chi-squared statistic vs uniform distribution
    Returns: (chi2_stat, p_value)
    """
    # Count frequencies
    counts = Counter(numbers)
    observed = np.array([counts.get(i, 0) for i in range(1, n_bins + 1)])

    # Expected uniform frequency
    expected = np.full(n_bins, len(numbers) / n_bins)

    # Chi-squared test
    chi2_stat, p_value = chisquare(observed, expected)

    return chi2_stat, p_value


def compute_shannon_entropy(numbers: List[int], n_bins: int = 54) -> float:
    """Compute Shannon entropy of number distribution"""
    counts = Counter(numbers)
    probs = np.array([counts.get(i, 0) for i in range(1, n_bins + 1)])
    probs = probs / probs.sum()

    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]

    return entropy(probs, base=2)


def compute_frequency_variance(numbers: List[int], n_bins: int = 54) -> float:
    """Compute variance of frequency distribution"""
    counts = Counter(numbers)
    freqs = np.array([counts.get(i, 0) for i in range(1, n_bins + 1)])
    return np.var(freqs)


def create_rolling_features(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Create rolling window features for change point detection
    Returns DataFrame with chi2, entropy, and variance features
    """
    print(f"Computing rolling features (window={window})...")

    features = []

    for i in range(len(df)):
        # Get window of draws (current and previous window-1)
        start_idx = max(0, i - window + 1)
        window_draws = df.iloc[start_idx:i+1]

        # Collect all numbers in window
        all_numbers = []
        for _, row in window_draws.iterrows():
            all_numbers.extend([row['N1'], row['N2'], row['N3'],
                               row['N4'], row['N5'], row['N6']])

        # Compute features
        chi2_stat, _ = compute_chi_squared(all_numbers)
        entr = compute_shannon_entropy(all_numbers)
        var = compute_frequency_variance(all_numbers)

        features.append({
            'chi2': chi2_stat,
            'entropy': entr,
            'variance': var
        })

    feature_df = pd.DataFrame(features)
    print(f"Created {len(feature_df)} feature vectors")

    return feature_df


def custom_cost_chi2(segment_data: np.ndarray) -> float:
    """
    Custom cost function based on chi-squared divergence
    Measures how much a segment deviates from uniform distribution
    """
    # Flatten all numbers in segment
    all_numbers = segment_data.flatten().astype(int)

    # Remove any invalid values
    all_numbers = all_numbers[(all_numbers >= 1) & (all_numbers <= 54)]

    if len(all_numbers) == 0:
        return 0.0

    # Compute chi-squared statistic
    chi2_stat, _ = compute_chi_squared(all_numbers.tolist())

    # Return negative chi2 (ruptures minimizes cost)
    return -chi2_stat


def detect_changepoints_pelt(df: pd.DataFrame, feature_df: pd.DataFrame,
                              max_changepoints: int = 10) -> Dict[str, Any]:
    """
    Detect change points using PELT algorithm with BIC model selection
    """
    print("\nRunning PELT change point detection...")

    # Prepare signal matrix (use all three features)
    signal = feature_df[['chi2', 'entropy', 'variance']].values

    # Normalize features to same scale
    signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)

    # Use Binary Segmentation (faster than PELT for model selection)
    # Test different numbers of change points using BIC
    bic_scores = []
    all_changepoints = []

    print("Testing models with 0-10 change points...")

    for n_bkps in range(0, max_changepoints + 1):
        if n_bkps == 0:
            # No change points - single segment
            # Compute BIC for single segment model
            n_params = signal.shape[1]  # mean of each feature
            n_samples = len(signal)

            # Log-likelihood (negative sum of squared errors)
            segment_mean = signal.mean(axis=0)
            sse = np.sum((signal - segment_mean) ** 2)
            log_likelihood = -0.5 * n_samples * np.log(sse / n_samples)

            # BIC = -2 * log_likelihood + k * log(n)
            bic = -2 * log_likelihood + n_params * np.log(n_samples)
            bic_scores.append(bic)
            all_changepoints.append([])
            print(f"  {n_bkps} change points: BIC = {bic:.2f}")
        else:
            # Use Binary Segmentation for speed (faster than PELT)
            print(f"  Testing {n_bkps} change points...", end=' ')
            algo_binseg = rpt.Binseg(model="l2", min_size=30).fit(signal)
            bkps = algo_binseg.predict(n_bkps=n_bkps)
            best_bkps = [b for b in bkps if b < len(signal)]

            all_changepoints.append(best_bkps)

            # Compute BIC for this segmentation
            n_segments = len(best_bkps) + 1
            n_params = n_segments * signal.shape[1]  # mean of each feature per segment
            n_samples = len(signal)

            # Compute log-likelihood based on segment means
            sse = 0.0
            prev_idx = 0
            for bkp in best_bkps + [len(signal)]:
                segment = signal[prev_idx:bkp]
                if len(segment) > 0:
                    segment_mean = segment.mean(axis=0)
                    sse += np.sum((segment - segment_mean) ** 2)
                prev_idx = bkp

            log_likelihood = -0.5 * n_samples * np.log(sse / n_samples + 1e-10)
            bic = -2 * log_likelihood + n_params * np.log(n_samples)
            bic_scores.append(bic)
            print(f"BIC = {bic:.2f}")

    # Find optimal number of change points (minimum BIC)
    optimal_n = np.argmin(bic_scores)
    optimal_changepoints = all_changepoints[optimal_n]

    print(f"\nOptimal number of change points: {optimal_n} (BIC = {bic_scores[optimal_n]:.2f})")
    print(f"Change point indices: {optimal_changepoints}")

    # Convert indices to dates
    changepoint_dates = [df.iloc[idx]['Date'].strftime('%Y-%m-%d')
                         for idx in optimal_changepoints]

    print(f"Change point dates: {changepoint_dates}")

    return {
        'optimal_n': optimal_n,
        'bic_scores': bic_scores,
        'changepoint_indices': optimal_changepoints,
        'changepoint_dates': changepoint_dates,
        'all_changepoints': all_changepoints
    }


def bootstrap_validation(df: pd.DataFrame, feature_df: pd.DataFrame,
                         changepoints: List[int], n_iterations: int = 100) -> Dict[int, float]:
    """
    Bootstrap validation of detected change points
    Returns stability score (frequency) for each change point
    """
    print(f"\nRunning bootstrap validation ({n_iterations} iterations)...")

    signal = feature_df[['chi2', 'entropy', 'variance']].values
    signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)

    # Count how often each change point appears in bootstrap samples
    changepoint_counts = Counter()
    total_samples = 0

    # Tolerance window for matching change points (±25 draws)
    tolerance = 25

    for i in range(n_iterations):
        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{n_iterations}...")

        # Bootstrap resample
        indices = np.random.choice(len(signal), size=len(signal), replace=True)
        indices = np.sort(indices)
        signal_boot = signal[indices]

        # Detect change points in bootstrap sample
        try:
            algo = rpt.Binseg(model="l2", min_size=30).fit(signal_boot)
            bkps = algo.predict(n_bkps=len(changepoints))
            bkps_clean = [b for b in bkps if b < len(signal_boot)]

            # Map bootstrap indices back to original indices
            bkps_original = [indices[b] if b < len(indices) else len(signal) - 1
                            for b in bkps_clean]

            # Match to original change points
            for cp in changepoints:
                for bp in bkps_original:
                    if abs(bp - cp) <= tolerance:
                        changepoint_counts[cp] += 1
                        break

            total_samples += 1
        except:
            continue

    # Compute stability scores
    stability = {cp: changepoint_counts[cp] / total_samples
                 for cp in changepoints}

    print("\nBootstrap stability scores:")
    for cp, score in stability.items():
        print(f"  Index {cp}: {score:.2f}")

    return stability


def analyze_segments(df: pd.DataFrame, changepoints: List[int]) -> List[Dict[str, Any]]:
    """
    Analyze each segment between change points
    """
    print("\nAnalyzing segments...")

    segments = []
    breakpoints = [0] + changepoints + [len(df)]

    for i in range(len(breakpoints) - 1):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]

        segment_df = df.iloc[start_idx:end_idx]

        # Collect all numbers in segment
        all_numbers = []
        for _, row in segment_df.iterrows():
            all_numbers.extend([row['N1'], row['N2'], row['N3'],
                               row['N4'], row['N5'], row['N6']])

        # Compute statistics
        chi2_stat, p_value = compute_chi_squared(all_numbers)
        entr = compute_shannon_entropy(all_numbers)

        # Find hot and cold numbers
        counts = Counter(all_numbers)
        hot_numbers = [num for num, _ in counts.most_common(5)]
        cold_numbers = [num for num, _ in counts.most_common()[-5:]]

        segment_info = {
            'start': int(start_idx),
            'end': int(end_idx),
            'start_date': segment_df.iloc[0]['Date'].strftime('%Y-%m-%d'),
            'end_date': segment_df.iloc[-1]['Date'].strftime('%Y-%m-%d'),
            'n_draws': len(segment_df),
            'chi2': float(chi2_stat),
            'p_value': float(p_value),
            'entropy': float(entr),
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers
        }

        segments.append(segment_info)

        print(f"Segment {i+1}: {segment_info['start_date']} to {segment_info['end_date']} "
              f"({segment_info['n_draws']} draws) - Chi2: {chi2_stat:.2f}, Entropy: {entr:.3f}")

    return segments


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file"""
    print(f"\nSaving results to {output_path}...")

    # Create results directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    results_native = convert_to_native(results)

    with open(output_path, 'w') as f:
        json.dump(results_native, f, indent=2)

    print("Results saved successfully!")


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("Change Point Detection Analysis - Lotto Texas")
    print("=" * 80)

    # Configuration
    csv_path = r"D:\projects\LottAI\lottotexas.csv"
    output_path = r"D:\projects\LottAI\results\changepoint_findings.json"
    window_size = 50
    max_changepoints = 10
    bootstrap_iterations = 100

    # Load data
    df = load_lottery_data(csv_path)

    # Create rolling features
    feature_df = create_rolling_features(df, window=window_size)

    # Detect change points using PELT
    cp_results = detect_changepoints_pelt(df, feature_df, max_changepoints=max_changepoints)

    # Bootstrap validation
    if len(cp_results['changepoint_indices']) > 0:
        stability = bootstrap_validation(df, feature_df,
                                        cp_results['changepoint_indices'],
                                        n_iterations=bootstrap_iterations)
        # Convert keys to strings for JSON serialization
        stability_str = {str(k): v for k, v in stability.items()}
    else:
        stability_str = {}

    # Analyze segments
    segments = analyze_segments(df, cp_results['changepoint_indices'])

    # Compile final results
    results = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_period': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d'),
            'n_draws': len(df)
        },
        'parameters': {
            'rolling_window': window_size,
            'max_changepoints_tested': max_changepoints,
            'bootstrap_iterations': bootstrap_iterations
        },
        'optimal_changepoints': cp_results['optimal_n'],
        'bic_scores': [float(x) for x in cp_results['bic_scores']],
        'changepoint_indices': cp_results['changepoint_indices'],
        'changepoint_dates': cp_results['changepoint_dates'],
        'segments': segments,
        'bootstrap_stability': stability_str
    }

    # Save results
    save_results(results, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total draws analyzed: {len(df)}")
    print(f"Optimal number of change points: {cp_results['optimal_n']}")
    print(f"Number of segments: {len(segments)}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
