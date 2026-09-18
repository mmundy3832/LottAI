#!/usr/bin/env python
"""
Matrix Profile Analysis for Lotto Texas Data
Using STUMPY library for motif and discord discovery

Encodes draws as 54-dimensional binary vectors and searches for:
- Motifs: repeated patterns (low distance pairs)
- Discords: anomalies (high distance patterns)
"""

import numpy as np
import pandas as pd
import stumpy
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
CSV_FILE = "lottotexas.csv"
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "matrix_profile_findings.json")
CUTOFF_DATE = "2006-04-01"
NUM_BALLS = 54
WINDOW_SIZES = [5, 7, 10, 15, 20]
TOP_N = 5  # Number of motifs/discords to extract per window size


def load_and_filter_data(csv_file: str, cutoff_date: str) -> pd.DataFrame:
    """Load lottery data and filter to post-2006 draws."""
    print(f"Loading data from {csv_file}...")

    # Read CSV - data format: Lotto Texas,MM,DD,YYYY,n1,n2,n3,n4,n5,n6
    df = pd.read_csv(csv_file, header=None)

    # Skip first row if it's a header
    if df.iloc[0, 0] == "Lotto Texas" and df.iloc[0, 1] == "11":
        df = df.iloc[1:].reset_index(drop=True)

    # Create date column from MM, DD, YYYY columns (indices 1, 2, 3)
    df['date'] = pd.to_datetime(
        df[3].astype(str) + '-' + df[1].astype(str) + '-' + df[2].astype(str),
        format='%Y-%m-%d'
    )

    # Extract the 6 drawn numbers (columns 4-9)
    df['numbers'] = df[[4, 5, 6, 7, 8, 9]].apply(
        lambda row: sorted([int(x) for x in row]), axis=1
    )

    # Filter to post-cutoff date
    df = df[df['date'] >= cutoff_date].reset_index(drop=True)

    print(f"Loaded {len(df)} draws from {df['date'].min()} to {df['date'].max()}")
    return df[['date', 'numbers']]


def encode_draw_as_binary(numbers: List[int], num_balls: int = NUM_BALLS) -> np.ndarray:
    """Encode a draw as a binary vector (1 if number drawn, 0 if not)."""
    vector = np.zeros(num_balls, dtype=np.float64)
    for num in numbers:
        if 1 <= num <= num_balls:
            vector[num - 1] = 1.0
    return vector


def create_time_series(df: pd.DataFrame) -> np.ndarray:
    """Create time series matrix from draws (T x 54)."""
    print("Encoding draws as 54-dimensional binary vectors...")
    time_series = np.array([
        encode_draw_as_binary(draw) for draw in df['numbers']
    ])
    print(f"Created time series: {time_series.shape} (draws x dimensions)")
    return time_series


def compute_matrix_profile(time_series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute matrix profile using STUMPY.

    Returns:
        profile: Matrix profile (distances)
        indices: Matrix profile indices (nearest neighbor indices)
    """
    print(f"  Computing matrix profile for window size {window_size}...")

    # For multivariate time series (T x d), mstump expects (d x T)
    # mstump returns a tuple: (distances, indices)
    # distances shape: (d, T-m+1) - distance for each dimension's time series
    # indices shape: (d, T-m+1) - nearest neighbor index for each dimension

    distances, mp_indices = stumpy.mstump(time_series.T, m=window_size)

    # distances has shape (d, T-m+1)
    # We want a single distance value per subsequence, so take mean across dimensions
    profile = np.mean(distances, axis=0)

    # For indices, we'll use the first dimension's indices
    # (or could use mode, but for lottery data they should be similar)
    indices = mp_indices[0, :].astype(int)

    return profile, indices


def find_motifs(profile: np.ndarray, indices: np.ndarray, top_n: int = TOP_N) -> List[Dict]:
    """
    Find top N motifs (repeated patterns with low distance).

    A motif is a pair of subsequences with minimum distance.
    """
    motifs = []
    used_indices = set()

    # Sort by distance (ascending) to find lowest distances first
    sorted_idx = np.argsort(profile)

    for idx in sorted_idx:
        if len(motifs) >= top_n:
            break

        if idx in used_indices:
            continue

        # Get the nearest neighbor index
        nn_idx = indices[idx]

        if nn_idx in used_indices or nn_idx < 0:
            continue

        # Ensure they're not trivially overlapping (within window size)
        # We'll skip this check for now as STUMPY handles exclusion zones

        motifs.append({
            'idx1': int(idx),
            'idx2': int(nn_idx),
            'distance': float(profile[idx])
        })

        used_indices.add(idx)
        used_indices.add(nn_idx)

    return motifs


def find_discords(profile: np.ndarray, top_n: int = TOP_N) -> List[Dict]:
    """
    Find top N discords (anomalies with high distance).

    A discord is a subsequence with maximum distance to its nearest neighbor.
    """
    discords = []

    # Sort by distance (descending) to find highest distances first
    sorted_idx = np.argsort(profile)[::-1]

    for idx in sorted_idx[:top_n]:
        discords.append({
            'idx': int(idx),
            'distance': float(profile[idx])
        })

    return discords


def analyze_window_size(time_series: np.ndarray, df: pd.DataFrame,
                       window_size: int) -> Dict:
    """Analyze a specific window size."""
    print(f"\nAnalyzing window size: {window_size}")

    if window_size > len(time_series):
        print(f"  Skipping: window size {window_size} exceeds time series length")
        return None

    # Compute matrix profile
    profile, indices = compute_matrix_profile(time_series, window_size)

    # Find motifs
    print(f"  Finding top {TOP_N} motifs...")
    motifs = find_motifs(profile, indices, TOP_N)

    # Add draw information to motifs
    for motif in motifs:
        idx1, idx2 = motif['idx1'], motif['idx2']
        motif['draws'] = f"{idx1}-{idx1+window_size-1} matches {idx2}-{idx2+window_size-1}"
        motif['date1'] = str(df.iloc[idx1]['date'].date())
        motif['date2'] = str(df.iloc[idx2]['date'].date())

    # Find discords
    print(f"  Finding top {TOP_N} discords...")
    discords = find_discords(profile, TOP_N)

    # Add draw information to discords
    for discord in discords:
        idx = discord['idx']
        discord['draws'] = f"{idx}-{idx+window_size-1}"
        discord['date'] = str(df.iloc[idx]['date'].date())

    print(f"  Found {len(motifs)} motifs and {len(discords)} discords")

    return {
        'motifs': motifs,
        'discords': discords,
        'profile_stats': {
            'mean_distance': float(np.mean(profile)),
            'std_distance': float(np.std(profile)),
            'min_distance': float(np.min(profile)),
            'max_distance': float(np.max(profile))
        }
    }


def find_best_motif_across_windows(results: Dict) -> Dict:
    """Find the best motif across all window sizes."""
    print("\nFinding best motif across all window sizes...")

    best_motif = None
    best_window_size = None
    min_distance = float('inf')

    for window_size, data in results['by_window_size'].items():
        if data and data['motifs']:
            top_motif = data['motifs'][0]  # First motif has lowest distance
            if top_motif['distance'] < min_distance:
                min_distance = top_motif['distance']
                best_motif = top_motif.copy()
                best_window_size = int(window_size)

    if best_motif:
        best_motif['window_size'] = best_window_size
        print(f"  Best motif: window_size={best_window_size}, distance={min_distance:.4f}")

    return best_motif


def find_best_discord_across_windows(results: Dict) -> Dict:
    """Find the best discord across all window sizes."""
    print("Finding best discord across all window sizes...")

    best_discord = None
    best_window_size = None
    max_distance = -float('inf')

    for window_size, data in results['by_window_size'].items():
        if data and data['discords']:
            top_discord = data['discords'][0]  # First discord has highest distance
            if top_discord['distance'] > max_distance:
                max_distance = top_discord['distance']
                best_discord = top_discord.copy()
                best_window_size = int(window_size)

    if best_discord:
        best_discord['window_size'] = best_window_size
        print(f"  Best discord: window_size={best_window_size}, distance={max_distance:.4f}")

    return best_discord


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("Matrix Profile Analysis for Lotto Texas")
    print("=" * 70)

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load and filter data
    df = load_and_filter_data(CSV_FILE, CUTOFF_DATE)

    # Create time series
    time_series = create_time_series(df)

    # Analyze each window size
    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_draws': len(df),
            'date_range': {
                'start': str(df['date'].min().date()),
                'end': str(df['date'].max().date())
            },
            'window_sizes': WINDOW_SIZES,
            'num_balls': NUM_BALLS,
            'top_n': TOP_N
        },
        'by_window_size': {}
    }

    for window_size in WINDOW_SIZES:
        try:
            window_results = analyze_window_size(time_series, df, window_size)
            results['by_window_size'][str(window_size)] = window_results
        except Exception as e:
            print(f"  ERROR analyzing window size {window_size}: {e}")
            results['by_window_size'][str(window_size)] = None

    # Find best motif and discord across all window sizes
    results['best_motif'] = find_best_motif_across_windows(results)
    results['best_discord'] = find_best_discord_across_windows(results)

    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 70)

    # Print summary
    print("\nSUMMARY")
    print("-" * 70)
    if results['best_motif']:
        bm = results['best_motif']
        print(f"Best Motif: window={bm['window_size']}, distance={bm['distance']:.4f}")
        print(f"  Draws: {bm['draws']}")
        print(f"  Dates: {bm['date1']} and {bm['date2']}")

    if results['best_discord']:
        bd = results['best_discord']
        print(f"\nBest Discord: window={bd['window_size']}, distance={bd['distance']:.4f}")
        print(f"  Draws: {bd['draws']}")
        print(f"  Date: {bd['date']}")

    print()


if __name__ == "__main__":
    main()
