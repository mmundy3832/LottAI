"""
Contrast Pattern Mining for Lotto Texas Data

Mines frequent itemsets across different contexts (temporal, seasonal, environmental)
and identifies emerging patterns and anti-patterns using growth rate analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import json
from pathlib import Path
import sys

# Progress logging
def log(message):
    """Print timestamped progress message"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

def load_lottery_data(filepath):
    """Load and parse lottery data, filtering to post-2006"""
    log(f"Loading lottery data from {filepath}")

    # Load CSV (format: Game, Month, Day, Year, N1, N2, N3, N4, N5, N6)
    df = pd.read_csv(filepath, header=None, names=['Game', 'Month', 'Day', 'Year', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6'])

    # Skip header if present (first row might be text)
    if df.iloc[0]['Game'] == 'Lotto Texas' or str(df.iloc[0]['Month']).isdigit() == False:
        df = df.iloc[1:].reset_index(drop=True)

    # Convert date columns to int
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    df['Year'] = df['Year'].astype(int)

    # Create date column
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Filter to 2006-04-01 and later
    df = df[df['Date'] >= '2006-04-01'].copy()

    # Convert number columns to int
    for col in ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']:
        df[col] = df[col].astype(int)

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    log(f"Loaded {len(df)} draws from {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df

def load_weather_data(filepath):
    """Load Austin weather data"""
    log(f"Loading weather data from {filepath}")

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    log(f"Loaded {len(df)} weather records from {df['date'].min().date()} to {df['date'].max().date()}")

    return df

def merge_lottery_weather(lottery_df, weather_df):
    """Merge lottery draws with weather data"""
    log("Merging lottery and weather data")

    # Convert Fahrenheit to Celsius and rename columns for consistency
    weather_df = weather_df.copy()
    weather_df['tavg'] = (weather_df['temp_mean_f'] - 32) * 5/9
    weather_df['tmax'] = (weather_df['temp_max_f'] - 32) * 5/9
    weather_df['tmin'] = (weather_df['temp_min_f'] - 32) * 5/9
    weather_df['pres'] = weather_df['pressure_msl_mean_hpa']

    merged = lottery_df.merge(
        weather_df[['date', 'tavg', 'tmax', 'tmin', 'pres']],
        left_on='Date',
        right_on='date',
        how='left'
    )

    # Check for missing weather data
    missing = merged['tavg'].isna().sum()
    if missing > 0:
        log(f"Warning: {missing} draws missing weather data")

    return merged

def draws_to_transactions(df):
    """Convert lottery draws to transaction format for mlxtend"""
    transactions = []
    for _, row in df.iterrows():
        transaction = [int(row['N1']), int(row['N2']), int(row['N3']),
                      int(row['N4']), int(row['N5']), int(row['N6'])]
        transactions.append(transaction)
    return transactions

def mine_frequent_itemsets(transactions, min_support=0.03):
    """Mine frequent itemsets using Apriori algorithm"""
    log(f"Mining frequent itemsets (min_support={min_support}, n_transactions={len(transactions)})")

    # Use TransactionEncoder to convert to one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, low_memory=True)

    # Convert frozensets to sorted lists for JSON serialization
    # Column name is 'itemsets' in mlxtend
    frequent_itemsets['itemsets_list'] = frequent_itemsets['itemsets'].apply(
        lambda x: sorted(list(x))
    )

    log(f"Found {len(frequent_itemsets)} frequent itemsets")

    return frequent_itemsets

def calculate_contrast_metrics(itemsets_a, itemsets_b, n_a, n_b, context_name):
    """
    Calculate growth rate and significance for patterns between two contexts

    itemsets_a: frequent itemsets from context A (target)
    itemsets_b: frequent itemsets from context B (baseline)
    n_a, n_b: number of transactions in each context
    """
    log(f"Calculating contrast metrics for {context_name}")

    # Create lookup for context B
    b_lookup = {}
    for _, row in itemsets_b.iterrows():
        key = tuple(row['itemsets_list'])
        b_lookup[key] = row['support']

    results = []
    chi2_tests = []

    for _, row_a in itemsets_a.iterrows():
        pattern = row_a['itemsets_list']
        key = tuple(pattern)
        support_a = row_a['support']

        # Find corresponding pattern in context B (or use very small value)
        support_b = b_lookup.get(key, 0.001)

        # Calculate growth rate
        growth_rate = support_a / support_b if support_b > 0 else float('inf')

        # Calculate observed counts
        count_a = int(support_a * n_a)
        count_b = int(support_b * n_b)

        # Skip if counts too small for chi-squared
        if count_a < 1 or count_b < 1:
            continue

        # Build contingency table for chi-squared test
        # [[pattern_in_A, not_pattern_in_A],
        #  [pattern_in_B, not_pattern_in_B]]
        contingency = np.array([
            [count_a, n_a - count_a],
            [count_b, n_b - count_b]
        ])

        # Chi-squared test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
        except:
            p_value = 1.0
            chi2 = 0.0

        # Calculate lift (observed / expected)
        expected_support = (count_a + count_b) / (n_a + n_b)
        lift = support_a / expected_support if expected_support > 0 else 1.0

        result = {
            'pattern': pattern,
            'support_a': float(support_a),
            'support_b': float(support_b),
            'count_a': count_a,
            'count_b': count_b,
            'growth_rate': float(growth_rate) if growth_rate != float('inf') else 999.0,
            'lift': float(lift),
            'chi2': float(chi2),
            'chi2_p': float(p_value)
        }

        results.append(result)
        chi2_tests.append(p_value)

    # Apply FDR correction (Benjamini-Hochberg)
    if len(chi2_tests) > 0:
        log(f"Applying FDR correction to {len(chi2_tests)} tests")
        reject, pvals_corrected, _, _ = multipletests(chi2_tests, alpha=0.05, method='fdr_bh')

        for i, result in enumerate(results):
            result['fdr_significant'] = bool(reject[i])
            result['fdr_p'] = float(pvals_corrected[i])

    # Sort by growth rate descending
    results.sort(key=lambda x: x['growth_rate'], reverse=True)

    log(f"Found {len(results)} contrast patterns")

    return results

def filter_emerging_patterns(results, min_growth=1.5, require_significance=True, p_threshold=0.05):
    """Filter for emerging patterns (high growth rate)"""
    emerging = [
        r for r in results
        if r['growth_rate'] >= min_growth
        and (not require_significance or r.get('chi2_p', 1.0) <= p_threshold)
    ]
    log(f"Found {len(emerging)} emerging patterns (growth >= {min_growth}, p <= {p_threshold})")
    return emerging

def filter_anti_patterns(results, max_growth=0.5, require_significance=True, p_threshold=0.05):
    """Filter for anti-patterns (negative growth)"""
    anti = [
        r for r in results
        if r['growth_rate'] <= max_growth
        and (not require_significance or r.get('chi2_p', 1.0) <= p_threshold)
    ]
    log(f"Found {len(anti)} anti-patterns (growth <= {max_growth}, p <= {p_threshold})")
    return anti

def temporal_contrast(df, min_support=0.03):
    """Recent 500 draws vs Previous 500 draws"""
    log("\n=== TEMPORAL CONTRAST: Recent vs Previous ===")

    if len(df) < 1000:
        log(f"Warning: Only {len(df)} draws available, adjusting temporal split")
        split = len(df) // 2
        recent = df.iloc[-split:].copy()
        previous = df.iloc[:split].copy()
    else:
        recent = df.iloc[-500:].copy()
        previous = df.iloc[-1000:-500].copy()

    log(f"Recent: {len(recent)} draws ({recent['Date'].min().date()} to {recent['Date'].max().date()})")
    log(f"Previous: {len(previous)} draws ({previous['Date'].min().date()} to {previous['Date'].max().date()})")

    recent_trans = draws_to_transactions(recent)
    previous_trans = draws_to_transactions(previous)

    recent_items = mine_frequent_itemsets(recent_trans, min_support)
    previous_items = mine_frequent_itemsets(previous_trans, min_support)

    contrast = calculate_contrast_metrics(
        recent_items, previous_items,
        len(recent_trans), len(previous_trans),
        "Temporal"
    )

    return {
        'emerging_patterns': filter_emerging_patterns(contrast, min_growth=1.3, p_threshold=0.10),
        'anti_patterns': filter_anti_patterns(contrast, max_growth=0.7, p_threshold=0.10),
        'all_patterns': contrast[:50]  # Top 50 by growth rate
    }

def seasonal_contrast(df, min_support=0.03):
    """Summer (Jun-Aug) vs Winter (Dec-Feb)"""
    log("\n=== SEASONAL CONTRAST: Summer vs Winter ===")

    df['month'] = df['Date'].dt.month

    summer = df[df['month'].isin([6, 7, 8])].copy()
    winter = df[df['month'].isin([12, 1, 2])].copy()

    log(f"Summer: {len(summer)} draws")
    log(f"Winter: {len(winter)} draws")

    if len(summer) < 10 or len(winter) < 10:
        log("Insufficient data for seasonal contrast")
        return {'emerging_patterns': [], 'anti_patterns': [], 'all_patterns': []}

    summer_trans = draws_to_transactions(summer)
    winter_trans = draws_to_transactions(winter)

    summer_items = mine_frequent_itemsets(summer_trans, min_support)
    winter_items = mine_frequent_itemsets(winter_trans, min_support)

    contrast = calculate_contrast_metrics(
        summer_items, winter_items,
        len(summer_trans), len(winter_trans),
        "Seasonal"
    )

    return {
        'emerging_patterns': filter_emerging_patterns(contrast, min_growth=1.3, p_threshold=0.10),
        'anti_patterns': filter_anti_patterns(contrast, max_growth=0.7, p_threshold=0.10),
        'all_patterns': contrast[:50]
    }

def pressure_contrast(df, min_support=0.03):
    """High pressure (>1018 hPa) vs Low pressure (<1012 hPa)"""
    log("\n=== PRESSURE CONTRAST: High vs Low ===")

    # Filter out missing pressure data
    df_clean = df[df['pres'].notna()].copy()

    high = df_clean[df_clean['pres'] > 1018].copy()
    low = df_clean[df_clean['pres'] < 1012].copy()

    log(f"High pressure: {len(high)} draws")
    log(f"Low pressure: {len(low)} draws")

    if len(high) < 10 or len(low) < 10:
        log("Insufficient data for pressure contrast")
        return {'emerging_patterns': [], 'anti_patterns': [], 'all_patterns': []}

    high_trans = draws_to_transactions(high)
    low_trans = draws_to_transactions(low)

    high_items = mine_frequent_itemsets(high_trans, min_support)
    low_items = mine_frequent_itemsets(low_trans, min_support)

    contrast = calculate_contrast_metrics(
        high_items, low_items,
        len(high_trans), len(low_trans),
        "Pressure"
    )

    return {
        'emerging_patterns': filter_emerging_patterns(contrast, min_growth=1.3, p_threshold=0.10),
        'anti_patterns': filter_anti_patterns(contrast, max_growth=0.7, p_threshold=0.10),
        'all_patterns': contrast[:50]
    }

def temperature_contrast(df, min_support=0.03):
    """Hot (>30°C) vs Cold (<15°C)"""
    log("\n=== TEMPERATURE CONTRAST: Hot vs Cold ===")

    # Filter out missing temperature data
    df_clean = df[df['tavg'].notna()].copy()

    hot = df_clean[df_clean['tavg'] > 30].copy()
    cold = df_clean[df_clean['tavg'] < 15].copy()

    log(f"Hot days: {len(hot)} draws")
    log(f"Cold days: {len(cold)} draws")

    if len(hot) < 10 or len(cold) < 10:
        log("Insufficient data for temperature contrast")
        return {'emerging_patterns': [], 'anti_patterns': [], 'all_patterns': []}

    hot_trans = draws_to_transactions(hot)
    cold_trans = draws_to_transactions(cold)

    hot_items = mine_frequent_itemsets(hot_trans, min_support)
    cold_items = mine_frequent_itemsets(cold_trans, min_support)

    contrast = calculate_contrast_metrics(
        hot_items, cold_items,
        len(hot_trans), len(cold_trans),
        "Temperature"
    )

    return {
        'emerging_patterns': filter_emerging_patterns(contrast, min_growth=1.3, p_threshold=0.10),
        'anti_patterns': filter_anti_patterns(contrast, max_growth=0.7, p_threshold=0.10),
        'all_patterns': contrast[:50]
    }

def main():
    """Main execution"""
    log("="*60)
    log("Contrast Pattern Mining - Lotto Texas")
    log("="*60)

    # File paths
    lottery_file = Path("D:/projects/LottAI/lottotexas.csv")
    weather_file = Path("D:/projects/LottAI/austin_weather_2006_2024.csv")
    output_file = Path("D:/projects/LottAI/results/contrast_mining_findings.json")

    # Ensure results directory exists
    output_file.parent.mkdir(exist_ok=True)

    # Load data
    lottery_df = load_lottery_data(lottery_file)
    weather_df = load_weather_data(weather_file)
    merged_df = merge_lottery_weather(lottery_df, weather_df)

    # Run all contrasts
    min_support = 0.03

    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_draws': len(lottery_df),
            'date_range': {
                'start': lottery_df['Date'].min().isoformat(),
                'end': lottery_df['Date'].max().isoformat()
            },
            'min_support': min_support,
            'growth_threshold': 1.3,
            'anti_pattern_threshold': 0.7,
            'p_threshold': 0.10
        },
        'temporal_contrast': temporal_contrast(lottery_df, min_support),
        'seasonal_contrast': seasonal_contrast(lottery_df, min_support),
        'pressure_contrast': pressure_contrast(merged_df, min_support),
        'temperature_contrast': temperature_contrast(merged_df, min_support)
    }

    # Save results
    log(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)

    for contrast_name in ['temporal_contrast', 'seasonal_contrast', 'pressure_contrast', 'temperature_contrast']:
        data = results[contrast_name]
        log(f"\n{contrast_name.replace('_', ' ').title()}:")
        log(f"  Emerging patterns: {len(data['emerging_patterns'])}")
        log(f"  Anti-patterns: {len(data['anti_patterns'])}")

        # Show top emerging pattern
        if data['emerging_patterns']:
            top = data['emerging_patterns'][0]
            log(f"  Top emerging: {top['pattern']} (growth: {top['growth_rate']:.2f}, p: {top.get('fdr_p', top['chi2_p']):.4f})")

        # Show top anti-pattern
        if data['anti_patterns']:
            top = data['anti_patterns'][0]
            log(f"  Top anti: {top['pattern']} (growth: {top['growth_rate']:.2f}, p: {top.get('fdr_p', top['chi2_p']):.4f})")

    log("\n" + "="*60)
    log("Contrast mining complete!")
    log("="*60)

if __name__ == '__main__':
    main()
