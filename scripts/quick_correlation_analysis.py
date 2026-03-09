#!/usr/bin/env python3
"""
Quick Feature Correlation Analysis
===================================

Simplified script that extracts features using existing V6 code and analyzes
correlations to identify redundant features BEFORE training.

This gives us data-driven answers to:
1. Which features are highly correlated (r > 0.9)?
2. Which feature groups have redundancy?
3. What can we safely remove for V6?
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List
import yfinance as yf
import warnings
from config.dimensions import DIMS
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.features.feature_extractor import FeatureExtractor
from fincoll.providers.yfinance_provider import YFinanceProvider

# Feature group definitions
FEATURE_GROUPS = {
    "raw_returns": list(range(0, 10)),
    "normalized_position": list(range(10, 15)),
    "moving_averages": list(range(15, 25)),
    "bollinger_bands": list(range(25, 28)),
    "volatility": list(range(28, 31)),
    "volume": list(range(31, 46)),
    "momentum": list(range(46, 81)),
    "advanced_tech": list(range(81, 131)),
    "velocity_accel": list(range(131, 151)),
    "news_sentiment": list(range(151, 171)),
    "fundamentals": list(range(171, 187)),
    "cross_asset_v6": list(range(187, 205)),
    "sector": list(range(205, 219)),
    "options": list(range(219, 229)),
    "support_resistance": list(range(229, 259)),
    "reserved": list(range(259, 263)),
    "senvec": list(range(263, 335))
}


def extract_features_simple(symbol: str, days: int = 30) -> np.ndarray:
    """Extract features for a single symbol using V6 extractor"""
    print(f"  Extracting features for {symbol}...", end="", flush=True)

    # Get data via yfinance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days+300)  # Extra for indicators

    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)

    if len(data) < 100:
        print(f" ❌ Insufficient data ({len(data)} days)")
        return None

    # Setup feature extractor
    provider = YFinanceProvider()
    extractor = FeatureExtractor(data_provider=provider)

    # Extract features for each day
    features_list = []
    for i in range(-days, 0):
        try:
            timestamp = data.index[i]
            # Extract DIMS.fincoll_total features
            features = extractor.extract_features(
                timestamp=timestamp,
                symbol=symbol,
                data_provider=provider
            )

            if features is not None and len(features) == 335:
                features_list.append(features)

        except Exception as e:
            continue

    if features_list:
        result = np.array(features_list)
        print(f" ✅ {len(features_list)} vectors ({result.shape})")
        return result
    else:
        print(f" ❌ No valid features")
        return None


def analyze_correlations(all_features: np.ndarray):
    """Analyze feature correlations"""
    print(f"\n🔍 Analyzing {all_features.shape[0]} feature vectors...")

    # Calculate correlation matrix
    print("  Computing correlation matrix (335x335)...")
    corr_matrix = np.corrcoef(all_features.T)

    # Find highly correlated pairs
    high_corr = []
    for i in range(335):
        for j in range(i+1, 335):
            corr = corr_matrix[i, j]
            if not np.isnan(corr) and abs(corr) > 0.9:
                group_i = get_feature_group(i)
                group_j = get_feature_group(j)
                high_corr.append({
                    'f1': i,
                    'f2': j,
                    'group1': group_i,
                    'group2': group_j,
                    'corr': corr
                })

    print(f"\n✅ Found {len(high_corr)} feature pairs with |r| > 0.9")

    if high_corr:
        df = pd.DataFrame(high_corr).sort_values('corr', ascending=False, key=abs)
        print("\nTop 20 Highly Correlated Pairs:")
        print(df.head(20).to_string(index=False))

    return corr_matrix, high_corr


def analyze_group_correlations(all_features: np.ndarray):
    """Analyze correlation between feature groups"""
    print("\n🔍 Analyzing inter-group correlations...")

    results = []

    for name1, indices1 in FEATURE_GROUPS.items():
        group1_data = all_features[:, indices1]

        # Skip if all zeros
        if np.all(group1_data == 0):
            print(f"  ⚠️ {name1}: All zeros, skipping")
            continue

        for name2, indices2 in FEATURE_GROUPS.items():
            if name1 >= name2:
                continue

            group2_data = all_features[:, indices2]

            if np.all(group2_data == 0):
                continue

            # Calculate average correlation
            correlations = []
            for f1 in group1_data.T:
                for f2 in group2_data.T:
                    r = np.corrcoef(f1, f2)[0, 1]
                    if not np.isnan(r):
                        correlations.append(abs(r))

            if correlations:
                avg_corr = np.mean(correlations)
                max_corr = np.max(correlations)

                results.append({
                    'group1': name1,
                    'group2': name2,
                    'avg_r': avg_corr,
                    'max_r': max_corr,
                    'dims1': len(indices1),
                    'dims2': len(indices2)
                })

    df = pd.DataFrame(results).sort_values('avg_r', ascending=False)

    print("\nTop 15 Correlated Group Pairs:")
    print(df.head(15).to_string(index=False))

    return df


def identify_zero_groups(all_features: np.ndarray):
    """Find feature groups that are all zeros"""
    print("\n🔍 Identifying zero-valued feature groups...")

    zero_groups = []
    partial_zero_groups = []

    for name, indices in FEATURE_GROUPS.items():
        group_data = all_features[:, indices]
        zero_pct = np.mean(group_data == 0) * 100

        if zero_pct == 100:
            zero_groups.append((name, len(indices)))
            print(f"  ❌ {name} ({len(indices)}D): 100% zeros")
        elif zero_pct > 90:
            partial_zero_groups.append((name, len(indices), zero_pct))
            print(f"  ⚠️ {name} ({len(indices)}D): {zero_pct:.1f}% zeros")

    return zero_groups, partial_zero_groups


def get_feature_group(index: int) -> str:
    """Get feature group name for index"""
    for name, indices in FEATURE_GROUPS.items():
        if index in indices:
            return name
    return "unknown"


def print_recommendations(zero_groups, high_corr, group_corr_df):
    """Print actionable recommendations"""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR V6 FEATURE REDUCTION")
    print("=" * 80)

    total_savings = 0

    print("\n1. REMOVE ZERO GROUPS (High Confidence)")
    for name, dims in zero_groups:
        print(f"   ❌ Remove {name} ({dims}D) - all zeros, no information")
        total_savings += dims

    print(f"\n2. INVESTIGATE HIGHLY CORRELATED PAIRS")
    if high_corr:
        # Group by feature groups
        group_pairs = {}
        for item in high_corr:
            key = tuple(sorted([item['group1'], item['group2']]))
            group_pairs[key] = group_pairs.get(key, 0) + 1

        for (g1, g2), count in sorted(group_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   🔍 {g1} ↔ {g2}: {count} highly correlated pairs (r > 0.9)")

    print(f"\n3. CONSIDER REDUCING HIGH-CORRELATION GROUPS")
    if not group_corr_df.empty:
        for _, row in group_corr_df.head(5).iterrows():
            if row['avg_r'] > 0.7:
                print(f"   ⚠️ {row['group1']} ↔ {row['group2']}: avg r={row['avg_r']:.2f}, max r={row['max_r']:.2f}")

    print(f"\n4. ESTIMATED DIMENSION REDUCTION")
    print(f"   Current: DIMS.fincoll_total")
    print(f"   Remove zeros: -{total_savings}D")
    print(f"   Proposed minimum: {335 - total_savings}D")

    # Test specific hypothesis from user
    print(f"\n5. TESTING YOUR HYPOTHESIS: Price history vs Technical indicators")
    if not group_corr_df.empty:
        price_tech_corr = group_corr_df[
            ((group_corr_df['group1'] == 'raw_returns') & (group_corr_df['group2'] == 'velocity_accel')) |
            ((group_corr_df['group1'] == 'raw_returns') & (group_corr_df['group2'] == 'momentum')) |
            ((group_corr_df['group1'] == 'normalized_position') & (group_corr_df['group2'] == 'bollinger_bands'))
        ]

        if not price_tech_corr.empty:
            print("   Found correlations:")
            for _, row in price_tech_corr.iterrows():
                print(f"     - {row['group1']} ↔ {row['group2']}: avg r={row['avg_r']:.2f}")
                if row['avg_r'] > 0.8:
                    print(f"       ✅ HIGH REDUNDANCY - Consider removing {row['group1']}")
        else:
            print("   ℹ️ Need more analysis to test hypothesis")

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("QUICK FEATURE CORRELATION ANALYSIS")
    print("=" * 80)

    # Extract features for a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    days = 30

    print(f"\n📊 Extracting features for {len(symbols)} symbols, {days} days each...")

    all_features_list = []
    for symbol in symbols:
        features = extract_features_simple(symbol, days)
        if features is not None:
            all_features_list.append(features)

    if not all_features_list:
        print("\n❌ ERROR: No features extracted")
        return 1

    # Concatenate all features
    all_features = np.vstack(all_features_list)
    print(f"\n✅ Total feature vectors: {all_features.shape}")

    # Analyze
    zero_groups, partial_zero = identify_zero_groups(all_features)
    corr_matrix, high_corr = analyze_correlations(all_features)
    group_corr_df = analyze_group_correlations(all_features)

    # Recommendations
    print_recommendations(zero_groups, high_corr, group_corr_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
