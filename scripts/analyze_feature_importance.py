#!/usr/bin/env python3
"""
Feature Importance and Redundancy Analysis
==========================================

Goal: Empirically determine which feature groups are redundant BEFORE training V6.

Approach:
1. Extract features for sample stocks
2. Calculate correlation matrix
3. Identify highly correlated features (r > 0.9)
4. Test which features affect V5 model predictions
5. Recommend features to remove for V6

Usage:
    python scripts/analyze_feature_importance.py --symbols AAPL MSFT GOOGL --days 60
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
from config.dimensions import DIMS
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.features.feature_extractor import FeatureExtractor
from fincoll.providers.tradestation_provider import TradeStationProvider
from fincoll.providers.yfinance_provider import YFinanceProvider

# Feature group definitions (matches V8_FEATURE_REDUCTION.md)
FEATURE_GROUPS = {
    "raw_returns": {
        "indices": list(range(0, 10)),
        "description": "Period returns (1d, 2d, 3d, 5d, 10d, 20d, 50d, 100d, 200d, YTD)",
        "hypothesis": "Redundant with velocity/acceleration derivatives"
    },
    "normalized_position": {
        "indices": list(range(10, 15)),
        "description": "Normalized position in 5d, 10d, 20d, 50d, 100d ranges",
        "hypothesis": "Redundant with Bollinger Bands"
    },
    "moving_averages": {
        "indices": list(range(15, 25)),
        "description": "MA ratios and slopes (5, 20, 50, 100, 200-day)",
        "hypothesis": "Partially redundant with MACD and Ichimoku"
    },
    "bollinger_bands": {
        "indices": list(range(25, 28)),
        "description": "Bollinger Band position and width",
        "hypothesis": "Keep (unique volatility signal)"
    },
    "volatility": {
        "indices": list(range(28, 31)),
        "description": "ATR and volatility measures",
        "hypothesis": "Keep (essential risk metric)"
    },
    "volume_ratios": {
        "indices": list(range(31, 46)),
        "description": "Volume ratios, changes, trends",
        "hypothesis": "Partially redundant with OBV"
    },
    "momentum_oscillators": {
        "indices": list(range(46, 81)),
        "description": "RSI, MACD, Stochastic, etc.",
        "hypothesis": "Some redundancy between oscillators"
    },
    "advanced_technical": {
        "indices": list(range(81, 131)),
        "description": "CCI, ADX, OBV, VWAP, Ichimoku, Fibonacci",
        "hypothesis": "Keep best performers from each category"
    },
    "velocity_accel": {
        "indices": list(range(131, 151)),
        "description": "Price/volume derivatives",
        "hypothesis": "Keep (NOT redundant, key signal)"
    },
    "news_sentiment": {
        "indices": list(range(151, 171)),
        "description": "Text-based sentiment scores",
        "hypothesis": "Keep (unique signal if available)"
    },
    "fundamentals": {
        "indices": list(range(171, 187)),
        "description": "PE, EPS, Beta, Market Cap, etc.",
        "hypothesis": "Keep (essential context)"
    },
    "cross_asset_v6": {
        "indices": list(range(187, 205)),
        "description": "Beta-residual momentum (V6 fix)",
        "hypothesis": "Keep (V6's key improvement)"
    },
    "sector": {
        "indices": list(range(205, 219)),
        "description": "Sector classification + relative performance",
        "hypothesis": "Keep (sector context)"
    },
    "options": {
        "indices": list(range(219, 229)),
        "description": "Options flow indicators",
        "hypothesis": "Keep (options flow signal)"
    },
    "support_resistance": {
        "indices": list(range(229, 259)),
        "description": "Support/resistance levels and breakouts",
        "hypothesis": "Keep (key levels)"
    },
    "reserved": {
        "indices": list(range(259, 263)),
        "description": "Unused placeholder",
        "hypothesis": "Remove (empty)"
    },
    "senvec": {
        "indices": list(range(263, 335)),
        "description": "Multi-source sentiment (DIMS.senvec_total)",
        "hypothesis": "Major redundancy, reduce to ~20D"
    }
}


def extract_features_for_symbols(symbols: List[str], days: int = 60) -> Dict[str, np.ndarray]:
    """Extract DIMS.fincoll_total features for given symbols"""
    print(f"\n📊 Extracting features for {len(symbols)} symbols...")

    # Initialize providers
    try:
        provider = TradeStationProvider()
        print("✅ Using TradeStation as data provider")
    except Exception as e:
        print(f"⚠️ TradeStation unavailable: {e}")
        provider = YFinanceProvider()
        print("✅ Using yfinance as fallback provider")

    extractor = FeatureExtractor(data_provider=provider)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 300)  # Extra data for indicators

    features_dict = {}
    for symbol in symbols:
        try:
            print(f"  Processing {symbol}...", end="")

            # Fetch OHLCV data
            data = provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )

            if data is None or len(data) < 100:
                print(f" ❌ Insufficient data")
                continue

            # Extract features for each day in the last 60 days
            features_list = []
            for i in range(-days, 0):
                try:
                    timestamp = data.index[i]
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
                features_dict[symbol] = np.array(features_list)
                print(f" ✅ {len(features_list)} feature vectors")
            else:
                print(f" ❌ No valid features")

        except Exception as e:
            print(f" ❌ Error: {e}")

    return features_dict


def analyze_feature_correlations(features_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Calculate correlation matrix and identify highly correlated features.

    Returns:
        DataFrame of feature pairs with correlation > 0.9
    """
    print("\n🔍 Analyzing feature correlations...")

    # Concatenate all features across symbols and time
    all_features = np.vstack(list(features_dict.values()))
    print(f"  Total feature vectors: {all_features.shape[0]}")
    print(f"  Feature dimensions: {all_features.shape[1]}")

    # Calculate correlation matrix
    print("  Computing correlation matrix...")
    corr_matrix = np.corrcoef(all_features.T)

    # Find high correlations (excluding diagonal)
    print("  Identifying highly correlated pairs...")
    high_corr = []

    for i in range(335):
        for j in range(i+1, 335):
            corr = corr_matrix[i, j]
            if not np.isnan(corr) and abs(corr) > 0.9:
                # Find which groups these features belong to
                group_i = get_feature_group(i)
                group_j = get_feature_group(j)

                high_corr.append({
                    'feature_i': i,
                    'feature_j': j,
                    'group_i': group_i,
                    'group_j': group_j,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })

    df_high_corr = pd.DataFrame(high_corr).sort_values('abs_correlation', ascending=False)

    print(f"\n✅ Found {len(df_high_corr)} feature pairs with |r| > 0.9")

    return df_high_corr


def get_feature_group(index: int) -> str:
    """Get feature group name for a given index"""
    for group_name, group_info in FEATURE_GROUPS.items():
        if index in group_info["indices"]:
            return group_name
    return "unknown"


def analyze_group_redundancy(features_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Calculate correlation between feature groups.

    Returns:
        DataFrame showing average correlation between groups
    """
    print("\n🔍 Analyzing inter-group correlations...")

    all_features = np.vstack(list(features_dict.values()))

    group_correlations = []

    for group1_name, group1_info in FEATURE_GROUPS.items():
        indices1 = group1_info["indices"]
        features1 = all_features[:, indices1]

        # Skip if all zeros
        if np.all(features1 == 0):
            continue

        for group2_name, group2_info in FEATURE_GROUPS.items():
            if group1_name >= group2_name:  # Avoid duplicates
                continue

            indices2 = group2_info["indices"]
            features2 = all_features[:, indices2]

            # Skip if all zeros
            if np.all(features2 == 0):
                continue

            # Calculate average correlation between groups
            correlations = []
            for i in range(features1.shape[1]):
                for j in range(features2.shape[1]):
                    corr = np.corrcoef(features1[:, i], features2[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if correlations:
                avg_corr = np.mean(correlations)
                max_corr = np.max(correlations)

                group_correlations.append({
                    'group1': group1_name,
                    'group2': group2_name,
                    'avg_correlation': avg_corr,
                    'max_correlation': max_corr,
                    'dims1': len(indices1),
                    'dims2': len(indices2)
                })

    df_group_corr = pd.DataFrame(group_correlations).sort_values('avg_correlation', ascending=False)

    return df_group_corr


def identify_zero_features(features_dict: Dict[str, np.ndarray]) -> List[str]:
    """Identify feature groups that are all zeros"""
    print("\n🔍 Identifying zero-valued features...")

    all_features = np.vstack(list(features_dict.values()))

    zero_groups = []
    for group_name, group_info in FEATURE_GROUPS.items():
        indices = group_info["indices"]
        group_features = all_features[:, indices]

        # Check if all values are zero
        if np.all(group_features == 0):
            zero_groups.append(group_name)
            print(f"  ❌ {group_name}: All zeros ({len(indices)}D)")
        elif np.mean(group_features == 0) > 0.9:
            print(f"  ⚠️ {group_name}: Mostly zeros ({np.mean(group_features == 0)*100:.1f}%)")

    return zero_groups


def generate_recommendations(
    high_corr_df: pd.DataFrame,
    group_corr_df: pd.DataFrame,
    zero_groups: List[str]
) -> pd.DataFrame:
    """Generate recommendations for V6 feature reduction"""
    print("\n💡 Generating V6 feature recommendations...")

    recommendations = []

    # Recommendation 1: Remove zero groups
    for group_name in zero_groups:
        group_info = FEATURE_GROUPS[group_name]
        recommendations.append({
            'priority': 'HIGH',
            'action': 'REMOVE',
            'group': group_name,
            'dimensions': len(group_info['indices']),
            'reason': 'All zeros (no data)',
            'confidence': 'Very High'
        })

    # Recommendation 2: Remove highly correlated feature groups
    for _, row in group_corr_df.head(10).iterrows():
        if row['avg_correlation'] > 0.85:
            # Determine which group to remove
            group1_dims = row['dims1']
            group2_dims = row['dims2']

            # Keep the smaller group, remove the larger (more redundancy)
            remove_group = row['group1'] if group1_dims > group2_dims else row['group2']
            keep_group = row['group2'] if group1_dims > group2_dims else row['group1']

            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'CONSIDER REMOVING',
                'group': remove_group,
                'dimensions': max(group1_dims, group2_dims),
                'reason': f"High correlation with {keep_group} (r={row['avg_correlation']:.2f})",
                'confidence': 'Medium'
            })

    # Recommendation 3: Address specific hypotheses from V8 document
    if not high_corr_df.empty:
        # Check raw returns vs velocity
        raw_returns_corr = high_corr_df[
            (high_corr_df['group_i'] == 'raw_returns') &
            (high_corr_df['group_j'] == 'velocity_accel')
        ]

        if not raw_returns_corr.empty:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'REMOVE',
                'group': 'raw_returns',
                'dimensions': 10,
                'reason': 'Confirmed: Redundant with velocity/acceleration derivatives',
                'confidence': 'High'
            })

    df_recommendations = pd.DataFrame(recommendations).sort_values(
        ['priority', 'dimensions'],
        ascending=[True, False]
    )

    return df_recommendations


def visualize_results(high_corr_df: pd.DataFrame, group_corr_df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots"""
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Feature pair correlations
    if not high_corr_df.empty:
        plt.figure(figsize=(12, 6))
        top_pairs = high_corr_df.head(20)
        plt.barh(range(len(top_pairs)), top_pairs['abs_correlation'])
        plt.yticks(range(len(top_pairs)),
                  [f"{row['group_i']}[{row['feature_i']}] ↔ {row['group_j']}[{row['feature_j']}]"
                   for _, row in top_pairs.iterrows()],
                  fontsize=8)
        plt.xlabel('Absolute Correlation')
        plt.title('Top 20 Highly Correlated Feature Pairs (|r| > 0.9)')
        plt.axvline(x=0.9, color='r', linestyle='--', label='Threshold (0.9)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_pair_correlations.png', dpi=150)
        print(f"  ✅ Saved: {output_dir / 'feature_pair_correlations.png'}")
        plt.close()

    # Plot 2: Group correlations
    if not group_corr_df.empty:
        plt.figure(figsize=(12, 8))
        top_groups = group_corr_df.head(15)
        plt.barh(range(len(top_groups)), top_groups['avg_correlation'])
        plt.yticks(range(len(top_groups)),
                  [f"{row['group1']} ↔ {row['group2']}" for _, row in top_groups.iterrows()],
                  fontsize=9)
        plt.xlabel('Average Correlation')
        plt.title('Top 15 Inter-Group Correlations')
        plt.axvline(x=0.7, color='orange', linestyle='--', label='Medium redundancy (0.7)')
        plt.axvline(x=0.85, color='r', linestyle='--', label='High redundancy (0.85)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'group_correlations.png', dpi=150)
        print(f"  ✅ Saved: {output_dir / 'group_correlations.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance and redundancy")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                       help="Symbols to analyze")
    parser.add_argument("--days", type=int, default=60,
                       help="Number of days to analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"),
                       help="Output directory for results")

    args = parser.parse_args()

    print("=" * 80)
    print("FEATURE IMPORTANCE & REDUNDANCY ANALYSIS")
    print("=" * 80)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Analysis period: {args.days} days")
    print(f"Output directory: {args.output_dir}")

    # Step 1: Extract features
    features_dict = extract_features_for_symbols(args.symbols, args.days)

    if not features_dict:
        print("\n❌ ERROR: No features extracted. Check data providers.")
        return 1

    # Step 2: Identify zero features
    zero_groups = identify_zero_features(features_dict)

    # Step 3: Analyze feature correlations
    high_corr_df = analyze_feature_correlations(features_dict)

    # Step 4: Analyze group correlations
    group_corr_df = analyze_group_redundancy(features_dict)

    # Step 5: Generate recommendations
    recommendations_df = generate_recommendations(high_corr_df, group_corr_df, zero_groups)

    # Step 6: Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(high_corr_df, group_corr_df, args.output_dir)

    # Step 7: Save results
    print("\n💾 Saving results...")
    args.output_dir.mkdir(exist_ok=True)

    high_corr_df.to_csv(args.output_dir / "high_correlations.csv", index=False)
    print(f"  ✅ Saved: {args.output_dir / 'high_correlations.csv'}")

    group_corr_df.to_csv(args.output_dir / "group_correlations.csv", index=False)
    print(f"  ✅ Saved: {args.output_dir / 'group_correlations.csv'}")

    recommendations_df.to_csv(args.output_dir / "recommendations.csv", index=False)
    print(f"  ✅ Saved: {args.output_dir / 'recommendations.csv'}")

    # Step 8: Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\n📊 Feature Vectors Analyzed: {sum(len(v) for v in features_dict.values())}")
    print(f"📊 Symbols Processed: {len(features_dict)}")

    print(f"\n❌ Zero Feature Groups: {len(zero_groups)}")
    for group in zero_groups:
        dims = len(FEATURE_GROUPS[group]['indices'])
        print(f"   - {group} ({dims}D)")

    print(f"\n🔗 Highly Correlated Pairs (|r| > 0.9): {len(high_corr_df)}")
    if not high_corr_df.empty:
        print("\nTop 10:")
        for _, row in high_corr_df.head(10).iterrows():
            print(f"   - f{row['feature_i']} ({row['group_i']}) ↔ f{row['feature_j']} ({row['group_j']}): r={row['correlation']:.3f}")

    print(f"\n🔗 Highly Correlated Groups (avg r > 0.7): {len(group_corr_df[group_corr_df['avg_correlation'] > 0.7])}")
    if not group_corr_df.empty:
        print("\nTop 5:")
        for _, row in group_corr_df.head(5).iterrows():
            print(f"   - {row['group1']} ↔ {row['group2']}: avg r={row['avg_correlation']:.3f}, max r={row['max_correlation']:.3f}")

    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 80)
    print(recommendations_df.to_string(index=False))

    # Calculate potential dimensionality reduction
    total_removable = recommendations_df[recommendations_df['action'].str.contains('REMOVE')]['dimensions'].sum()
    print(f"\n📉 Potential Dimension Reduction: {total_removable}D")
    print(f"   Current: DIMS.fincoll_total")
    print(f"   Proposed: {335 - total_removable}D")
    print(f"   Reduction: {total_removable/335*100:.1f}%")

    print("\n✅ Analysis complete! Check the output directory for detailed results.")
    print(f"   Output: {args.output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
