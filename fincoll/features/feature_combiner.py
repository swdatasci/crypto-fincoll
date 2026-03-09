"""
Feature Combiner - Unified Integration Layer

KEY INSIGHT: Combine all three feature streams intelligently!
Each stream answers a different question:
- Technicals (WHAT): What is happening to price/volume?
- Fundamentals (WHO): What kind of company is this?
- Sentiment (WHY): Why is the price behaving this way?

Together, they provide complete market context for predictions.

Agent Delta - Phase 1 FEATURE INTEGRATION stream
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .fundamental_embeddings import FundamentalEmbedding, CompanyContext
from .fundamental_history import FundamentalHistoryTracker
from .market_regime import MarketRegimeClassifier, RegimeContext
from ..fundamentals.fundamental_collector import FundamentalCollector
from ..sentiment.news_sentiment import NewsSentimentAnalyzer
from ..technicals.advanced_indicators import TechnicalIndicatorCalculator


@dataclass
class UnifiedFeatureVector:
    """
    Unified feature vector combining all three streams.

    This is what gets fed to the model for training/prediction.
    """

    # Symbol and timestamp
    symbol: str
    timestamp: datetime

    # Price patterns (WHAT is happening) - 20 normalized technical indicators
    technicals: Dict[str, float]

    # Company context (WHO is this company) - 26 dimensional embedding
    company_context: CompanyContext

    # Market regime (WHY is it happening) - 15 dimensional embedding
    regime_context: RegimeContext

    # Additional metadata
    days_until_earnings: Optional[int]  # Risk indicator
    fundamental_changed_recently: bool  # True if fundamentals updated in last 7 days

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary for easier inspection.

        Returns:
            Dictionary with all features flattened
        """
        features = {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'days_until_earnings': self.days_until_earnings,
            'fundamental_changed_recently': float(self.fundamental_changed_recently),
        }

        # Add technicals (prefix with 'tech_')
        for key, value in self.technicals.items():
            features[f'tech_{key}'] = value

        # Add company context (prefix with 'company_')
        features['company_sector'] = self.company_context.sector.name
        features['company_size'] = self.company_context.size_category.name
        features['company_style'] = self.company_context.growth_style.name
        features['company_growth_score'] = self.company_context.growth_score
        features['company_quality_score'] = self.company_context.quality_score
        features['company_leverage_score'] = self.company_context.leverage_score
        features['company_profitability_score'] = self.company_context.profitability_score

        # Add regime context (prefix with 'regime_')
        features['regime'] = self.regime_context.regime.name
        features['regime_fear_score'] = self.regime_context.fear_score
        features['regime_greed_score'] = self.regime_context.greed_score
        features['regime_uncertainty_score'] = self.regime_context.uncertainty_score
        features['regime_sentiment_score'] = self.regime_context.sentiment_score
        features['regime_sentiment_std'] = self.regime_context.sentiment_std
        features['regime_is_high_volatility'] = float(self.regime_context.is_high_volatility)
        features['regime_is_extreme'] = float(self.regime_context.is_extreme_sentiment)

        return features

    def to_model_input(self) -> np.ndarray:
        """
        Convert to numerical vector for model input.

        Returns:
            Array with shape (technical_dims + company_dims + regime_dims + metadata)
            = (20 + 26 + 15 + 2) = 63 features
        """
        # Technical indicators (20 features - use main ones)
        tech_keys = [
            'rsi_14', 'macd', 'macd_histogram', 'stochastic_k', 'bb_percent',
            'adx', 'cci', 'williams_r', 'atr', 'cmf',
            'sma_20', 'ema_12', 'volume_roc', 'obv', 'historical_volatility',
            'bb_width', 'stochastic_d', 'macd_signal', 'sma_50', 'ema_26'
        ]
        tech_vector = np.array([self.technicals.get(k, 0.0) for k in tech_keys])

        # Company context (26 features)
        company_vector = self.company_context.to_vector()

        # Regime context (15 features)
        regime_vector = self.regime_context.to_vector()

        # Metadata (2 features)
        metadata_vector = np.array([
            self.days_until_earnings or 90.0,  # Default to 90 if unknown
            float(self.fundamental_changed_recently)
        ])

        # Normalize days_until_earnings to 0-1 (90 days = 0.5)
        metadata_vector[0] = np.clip(metadata_vector[0] / 180.0, 0, 1)

        # Concatenate all features
        return np.concatenate([tech_vector, company_vector, regime_vector, metadata_vector])

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of each feature component."""
        return {
            'technicals': 20,
            'company_context': 26,
            'regime_context': 15,
            'metadata': 2,
            'total': 63
        }


class FeatureCombiner:
    """
    Combines all three feature streams into unified vectors.

    This is the main entry point for feature extraction.
    Usage:
        combiner = FeatureCombiner()
        features = combiner.extract_features("AAPL", start_date, end_date)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the feature combiner.

        Args:
            cache_dir: Base directory for caching (default: data/cache)
        """
        # Initialize all collectors
        self.fundamental_collector = FundamentalCollector(cache_dir=cache_dir)
        self.fundamental_embedder = FundamentalEmbedding()
        self.fundamental_history = FundamentalHistoryTracker(cache_dir=cache_dir)
        self.sentiment_analyzer = NewsSentimentAnalyzer(cache_dir=cache_dir)
        self.regime_classifier = MarketRegimeClassifier()
        self.technical_calculator = TechnicalIndicatorCalculator(normalize=True)

        print("FeatureCombiner initialized with all three streams")

    def extract_features_single_point(self,
                                      symbol: str,
                                      timestamp: Optional[datetime] = None,
                                      ohlcv_data: Optional[pd.DataFrame] = None,
                                      use_cache: bool = True) -> UnifiedFeatureVector:
        """
        Extract unified features for a single point in time.

        Args:
            symbol: Stock ticker symbol
            timestamp: Target timestamp (default: now)
            ohlcv_data: Pre-fetched OHLCV data (if None, will fetch)
            use_cache: Whether to use cached data

        Returns:
            UnifiedFeatureVector with all features
        """
        if timestamp is None:
            timestamp = datetime.now()

        print(f"\nExtracting features for {symbol} at {timestamp.date()}...")

        # 1. Extract fundamentals (WHO is this company)
        print("  [1/3] Extracting fundamentals (company context)...")
        fundamentals = self.fundamental_collector.collect_fundamentals(symbol, use_cache=use_cache)
        company_context = self.fundamental_embedder.create_context(fundamentals, symbol)

        # Check if fundamentals changed recently
        fundamental_changed_recently = self._check_fundamental_change(symbol, timestamp)

        # Get days until earnings
        days_until_earnings = self.fundamental_history.get_days_until_earnings(symbol)

        # 2. Extract sentiment (WHY is price behaving this way)
        print("  [2/3] Extracting sentiment (market regime)...")
        sentiment_features = self.sentiment_analyzer.analyze(symbol, use_cache=use_cache)
        regime_context = self.regime_classifier.classify(sentiment_features)

        # 3. Extract technicals (WHAT is happening)
        print("  [3/3] Calculating technical indicators...")
        if ohlcv_data is None:
            # Fetch OHLCV data (need historical for indicators)
            import yfinance as yf
            end_date = timestamp
            start_date = timestamp - timedelta(days=365)  # 1 year for indicators

            ticker = yf.Ticker(symbol)
            ohlcv_data = ticker.history(start=start_date, end=end_date)

        # Calculate all technical indicators
        df_with_technicals = self.technical_calculator.calculate_all(ohlcv_data)

        # Get latest values (at timestamp)
        if timestamp.date() in df_with_technicals.index:
            latest_technicals = df_with_technicals.loc[timestamp.date()].to_dict()
        else:
            # Use most recent available
            latest_technicals = df_with_technicals.iloc[-1].to_dict()

        # Create unified vector
        unified = UnifiedFeatureVector(
            symbol=symbol,
            timestamp=timestamp,
            technicals=latest_technicals,
            company_context=company_context,
            regime_context=regime_context,
            days_until_earnings=days_until_earnings,
            fundamental_changed_recently=fundamental_changed_recently
        )

        print(f"  ✓ Feature extraction complete!")
        return unified

    def extract_features_timeseries(self,
                                   symbol: str,
                                   start_date: datetime,
                                   end_date: datetime,
                                   use_cache: bool = True) -> pd.DataFrame:
        """
        Extract unified features for a time series (multiple dates).

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            DataFrame with one row per date, all features as columns
        """
        print(f"\nExtracting time series features for {symbol}")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")

        # Fetch OHLCV data once for all dates
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        # Add buffer for technical indicators
        data_start = start_date - timedelta(days=365)
        ohlcv_data = ticker.history(start=data_start, end=end_date)

        if ohlcv_data.empty:
            print(f"  ✗ No OHLCV data available for {symbol}")
            return pd.DataFrame()

        # Calculate technicals once for all dates
        print("  [1/3] Calculating technical indicators...")
        df_with_technicals = self.technical_calculator.calculate_all(ohlcv_data)

        # Get fundamentals (WHO - doesn't change often)
        print("  [2/3] Extracting fundamentals...")
        fundamentals = self.fundamental_collector.collect_fundamentals(symbol, use_cache=use_cache)
        company_context = self.fundamental_embedder.create_context(fundamentals, symbol)

        # Get sentiment (WHY - changes with news)
        print("  [3/3] Extracting sentiment...")
        sentiment_features = self.sentiment_analyzer.analyze(symbol, use_cache=use_cache)
        regime_context = self.regime_classifier.classify(sentiment_features)

        # Get days until earnings
        days_until_earnings = self.fundamental_history.get_days_until_earnings(symbol)

        # Build time series
        print("  Building time series...")
        features_list = []

        # Ensure timezone consistency for date filtering
        # Convert start_date and end_date to tz-aware if df index is tz-aware
        if hasattr(df_with_technicals.index, 'tz') and df_with_technicals.index.tz is not None:
            # DataFrame has timezone, make our dates tz-aware
            import pytz
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=pytz.UTC)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=pytz.UTC)
        else:
            # DataFrame is tz-naive, make our dates tz-naive
            if start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)

        # Filter to requested date range
        df_range = df_with_technicals.loc[start_date:end_date]

        for date_idx in df_range.index:
            timestamp = pd.to_datetime(date_idx)

            # Get technical indicators for this date
            latest_technicals = df_range.loc[date_idx].to_dict()

            # Check if fundamentals changed recently (within 7 days of this date)
            fundamental_changed_recently = self._check_fundamental_change(symbol, timestamp)

            # Create unified vector
            unified = UnifiedFeatureVector(
                symbol=symbol,
                timestamp=timestamp,
                technicals=latest_technicals,
                company_context=company_context,
                regime_context=regime_context,
                days_until_earnings=days_until_earnings,
                fundamental_changed_recently=fundamental_changed_recently
            )

            # Convert to flat dictionary
            features_list.append(unified.to_feature_dict())

        # Convert to DataFrame
        df_features = pd.DataFrame(features_list)
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        df_features.set_index('timestamp', inplace=True)

        print(f"  ✓ Extracted {len(df_features)} time points")
        print(f"  ✓ Total features: {len(df_features.columns)}")

        return df_features

    def _check_fundamental_change(self, symbol: str, timestamp: datetime) -> bool:
        """
        Check if fundamentals changed recently (within 7 days).

        This is useful for identifying earnings release dates (high volatility).

        Args:
            symbol: Stock ticker
            timestamp: Target date

        Returns:
            True if fundamentals changed recently
        """
        # Get next earnings date
        earnings_date = self.fundamental_history.get_next_earnings_date(symbol)

        if earnings_date is None:
            return False

        # Ensure both are timezone-naive for comparison
        if earnings_date.tzinfo is not None:
            earnings_date = earnings_date.replace(tzinfo=None)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        # Check if within 7 days after earnings (fundamentals update period)
        days_after_earnings = (timestamp - earnings_date).days

        return 0 <= days_after_earnings <= 7

    def get_feature_summary(self, unified: UnifiedFeatureVector) -> str:
        """
        Generate human-readable summary of unified features.

        Args:
            unified: UnifiedFeatureVector

        Returns:
            String summary
        """
        summary = f"""
Unified Feature Summary for {unified.symbol}
============================================
Timestamp: {unified.timestamp.strftime('%Y-%m-%d')}

WHAT (Technicals - Price Patterns):
-----------------------------------
RSI (14): {unified.technicals.get('rsi_14', 0):.3f}
MACD: {unified.technicals.get('macd', 0):.3f}
Bollinger %B: {unified.technicals.get('bb_percent', 0):.3f}
ADX: {unified.technicals.get('adx', 0):.3f}
Volume ROC: {unified.technicals.get('volume_roc', 0):.3f}

WHO (Company Context):
----------------------
{self.fundamental_embedder.describe_context(unified.company_context)}

WHY (Market Regime):
-------------------
{self.regime_classifier.explain_regime(unified.regime_context)}

Additional Context:
------------------
Days Until Earnings: {unified.days_until_earnings or 'Unknown'}
Fundamentals Changed Recently: {'Yes' if unified.fundamental_changed_recently else 'No'}

Feature Dimensions:
------------------
{unified.get_feature_dimensions()}

Model Input Vector Shape: {unified.to_model_input().shape}
"""
        return summary


if __name__ == "__main__":
    # Quick test
    print("Testing FeatureCombiner on AAPL...")

    combiner = FeatureCombiner()

    # Test single point extraction
    print("\n" + "="*60)
    print("TEST 1: Single Point Feature Extraction")
    print("="*60)

    unified = combiner.extract_features_single_point("AAPL")
    print(combiner.get_feature_summary(unified))

    # Test model input conversion
    print("\n" + "="*60)
    print("TEST 2: Model Input Vector")
    print("="*60)

    model_input = unified.to_model_input()
    print(f"Model input shape: {model_input.shape}")
    print(f"First 10 features: {model_input[:10]}")
    print(f"Min value: {model_input.min():.3f}")
    print(f"Max value: {model_input.max():.3f}")
    print(f"Mean value: {model_input.mean():.3f}")

    # Test time series extraction (last 30 days)
    print("\n" + "="*60)
    print("TEST 3: Time Series Feature Extraction")
    print("="*60)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    df_features = combiner.extract_features_timeseries("AAPL", start_date, end_date)
    print(f"\nTime series shape: {df_features.shape}")
    print(f"Columns: {list(df_features.columns)[:10]}...")
    print(f"\nLast 5 days:")
    print(df_features[['company_growth_score', 'regime', 'regime_fear_score', 'tech_rsi_14']].tail())
