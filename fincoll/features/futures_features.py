"""
Futures Market Features - Critical for Market Context

Adds futures data to predictions for macro market awareness.
Fetches ES, NQ, VIX, CL, GC futures from TradeStation.

Features: base vector → +DIMS.fincoll_futures (futures) from central config
- Futures prices (5 × 1 = 5D): ES, NQ, VIX, CL, GC normalized
- Futures momentum (5 × 2 = 10D): 1-day and 5-day returns
- Futures volatility (5 × 2 = 10D): 5-day and 20-day rolling vol

Usage:
    extractor = FuturesFeatureExtractor()
    futures_features = extractor.extract_features()  # 25D array
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging
import threading

from config.dimensions import DIMS
from fincoll.providers.tradestation_trading_provider import TradeStationTradingProvider


class FuturesFeatureExtractor:
    """
    Extract macro futures features for market context awareness

    TradeStation-verified futures symbols:
    - @ES.D: E-mini S&P 500 (continuous)
    - @NQ.D: E-mini Nasdaq (continuous)
    - @VX: VIX futures (front month)
    - @CL: Crude Oil (front month)
    - @GC: Gold (front month)

    IMPORTANT: Uses class-level cache shared across all instances to prevent
    redundant API calls when multiple workers extract features in parallel.
    """

    FUTURES_SYMBOLS = ["@ES.D", "@NQ.D", "@VX", "@CL", "@GC"]

    # Normalization constants (approximate typical ranges)
    NORM_RANGES = {
        "@ES.D": (3000, 7000),  # S&P typically 3k-7k
        "@NQ.D": (10000, 30000),  # Nasdaq typically 10k-30k
        "@VX": (10, 40),  # VIX typically 10-40
        "@CL": (40, 120),  # Oil typically $40-$120
        "@GC": (1500, 2500),  # Gold typically $1500-$2500
    }

    # CLASS-LEVEL cache shared across all instances (thread-safe)
    _cache_lock = threading.Lock()
    _cache = {}
    _cache_time = None
    _cache_ttl = timedelta(minutes=5)

    def __init__(self, collector: Optional[TradeStationTradingProvider] = None):
        """
        Initialize futures feature extractor

        Args:
            collector: TradeStation trading provider (creates new one if not provided)
        """
        self.collector = collector or TradeStationTradingProvider()
        self.logger = logging.getLogger(__name__)

    def _fetch_futures_data(self, days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical futures data for all symbols (with shared cache)

        Args:
            days_back: Number of days of history to fetch

        Returns:
            Dict mapping symbol → DataFrame with OHLCV data

        Note:
            Uses class-level cache to prevent redundant API calls across
            parallel workers. Thread-safe with lock.
        """
        # Check cache (with lock for thread safety)
        with FuturesFeatureExtractor._cache_lock:
            if (
                FuturesFeatureExtractor._cache_time
                and datetime.now() - FuturesFeatureExtractor._cache_time
                < FuturesFeatureExtractor._cache_ttl
            ):
                cache_age = (
                    datetime.now() - FuturesFeatureExtractor._cache_time
                ).total_seconds()
                self.logger.debug(f"✓ Using cached futures data ({cache_age:.0f}s old)")
                return FuturesFeatureExtractor._cache

        # Cache miss - fetch data (one thread will fetch, others will wait)
        with FuturesFeatureExtractor._cache_lock:
            # Double-check cache after acquiring lock (another thread may have fetched)
            if (
                FuturesFeatureExtractor._cache_time
                and datetime.now() - FuturesFeatureExtractor._cache_time
                < FuturesFeatureExtractor._cache_ttl
            ):
                cache_age = (
                    datetime.now() - FuturesFeatureExtractor._cache_time
                ).total_seconds()
                self.logger.debug(
                    f"✓ Using cached futures data from other thread ({cache_age:.0f}s old)"
                )
                return FuturesFeatureExtractor._cache

            self.logger.info("Fetching fresh futures data for all symbols...")
            results = {}

            for symbol in self.FUTURES_SYMBOLS:
                try:
                    df = self.collector._get_historical_bars(
                        symbol, interval="1d", bar_count=days_back
                    )

                    if df is not None and not df.empty:
                        results[symbol] = df
                        self.logger.debug(f"  ✓ {symbol}: {len(df)} bars")
                    else:
                        self.logger.warning(f"  ✗ {symbol}: No data, using zeros")
                        results[symbol] = None

                except Exception as e:
                    self.logger.error(f"  ✗ {symbol}: Error - {e}")
                    results[symbol] = None

            # Update CLASS-LEVEL cache (not instance cache!)
            FuturesFeatureExtractor._cache = results
            FuturesFeatureExtractor._cache_time = datetime.now()
            self.logger.info(
                f"✓ Futures cache updated ({len(results)} symbols, TTL={FuturesFeatureExtractor._cache_ttl.total_seconds():.0f}s)"
            )

            return results

    def _normalize_price(self, symbol: str, price: float) -> float:
        """
        Normalize price to 0-1 range based on typical symbol range

        Args:
            symbol: Futures symbol
            price: Raw price

        Returns:
            Normalized price (0-1)
        """
        min_val, max_val = self.NORM_RANGES.get(symbol, (0, 1))

        # Clip to range and normalize
        clipped = np.clip(price, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)

        return float(normalized)

    def _calc_momentum(
        self, df: pd.DataFrame, periods: list = [1, 5]
    ) -> Dict[str, float]:
        """
        Calculate momentum (returns) for given periods

        Args:
            df: DataFrame with 'close' column
            periods: List of periods to calculate returns

        Returns:
            Dict mapping period → return percentage
        """
        if df is None or df.empty:
            return {f"{p}d": 0.0 for p in periods}

        results = {}

        for period in periods:
            if len(df) < period + 1:
                results[f"{period}d"] = 0.0
                continue

            # Calculate simple return
            current_price = float(df.iloc[-1]["close"])
            past_price = float(df.iloc[-period - 1]["close"])

            return_pct = ((current_price - past_price) / past_price) * 100
            results[f"{period}d"] = float(return_pct)

        return results

    def _calc_volatility(
        self, df: pd.DataFrame, periods: list = [5, 20]
    ) -> Dict[str, float]:
        """
        Calculate rolling volatility for given periods

        Args:
            df: DataFrame with 'close' column
            periods: List of periods for volatility calculation

        Returns:
            Dict mapping period → annualized volatility
        """
        if df is None or df.empty:
            return {f"{p}d": 0.0 for p in periods}

        results = {}

        # Calculate daily returns
        df = df.copy()
        df["returns"] = df["close"].pct_change()

        for period in periods:
            if len(df) < period:
                results[f"{period}d"] = 0.0
                continue

            # Calculate rolling standard deviation (annualized)
            volatility = df["returns"].rolling(window=period).std().iloc[-1]
            annualized_vol = volatility * np.sqrt(252)  # Trading days per year

            results[f"{period}d"] = float(annualized_vol * 100)  # As percentage

        return results

    def extract_features(self, days_back: int = 30) -> np.ndarray:
        """
        Extract all futures features

        Returns 25D feature vector:
        - Prices (5D): Normalized current prices
        - Momentum 1d (5D): 1-day returns
        - Momentum 5d (5D): 5-day returns
        - Volatility 5d (5D): 5-day rolling volatility
        - Volatility 20d (5D): 20-day rolling volatility

        Args:
            days_back: Days of history to fetch

        Returns:
            25D numpy array of futures features
        """
        futures_data = self._fetch_futures_data(days_back=days_back)

        features = []

        for symbol in self.FUTURES_SYMBOLS:
            df = futures_data.get(symbol)

            if df is None or df.empty:
                # No data - use zeros (5 features per symbol)
                features.extend([0.0] * 5)
                continue

            # 1. Normalized current price
            current_price = float(df.iloc[-1]["close"])
            norm_price = self._normalize_price(symbol, current_price)
            features.append(norm_price)

            # 2-3. Momentum (1d, 5d returns)
            momentum = self._calc_momentum(df, periods=[1, 5])
            features.append(
                momentum["1d"] / 100.0
            )  # Normalize to -1 to 1 range roughly
            features.append(momentum["5d"] / 100.0)

            # 4-5. Volatility (5d, 20d rolling vol)
            volatility = self._calc_volatility(df, periods=[5, 20])
            features.append(volatility["5d"] / 100.0)  # Normalize
            features.append(volatility["20d"] / 100.0)

        # Should have 5 symbols × 5 features = 25 total
        assert len(features) == 25, f"Expected 25 features, got {len(features)}"

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> list:
        """Get names of all futures features"""
        names = []

        for symbol in self.FUTURES_SYMBOLS:
            symbol_clean = symbol.replace("@", "").replace(".D", "")
            names.extend(
                [
                    f"futures_{symbol_clean}_price",
                    f"futures_{symbol_clean}_mom_1d",
                    f"futures_{symbol_clean}_mom_5d",
                    f"futures_{symbol_clean}_vol_5d",
                    f"futures_{symbol_clean}_vol_20d",
                ]
            )

        return names

    def get_current_snapshot(self) -> Dict:
        """
        Get human-readable snapshot of current futures state

        Returns:
            Dict with current futures prices, momentum, volatility
        """
        futures_data = self._fetch_futures_data(days_back=30)

        snapshot = {}

        for symbol in self.FUTURES_SYMBOLS:
            df = futures_data.get(symbol)

            if df is None or df.empty:
                snapshot[symbol] = {"status": "NO DATA"}
                continue

            current_price = float(df.iloc[-1]["close"])
            momentum = self._calc_momentum(df)
            volatility = self._calc_volatility(df)

            snapshot[symbol] = {
                "price": current_price,
                "price_norm": self._normalize_price(symbol, current_price),
                "momentum_1d": momentum["1d"],
                "momentum_5d": momentum["5d"],
                "volatility_5d": volatility["5d"],
                "volatility_20d": volatility["20d"],
            }

        return snapshot


if __name__ == "__main__":
    # Test futures feature extraction
    import json

    extractor = FuturesFeatureExtractor()

    print("=" * 70)
    print("FUTURES FEATURE EXTRACTION TEST")
    print("=" * 70)

    # Get features
    features = extractor.extract_features()

    print(f"\n✓ Extracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

    # Get human-readable snapshot
    snapshot = extractor.get_current_snapshot()

    print("\n" + "=" * 70)
    print("CURRENT FUTURES SNAPSHOT")
    print("=" * 70)

    for symbol, data in snapshot.items():
        print(f"\n{symbol}:")
        print(json.dumps(data, indent=2))

    # Get feature names
    feature_names = extractor.get_feature_names()
    print("\n" + "=" * 70)
    print(f"FEATURE NAMES ({len(feature_names)} total)")
    print("=" * 70)
    for i, name in enumerate(feature_names):
        print(f"{i:2d}. {name:40s} = {features[i]:8.4f}")
