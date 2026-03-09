#!/usr/bin/env python3
"""
Feature Extractor - Config-Driven Multi-Modal Features

This module extracts all features for training based on config:
- Technical (81D): Price, volume, indicators
- Advanced Technical (50D): Extended oscillators, divergence
- Velocity/Accel (20D): Momentum calculations
- News Sentiment (20D): Alpha Vantage news sentiment
- Fundamentals (16D): PE, ROE, Beta, margins (FULLY POPULATED)
- Cross-Asset (18D): SPY correlation, VIX, yields
- Sector/Industry (14D): Relative performance
- Options Flow (10D): Put/call ratio, IV
- Support/Resistance (30D): Key price levels
- VWAP (5D): Institutional flow signals
- SenVec (DIMS.senvec_total): Real-time sentiment features
- Futures (25D): Macro market context
- Finnhub Fundamentals (15D): Earnings, Insider, Analyst [NEW]

Total: DIMS.fincoll_total (dynamically loaded from feature_dimensions.yaml)
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Import centralized dimensions config
from config.dimensions import DIMS

logger = logging.getLogger(__name__)

# Import SenVec integration
try:
    from fincoll.utils.senvec_integration import (
        check_senvec_health,
        get_senvec_features,
    )

    SENVEC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SenVec integration not available: {e}")
    SENVEC_AVAILABLE = False

# Import Futures integration
try:
    from fincoll.features.futures_features import FuturesFeatureExtractor

    FUTURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Futures integration not available: {e}")
    FUTURES_AVAILABLE = False

# Import Early Signal features (momentum/acceleration/volume derivatives)
try:
    from fincoll.features.early_signal_features import extract_early_signal_features

    EARLY_SIGNAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Early signal features not available: {e}")
    EARLY_SIGNAL_AVAILABLE = False

# Import Market-Neutral features (beta-adjusted, relative strength, sector-relative)
try:
    from fincoll.features.market_neutral import extract_market_neutral_features

    MARKET_NEUTRAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Market-neutral features not available: {e}")
    MARKET_NEUTRAL_AVAILABLE = False

# Import Advanced Risk features (VaR, CVaR, downside deviation, etc.)
try:
    from fincoll.features.advanced_risk import extract_advanced_risk_features

    ADVANCED_RISK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced risk features not available: {e}")
    ADVANCED_RISK_AVAILABLE = False

# Import Momentum Variations features (Carhart 12-1, multi-horizon)
try:
    from fincoll.features.momentum_variations import (
        extract_momentum_variations_features,
    )

    MOMENTUM_VARIATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Momentum variations features not available: {e}")
    MOMENTUM_VARIATIONS_AVAILABLE = False

# Import Cross-Market features (equity indexes, futures, regime detection)
try:
    from fincoll.features.cross_market_features import CrossMarketFeatureExtractor

    CROSS_MARKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cross-market features not available: {e}")
    CROSS_MARKET_AVAILABLE = False

# Import Crypto Market features (multi-provider round-robin)
try:
    from fincoll.features.crypto_market_features import (
        MultiProviderCryptoFeatureExtractor,
    )

    CRYPTO_MARKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Crypto market features not available: {e}")
    CRYPTO_MARKET_AVAILABLE = False


class FeatureExtractor:
    """
    Extracts DIMS.fincoll_total multi-modal features from OHLCV data and external sources

    This is the core feature engineering that enables symbol-agnostic learning.
    By including news, fundamentals, cross-asset data, and real-time sentiment (SenVec),
    the model learns WHY prices move, not just THAT they moved.
    """

    # CLASS-LEVEL caches shared across all instances (thread-safe)
    # Pattern follows FuturesFeatureExtractor for maximum performance
    import threading

    _spy_cache_lock = threading.Lock()
    _spy_cache = {}  # {timestamp_bucket: (data, timestamp)}

    _news_cache_lock = threading.Lock()
    _news_cache = {}  # {cache_key: (data, timestamp)}

    _sector_etf_cache_lock = threading.Lock()
    _sector_etf_cache = {}  # {etf_timestamp_bucket: (data, timestamp)}

    _sector_cache_lock = threading.Lock()
    _sector_cache = {}  # {symbol: sector_name}

    _fundamental_cache_lock = threading.Lock()
    _fundamental_cache = {}  # {cache_key: (data, timestamp)} - Thread-safe shared cache for fundamentals

    def __init__(
        self,
        alpha_vantage_client=None,
        cache_fundamentals: bool = True,
        cache_news: bool = True,
        enable_senvec: bool = True,
        enable_futures: bool = True,
        enable_finnhub: bool = True,
        enable_market_neutral: bool = True,
        enable_advanced_risk: bool = True,
        enable_momentum_variations: bool = True,
        data_provider=None,
        enable_influxdb_storage: bool = False,
        auto_batch_threshold: int = 100,
    ):
        """
        Initialize feature extractor

        Args:
            alpha_vantage_client: AlphaVantageClient instance for news/fundamentals
            cache_fundamentals: Cache fundamental data (24-hour TTL)
            cache_news: Cache news data (5-minute TTL)
            enable_senvec: Enable SenVec features (dimension from config)
            enable_futures: Enable futures features (25D macro context)
            enable_finnhub: Enable Finnhub fundamentals (15D earnings/insider/analyst)
            enable_market_neutral: Enable market-neutral features (17D beta-adjusted, relative strength)
            enable_advanced_risk: Enable advanced risk features (8D VaR, CVaR, downside deviation)
            enable_momentum_variations: Enable momentum variations (6D Carhart, multi-horizon)
            data_provider: Data provider for SPY data (yfinance or tradestation)
            enable_influxdb_storage: Store feature vectors to InfluxDB (for training data collection)
            auto_batch_threshold: Auto-enable batch mode when processing >= this many symbols (default: 100)
        """
        self.av_client = alpha_vantage_client
        self.cache_fundamentals = cache_fundamentals
        self.cache_news = cache_news
        self.enable_senvec = enable_senvec and SENVEC_AVAILABLE
        self.enable_futures = enable_futures and FUTURES_AVAILABLE
        self.enable_finnhub = enable_finnhub
        self.enable_market_neutral = enable_market_neutral and MARKET_NEUTRAL_AVAILABLE
        self.enable_advanced_risk = enable_advanced_risk and ADVANCED_RISK_AVAILABLE
        self.enable_momentum_variations = (
            enable_momentum_variations and MOMENTUM_VARIATIONS_AVAILABLE
        )
        self.auto_batch_threshold = auto_batch_threshold

        # AUTO-INITIALIZE DATA PROVIDER IF NONE PROVIDED
        if data_provider is None:
            logger.warning(
                "⚠️  No data_provider passed to FeatureExtractor - auto-initializing MultiProviderFetcher"
            )
            try:
                from ..providers.multi_provider_fetcher import MultiProviderFetcher

                data_provider = MultiProviderFetcher()
                logger.info(
                    f"✅ Auto-initialized MultiProviderFetcher: {list(data_provider.providers.keys())}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to auto-initialize MultiProviderFetcher: {e}")
                data_provider = None
        else:
            logger.info(f"✅ Data provider passed: {type(data_provider).__name__}")

        self.data_provider = data_provider
        self.enable_influxdb_storage = enable_influxdb_storage

        # Finnhub API key
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        if self.enable_finnhub and not self.finnhub_api_key:
            logger.warning("Finnhub API key not found - Finnhub features will be zeros")
            self.enable_finnhub = False

        # Initialize futures extractor if enabled
        if self.enable_futures:
            try:
                self.futures_extractor = FuturesFeatureExtractor()
                logger.info("Futures features enabled (ES, NQ, VIX, CL, GC)")
            except Exception as e:
                logger.warning(f"Could not initialize futures extractor: {e}")
                self.enable_futures = False
                self.futures_extractor = None
        else:
            self.futures_extractor = None

        # Initialize cross-market extractor (NEW - 20D equities/futures correlations)
        self.enable_cross_market = CROSS_MARKET_AVAILABLE
        if self.enable_cross_market:
            try:
                self.cross_market_extractor = CrossMarketFeatureExtractor()
                logger.info(
                    "Cross-market features enabled (SPY, QQQ, VIX, ES, NQ - 20D)"
                )
            except Exception as e:
                logger.warning(f"Could not initialize cross-market extractor: {e}")
                self.enable_cross_market = False
                self.cross_market_extractor = None
        else:
            self.cross_market_extractor = None

        # Initialize crypto market extractor (NEW - 156D multi-provider features)
        self.enable_crypto_market = CRYPTO_MARKET_AVAILABLE
        if self.enable_crypto_market:
            try:
                self.crypto_market_extractor = MultiProviderCryptoFeatureExtractor()
                logger.info(
                    "Crypto market features enabled (CoinGecko, CryptoCompare, CoinMarketCap - 156D)"
                )
            except Exception as e:
                logger.warning(f"Could not initialize crypto market extractor: {e}")
                self.enable_crypto_market = False
                self.crypto_market_extractor = None
        else:
            self.crypto_market_extractor = None

        # Initialize InfluxDB saver if enabled
        if self.enable_influxdb_storage:
            try:
                from fincoll.storage.influxdb_saver import InfluxDBFeatureSaver

                self.influxdb_saver = InfluxDBFeatureSaver()
                logger.info(
                    f"InfluxDB storage enabled (config_version={self.influxdb_saver.config_version})"
                )
            except Exception as e:
                logger.warning(f"Could not initialize InfluxDB saver: {e}")
                self.enable_influxdb_storage = False
                self.influxdb_saver = None
        else:
            self.influxdb_saver = None

        # Instance-level caches (for senvec stale-while-revalidate)
        self._senvec_cache = {}  # {(symbol, date): (data, timestamp)} - NEW for stale-while-revalidate

        # NOTE: SPY, news, sector, sector_etf, fundamentals now use CLASS-LEVEL caches (thread-safe)
        # (see class variables above for thread-safe shared caches)

        # Batch data prefetch (Tier 2 optimization)
        # Pre-fetched batch data to avoid redundant API calls
        self._batch_senvec_data = {}  # {(symbol, date): features} - Set by prepare_batch_data()
        self._batch_fundamental_data = {}  # {(symbol, date): features} - Set by prepare_batch_data()
        self._batch_timestamp = None  # Timestamp for batch data validity
        self._batch_symbols = set()  # Track symbols in current batch for auto-batching
        self._auto_batch_enabled = False  # Flag to track if auto-batching is active

        # Load cache config from YAML (stale-while-revalidate)
        import yaml

        from config.dimensions import DIMS

        config_path = (
            Path(__file__).parent.parent / "config" / "feature_dimensions.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.cache_config = config.get("cache", {})

        # Legacy TTLs for backwards compatibility
        self.fundamental_ttl = timedelta(hours=24)
        self.news_ttl = timedelta(minutes=5)
        self.spy_ttl = timedelta(hours=4)  # SPY data cache (4-hour window)

        # Cache statistics (for monitoring)
        self.cache_stats = {
            "news": {"fresh": 0, "stale": 0, "miss": 0},
            "senvec_social": {"fresh": 0, "stale": 0, "miss": 0},
            "senvec_news": {"fresh": 0, "stale": 0, "miss": 0},
            "fundamentals": {"fresh": 0, "stale": 0, "miss": 0},
            "fundamentals": {"fresh": 0, "stale": 0, "miss": 0},
        }

        # Check SenVec availability
        # IMPORTANT: If you see all zeros in SenVec features, check:
        # 1. Redis DB 1 cache may be stale (288+ entries from before services started)
        #    Fix: cd /home/rford/caelum/ss/senvec && .venv/bin/python3 -c "import redis; r = redis.Redis(host='10.32.3.27', port=6379, db=1, decode_responses=True); r.delete(*r.keys('senvec:features:*'))"
        # 2. Backend services may not be running (senvec-alphavantage, senvec-social, senvec-news)
        #    Fix: pm2 list | grep senvec
        # 3. Aggregator may need restart to refresh backend connections
        #    Fix: pm2 restart senvec-aggregator
        # Session 2026-01-20: Discovered stale Redis cache in DB 1 causing all zeros despite healthy services
        if enable_senvec and SENVEC_AVAILABLE:
            senvec_healthy = check_senvec_health()
            if not senvec_healthy:
                logger.warning(
                    "SenVec services not fully healthy - will use zeros as fallback"
                )
        elif enable_senvec and not SENVEC_AVAILABLE:
            logger.warning("SenVec requested but not available - will use zeros")

    def get_cache_stats(self) -> dict:
        """
        Get cache hit/miss statistics for monitoring.

        Returns:
            Dictionary with cache statistics per component
        """
        stats = {}
        for component, counts in self.cache_stats.items():
            total = sum(counts.values())
            if total > 0:
                stats[component] = {
                    **counts,
                    "total": total,
                    "hit_rate": (counts["fresh"] + counts["stale"]) / total,
                    "fresh_rate": counts["fresh"] / total,
                    "stale_rate": counts["stale"] / total,
                    "miss_rate": counts["miss"] / total,
                }
        return stats

    def log_cache_stats(self):
        """Log cache statistics for monitoring"""
        stats = self.get_cache_stats()
        if stats:
            logger.info("=" * 60)
            logger.info("Cache Statistics")
            logger.info("=" * 60)
            for component, data in stats.items():
                logger.info(
                    f"{component:20s} | "
                    f"Fresh: {data['fresh']:4d} ({data['fresh_rate']:.1%}) | "
                    f"Stale: {data['stale']:4d} ({data['stale_rate']:.1%}) | "
                    f"Miss: {data['miss']:4d} ({data['miss_rate']:.1%}) | "
                    f"Hit Rate: {data['hit_rate']:.1%}"
                )
            logger.info("=" * 60)

    def _apply_decay(
        self, cached_value: np.ndarray, age_seconds: float, half_life: float
    ) -> np.ndarray:
        """
        Apply exponential decay to cached features (stale-while-revalidate).

        Sentiment doesn't instantly reset to zero - it decays over time.
        Example: Positive earnings news Monday still relevant Tuesday (decayed).

        Args:
            cached_value: Original cached feature vector
            age_seconds: How old is the cached data (seconds)
            half_life: Seconds for value to decay by 50% (None = no decay)

        Returns:
            Decayed feature vector

        Examples:
            age=0s, half_life=24h → decay=1.0 (100% of original)
            age=24h, half_life=24h → decay=0.5 (50% of original)
            age=48h, half_life=24h → decay=0.25 (25% of original)
        """
        if half_life is None or half_life == 0:
            return cached_value  # No decay (fundamentals, cross-asset)

        # Exponential decay: value = original * 0.5^(age/half_life)
        decay_factor = 0.5 ** (age_seconds / half_life)
        return cached_value * decay_factor

    def _get_cached_or_fetch(
        self,
        cache_key: str,
        cache_dict: dict,
        component: str,
        fetch_fn,
        symbol: str,
        cache_lock=None,
    ) -> tuple[np.ndarray, str]:
        """
        Three-tier cache: FRESH → STALE → EXPIRED (stale-while-revalidate).

        Args:
            cache_key: Key for this cache entry
            cache_dict: Cache dictionary to use (instance or class-level)
            component: Component name ('news', 'senvec_social', etc.)
            fetch_fn: Function to fetch fresh data (callable, no args)
            symbol: Symbol for logging
            cache_lock: Optional threading.Lock for thread-safe class-level caches

        Returns:
            (features, cache_status) where cache_status = 'fresh'|'stale'|'miss'|'error'
        """
        cache_cfg = self.cache_config.get(component, {})
        fresh_ttl = cache_cfg.get("fresh_ttl", 3600)  # Default 1 hour
        stale_ttl = cache_cfg.get("stale_ttl", 86400)  # Default 24 hours
        decay_half_life = cache_cfg.get("decay_half_life")

        # Helper function for cache check (with or without lock)
        def check_cache():
            if cache_key in cache_dict:
                cached_data, cached_time = cache_dict[cache_key]
                age_seconds = (datetime.now() - cached_time).total_seconds()
                return cached_data, cached_time, age_seconds
            return None, None, None

        # Helper function to update cache (with or without lock)
        def update_cache(features):
            cache_dict[cache_key] = (features, datetime.now())

        # Check cache (with lock if provided)
        if cache_lock:
            with cache_lock:
                cached_data, cached_time, age_seconds = check_cache()
        else:
            cached_data, cached_time, age_seconds = check_cache()

        # Tier 1: FRESH - serve immediately
        if cached_data is not None and age_seconds < fresh_ttl:
            self.cache_stats[component]["fresh"] += 1
            logger.debug(f"{symbol} {component}: FRESH cache ({age_seconds:.0f}s old)")
            return cached_data, "fresh"

        # Tier 2: STALE - serve with decay
        if cached_data is not None and age_seconds < stale_ttl:
            self.cache_stats[component]["stale"] += 1
            decayed = self._apply_decay(cached_data, age_seconds, decay_half_life)
            logger.info(
                f"{symbol} {component}: STALE cache ({age_seconds / 3600:.1f}h old, "
                f"decay={0.5 ** (age_seconds / decay_half_life) if decay_half_life else 1.0:.3f})"
            )
            # TODO: Trigger async refresh (low priority background task)
            return decayed, "stale"

        # Tier 3: EXPIRED or NO CACHE - must fetch
        try:
            # Use lock for fetch if provided (double-check pattern)
            if cache_lock:
                with cache_lock:
                    # Double-check cache after acquiring lock
                    recheck_data, recheck_time, recheck_age = check_cache()
                    if recheck_data is not None and recheck_age < fresh_ttl:
                        self.cache_stats[component]["fresh"] += 1
                        return recheck_data, "fresh"

                    # Fetch and cache in one critical section
                    self.cache_stats[component]["miss"] += 1
                    features = fetch_fn()
                    update_cache(features)
                    logger.debug(f"{symbol} {component}: FETCHED fresh data")
                    return features, "miss"
            else:
                # No lock - simple fetch
                self.cache_stats[component]["miss"] += 1
                features = fetch_fn()
                update_cache(features)
                logger.debug(f"{symbol} {component}: FETCHED fresh data")
                return features, "miss"

        except Exception as e:
            # API failed - use stale cache if available (even if expired)
            if cached_data is not None:
                age_seconds = (datetime.now() - cached_time).total_seconds()
                decayed = self._apply_decay(cached_data, age_seconds, decay_half_life)
                logger.warning(
                    f"{symbol} {component}: API FAILED, using EXPIRED cache "
                    f"({age_seconds / 3600:.1f}h old): {e}"
                )
                return decayed, "error"
            else:
                # No cache at all - let exception propagate to caller
                # Caller will handle gracefully (e.g., return zeros for Finnhub)
                logger.warning(
                    f"{symbol} {component}: API FAILED, no cache available, propagating error: {e}"
                )
                raise

    def extract_features(
        self, ohlcv_data: pd.DataFrame, symbol: str, timestamp: datetime
    ) -> np.ndarray:
        """
        Extract all features for a single bar (dimension from config)

        Args:
            ohlcv_data: DataFrame with OHLCV data (must include enough history for calculations)
            symbol: Stock ticker symbol
            timestamp: Bar timestamp

        Returns:
            np.ndarray of shape (DIMS.fincoll_total,) with all features
        """
        # VALIDATION: Ensure timestamp is a valid datetime object
        if timestamp is None or pd.isna(timestamp):
            logger.error(f"❌ extract_features received null timestamp for {symbol}")
            raise ValueError(
                f"Invalid timestamp for {symbol}: timestamp is None or NaN"
            )
        elif isinstance(timestamp, int):
            logger.error(
                f"❌ extract_features received integer timestamp: {timestamp} for {symbol}"
            )
            raise ValueError(
                f"Invalid timestamp type for {symbol}: expected datetime, got int: {timestamp}"
            )
        elif isinstance(timestamp, pd.Timestamp):
            if pd.isna(timestamp):
                logger.error(
                    f"❌ extract_features received NaT (Not a Time) for {symbol}"
                )
                raise ValueError(f"Invalid timestamp for {symbol}: timestamp is NaT")
            timestamp = timestamp.to_pydatetime()
        elif not isinstance(timestamp, datetime):
            try:
                timestamp = pd.Timestamp(timestamp).to_pydatetime()
                if pd.isna(timestamp):
                    raise ValueError("Converted to NaT")
            except Exception as e:
                logger.error(
                    f"❌ Failed to convert timestamp for {symbol}: {timestamp} (type: {type(timestamp)})"
                )
                raise ValueError(f"Invalid timestamp for {symbol}: {timestamp}")

        features = []

        # 1. Technical features (81D) - f0-f80
        technical = self._extract_technical_features(ohlcv_data)
        features.append(technical)

        # 2. Advanced technical (50D) - f81-f130
        advanced_technical = self._extract_advanced_technical(ohlcv_data)
        features.append(advanced_technical)

        # 3. Velocity/Acceleration (20D) - f131-f150
        velocity_accel = self._extract_velocity_accel(ohlcv_data)
        features.append(velocity_accel)

        # PARALLEL API CALLS (Tier 1 Optimization: 2.5x speedup)
        # Independent API calls executed in parallel to reduce latency
        from concurrent.futures import ThreadPoolExecutor

        api_features = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all independent API calls simultaneously
            futures_dict = {
                "news": executor.submit(self._extract_news_features, symbol, timestamp),
                "fundamentals": executor.submit(
                    self._extract_fundamental_features, symbol
                ),
                "senvec": executor.submit(
                    self._extract_senvec_features, symbol, timestamp
                ),
                "futures": executor.submit(self._extract_futures_features),
                "finnhub": executor.submit(self._extract_finnhub_fundamentals, symbol),
            }

            # Collect results as they complete
            for key, future in futures_dict.items():
                try:
                    api_features[key] = future.result()
                except Exception as e:
                    logger.warning(
                        f"Parallel API call '{key}' failed for {symbol}: {e}"
                    )
                    # Use zeros as fallback based on expected dimensions
                    if key == "news":
                        api_features[key] = np.zeros(20, dtype=np.float32)
                    elif key == "fundamentals":
                        api_features[key] = np.zeros(16, dtype=np.float32)
                    elif key == "senvec":
                        api_features[key] = np.zeros(
                            DIMS.senvec_total, dtype=np.float32
                        )
                    elif key == "futures":
                        api_features[key] = np.zeros(25, dtype=np.float32)
                    elif key == "finnhub":
                        api_features[key] = np.zeros(15, dtype=np.float32)

        # 4. Crypto-Specific Features (34D) - On-chain metrics, funding rates, liquidations
        # This is crypto's equivalent of fundamentals - on-chain data instead of earnings
        crypto_specific_features = self._extract_crypto_specific_features(
            symbol, timestamp, ohlcv_data
        )
        features.append(crypto_specific_features)

        # 5. Cross-Crypto Correlations (30D) - BTC dominance, market cap ratios, altcoin season
        # This is crypto's equivalent of sector analysis - cross-crypto relationships
        cross_crypto_features = self._extract_cross_crypto_features(
            symbol, timestamp, ohlcv_data
        )
        features.append(cross_crypto_features)

        # 6. Cross-Market Equities (20D) - NEW [MARKET REGIME]
        # Multi-index correlations (SPY, QQQ, DIA, IWM, VIX) + futures (ES, NQ, YM)
        # Regime detection (flight-to-crypto, risk-on/off, divergence)
        if self.enable_cross_market:
            try:
                cross_market_features = self.cross_market_extractor.extract(
                    symbol=symbol, crypto_prices=ohlcv_data, timestamp=timestamp
                )
                features.append(cross_market_features)
            except Exception as e:
                logger.warning(
                    f"Cross-market feature extraction failed for {symbol}: {e}"
                )
                features.append(
                    np.zeros(DIMS.fincoll_cross_market_equities, dtype=np.float32)
                )
        else:
            features.append(
                np.zeros(DIMS.fincoll_cross_market_equities, dtype=np.float32)
            )

        # 19. Crypto Market Extended (156D) - NEW [CRYPTO FUNDAMENTALS]
        # Multi-provider round-robin (CoinGecko, CryptoCompare, CoinMarketCap)
        # Token metadata, pools, exchanges, treasuries, categorization, fundamentals
        if self.enable_crypto_market:
            try:
                crypto_market_features = self.crypto_market_extractor.extract(
                    symbol=symbol, timestamp=timestamp
                )
                features.append(crypto_market_features)
            except Exception as e:
                logger.warning(
                    f"Crypto market feature extraction failed for {symbol}: {e}"
                )
                features.append(
                    np.zeros(DIMS.fincoll_crypto_market_extended, dtype=np.float32)
                )
        else:
            features.append(
                np.zeros(DIMS.fincoll_crypto_market_extended, dtype=np.float32)
            )

        # Concatenate all features
        all_features = np.concatenate(features)

        # DEBUG: Log feature breakdown
        if all_features.shape[0] != DIMS.fincoll_total:
            logger.error(f"\n{'=' * 80}\nDIMENSION MISMATCH DEBUG\n{'=' * 80}")
            logger.error(
                f"Expected: {DIMS.fincoll_total}D, Got: {all_features.shape[0]}D"
            )
            logger.error(f"Component count: {len(features)}")
            logger.error("Breakdown:")
            for i, f in enumerate(features, 1):
                logger.error(f"  {i:2d}. {f.shape[0]:3d}D")
            logger.error(f"Total: {sum(f.shape[0] for f in features)}D")
            logger.error("=" * 80)

        # Dimensions are validated against DIMS.fincoll_total
        assert all_features.shape[0] == DIMS.fincoll_total, (
            f"Expected {DIMS.fincoll_total} features, got {all_features.shape[0]}"
        )

        return all_features

    def extract_features_batch(
        self,
        ohlcv_data_dict: dict,
        symbols: List[str],
        timestamp: datetime,
        auto_batch: bool = True,
    ) -> dict:
        """
        Extract features for multiple symbols with automatic batch optimization.

        This method automatically enables batch data prefetching when processing
        >= auto_batch_threshold symbols (default: 100), achieving 5-10x speedup.

        Args:
            ohlcv_data_dict: Dict mapping symbol -> OHLCV DataFrame
            symbols: List of symbols to process
            timestamp: Timestamp for feature extraction
            auto_batch: Enable automatic batch mode (default: True)

        Returns:
            Dict mapping symbol -> feature vector (np.ndarray)

        Example:
            >>> extractor = FeatureExtractor()
            >>> ohlcv_dict = {sym: fetch_ohlcv(sym) for sym in symbols}
            >>> features = extractor.extract_features_batch(ohlcv_dict, symbols, datetime.now())
            >>> # Batch mode automatically enabled for >100 symbols
        """
        # Auto-enable batching if threshold met
        if auto_batch and len(symbols) >= self.auto_batch_threshold:
            self.enable_auto_batching(symbols, timestamp)
        else:
            if auto_batch:
                logger.debug(
                    f"Batch mode not activated: {len(symbols)} symbols < {self.auto_batch_threshold} threshold"
                )

        # Extract features for all symbols
        results = {}
        for symbol in symbols:
            if symbol not in ohlcv_data_dict:
                logger.warning(f"Skipping {symbol}: no OHLCV data in ohlcv_data_dict")
                continue

            try:
                features = self.extract_features(
                    ohlcv_data_dict[symbol], symbol, timestamp
                )
                results[symbol] = features
            except Exception as e:
                logger.error(f"Feature extraction failed for {symbol}: {e}")
                continue

        # Log batch performance summary
        if self._auto_batch_enabled:
            logger.info(
                f"✅ Batch extraction complete: {len(results)}/{len(symbols)} symbols processed "
                f"with batch optimization"
            )

        # Clear batch data to free memory
        if auto_batch:
            self.clear_batch_data()

        return results

    def prepare_batch_data(self, symbols: List[str], timestamp: datetime) -> None:
        """
        Pre-fetch batch data for multiple symbols (Tier 2 Optimization)

        This method fetches SenVec and fundamentals data in batch mode, which is
        significantly faster than individual API calls. The pre-fetched data is
        stored in instance variables and used by extract_features() automatically.

        Args:
            symbols: List of stock tickers to fetch data for
            timestamp: Current timestamp for SenVec features

        Usage:
            extractor.prepare_batch_data(['AAPL', 'MSFT', 'GOOGL'], datetime.now())
            features = {sym: extractor.extract_features(ohlcv[sym], sym, ts) for sym in symbols}
        """
        logger.info(f"Pre-fetching batch data for {len(symbols)} symbols...")

        # 1. Batch SenVec features (10x speedup)
        if self.enable_senvec and SENVEC_AVAILABLE:
            try:
                from fincoll.utils.senvec_integration import get_senvec_features_batch

                date_str = timestamp.strftime("%Y-%m-%d")
                senvec_batch = get_senvec_features_batch(symbols, date=date_str)

                # CRITICAL: Validate batch data before storing
                # Check AlphaVantage component (first 18D) is not all zeros
                valid_batch = {}
                invalid_symbols = []
                for sym, features in senvec_batch.items():
                    av_component = features[0:18]  # AlphaVantage critical component
                    zero_pct = (av_component == 0).sum() / 18 * 100

                    if zero_pct >= 100.0:
                        invalid_symbols.append(sym)
                        logger.warning(
                            f"  {sym}: AlphaVantage component is 100% zeros - REJECTING batch data"
                        )
                    else:
                        valid_batch[(sym, date_str)] = features

                if invalid_symbols:
                    logger.error(
                        f"⚠️  CRITICAL: {len(invalid_symbols)}/{len(symbols)} symbols have invalid "
                        f"AlphaVantage data (all zeros). Service may be down or returning stale cache."
                    )
                    logger.error(
                        f"Invalid symbols: {', '.join(invalid_symbols[:5])}"
                        f"{' ...' if len(invalid_symbols) > 5 else ''}"
                    )

                    # If >50% of symbols are invalid, BLOCK batch mode entirely
                    if len(invalid_symbols) / len(symbols) > 0.5:
                        logger.error(
                            "❌ BLOCKING BATCH MODE: >50% of symbols have invalid data. "
                            "Training will use individual API calls (slower but safer)."
                        )
                        self._batch_senvec_data = {}
                        return  # Skip batch mode

                # Store only valid batch data
                self._batch_senvec_data = valid_batch
                logger.info(
                    f"✓ Batch SenVec fetched: {len(valid_batch)} valid symbols "
                    f"({len(invalid_symbols)} rejected) for date {date_str}"
                )
            except Exception as e:
                logger.warning(f"Batch SenVec fetch failed: {e}")
                self._batch_senvec_data = {}

        # 2. Batch fundamentals via yfinance (5-10x speedup)
        if self.av_client or self.data_provider:
            try:
                import yfinance as yf

                # Fetch all tickers at once
                tickers = yf.Tickers(" ".join(symbols))

                # Extract fundamentals for each symbol
                batch_fundamentals = {}
                for symbol in symbols:
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info

                        # Extract same features as _extract_fundamental_features()
                        from data.providers.alphavantage_client import (
                            extract_fundamental_features,
                        )

                        features = extract_fundamental_features(info)
                        batch_fundamentals[symbol] = features
                    except Exception as e:
                        logger.debug(
                            f"Batch fundamental extraction failed for {symbol}: {e}"
                        )
                        batch_fundamentals[symbol] = {}

                self._batch_fundamental_data = batch_fundamentals
                logger.info(
                    f"✓ Batch fundamentals fetched: {len(batch_fundamentals)} symbols"
                )
            except Exception as e:
                logger.warning(f"Batch fundamentals fetch failed: {e}")
                self._batch_fundamental_data = {}

        # Store timestamp for validity checking
        self._batch_timestamp = timestamp

    def clear_batch_data(self) -> None:
        """Clear pre-fetched batch data to free memory"""
        self._batch_senvec_data = {}
        self._batch_fundamental_data = {}
        self._batch_timestamp = None
        self._batch_symbols = set()
        self._auto_batch_enabled = False

    def enable_auto_batching(self, symbols: List[str], timestamp: datetime) -> None:
        """
        Enable auto-batching mode for processing multiple symbols.

        This method is automatically called by extract_features() when it detects
        batch processing (>= auto_batch_threshold symbols). It can also be called
        manually to explicitly enable batching.

        Args:
            symbols: List of symbols that will be processed
            timestamp: Timestamp for the batch processing
        """
        if len(symbols) < self.auto_batch_threshold:
            logger.debug(
                f"Auto-batching skipped: {len(symbols)} symbols < {self.auto_batch_threshold} threshold"
            )
            return

        logger.info(
            f"🚀 AUTO-BATCHING ENABLED: Processing {len(symbols)} symbols "
            f"(threshold: {self.auto_batch_threshold})"
        )

        # Prepare batch data
        self.prepare_batch_data(symbols, timestamp)

        # Mark auto-batching as active
        self._auto_batch_enabled = True
        self._batch_symbols = set(symbols)

        logger.info(
            f"✅ Batch data prepared: {len(self._batch_senvec_data)} SenVec entries, "
            f"{len(self._batch_fundamental_data)} fundamental entries"
        )

    def save_features_to_influxdb(
        self,
        symbol: str,
        timestamp: datetime,
        features: np.ndarray,
        source: str = "tradestation",
    ) -> bool:
        """
        Save feature vector to InfluxDB (if enabled).

        This should be called after extract_features() to store the exact
        feature vector that was generated, preserving temporal data like
        SenVec sentiment that cannot be reconstructed later.

        Args:
            symbol: Stock ticker
            timestamp: When features were generated
            features: Feature array from extract_features()
            source: Data source (tradestation, yfinance, etc.)

        Returns:
            True if saved successfully (or storage disabled), False on error

        Example:
            features = extractor.extract_features(ohlcv, 'AAPL', datetime.now())
            extractor.save_features_to_influxdb('AAPL', datetime.now(), features)
        """
        if not self.enable_influxdb_storage or self.influxdb_saver is None:
            return True  # Not an error - storage just disabled

        try:
            return self.influxdb_saver.save_feature_vector(
                symbol=symbol, timestamp=timestamp, features=features, source=source
            )
        except Exception as e:
            logger.error(f"Failed to save features to InfluxDB: {e}")
            return False

    def _extract_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 81D technical features (f0-f80)

        Features include:
        - Price features: returns, normalized position, momentum, ranges
        - Volume features: ratios, trends, spikes
        - Technical indicators: MAs, EMAs, Bollinger bands
        - Volatility features: ATR, std dev, volatility ratios
        - Momentum features: RSI-like calculations
        - Time-based features: hour, day of week, etc.
        """
        features = []

        # Ensure we have enough data
        if len(df) < 50:
            return np.zeros(81)

        # Get current bar
        close = df["close"].iloc[-1]
        volume = df["volume"].iloc[-1]

        # Price returns (10D)
        for period in [1, 2, 3, 5, 10, 20, 30, 60, 120, 240]:
            if len(df) > period:
                ret = (close - df["close"].iloc[-period - 1]) / df["close"].iloc[
                    -period - 1
                ]
                features.append(ret)
            else:
                features.append(0.0)

        # Normalized position in recent range (5D)
        for period in [10, 20, 50, 100, 200]:
            if len(df) > period:
                recent_high = df["high"].iloc[-period:].max()
                recent_low = (
                    df["low"].iloc[-period:].min()
                )  # FIXED: was .max(), should be .min()
                if recent_high > recent_low:
                    norm_pos = (close - recent_low) / (recent_high - recent_low)
                    features.append(norm_pos)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)

        # Volume features (10D)
        avg_volume_5 = df["volume"].iloc[-5:].mean() if len(df) >= 5 else volume
        avg_volume_20 = df["volume"].iloc[-20:].mean() if len(df) >= 20 else volume
        avg_volume_50 = df["volume"].iloc[-50:].mean() if len(df) >= 50 else volume

        features.append(volume / avg_volume_5 if avg_volume_5 > 0 else 1.0)
        features.append(volume / avg_volume_20 if avg_volume_20 > 0 else 1.0)
        features.append(volume / avg_volume_50 if avg_volume_50 > 0 else 1.0)
        features.append(avg_volume_5 / avg_volume_20 if avg_volume_20 > 0 else 1.0)
        features.append(avg_volume_20 / avg_volume_50 if avg_volume_50 > 0 else 1.0)

        # Volume trend (5D)
        for period in [5, 10, 20, 30, 50]:
            if len(df) >= period:
                vol_change = (volume - df["volume"].iloc[-period]) / df["volume"].iloc[
                    -period
                ]
                features.append(vol_change)
            else:
                features.append(0.0)

        # Moving averages (10D)
        for period in [5, 10, 20, 50, 100]:
            if len(df) >= period:
                ma = df["close"].iloc[-period:].mean()
                ma_ratio = (close - ma) / ma
                features.append(ma_ratio)
                # MA slope
                if len(df) >= period + 5:
                    ma_prev = df["close"].iloc[-period - 5 : -5].mean()
                    ma_slope = (ma - ma_prev) / ma_prev
                    features.append(ma_slope)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
                features.append(0.0)

        # Bollinger bands (3D)
        if len(df) >= 20:
            ma20 = df["close"].iloc[-20:].mean()
            std20 = df["close"].iloc[-20:].std()
            upper_band = ma20 + 2 * std20
            lower_band = ma20 - 2 * std20
            bb_position = (
                (close - lower_band) / (upper_band - lower_band)
                if upper_band > lower_band
                else 0.5
            )
            bb_width = (upper_band - lower_band) / ma20
            features.append(bb_position)
            features.append(bb_width)
            features.append(
                1.0 if close > upper_band else (-1.0 if close < lower_band else 0.0)
            )
        else:
            features.extend([0.5, 0.0, 0.0])

        # ATR (3D)
        for period in [10, 20, 50]:
            if len(df) >= period + 1:
                high_low = df["high"].iloc[-period:] - df["low"].iloc[-period:]
                high_close = abs(
                    df["high"].iloc[-period:] - df["close"].shift(1).iloc[-period:]
                )
                low_close = abs(
                    df["low"].iloc[-period:] - df["close"].shift(1).iloc[-period:]
                )
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.mean()
                atr_pct = atr / close if close > 0 else 0.0
                features.append(atr_pct)
            else:
                features.append(0.0)

        # Volatility (5D)
        for period in [5, 10, 20, 50, 100]:
            if len(df) >= period:
                returns = df["close"].pct_change().iloc[-period:]
                vol = returns.std()
                features.append(vol)
            else:
                features.append(0.0)

        # Momentum indicators (10D)
        for period in [5, 10, 14, 20, 30]:
            if len(df) >= period + 1:
                # RSI-like calculation
                changes = df["close"].diff().iloc[-period:]
                gains = changes.where(changes > 0, 0.0).mean()
                losses = -changes.where(changes < 0, 0.0).mean()
                if losses != 0:
                    rs = gains / losses
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100.0
                features.append(rsi / 100.0)  # Normalize to 0-1

                # Rate of change
                roc = (close - df["close"].iloc[-period - 1]) / df["close"].iloc[
                    -period - 1
                ]
                features.append(roc)
            else:
                features.append(0.5)
                features.append(0.0)

        # Time-based features (5D)
        timestamp = (
            df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        )
        features.append(timestamp.hour / 24.0)  # Hour of day
        features.append(timestamp.weekday() / 7.0)  # Day of week
        features.append(timestamp.day / 31.0)  # Day of month
        features.append((timestamp.month - 1) / 12.0)  # Month of year
        features.append(
            1.0 if timestamp.hour < 10 or timestamp.hour > 15 else 0.0
        )  # Market open/close

        # Pad or truncate to exactly 81D
        features = features[:81]
        while len(features) < 81:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _extract_advanced_technical(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 50D advanced technical indicators (f81-f130)

        Includes:
        - MACD variations (5D)
        - Stochastic oscillators (5D)
        - Williams %R (3D)
        - CCI (Commodity Channel Index) (3D)
        - OBV (On-Balance Volume) (5D)
        - ADX (Average Directional Index) (3D)
        - Parabolic SAR (2D)
        - Ichimoku Cloud components (5D)
        - Volume-weighted indicators (10D)
        - Divergence signals (5D)
        - Mean reversion indicators (4D)
        """
        features = []

        if len(df) < 50:
            return np.zeros(50, dtype=np.float32)

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        # === MACD Variations (5D) ===
        # Standard MACD (12, 26, 9)
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd_line = ema12[-1] - ema26[-1]

        # MACD signal line (9-period EMA of MACD)
        macd_history = ema12[-26:] - ema26[-26:] if len(close) >= 26 else np.array([0])
        signal_line = self._ema(macd_history, 9)[-1] if len(macd_history) >= 9 else 0

        macd_histogram = macd_line - signal_line

        features.append(macd_line / close[-1])  # Normalized MACD
        features.append(signal_line / close[-1])  # Normalized signal
        features.append(macd_histogram / close[-1])  # Normalized histogram
        features.append(1.0 if macd_line > signal_line else -1.0)  # Bullish/bearish
        features.append(
            (macd_line - signal_line) / abs(macd_line) if macd_line != 0 else 0.0
        )  # Divergence strength

        # === Stochastic Oscillators (5D) ===
        # Fast Stochastic (14, 3)
        stoch_k, stoch_d = self._stochastic(high, low, close, k_period=14, d_period=3)
        features.append(stoch_k / 100.0)  # %K (0-100 normalized to 0-1)
        features.append(stoch_d / 100.0)  # %D (0-100 normalized to 0-1)
        features.append(
            1.0 if stoch_k > 80 else (-1.0 if stoch_k < 20 else 0.0)
        )  # Overbought/oversold
        features.append(1.0 if stoch_k > stoch_d else -1.0)  # Bullish/bearish crossover
        features.append((stoch_k - stoch_d) / 100.0)  # K-D spread

        # === Williams %R (3D) ===
        williams_r = self._williams_r(high, low, close, period=14)
        features.append(
            (williams_r + 100) / 100.0
        )  # FIXED: Normalized to true 0-1 range (was -1 to 0)
        features.append(
            1.0 if williams_r < -80 else (-1.0 if williams_r > -20 else 0.0)
        )  # Oversold/overbought
        features.append(
            (williams_r + 100) / 50.0
        )  # FIXED: Scaled to 0-2 range (was 0-2 range incorrect)

        # === CCI - Commodity Channel Index (3D) ===
        cci = self._cci(high, low, close, period=20)
        features.append(np.clip(cci / 200.0, -1, 1))  # Normalized and clipped
        features.append(
            1.0 if cci > 100 else (-1.0 if cci < -100 else 0.0)
        )  # Overbought/oversold
        features.append(cci / 100.0)  # Raw CCI scaled

        # === OBV - On-Balance Volume (5D) ===
        obv = self._obv(close, volume)
        obv_ma20 = np.mean(obv[-20:]) if len(obv) >= 20 else obv[-1]
        obv_ma50 = np.mean(obv[-50:]) if len(obv) >= 50 else obv[-1]

        features.append(
            (obv[-1] - obv_ma20) / obv_ma20 if obv_ma20 != 0 else 0.0
        )  # OBV vs 20-MA
        features.append(
            (obv[-1] - obv_ma50) / obv_ma50 if obv_ma50 != 0 else 0.0
        )  # OBV vs 50-MA
        features.append(
            (obv_ma20 - obv_ma50) / obv_ma50 if obv_ma50 != 0 else 0.0
        )  # OBV MA divergence

        # OBV trend
        obv_slope = (obv[-1] - obv[-10]) / 10 if len(obv) >= 10 else 0
        features.append(obv_slope / abs(obv[-1]) if obv[-1] != 0 else 0.0)
        features.append(1.0 if obv_slope > 0 else -1.0)  # OBV direction

        # === ADX - Average Directional Index (3D) ===
        adx, plus_di, minus_di = self._adx(high, low, close, period=14)
        features.append(adx / 100.0)  # ADX strength (0-100 normalized)
        features.append(plus_di / 100.0 - minus_di / 100.0)  # DI spread
        features.append(1.0 if plus_di > minus_di else -1.0)  # Trend direction

        # === Parabolic SAR (2D) ===
        sar = self._parabolic_sar(high, low, close)
        features.append(1.0 if close[-1] > sar else -1.0)  # Above/below SAR
        features.append((close[-1] - sar) / close[-1])  # Distance to SAR

        # === Ichimoku Cloud Components (5D) ===
        tenkan, kijun, senkou_a, senkou_b = self._ichimoku(high, low, close)
        features.append((close[-1] - senkou_a) / close[-1])  # Distance to Senkou A
        features.append((close[-1] - senkou_b) / close[-1])  # Distance to Senkou B
        features.append(
            1.0 if close[-1] > max(senkou_a, senkou_b) else -1.0
        )  # Above/below cloud
        features.append((tenkan - kijun) / close[-1])  # TK spread
        features.append(1.0 if tenkan > kijun else -1.0)  # Bullish/bearish TK cross

        # === Volume-Weighted Indicators (10D) ===
        # VWAP
        vwap = self._vwap(high, low, close, volume)
        features.append((close[-1] - vwap) / vwap)  # Distance to VWAP
        features.append(1.0 if close[-1] > vwap else -1.0)  # Above/below VWAP

        # Money Flow Index (MFI)
        mfi = self._mfi(high, low, close, volume, period=14)
        features.append(mfi / 100.0)  # Normalized MFI
        features.append(
            1.0 if mfi > 80 else (-1.0 if mfi < 20 else 0.0)
        )  # Overbought/oversold

        # Volume Price Trend (VPT)
        vpt = self._vpt(close, volume)
        vpt_ma = np.mean(vpt[-20:]) if len(vpt) >= 20 else vpt[-1]
        features.append((vpt[-1] - vpt_ma) / abs(vpt_ma) if vpt_ma != 0 else 0.0)

        # Chaikin Money Flow (CMF)
        cmf = self._cmf(high, low, close, volume, period=20)
        features.append(cmf)  # Already normalized around 0
        features.append(
            1.0 if cmf > 0.1 else (-1.0 if cmf < -0.1 else 0.0)
        )  # Accumulation/distribution

        # Accumulation/Distribution Line (ADL)
        adl = self._adl(high, low, close, volume)
        adl_slope = (adl[-1] - adl[-10]) / 10 if len(adl) >= 10 else 0
        features.append(adl_slope / abs(adl[-1]) if adl[-1] != 0 else 0.0)
        features.append(1.0 if adl_slope > 0 else -1.0)

        # Price Volume Trend
        features.append(
            (vpt[-1] - vpt[-10]) / 10 / abs(vpt[-1])
            if len(vpt) >= 10 and vpt[-1] != 0
            else 0.0
        )

        # === Divergence Signals (5D) ===
        # Price-RSI divergence (calculated earlier in technical features, so derive from price)
        price_slope = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

        # Price making higher highs but indicator making lower highs (bearish divergence)
        recent_high = np.max(high[-20:]) if len(high) >= 20 else high[-1]
        features.append(
            1.0 if close[-1] >= recent_high * 0.98 else 0.0
        )  # Near recent high

        # Price-volume divergence
        vol_slope = (
            (volume[-1] - np.mean(volume[-20:])) / np.mean(volume[-20:])
            if len(volume) >= 20
            else 0
        )
        features.append(
            1.0 if price_slope > 0 and vol_slope < 0 else 0.0
        )  # Bearish divergence
        features.append(
            1.0 if price_slope < 0 and vol_slope > 0 else 0.0
        )  # Bullish divergence (sell climax)

        # MACD-Price divergence
        features.append(
            1.0 if price_slope > 0 and macd_histogram < 0 else 0.0
        )  # Bearish divergence
        features.append(
            1.0 if price_slope < 0 and macd_histogram > 0 else 0.0
        )  # Bullish divergence

        # === Mean Reversion Indicators (4D) ===
        # Bollinger Band position (already in technical, but add reversal signals)
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(
            close, period=20, num_std=2
        )
        bb_position = (
            (close[-1] - bb_lower) / (bb_upper - bb_lower)
            if bb_upper > bb_lower
            else 0.5
        )

        features.append(
            1.0 if bb_position > 0.95 else 0.0
        )  # Near upper band (potential reversal)
        features.append(
            1.0 if bb_position < 0.05 else 0.0
        )  # Near lower band (potential bounce)

        # Z-Score for mean reversion
        z_score = (
            (close[-1] - np.mean(close[-20:])) / np.std(close[-20:])
            if len(close) >= 20 and np.std(close[-20:]) > 0
            else 0
        )
        features.append(z_score / 3.0)  # Normalized (3 std devs = extreme)
        features.append(1.0 if abs(z_score) > 2 else 0.0)  # Extreme deviation signal

        # Pad or truncate to exactly 50D
        features = features[:50]
        while len(features) < 50:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    # Helper functions for advanced technical indicators

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([np.mean(data)])

        ema = np.zeros(len(data))
        ema[0] = data[0]
        multiplier = 2.0 / (period + 1)

        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    def _stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator (%K and %D)"""
        if len(close) < k_period:
            return 50.0, 50.0

        lowest_low = np.min(low[-k_period:])
        highest_high = np.max(high[-k_period:])

        if highest_high == lowest_low:
            k = 50.0
        else:
            k = 100.0 * (close[-1] - lowest_low) / (highest_high - lowest_low)

        # %D is typically 3-period SMA of %K (simplified here)
        d = k  # Simplified - in full implementation, would calculate from recent %K values

        return k, d

    def _williams_r(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """Calculate Williams %R"""
        if len(close) < period:
            return -50.0

        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])

        if highest_high == lowest_low:
            return -50.0

        williams = -100.0 * (highest_high - close[-1]) / (highest_high - lowest_low)
        return williams

    def _cci(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20
    ) -> float:
        """Calculate Commodity Channel Index"""
        if len(close) < period:
            return 0.0

        typical_price = (high[-period:] + low[-period:] + close[-period:]) / 3.0
        sma_tp = np.mean(typical_price)
        mean_deviation = np.mean(np.abs(typical_price - sma_tp))

        if mean_deviation == 0:
            return 0.0

        cci = (typical_price[-1] - sma_tp) / (0.015 * mean_deviation)
        return cci

    def _obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(close))
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return obv

    def _adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> Tuple[float, float, float]:
        """Calculate Average Directional Index (ADX) and Directional Indicators"""
        if len(close) < period + 1:
            return 0.0, 0.0, 0.0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed TR and DM
        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0

        # ADX
        dx = (
            100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            if (plus_di + minus_di) > 0
            else 0
        )
        adx = dx  # Simplified - full implementation would smooth this

        return adx, plus_di, minus_di

    def _parabolic_sar(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> float:
        """Calculate Parabolic SAR (simplified)"""
        if len(close) < 5:
            return close[-1]

        # Simplified SAR calculation
        # In full implementation, this would track trend direction and acceleration
        recent_high = np.max(high[-5:])
        recent_low = np.min(low[-5:])

        if close[-1] > (recent_high + recent_low) / 2:
            # Uptrend - SAR below price
            return recent_low
        else:
            # Downtrend - SAR above price
            return recent_high

    def _ichimoku(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Calculate Ichimoku Cloud components"""
        if len(close) < 52:
            return close[-1], close[-1], close[-1], close[-1]

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2

        # Senkou Span A: (Tenkan + Kijun) / 2 projected 26 periods ahead
        senkou_a = (tenkan + kijun) / 2

        # Senkou Span B: (52-period high + 52-period low) / 2 projected 26 periods ahead
        senkou_b = (np.max(high[-52:]) + np.min(low[-52:])) / 2

        return tenkan, kijun, senkou_a, senkou_b

    def _vwap(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> float:
        """Calculate Volume Weighted Average Price"""
        if len(close) < 1:
            return close[-1] if len(close) > 0 else 0

        typical_price = (high + low + close) / 3.0
        vwap = (
            np.sum(typical_price * volume) / np.sum(volume)
            if np.sum(volume) > 0
            else close[-1]
        )

        return vwap

    def _mfi(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate Money Flow Index"""
        if len(close) < period + 1:
            return 50.0

        typical_price = (
            high[-period - 1 :] + low[-period - 1 :] + close[-period - 1 :]
        ) / 3.0
        money_flow = typical_price * volume[-period - 1 :]

        positive_flow = np.sum(money_flow[1:][typical_price[1:] > typical_price[:-1]])
        negative_flow = np.sum(money_flow[1:][typical_price[1:] < typical_price[:-1]])

        if negative_flow == 0:
            return 100.0

        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    def _vpt(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Price Trend"""
        vpt = np.zeros(len(close))
        vpt[0] = volume[0]

        for i in range(1, len(close)):
            vpt[i] = (
                vpt[i - 1] + volume[i] * (close[i] - close[i - 1]) / close[i - 1]
                if close[i - 1] != 0
                else vpt[i - 1]
            )

        return vpt

    def _cmf(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 20,
    ) -> float:
        """Calculate Chaikin Money Flow"""
        if len(close) < period:
            return 0.0

        money_flow_multiplier = (
            (close[-period:] - low[-period:]) - (high[-period:] - close[-period:])
        ) / (high[-period:] - low[-period:])
        money_flow_multiplier = np.where(
            high[-period:] == low[-period:], 0, money_flow_multiplier
        )

        money_flow_volume = money_flow_multiplier * volume[-period:]
        cmf = (
            np.sum(money_flow_volume) / np.sum(volume[-period:])
            if np.sum(volume[-period:]) > 0
            else 0
        )

        return cmf

    def _adl(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """Calculate Accumulation/Distribution Line"""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_multiplier = np.where(high == low, 0, money_flow_multiplier)

        money_flow_volume = money_flow_multiplier * volume
        adl = np.cumsum(money_flow_volume)

        return adl

    def _bollinger_bands(
        self, close: np.ndarray, period: int = 20, num_std: int = 2
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(close) < period:
            return close[-1], close[-1], close[-1]

        middle = np.mean(close[-period:])
        std = np.std(close[-period:])

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return upper, middle, lower

    def _extract_velocity_accel(self, df: pd.DataFrame) -> np.ndarray:
        """Extract 20D velocity/acceleration features (f131-f150)"""
        features = []

        if len(df) < 5:
            return np.zeros(20, dtype=np.float32)

        # Price velocity (3-point, 5-point)
        prices = df["close"].values
        if len(prices) >= 3:
            vel_3pt = (prices[-1] - prices[-3]) / 2
            features.append(vel_3pt / prices[-1])
        else:
            features.append(0.0)

        if len(prices) >= 5:
            vel_5pt = (prices[-1] - prices[-5]) / 4
            features.append(vel_5pt / prices[-1])
        else:
            features.append(0.0)

        # Volume velocity
        volumes = df["volume"].values
        if len(volumes) >= 3:
            vol_vel = (volumes[-1] - volumes[-3]) / 2
            avg_vol = volumes[-3:].mean()
            features.append(vol_vel / avg_vol if avg_vol > 0 else 0.0)
        else:
            features.append(0.0)

        # Price acceleration
        if len(prices) >= 4:
            accel = prices[-1] - 2 * prices[-2] + prices[-3]
            features.append(accel / prices[-1])
        else:
            features.append(0.0)

        # Momentum calculations (10-bar, 20-bar, 30-bar)
        for period in [10, 20, 30]:
            if len(df) >= period:
                momentum = (prices[-1] - prices[-period]) / prices[-period]
                features.append(momentum)
            else:
                features.append(0.0)

        # Rate of change
        for period in [10, 20]:
            if len(df) >= period:
                roc = (prices[-1] - prices[-period]) / prices[-period]
                features.append(roc)
            else:
                features.append(0.0)

        # Pad to 20D
        while len(features) < 20:
            features.append(0.0)

        return np.array(features[:20], dtype=np.float32)

    def _extract_news_features(self, symbol: str, timestamp) -> np.ndarray:
        """
        Extract 20D news sentiment features (f151-f170).
        Delegates to senvec-news-internal (port 18005) which handles caching,
        dual-source collection (Alpha Vantage 10D + FinLight 10D), and MongoDB storage.

        Args:
            symbol: Stock symbol
            timestamp: Unused (news-internal manages its own freshness)

        Returns:
            20D news sentiment features (f151-f170) as float32 np.ndarray
        """
        try:
            resp = requests.get(
                f"http://10.32.3.27:18005/features/{symbol}",
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            arr = np.array(features, dtype=np.float32)
            if arr.shape != (20,):
                logger.warning(
                    f"{symbol} news-internal returned {arr.shape}, padding to 20D"
                )
                arr = np.pad(arr, (0, max(0, 20 - len(arr))))[:20].astype(np.float32)
            return arr
        except Exception as e:
            logger.error(f"{symbol} news-internal (18005) error: {e} — returning zeros")
            return np.zeros(20, dtype=np.float32)

    def _extract_fundamental_features(self, symbol: str) -> np.ndarray:
        """
        Extract 16D fundamental features (f171-f186)

        Data sources:
        - Alpha Vantage: PE, PB, Dividend Yield, Profit Margin, ROE, Earnings/Revenue Growth, Beta
        - Finnhub: PEG, P/S, EV/Revenue, Operating Margin, ROA, Current Ratio, Debt-to-Equity, Asset Turnover

        ALL SLOTS NOW POPULATED (no more TODOs)

        Caching strategy (from YAML config):
        - FRESH (0-24h): Serve cached, no API call (fundamentals don't change daily)
        - STALE (24h-30d): Serve cached with NO decay (fundamentals persist)
        - EXPIRED (>30d): Must fetch fresh or fail
        """
        # TIER 2 OPTIMIZATION: Check for pre-fetched batch data
        if symbol in self._batch_fundamental_data:
            if self._auto_batch_enabled:
                logger.debug(
                    f"🚀 {symbol} Fundamentals: Using AUTO-BATCH data (5-10x speedup)"
                )
            else:
                logger.debug(f"{symbol} Fundamentals: Using BATCH data (5-10x speedup)")
            av_features = self._batch_fundamental_data[symbol]
            # Skip to feature assembly using batch data
            finnhub_metrics = self._get_finnhub_basic_metrics(symbol)
            features = np.array(
                [
                    # Valuation (6D)
                    av_features.get("pe_ratio", 0.0),  # f171 Alpha Vantage
                    av_features.get("pb_ratio", 0.0),  # f172 Alpha Vantage
                    av_features.get("dividend_yield", 0.0),  # f173 Alpha Vantage
                    finnhub_metrics.get("pegTTM", 0.0),  # f174 Finnhub
                    finnhub_metrics.get("psTTM", 0.0),  # f175 Finnhub
                    finnhub_metrics.get("evRevenueTTM", 0.0),  # f176 Finnhub
                    # Profitability (4D)
                    av_features.get("profit_margin", 0.0),  # f177 Alpha Vantage
                    finnhub_metrics.get("operatingMarginTTM", 0.0),  # f178 Finnhub
                    finnhub_metrics.get("roaTTM", 0.0),  # f179 Finnhub
                    av_features.get("roe", 0.0),  # f180 Alpha Vantage
                    # Growth (2D)
                    av_features.get("earnings_growth", 0.0),  # f181 Alpha Vantage
                    av_features.get("revenue_growth", 0.0),  # f182 Alpha Vantage
                    # Financial Health (2D)
                    finnhub_metrics.get("currentRatioQuarterly", 0.0),  # f183 Finnhub
                    finnhub_metrics.get(
                        "longTermDebt/equityQuarterly", 0.0
                    ),  # f184 Finnhub
                    # Other (2D)
                    av_features.get("beta", 1.0),  # f185 Alpha Vantage
                    finnhub_metrics.get("assetTurnoverTTM", 0.0),  # f186 Finnhub
                ],
                dtype=np.float32,
            )
            return features

        # Get Alpha Vantage fundamentals using stale-while-revalidate cache
        av_features = {}
        if self.av_client:

            def fetch_av_fundamentals():
                """Fetch fresh fundamental data from Alpha Vantage"""
                from data.providers.alphavantage_client import (
                    extract_fundamental_features,
                )

                overview = self.av_client.get_company_overview(symbol)
                return extract_fundamental_features(overview)

            try:
                cache_key = f"av_{symbol}"
                av_features, cache_status = self._get_cached_or_fetch(
                    cache_key=cache_key,
                    cache_dict=FeatureExtractor._fundamental_cache,
                    component="fundamentals",
                    fetch_fn=fetch_av_fundamentals,
                    symbol=symbol,
                    cache_lock=FeatureExtractor._fundamental_cache_lock,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get Alpha Vantage fundamentals for {symbol}: {e}"
                )
                av_features = {}

        # Get Finnhub metrics to fill gaps
        finnhub_metrics = self._get_finnhub_basic_metrics(symbol)

        # Convert to array (16D - FULLY POPULATED)
        features = np.array(
            [
                # Valuation (6D)
                av_features.get("pe_ratio", 0.0),  # f171 Alpha Vantage
                av_features.get("pb_ratio", 0.0),  # f172 Alpha Vantage
                av_features.get("dividend_yield", 0.0),  # f173 Alpha Vantage
                finnhub_metrics.get("pegTTM", 0.0),  # f174 Finnhub ✅ FILLED
                finnhub_metrics.get("psTTM", 0.0),  # f175 Finnhub ✅ FILLED
                finnhub_metrics.get("evRevenueTTM", 0.0),  # f176 Finnhub ✅ FILLED
                # Profitability (4D)
                av_features.get("profit_margin", 0.0),  # f177 Alpha Vantage
                finnhub_metrics.get(
                    "operatingMarginTTM", 0.0
                ),  # f178 Finnhub ✅ FILLED
                finnhub_metrics.get("roaTTM", 0.0),  # f179 Finnhub ✅ FILLED
                av_features.get("roe", 0.0),  # f180 Alpha Vantage
                # Growth (2D)
                av_features.get("earnings_growth", 0.0),  # f181 Alpha Vantage
                av_features.get("revenue_growth", 0.0),  # f182 Alpha Vantage
                # Financial Health (2D)
                finnhub_metrics.get(
                    "currentRatioQuarterly", 0.0
                ),  # f183 Finnhub ✅ FILLED
                finnhub_metrics.get(
                    "longTermDebt/equityQuarterly", 0.0
                ),  # f184 Finnhub ✅ FILLED
                # Other (2D)
                av_features.get("beta", 1.0),  # f185 Alpha Vantage
                finnhub_metrics.get(
                    "assetTurnoverTTM", 0.0
                ),  # f186 Finnhub ✅ FILLED (was Reserved)
            ],
            dtype=np.float32,
        )

        return features

    def _extract_cross_asset_features(
        self, timestamp: datetime, ohlcv_data: pd.DataFrame, symbol: str, beta: float
    ) -> np.ndarray:
        """
        Extract 18D cross-asset signals (f187-f204) - V6 Beta-Adjusted Residual Momentum

        Implements residual momentum strategy:
        - Expected Return: E[R] = beta × R_market
        - Residual: ε = R_stock - E[R] (deviation from beta expectation)
        - Z-Score: Z = ε / σ_ε (outlier detection)
        - Acceleration: Second derivative of residual (breaking away signal)

        Args:
            timestamp: Current bar timestamp
            ohlcv_data: Stock OHLCV data
            symbol: Stock ticker
            beta: Stock beta from fundamentals

        Returns:
            18D feature array:
            - 1-day horizon (6D): f187-f192
            - 5-day horizon (6D): f193-f198
            - 20-day horizon (6D): f199-f204
        """
        try:
            features = []

            # Get SPY data with caching
            spy_data = self._get_spy_data(timestamp, lookback=60)
            if spy_data is None or len(spy_data) < 20:
                logger.warning(f"Insufficient SPY data for {timestamp}, using zeros")
                return np.zeros(18, dtype=np.float32)

            # Calculate for multiple horizons (1d, 5d, 20d)
            horizons = [1, 5, 20]

            for horizon in horizons:
                # Calculate returns
                if len(ohlcv_data) < horizon + 5 or len(spy_data) < horizon + 5:
                    # Not enough data for this horizon
                    features.extend([0.0] * 6)
                    continue

                # Market return (SPY)
                spy_return = (
                    spy_data["close"].iloc[-1] / spy_data["close"].iloc[-horizon - 1]
                ) - 1.0
                features.append(spy_return)

                # Stock return
                stock_return = (
                    ohlcv_data["close"].iloc[-1]
                    / ohlcv_data["close"].iloc[-horizon - 1]
                ) - 1.0

                # Expected return based on beta
                expected_return = beta * spy_return

                # Residual (deviation from beta expectation)
                residual = stock_return - expected_return
                features.append(residual)

                # Calculate residual history for Z-score and derivatives
                residual_history = self._calculate_residual_history(
                    symbol, ohlcv_data, spy_data, beta, horizon, lookback=60
                )

                # Z-score of residual (outlier detection)
                if len(residual_history) >= 2:
                    residual_std = np.std(residual_history)
                    residual_zscore = (
                        residual / residual_std if residual_std > 1e-8 else 0.0
                    )
                    features.append(residual_zscore)
                else:
                    features.append(0.0)

                # Residual velocity (first derivative)
                if len(residual_history) >= 2:
                    residual_velocity = residual - residual_history[-2]
                    features.append(residual_velocity)
                else:
                    features.append(0.0)

                # Residual acceleration (second derivative) - KEY SIGNAL
                if len(residual_history) >= 3:
                    prev_velocity = residual_history[-2] - residual_history[-3]
                    residual_accel = (
                        residual_velocity - prev_velocity
                        if len(residual_history) >= 2
                        else 0.0
                    )
                    features.append(residual_accel)
                else:
                    features.append(0.0)

                # Residual jerk (third derivative) - regime shift detector
                if len(residual_history) >= 4:
                    prev_accel = (residual_history[-2] - residual_history[-3]) - (
                        residual_history[-3] - residual_history[-4]
                    )
                    residual_jerk = (
                        residual_accel - prev_accel
                        if len(residual_history) >= 3
                        else 0.0
                    )
                    features.append(residual_jerk)
                else:
                    features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract cross-asset features for {symbol}: {e}")
            return np.zeros(18, dtype=np.float32)

    def _get_spy_data(
        self, timestamp: datetime, lookback: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Fetch SPY data with caching

        Args:
            timestamp: Current timestamp (datetime or pandas Timestamp or int)
            lookback: Number of days of history to fetch

        Returns:
            DataFrame with SPY OHLCV data or None if unavailable
        """
        try:
            # Convert timestamp to datetime if it's an int or pandas Timestamp
            if timestamp is None or pd.isna(timestamp):
                logger.error(f"❌ _get_spy_data received null/NaN timestamp")
                raise ValueError("Invalid timestamp: timestamp is None or NaN")
            elif isinstance(timestamp, int):
                # BUGFIX: Integers should NOT be interpreted as nanoseconds
                # If we receive an int, it's likely a mistake - log and reject
                logger.error(
                    f"❌ _get_spy_data received integer timestamp: {timestamp}. Expected datetime object!"
                )
                raise ValueError(
                    f"Invalid timestamp type: {type(timestamp)}. Expected datetime, got int: {timestamp}"
                )
            elif isinstance(timestamp, pd.Timestamp):
                if pd.isna(timestamp):
                    logger.error(f"❌ _get_spy_data received NaT (Not a Time)")
                    raise ValueError("Invalid timestamp: timestamp is NaT")
                timestamp = timestamp.to_pydatetime()
            elif isinstance(timestamp, str):
                # Convert string timestamp to datetime
                try:
                    timestamp = pd.to_datetime(timestamp).to_pydatetime()
                    if pd.isna(timestamp):
                        raise ValueError("Converted to NaT")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to convert string timestamp to datetime: {timestamp}"
                    )
                    raise ValueError(f"Invalid timestamp string: {timestamp}")
            elif not isinstance(timestamp, datetime):
                # Try to convert other types to datetime
                try:
                    timestamp = pd.Timestamp(timestamp).to_pydatetime()
                    if pd.isna(timestamp):
                        raise ValueError("Converted to NaT")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to convert timestamp to datetime: {timestamp} (type: {type(timestamp)})"
                    )
                    raise ValueError(f"Invalid timestamp: {timestamp}")

            # Final type check to ensure timestamp is datetime before date arithmetic
            if not isinstance(timestamp, datetime):
                logger.error(
                    f"❌ Timestamp conversion failed: {timestamp} (type: {type(timestamp)})"
                )
                raise ValueError(
                    f"Timestamp must be datetime object, got {type(timestamp)}"
                )

            # Check cache (bucket by hour to allow reuse)
            cache_key = pd.Timestamp(timestamp).strftime("%Y-%m-%d-%H")

            # Check CLASS-LEVEL cache with lock (thread-safe)
            with FeatureExtractor._spy_cache_lock:
                if cache_key in FeatureExtractor._spy_cache:
                    cached_data, cached_time = FeatureExtractor._spy_cache[cache_key]
                    if datetime.now() - cached_time < self.spy_ttl:
                        return cached_data

            # Cache miss - fetch data with lock (one thread fetches, others wait)
            with FeatureExtractor._spy_cache_lock:
                # Double-check cache after acquiring lock (another thread may have fetched)
                if cache_key in FeatureExtractor._spy_cache:
                    cached_data, cached_time = FeatureExtractor._spy_cache[cache_key]
                    if datetime.now() - cached_time < self.spy_ttl:
                        return cached_data

                # Fetch SPY data
                if self.data_provider is None:
                    logger.warning("No data provider configured for SPY data")
                    return None

                # Calculate date range - ensure datetime objects for arithmetic
                end_date = pd.to_datetime(timestamp).to_pydatetime()
                start_date = pd.to_datetime(timestamp).to_pydatetime() - timedelta(
                    days=lookback + 30
                )  # Extra buffer for market holidays

                # Fetch from provider (pass datetime objects, not strings)
                spy_data = self.data_provider.get_historical_bars(
                    symbol="SPY",
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                )

                if spy_data is None or len(spy_data) == 0:
                    logger.debug(f"No SPY data available for {timestamp}")
                    return None

                # Cache result in CLASS-LEVEL cache
                FeatureExtractor._spy_cache[cache_key] = (spy_data, datetime.now())

                return spy_data

        except Exception as e:
            logger.warning(f"Failed to fetch SPY data: {e}")
            return None

    def _calculate_residual_history(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        beta: float,
        horizon: int,
        lookback: int = 60,
    ) -> List[float]:
        """
        Calculate historical residuals for Z-score and derivative calculations

        Args:
            symbol: Stock ticker
            ohlcv_data: Stock OHLCV data
            spy_data: SPY OHLCV data
            beta: Stock beta
            horizon: Return horizon (1, 5, or 20 days)
            lookback: Number of historical residuals to calculate

        Returns:
            List of historical residuals (oldest to newest)
        """
        try:
            residuals = []

            # Calculate residuals for each historical point
            max_lookback = min(
                lookback, len(ohlcv_data) - horizon - 1, len(spy_data) - horizon - 1
            )

            for i in range(max_lookback, 0, -1):
                # Stock return at time t-i
                stock_return = (
                    ohlcv_data["close"].iloc[-i]
                    / ohlcv_data["close"].iloc[-i - horizon]
                ) - 1.0

                # Market return at time t-i
                spy_return = (
                    spy_data["close"].iloc[-i] / spy_data["close"].iloc[-i - horizon]
                ) - 1.0

                # Expected return
                expected_return = beta * spy_return

                # Residual
                residual = stock_return - expected_return
                residuals.append(residual)

            return residuals

        except Exception as e:
            logger.warning(f"Failed to calculate residual history for {symbol}: {e}")
            return []

    def _get_symbol_sector(self, symbol: str) -> str:
        """
        Get sector classification for a symbol using yfinance

        Args:
            symbol: Stock ticker

        Returns:
            Sector name or empty string if unavailable
        """
        try:
            # Check CLASS-LEVEL cache first (thread-safe)
            with FeatureExtractor._sector_cache_lock:
                if symbol in FeatureExtractor._sector_cache:
                    return FeatureExtractor._sector_cache[symbol]

            # Fetch from yfinance (with lock to prevent duplicate fetches)
            with FeatureExtractor._sector_cache_lock:
                # Double-check after acquiring lock
                if symbol in FeatureExtractor._sector_cache:
                    return FeatureExtractor._sector_cache[symbol]

                import yfinance as yf

                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Get sector (use GICS sector classification)
                sector = info.get("sector", "")

                # Cache result in CLASS-LEVEL cache
                FeatureExtractor._sector_cache[symbol] = sector

                return sector

        except Exception as e:
            logger.warning(f"Failed to get sector for {symbol}: {e}")
            return ""

    def _get_sector_etf_data(
        self, etf_symbol: str, timestamp: datetime, lookback: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch sector ETF data with caching (similar to SPY data fetch)

        Args:
            etf_symbol: Sector ETF ticker (XLK, XLF, etc.)
            timestamp: Current timestamp (datetime or pandas Timestamp or int)
            lookback: Number of days of history to fetch

        Returns:
            DataFrame with sector ETF OHLCV data or None if unavailable
        """
        try:
            # Convert timestamp to datetime if it's an int or pandas Timestamp
            if timestamp is None or pd.isna(timestamp):
                logger.error(
                    f"❌ _get_sector_etf_data received null/NaN timestamp for {etf_symbol}"
                )
                return None
            elif isinstance(timestamp, int):
                logger.error(
                    f"❌ _get_sector_etf_data received integer timestamp: {timestamp}"
                )
                return None
            elif isinstance(timestamp, pd.Timestamp):
                if pd.isna(timestamp):
                    logger.error(
                        f"❌ _get_sector_etf_data received NaT for {etf_symbol}"
                    )
                    return None
                timestamp = timestamp.to_pydatetime()
            elif isinstance(timestamp, str):
                # Convert string timestamp to datetime
                try:
                    timestamp = pd.to_datetime(timestamp).to_pydatetime()
                    if pd.isna(timestamp):
                        raise ValueError("Converted to NaT")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to convert string timestamp for {etf_symbol}: {timestamp}"
                    )
                    return None
            elif not isinstance(timestamp, datetime):
                try:
                    timestamp = pd.Timestamp(timestamp).to_pydatetime()
                    if pd.isna(timestamp):
                        raise ValueError("Converted to NaT")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to convert timestamp for {etf_symbol}: {timestamp} (type: {type(timestamp)})"
                    )
                    return None

            # Final type check
            if not isinstance(timestamp, datetime):
                logger.error(
                    f"❌ Timestamp conversion failed for {etf_symbol}: {timestamp} (type: {type(timestamp)})"
                )
                return None

            # Check CLASS-LEVEL cache (bucket by hour to allow reuse, thread-safe)
            cache_key = (
                f"{etf_symbol}_{pd.Timestamp(timestamp).strftime('%Y-%m-%d-%H')}"
            )

            with FeatureExtractor._sector_etf_cache_lock:
                if cache_key in FeatureExtractor._sector_etf_cache:
                    cached_data, cached_time = FeatureExtractor._sector_etf_cache[
                        cache_key
                    ]
                    if datetime.now() - cached_time < self.spy_ttl:  # Reuse SPY TTL
                        return cached_data

            # Cache miss - fetch data with lock (one thread fetches, others wait)
            with FeatureExtractor._sector_etf_cache_lock:
                # Double-check cache after acquiring lock (another thread may have fetched)
                if cache_key in FeatureExtractor._sector_etf_cache:
                    cached_data, cached_time = FeatureExtractor._sector_etf_cache[
                        cache_key
                    ]
                    if datetime.now() - cached_time < self.spy_ttl:
                        return cached_data

                # Fetch sector ETF data
                if self.data_provider is None:
                    logger.warning(f"No data provider configured for {etf_symbol} data")
                    return None

                # Calculate date range - ensure datetime objects for arithmetic
                end_date = pd.to_datetime(timestamp).to_pydatetime()
                start_date = pd.to_datetime(timestamp).to_pydatetime() - timedelta(
                    days=lookback + 30
                )  # Extra buffer for market holidays

                # Fetch from provider (pass datetime objects, not strings)
                etf_data = self.data_provider.get_historical_bars(
                    symbol=etf_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                )

                if etf_data is None or len(etf_data) < 5:
                    logger.debug(f"Insufficient {etf_symbol} data for {timestamp}")
                    return None

                # Normalize column names (yfinance returns Title Case, TradeStation lowercase)
                etf_data.columns = [
                    c.lower().replace(" ", "_") for c in etf_data.columns
                ]

                # Cache result in CLASS-LEVEL cache
                FeatureExtractor._sector_etf_cache[cache_key] = (
                    etf_data,
                    datetime.now(),
                )

                return etf_data

        except Exception as e:
            logger.warning(f"Failed to fetch {etf_symbol} data: {e}")
            return None

    def _extract_sector_features(self, symbol: str, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 14D sector/industry positioning (f205-f218)

        Features:
        - f205-f215: One-hot sector classification (11 sectors)
        - f216: 1-day relative performance (stock - sector ETF)
        - f217: 5-day relative performance
        - f218: 20-day relative performance
        """
        try:
            # Sector ETF mapping (GICS sectors)
            SECTOR_ETFS = {
                "Technology": "XLK",
                "Financials": "XLF",
                "Healthcare": "XLV",
                "Energy": "XLE",
                "Consumer Cyclical": "XLY",
                "Consumer Defensive": "XLP",
                "Industrials": "XLI",
                "Basic Materials": "XLB",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Communication Services": "XLC",
            }

            # Get sector classification for symbol
            sector = self._get_symbol_sector(symbol)

            # One-hot encode sector (f205-f215)
            sector_onehot = [0.0] * 11
            if sector in SECTOR_ETFS:
                sector_index = list(SECTOR_ETFS.keys()).index(sector)
                sector_onehot[sector_index] = 1.0

            # Get sector ETF ticker
            sector_etf = SECTOR_ETFS.get(sector)
            if sector_etf is None or len(df) < 20:
                # No sector data or insufficient history
                return np.array(sector_onehot + [0.0, 0.0, 0.0], dtype=np.float32)

            # Fetch sector ETF data
            timestamp = (
                pd.Timestamp(df.index[-1])
                if isinstance(df.index[-1], (str, pd.Timestamp))
                else datetime.now()
            )
            sector_data = self._get_sector_etf_data(sector_etf, timestamp, lookback=30)

            if sector_data is None or len(sector_data) < 20:
                logger.warning(
                    f"Insufficient sector ETF data for {symbol} ({sector_etf})"
                )
                return np.array(sector_onehot + [0.0, 0.0, 0.0], dtype=np.float32)

            # Calculate relative performance for different horizons
            horizons = [1, 5, 20]
            relative_performance = []

            for horizon in horizons:
                if len(df) < horizon + 1 or len(sector_data) < horizon + 1:
                    relative_performance.append(0.0)
                    continue

                # Stock return
                stock_return = (
                    df["close"].iloc[-1] / df["close"].iloc[-horizon - 1]
                ) - 1.0

                # Sector ETF return
                sector_return = (
                    sector_data["close"].iloc[-1]
                    / sector_data["close"].iloc[-horizon - 1]
                ) - 1.0

                # Relative performance (alpha vs sector)
                rel_perf = stock_return - sector_return
                relative_performance.append(rel_perf)

            return np.array(sector_onehot + relative_performance, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract sector features for {symbol}: {e}")
            return np.zeros(14, dtype=np.float32)

    def _extract_vwap_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 5D VWAP-based features (institutional flow signals)

        Features:
        - Distance from VWAP (% deviation)
        - VWAP slope (momentum indicator)
        - Cumulative volume delta from VWAP
        - Volume-weighted price momentum
        - VWAP mean reversion signal
        """
        try:
            if len(df) < 20:
                return np.zeros(5, dtype=np.float32)

            # Calculate VWAP over different windows
            # VWAP = cumsum(price * volume) / cumsum(volume)
            close = df["close"].values
            volume = df["volume"].values

            # Today's VWAP (cumulative for current session)
            vwap = np.cumsum(close * volume) / np.cumsum(volume)
            current_vwap = vwap[-1]
            current_price = close[-1]

            # Feature 1: Distance from VWAP (% deviation)
            distance_from_vwap = (
                (current_price - current_vwap) / current_vwap
                if current_vwap > 0
                else 0.0
            )

            # Feature 2: VWAP slope (5-day momentum)
            if len(vwap) >= 5:
                vwap_5d_ago = vwap[-5]
                vwap_slope = (
                    (current_vwap - vwap_5d_ago) / vwap_5d_ago
                    if vwap_5d_ago > 0
                    else 0.0
                )
            else:
                vwap_slope = 0.0

            # Feature 3: Cumulative delta (volume-weighted price deviation)
            # Positive = price consistently above VWAP (buying pressure)
            # Negative = price consistently below VWAP (selling pressure)
            price_deviations = close - vwap
            cumulative_delta = (
                np.sum(price_deviations * volume) / np.sum(volume)
                if np.sum(volume) > 0
                else 0.0
            )

            # Feature 4: Volume-weighted price momentum (recent vs old)
            if len(df) >= 10:
                recent_vwap = (
                    np.sum(close[-5:] * volume[-5:]) / np.sum(volume[-5:])
                    if np.sum(volume[-5:]) > 0
                    else current_price
                )
                older_vwap = (
                    np.sum(close[-10:-5] * volume[-10:-5]) / np.sum(volume[-10:-5])
                    if np.sum(volume[-10:-5]) > 0
                    else recent_vwap
                )
                vw_momentum = (
                    (recent_vwap - older_vwap) / older_vwap if older_vwap > 0 else 0.0
                )
            else:
                vw_momentum = 0.0

            # Feature 5: Mean reversion signal
            # Z-score of current price vs VWAP distribution
            if len(vwap) >= 20:
                vwap_std = np.std(close[-20:] - vwap[-20:])
                mean_reversion = (
                    (current_price - current_vwap) / vwap_std
                    if vwap_std > 1e-8
                    else 0.0
                )
            else:
                mean_reversion = 0.0

            return np.array(
                [
                    distance_from_vwap,
                    vwap_slope,
                    cumulative_delta,
                    vw_momentum,
                    mean_reversion,
                ],
                dtype=np.float32,
            )

        except Exception as e:
            logger.warning(f"Failed to extract VWAP features: {e}")
            return np.zeros(5, dtype=np.float32)

    def _extract_options_features(self, symbol: str, timestamp: datetime) -> np.ndarray:
        """Extract 10D options flow features (f225-f234)"""
        # Placeholder: Put/call ratio, IV, options volume
        # TODO: Implement options data fetching
        return np.zeros(10, dtype=np.float32)

    def _extract_crypto_specific_features(
        self, symbol: str, timestamp: datetime, ohlcv_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract 34D crypto-specific features (on-chain metrics, funding rates, liquidations).

        This is crypto's equivalent of fundamentals - instead of PE ratios and earnings,
        we use on-chain data like exchange flows, whale transactions, and derivatives data.

        TODO: Implement actual extraction from CoinGecko/on-chain APIs
        For now returns zeros - will be replaced with real data from crypto_market_extractor
        """
        return np.zeros(DIMS.fincoll_cross_asset, dtype=np.float32)

    def _extract_cross_crypto_features(
        self, symbol: str, timestamp: datetime, ohlcv_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract 30D cross-crypto correlation features (BTC dominance, market cap ratios).

        This is crypto's equivalent of sector analysis - instead of comparing to sector ETFs,
        we compare to BTC dominance, total market cap, and altcoin season indicators.

        TODO: Implement actual extraction from CoinGecko market data
        For now returns zeros - will be replaced with real data from crypto_market_extractor
        """
        return np.zeros(DIMS.fincoll_early_signal, dtype=np.float32)

    def _extract_support_resistance(self, df: pd.DataFrame) -> np.ndarray:
        """Extract 30D support/resistance levels (f229-f258)"""
        features = []

        if len(df) < 20:
            return np.zeros(30, dtype=np.float32)

        close = df["close"].iloc[-1]

        # Session high/low (6D)
        session_high = df["high"].iloc[-1]
        session_low = df["low"].iloc[-1]
        prev_high = df["high"].iloc[-2] if len(df) >= 2 else session_high
        prev_low = df["low"].iloc[-2] if len(df) >= 2 else session_low
        prev2_high = df["high"].iloc[-3] if len(df) >= 3 else prev_high
        prev2_low = df["low"].iloc[-3] if len(df) >= 3 else prev_low

        features.extend(
            [
                (session_high - close) / close,
                (close - session_low) / close,
                (prev_high - close) / close,
                (close - prev_low) / close,
                (prev2_high - close) / close,
                (close - prev2_low) / close,
            ]
        )

        # Key support/resistance levels (24D = 12 levels × 2 features each)
        # Find local highs and lows in recent history
        window = min(50, len(df))
        highs = df["high"].iloc[-window:]
        lows = df["low"].iloc[-window:]

        # Top 6 resistance levels (local highs)
        resistance_levels = highs.nlargest(6).values
        for level in resistance_levels:
            features.append((level - close) / close)  # Distance to resistance
            features.append(1.0)  # Strength placeholder

        # Top 6 support levels (local lows)
        support_levels = lows.nsmallest(6).values
        for level in support_levels:
            features.append((close - level) / close)  # Distance to support
            features.append(1.0)  # Strength placeholder

        # Pad or truncate to 30D
        features = features[:30]
        while len(features) < 30:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _extract_senvec_features(self, symbol: str, timestamp: datetime) -> np.ndarray:
        """
        Extract {DIMS.senvec_total}D SenVec sentiment features (dimension from config)
        Uses stale-while-revalidate caching with per-component decay.

        Features from SenVec microservices:
        - f264-f281 (18D): Cross-asset signals (Alpha Vantage)
        - f282-f304 (23D): Social sentiment (Twitter, Reddit, StockTwits)
        - f305-f312 (8D): News sentiment (FinLight)

        Caching strategy:
        - Social: FRESH 2h, STALE 3d, decay 12h (buzz decays fast)
        - News: FRESH 1h, STALE 7d, decay 24h (persists longer)
        - Total vector cached, but decay applied per component

        Args:
            symbol: Stock ticker symbol
            timestamp: Bar timestamp (datetime or pandas Timestamp or int)

        Returns:
            {DIMS.senvec_total}D numpy array with SenVec features
        """
        if not self.enable_senvec:
            logger.debug(f"SenVec disabled, returning zeros for {symbol}")
            return np.zeros(DIMS.senvec_total, dtype=np.float32)

        # Convert timestamp to datetime if it's an int or pandas Timestamp
        if isinstance(timestamp, int):
            timestamp = pd.Timestamp(timestamp)
        elif not isinstance(timestamp, (datetime, pd.Timestamp)):
            timestamp = pd.Timestamp(timestamp)

        # Date for cache key
        date_str = pd.Timestamp(timestamp).strftime("%Y-%m-%d")
        cache_key = (symbol, date_str)

        # TIER 2 OPTIMIZATION: Check for pre-fetched batch data (FIXED: with date check)
        batch_key = (symbol, date_str)
        if batch_key in self._batch_senvec_data:
            if self._auto_batch_enabled:
                logger.debug(
                    f"🚀 {symbol} SenVec: Using AUTO-BATCH data for {date_str} (10x speedup)"
                )
            else:
                logger.debug(
                    f"{symbol} SenVec: Using BATCH data for {date_str} (10x speedup)"
                )
            return self._batch_senvec_data[batch_key]

        # Define fetch function
        def fetch_fresh_senvec():
            senvec_features = get_senvec_features(
                symbol=symbol, date=date_str, fallback_zeros=True
            )

            # Ensure correct shape
            if senvec_features.shape != (DIMS.senvec_total,):
                logger.warning(
                    f"Expected {DIMS.senvec_total}D SenVec features for {symbol}, "
                    f"got {senvec_features.shape}"
                )
                senvec_features = np.pad(
                    senvec_features,
                    (0, max(0, DIMS.senvec_total - len(senvec_features))),
                )[: DIMS.senvec_total]

            senvec_features = senvec_features.astype(np.float32)

            # ZPADDING: zero out the 8D senvec_news slot (indices 41-48 of 49D)
            # Port 18004 (senvec-news) is permanently disabled; this slot was populated
            # by the old FinLight-mislabeled service. Slot is preserved for retraining later.
            if len(senvec_features) >= 49:
                senvec_features[41:49] = 0.0

            return senvec_features

        # Use three-tier cache for social component (most volatile)
        # Note: We cache the entire vector but could split by component in future
        try:
            # Use 'senvec_social' config (most conservative - shortest TTL)
            features, cache_status = self._get_cached_or_fetch(
                cache_key=cache_key,
                cache_dict=self._senvec_cache,
                component="senvec_social",  # Use social config (shortest decay)
                fetch_fn=fetch_fresh_senvec,
                symbol=symbol,
            )
            return features

        except Exception as e:
            # All tiers failed (no cache + API down)
            logger.error(f"{symbol} SenVec: Complete failure, returning zeros: {e}")
            return np.zeros(DIMS.senvec_total, dtype=np.float32)

    def _extract_futures_features(self) -> np.ndarray:
        """
        Extract 25D futures market features (f336-f360)

        Returns macro market context from ES, NQ, VIX, CL, GC futures.
        Critical for understanding overall market regime.

        Features:
        - Futures prices (5D): Normalized current prices
        - Futures momentum (10D): 1d and 5d returns for each
        - Futures volatility (10D): 5d and 20d rolling vol for each

        Returns:
            np.ndarray of shape (25,) with futures features, or zeros if unavailable
        """
        if not self.enable_futures or self.futures_extractor is None:
            return np.zeros(25, dtype=np.float32)

        try:
            futures_features = self.futures_extractor.extract_features()
            return futures_features.astype(np.float32)

        except Exception as e:
            logger.error(f"Futures feature extraction failed: {e}")
            return np.zeros(25, dtype=np.float32)

    def _get_finnhub_basic_metrics(self, symbol: str) -> dict:
        """
        Get basic financial metrics from Finnhub (free tier)

        API: /stock/metric?symbol={symbol}&metric=all
        Free tier: 60 calls/min

        Returns dict with keys: pegTTM, psTTM, evRevenueTTM, operatingMarginTTM,
                               roaTTM, currentRatioQuarterly, longTermDebt/equityQuarterly, etc.

        Caching strategy (from YAML config):
        - FRESH (0-24h): Serve cached, no API call
        - STALE (24h-30d): Serve cached with NO decay (fundamentals persist)
        - EXPIRED (>30d): Must fetch fresh or fail
        """
        if not self.enable_finnhub:
            return {}

        def fetch_finnhub_metrics():
            """Fetch fresh metrics from Finnhub API"""
            url = "https://finnhub.io/api/v1/stock/metric"
            params = {"symbol": symbol, "metric": "all", "token": self.finnhub_api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return data.get("metric", {})

        try:
            cache_key = f"finnhub_metrics_{symbol}"
            metrics, cache_status = self._get_cached_or_fetch(
                cache_key=cache_key,
                cache_dict=FeatureExtractor._fundamental_cache,
                component="fundamentals",
                fetch_fn=fetch_finnhub_metrics,
                symbol=symbol,
                cache_lock=FeatureExtractor._fundamental_cache_lock,
            )
            return metrics

        except Exception as e:
            logger.warning(f"Failed to get Finnhub metrics for {symbol}: {e}")
            return {}

    def _extract_finnhub_fundamentals(self, symbol: str) -> np.ndarray:
        """
        Extract 15D Finnhub fundamental features (f361-f375)

        Leading indicators from Finnhub free tier:
        - Earnings Surprises (4D): f361-f364
        - Insider Transactions (5D): f365-f369
        - Analyst Recommendations (6D): f370-f375

        Returns:
            15D numpy array
        """
        if not self.enable_finnhub:
            return np.zeros(15, dtype=np.float32)

        # Skip crypto symbols - they don't have earnings, insider trades, or analyst coverage
        # Crypto identifiers: -USD, -USDT, -USDC, etc.
        if "-USD" in symbol or symbol.endswith(("USD", "USDT", "USDC", "BTC", "ETH")):
            return np.zeros(15, dtype=np.float32)

        try:
            # PARALLELIZE Finnhub API calls (3 concurrent instead of 3 sequential)
            # This reduces timeout from 6s to 2s if all calls timeout
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=3) as executor:
                earnings_future = executor.submit(self._get_finnhub_earnings, symbol)
                insider_future = executor.submit(self._get_finnhub_insider, symbol)
                analyst_future = executor.submit(self._get_finnhub_analyst, symbol)

                # Wait for all with timeout
                earnings = earnings_future.result(timeout=3)
                insider = insider_future.result(timeout=3)
                analyst = analyst_future.result(timeout=3)

            # Combine into 15D vector
            features = np.concatenate([earnings, insider, analyst])

            return features

        except Exception as e:
            logger.warning(f"Failed to get Finnhub fundamentals for {symbol}: {e}")
            return np.zeros(15, dtype=np.float32)

    def _get_finnhub_earnings(self, symbol: str) -> np.ndarray:
        """
        Extract 4D earnings surprise features

        API: /stock/earnings
        Free tier: 60 calls/min

        Returns:
            4D array: [surprise_pct, beat_streak, avg_surprise, surprise_volatility]

        Caching strategy: Uses 'fundamentals' config (fresh_ttl=24h, stale_ttl=30d)
        """

        def fetch_finnhub_earnings():
            """Fetch fresh earnings data from Finnhub API"""
            url = "https://finnhub.io/api/v1/stock/earnings"
            params = {"symbol": symbol, "token": self.finnhub_api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse last 4 quarters
            earnings_data = data[:4] if len(data) >= 4 else data

            if not earnings_data:
                return np.zeros(4, dtype=np.float32)

            # Calculate features
            surprise_pcts = []
            beat_streak = 0

            for quarter in earnings_data:
                actual = quarter.get("actual", 0)
                estimate = quarter.get("estimate", 0)

                if estimate != 0:
                    surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                    surprise_pcts.append(surprise_pct)

                    # Track beat streak (consecutive positive surprises)
                    if surprise_pct > 0:
                        beat_streak += 1
                    else:
                        break  # Streak broken

            # Feature calculations
            surprise_pct = surprise_pcts[0] if surprise_pcts else 0.0  # Most recent
            avg_surprise = np.mean(surprise_pcts) if surprise_pcts else 0.0
            surprise_volatility = (
                np.std(surprise_pcts) if len(surprise_pcts) > 1 else 0.0
            )

            return np.array(
                [
                    surprise_pct,  # f361: Most recent surprise %
                    beat_streak,  # f362: Consecutive beats
                    avg_surprise,  # f363: Average surprise (last 4Q)
                    surprise_volatility,  # f364: Surprise consistency
                ],
                dtype=np.float32,
            )

        try:
            cache_key = f"finnhub_earnings_{symbol}"
            features, cache_status = self._get_cached_or_fetch(
                cache_key=cache_key,
                cache_dict=FeatureExtractor._fundamental_cache,
                component="fundamentals",
                fetch_fn=fetch_finnhub_earnings,
                symbol=symbol,
                cache_lock=FeatureExtractor._fundamental_cache_lock,
            )
            return features

        except Exception as e:
            logger.warning(f"Failed to get earnings for {symbol}: {e}")
            return np.zeros(4, dtype=np.float32)

    def _get_finnhub_insider(self, symbol: str) -> np.ndarray:
        """
        Extract 5D insider transaction features

        API: /stock/insider-transactions
        Free tier: 60 calls/min

        Returns:
            5D array: [buy_sell_ratio, net_shares, avg_price, transaction_count, buying_acceleration]

        Caching strategy: Uses 'fundamentals' config (fresh_ttl=24h, stale_ttl=30d)
        """

        def fetch_finnhub_insider():
            """Fetch fresh insider transaction data from Finnhub API"""
            url = "https://finnhub.io/api/v1/stock/insider-transactions"
            params = {"symbol": symbol, "token": self.finnhub_api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            transactions = data.get("data", [])

            if not transactions:
                return np.zeros(5, dtype=np.float32)

            # Filter to last 30 days
            now = datetime.now()
            recent_30d = [
                t
                for t in transactions
                if (now - datetime.fromisoformat(t["transactionDate"])).days <= 30
            ]

            # Filter to 30-60 days ago (for acceleration)
            prior_30d = [
                t
                for t in transactions
                if 30 < (now - datetime.fromisoformat(t["transactionDate"])).days <= 60
            ]

            # Calculate metrics
            def calc_metrics(txns):
                if not txns:
                    return 0, 0, 0, 0

                buys = sum(1 for t in txns if t.get("transactionCode") == "P")
                sells = sum(1 for t in txns if t.get("transactionCode") == "S")

                buy_sell_ratio = (
                    buys / sells if sells > 0 else (buys if buys > 0 else 0)
                )

                net_shares = sum(t.get("change", 0) for t in txns)

                prices = [
                    t.get("transactionPrice", 0)
                    for t in txns
                    if t.get("transactionPrice")
                ]
                avg_price = np.mean(prices) if prices else 0

                return buy_sell_ratio, net_shares, avg_price, len(txns)

            # Recent metrics
            buy_sell_ratio, net_shares, avg_price, count = calc_metrics(recent_30d)

            # Prior metrics (for acceleration)
            prior_ratio, _, _, prior_count = calc_metrics(prior_30d)

            # Buying acceleration (recent vs prior)
            buying_acceleration = (
                buy_sell_ratio / prior_ratio if prior_ratio > 0 else 1.0
            )

            return np.array(
                [
                    buy_sell_ratio,  # f365: Buys / Sells (last 30d)
                    net_shares / 1000,  # f366: Net shares (scaled down)
                    avg_price,  # f367: Avg transaction price
                    count,  # f368: Transaction count
                    buying_acceleration,  # f369: Recent vs prior buying
                ],
                dtype=np.float32,
            )

        try:
            cache_key = f"finnhub_insider_{symbol}"
            features, cache_status = self._get_cached_or_fetch(
                cache_key=cache_key,
                cache_dict=FeatureExtractor._fundamental_cache,
                component="fundamentals",
                fetch_fn=fetch_finnhub_insider,
                symbol=symbol,
                cache_lock=FeatureExtractor._fundamental_cache_lock,
            )
            return features

        except Exception as e:
            logger.warning(f"Failed to get insider transactions for {symbol}: {e}")
            return np.zeros(5, dtype=np.float32)

    def _get_finnhub_analyst(self, symbol: str) -> np.ndarray:
        """
        Extract 6D analyst recommendation features

        API: /stock/recommendation
        Free tier: 60 calls/min

        Returns:
            6D array: [buy_pct, consensus_score, upgrade_momentum, coverage_count,
                       strong_conviction_pct, bullish_trend]

        Caching strategy: Uses 'fundamentals' config (fresh_ttl=24h, stale_ttl=30d)
        """

        def fetch_finnhub_analyst():
            """Fetch fresh analyst recommendation data from Finnhub API"""
            url = "https://finnhub.io/api/v1/stock/recommendation"
            params = {"symbol": symbol, "token": self.finnhub_api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data or len(data) == 0:
                return np.zeros(6, dtype=np.float32)

            # Get most recent month
            latest = data[0]

            strong_buy = latest.get("strongBuy", 0)
            buy = latest.get("buy", 0)
            hold = latest.get("hold", 0)
            sell = latest.get("sell", 0)
            strong_sell = latest.get("strongSell", 0)

            total = strong_buy + buy + hold + sell + strong_sell

            if total == 0:
                return np.zeros(6, dtype=np.float32)

            # Buy percentage
            buy_pct = (strong_buy + buy) / total

            # Consensus score (1=sell, 5=strongBuy)
            consensus_score = (
                strong_sell * 1 + sell * 2 + hold * 3 + buy * 4 + strong_buy * 5
            ) / total

            # Strong conviction percentage
            strong_conviction_pct = (strong_buy + strong_sell) / total

            # Upgrade momentum (month-over-month change in consensus)
            if len(data) > 1:
                prior = data[1]
                prior_total = sum(
                    [
                        prior.get("strongBuy", 0),
                        prior.get("buy", 0),
                        prior.get("hold", 0),
                        prior.get("sell", 0),
                        prior.get("strongSell", 0),
                    ]
                )

                if prior_total > 0:
                    prior_consensus = (
                        prior.get("strongSell", 0) * 1
                        + prior.get("sell", 0) * 2
                        + prior.get("hold", 0) * 3
                        + prior.get("buy", 0) * 4
                        + prior.get("strongBuy", 0) * 5
                    ) / prior_total

                    upgrade_momentum = consensus_score - prior_consensus
                    bullish_trend = (
                        1
                        if upgrade_momentum > 0.1
                        else (-1 if upgrade_momentum < -0.1 else 0)
                    )
                else:
                    upgrade_momentum = 0
                    bullish_trend = 0
            else:
                upgrade_momentum = 0
                bullish_trend = 0

            return np.array(
                [
                    buy_pct,  # f370: (strongBuy + buy) / total
                    consensus_score,  # f371: Weighted score (1-5)
                    upgrade_momentum,  # f372: Month-over-month change
                    total,  # f373: Total analyst coverage
                    strong_conviction_pct,  # f374: Strong opinions
                    bullish_trend,  # f375: 1=improving, -1=declining, 0=stable
                ],
                dtype=np.float32,
            )

        try:
            cache_key = f"finnhub_analyst_{symbol}"
            features, cache_status = self._get_cached_or_fetch(
                cache_key=cache_key,
                cache_dict=FeatureExtractor._fundamental_cache,
                component="fundamentals",
                fetch_fn=fetch_finnhub_analyst,
                symbol=symbol,
                cache_lock=FeatureExtractor._fundamental_cache_lock,
            )
            return features

        except Exception as e:
            logger.warning(f"Failed to get analyst recommendations for {symbol}: {e}")
            return np.zeros(6, dtype=np.float32)


if __name__ == "__main__":
    # Test feature extractor
    print("Testing Feature Extractor ({DIMS.fincoll_total}D)...")

    # Create dummy OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=200, freq="1min")
    df = pd.DataFrame(
        {
            "open": np.random.randn(200).cumsum() + 100,
            "high": np.random.randn(200).cumsum() + 101,
            "low": np.random.randn(200).cumsum() + 99,
            "close": np.random.randn(200).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 200),
        },
        index=dates,
    )

    # Create extractor (without Alpha Vantage for testing)
    extractor = FeatureExtractor(alpha_vantage_client=None, enable_senvec=True)

    # Extract features
    features = extractor.extract_features(df, "AAPL", datetime.now())

    print(f"\n✅ Extracted {len(features)} features")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

    # Break down by category
    print(f"\nFeature breakdown:")
    print(f"  Technical (f0-f80): {features[0:81].shape}")
    print(f"  Advanced Technical (f81-f130): {features[81:131].shape}")
    print(f"  Velocity/Accel (f131-f150): {features[131:151].shape}")
    print(f"  News Sentiment (f151-f170): {features[151:171].shape}")
    print(f"  Fundamentals (f171-f186): {features[171:187].shape}")
    print(f"  Cross-Asset (f187-f204): {features[187:205].shape}")
    print(f"  Sector/Industry (f205-f218): {features[205:219].shape}")
    print(f"  Options Flow (f219-f228): {features[219:229].shape}")
    print(f"  Support/Resistance (f229-f258): {features[229:259].shape}")
    print(f"  VWAP (f259-f263): {features[259:264].shape}")
    print(f"  SenVec (f264-f335): {features[264:336].shape}")
    print(f"  Futures (f336-f360): {features[336:361].shape}")
    print(f"  Finnhub Fundamentals (f361-f375): {features[361:376].shape}")

    print(f"\n✅ Feature extractor ({DIMS.fincoll_total}D) ready for training!")
