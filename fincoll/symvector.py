"""
SymVector - Symbol Vector Lifecycle Manager

One instance per symbol/date that manages the complete lifecycle:
1. Initialize with previous vector (DB or Redis fallback to SPY)
2. Gather data pieces in parallel (SenVec, pricing, technical, fundamentals)
3. Merge into existing XXXD-Vector (incremental update)
4. Store data triad (raw data + input vector) to database
5. Send to FinVec for prediction
6. Receive output vector from FinVec
7. Store output vector with triad
8. Forward to requestor (PIM) with symbol, timestamp, record_id

Author: Roderick Ford & Claude Code
Date: 2026-02-10
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
import redis
import json
import os
from pathlib import Path

# P0 FIX: Remove sys.path manipulation - use proper imports
# If config.dimensions not found, it means fincoll package not installed properly
try:
    from config.dimensions import DIMS
except ImportError as e:
    raise ImportError(
        "Failed to import config.dimensions. Please install fincoll package properly:\n"
        "  cd /home/rford/caelum/ss/fincoll && pip install -e .\n"
        f"Error: {e}"
    )

logger = logging.getLogger(__name__)

# P0 FIX: Configuration for timeouts and security
# P2 FIX: Named constants instead of magic numbers
class SymVectorConfig:
    """Configuration for SymVector processing with security and performance settings"""

    # Named constants (P2 fix from code review)
    DEFAULT_SENVEC_TIMEOUT_SECONDS = 5.0
    DEFAULT_PRICING_TIMEOUT_SECONDS = 10.0
    DEFAULT_FUNDAMENTAL_TIMEOUT_SECONDS = 5.0
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
    DEFAULT_CIRCUIT_BREAKER_TIMEOUT_SECONDS = 60
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
    DEFAULT_RETRY_BACKOFF_MAX_SECONDS = 10.0
    DEFAULT_MAX_CONCURRENT_SYMBOLS = 100
    DEFAULT_BATCH_SIZE = 200

    # Timeouts (seconds)
    SENVEC_TIMEOUT = float(os.getenv("SENVEC_TIMEOUT", str(DEFAULT_SENVEC_TIMEOUT_SECONDS)))
    PRICING_TIMEOUT = float(os.getenv("PRICING_TIMEOUT", str(DEFAULT_PRICING_TIMEOUT_SECONDS)))
    FUNDAMENTAL_TIMEOUT = float(os.getenv("FUNDAMENTAL_TIMEOUT", str(DEFAULT_FUNDAMENTAL_TIMEOUT_SECONDS)))

    # Circuit breaker settings
    CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", str(DEFAULT_CIRCUIT_BREAKER_THRESHOLD)))
    CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", str(DEFAULT_CIRCUIT_BREAKER_TIMEOUT_SECONDS)))

    # API Authentication (P0: Required for production)
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    SENVEC_API_KEY = os.getenv("SENVEC_API_KEY")

    # P1: Retry logic with exponential backoff
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", str(DEFAULT_RETRY_BACKOFF_SECONDS)))
    RETRY_BACKOFF_MAX = float(os.getenv("RETRY_BACKOFF_MAX", str(DEFAULT_RETRY_BACKOFF_MAX_SECONDS)))

    # P1: Increased batch processing
    MAX_CONCURRENT_SYMBOLS = int(os.getenv("MAX_CONCURRENT_SYMBOLS", str(DEFAULT_MAX_CONCURRENT_SYMBOLS)))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))


# P0 FIX: Circuit breaker for external services
class CircuitBreaker:
    """Simple circuit breaker to prevent cascading failures"""

    def __init__(self, threshold: int, timeout: int):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call_failed(self):
        """Record a failure"""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()
        if self.failures >= self.threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")

    def call_succeeded(self):
        """Record a success"""
        self.failures = 0
        self.state = "closed"

    def can_attempt(self) -> bool:
        """Check if we can attempt a call"""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed > self.timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False

        # half_open state - allow one attempt
        return True


# Global circuit breakers for external services
_senvec_breaker = CircuitBreaker(
    threshold=SymVectorConfig.CIRCUIT_BREAKER_THRESHOLD,
    timeout=SymVectorConfig.CIRCUIT_BREAKER_TIMEOUT
)
_pricing_breaker = CircuitBreaker(
    threshold=SymVectorConfig.CIRCUIT_BREAKER_THRESHOLD,
    timeout=SymVectorConfig.CIRCUIT_BREAKER_TIMEOUT
)
_fundamental_breaker = CircuitBreaker(
    threshold=SymVectorConfig.CIRCUIT_BREAKER_THRESHOLD,
    timeout=SymVectorConfig.CIRCUIT_BREAKER_TIMEOUT
)


# P1 FIX: Retry helper with exponential backoff
async def retry_with_backoff(coro_func, max_retries=None, backoff=None, backoff_max=None):
    """
    Retry an async function with exponential backoff

    Args:
        coro_func: Coroutine function to retry
        max_retries: Maximum retry attempts (default: config value)
        backoff: Initial backoff seconds (default: config value)
        backoff_max: Maximum backoff seconds (default: config value)

    Returns:
        Result of coro_func or raises last exception
    """
    max_retries = max_retries or SymVectorConfig.MAX_RETRIES
    backoff = backoff or SymVectorConfig.RETRY_BACKOFF
    backoff_max = backoff_max or SymVectorConfig.RETRY_BACKOFF_MAX

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = min(backoff * (2 ** attempt), backoff_max)
                logger.debug(f"Retry {attempt+1}/{max_retries} after {wait_time:.1f}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"All {max_retries} retries exhausted: {e}")

    raise last_exception


# Lazy imports for optional dependencies
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    logger.warning("pandas_ta not available - technical indicators will be limited")
    PANDAS_TA_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not available - fundamentals fallback disabled")
    YFINANCE_AVAILABLE = False


class SymVectorState(Enum):
    """SymVector lifecycle states"""
    INITIALIZED = "initialized"      # Created with symbol/date
    GATHERING = "gathering"           # Collecting data pieces
    READY = "ready"                   # All data collected, vector merged
    PREDICTING = "predicting"         # Sent to FinVec, waiting for output
    COMPLETE = "complete"             # Output received, stored, ready to forward
    FAILED = "failed"                 # Error occurred


class SymVector:
    """
    Symbol Vector - Complete lifecycle manager for one symbol at one point in time

    Manages:
    - Data collection (parallel queries)
    - Vector merging (incremental updates from previous vector)
    - Storage (raw data + input vector + output vector)
    - FinVec interaction (send input, receive output)
    - Response forwarding (to PIM with full context)
    """

    def __init__(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
        redis_client: Optional[redis.Redis] = None,
        influxdb_store = None,
        feature_extractor = None,
        finvec_engine = None,
    ):
        """
        Initialize SymVector for a symbol at a point in time

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            timestamp: Point in time (default: now)
            redis_client: Redis client for previous vector cache
            influxdb_store: InfluxDB storage for data triad
            feature_extractor: Feature extractor for creating vectors
            finvec_engine: FinVec engine for predictions
        """
        self.symbol = symbol
        self.timestamp = timestamp or datetime.utcnow()
        self.state = SymVectorState.INITIALIZED

        # External dependencies
        self.redis = redis_client
        self.store = influxdb_store
        self.extractor = feature_extractor
        self.finvec = finvec_engine

        # Data pieces (collected in parallel)
        self.previous_vector: Optional[np.ndarray] = None  # Starting point
        self.senvec_data: Optional[Dict] = None            # Sentiment data
        self.pricing_data: Optional[Dict] = None           # OHLCV data
        self.technical_data: Optional[Dict] = None         # Technical indicators
        self.fundamental_data: Optional[Dict] = None       # Fundamentals

        # Vectors (created from data)
        self.input_vector: Optional[np.ndarray] = None     # XXXD merged vector
        self.output_vector: Optional[Dict] = None          # Predictions from FinVec

        # Metadata
        self.raw_data: Dict[str, Any] = {}                 # All raw data collected
        self.record_id: Optional[str] = None               # Database record ID
        self.error: Optional[str] = None                   # Error message if failed
        self.timings: Dict[str, float] = {}                # Performance tracking

        logger.debug(f"SymVector initialized: {symbol} at {timestamp}")

    async def gather_and_process(self) -> bool:
        """
        Main workflow: Gather data → Merge → Store → Predict → Store → Complete

        Returns:
            True if successful, False if failed
        """
        try:
            start_time = asyncio.get_event_loop().time()

            # Step 1: Get previous vector (starting point)
            await self._get_previous_vector()

            # Step 2: Gather all data pieces in parallel
            self.state = SymVectorState.GATHERING
            success = await self._gather_all_data()
            if not success:
                self.state = SymVectorState.FAILED
                return False

            # Step 3: Merge into XXXD-Vector
            self.state = SymVectorState.READY
            self.input_vector = await self._merge_vector()

            # Step 4: Store raw data + input vector
            await self._store_input_triad()

            # Step 5: Send to FinVec for prediction (OPTIONAL in Phase 1)
            # Phase 1: Collect input vectors WITHOUT predictions
            # Phase 2: Will create training labels from actual market outcomes
            # Phase 3: Train model, then enable predictions
            if self.finvec:
                self.state = SymVectorState.PREDICTING
                self.output_vector = await self._get_prediction()
                if not self.output_vector:
                    logger.warning(
                        f"{self.symbol}: Prediction failed, continuing anyway (Phase 1 - input collection)"
                    )
                    self.output_vector = None
            else:
                logger.info(
                    f"{self.symbol}: Skipping prediction (Phase 1 - will train model after data collection)"
                )
                self.output_vector = None

            # Step 6: Store output vector with triad (if we have one)
            if self.output_vector:
                await self._store_output_vector()
            else:
                logger.debug(
                    f"{self.symbol}: No output vector (Phase 1 - will create labels from actuals during training)"
                )

            # Step 7: Complete
            self.state = SymVectorState.COMPLETE
            self.timings['total'] = asyncio.get_event_loop().time() - start_time

            # Calculate vector quality metrics for monitoring
            non_zero_pct = 0
            if self.input_vector is not None:
                non_zero_count = np.count_nonzero(self.input_vector)
                non_zero_pct = 100 * non_zero_count / len(self.input_vector)

            logger.info(
                f"✅ SymVector complete: {self.symbol} "
                f"(state={self.state.value}, "
                f"input={self.input_vector.shape if self.input_vector is not None else None}, "
                f"filled={non_zero_pct:.1f}%, "
                f"prediction={'yes' if self.output_vector else 'phase1'}, "
                f"time={self.timings['total']:.2f}s)"
            )
            return True

        except Exception as e:
            self.state = SymVectorState.FAILED
            self.error = str(e)
            logger.error(f"❌ SymVector failed for {self.symbol}: {e}", exc_info=True)
            return False

    async def _get_previous_vector(self):
        """
        Get previous vector as starting point

        P1 FIX: Simplified fallback hierarchy (removed Redis/SPY middle layer)
        Priority:
        1. Database: Nearest tick (not bar) for this symbol
        2. Zero vector: If nothing available
        """
        start = asyncio.get_event_loop().time()

        try:
            # Try database first (nearest tick for this symbol)
            if self.store:
                result = self.store.get_latest_vector(
                    self.symbol,
                    before=self.timestamp
                )
                if result:
                    prev_timestamp, prev_vector = result
                    self.previous_vector = prev_vector
                    logger.debug(
                        f"{self.symbol}: Using previous vector from {prev_timestamp} "
                        f"(shape={prev_vector.shape})"
                    )
                    self.timings['previous_vector'] = asyncio.get_event_loop().time() - start
                    return

            # P1 FIX: Direct fallback to zero vector (removed Redis/SPY middle layer)
            # Reasoning: Redis adds latency without significant benefit for initial vectors
            # Zero vectors work fine as starting point for first-time symbols
            self.previous_vector = np.zeros(DIMS.fincoll_total, dtype=np.float32)
            logger.warning(
                f"{self.symbol}: No previous vector found, using zero vector "
                f"(dim={DIMS.fincoll_total})"
            )

        except Exception as e:
            logger.error(f"Error getting previous vector for {self.symbol}: {e}")
            # Fallback to zero vector
            self.previous_vector = np.zeros(DIMS.fincoll_total, dtype=np.float32)

        self.timings['previous_vector'] = asyncio.get_event_loop().time() - start

    async def _gather_all_data(self) -> bool:
        """
        Gather all data pieces in parallel

        Queries (not ordered, run in parallel):
        - SenVec data (sentiment)
        - Pricing data (OHLCV)
        - Technical data (indicators)
        - Fundamental data (financials)

        Returns:
            True if all required data collected, False otherwise
        """
        start = asyncio.get_event_loop().time()

        # Launch all queries in parallel
        tasks = [
            self._query_senvec(),
            self._query_pricing(),
            self._query_technical(),
            self._query_fundamental(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.warning(
                f"{self.symbol}: {len(failures)}/{len(tasks)} data queries failed"
            )
            # Continue if at least pricing data succeeded (minimum requirement)
            if not self.pricing_data:
                logger.error(f"{self.symbol}: Pricing data missing (required)")
                return False

        self.timings['gather_data'] = asyncio.get_event_loop().time() - start
        return True

    async def _query_senvec(self):
        """Query SenVec for sentiment data with timeout and circuit breaker"""
        start = asyncio.get_event_loop().time()

        # P0 FIX: Check circuit breaker before attempting call
        if not _senvec_breaker.can_attempt():
            logger.warning(f"{self.symbol}: SenVec circuit breaker open, using fallback")
            self._set_senvec_fallback("Circuit breaker open")
            self.timings['senvec'] = asyncio.get_event_loop().time() - start
            return

        try:
            # Use SenVec integration module (sync, but called in async context)
            from utils.senvec_integration import get_senvec_features

            # P0 FIX: Add timeout to SenVec call (5 seconds max)
            loop = asyncio.get_event_loop()
            senvec_task = loop.run_in_executor(
                None,  # default executor
                get_senvec_features,
                self.symbol,
                self.timestamp.strftime('%Y-%m-%d'),
                True  # fallback_zeros
            )

            # Apply timeout
            senvec_features = await asyncio.wait_for(
                senvec_task,
                timeout=SymVectorConfig.SENVEC_TIMEOUT
            )

            # Store as dict for raw data
            self.senvec_data = {
                "features": senvec_features.tolist() if senvec_features is not None else [],
                "dimension": DIMS.senvec_total,
                "source": "senvec_api",
                "timestamp": self.timestamp.isoformat(),
            }

            # P0 FIX: Record success in circuit breaker
            _senvec_breaker.call_succeeded()

            logger.debug(
                f"{self.symbol}: SenVec data collected "
                f"({DIMS.senvec_total}D, non_zero={np.count_nonzero(senvec_features)})"
            )

        except asyncio.TimeoutError:
            # P0 FIX: Handle timeout specifically
            logger.warning(f"{self.symbol}: SenVec timeout after {SymVectorConfig.SENVEC_TIMEOUT}s")
            _senvec_breaker.call_failed()
            self._set_senvec_fallback(f"Timeout after {SymVectorConfig.SENVEC_TIMEOUT}s")

        except Exception as e:
            logger.warning(f"{self.symbol}: SenVec query failed: {e}")
            _senvec_breaker.call_failed()
            self._set_senvec_fallback(str(e))

        finally:
            self.timings['senvec'] = asyncio.get_event_loop().time() - start

    def _set_senvec_fallback(self, error: str):
        """Set SenVec fallback data"""
        self.senvec_data = {
            "features": [0.0] * DIMS.senvec_total,
            "dimension": DIMS.senvec_total,
            "source": "fallback_zeros",
            "error": error,
        }

    async def _query_pricing(self):
        """Query pricing data (OHLCV) with timeout and circuit breaker"""
        start = asyncio.get_event_loop().time()

        # P0 FIX: Check circuit breaker
        if not _pricing_breaker.can_attempt():
            logger.error(f"{self.symbol}: Pricing circuit breaker open")
            self.pricing_data = None
            self.timings['pricing'] = asyncio.get_event_loop().time() - start
            raise RuntimeError("Pricing circuit breaker open - service unavailable")

        try:
            # Use yfinance directly (simpler, no complex dependencies)
            import yfinance as yf
            from datetime import timedelta

            # Use yfinance to fetch data with timeout
            loop = asyncio.get_event_loop()

            def fetch_yf_data():
                ticker = yf.Ticker(self.symbol)
                # Get last 100 days of data
                df = ticker.history(period="100d", interval="1d")
                return df

            # P0 FIX: Add timeout to pricing call
            pricing_task = loop.run_in_executor(None, fetch_yf_data)
            df = await asyncio.wait_for(pricing_task, timeout=SymVectorConfig.PRICING_TIMEOUT)

            if df is None or df.empty:
                raise ValueError(f"No pricing data returned for {self.symbol}")

            # Get latest bar
            latest = df.iloc[-1]

            # Store as dict for raw data
            self.pricing_data = {
                "open": float(latest['Open']),
                "high": float(latest['High']),
                "low": float(latest['Low']),
                "close": float(latest['Close']),
                "volume": int(latest['Volume']) if 'Volume' in latest else 0,
                "bars_count": len(df),
                "df": df,  # Keep full dataframe for technical indicators
                "source": "yfinance",
                "timestamp": self.timestamp.isoformat(),
            }
            # P0 FIX: Record success in circuit breaker
            _pricing_breaker.call_succeeded()

            logger.debug(
                f"{self.symbol}: Pricing data collected "
                f"({len(df)} bars, close={latest['Close']:.2f})"
            )

        except asyncio.TimeoutError:
            # P0 FIX: Handle timeout specifically
            logger.error(f"{self.symbol}: Pricing timeout after {SymVectorConfig.PRICING_TIMEOUT}s")
            _pricing_breaker.call_failed()
            self.pricing_data = None
            raise RuntimeError(f"Pricing timeout after {SymVectorConfig.PRICING_TIMEOUT}s")

        except Exception as e:
            logger.error(f"{self.symbol}: Pricing query failed: {e}")
            _pricing_breaker.call_failed()
            self.pricing_data = None
            raise  # Pricing is required
        finally:
            self.timings['pricing'] = asyncio.get_event_loop().time() - start

    async def _query_technical(self):
        """Calculate technical indicators from pricing data"""
        start = asyncio.get_event_loop().time()
        try:
            # Technical indicators calculated from pricing data
            # Need pricing_data to be available
            if not self.pricing_data or 'df' not in self.pricing_data:
                raise ValueError("Pricing data required for technical indicators")

            df = self.pricing_data['df']

            # Calculate basic indicators (requires pandas_ta)
            if not PANDAS_TA_AVAILABLE:
                raise ImportError("pandas_ta required for technical indicators")

            # RSI (14-day) - yfinance uses capitalized column names
            rsi = ta.rsi(df['Close'], length=14)
            rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0

            # MACD
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            macd_val = float(macd['MACD_12_26_9'].iloc[-1]) if 'MACD_12_26_9' in macd.columns else 0.0

            # Bollinger Bands
            bbands = ta.bbands(df['Close'], length=20, std=2)
            bb_upper = float(bbands['BBU_20_2.0'].iloc[-1]) if 'BBU_20_2.0' in bbands.columns else 0.0
            bb_lower = float(bbands['BBL_20_2.0'].iloc[-1]) if 'BBL_20_2.0' in bbands.columns else 0.0
            bb_mid = float(bbands['BBM_20_2.0'].iloc[-1]) if 'BBM_20_2.0' in bbands.columns else 0.0

            # ATR (Average True Range)
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            atr_val = float(atr.iloc[-1]) if not atr.empty else 0.0

            # Store as dict
            self.technical_data = {
                "rsi": rsi_val,
                "macd": macd_val,
                "bollinger_upper": bb_upper,
                "bollinger_middle": bb_mid,
                "bollinger_lower": bb_lower,
                "atr": atr_val,
                "source": "pandas_ta",
                "timestamp": self.timestamp.isoformat(),
            }
            logger.debug(
                f"{self.symbol}: Technical data calculated "
                f"(RSI={rsi_val:.1f}, MACD={macd_val:.3f})"
            )
        except Exception as e:
            logger.warning(f"{self.symbol}: Technical calculation failed: {e}")
            # Fallback to zeros/defaults
            self.technical_data = {
                "rsi": 50.0,
                "macd": 0.0,
                "bollinger_upper": 0.0,
                "bollinger_middle": 0.0,
                "bollinger_lower": 0.0,
                "atr": 0.0,
                "source": "fallback",
                "error": str(e),
            }
        finally:
            self.timings['technical'] = asyncio.get_event_loop().time() - start

    async def _query_fundamental(self):
        """Query fundamental data (PE, market cap, etc.) with timeout and circuit breaker"""
        start = asyncio.get_event_loop().time()

        # P0 FIX: Check circuit breaker
        if not _fundamental_breaker.can_attempt():
            logger.warning(f"{self.symbol}: Fundamental circuit breaker open, using fallback")
            self._set_fundamental_fallback("Circuit breaker open")
            self.timings['fundamental'] = asyncio.get_event_loop().time() - start
            return

        try:
            # P0 FIX: Use config for API key (with validation)
            import requests

            finnhub_key = SymVectorConfig.FINNHUB_API_KEY

            # P0 FIX: Validate API key is present for production
            if not finnhub_key:
                logger.debug(f"{self.symbol}: FINNHUB_API_KEY not set, using yfinance fallback")
                # Skip to yfinance fallback
                raise ValueError("Finnhub API key not configured")

            loop = asyncio.get_event_loop()

            # P0 FIX: Make API call with timeout
            def fetch_finnhub():
                url = f"https://finnhub.io/api/v1/stock/metric"
                params = {
                    "symbol": self.symbol,
                    "metric": "all",
                    "token": finnhub_key
                }
                # P0 FIX: Add timeout to requests.get
                response = requests.get(url, params=params, timeout=SymVectorConfig.FUNDAMENTAL_TIMEOUT)

                # P0 FIX: Check for rate limiting
                if response.status_code == 429:
                    raise RuntimeError("Finnhub API rate limit exceeded")
                elif response.status_code == 401:
                    raise RuntimeError("Finnhub API authentication failed")
                elif response.status_code == 200:
                    return response.json()
                else:
                    raise RuntimeError(f"Finnhub API returned {response.status_code}")

            finnhub_task = loop.run_in_executor(None, fetch_finnhub)
            data = await asyncio.wait_for(finnhub_task, timeout=SymVectorConfig.FUNDAMENTAL_TIMEOUT)

            if data and 'metric' in data:
                metrics = data['metric']
                self.fundamental_data = {
                    "pe_ratio": metrics.get("peExclExtraTTM", 0.0),
                    "market_cap": metrics.get("marketCapitalization", 0.0),
                    "beta": metrics.get("beta", 1.0),
                    "roe": metrics.get("roeTTM", 0.0),
                    "roa": metrics.get("roaTTM", 0.0),
                    "current_ratio": metrics.get("currentRatioQuarterly", 0.0),
                    "source": "finnhub",
                    "timestamp": self.timestamp.isoformat(),
                }

                # P0 FIX: Record success
                _fundamental_breaker.call_succeeded()

                logger.debug(
                    f"{self.symbol}: Fundamental data collected from Finnhub "
                    f"(PE={self.fundamental_data['pe_ratio']:.1f})"
                )
                return

            # Fallback: Use yfinance info (if available)
            if not YFINANCE_AVAILABLE:
                raise ImportError("yfinance not available for fundamentals fallback")

            loop = asyncio.get_event_loop()
            def fetch_yfinance():
                ticker = yf.Ticker(self.symbol)
                info = ticker.info
                return info

            # P0 FIX: Add timeout to yfinance fallback
            yf_task = loop.run_in_executor(None, fetch_yfinance)
            info = await asyncio.wait_for(yf_task, timeout=SymVectorConfig.FUNDAMENTAL_TIMEOUT)

            self.fundamental_data = {
                "pe_ratio": info.get("trailingPE", info.get("forwardPE", 0.0)),
                "market_cap": info.get("marketCap", 0.0),
                "beta": info.get("beta", 1.0),
                "roe": info.get("returnOnEquity", 0.0),
                "roa": info.get("returnOnAssets", 0.0),
                "current_ratio": info.get("currentRatio", 0.0),
                "source": "yfinance",
                "timestamp": self.timestamp.isoformat(),
            }

            # P0 FIX: Record success
            _fundamental_breaker.call_succeeded()

            logger.debug(
                f"{self.symbol}: Fundamental data collected from yfinance "
                f"(PE={self.fundamental_data['pe_ratio']:.1f})"
            )

        except asyncio.TimeoutError:
            # P0 FIX: Handle timeout
            logger.warning(f"{self.symbol}: Fundamental timeout after {SymVectorConfig.FUNDAMENTAL_TIMEOUT}s")
            _fundamental_breaker.call_failed()
            self._set_fundamental_fallback(f"Timeout after {SymVectorConfig.FUNDAMENTAL_TIMEOUT}s")

        except Exception as e:
            logger.warning(f"{self.symbol}: Fundamental query failed: {e}")
            _fundamental_breaker.call_failed()
            self._set_fundamental_fallback(str(e))

        finally:
            self.timings['fundamental'] = asyncio.get_event_loop().time() - start

    def _set_fundamental_fallback(self, error: str):
        """Set fundamental fallback data"""
        self.fundamental_data = {
            "pe_ratio": 0.0,
            "market_cap": 0.0,
            "beta": 1.0,
            "roe": 0.0,
            "roa": 0.0,
            "current_ratio": 0.0,
            "source": "fallback",
            "error": error,
        }

    async def _merge_vector(self) -> np.ndarray:
        """
        Merge new data into previous vector to create input vector

        Process:
        1. Use FeatureExtractor to create full 414D vector from collected data
        2. If available, use previous_vector as baseline for incremental updates
        3. Extract features from: pricing (OHLCV), senvec (sentiment), technical, fundamentals

        Returns:
            414D input vector ready for FinVec
        """
        start = asyncio.get_event_loop().time()

        try:
            # Store raw data for database
            self.raw_data = {
                "symbol": self.symbol,
                "timestamp": self.timestamp.isoformat(),
                "senvec": self.senvec_data,
                "pricing": {k: v for k, v in self.pricing_data.items() if k != 'df'} if self.pricing_data else None,
                "technical": self.technical_data,
                "fundamental": self.fundamental_data,
            }

            # Use FeatureExtractor to create full vector if available
            if self.extractor and self.pricing_data and 'df' in self.pricing_data:
                # Extract features from OHLCV dataframe
                df = self.pricing_data['df']

                # Call feature extractor synchronously
                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(
                    None,
                    self.extractor.extract_features,
                    df,
                    self.symbol,
                    self.timestamp
                )

                if vector is not None:
                    logger.debug(
                        f"{self.symbol}: Vector extracted via FeatureExtractor "
                        f"(shape={vector.shape}, non_zero={np.count_nonzero(vector)})"
                    )
                    self.timings['merge_vector'] = asyncio.get_event_loop().time() - start
                    return vector

            # Fallback: Manual feature construction
            logger.warning(f"{self.symbol}: FeatureExtractor unavailable, using manual merge")

            # Start with previous vector as baseline
            vector = self.previous_vector.copy()

            # Update SenVec portion (dimensions from DIMS.fincoll_senvec)
            if self.senvec_data and 'features' in self.senvec_data:
                senvec_features = np.array(self.senvec_data['features'], dtype=np.float32)
                # SenVec goes in positions based on DIMS config
                # For now, put at end (adjust based on actual dimension layout)
                senvec_start = DIMS.fincoll_total - DIMS.senvec_total
                vector[senvec_start:senvec_start+len(senvec_features)] = senvec_features[:DIMS.senvec_total]

            # Update pricing features (first ~81D typically)
            if self.pricing_data:
                # Simple price normalization (close price as % change)
                close = self.pricing_data.get('close', 0.0)
                prev_close = self.pricing_data['df']['Close'].iloc[-2] if len(self.pricing_data['df']) > 1 else close
                price_change = (close - prev_close) / prev_close if prev_close > 0 else 0.0

                # Update first few dimensions with price features
                vector[0] = price_change  # % change
                vector[1] = self.pricing_data.get('volume', 0.0) / 1e6  # volume in millions
                # ... (full implementation would update all 81 technical dimensions)

            logger.debug(
                f"{self.symbol}: Vector manually merged "
                f"(shape={vector.shape}, non_zero={np.count_nonzero(vector)})"
            )

            self.timings['merge_vector'] = asyncio.get_event_loop().time() - start
            return vector

        except Exception as e:
            logger.error(f"{self.symbol}: Vector merge failed: {e}", exc_info=True)
            # Fallback to previous vector
            self.timings['merge_vector'] = asyncio.get_event_loop().time() - start
            return self.previous_vector

    async def _store_input_triad(self):
        """Store raw data + input vector to database"""
        if not self.store:
            logger.warning(f"{self.symbol}: No storage available, skipping")
            return

        try:
            # Store input vector
            self.store.save_feature_vector(
                symbol=self.symbol,
                timestamp=self.timestamp,
                features=self.input_vector,
                source="symvector",
                version="v7"
            )

            # Store raw data
            self.store.save_raw_market_data(
                symbol=self.symbol,
                timestamp=self.timestamp,
                data=self.raw_data,
                source="symvector"
            )

            # Generate record ID for reference
            self.record_id = f"{self.symbol}_{self.timestamp.isoformat()}"

            logger.debug(f"{self.symbol}: Input triad stored (record_id={self.record_id})")

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to store input triad: {e}")

    async def _get_prediction(self) -> Optional[Dict]:
        """
        Send input vector to FinVec and wait for output

        Returns:
            Prediction dict from FinVec or None if failed
        """
        if not self.finvec:
            logger.warning(f"{self.symbol}: No FinVec engine available")
            return None

        start = asyncio.get_event_loop().time()

        try:
            # Prepare input for FinVec engine
            # The engine expects OHLCV data, which it will featurize internally
            if not self.pricing_data or 'df' not in self.pricing_data:
                raise ValueError("Pricing data required for FinVec prediction")

            df = self.pricing_data['df']

            # Call FinVec engine in executor (it's synchronous)
            loop = asyncio.get_event_loop()

            def call_finvec():
                # FinVec engine predict method
                # Depending on the engine type, call appropriate method
                if hasattr(self.finvec, 'predict'):
                    # Direct prediction from features
                    result = self.finvec.predict(self.input_vector)
                    return result
                elif hasattr(self.finvec, 'process_batch'):
                    # Engine-style batch processing
                    result = self.finvec.process_batch([self.symbol])
                    if result and self.symbol in result:
                        return result[self.symbol]
                else:
                    raise NotImplementedError(f"FinVec engine method not recognized")

            result = await loop.run_in_executor(None, call_finvec)

            # Convert result to standard format
            if isinstance(result, dict):
                # Already in dict format
                prediction = result
            else:
                # Convert model output to dict format
                prediction = {
                    "symbol": self.symbol,
                    "velocities": self._parse_velocity_output(result),
                    "timestamp": self.timestamp.isoformat(),
                }

            self.timings['prediction'] = asyncio.get_event_loop().time() - start
            logger.debug(
                f"{self.symbol}: Prediction received from FinVec "
                f"(time={self.timings['prediction']:.2f}s)"
            )

            return prediction

        except Exception as e:
            logger.error(f"{self.symbol}: FinVec prediction failed: {e}", exc_info=True)
            return None

    def _parse_velocity_output(self, output) -> List[Dict]:
        """
        Parse FinVec output into velocity format

        Args:
            output: Raw output from FinVec engine

        Returns:
            List of velocity dicts (one per timeframe)
        """
        # Default timeframes
        timeframes = ["1min", "5min", "15min", "1hour", "daily"]
        velocities = []

        try:
            # If output is numpy array or tensor
            if hasattr(output, 'shape'):
                # Assume output is [velocity_1m, velocity_5m, velocity_15m, velocity_1h, velocity_1d, ...]
                output_arr = output.detach().cpu().numpy() if hasattr(output, 'detach') else output
                output_arr = output_arr.flatten()

                for i, tf in enumerate(timeframes):
                    if i < len(output_arr):
                        velocities.append({
                            "timeframe": tf,
                            "velocity": float(output_arr[i]),
                            "confidence": 0.75,  # Default confidence
                        })
            else:
                # Fallback to default values
                for tf in timeframes:
                    velocities.append({
                        "timeframe": tf,
                        "velocity": 0.0,
                        "confidence": 0.0,
                    })

        except Exception as e:
            logger.warning(f"Failed to parse velocity output: {e}")
            # Return defaults
            for tf in timeframes:
                velocities.append({
                    "timeframe": tf,
                    "velocity": 0.0,
                    "confidence": 0.0,
                })

        return velocities

    async def _store_output_vector(self):
        """Store output vector with triad"""
        if not self.store or not self.output_vector:
            logger.warning(f"{self.symbol}: Cannot store output vector")
            return

        try:
            # Extract velocities
            velocities = self.output_vector.get("velocities", [])
            velocity_map = {}
            for v in velocities:
                tf = v.get("timeframe", "")
                velocity_map[tf] = v.get("velocity", 0.0)

            # Calculate average confidence
            confidences = [v.get("confidence", 0.0) for v in velocities]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Store to database
            self.store.save_prediction_output(
                symbol=self.symbol,
                timestamp=self.timestamp,
                velocity_1m=velocity_map.get("1min", 0.0),
                velocity_15m=velocity_map.get("15min", 0.0),
                velocity_1h=velocity_map.get("1hour", 0.0),
                velocity_1d=velocity_map.get("daily", 0.0),
                confidence=avg_confidence,
                model_version="v7"
            )

            logger.debug(f"{self.symbol}: Output vector stored")

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to store output vector: {e}")

    def to_response(self) -> Dict:
        """
        Format for PIM response

        Returns:
            Dict with symbol, timestamp, record_id, and predictions
        """
        if self.state != SymVectorState.COMPLETE:
            return {
                "symbol": self.symbol,
                "timestamp": self.timestamp.isoformat(),
                "state": self.state.value,
                "error": self.error,
            }

        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "record_id": self.record_id,  # For full triad details
            "prediction": self.output_vector,
            "timings": self.timings,
            "state": self.state.value,
        }

    def __repr__(self):
        return (
            f"SymVector({self.symbol}, {self.timestamp.isoformat()}, "
            f"state={self.state.value})"
        )


# ============================================================================
# Batch Processing
# ============================================================================

async def process_symbol_batch(
    symbols: List[str],
    timestamp: Optional[datetime] = None,
    redis_client = None,
    influxdb_store = None,
    feature_extractor = None,
    finvec_engine = None,
    max_concurrent: int = None,  # P1: Use config default (100 instead of 50)
) -> List[Dict]:
    """
    Process a batch of symbols, returning results one at a time as they complete

    Args:
        symbols: List of stock symbols
        timestamp: Point in time (default: now)
        redis_client: Redis client for caching
        influxdb_store: InfluxDB storage
        feature_extractor: Feature extractor
        finvec_engine: FinVec engine
        max_concurrent: Maximum concurrent SymVector instances (default: SymVectorConfig.MAX_CONCURRENT_SYMBOLS)

    Returns:
        List of response dicts (one per symbol, in completion order)
    """
    max_concurrent = max_concurrent or SymVectorConfig.MAX_CONCURRENT_SYMBOLS  # P1: Default to 100
    timestamp = timestamp or datetime.utcnow()
    results = []

    # Create SymVector instances
    symvectors = [
        SymVector(
            symbol=symbol,
            timestamp=timestamp,
            redis_client=redis_client,
            influxdb_store=influxdb_store,
            feature_extractor=feature_extractor,
            finvec_engine=finvec_engine,
        )
        for symbol in symbols
    ]

    # Process in batches (max_concurrent at a time)
    for i in range(0, len(symvectors), max_concurrent):
        batch = symvectors[i:i+max_concurrent]

        # Process batch in parallel
        tasks = [sv.gather_and_process() for sv in batch]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results as they complete
        for sv in batch:
            results.append(sv.to_response())

    logger.info(
        f"Batch complete: {len(symbols)} symbols processed, "
        f"{sum(1 for r in results if r.get('state') == 'complete')} successful"
    )

    return results
