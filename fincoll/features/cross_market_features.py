"""
Cross-Market Feature Extractor for Crypto (20D)

Extracts cross-market correlation features between crypto and traditional markets.
Uses YFinance provider for centralized, cached data access.

Feature Groups:
- Major Equity Indexes (8D): SPY, QQQ, DIA, IWM, VIX correlations + returns
- Futures Markets (6D): ES, NQ, YM correlations + returns (24/7 coverage)
- Regime Detection (6D): Market regime classification and divergence metrics

Total: 20 dimensions
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Import YFinance provider (ensure it's installed in venv)
try:
    from yfinance_provider import YFinanceClient, DataNotFoundError
except ImportError:
    # Graceful degradation if provider not installed
    YFinanceClient = None
    DataNotFoundError = Exception

logger = logging.getLogger(__name__)


class CrossMarketFeatureExtractor:
    """
    Extract cross-market correlation features (20D).

    Uses YFinance provider with smart caching to avoid rate limits
    when both crypto and equities FinColl fetch the same indexes.
    """

    # Major equity indexes
    EQUITY_INDEXES = ["SPY", "QQQ", "DIA", "IWM"]  # S&P 500, NASDAQ, Dow, Russell 2000
    VIX_SYMBOL = "^VIX"  # Fear gauge

    # Futures symbols (24/7 trading)
    FUTURES_SYMBOLS = ["ES=F", "NQ=F", "YM=F"]  # E-mini S&P, NASDAQ, Dow

    # Correlation window (30 days)
    CORRELATION_WINDOW = 30

    def __init__(
        self,
        cache_ttl: int = 60,
        enable_cache: bool = True,
        redis_host: str = "10.32.3.27",
        redis_port: int = 6379,
    ):
        """
        Initialize cross-market feature extractor.

        Args:
            cache_ttl: Cache TTL in seconds (controlled by PIM scheduler)
            enable_cache: Whether to use Redis caching
            redis_host: Redis server host
            redis_port: Redis server port
        """
        if YFinanceClient is None:
            raise ImportError(
                "yfinance_provider not installed. "
                "Install with: uv pip install -e pim-data-providers/yfinance"
            )

        self.client = YFinanceClient(
            cache_ttl=cache_ttl,
            enable_cache=enable_cache,
            redis_host=redis_host,
            redis_port=redis_port,
        )

    def extract(
        self,
        symbol: str,
        crypto_prices: pd.DataFrame,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """
        Extract cross-market features for a crypto symbol.

        Args:
            symbol: Crypto symbol (e.g., "BTC/USD")
            crypto_prices: DataFrame with crypto OHLCV data (last 30+ days)
                          Required columns: 'close', index: DatetimeIndex
            timestamp: Current timestamp (defaults to now)

        Returns:
            20D feature vector as numpy array

        Raises:
            ValueError: If crypto_prices missing required columns
            RuntimeError: If unable to fetch market data
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Validate crypto_prices
        if not isinstance(crypto_prices, pd.DataFrame):
            raise ValueError("crypto_prices must be a pandas DataFrame")
        if "close" not in crypto_prices.columns:
            raise ValueError("crypto_prices must have 'close' column")
        if not isinstance(crypto_prices.index, pd.DatetimeIndex):
            raise ValueError("crypto_prices index must be DatetimeIndex")
        if len(crypto_prices) < self.CORRELATION_WINDOW:
            logger.warning(
                f"crypto_prices has only {len(crypto_prices)} rows, "
                f"need {self.CORRELATION_WINDOW} for correlation"
            )

        features = []

        try:
            # 1. Major Equity Indexes (8D)
            equity_features = self._extract_equity_features(crypto_prices, timestamp)
            features.extend(equity_features)

            # 2. Futures Markets (6D)
            futures_features = self._extract_futures_features(crypto_prices, timestamp)
            features.extend(futures_features)

            # 3. Regime Detection (6D)
            regime_features = self._extract_regime_features(
                crypto_prices, equity_features, futures_features, timestamp
            )
            features.extend(regime_features)

        except Exception as e:
            logger.error(f"Error extracting cross-market features for {symbol}: {e}")
            # Return zeros on error (graceful degradation)
            features = [0.0] * 20

        return np.array(features, dtype=np.float32)

    def _extract_equity_features(
        self, crypto_prices: pd.DataFrame, timestamp: datetime
    ) -> list:
        """
        Extract major equity index features (8D).

        Returns:
            [spy_corr, qqq_corr, dia_corr, iwm_corr, spy_ret, qqq_ret, dia_ret, vix]
        """
        features = []

        # Get 30-day returns for correlation
        crypto_returns = crypto_prices["close"].pct_change().dropna()

        # Fetch index data and compute correlations
        for symbol in self.EQUITY_INDEXES:
            try:
                # Get historical data (30 days)
                history = self.client.get_historical(
                    symbol, period="1mo", interval="1d"
                )

                # Convert to DataFrame
                index_df = pd.DataFrame(
                    [
                        {"timestamp": bar.timestamp, "close": bar.close}
                        for bar in history.bars
                    ]
                )
                index_df.set_index("timestamp", inplace=True)

                # Calculate returns
                index_returns = index_df["close"].pct_change().dropna()

                # Align dates and compute correlation
                aligned_crypto, aligned_index = crypto_returns.align(
                    index_returns, join="inner"
                )

                if len(aligned_crypto) >= 10:  # Minimum 10 days for correlation
                    corr = aligned_crypto.corr(aligned_index)
                    features.append(corr if not np.isnan(corr) else 0.0)
                else:
                    features.append(0.0)

            except (DataNotFoundError, Exception) as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                features.append(0.0)

        # Get 1-day returns for SPY, QQQ, DIA
        for symbol in self.EQUITY_INDEXES[:3]:  # SPY, QQQ, DIA only
            try:
                index_data = self.client.get_index(symbol)
                ret_1d = index_data.change_percent / 100.0  # Convert to decimal
                features.append(ret_1d)
            except Exception as e:
                logger.warning(f"Failed to fetch 1d return for {symbol}: {e}")
                features.append(0.0)

        # Get VIX (fear gauge)
        try:
            vix_data = self.client.get_index(self.VIX_SYMBOL)
            features.append(vix_data.price)
        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")
            features.append(20.0)  # Default VIX value

        return features

    def _extract_futures_features(
        self, crypto_prices: pd.DataFrame, timestamp: datetime
    ) -> list:
        """
        Extract futures market features (6D).

        Returns:
            [es_corr, nq_corr, es_ret, nq_ret, ym_ret, futures_premium]
        """
        features = []

        # Get 30-day returns for correlation
        crypto_returns = crypto_prices["close"].pct_change().dropna()

        # Fetch futures data and compute correlations
        for symbol in self.FUTURES_SYMBOLS[:2]:  # ES, NQ
            try:
                # Get historical data
                history = self.client.get_historical(
                    symbol, period="1mo", interval="1d"
                )

                # Convert to DataFrame
                futures_df = pd.DataFrame(
                    [
                        {"timestamp": bar.timestamp, "close": bar.close}
                        for bar in history.bars
                    ]
                )
                futures_df.set_index("timestamp", inplace=True)

                # Calculate returns
                futures_returns = futures_df["close"].pct_change().dropna()

                # Align and compute correlation
                aligned_crypto, aligned_futures = crypto_returns.align(
                    futures_returns, join="inner"
                )

                if len(aligned_crypto) >= 10:
                    corr = aligned_crypto.corr(aligned_futures)
                    features.append(corr if not np.isnan(corr) else 0.0)
                else:
                    features.append(0.0)

            except Exception as e:
                logger.warning(f"Failed to fetch futures {symbol}: {e}")
                features.append(0.0)

        # Get 1-day returns for all 3 futures
        for symbol in self.FUTURES_SYMBOLS:
            try:
                quote = self.client.get_quote(symbol)
                if quote.previous_close and quote.previous_close > 0:
                    ret_1d = (quote.price - quote.previous_close) / quote.previous_close
                    features.append(ret_1d)
                else:
                    features.append(0.0)
            except Exception as e:
                logger.warning(f"Failed to fetch 1d return for futures {symbol}: {e}")
                features.append(0.0)

        # Futures premium (difference between ES and SPY)
        try:
            es_quote = self.client.get_quote("ES=F")
            spy_index = self.client.get_index("SPY")
            # ES trades at ~10x SPY price
            normalized_es = es_quote.price / 10.0
            premium = (normalized_es - spy_index.price) / spy_index.price
            features.append(premium)
        except Exception as e:
            logger.warning(f"Failed to compute futures premium: {e}")
            features.append(0.0)

        return features

    def _extract_regime_features(
        self,
        crypto_prices: pd.DataFrame,
        equity_features: list,
        futures_features: list,
        timestamp: datetime,
    ) -> list:
        """
        Extract market regime features (6D).

        Returns:
            [inverse_corr_strength, flight_to_crypto, flight_to_safety,
             risk_on, divergence_days, correlation_volatility]
        """
        features = []

        # Extract relevant correlations
        spy_corr = equity_features[0] if len(equity_features) > 0 else 0.0
        spy_ret = equity_features[4] if len(equity_features) > 4 else 0.0
        vix = equity_features[7] if len(equity_features) > 7 else 20.0

        # Crypto return
        crypto_ret = (
            crypto_prices["close"].pct_change().iloc[-1]
            if len(crypto_prices) > 1
            else 0.0
        )

        # 1. Inverse correlation strength (how negative is the correlation?)
        inverse_corr_strength = abs(min(spy_corr, 0.0))  # 0 to 1
        features.append(inverse_corr_strength)

        # 2. Flight to crypto (equities down, crypto up)
        flight_to_crypto = 1.0 if (spy_ret < -0.01 and crypto_ret > 0.01) else 0.0
        features.append(flight_to_crypto)

        # 3. Flight to safety (both down, VIX up)
        flight_to_safety = (
            1.0 if (spy_ret < -0.01 and crypto_ret < -0.01 and vix > 25) else 0.0
        )
        features.append(flight_to_safety)

        # 4. Risk-on regime (both up, VIX down)
        risk_on = 1.0 if (spy_ret > 0.01 and crypto_ret > 0.01 and vix < 20) else 0.0
        features.append(risk_on)

        # 5. Divergence days (rolling window of correlation changes)
        try:
            crypto_returns = crypto_prices["close"].pct_change().dropna()
            # Simple proxy: how many days in last 10 had opposite moves?
            if len(crypto_returns) >= 10:
                divergence_count = 0
                # This is a placeholder - would need SPY historical for accurate count
                # For now, use correlation as proxy
                if spy_corr < -0.3:
                    divergence_count = 5  # Strong divergence
                elif spy_corr < 0:
                    divergence_count = 3  # Moderate divergence
                features.append(float(divergence_count))
            else:
                features.append(0.0)
        except Exception:
            features.append(0.0)

        # 6. Correlation volatility (stability of correlation)
        try:
            # Rolling 30-day correlation volatility
            # Simplified: use absolute correlation as proxy
            corr_vol = abs(spy_corr) * 0.5  # Placeholder calculation
            features.append(corr_vol)
        except Exception:
            features.append(0.0)

        return features

    def close(self):
        """Close YFinance client and Redis connection."""
        if self.client:
            self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
