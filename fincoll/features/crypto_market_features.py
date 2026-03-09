"""
Crypto Market Feature Extractor with Multi-Provider Fallback (126D)

Implements round-robin provider strategy to maximize combined free rate limits:
- CoinGecko: 50 calls/min (primary)
- CryptoCompare: 138 calls/hour (backup)
- CoinMarketCap: 13 calls/hour (fallback)

Feature Groups (126D Total):
- Token Metadata (18D): Holders, audits, age, legitimacy signals
- Onchain Pools (14D): DEX liquidity, TVL, slippage risk
- Crypto Market (9D): BTC dominance, market cap, altcoin season
- Exchanges & Derivatives (11D): Listing quality, CEX/DEX ratio
- Public Treasuries (6D): Institutional holdings (MicroStrategy, Tesla)
- Categorization (31D): Layer 1/2, DeFi, NFT, consensus type
- Fundamentals (REMOVED - 0D): Now provided by Binance GPU-accelerated service (18D)
- Dynamic Clustering (12D): Correlation-based groupings
- Macro Factors (20D): Economic indicators, VIX, DXY (reduced from 25D)
- Sentiment Analysis (5D): Fear/Greed, social media sentiment

Total: 126 dimensions (156D - 30D moved to Binance GPU fundamentals)
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Available crypto data providers."""

    COINGECKO = "coingecko"
    CRYPTOCOMPARE = "cryptocompare"
    COINMARKETCAP = "coinmarketcap"


class ProviderConfig:
    """Configuration for crypto data providers."""

    COINGECKO = {
        "base_url": "https://api.coingecko.com/api/v3",
        "api_key": "CG-p5ZWnSsvU1AArBuf9s2iFf4A",  # Demo plan
        "rate_limit": 50,  # calls/min
        "priority": 1,  # Primary provider
    }

    CRYPTOCOMPARE = {
        "base_url": "https://min-api.cryptocompare.com/data",
        "api_key": None,  # TODO: Add API key when registered
        "rate_limit": 138,  # calls/hour (generous free tier)
        "priority": 2,  # Backup provider
    }

    COINMARKETCAP = {
        "base_url": "https://pro-api.coinmarketcap.com/v1",
        "api_key": None,  # TODO: Add API key when registered
        "rate_limit": 13,  # calls/hour
        "priority": 3,  # Fallback provider
    }


class MultiProviderCryptoFeatureExtractor:
    """
    Extract crypto market features (156D) with multi-provider fallback.

    Uses round-robin strategy to maximize combined free rate limits:
    - Primary: CoinGecko (50 calls/min)
    - Backup: CryptoCompare (138 calls/hour)
    - Fallback: CoinMarketCap (13 calls/hour)

    Combined capacity: ~60+ calls/minute with intelligent distribution.
    """

    def __init__(
        self,
        redis_host: str = "10.32.3.27",
        redis_port: int = 6379,
        cache_ttl: int = 300,  # 5 minutes default
        enable_cache: bool = True,
    ):
        """
        Initialize multi-provider crypto feature extractor.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            cache_ttl: Cache TTL in seconds
            enable_cache: Whether to use Redis caching
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache

        # Provider round-robin state
        self.current_provider = DataProvider.COINGECKO
        self.provider_failures: Dict[DataProvider, int] = {
            DataProvider.COINGECKO: 0,
            DataProvider.CRYPTOCOMPARE: 0,
            DataProvider.COINMARKETCAP: 0,
        }

        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Caelum-Crypto-FinColl/1.0",
            }
        )

    def extract(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """
        Extract crypto market features for a symbol (126D).

        NOTE: Fundamentals (ATH/ATL, volatility, Sharpe, technical indicators) are now
        provided by Binance GPU-accelerated service (18D) for ~65x performance improvement.

        Args:
            symbol: Crypto symbol (e.g., "bitcoin", "ethereum", "BTC")
            timestamp: Current timestamp (defaults to now)

        Returns:
            126D feature vector as numpy array
        """
        if timestamp is None:
            timestamp = datetime.now()

        features = []

        try:
            # Normalize symbol to CoinGecko ID format
            coin_id = self._normalize_symbol(symbol)

            # 1. Token Metadata (18D)
            token_metadata = self._extract_token_metadata(coin_id)
            features.extend(token_metadata)

            # 2. Onchain Pools (14D)
            onchain_pools = self._extract_onchain_pools(coin_id)
            features.extend(onchain_pools)

            # 3. Crypto Market (9D)
            crypto_market = self._extract_crypto_market(coin_id)
            features.extend(crypto_market)

            # 4. Exchanges & Derivatives (11D)
            exchanges_derivatives = self._extract_exchanges_derivatives(coin_id)
            features.extend(exchanges_derivatives)

            # 5. Public Treasuries (6D)
            public_treasuries = self._extract_public_treasuries(coin_id)
            features.extend(public_treasuries)

            # 6. Categorization (31D)
            categorization = self._extract_categorization(coin_id)
            features.extend(categorization)

            # 7. Fundamentals (REMOVED - now from Binance GPU service)
            # Previously 25D, now 0D here, 18D from Binance provider

            # 8. Dynamic Clustering (12D)
            clustering = self._extract_dynamic_clustering(coin_id)
            features.extend(clustering)

            # 9. Macro Factors (20D - reduced from 25D)
            macro_factors = self._extract_macro_factors()
            features.extend(macro_factors)

            # 10. Sentiment Analysis (5D)
            sentiment = self._extract_sentiment(coin_id)
            features.extend(sentiment)

        except Exception as e:
            logger.error(f"Error extracting crypto features for {symbol}: {e}")
            # Return zeros on error (graceful degradation)
            features = [0.0] * 126

        # Ensure exactly 126 dimensions
        if len(features) < 126:
            features.extend([0.0] * (126 - len(features)))
        elif len(features) > 126:
            features = features[:126]

        return np.array(features, dtype=np.float32)

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to CoinGecko coin ID format.

        Args:
            symbol: Raw symbol (e.g., "BTC", "BTC/USD", "bitcoin")

        Returns:
            CoinGecko coin ID (e.g., "bitcoin")
        """
        # Remove trading pair suffixes
        symbol = symbol.replace("/USD", "").replace("/USDT", "").strip()

        # Common mappings
        symbol_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDT": "tether",
            "BNB": "binancecoin",
            "SOL": "solana",
            "ADA": "cardano",
            "DOGE": "dogecoin",
            "XRP": "ripple",
            "MATIC": "matic-network",
            "DOT": "polkadot",
        }

        return symbol_map.get(symbol.upper(), symbol.lower())

    def _fetch_with_fallback(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        provider: Optional[DataProvider] = None,
    ) -> Optional[Dict]:
        """
        Fetch data with multi-provider fallback.

        Tries providers in round-robin order:
        1. Current provider (based on last successful call)
        2. Next provider (if current fails)
        3. Final fallback (if both fail)

        Args:
            endpoint: API endpoint (provider-specific)
            params: Query parameters
            provider: Force specific provider (optional)

        Returns:
            JSON response or None on failure
        """
        providers_to_try = (
            [provider]
            if provider
            else [
                DataProvider.COINGECKO,
                DataProvider.CRYPTOCOMPARE,
                DataProvider.COINMARKETCAP,
            ]
        )

        for prov in providers_to_try:
            try:
                data = self._fetch_from_provider(endpoint, params, prov)
                if data:
                    self.current_provider = prov  # Update round-robin state
                    self.provider_failures[prov] = 0  # Reset failure count
                    return data
            except Exception as e:
                self.provider_failures[prov] += 1
                logger.warning(f"Provider {prov.value} failed: {e}")
                continue

        logger.error(f"All providers failed for endpoint: {endpoint}")
        return None

    def _fetch_from_provider(
        self,
        endpoint: str,
        params: Optional[Dict],
        provider: DataProvider,
    ) -> Optional[Dict]:
        """
        Fetch data from a specific provider.

        Args:
            endpoint: Provider-specific endpoint
            params: Query parameters
            provider: Which provider to use

        Returns:
            JSON response or None
        """
        if provider == DataProvider.COINGECKO:
            return self._fetch_coingecko(endpoint, params)
        elif provider == DataProvider.CRYPTOCOMPARE:
            return self._fetch_cryptocompare(endpoint, params)
        elif provider == DataProvider.COINMARKETCAP:
            return self._fetch_coinmarketcap(endpoint, params)
        else:
            return None

    def _fetch_coingecko(self, endpoint: str, params: Optional[Dict]) -> Optional[Dict]:
        """Fetch from CoinGecko API."""
        config = ProviderConfig.COINGECKO
        url = f"{config['base_url']}/{endpoint}"

        if params is None:
            params = {}
        params["x_cg_demo_api_key"] = config["api_key"]

        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_cryptocompare(
        self, endpoint: str, params: Optional[Dict]
    ) -> Optional[Dict]:
        """Fetch from CryptoCompare API."""
        config = ProviderConfig.CRYPTOCOMPARE

        if config["api_key"] is None:
            logger.warning("CryptoCompare API key not configured, skipping")
            return None

        url = f"{config['base_url']}/{endpoint}"

        if params is None:
            params = {}
        params["api_key"] = config["api_key"]

        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_coinmarketcap(
        self, endpoint: str, params: Optional[Dict]
    ) -> Optional[Dict]:
        """Fetch from CoinMarketCap API."""
        config = ProviderConfig.COINMARKETCAP

        if config["api_key"] is None:
            logger.warning("CoinMarketCap API key not configured, skipping")
            return None

        url = f"{config['base_url']}/{endpoint}"
        headers = {"X-CMC_PRO_API_KEY": config["api_key"]}

        response = self.session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Feature Extraction Methods (156D)
    # =========================================================================

    def _extract_token_metadata(self, coin_id: str) -> List[float]:
        """
        Extract token metadata features (18D).

        Features:
        - Number of holders (normalized)
        - Smart contract audits (binary flags)
        - Token age (days since creation)
        - Legitimacy signals (governance, KYC, legal)
        - GitHub stars/forks (developer interest)
        - Whitepaper availability

        Returns:
            18D feature vector
        """
        features = []

        try:
            # Fetch detailed coin data from CoinGecko
            data = self._fetch_with_fallback(f"coins/{coin_id}")

            if data:
                # 1. Market cap rank (inverse - lower is better)
                rank = data.get("market_cap_rank", 1000)
                features.append(1.0 / rank if rank > 0 else 0.0)

                # 2-3. Developer activity
                dev_data = data.get("developer_data", {})
                features.append(float(dev_data.get("forks", 0)) / 1000.0)  # Normalized
                features.append(float(dev_data.get("stars", 0)) / 1000.0)

                # 4-5. Community metrics
                community_data = data.get("community_data", {})
                features.append(
                    float(community_data.get("twitter_followers", 0)) / 100000.0
                )
                features.append(
                    float(community_data.get("reddit_subscribers", 0)) / 10000.0
                )

                # 6. Liquidity score
                features.append(float(data.get("liquidity_score", 0.0)))

                # 7. Public interest score
                features.append(float(data.get("public_interest_score", 0.0)))

                # 8. Coingecko score
                features.append(float(data.get("coingecko_score", 0.0)) / 100.0)

                # 9. Developer score
                features.append(float(data.get("developer_score", 0.0)) / 100.0)

                # 10. Community score
                features.append(float(data.get("community_score", 0.0)) / 100.0)

                # 11-18. Placeholder for additional metadata
                # (Contract audits, holder counts, etc. - may require paid tier)
                features.extend([0.0] * 8)
            else:
                features = [0.0] * 18

        except Exception as e:
            logger.warning(f"Failed to extract token metadata for {coin_id}: {e}")
            features = [0.0] * 18

        return features[:18]

    def _extract_onchain_pools(self, coin_id: str) -> List[float]:
        """Extract onchain liquidity pool features (14D)."""
        # Placeholder - requires onchain data APIs (Dune, Flipside, etc.)
        return [0.0] * 14

    def _extract_crypto_market(self, coin_id: str) -> List[float]:
        """
        Extract crypto market features (9D).

        Features:
        - BTC dominance
        - Market cap (total crypto)
        - 24h volume (total crypto)
        - Altcoin season indicator
        - Market cap change (24h)
        - DeFi market cap
        - NFT market cap
        - Stablecoin market cap
        - Active markets count

        Returns:
            9D feature vector
        """
        features = []

        try:
            # Fetch global market data
            global_data = self._fetch_with_fallback("global")

            if global_data and "data" in global_data:
                data = global_data["data"]

                # 1. BTC dominance
                btc_dom = data.get("market_cap_percentage", {}).get("btc", 56.0)
                features.append(btc_dom / 100.0)

                # 2. Total market cap (normalized by $1T)
                total_mc = data.get("total_market_cap", {}).get("usd", 0)
                features.append(total_mc / 1e12)

                # 3. 24h volume (normalized by $100B)
                total_vol = data.get("total_volume", {}).get("usd", 0)
                features.append(total_vol / 1e11)

                # 4. Altcoin season (1 - BTC dominance)
                features.append(1.0 - (btc_dom / 100.0))

                # 5. Market cap change (24h)
                mc_change = data.get("market_cap_change_percentage_24h_usd", 0.0)
                features.append(mc_change / 100.0)

                # 6. Active cryptocurrencies
                active_cryptos = data.get("active_cryptocurrencies", 0)
                features.append(active_cryptos / 20000.0)  # Normalized

                # 7-9. DeFi/NFT/Stablecoin market caps (placeholder - may require separate endpoints)
                features.extend([0.0] * 3)
            else:
                features = [0.0] * 9

        except Exception as e:
            logger.warning(f"Failed to extract crypto market features: {e}")
            features = [0.0] * 9

        return features[:9]

    def _extract_exchanges_derivatives(self, coin_id: str) -> List[float]:
        """Extract exchange and derivatives features (11D)."""
        # Placeholder - requires exchange API integrations
        return [0.0] * 11

    def _extract_public_treasuries(self, coin_id: str) -> List[float]:
        """Extract public treasury holdings features (6D)."""
        # Placeholder - requires BitcoinTreasuries.org API or similar
        return [0.0] * 6

    def _extract_categorization(self, coin_id: str) -> List[float]:
        """
        Extract categorization features (31D).

        Features:
        - One-hot encoding for primary category (Layer 1, Layer 2, DeFi, etc.)
        - Multi-label encoding for secondary categories
        - Consensus type (PoW, PoS, PoA, etc.)

        Returns:
            31D feature vector
        """
        features = []

        try:
            # Fetch coin categories
            data = self._fetch_with_fallback(f"coins/{coin_id}")

            if data:
                categories = data.get("categories", [])

                # Define major categories
                major_categories = [
                    "Layer 1 (L1)",
                    "Layer 2 (L2)",
                    "Decentralized Finance (DeFi)",
                    "Non-Fungible Tokens (NFT)",
                    "Meme",
                    "Exchange-based Tokens",
                    "Smart Contract Platform",
                    "Stablecoins",
                    "Privacy Coins",
                    "Governance",
                ]

                # One-hot encoding (first 10D)
                for cat in major_categories:
                    features.append(1.0 if cat in categories else 0.0)

                # Placeholder for additional categorization (21D)
                features.extend([0.0] * 21)
            else:
                features = [0.0] * 31

        except Exception as e:
            logger.warning(f"Failed to extract categorization for {coin_id}: {e}")
            features = [0.0] * 31

        return features[:31]

    def _extract_fundamentals(self, coin_id: str) -> List[float]:
        """
        Extract fundamental features (25D).

        Features:
        - Supply metrics (circulating, total, max)
        - Market cap metrics
        - Price metrics (ATH, ATL, price change %)
        - Volume metrics
        - Developer activity

        Returns:
            25D feature vector
        """
        features = []

        try:
            data = self._fetch_with_fallback(f"coins/{coin_id}")

            if data:
                market_data = data.get("market_data", {})

                # 1. Circulating supply (normalized by max supply)
                circ_supply = market_data.get("circulating_supply", 0)
                max_supply = market_data.get("max_supply", circ_supply)
                features.append(circ_supply / max_supply if max_supply > 0 else 1.0)

                # 2. Market cap (normalized by $1T)
                mc_usd = market_data.get("market_cap", {}).get("usd", 0)
                features.append(mc_usd / 1e12)

                # 3. Fully diluted valuation (normalized by $1T)
                fdv = market_data.get("fully_diluted_valuation", {}).get("usd", 0)
                features.append(fdv / 1e12)

                # 4. Price change (7d)
                price_change_7d = market_data.get("price_change_percentage_7d", 0.0)
                features.append(price_change_7d / 100.0)

                # 5. Price change (30d)
                price_change_30d = market_data.get("price_change_percentage_30d", 0.0)
                features.append(price_change_30d / 100.0)

                # 6. ATH change %
                ath_change = market_data.get("ath_change_percentage", {}).get("usd", 0)
                features.append(ath_change / 100.0)

                # 7. ATL change %
                atl_change = market_data.get("atl_change_percentage", {}).get("usd", 0)
                features.append(atl_change / 100.0)

                # 8-25. Additional fundamental metrics (placeholder)
                features.extend([0.0] * 18)
            else:
                features = [0.0] * 25

        except Exception as e:
            logger.warning(f"Failed to extract fundamentals for {coin_id}: {e}")
            features = [0.0] * 25

        return features[:25]

    def _extract_dynamic_clustering(self, coin_id: str) -> List[float]:
        """Extract dynamic clustering features (12D)."""
        # Placeholder - requires correlation matrix computation
        return [0.0] * 12

    def _extract_macro_factors(self) -> List[float]:
        """Extract macro economic factors (20D - reduced from 25D)."""
        # Placeholder - requires macro economic data APIs
        # (Federal Reserve API, Bloomberg, etc.)
        # Reduced from 25D to 20D to make room for Binance GPU fundamentals (18D)
        return [0.0] * 20

    def _extract_sentiment(self, coin_id: str) -> List[float]:
        """
        Extract sentiment features (5D).

        Features:
        - Fear & Greed Index
        - Social media sentiment (Twitter, Reddit)
        - News sentiment
        - Search trends

        Returns:
            5D feature vector
        """
        # Placeholder - requires sentiment APIs
        # (Alternative.me Fear & Greed, LunarCrush, etc.)
        return [0.0] * 5

    def close(self):
        """Close HTTP session."""
        if self.session:
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


class BinanceGPUFundamentalsExtractor:
    """
    Extracts GPU-accelerated fundamentals from Binance provider (18D).

    Fetches from Binance HTTP provider (port 9004) which uses GPU-computed
    indicators for ~65x performance improvement vs Python.

    Features (18D):
    - ATH metrics (6D): current_price, ath, distance%, atl, distance%, metadata
    - Volatility (2D): 30d, 7d
    - Sharpe (1D): 30d risk-adjusted return
    - SMA (3D): 7, 30, 90 day
    - RSI (1D): GPU-computed
    - MACD (3D): line, signal, histogram (GPU-computed)
    - ATR (1D): GPU-computed
    - ADX (1D): GPU-computed
    """

    BINANCE_PROVIDER_URL = "http://10.32.3.27:9004"

    def __init__(self, enable_cache: bool = True, cache_ttl: int = 60):
        """Initialize Binance fundamentals extractor."""
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.session = requests.Session()

    def extract(self, symbol: str, timestamp: Optional[datetime] = None) -> np.ndarray:
        """
        Extract Binance GPU fundamentals for a symbol (18D).

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH", "BTCUSDT")
            timestamp: Not used (for compatibility with other extractors)

        Returns:
            18D feature vector with GPU-accelerated indicators
        """
        try:
            # Normalize symbol to Binance format (e.g., BTC -> BTCUSDT)
            binance_symbol = self._normalize_symbol(symbol)

            # Fetch from Binance provider
            response = self.session.get(
                f"{self.BINANCE_PROVIDER_URL}/fundamentals/{binance_symbol}", timeout=5
            )
            response.raise_for_status()
            data = response.json()

            # Extract 18D features
            features = [
                # ATH metrics (6D)
                data.get("current_price", 0.0),
                data.get("ath", 0.0),
                data.get("distance_from_ath_percent", 0.0),
                data.get("atl", 0.0),
                data.get("distance_from_atl_percent", 0.0),
                0.0,  # metadata placeholder
                # Volatility (2D)
                data.get("volatility_30d", 0.0),
                data.get("volatility_7d", 0.0),
                # Sharpe (1D)
                data.get("sharpe_ratio_30d") or 0.0,
                # SMA (3D)
                data.get("sma_7", 0.0),
                data.get("sma_30", 0.0),
                data.get("sma_90", 0.0),
                # GPU-computed indicators (6D)
                data.get("rsi_14", 0.0),
                data.get("macd_line", 0.0) if "macd_line" in data else 0.0,
                data.get("macd_signal", 0.0) if "macd_signal" in data else 0.0,
                data.get("macd_histogram", 0.0) if "macd_histogram" in data else 0.0,
                data.get("atr_14", 0.0) if "atr_14" in data else 0.0,
                data.get("adx_14", 0.0) if "adx_14" in data else 0.0,
            ]

            logger.debug(
                f"Fetched Binance GPU fundamentals for {binance_symbol}: {len(features)}D"
            )
            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to fetch Binance fundamentals for {symbol}: {e}")
            # Return zeros on error (graceful degradation)
            return np.zeros(18, dtype=np.float32)

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to Binance format.

        Examples:
            BTC -> BTC
            bitcoin -> BTC
            BTCUSDT -> BTC
        """
        # Common mappings
        symbol_map = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "binancecoin": "BNB",
            "solana": "SOL",
            "cardano": "ADA",
            "polkadot": "DOT",
            "avalanche": "AVAX",
            "polygon": "MATIC",
            "chainlink": "LINK",
            "litecoin": "LTC",
        }

        symbol_lower = symbol.lower()

        # Check if it's a CoinGecko ID
        if symbol_lower in symbol_map:
            return symbol_map[symbol_lower]

        # Remove USDT suffix if present
        if symbol.upper().endswith("USDT"):
            return symbol[:-4].upper()

        # Return as-is (assume it's already in correct format)
        return symbol.upper()

    def close(self):
        """Close HTTP session."""
        if self.session:
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
