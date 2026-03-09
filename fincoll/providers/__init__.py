"""
Data Providers for FinColl

All data access goes through MultiProviderFetcher — the single gateway.
Underlying providers are registered at server startup.

Provider hierarchy:
- BaseTradingProvider: abstract base for all new-style providers
- MultiProviderFetcher: gateway — tries providers in priority order
  - TradeStationTradingProvider (primary, added when credentials available)
  - AlpacaTradingProvider (fallback)
  - PublicTradingProvider (fallback)
  - YFinanceProviderWrapper (last resort, via multi_provider_fetcher.py)
"""

from .base_trading_provider import BaseTradingProvider
from .multi_provider_fetcher import DataType, MultiProviderFetcher

__all__ = [
    "BaseTradingProvider",
    "MultiProviderFetcher",
    "DataType",
]

# Optional providers (require pim-api-clients SDK)
try:
    from .tradestation_trading_provider import TradeStationTradingProvider

    __all__.append("TradeStationTradingProvider")
except ImportError:
    pass

try:
    from .alpaca_trading_provider import AlpacaTradingProvider

    __all__.append("AlpacaTradingProvider")
except ImportError:
    pass

try:
    from .public_trading_provider import PublicTradingProvider

    __all__.append("PublicTradingProvider")
except ImportError:
    pass
