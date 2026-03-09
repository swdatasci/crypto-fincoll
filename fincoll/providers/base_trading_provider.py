"""Base class for trading data providers."""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
import pytz

import pandas as pd


class CircuitBreaker:
    """Simple circuit breaker to prevent cascading failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self._open = False

    def is_open(self) -> bool:
        """Check if circuit breaker is open (tripped)."""
        if not self._open:
            return False

        # Check if recovery timeout has passed
        if (
            self.last_failure_time
            and (time.time() - self.last_failure_time) > self.recovery_timeout
        ):
            self._open = False
            self.failures = 0
            return False

        return True

    def record_failure(self) -> None:
        """Record a failure and potentially trip the circuit."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self._open = True

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failures = 0


class BaseTradingProvider(ABC):
    """
    Abstract base class for all trading data providers.

    Implements circuit breaker pattern for resilience.
    Rate limiting is handled by the underlying pim-api-clients SDKs.
    """

    def __init__(self, name: str):
        self.name = name
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

    def get_historical_bars(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bar_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Safely get historical bars with circuit breaker protection.
        """
        if self.circuit_breaker.is_open():
            raise ConnectionError(f"Circuit breaker open for {self.name}")

        try:
            result = self._get_historical_bars(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                bar_count=bar_count,
            )
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

    @abstractmethod
    def _get_historical_bars(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bar_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Implementation-specific method to fetch historical bars.
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        """
        pass

    def is_healthy(self) -> bool:
        """Check if provider is healthy (circuit breaker not tripped)."""
        return not self.circuit_breaker.is_open()

    def is_market_open(self) -> bool:
        """
        Check whether the US equity market is currently open.

        Pure time-based check (NYSE regular hours, Mon–Fri 09:30–16:00 ET).
        Does not account for early closes or market holidays.
        """
        try:
            et = pytz.timezone("US/Eastern")
            now_et = datetime.now(et)
            if now_et.weekday() >= 5:  # Saturday or Sunday
                return False
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= now_et <= market_close
        except Exception:
            return False

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get a quote dict for *symbol*.

        Default implementation wraps ``get_current_price()`` into the standard
        quote schema.  Providers that can return richer data (bid/ask/volume)
        should override this method.

        Returns a dict with at least::

            {
                "symbol": str,
                "price":  float | None,
                "bid":    float | None,
                "ask":    float | None,
                "volume": float | None,
                "provider": str,
            }
        """
        try:
            price = self.get_current_price(symbol)
        except Exception:
            price = None

        return {
            "symbol": symbol,
            "price": price,
            "bid": None,
            "ask": None,
            "volume": None,
            "provider": self.name,
        }
