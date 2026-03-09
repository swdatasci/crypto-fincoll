"""
SenVec Cache Controller

Manages SenVec cache refresh based on market hours:
- Starts cache when market opens
- Stops cache when market closes
- Monitors for rate limit errors
- Manages symbol lists
"""

import httpx
import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict
from ..providers.tradestation_trading_provider import TradeStationTradingProvider

logger = logging.getLogger(__name__)


class SenVecController:
    """
    Controls SenVec cache refresh based on market hours.

    Uses TradeStation API to detect market open/close and holidays.
    Sends control signals to SenVec aggregator via REST API.
    """

    def __init__(
        self,
        senvec_url: Optional[str] = None,
        check_interval: int = 60,  # Check market status every 60 seconds
        cache_interval: int = 300,  # SenVec cache refresh interval
        symbols: Optional[List[str]] = None,
    ):
        """
        Initialize SenVec controller

        Args:
            senvec_url: Base URL for SenVec aggregator service
            check_interval: How often to check market status (seconds)
            cache_interval: How often SenVec should refresh cache (seconds)
            symbols: List of symbols to cache (None = use SenVec's default universe)
        """
        # Read from environment if not provided
        senvec_url = senvec_url or os.getenv("SENVEC_URL", "http://localhost:18000")
        self.senvec_url = senvec_url.rstrip("/")
        self.check_interval = check_interval
        self.cache_interval = cache_interval
        self.symbols = symbols

        # TradeStationTradingProvider for market hours (new-style provider)
        self.ts_provider = TradeStationTradingProvider()

        # State tracking
        self._running = False
        self._cache_enabled = False
        self._last_market_status = None

        self.client = httpx.AsyncClient(timeout=30.0)

    async def start(self):
        """Start the controller loop"""
        self._running = True
        logger.info("=" * 60)
        logger.info("SenVec Cache Controller Starting")
        logger.info(f"  SenVec URL: {self.senvec_url}")
        logger.info(f"  Check interval: {self.check_interval}s")
        logger.info(f"  Cache interval: {self.cache_interval}s")
        if self.symbols:
            logger.info(f"  Symbols: {len(self.symbols)} specified")
        else:
            logger.info("  Symbols: Using SenVec default universe")
        logger.info("=" * 60)

        # Start monitoring loop
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the controller and disable SenVec cache"""
        self._running = False

        # Stop SenVec cache if it's running
        if self._cache_enabled:
            await self._disable_senvec_cache()

        await self.client.aclose()
        logger.info("SenVec Cache Controller Stopped")

    async def _monitor_loop(self):
        """Main monitoring loop - checks market status periodically"""
        while self._running:
            try:
                # Check if TradeStation is available
                if not self.ts_provider.is_healthy():
                    logger.warning(
                        "TradeStation not healthy - cannot check market hours"
                    )
                    await asyncio.sleep(self.check_interval)
                    continue

                # Check current market status
                is_open = self.ts_provider.is_market_open()

                # Detect market status changes
                if is_open != self._last_market_status:
                    if is_open:
                        logger.info("🟢 Market OPENED - Enabling SenVec cache")
                        await self._enable_senvec_cache()
                    else:
                        logger.info("🔴 Market CLOSED - Disabling SenVec cache")
                        await self._disable_senvec_cache()

                    self._last_market_status = is_open

                # Periodically check SenVec status for errors (every 5 minutes when market open)
                if is_open and self._cache_enabled:
                    await self._check_senvec_status()

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in SenVec controller loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def _enable_senvec_cache(self):
        """Enable SenVec cache refresh"""
        try:
            url = f"{self.senvec_url}/cache/control/start"

            payload = {"interval_seconds": self.cache_interval, "skip_on_error": True}

            if self.symbols:
                payload["symbols"] = self.symbols

            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.info(
                f"✅ SenVec cache enabled: {result.get('symbols_count', 0)} symbols"
            )
            self._cache_enabled = True

        except Exception as e:
            logger.error(f"Failed to enable SenVec cache: {e}")
            self._cache_enabled = False

    async def _disable_senvec_cache(self):
        """Disable SenVec cache refresh"""
        try:
            url = f"{self.senvec_url}/cache/control/stop"

            response = await self.client.post(url)
            response.raise_for_status()

            result = response.json()
            logger.info(
                f"⏸️  SenVec cache disabled: {result.get('symbols_refreshed', 0)} symbols refreshed total"
            )
            self._cache_enabled = False

        except Exception as e:
            logger.error(f"Failed to disable SenVec cache: {e}")

    async def _check_senvec_status(self):
        """Check SenVec status and log any rate limit errors"""
        try:
            url = f"{self.senvec_url}/cache/control/status"

            response = await self.client.get(url)
            response.raise_for_status()

            status = response.json()

            # Check for rate limit errors
            rate_limit_errors = status.get("rate_limit_errors", [])
            if rate_limit_errors:
                logger.warning("=" * 60)
                logger.warning(
                    f"⚠️  RATE LIMIT ERRORS DETECTED: {len(rate_limit_errors)} errors"
                )
                for error in rate_limit_errors[-3:]:  # Show last 3
                    logger.warning(
                        f"  Symbol: {error.get('symbol')}, Time: {error.get('timestamp')}"
                    )
                    logger.warning(f"  Error: {error.get('error')}")
                logger.warning("=" * 60)

            # Log normal status every 5 checks (5 minutes with 60s interval)
            if status.get("cycle_count", 0) % 5 == 0:
                logger.info(
                    f"SenVec status: {status.get('symbols_refreshed', 0)} symbols, "
                    f"{status.get('errors', 0)} errors, "
                    f"Cycle #{status.get('cycle_count', 0)}"
                )

        except Exception as e:
            logger.error(f"Failed to check SenVec status: {e}")

    async def update_symbols(self, symbols: List[str], action: str = "replace"):
        """
        Update the symbol list in SenVec cache

        Args:
            symbols: List of symbols
            action: "replace", "add", or "remove"
        """
        try:
            url = f"{self.senvec_url}/cache/control/symbols"

            payload = {"symbols": symbols, "action": action}

            response = await self.client.put(url, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.info(
                f"📝 Updated SenVec symbols ({action}): {result.get('symbols_count', 0)} total"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to update SenVec symbols: {e}")
            raise

    async def get_status(self) -> Dict:
        """
        Get current SenVec cache status

        Returns:
            Dict with cache status information
        """
        try:
            url = f"{self.senvec_url}/cache/control/status"

            response = await self.client.get(url)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Failed to get SenVec status: {e}")
            return {"error": str(e), "enabled": False, "running": False}
