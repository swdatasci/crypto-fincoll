#!/usr/bin/env python3
"""
Finnhub Cache Warming Script

Pre-populates Finnhub fundamental data cache for active symbols to prevent
first-request timeouts. Run this periodically (e.g., daily) to keep cache warm.

Usage:
    python scripts/warm_finnhub_cache.py --symbols AAPL,MSFT,GOOGL
    python scripts/warm_finnhub_cache.py --file symbols.txt
    python scripts/warm_finnhub_cache.py --universe pim  # Load from PIM universe

Features:
    - Parallel cache warming (default: 5 concurrent)
    - Rate-limited (respects Finnhub free tier: 60 req/min)
    - Progress tracking
    - Error reporting
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List
import time

# Add fincoll to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.features.feature_extractor import FeatureExtractor
from fincoll.config.dimensions import DIMS
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


async def warm_symbol(
    extractor: FeatureExtractor, symbol: str, semaphore: asyncio.Semaphore
) -> tuple[str, bool, str]:
    """
    Warm cache for a single symbol.

    Returns:
        (symbol, success, message)
    """
    async with semaphore:
        try:
            # Call all three Finnhub endpoints to populate cache
            # These methods use _get_cached_or_fetch which will populate cache
            start_time = time.time()

            # Create dummy OHLCV data (not used for fundamentals)
            dummy_df = pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp.now()],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000000],
                }
            )

            # Extract fundamentals (this will cache)
            features = await asyncio.to_thread(
                extractor._extract_finnhub_fundamentals, symbol
            )

            elapsed = time.time() - start_time

            if np.any(features != 0):
                return (symbol, True, f"cached in {elapsed:.2f}s")
            else:
                return (symbol, False, f"returned zeros in {elapsed:.2f}s (no data)")

        except Exception as e:
            return (symbol, False, f"failed: {e}")


async def warm_cache_batch(symbols: List[str], max_concurrent: int = 5):
    """
    Warm cache for multiple symbols with rate limiting.

    Finnhub free tier: 60 req/min → ~1 req/sec safe limit
    Each symbol makes 3 API calls (earnings, insider, analyst)
    So max ~20 symbols/min = 1 symbol every 3 seconds

    With max_concurrent=5 and proper delays, we stay under limit.
    """
    extractor = FeatureExtractor(
        enable_senvec=False,  # Don't need SenVec for fundamentals
        enable_finnhub=True,
        enable_futures=False,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(
        f"Warming cache for {len(symbols)} symbols (max {max_concurrent} concurrent)..."
    )

    tasks = [warm_symbol(extractor, symbol, semaphore) for symbol in symbols]

    # Gather with progress tracking
    success_count = 0
    fail_count = 0

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        symbol, success, message = await coro

        if success:
            logger.info(f"[{i}/{len(symbols)}] ✅ {symbol}: {message}")
            success_count += 1
        else:
            logger.warning(f"[{i}/{len(symbols)}] ❌ {symbol}: {message}")
            fail_count += 1

        # Rate limit: Wait 3 seconds between symbols (20 symbols/min)
        if i < len(symbols):
            await asyncio.sleep(3)

    logger.info(f"Cache warming complete: {success_count} success, {fail_count} failed")


def load_pim_universe() -> List[str]:
    """Load symbols from PIM universe configuration"""
    try:
        # Try to import from PIM
        pim_path = Path(__file__).parent.parent.parent / "PassiveIncomeMaximizer"
        sys.path.insert(0, str(pim_path))

        from engine.config.symbol_universe import SYMBOL_UNIVERSE

        return SYMBOL_UNIVERSE
    except Exception as e:
        logger.error(f"Failed to load PIM universe: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Warm Finnhub cache for active symbols"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing symbols (one per line)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=["pim"],
        help="Load symbols from a predefined universe (pim = PassiveIncomeMaximizer)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Max concurrent API calls (default: 5)",
    )

    args = parser.parse_args()

    # Load symbols
    symbols = []

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.file:
        with open(args.file) as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
    elif args.universe == "pim":
        symbols = load_pim_universe()
    else:
        parser.print_help()
        sys.exit(1)

    if not symbols:
        logger.error("No symbols to warm!")
        sys.exit(1)

    logger.info(f"Loaded {len(symbols)} symbols")

    # Run warming
    asyncio.run(warm_cache_batch(symbols, max_concurrent=args.concurrent))


if __name__ == "__main__":
    main()
