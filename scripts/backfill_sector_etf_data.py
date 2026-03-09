#!/usr/bin/env python3
"""
Backfill Sector ETF Historical Data

This script backfills historical data for all sector ETFs used in feature extraction.
Sector ETFs are used to calculate relative performance features (alpha vs sector).

Sector ETFs (GICS classification):
- XLK (Technology)
- XLF (Financials)
- XLV (Healthcare)
- XLE (Energy)
- XLY (Consumer Cyclical)
- XLP (Consumer Defensive)
- XLI (Industrials)
- XLU (Utilities)
- XLRE (Real Estate)
- XLB (Materials)
- XLC (Communication Services)

Usage:
    python scripts/backfill_sector_etf_data.py --days 365 --provider tradestation
    python scripts/backfill_sector_etf_data.py --days 90 --provider alpaca
    python scripts/backfill_sector_etf_data.py --etfs XLV XLK XLF --days 180
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.providers.tradestation_provider import TradeStationTradingProvider
from fincoll.providers.alpaca_provider import AlpacaTradingProvider

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# All sector ETFs used in feature extraction
SECTOR_ETFS = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Healthcare
    "XLE",  # Energy
    "XLY",  # Consumer Cyclical
    "XLP",  # Consumer Defensive
    "XLI",  # Industrials
    "XLU",  # Utilities
    "XLRE",  # Real Estate
    "XLB",  # Materials
    "XLC",  # Communication Services
]


def backfill_sector_etfs(
    etfs: list[str],
    days: int,
    provider_name: str = "tradestation",
    interval: str = "1d",
):
    """
    Backfill historical data for sector ETFs

    Args:
        etfs: List of ETF symbols to backfill
        days: Number of days of history to fetch
        provider_name: Data provider to use ('tradestation' or 'alpaca')
        interval: Bar interval (default '1d' for daily)
    """
    # Initialize provider
    if provider_name.lower() == "tradestation":
        logger.info("🔌 Initializing TradeStation provider...")
        provider = TradeStationTradingProvider()
    elif provider_name.lower() == "alpaca":
        logger.info("🔌 Initializing Alpaca provider...")
        provider = AlpacaTradingProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(
        f"📅 Backfilling {len(etfs)} sector ETFs from {start_date.date()} to {end_date.date()}"
    )
    logger.info(f"📊 Provider: {provider_name}, Interval: {interval}")

    results = {"success": [], "failed": [], "total_bars": 0}

    for etf in etfs:
        try:
            logger.info(f"📈 Fetching {etf}...")

            # Fetch historical data
            data = provider.get_historical_bars(
                symbol=etf, start_date=start_date, end_date=end_date, interval=interval
            )

            if data is not None and len(data) > 0:
                bars_count = len(data)
                results["success"].append(etf)
                results["total_bars"] += bars_count
                logger.info(f"  ✅ {etf}: {bars_count} bars fetched")

                # Show sample of data
                logger.info(f"  📊 Date range: {data.index[0]} to {data.index[-1]}")
                logger.info(f"  💰 Latest close: ${data['close'].iloc[-1]:.2f}")
            else:
                results["failed"].append(etf)
                logger.warning(f"  ❌ {etf}: No data returned")

        except Exception as e:
            results["failed"].append(etf)
            logger.error(f"  ❌ {etf}: Failed - {e}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 BACKFILL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successful: {len(results['success'])}/{len(etfs)}")
    logger.info(f"❌ Failed: {len(results['failed'])}/{len(etfs)}")
    logger.info(f"📈 Total bars fetched: {results['total_bars']:,}")

    if results["success"]:
        logger.info(f"\n✅ Success: {', '.join(results['success'])}")

    if results["failed"]:
        logger.warning(f"\n❌ Failed: {', '.join(results['failed'])}")

    logger.info("\n💡 Note: Data is cached in the FeatureExtractor class-level cache")
    logger.info("   and will be reused across all feature extraction calls.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical data for sector ETFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill all sector ETFs for 1 year using TradeStation
  python scripts/backfill_sector_etf_data.py --days 365
  
  # Backfill specific ETFs for 90 days using Alpaca
  python scripts/backfill_sector_etf_data.py --etfs XLV XLK XLF --days 90 --provider alpaca
  
  # Backfill all ETFs for 2 years
  python scripts/backfill_sector_etf_data.py --days 730
        """,
    )

    parser.add_argument(
        "--etfs",
        nargs="+",
        default=SECTOR_ETFS,
        help="Sector ETFs to backfill (default: all 11 sector ETFs)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to fetch (default: 365)",
    )

    parser.add_argument(
        "--provider",
        choices=["tradestation", "alpaca"],
        default="tradestation",
        help="Data provider to use (default: tradestation)",
    )

    parser.add_argument(
        "--interval", default="1d", help="Bar interval (default: 1d for daily bars)"
    )

    args = parser.parse_args()

    # Run backfill
    results = backfill_sector_etfs(
        etfs=args.etfs,
        days=args.days,
        provider_name=args.provider,
        interval=args.interval,
    )

    # Exit with error code if any failed
    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
