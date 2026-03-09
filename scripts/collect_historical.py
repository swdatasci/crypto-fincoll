#!/usr/bin/env python3
"""
Unified Historical Data Collector

Smart symbol selection + OHLCV download from yfinance → InfluxDB.
Fully non-interactive: safe for nohup/background use.

Usage examples:
  # Auto-select 50 diversified symbols, 2020-present
  python scripts/collect_historical.py --auto 50

  # Full tradable universe
  python scripts/collect_historical.py --universe tradable

  # Specific sectors
  python scripts/collect_historical.py --sectors tech,healthcare --count-per-sector 5

  # Manual symbol list
  python scripts/collect_historical.py --symbols AAPL MSFT NVDA

  # Estimate only (no download)
  python scripts/collect_historical.py --universe tradable --estimate-only

  # Skip confirmation prompt
  python scripts/collect_historical.py --auto 50 --yes
"""

import sys
import random
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Load .env from fincoll repo root (provides INFLUXDB_TOKEN etc.)
# ---------------------------------------------------------------------------
def _load_dotenv():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    import os

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key not in os.environ:  # don't override explicit env vars
                os.environ[key] = val


_load_dotenv()


# ---------------------------------------------------------------------------
# Symbol universe - imported directly from finvec (no torch dependency chain)
# ---------------------------------------------------------------------------
# Walk up from scripts/ → fincoll-repo/ → ss/ → finvec/configs
# Works both when running from caelum-supersystem/fincoll and ss/fincoll
# because both are the same NFS directory.
def _find_finvec_configs() -> Path:
    """Return path to finvec/configs regardless of how this script was invoked."""
    script_dir = Path(__file__).resolve().parent
    # Try relative paths at various depths
    candidates = [
        script_dir.parents[1]
        / "finvec"
        / "configs",  # ss/fincoll/scripts → ss/finvec/configs
        script_dir.parents[2] / "finvec" / "configs",  # deeper nesting
        Path("/home/rford/caelum/ss/finvec/configs"),  # absolute fallback
    ]
    for p in candidates:
        if (p / "data_symbols.py").exists():
            return p
    return candidates[-1]  # last resort


_FINVEC_CONFIGS = _find_finvec_configs()
if str(_FINVEC_CONFIGS) not in sys.path:
    sys.path.insert(0, str(_FINVEC_CONFIGS))

try:
    from data_symbols import (
        get_symbol_list,
        get_custom_symbol_list,
        TRADABLE_SYMBOLS,
        DIVERSIFIED_SYMBOLS,
        FULL_MARKET_SYMBOLS,
        TECH_SYMBOLS,
    )

    _SYMBOLS_AVAILABLE = True
except ImportError:
    _SYMBOLS_AVAILABLE = False
    # Fallback mini-universe if finvec is not on the path
    TRADABLE_SYMBOLS = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "JPM",
        "BAC",
        "JNJ",
        "UNH",
        "XOM",
        "CVX",
        "SPY",
        "QQQ",
    ]

    def get_symbol_list(category="diversified"):
        return TRADABLE_SYMBOLS

    def get_custom_symbol_list(**kwargs):
        return TRADABLE_SYMBOLS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("collect_historical")


# ---------------------------------------------------------------------------
# InfluxDB writer (gracefully degrades to CSV when client unavailable)
# ---------------------------------------------------------------------------
class OHLCVStore:
    """Write OHLCV rows to InfluxDB, with CSV fallback."""

    def __init__(
        self,
        influx_url: str = "http://10.32.3.27:8086",
        influx_token: str = "",
        influx_org: str = "caelum",
        influx_bucket: str = "training_data",
        csv_fallback: Optional[Path] = None,
    ):
        self.bucket = influx_bucket
        self.csv_path = csv_fallback
        self._write_api = None
        self._Point = None

        try:
            from influxdb_client import InfluxDBClient, Point, WriteOptions  # type: ignore
            from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore

            client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
            self._write_api = client.write_api(write_options=SYNCHRONOUS)
            self._Point = Point
            self._org = influx_org
            logger.info(
                "InfluxDB writer ready at %s (bucket=%s)", influx_url, influx_bucket
            )
        except Exception as exc:
            logger.warning("InfluxDB not available (%s); using CSV fallback.", exc)
            if csv_fallback is None:
                self.csv_path = Path("/tmp/ohlcv_collection.csv")
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with open(self.csv_path, "w") as f:
                    f.write("symbol,date,open,high,low,close,volume\n")
            logger.info("CSV fallback: %s", self.csv_path)

    @property
    def mode(self) -> str:
        return "influxdb" if self._write_api else "csv"

    def write(self, symbol: str, date: datetime, row: Dict[str, float]) -> bool:
        if self._write_api and self._Point:
            try:
                point = (
                    self._Point("ohlcv")
                    .tag("symbol", symbol)
                    .field("open", row["open"])
                    .field("high", row["high"])
                    .field("low", row["low"])
                    .field("close", row["close"])
                    .field("volume", row["volume"])
                    .time(date)
                )
                self._write_api.write(bucket=self.bucket, org=self._org, record=point)
                return True
            except Exception as exc:
                logger.error("InfluxDB write failed for %s @ %s: %s", symbol, date, exc)
                return False
        else:
            # CSV fallback
            try:
                with open(self.csv_path, "a") as f:
                    f.write(
                        f"{symbol},{date.strftime('%Y-%m-%d')},"
                        f"{row['open']},{row['high']},{row['low']},"
                        f"{row['close']},{row['volume']}\n"
                    )
                return True
            except Exception as exc:
                logger.error("CSV write failed: %s", exc)
                return False


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class HistoricalCollector:
    """Download OHLCV from yfinance and store to InfluxDB/CSV."""

    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 10,
        rate_limit_secs: float = 0.5,
        store: Optional[OHLCVStore] = None,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.rate_limit = rate_limit_secs
        self.store = store or OHLCVStore()

        self.stats: Dict = {
            "success": 0,
            "failed": 0,
            "total_days": 0,
            "failed_symbols": [],
            "start_time": datetime.now(),
        }

    # ------------------------------------------------------------------
    def estimate(self) -> Dict:
        trading_days = len(pd.date_range(self.start_date, self.end_date, freq="B"))
        per_symbol_secs = trading_days * 0.002  # yfinance batch download is fast
        total_secs = len(self.symbols) * (per_symbol_secs + self.rate_limit)
        return {
            "symbols": len(self.symbols),
            "trading_days": trading_days,
            "estimated_hours": total_secs / 3600,
            "estimated_minutes": total_secs / 60,
        }

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        logger.info("=" * 72)
        logger.info("HISTORICAL OHLCV COLLECTION")
        logger.info("  Symbols  : %d", len(self.symbols))
        logger.info(
            "  Range    : %s → %s",
            self.start_date.strftime("%Y-%m-%d"),
            self.end_date.strftime("%Y-%m-%d"),
        )
        logger.info("  Storage  : %s", self.store.mode)
        logger.info("=" * 72)

        batches = [
            self.symbols[i : i + self.batch_size]
            for i in range(0, len(self.symbols), self.batch_size)
        ]

        for b_idx, batch in enumerate(tqdm(batches, desc="Batches", unit="batch")):
            logger.info("Batch %d/%d: %s", b_idx + 1, len(batches), batch)
            for sym in batch:
                self._process(sym)
                time.sleep(self.rate_limit)

        self._print_summary()
        return self.stats

    # ------------------------------------------------------------------
    def _process(self, symbol: str):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                interval="1d",
            )

            if df.empty:
                logger.warning("No data returned for %s", symbol)
                self.stats["failed"] += 1
                self.stats["failed_symbols"].append(symbol)
                return

            days_stored = 0
            for ts, row in df.iterrows():
                dt = ts.to_pydatetime()
                ohlcv = {
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),  # InfluxDB schema: integer
                }
                if self.store.write(symbol, dt, ohlcv):
                    days_stored += 1

            self.stats["success"] += 1
            self.stats["total_days"] += days_stored
            logger.info("  ✓ %-8s  %d days", symbol, days_stored)

        except Exception as exc:
            logger.error("Failed %s: %s", symbol, exc)
            self.stats["failed"] += 1
            self.stats["failed_symbols"].append(symbol)

    # ------------------------------------------------------------------
    def _print_summary(self):
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
        logger.info("=" * 72)
        logger.info("COLLECTION COMPLETE")
        logger.info("  Duration  : %.1f min", elapsed / 60)
        logger.info(
            "  Success   : %d / %d symbols", self.stats["success"], len(self.symbols)
        )
        logger.info("  Failed    : %d", self.stats["failed"])
        logger.info("  Total days: %d", self.stats["total_days"])
        if self.stats["failed_symbols"]:
            logger.info("  Failed syms: %s", self.stats["failed_symbols"])
        logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Symbol selection helpers
# ---------------------------------------------------------------------------
def select_symbols(args) -> List[str]:
    """Resolve symbols from CLI arguments."""
    if args.symbols:
        return args.symbols

    if args.universe:
        return list(get_symbol_list(args.universe))

    if args.sectors:
        sector_list = [s.strip() for s in args.sectors.split(",")]
        n = args.count_per_sector or 5
        return get_custom_symbol_list(sectors=sector_list, include_indices=True)[
            : n * len(sector_list)
        ]

    if args.auto:
        universe = list(get_symbol_list("diversified"))
        n = min(args.auto, len(universe))
        return random.sample(universe, n)

    # Default: full tradable universe
    return list(get_symbol_list("tradable"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect historical OHLCV data (non-interactive, no torch deps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Symbol selection (mutually exclusive) --
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+", metavar="SYM", help="Explicit symbol list")
    sym.add_argument(
        "--universe",
        metavar="NAME",
        choices=[
            "demo",
            "tech",
            "diversified",
            "index",
            "full",
            "tradable",
            "crypto",
            "forex",
            "commodity",
        ],
        help="Named symbol universe from finvec/configs/data_symbols.py",
    )
    sym.add_argument(
        "--sectors", metavar="tech,healthcare,...", help="Comma-separated sector names"
    )
    sym.add_argument(
        "--auto",
        type=int,
        metavar="N",
        help="Randomly pick N symbols from the diversified universe",
    )

    # -- Sector sub-option --
    p.add_argument(
        "--count-per-sector",
        type=int,
        default=5,
        metavar="N",
        help="Symbols per sector when using --sectors (default: 5)",
    )

    # -- Date range --
    p.add_argument(
        "--start-date",
        metavar="YYYY-MM-DD",
        type=lambda d: datetime.strptime(d, "%Y-%m-%d"),
        default=datetime(2020, 1, 1),
        help="Start date (default: 2020-01-01)",
    )
    p.add_argument(
        "--end-date",
        metavar="YYYY-MM-DD or 'today'",
        default="today",
        help="End date (default: today)",
    )

    # -- Collection options --
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Symbols per batch (default: 10)",
    )
    p.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        metavar="SECS",
        help="Pause between symbols in seconds (default: 0.5)",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible symbol selection",
    )

    # -- InfluxDB (defaults read from env, which is loaded from .env above) --
    import os

    p.add_argument(
        "--influx-url",
        default=os.environ.get("INFLUXDB_URL", "http://10.32.3.27:8086"),
        help="InfluxDB URL (env: INFLUXDB_URL)",
    )
    p.add_argument(
        "--influx-token",
        default=os.environ.get("INFLUXDB_TOKEN", ""),
        metavar="TOKEN",
        help="InfluxDB auth token (env: INFLUXDB_TOKEN)",
    )
    p.add_argument(
        "--influx-org",
        default=os.environ.get("INFLUXDB_ORG", "caelum"),
        help="InfluxDB org (env: INFLUXDB_ORG)",
    )
    p.add_argument(
        "--influx-bucket",
        default=os.environ.get("INFLUXDB_BUCKET", "market_data"),
        help="InfluxDB bucket (env: INFLUXDB_BUCKET, default: market_data)",
    )
    p.add_argument(
        "--csv-fallback", metavar="PATH", help="CSV file path if InfluxDB unavailable"
    )

    # -- Behaviour flags --
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print time estimate and exit without downloading",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt (required for non-interactive/nohup)",
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle symbol order before collecting"
    )

    return p


def parse_end_date(raw: str) -> datetime:
    if raw.lower() == "today":
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime.strptime(raw, "%Y-%m-%d")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Seed RNG
    if args.random_seed is not None:
        random.seed(args.random_seed)

    # Resolve end date
    end_date = parse_end_date(args.end_date)

    # Select symbols
    symbols = select_symbols(args)
    if not symbols:
        logger.error(
            "No symbols selected. Use --symbols, --universe, --sectors, or --auto."
        )
        sys.exit(1)

    if args.shuffle:
        random.shuffle(symbols)

    # Print selection
    print("=" * 72)
    print(f"SYMBOL SELECTION  ({len(symbols)} symbols)")
    print("=" * 72)
    if len(symbols) <= 30:
        print("  " + "  ".join(symbols))
    else:
        preview = "  ".join(symbols[:20])
        print(f"  {preview}  ... (+{len(symbols) - 20} more)")
    print()

    # Estimate
    store = OHLCVStore(
        influx_url=args.influx_url,
        influx_token=args.influx_token,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        csv_fallback=Path(args.csv_fallback) if args.csv_fallback else None,
    )
    collector = HistoricalCollector(
        symbols=symbols,
        start_date=args.start_date,
        end_date=end_date,
        batch_size=args.batch_size,
        rate_limit_secs=args.rate_limit,
        store=store,
    )
    est = collector.estimate()
    print("=" * 72)
    print("ESTIMATE")
    print("=" * 72)
    print(f"  Symbols      : {est['symbols']}")
    print(f"  Trading days : {est['trading_days']}")
    print(f"  Storage      : {store.mode}")
    print(
        f"  Est. time    : {est['estimated_hours']:.1f} h  ({est['estimated_minutes']:.0f} min)"
    )
    print("=" * 72)

    if args.estimate_only:
        logger.info("--estimate-only set; exiting without collection.")
        return

    # Confirmation (skip with --yes)
    if not args.yes:
        try:
            resp = input("\nStart collection? [yes/no]: ").strip().lower()
        except EOFError:
            resp = "no"
        if resp != "yes":
            logger.info("Collection cancelled.")
            return

    # Run
    collector.run()


if __name__ == "__main__":
    main()
