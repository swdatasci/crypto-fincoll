#!/usr/bin/env python3
"""
FinVec Bootstrapping Workflow

Executes the complete bootstrapping pipeline to prepare training data for finvec models:
1. Fetch liquid symbols from market
2. Filter by liquidity threshold (TradabilityEvaluator)
3. Generate partial symvectors (414D, no zero exclusion)
4. Store data triads to InfluxDB (raw data + input vector + output)
5. Prepare dataset for Phase 2 (label generation)

Phase 1: Collect input vectors WITHOUT predictions
Phase 2: Generate labels from actual market outcomes
Phase 3: Train 414D model on labeled dataset

Author: Roderick Ford & Claude Code
Date: 2026-02-15
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path

# Add fincoll to path (we're in fincoll/workflows/ now)
fincoll_path = Path(__file__).parent.parent
sys.path.insert(0, str(fincoll_path))

# Try to import only what we need (no SymVector wrapper - we use FeatureExtractor directly)
try:
    from config.dimensions import DIMS
    from tasks.symvector_tasks import process_symbol_batch, aggregate_results
    from fincoll.providers.yfinance_provider import YFinanceProvider

    FINCOLL_AVAILABLE = True
    YFINANCE_AVAILABLE = True
except ImportError as e:
    FINCOLL_AVAILABLE = False
    YFINANCE_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Try to import PIM TradabilityEvaluator (optional - cross-project dependency)
try:
    pim_path = Path(__file__).parent.parent.parent / 'PassiveIncomeMaximizer' / 'engine'
    sys.path.insert(0, str(pim_path))
    from pim.execution.tradability_evaluator import TradabilityEvaluator
    TRADABILITY_AVAILABLE = True
except ImportError:
    TRADABILITY_AVAILABLE = False

if not FINCOLL_AVAILABLE:
    print(f"❌ Import error: {IMPORT_ERROR}")
    print("   Missing fincoll modules. Please ensure:")
    print("   1. Running from fincoll directory")
    print("   2. Required packages are installed")
    print("\n   Try:")
    print("   cd /home/rford/caelum/caelum-supersystem/fincoll")
    print("   uv run python workflows/bootstrap_finvec.py --help")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bootstrap_finvec.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FinVecBootstrapper:
    """
    Manages the bootstrap workflow for finvec training data collection
    """

    def __init__(
        self,
        liquidity_threshold: float = 0.7,
        symbols_per_batch: int = 50,
        max_symbols: int = 500,
        phase: int = 1,
        use_real_vectors: bool = False
    ):
        self.liquidity_threshold = liquidity_threshold
        self.symbols_per_batch = symbols_per_batch
        self.max_symbols = max_symbols
        self.phase = phase
        self.use_real_vectors = use_real_vectors

        self.tradability_evaluator = None
        self.data_provider = None

        # Initialize data provider if using real vectors
        if use_real_vectors:
            if not YFINANCE_AVAILABLE:
                logger.error(f"Cannot use real vectors: YFinance provider not available")
                logger.error(f"  Import error: {YFINANCE_IMPORT_ERROR if 'YFINANCE_IMPORT_ERROR' in globals() else 'Unknown'}")
                raise ImportError("YFinanceProvider required for real vector generation")

            self.data_provider = YFinanceProvider()
            logger.info("  Data provider: YFinance (initialized)")

        self.stats = {
            'total_symbols': 0,
            'liquid_symbols': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }

        logger.info(f"FinVec Bootstrapper initialized:")
        logger.info(f"  Liquidity threshold: {liquidity_threshold}")
        logger.info(f"  Batch size: {symbols_per_batch}")
        logger.info(f"  Max symbols: {max_symbols}")
        logger.info(f"  Phase: {phase}")
        logger.info(f"  Vector mode: {'REAL 414D' if use_real_vectors else 'PLACEHOLDER'}")

    def fetch_market_symbols(self) -> List[str]:
        """Fetch symbols from market (placeholder - implement with real data source)"""
        logger.info("Fetching symbols from market...")

        # TODO: Replace with actual market data source (TradeStation, Alpaca, etc.)
        # For now, return a sample list
        symbols = [
            # Large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
            # Large cap other
            'JPM', 'V', 'WMT', 'JNJ', 'PG', 'UNH', 'HD', 'MA',
            # Mid cap
            'SQ', 'SNAP', 'UBER', 'LYFT', 'PINS', 'DOCU', 'ZM', 'PTON',
            # ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
            # Crypto-related
            'COIN', 'MARA', 'RIOT',
        ]

        self.stats['total_symbols'] = len(symbols)
        logger.info(f"  Found {len(symbols)} symbols")

        return symbols[:self.max_symbols]

    def filter_by_liquidity(self, symbols: List[str]) -> List[str]:
        """Filter symbols by liquidity threshold"""
        logger.info(f"Filtering symbols by liquidity (threshold: {self.liquidity_threshold})...")

        # TODO: Initialize TradabilityEvaluator with real data provider
        # For now, use mock filtering logic
        liquid_symbols = []

        # Large caps are generally liquid
        high_liquidity_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'WMT', 'JNJ', 'PG', 'UNH',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'
        }

        for symbol in symbols:
            if symbol in high_liquidity_symbols:
                liquid_symbols.append(symbol)
                logger.debug(f"  ✓ {symbol} - High liquidity")
            else:
                logger.debug(f"  ✗ {symbol} - Low liquidity")

        self.stats['liquid_symbols'] = len(liquid_symbols)
        logger.info(f"  {len(liquid_symbols)}/{len(symbols)} symbols pass liquidity filter")

        return liquid_symbols

    def process_batch(self, symbols: List[str], timestamp: datetime) -> Dict:
        """Process a batch of symbols"""
        logger.info(f"Processing batch of {len(symbols)} symbols...")

        # Use async processing with real vectors if enabled
        result = asyncio.run(
            process_symbol_batch(
                symbols=symbols,
                timestamp=timestamp,
                use_real_vectors=self.use_real_vectors,
                data_provider=self.data_provider
            )
        )

        logger.info(f"  Batch complete: {result['successful']}/{result['total']} successful")

        return result

    def run(self):
        """Execute complete bootstrapping workflow"""
        self.stats['start_time'] = datetime.utcnow()
        logger.info("=" * 80)
        logger.info("FINVEC BOOTSTRAPPING WORKFLOW - PHASE {}".format(self.phase))
        logger.info("=" * 80)

        try:
            # Step 1: Fetch symbols
            symbols = self.fetch_market_symbols()

            # Step 2: Filter by liquidity
            liquid_symbols = self.filter_by_liquidity(symbols)

            if not liquid_symbols:
                logger.error("No liquid symbols found! Aborting.")
                return False

            # Step 3: Process in batches
            timestamp = datetime.utcnow()
            total_batches = (len(liquid_symbols) + self.symbols_per_batch - 1) // self.symbols_per_batch

            logger.info(f"Processing {len(liquid_symbols)} symbols in {total_batches} batches...")

            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.symbols_per_batch
                end_idx = min(start_idx + self.symbols_per_batch, len(liquid_symbols))
                batch_symbols = liquid_symbols[start_idx:end_idx]

                logger.info(f"Batch {batch_idx + 1}/{total_batches}: {len(batch_symbols)} symbols")

                batch_result = self.process_batch(batch_symbols, timestamp)

                self.stats['processed'] += batch_result['total']
                self.stats['successful'] += batch_result['successful']
                self.stats['failed'] += batch_result['failed']

            # Step 4: Final summary
            self.stats['end_time'] = datetime.utcnow()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            logger.info("=" * 80)
            logger.info("BOOTSTRAPPING COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration:.1f}s")
            logger.info(f"Total symbols: {self.stats['total_symbols']}")
            logger.info(f"Liquid symbols: {self.stats['liquid_symbols']}")
            logger.info(f"Processed: {self.stats['processed']}")
            logger.info(f"Successful: {self.stats['successful']}")
            logger.info(f"Failed: {self.stats['failed']}")
            logger.info(f"Success rate: {self.stats['successful'] / max(self.stats['processed'], 1) * 100:.1f}%")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"Bootstrapping failed: {e}", exc_info=True)
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FinVec Bootstrapping Workflow')

    parser.add_argument(
        '--liquidity-threshold',
        type=float,
        default=0.7,
        help='Minimum liquidity score (0-1, default: 0.7)'
    )

    parser.add_argument(
        '--symbols-per-batch',
        type=int,
        default=50,
        help='Number of symbols per batch (default: 50)'
    )

    parser.add_argument(
        '--max-symbols',
        type=int,
        default=500,
        help='Maximum total symbols to process (default: 500)'
    )

    parser.add_argument(
        '--phase',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Bootstrap phase: 1=collect vectors, 2=label generation, 3=training (default: 1)'
    )

    parser.add_argument(
        '--use-real-vectors',
        action='store_true',
        help='Use real 414D vector generation via FeatureExtractor (requires yfinance). Default: placeholder mode'
    )

    args = parser.parse_args()

    # Create and run bootstrapper
    bootstrapper = FinVecBootstrapper(
        liquidity_threshold=args.liquidity_threshold,
        symbols_per_batch=args.symbols_per_batch,
        max_symbols=args.max_symbols,
        phase=args.phase,
        use_real_vectors=args.use_real_vectors
    )

    success = bootstrapper.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
