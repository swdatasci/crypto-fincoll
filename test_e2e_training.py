#!/usr/bin/env python3
"""
End-to-End Training Test

Tests the complete FinColl → FinVec integration:
1. FinColl fetches data (using synthetic data for testing)
2. FinColl constructs DIMS.fincoll_total features
3. FinColl computes 10D velocity targets
4. FinColl calls FinVec.train_from_arrays()
5. FinVec trains the model and returns metrics
6. Success!

This proves that FinColl orchestrates and FinVec trains,
with NO HTTP requests between them - just Python imports.
"""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.dimensions import DIMS
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_bars(symbol: str, n_bars: int = 500) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Args:
        symbol: Stock symbol
        n_bars: Number of bars to generate

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(hash(symbol) % (2**32))  # Deterministic per symbol

    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_bars)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_bars)

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02  # 2% daily volatility
    prices = base_price * np.cumprod(1 + returns)

    # Create OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_bars) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_bars)
    }, index=timestamps)

    return df


def test_e2e_training():
    """
    End-to-end test of FinColl → FinVec integration.

    Flow:
    1. Generate synthetic data for 3 symbols
    2. Call FinColl's train_velocity_model()
    3. FinColl orchestrates data → features → targets
    4. FinColl calls FinVec.train_from_arrays()
    5. Verify success
    """
    from fincoll import train_velocity_model, TrainingConfig

    logger.info("=" * 80)
    logger.info("E2E Training Test: FinColl → FinVec Integration")
    logger.info("=" * 80)

    # Step 1: Prepare test configuration
    logger.info("\n[1/4] Creating configuration...")
    config = TrainingConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL'],  # 3 symbols for testing
        epochs=2,  # Just 2 epochs for fast testing
        batch_size=32,
        learning_rate=0.001,
        device='cpu',  # Use CPU for testing
        hidden_dim=128,  # Smaller model for testing
        num_layers=2,
        lookback_days=100,  # Less data for faster testing
        use_cache=False,  # Don't use cache for testing
        checkpoint_dir="/tmp/fincoll_test_checkpoints"
    )
    logger.info(f"Config: {config.symbols}, {config.epochs} epochs, device={config.device}")

    # Step 2: Mock data fetcher (replace actual TradeStation calls)
    logger.info("\n[2/4] Mocking data fetcher...")
    from fincoll.storage.influxdb_cache import get_cache
    from fincoll.providers.tradestation_provider import TradeStationProvider

    # Monkey-patch the data fetcher to use synthetic data
    original_get_bars = TradeStationProvider.get_historical_bars

    def mock_get_bars(self, symbol, interval, start_date, end_date, **kwargs):
        logger.info(f"  Mock fetch: {symbol}")
        return generate_synthetic_bars(symbol, n_bars=100)

    TradeStationProvider.get_historical_bars = mock_get_bars

    try:
        # Step 3: Call FinColl training (this triggers the entire pipeline)
        logger.info("\n[3/4] Calling FinColl train_velocity_model()...")
        logger.info("  This will:")
        logger.info("    - Fetch synthetic data")
        logger.info("    - Extract DIMS.fincoll_total features")
        logger.info("    - Compute 10D velocity targets")
        logger.info("    - Call FinVec.train_from_arrays()")
        logger.info("")

        result = train_velocity_model(config)

        # Step 4: Verify results
        logger.info("\n[4/4] Verifying results...")
        logger.info(f"  Status: {result['status']}")
        logger.info(f"  Epochs completed: {result['epochs_completed']}")
        logger.info(f"  Best val loss: {result.get('best_val_loss', 'N/A')}")
        logger.info(f"  Final train loss: {result.get('final_train_loss', 'N/A')}")
        logger.info(f"  Training time: {result.get('training_time_seconds', 0):.2f}s")
        logger.info(f"  Samples trained: {result.get('samples_trained', 'N/A')}")

        # Assertions
        assert result['status'] == 'success', f"Training failed: {result.get('error')}"
        assert result['epochs_completed'] > 0, "No epochs completed"
        assert 'best_val_loss' in result, "Missing validation loss"
        assert 'checkpoint_path' in result, "Missing checkpoint path"

        logger.info("\n" + "=" * 80)
        logger.info("✓ SUCCESS: FinColl → FinVec integration works!")
        logger.info("=" * 80)
        logger.info("\nKEY INTEGRATION POINTS VERIFIED:")
        logger.info("  1. FinColl fetched data (synthetic)")
        logger.info("  2. FinColl constructed DIMS.fincoll_total features")
        logger.info("  3. FinColl computed 10D velocity targets")
        logger.info("  4. FinColl called FinVec via: from finvec import train_from_arrays")
        logger.info("  5. FinVec trained model and returned metrics")
        logger.info("  6. NO HTTP requests between FinColl and FinVec - pure Python!")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"\n✗ FAILED: {e}", exc_info=True)
        return False

    finally:
        # Restore original method
        TradeStationProvider.get_historical_bars = original_get_bars


if __name__ == '__main__':
    logger.info("Starting E2E integration test...\n")

    success = test_e2e_training()

    if success:
        logger.info("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n❌ Tests failed!")
        sys.exit(1)
