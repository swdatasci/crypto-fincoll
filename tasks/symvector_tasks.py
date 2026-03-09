"""
SymVector Batch Processing Tasks

Provides async task functions for processing symbol batches during bootstrapping.
Uses FeatureExtractor (the actual 414D generator) for real vector generation.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import FeatureExtractor and DIMS
# These are only needed for use_real_vectors=True mode
try:
    # Add fincoll to path
    fincoll_path = Path(__file__).parent.parent
    sys.path.insert(0, str(fincoll_path))

    from fincoll.features.feature_extractor import FeatureExtractor
    from config.dimensions import DIMS

    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    FEATURE_EXTRACTOR_AVAILABLE = False
    FEATURE_EXTRACTOR_ERROR = str(e)
    logger.warning(f"FeatureExtractor not available: {e}")
    logger.warning("  Will only support placeholder mode (use_real_vectors=False)")


async def process_symbol_batch(
    symbols: List[str],
    timestamp: datetime = None,
    use_real_vectors: bool = False,
    data_provider=None
) -> Dict[str, Any]:
    """
    Process a batch of symbols to generate partial symvectors.

    Args:
        symbols: List of stock symbols to process
        timestamp: Target timestamp for data collection (default: now)
        use_real_vectors: If True, use FeatureExtractor for real 414D generation
        data_provider: Data provider for fetching OHLCV data (required if use_real_vectors=True)

    Returns:
        Dictionary with batch results:
        {
            'total': int,
            'successful': int,
            'failed': int,
            'results': List[Dict]
        }
    """
    timestamp = timestamp or datetime.utcnow()

    logger.info(f"Processing batch of {len(symbols)} symbols at {timestamp}")
    logger.info(f"  Mode: {'REAL 414D VECTORS' if use_real_vectors else 'PLACEHOLDER'}")

    results = {
        'total': len(symbols),
        'successful': 0,
        'failed': 0,
        'results': []
    }

    # Initialize FeatureExtractor if using real vectors
    feature_extractor = None
    expected_dim = 414  # Default if DIMS not available

    if use_real_vectors:
        if not FEATURE_EXTRACTOR_AVAILABLE:
            logger.error(f"use_real_vectors=True but FeatureExtractor not available: {FEATURE_EXTRACTOR_ERROR if 'FEATURE_EXTRACTOR_ERROR' in globals() else 'Import failed'}")
            logger.error("  Falling back to placeholder mode.")
            use_real_vectors = False
        elif data_provider is None:
            logger.warning("use_real_vectors=True but no data_provider supplied. Falling back to placeholder mode.")
            use_real_vectors = False
        else:
            try:
                feature_extractor = FeatureExtractor(
                    data_provider=data_provider,
                    enable_senvec=True,
                    enable_futures=True,
                    enable_finnhub=True,
                    cache_fundamentals=True,
                    cache_news=True
                )
                # Get expected dimensions from config if available
                if FEATURE_EXTRACTOR_AVAILABLE and 'DIMS' in dir():
                    expected_dim = DIMS.fincoll_total
                logger.info(f"  FeatureExtractor initialized (target: {expected_dim}D vectors)")
            except Exception as e:
                logger.error(f"Failed to initialize FeatureExtractor: {e}")
                use_real_vectors = False

    # Process each symbol
    for symbol in symbols:
        try:
            if use_real_vectors and feature_extractor:
                # REAL 414D VECTOR GENERATION
                # Fetch OHLCV data (need enough history for indicators)
                lookback_days = 90  # 90 days for technical indicators
                end_date = timestamp
                start_date = end_date - timedelta(days=lookback_days)

                # Fetch data from provider (using BaseDataProvider interface)
                ohlcv_data = data_provider.get_historical_bars(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    interval='1d'  # Daily bars
                )

                if ohlcv_data is None or len(ohlcv_data) == 0:
                    raise ValueError(f"No OHLCV data available for {symbol}")

                # Generate 414D vector using FeatureExtractor
                vector = feature_extractor.extract_features(
                    ohlcv_data=ohlcv_data,
                    symbol=symbol,
                    timestamp=timestamp
                )

                # Verify vector dimensions
                actual_dim = len(vector)
                if actual_dim != expected_dim:
                    logger.warning(
                        f"Vector dimension mismatch: expected {expected_dim}D, got {actual_dim}D"
                    )
                    # Update expected_dim to actual for reporting
                    expected_dim = actual_dim

                result = {
                    'symbol': symbol,
                    'status': 'success',
                    'timestamp': timestamp,
                    'vector_size': actual_dim,
                    'vector': vector.tolist(),  # Convert numpy array to list for JSON
                    'has_prediction': False,  # Phase 1: no predictions yet
                    'mode': 'REAL_VECTOR'
                }

                logger.info(f"  ✓ {symbol}: Generated {actual_dim}D vector (REAL)")

            else:
                # PLACEHOLDER MODE (for testing workflow without data providers)
                result = {
                    'symbol': symbol,
                    'status': 'success',
                    'timestamp': timestamp,
                    'vector_size': expected_dim,
                    'vector': None,  # Placeholder - no actual vector
                    'has_prediction': False,
                    'mode': 'PLACEHOLDER'
                }

                logger.debug(f"  ✓ {symbol}: Placeholder (workflow test, {expected_dim}D)")

            results['successful'] += 1
            results['results'].append(result)

        except Exception as e:
            logger.error(f"  ✗ {symbol}: {e}", exc_info=True)
            results['failed'] += 1
            results['results'].append({
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'mode': 'REAL_VECTOR' if use_real_vectors else 'PLACEHOLDER'
            })

    logger.info(f"Batch complete: {results['successful']}/{results['total']} successful")

    return results


def aggregate_results(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple batches.

    Args:
        batch_results: List of batch result dictionaries

    Returns:
        Aggregated statistics dictionary
    """
    aggregated = {
        'total_batches': len(batch_results),
        'total_symbols': 0,
        'successful': 0,
        'failed': 0,
        'all_results': []
    }

    for batch in batch_results:
        aggregated['total_symbols'] += batch.get('total', 0)
        aggregated['successful'] += batch.get('successful', 0)
        aggregated['failed'] += batch.get('failed', 0)
        aggregated['all_results'].extend(batch.get('results', []))

    if aggregated['total_symbols'] > 0:
        aggregated['success_rate'] = aggregated['successful'] / aggregated['total_symbols']
    else:
        aggregated['success_rate'] = 0.0

    return aggregated
