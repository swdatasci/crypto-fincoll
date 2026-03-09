"""
Feature Constructor - Build Feature Vectors from Central Config

This module constructs complete feature vectors from:
- Technical indicators (from FeatureExtractor)
- SenVec sentiment (via HTTP API)
- Sector embeddings
- Cross-asset signals

FinColl's job is to orchestrate data collection and feature construction.
FinVec only receives the complete feature vectors.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime

from config.dimensions import DIMS

logger = logging.getLogger(__name__)


def construct_features(
    bars: Dict[str, pd.DataFrame],
    sentiment_data: Optional[pd.DataFrame] = None,
    include_sentiment: bool = True,
    include_fundamentals: bool = True
) -> Dict[str, np.ndarray]:
    """
    Construct feature vectors for all symbols using the central dimension config.

    This is FinColl's orchestration layer - it handles:
    1. Technical feature extraction
    2. SenVec sentiment integration (config-sized)
    3. Fundamental data integration
    4. Cross-asset signals (SPY, VIX)
    5. Sector/industry embeddings

    Args:
        bars: Dict of symbol -> OHLCV DataFrame
        sentiment_data: Optional sentiment data from SenVec
        include_sentiment: Include SenVec features (dimension from config)
        include_fundamentals: Include fundamental features

    Returns:
        Dict of symbol -> feature array [N, feature_dim]
        where feature_dim is DIMS.fincoll_total

    Note:
        This is a STUB implementation. Real implementation should:
        - Use FeatureExtractor for each symbol
        - Call SenVec API for sentiment features
        - Fetch SPY/VIX for cross-asset signals
        - Handle missing data gracefully
    """
    from ..features.feature_extractor import FeatureExtractor

    logger.info(f"Constructing features for {len(bars)} symbols")

    # Initialize feature extractor
    extractor = FeatureExtractor(
        alpha_vantage_client=None,  # TODO: Pass real client
        enable_senvec=include_sentiment,
        enable_futures=True,
        data_provider=None  # TODO: Pass real provider
    )

    features_dict = {}

    for symbol, df in bars.items():
        try:
            # Extract features for each bar in the DataFrame
            # For now, just extract features for the last bar as a demo
            if len(df) == 0:
                logger.warning(f"Empty DataFrame for {symbol}, skipping")
                continue

            # Get timestamp of last bar
            timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()

            # Extract features (DIMS.fincoll_total)
            features = extractor.extract_features(
                ohlcv_data=df,
                symbol=symbol,
                timestamp=timestamp
            )

            # For training, we need a sequence of features, not just one
            # This is a simplified version - real implementation should extract
            # features for all bars in the DataFrame
            all_features = []
            for i in range(len(df)):
                timestamp_i = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
                features_i = extractor.extract_features(
                    ohlcv_data=df.iloc[:i+1],  # Use data up to this point
                    symbol=symbol,
                    timestamp=timestamp_i
                )
                all_features.append(features_i)

            # Stack into array [N, DIMS.fincoll_total]
            features_array = np.stack(all_features)
            features_dict[symbol] = features_array

            logger.info(f"Extracted features for {symbol}: {features_array.shape}")

        except Exception as e:
            logger.error(f"Failed to extract features for {symbol}: {e}")
            continue

    return features_dict


def construct_features_batch(
    bars: Dict[str, pd.DataFrame],
    sentiment_url: Optional[str] = "http://localhost:18000",
    include_sentiment: bool = True,
    include_fundamentals: bool = True
) -> Dict[str, np.ndarray]:
    """
    Batch-optimized feature construction.

    Similar to construct_features() but optimized for batch processing:
    - Parallel sentiment API calls
    - Batch SPY/VIX fetching
    - Vectorized technical indicators

    Args:
        bars: Dict of symbol -> OHLCV DataFrame
        sentiment_url: SenVec API URL
        include_sentiment: Include SenVec features
        include_fundamentals: Include fundamental features

    Returns:
        Dict of symbol -> feature array [N, feature_dim]
    """
    # For now, just call the regular constructor
    # TODO: Implement actual batch optimizations
    return construct_features(
        bars=bars,
        sentiment_data=None,
        include_sentiment=include_sentiment,
        include_fundamentals=include_fundamentals
    )


def validate_features(features: np.ndarray, expected_dim: int = DIMS.fincoll_total) -> bool:
    """
    Validate feature array shape and values.

    Args:
        features: Feature array [N, feature_dim]
        expected_dim: Expected feature dimension (defaults to DIMS.fincoll_total)

    Returns:
        True if valid, False otherwise
    """
    if features.ndim != 2:
        logger.error(f"Expected 2D array, got {features.ndim}D")
        return False

    if features.shape[1] != expected_dim:
        logger.error(f"Expected {expected_dim} features, got {features.shape[1]}")
        return False

    if np.isnan(features).any():
        logger.warning(f"Features contain NaN values: {np.isnan(features).sum()} / {features.size}")
        # Don't fail on NaN, just warn

    if np.isinf(features).any():
        logger.warning(f"Features contain Inf values: {np.isinf(features).sum()} / {features.size}")
        # Don't fail on Inf, just warn

    return True
