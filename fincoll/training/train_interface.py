"""
Training Interface - Python Library API for Model Training

Pure Python functions for training velocity models.

Flow:
    1. Fetch historical data (cache-first)
    2. Extract features for all symbols
    3. Create PyTorch Dataset
    4. Train using FinVec training code
    5. Save checkpoints

Usage:
    from fincoll import train_velocity_model, TrainingConfig

    config = TrainingConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        epochs=100,
        device='cuda'
    )

    metrics = train_velocity_model(config)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import FinColl components
from ..storage.influxdb_cache import get_cache as get_influx_cache
from ..providers.multi_provider_fetcher import MultiProviderFetcher
from ..features.feature_extractor import FeatureExtractor
from config.dimensions import DIMS  # type: ignore[import]

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for velocity model training"""

    # Data
    symbols: List[str]  # Symbols to train on
    lookback_days: int = 365 * 2  # 2 years of daily data
    train_split: float = 0.8  # Train/val split ratio

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cuda"  # 'cpu' or 'cuda'

    # Model
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2

    # Checkpointing
    checkpoint_dir: Optional[str] = None  # Where to save models
    save_interval: int = 10  # Save every N epochs

    # Features
    include_sentiment: bool = True
    include_fundamentals: bool = True

    # Data fetching
    use_cache: bool = True
    max_workers: int = 4  # Parallel data fetching


def train_velocity_model(config: TrainingConfig) -> Dict[str, Any]:
    """
    Train velocity model on historical data.

    This is the PRIMARY function for training from PIM or scripts.

    Args:
        config: Training configuration

    Returns:
        Training metrics:
        {
            "status": "success" | "failed",
            "epochs_completed": int,
            "best_val_loss": float,
            "final_train_loss": float,
            "final_val_loss": float,
            "checkpoint_path": str,
            "training_time_seconds": float,
            "samples_trained": int,
            "metadata": {
                "symbols": [str],
                "feature_dim": int,
                "model_params": int,
                ...
            }
        }

    Raises:
        ValueError: If config is invalid
        RuntimeError: If training fails
    """
    start_time = datetime.now()

    logger.info(f"Starting velocity model training with {len(config.symbols)} symbols")

    try:
        # Step 1: Fetch historical data for all symbols
        logger.info("Fetching historical data...")
        symbol_data = _fetch_training_data(config)

        # Step 2: Extract features and labels
        logger.info("Extracting features and labels...")
        features, labels = _extract_training_features(symbol_data, config)

        # Step 3: Create train/val split
        logger.info("Creating train/val split...")
        train_data, val_data = _create_train_val_split(features, labels, config)

        # Step 4: Prepare FinVec configuration and dataset
        logger.info("Preparing FinVec configuration...")
        finvec_data = _create_model_and_trainer(train_data, val_data, config)

        # Step 5: Train model via FinVec
        logger.info(f"Training via FinVec for {config.epochs} epochs...")
        train_metrics = _train_model(finvec_data, None, config)

        # Step 6: Checkpoint already saved by FinVec
        checkpoint_path = train_metrics.get("checkpoint_path", "unknown")

        # Step 7: Compile results
        training_time = (datetime.now() - start_time).total_seconds()

        results = {
            "status": "success",
            "epochs_completed": train_metrics["epochs_completed"],
            "best_val_loss": train_metrics["best_val_loss"],
            "final_train_loss": train_metrics["final_train_loss"],
            "final_val_loss": train_metrics["final_val_loss"],
            "checkpoint_path": checkpoint_path,
            "training_time_seconds": round(training_time, 2),
            "samples_trained": train_metrics.get("samples_trained", len(features)),
            "metadata": {
                "symbols": config.symbols,
                "feature_dim": features.shape[1] if len(features) > 0 else 0,
                "model_params": train_metrics.get("model_params", 0),
                "lookback_days": config.lookback_days,
                "batch_size": config.batch_size,
                "device": config.device,
                "timestamp": datetime.now().isoformat(),
            },
        }

        logger.info(f"Training completed in {training_time:.0f}s")
        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# ==================== PRIVATE HELPER FUNCTIONS ====================


def _fetch_training_data(config: TrainingConfig) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for all symbols.

    Uses cache-first strategy with parallel fetching.

    Args:
        config: Training config

    Returns:
        Dict of symbol -> bars DataFrame
    """
    symbol_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.lookback_days)

    cache = get_influx_cache() if config.use_cache else None
    provider = MultiProviderFetcher()

    for symbol in config.symbols:
        try:
            # Try cache first
            bars = None
            if cache:
                bars = cache.get_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                    source="tradestation",
                )

            # Cache miss - fetch from provider
            if bars is None or bars.empty:
                logger.info(f"Cache miss for {symbol}, fetching from provider")
                bars = provider.get_historical_bars(
                    symbol=symbol,
                    interval="1d",
                    start_date=start_date,
                    end_date=end_date,
                )

                # Store in cache
                if cache and bars is not None:
                    cache.store_bars(symbol, bars, interval="1d", source="tradestation")

            if bars is not None and not bars.empty:
                symbol_data[symbol] = bars
                logger.info(f"Fetched {len(bars)} bars for {symbol}")

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")

    logger.info(f"Fetched data for {len(symbol_data)}/{len(config.symbols)} symbols")
    return symbol_data


def _extract_training_features(
    symbol_data: Dict[str, pd.DataFrame], config: TrainingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels for all symbols.

    This orchestrates the complete pipeline:
    1. Construct features (dimension from config)
    2. Compute velocity labels (8D: 4 horizons × 2 directions)
    3. Align features and labels
    4. Clean NaN/Inf values

    Args:
        symbol_data: Dict of symbol -> bars
        config: Training config

    Returns:
        (features_array, labels_array)
        - features: [N, DIMS.fincoll_total] numpy array
        - labels: [N, 8] numpy array (4 horizons × 2 directions)
    """
    from ..features.constructor import construct_features
    from ..training.target_computer import (
        compute_velocity_targets_per_symbol,
        align_features_and_targets,
    )

    logger.info(f"Extracting features for {len(symbol_data)} symbols...")

    # Step 1: Construct features (DIMS.fincoll_total)
    features_dict = construct_features(
        bars=symbol_data,
        sentiment_data=None,  # TODO: Fetch from SenVec if enabled
        include_sentiment=config.include_sentiment,
        include_fundamentals=config.include_fundamentals,
    )

    if not features_dict:
        raise RuntimeError("No features extracted from any symbols")

    # Step 2: Compute velocity labels (8D: 4 horizons × 2 directions)
    labels_dict = compute_velocity_targets_per_symbol(bars_by_symbol=symbol_data)

    if not labels_dict:
        raise RuntimeError("No labels computed for any symbols")

    # Step 3: Align features and labels
    features_array, labels_array = align_features_and_targets(
        features_dict, labels_dict
    )

    if len(features_array) == 0:
        raise RuntimeError("No valid samples after alignment")

    logger.info(
        f"Extracted features: {features_array.shape}, labels: {labels_array.shape}"
    )
    return features_array, labels_array


def _create_train_val_split(
    features: np.ndarray, labels: np.ndarray, config: TrainingConfig
) -> tuple[tuple, tuple]:
    """
    Create train/val split.

    Args:
        features: Feature array [N, feature_dim]
        labels: Label array [N, output_dim]
        config: Training config

    Returns:
        ((train_features, train_labels), (val_features, val_labels))
    """
    split_idx = int(len(features) * config.train_split)

    train_features = features[:split_idx]
    train_labels = labels[:split_idx]

    val_features = features[split_idx:]
    val_labels = labels[split_idx:]

    logger.info(f"Train split: {len(train_features)} samples")
    logger.info(f"Val split: {len(val_features)} samples")

    return (train_features, train_labels), (val_features, val_labels)


def _create_model_and_trainer(
    train_data: tuple, val_data: tuple, config: TrainingConfig
):
    """
    Prepare FinVec configuration.

    This is called before training but doesn't create anything -
    FinVec will create the model internally.

    Returns:
        (features, targets) combined from train/val data
    """
    try:
        # Import FinVec configuration
        from finvec import ArrayTrainingConfig

        # Combine train and val data (FinVec will split internally)
        train_features, train_labels = train_data
        val_features, val_labels = val_data

        all_features = np.vstack([train_features, val_features])
        all_labels = np.vstack([train_labels, val_labels])

        # Create FinVec config
        finvec_config = ArrayTrainingConfig(
            input_dim=all_features.shape[1],
            output_dim=all_labels.shape[1],
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            checkpoint_dir=config.checkpoint_dir,
            save_interval=config.save_interval,
        )

        return (all_features, all_labels, finvec_config)

    except ImportError as e:
        logger.error(f"Failed to import FinVec: {e}")
        raise RuntimeError("FinVec not available - check installation") from e


def _train_model(model, trainer_config, config: TrainingConfig) -> Dict[str, Any]:
    """
    Train model using FinVec's array-based training.

    This is the KEY INTEGRATION POINT: FinColl → FinVec via Python import.

    Returns:
        Training metrics
    """
    try:
        # Unpack the tuple from _create_model_and_trainer
        all_features, all_labels, finvec_config = model

        # Import FinVec training function
        from finvec import train_from_arrays

        logger.info(f"Calling FinVec training with {len(all_features)} samples...")

        # Call FinVec training (THIS IS THE INTEGRATION!)
        metrics = train_from_arrays(
            features=all_features, targets=all_labels, config=finvec_config
        )

        # Return standardized metrics
        return {
            "epochs_completed": metrics.get("epochs_completed", config.epochs),
            "best_val_loss": metrics.get("best_val_loss", 0.0),
            "final_train_loss": metrics.get("final_train_loss", 0.0),
            "final_val_loss": metrics.get("final_val_loss", 0.0),
        }

    except ImportError as e:
        logger.error(f"Failed to import FinVec: {e}")
        raise RuntimeError("FinVec not available - check installation") from e
    except Exception as e:
        logger.error(f"FinVec training failed: {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {e}") from e


def _save_checkpoint(model, metrics: Dict[str, Any], config: TrainingConfig) -> str:
    """
    Save model checkpoint.

    Args:
        model: Trained model
        metrics: Training metrics
        config: Training config

    Returns:
        Path to saved checkpoint
    """
    import torch
    from config.dimensions import DIMS

    if config.checkpoint_dir is None:
        # Default: save to finvec/checkpoints/velocity/
        checkpoint_dir = (
            Path(__file__).parent.parent.parent.parent
            / "finvec"
            / "checkpoints"
            / "velocity"
        )
    else:
        checkpoint_dir = Path(config.checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"velocity_model_{timestamp}.pt"

    # Save actual model checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": {
            "checkpoint_dir": str(config.checkpoint_dir)
            if config.checkpoint_dir
            else None,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
        "timestamp": timestamp,
        "feature_dim": DIMS.fincoll_total,
        "config_version": None,  # Will be set if config_version module available
    }

    # Add config version if available
    try:
        from fincoll.storage.config_version import get_config_version

        checkpoint["config_version"] = get_config_version()
    except ImportError:
        pass

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✅ Saved model checkpoint to {checkpoint_path}")
    logger.info(f"   Metrics: {metrics}")
    logger.info(f"   Feature dim: {DIMS.fincoll_total}D")

    return str(checkpoint_path)
