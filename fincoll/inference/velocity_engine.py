"""
Velocity Inference Engine for FinColl

Wraps the VelocityTransformerModel to provide velocity predictions from config-sized feature vectors.
No separate velocity server needed - FinColl does feature extraction + inference directly.

NOTE: The checkpoint was trained with VelocityTransformerModel (from train_velocity.py),
NOT SimpleVelocityModel. This engine handles dynamic input dimension matching.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from config.dimensions import DIMS

# Add finvec to path for model imports
finvec_path = Path(__file__).parent.parent.parent.parent / "finvec"
sys.path.insert(0, str(finvec_path))

from train_velocity import VelocityTrainingConfig, VelocityTransformerModel

logger = logging.getLogger(__name__)


class VelocityEngine:
    """
    Inference engine for velocity predictions.

    Takes feature vectors from FinColl and returns velocity format predictions.
    Model was trained with VelocityTransformerModel on engineered features.
    Handles dynamic input dimension matching from checkpoint.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Initialize velocity engine.

        Args:
            checkpoint_path: Path to velocity model checkpoint
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        logger.info(f"VelocityEngine using device: {self.device}")

        # Load model and get expected input dimension
        self.model, self.input_dim = self._load_model(checkpoint_path)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"VelocityEngine initialized ({param_count:,} parameters, input_dim={self.input_dim})"
        )

    def _get_device(self, device: str) -> torch.device:
        """Get torch device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, checkpoint_path: str):
        """
        Load velocity model from checkpoint.

        Uses VelocityTransformerModel with dynamic input dimension matching
        to handle checkpoints trained with different input dimensions.

        Args:
            checkpoint_path: Path to .pt file

        Returns:
            Tuple of (model, input_dim)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Velocity checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading velocity model from {checkpoint_path}")

        # Load checkpoint (PyTorch 2.6+ requires weights_only=False for numpy objects)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Get config from checkpoint
        config_dict = checkpoint.get("config", {})

        # Reconstruct config from checkpoint (filter to valid VelocityTrainingConfig fields)
        valid_config_fields = {
            "d_model",
            "n_heads",
            "n_layers",
            "dropout",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "gradient_clip_norm",
            "symbols",
            "symbol_set",
            "train_split",
            "val_split",
            "sequence_length",
            "timeframes",
            "velocity_weight",
            "bars_weight",
            "confidence_weight",
            "spike_weight",
            "device",
            "num_workers",
            "mixed_precision",
            "checkpoint_dir",
            "save_every",
            "resume_from",
            "use_wandb",
            "wandb_project",
            "log_every",
            "fincoll_url",
            "use_cached_data",
            "cache_dir",
        }
        filtered_config = {
            k: v for k, v in config_dict.items() if k in valid_config_fields
        }
        config = VelocityTrainingConfig(**filtered_config)

        # Infer input_dim from checkpoint weights (handles dimension mismatch)
        state_dict_key = (
            "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
        )
        state_dict = checkpoint[state_dict_key]

        # Get actual input dimension from input_projection.0.weight [d_model, input_dim]
        checkpoint_input_dim = state_dict["input_projection.0.weight"].shape[1]

        # Create model (will have default input_dim)
        model = VelocityTransformerModel(config)

        # If dimensions don't match, recreate input_projection with correct dimensions
        current_input_dim = model.input_projection[0].in_features
        if current_input_dim != checkpoint_input_dim:
            logger.info(
                f"Adjusting input_projection: {current_input_dim}D -> {checkpoint_input_dim}D"
            )
            model.input_projection = nn.Sequential(
                nn.Linear(checkpoint_input_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

        # Now load the full state dict
        # Try strict loading first (ideal case)
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Checkpoint loaded successfully (exact match)")
        except RuntimeError as e:
            if "Unexpected key(s) in state_dict" in str(e):
                logger.warning(
                    f"Checkpoint contains extra keys (likely from previous architecture). "
                    f"Filtering to match current model..."
                )

                # Filter state_dict to only include keys present in current model
                model_keys = set(model.state_dict().keys())
                filtered_state = {
                    k: v for k, v in state_dict.items() if k in model_keys
                }

                # Load with strict=False to allow missing keys
                missing_keys, unexpected_keys = model.load_state_dict(
                    filtered_state, strict=False
                )

                logger.info(
                    f"Loaded {len(filtered_state)}/{len(state_dict)} checkpoint parameters"
                )
                if missing_keys:
                    logger.warning(
                        f"Missing keys (will use random init): {len(missing_keys)} keys"
                    )
                if unexpected_keys:
                    logger.warning(
                        f"Skipped unexpected keys: {len(unexpected_keys)} keys"
                    )
            else:
                # Different error - re-raise
                raise

        model.to(self.device)

        epoch = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get(
            "val_loss", checkpoint.get("best_val_loss", "unknown")
        )

        logger.info(
            f"VelocityTransformerModel loaded (epoch {epoch}, d_model={config.d_model}, "
            f"n_layers={config.n_layers}, input_dim={checkpoint_input_dim})"
        )

        return model, checkpoint_input_dim

    def predict(
        self, feature_vector: np.ndarray, symbol: str, current_price: float
    ) -> Dict[str, Any]:
        """
        Generate velocity prediction from feature vector.

        Args:
            feature_vector: numpy array from FinColl feature extraction
            symbol: Stock symbol
            current_price: Current price for return calculation

        Returns:
            Velocity prediction in format:
            {
                "symbol": str,
                "timestamp": str,
                "current_price": float,
                "velocities": [
                    {
                        "velocity": float,        # % return per bar
                        "timeframe": str,         # "1min", "5min", etc.
                        "bars": int,              # Bars until target
                        "seconds": int,           # Time in seconds
                        "direction": str,         # "LONG" or "SHORT"
                        "confidence": float,      # 0-1 scale
                        "expected_return": float  # velocity * bars
                    },
                    ...
                ],
                "best_opportunity": {...},  # Highest |velocity|
                "spike_alert": {...},       # Divergence detection
                "metadata": {...}
            }
        """
        # CRITICAL: Validate feature vector for NaN/Inf before processing
        if isinstance(feature_vector, np.ndarray):
            nan_count = np.isnan(feature_vector).sum()
            inf_count = np.isinf(feature_vector).sum()
            if nan_count > 0 or inf_count > 0:
                raise ValueError(
                    f"Invalid feature vector for {symbol}: "
                    f"{nan_count} NaN values, {inf_count} Inf values. "
                    f"Cannot perform inference with corrupted features."
                )
            features = torch.from_numpy(feature_vector).float()
        else:
            features = feature_vector.float()

        # Ensure correct shape [1, seq_len, input_dim]
        if features.dim() == 1:
            # Single feature vector - expand to [1, 1, input_dim]
            features = features.unsqueeze(0).unsqueeze(0)
        elif features.dim() == 2:
            # [seq_len, input_dim] -> [1, seq_len, input_dim]
            features = features.unsqueeze(0)

        # Pad or truncate to match model's expected input dimension
        current_dim = features.shape[-1]
        if current_dim < self.input_dim:
            padding = torch.zeros(
                features.shape[0], features.shape[1], self.input_dim - current_dim
            )
            features = torch.cat([features, padding], dim=-1)
        elif current_dim > self.input_dim:
            features = features[:, :, : self.input_dim]

        features = features.to(self.device)

        # Run inference - VelocityTransformerModel returns a dictionary
        with torch.no_grad():
            outputs = self.model(features)

        # Format response
        return self._format_response(
            symbol=symbol, current_price=current_price, outputs=outputs
        )

    def _format_response(
        self, symbol: str, current_price: float, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Format VelocityTransformerModel output into velocity response structure.

        Args:
            symbol: Stock symbol
            current_price: Current price
            outputs: Model output dictionary with velocity tensors

        Returns:
            Formatted velocity response
        """
        from models.heads.velocity_heads import STANDARD_TIMEFRAMES

        velocities = []

        # Get tensor outputs (batch size 1, so index [0])
        all_long_vel = outputs["all_long_velocities"][0].cpu().numpy()
        all_short_vel = outputs["all_short_velocities"][0].cpu().numpy()
        all_long_bars = outputs["all_long_bars"][0].cpu().numpy()
        all_short_bars = outputs["all_short_bars"][0].cpu().numpy()

        # Get acceleration outputs (new in DIMS.fincoll_total model with learnable sigmoid)
        all_long_accel = (
            outputs.get("all_long_accelerations", [np.zeros_like(all_long_vel)])[0]
            .cpu()
            .numpy()
            if "all_long_accelerations" in outputs
            else np.zeros_like(all_long_vel)
        )
        all_short_accel = (
            outputs.get("all_short_accelerations", [np.zeros_like(all_short_vel)])[0]
            .cpu()
            .numpy()
            if "all_short_accelerations" in outputs
            else np.zeros_like(all_short_vel)
        )
        all_accel_weight_long = (
            outputs.get("all_accel_weights_long", [np.ones_like(all_long_vel)])[0]
            .cpu()
            .numpy()
            if "all_accel_weights_long" in outputs
            else np.ones_like(all_long_vel)
        )
        all_accel_weight_short = (
            outputs.get("all_accel_weights_short", [np.ones_like(all_short_vel)])[0]
            .cpu()
            .numpy()
            if "all_accel_weights_short" in outputs
            else np.ones_like(all_short_vel)
        )

        # Get per-timeframe predictions for confidence
        tf_preds = outputs.get("timeframe_predictions", {})

        for i, tf in enumerate(STANDARD_TIMEFRAMES):
            # Get confidence if available
            tf_pred = tf_preds.get(tf.name, {})
            long_conf = (
                float(
                    tf_pred.get("long_confidence", torch.tensor([[0.5]]))[0, 0]
                    .cpu()
                    .item()
                )
                if "long_confidence" in tf_pred
                else 0.5
            )
            short_conf = (
                float(
                    tf_pred.get("short_confidence", torch.tensor([[0.5]]))[0, 0]
                    .cpu()
                    .item()
                )
                if "short_confidence" in tf_pred
                else 0.5
            )

            # Long velocity
            long_vel = float(all_long_vel[i])
            long_bars = max(1, int(all_long_bars[i]))
            long_accel = float(all_long_accel[i])
            long_accel_weight = float(all_accel_weight_long[i])
            long_weighted_score = abs(long_vel) * long_accel_weight

            velocities.append(
                {
                    "velocity": long_vel,
                    "timeframe": tf.name,
                    "bars": long_bars,
                    "seconds": tf.seconds * long_bars,
                    "direction": "LONG",
                    "confidence": long_conf,
                    "expected_return": long_vel * long_bars,
                    "acceleration": long_accel,
                    "accel_weight": long_accel_weight,
                    "weighted_score": long_weighted_score,
                }
            )

            # Short velocity
            short_vel = float(all_short_vel[i])
            short_bars = max(1, int(all_short_bars[i]))
            short_accel = float(all_short_accel[i])
            short_accel_weight = float(all_accel_weight_short[i])
            short_weighted_score = abs(short_vel) * short_accel_weight

            velocities.append(
                {
                    "velocity": short_vel,
                    "timeframe": tf.name,
                    "bars": short_bars,
                    "seconds": tf.seconds * short_bars,
                    "direction": "SHORT",
                    "confidence": short_conf,
                    "expected_return": short_vel * short_bars,
                    "acceleration": short_accel,
                    "accel_weight": short_accel_weight,
                    "weighted_score": short_weighted_score,
                }
            )

        # Sort by weighted_score (acceleration-aware, highest first)
        # This prioritizes opportunities with high velocity AND high acceleration
        velocities.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Best opportunity
        best_opportunity = velocities[0] if velocities else None

        # Spike alert
        spike_detected = bool(outputs["spike_detected"][0, 0].cpu().item() > 0.5)
        spike_magnitude = float(outputs["spike_magnitude"][0, 0].cpu().item())
        spike_direction = float(outputs["spike_direction"][0, 0].cpu().item())

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "velocities": velocities,
            "best_opportunity": best_opportunity,
            "spike_alert": {
                "detected": spike_detected,
                "magnitude": spike_magnitude,
                "direction": spike_direction,
            },
            "metadata": {
                "model": "velocity_transformer_v1",
                "input_dim": self.input_dim,
                "device": str(self.device),
                "timeframes": [tf.name for tf in STANDARD_TIMEFRAMES],
            },
        }


# Global engine instance (singleton)
_velocity_engine_instance: Optional[VelocityEngine] = None


def get_velocity_engine(
    checkpoint_path: Optional[str] = None, device: str = "auto"
) -> VelocityEngine:
    """
    Get or create velocity engine (singleton).

    Args:
        checkpoint_path: Path to checkpoint (only used on first call)
        device: Device to use ('cpu', 'cuda', 'auto')

    Returns:
        VelocityEngine instance
    """
    global _velocity_engine_instance

    if _velocity_engine_instance is None:
        # Get checkpoint path from environment or use default
        if checkpoint_path is None:
            checkpoint_path = os.getenv(
                "VELOCITY_CHECKPOINT",
                str(
                    Path(__file__).parent.parent.parent.parent
                    / "finvec"
                    / "checkpoints"
                    / "velocity"
                    / "best_model.pt"
                ),
            )

        # Get device from environment
        device_env = os.getenv("VELOCITY_DEVICE")
        if device_env:
            device = device_env

        logger.info(f"Creating VelocityEngine with checkpoint: {checkpoint_path}")
        _velocity_engine_instance = VelocityEngine(checkpoint_path, device)

    return _velocity_engine_instance
