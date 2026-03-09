"""FinColl Training Module

Training interface for velocity models.
"""

from .train_interface import (
    train_velocity_model,
    TrainingConfig
)

__all__ = [
    'train_velocity_model',
    'TrainingConfig',
]
