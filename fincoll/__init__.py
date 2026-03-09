"""FinColl - Financial Data Collection & Feature Orchestration

Pure Python library for PIM integration.

Architecture:
    InfluxDB Cache -> TradeStation -> SenVec -> Features -> FinVec -> Predictions

Key Functions:
    - get_velocity_predictions(): Get ML predictions for a symbol
    - train_velocity_model(): Train velocity model on historical data

Usage:
    from fincoll import get_velocity_predictions

    predictions = get_velocity_predictions(
        symbol='AAPL',
        current_price=175.50,
        use_cache=True  # Check InfluxDB first
    )

    # predictions = {
    #     'symbol': 'AAPL',
    #     'timestamp': '2025-12-22T...',
    #     'current_price': 175.50,
    #     'velocities': [...],  # Multi-timeframe predictions
    #     'best_opportunity': {...}  # Highest velocity
    # }
"""

from .api.pim_interface import VelocityPredictionConfig, get_velocity_predictions
from .training.train_interface import TrainingConfig, train_velocity_model

__version__ = "1.0.0"

__all__ = [
    # Inference API
    "get_velocity_predictions",
    "VelocityPredictionConfig",
    # Training API
    "train_velocity_model",
    "TrainingConfig",
]
