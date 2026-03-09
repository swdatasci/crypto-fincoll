"""
Model Registry Client for FinVec Integration

This client enables FinVec training scripts to auto-register model checkpoints
in the PIM Model Registry after training completes.

Usage:
    from fincoll.model_registry_client import ModelRegistryClient

    client = ModelRegistryClient()
    result = client.register_checkpoint(
        checkpoint_path='checkpoints/epoch50.pt',
        metrics={'train_loss': 0.025, 'val_loss': 0.031, ...},
        training_config={'epochs': 50, 'batch_size': 64, ...}
    )
"""

import requests
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelRegistryClient:
    """Client for PIM Model Registry API

    Connects to PassiveIncomeMaximizer's Express API to register trained
    FinVec checkpoints and retrieve model metadata.
    """

    def __init__(self, base_url: str = 'http://10.32.3.27:5000'):
        """Initialize Model Registry client

        Args:
            base_url: PIM Express API base URL (default: http://10.32.3.27:5000)
        """
        self.base_url = base_url
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_checkpoint(
        self,
        checkpoint_path: str,
        metrics: Dict[str, float],
        training_config: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a FinVec checkpoint in PIM Model Registry

        Args:
            checkpoint_path: Path to .pt checkpoint file
            metrics: Training metrics
                - train_loss: Training loss (required)
                - val_loss: Validation loss (required)
                - test_loss: Test loss (optional)
                - sharpe_ratio: Sharpe ratio (optional)
                - win_rate: Win rate (optional)
                - avg_profit: Average profit (optional)
                - max_drawdown: Maximum drawdown (optional)
            training_config: Training configuration
                - epochs: Total epochs (required)
                - current_epoch: Current epoch (required)
                - batch_size: Batch size (required)
                - learning_rate: Learning rate (required)
                - symbols: Symbols trained on (optional)
                - dropout: Dropout rate (optional)
                - data_path: Data path (optional)
                - training_duration_seconds: Training duration (optional)
                - gpu: GPU device (optional)
            dataset_info: Dataset information (optional)
                - total_samples: Total samples
                - train_samples: Training samples
                - val_samples: Validation samples
                - test_samples: Test samples
                - symbols: Symbol list
                - date_range: {'start': '2023-01-01', 'end': '2024-12-31'}
                - data_quality: Data quality score (0-1)
            notes: Additional notes (optional)

        Returns:
            {'success': True, 'version': 'epoch50_2026-02-08', 'version_id': 123}
            OR
            {'success': False, 'error': 'error message'}
        """
        import os

        # Generate version string
        epoch = training_config.get('current_epoch', training_config.get('epochs', 0))
        date_str = datetime.now().strftime('%Y-%m-%d')
        version = f"epoch{epoch}_{date_str}"

        # Build payload matching PIM Model Registry API schema
        payload = {
            'version': version,
            'checkpoint_path': checkpoint_path,
            'model_type': 'transformer',
            'timestamp': datetime.now().isoformat(),

            # Training configuration (camelCase for TypeScript compatibility)
            'trainingConfig': {
                'modelType': 'transformer',
                'architecture': training_config.get('architecture', 'transformer_d512_h8_l6'),
                'hyperparameters': {
                    'epochs': training_config.get('epochs'),
                    'batch_size': training_config.get('batch_size'),
                    'learning_rate': training_config.get('learning_rate'),
                    'dropout': training_config.get('dropout', 0.1),
                },
                'trainingDataPath': training_config.get('data_path', ''),
                'checkpointPath': checkpoint_path,
                'trainingDuration': training_config.get('training_duration_seconds', 0),
                'gpuUsed': training_config.get('gpu', 'unknown')
            },

            # Performance metrics (camelCase for TypeScript compatibility)
            'performance': {
                'trainLoss': metrics.get('train_loss'),
                'valLoss': metrics.get('val_loss'),
                'testLoss': metrics.get('test_loss'),
                'trainAccuracy': metrics.get('train_accuracy', 0.0),
                'valAccuracy': metrics.get('val_accuracy', 0.0),
                'testAccuracy': metrics.get('test_accuracy', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1Score': metrics.get('f1_score', 0.0),
                'sharpeRatio': metrics.get('sharpe_ratio'),
                'maxDrawdown': metrics.get('max_drawdown'),
                'winRate': metrics.get('win_rate'),
                'avgProfit': metrics.get('avg_profit')
            },

            # Dataset information (camelCase for TypeScript compatibility)
            'datasetInfo': dataset_info or {
                'totalSamples': training_config.get('total_samples', 0),
                'trainSamples': training_config.get('train_samples', 0),
                'valSamples': training_config.get('val_samples', 0),
                'testSamples': training_config.get('test_samples', 0),
                'symbols': training_config.get('symbols', []),
                'dateRange': {
                    'start': training_config.get('start_date', '2023-01-01'),
                    'end': training_config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
                },
                'dataQuality': 0.95  # Default quality score
            },

            # Deployment metadata
            'deploymentStatus': 'registered',
            'tags': ['transformer', 'auto-registered'],
            'notes': notes or f"Auto-registered from FinVec training on {date_str}"
        }

        try:
            self.logger.info(f"Registering checkpoint: {version}")
            response = requests.post(
                f"{self.base_url}/api/finvec/models/register",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

            self.logger.info(f"✅ Checkpoint registered: {version}")
            return {
                'success': True,
                'version': version,
                **result
            }

        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to register checkpoint: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def get_best_model(
        self,
        metric: str = 'sharpe_ratio',
        limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Get best model(s) by metric

        Args:
            metric: Metric to sort by (default: sharpe_ratio)
                Options: sharpe_ratio, win_rate, train_loss, val_loss
            limit: Number of results to return (default: 1)

        Returns:
            Model metadata dict or None if request fails
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/finvec/models/best",
                params={'metric': metric, 'limit': limit},
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get best model: {e}")
            return None

    def list_models(
        self,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Optional[Dict[str, Any]]:
        """List registered models

        Args:
            status: Filter by deployment status (registered, canary, active, archived)
            limit: Number of results to return (default: 10)
            offset: Pagination offset (default: 0)

        Returns:
            {'models': [...], 'total': N} or None if request fails
        """
        try:
            params = {'limit': limit, 'offset': offset}
            if status:
                params['status'] = status

            response = requests.get(
                f"{self.base_url}/api/finvec/models/list",
                params=params,
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list models: {e}")
            return None

    def get_model(self, version: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by version

        Args:
            version: Model version (e.g., 'v7_epoch50_2026-02-08')

        Returns:
            Model metadata dict or None if request fails
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/finvec/models/{version}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get model {version}: {e}")
            return None

    def health_check(self) -> bool:
        """Check if PIM Model Registry API is accessible

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/health",
                timeout=5
            )
            return response.status_code == 200

        except requests.exceptions.RequestException:
            return False
