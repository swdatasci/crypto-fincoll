"""
A/B Testing Manager for FinColl API

Manages model variant serving for canary deployments and A/B testing.
Implements hash-based traffic splitting for consistent variant assignment.

Architecture:
    - Control model: Current production model
    - Treatment model: New model being tested
    - Traffic split: Percentage of traffic routed to treatment (e.g., 10%)
    - Hash-based assignment: Consistent symbol-to-variant mapping
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ABExperiment:
    """A/B test experiment configuration"""
    experiment_id: str
    control_model: str
    treatment_model: str
    traffic_split: float  # 0.0 - 1.0 (e.g., 0.1 = 10% treatment)
    status: str  # 'running', 'paused', 'completed'
    start_time: datetime
    end_time: Optional[datetime] = None


class ABTestingManager:
    """
    Manages A/B testing for model deployments in FinColl.

    Uses hash-based traffic splitting for consistent variant assignment.
    Symbols consistently map to same variant throughout experiment.
    """

    def __init__(self):
        self.active_experiment: Optional[ABExperiment] = None
        self._control_model = None
        self._treatment_model = None

        logger.info("ABTestingManager initialized")

    def load_experiment(self, experiment_config: Dict):
        """
        Load A/B test experiment configuration

        Args:
            experiment_config: {
                'experiment_id': 'exp_123',
                'control_model': 'epoch45_2026-02-01',
                'treatment_model': 'epoch50_2026-02-08',
                'traffic_split': 0.1,  # 10% to treatment
                'status': 'running'
            }
        """
        self.active_experiment = ABExperiment(
            experiment_id=experiment_config['experiment_id'],
            control_model=experiment_config['control_model'],
            treatment_model=experiment_config['treatment_model'],
            traffic_split=experiment_config['traffic_split'],
            status=experiment_config.get('status', 'running'),
            start_time=datetime.now(),
            end_time=None
        )

        logger.info(
            f"📊 A/B Test Loaded: {self.active_experiment.experiment_id} "
            f"(Control: {self.active_experiment.control_model}, "
            f"Treatment: {self.active_experiment.treatment_model}, "
            f"Split: {self.active_experiment.traffic_split * 100:.0f}%)"
        )

    def assign_variant(self, symbol: str) -> tuple[str, str]:
        """
        Assign variant (control or treatment) for a symbol

        Uses consistent hash-based assignment so same symbol
        always gets same variant throughout experiment.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Tuple of (variant, model_version)
            - variant: 'control' or 'treatment'
            - model_version: Model version to use
        """
        if not self.active_experiment or self.active_experiment.status != 'running':
            return ('control', 'active')  # Default: use active model

        # Hash-based assignment for consistency
        hash_val = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
        normalized = (hash_val % 100) / 100.0  # Normalize to 0-1

        if normalized < self.active_experiment.traffic_split:
            # Treatment group
            return ('treatment', self.active_experiment.treatment_model)
        else:
            # Control group
            return ('control', self.active_experiment.control_model)

    def get_variant_metadata(self, symbol: str) -> Dict:
        """
        Get A/B test metadata for prediction response

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with A/B test metadata
        """
        if not self.active_experiment or self.active_experiment.status != 'running':
            return {
                'ab_test_active': False
            }

        variant, model_version = self.assign_variant(symbol)

        return {
            'ab_test_active': True,
            'experiment_id': self.active_experiment.experiment_id,
            'variant': variant,
            'model_version': model_version,
            'traffic_split': self.active_experiment.traffic_split,
            'start_time': self.active_experiment.start_time.isoformat()
        }

    def stop_experiment(self):
        """Stop the active experiment"""
        if self.active_experiment:
            self.active_experiment.status = 'completed'
            self.active_experiment.end_time = datetime.now()

            logger.info(
                f"📊 A/B Test Stopped: {self.active_experiment.experiment_id}"
            )

        self.active_experiment = None
        self._control_model = None
        self._treatment_model = None

    def get_experiment_status(self) -> Optional[Dict]:
        """Get current experiment status"""
        if not self.active_experiment:
            return None

        return {
            'experiment_id': self.active_experiment.experiment_id,
            'control_model': self.active_experiment.control_model,
            'treatment_model': self.active_experiment.treatment_model,
            'traffic_split': self.active_experiment.traffic_split,
            'status': self.active_experiment.status,
            'start_time': self.active_experiment.start_time.isoformat(),
            'end_time': self.active_experiment.end_time.isoformat() if self.active_experiment.end_time else None
        }


# Global singleton instance
ab_manager = ABTestingManager()
