#!/usr/bin/env python3
"""
Historical Feature Updater

Backfills new features into historical training data with zero values,
enabling seamless integration of new features without losing accumulated
training data.

This implements the strategy discussed in:
docs/QUANT_STRATEGIES_EXPANSION.md
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import yaml
import hashlib
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.config.dimensions import DIMS
from fincoll.config.feature_dimensions import load_dimension_config
from fincoll.features.constructor import FeatureConstructor
from fincoll.storage.influxdb_saver import InfluxDBSaver
from fincoll.storage.influxdb_loader import InfluxDBLoader
from fincoll.utils.logger import setup_logger


class HistoricalFeatureUpdater:
    """
    Backfills new features into historical data with zero values.

    This enables training data accumulation even as features evolve.
    """

    def __init__(
        self,
        data_provider: Optional[str] = None,
        batch_size: int = 50,
        max_workers: int = 4,
        dry_run: bool = False,
    ):
        """
        Initialize the historical feature updater.

        Args:
            data_provider: Data provider for feature extraction ('tradestation', 'yfinance', etc.)
            batch_size: Number of symbols to process in parallel
            max_workers: Maximum parallel workers for processing
            dry_run: If True, don't save to database (preview only)
        """
        self.data_provider = data_provider or "yfinance"  # Safe default
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.dry_run = dry_run

        # Initialize components
        self.logger = setup_logger("historical_updater")
        self.feature_constructor = FeatureConstructor()
        self.influx_saver = InfluxDBSaver()
        self.influx_loader = InfluxDBLoader()

        # Load dimension configuration
        try:
            self.dim_config = load_dimension_config() if load_dimension_config else {}
        except Exception as e:
            logging.warning(f"Could not load dimension config: {e}")
            self.dim_config = {}

        self.config_hash = self._compute_config_hash()

        # Track statistics
        self.stats = {
            "processed_days": 0,
            "processed_symbols": 0,
            "updated_samples": 0,
            "failed_samples": 0,
            "new_features_added": 0,
        }

    def _compute_config_hash(self) -> str:
        """Compute MD5 hash of current feature dimension config."""
        config_path = (
            Path(__file__).parent.parent / "config" / "feature_dimensions.yaml"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "rb") as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()[:8]

    def identify_new_features(self) -> List[Tuple[str, int]]:
        """
        Identify features that have been added but not implemented.

        Returns:
            List of (feature_name, dimension_count) tuples for unimplemented features
        """
        new_features = []

        # Check each component in the config
        for component_name, component_config in self.dim_config["fincoll"][
            "components"
        ].items():
            dims = (
                component_config
                if isinstance(component_config, int)
                else component_config.get("dimensions", 0)
            )

            if dims > 0:
                # Check if this feature is actually implemented
                is_implemented = self._check_feature_implemented(component_name)
                if not is_implemented:
                    new_features.append((component_name, dims))
                    self.logger.info(
                        f"Feature not implemented: {component_name} ({dims}D)"
                    )

        return new_features

    def _check_feature_implemented(self, component_name: str) -> bool:
        """Check if a feature component is actually implemented in the extractor."""
        # This would check if the feature extractor actually populates this component
        # For now, we'll check the validation thresholds - 1.00 means not implemented
        thresholds = self.dim_config.get("validation", {}).get("service_thresholds", {})

        if component_name in thresholds:
            # If threshold is 1.00 (100% zeros acceptable), it's not implemented
            return thresholds[component_name] < 1.0

        # Default to checking if extractor has the method
        extractor_methods = [
            f"extract_{component_name}",
            f"get_{component_name}",
            component_name,
        ]

        for method_name in extractor_methods:
            if hasattr(self.feature_constructor, method_name):
                # Method exists - assume implemented
                return True

        # Check historical logs for recent non-zero values
        return self._check_historical_nonzeros(component_name)

    def _check_historical_nonzeros(self, component_name: str) -> bool:
        """Check historical data for evidence of feature implementation."""
        try:
            # Query last 30 days for this component
            df = self.influx_loader.load_features(
                symbols=["AAPL"],  # Test with a single symbol
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
            )

            if df.empty:
                return False

            # Check if any non-zero values exist for this component's dimensions
            component_dims = self._get_component_dimensions(component_name)
            if not component_dims:
                return False

            # Sample a few dimensions from this component
            sample_dims = component_dims[: min(3, len(component_dims))]

            for dim in sample_dims:
                if dim in df.columns and df[dim].abs().sum() > 0:
                    return True

            return False

        except Exception as e:
            self.logger.warning(
                f"Could not check historical data for {component_name}: {e}"
            )
            return False

    def _get_component_dimensions(self, component_name: str) -> List[str]:
        """Get the dimension names/ranges for a component."""
        # Map component names to dimension ranges
        dim_mapping = {
            "technical": [f"f{i:03d}" for i in range(0, 81)],
            "advanced_technical": [f"f{i:03d}" for i in range(81, 131)],
            "velocity": [f"f{i:03d}" for i in range(131, 151)],
            "news": [f"f{i:03d}" for i in range(151, 171)],
            "fundamentals": [f"f{i:03d}" for i in range(171, 187)],
            "cross_asset": [f"f{i:03d}" for i in range(187, 205)],
            "sector": [f"f{i:03d}" for i in range(205, 219)],
            "options": [f"f{i:03d}" for i in range(219, 229)],
            "support_resistance": [f"f{i:03d}" for i in range(229, 259)],
            "vwap": [f"f{i:03d}" for i in range(259, 264)],
            "senvec": [f"f{i:03d}" for i in range(264, 313)],
            "futures": [f"f{i:03d}" for i in range(313, 338)],
            "finnhub": [f"f{i:03d}" for i in range(338, 353)],
            "early_signal": [f"f{i:03d}" for i in range(376, 406)],
            "market_neutral": [f"f{i:03d}" for i in range(406, 423)],
            "advanced_risk": [f"f{i:03d}" for i in range(423, 431)],
            "momentum_variations": [f"f{i:03d}" for i in range(431, 437)],
        }

        return dim_mapping.get(component_name, [])

        return dim_mapping.get(component_name, [])

    def backfill_range(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        force_reprocess: bool = False,
    ) -> Dict:
        """
        Backfill new features for specified date range and symbols.

        Args:
            symbols: List of symbols to backfill
            start_date: Start date for backfill
            end_date: End date for backfill
            force_reprocess: If True, reprocess even if data already exists

        Returns:
            Dictionary with statistics
        """
        self.logger.info(
            f"Starting backfill: {len(symbols)} symbols, {start_date} to {end_date}"
        )

        # Identify new/unimplemented features
        new_features = self.identify_new_features()
        if not new_features:
            self.logger.info(
                "No new features to backfill. All components appear implemented."
            )
            return self.stats

        self.logger.info(f"Found {len(new_features)} new features to backfill:")
        for feature_name, dims in new_features:
            self.logger.info(f"  - {feature_name}: {dims} dimensions")
            self.stats["new_features_added"] += dims

        # Process in batches
        date_range = pd.date_range(start_date, end_date, freq="D")

        for i in tqdm(
            range(0, len(symbols), self.batch_size), desc="Processing symbol batches"
        ):
            batch_symbols = symbols[i : i + self.batch_size]

            for current_date in tqdm(
                date_range,
                desc=f"Backfilling {len(batch_symbols)} symbols",
                leave=False,
            ):
                self._process_date_batch(
                    batch_symbols, current_date, new_features, force_reprocess
                )

        self._print_summary()
        return self.stats

    def _process_date_batch(
        self,
        symbols: List[str],
        current_date: datetime,
        new_features: List[Tuple[str, int]],
        force_reprocess: bool,
    ):
        """Process a batch of symbols for a specific date."""

        for symbol in symbols:
            try:
                # Check if we already have data for this symbol/date
                if not force_reprocess and self._has_existing_data(
                    symbol, current_date
                ):
                    continue

                # Extract existing features (will populate implemented features)
                existing_features = self._extract_existing_features(
                    symbol, current_date
                )

                if existing_features is None or len(existing_features) == 0:
                    self.logger.debug(
                        f"No existing data for {symbol} on {current_date}"
                    )
                    continue

                # Augment with zeros for new features
                updated_features = self._augment_with_zeros(
                    existing_features, new_features
                )

                # Save to database
                if not self.dry_run:
                    self._save_features(symbol, current_date, updated_features)

                self.stats["updated_samples"] += 1

            except Exception as e:
                self.logger.error(f"Error processing {symbol} on {current_date}: {e}")
                self.stats["failed_samples"] += 1

    def _has_existing_data(self, symbol: str, date: datetime) -> bool:
        """Check if training data already exists for symbol/date."""
        try:
            df = self.influx_loader.load_features(
                symbols=[symbol], start_date=date, end_date=date + timedelta(days=1)
            )
            return not df.empty
        except Exception as e:
            self.logger.debug(f"Could not check existing data for {symbol}: {e}")
            return False

    def _extract_existing_features(
        self, symbol: str, date: datetime
    ) -> Optional[np.ndarray]:
        """Extract existing features for a symbol/date."""
        try:
            # Use the feature constructor to extract what data is available
            features = self.feature_constructor.construct(
                symbol=symbol, date=date, provider=self.data_provider
            )
            return features
        except Exception as e:
            self.logger.debug(f"Could not extract features for {symbol}: {e}")
            return None

    def _augment_with_zeros(
        self, existing_features: np.ndarray, new_features: List[Tuple[str, int]]
    ) -> np.ndarray:
        """Augment existing features with zeros for new features."""

        # Start with existing features
        if len(existing_features.shape) == 1:
            # Single vector
            augmented = existing_features.copy()
        else:
            # Multiple vectors (time series) - process each
            augmented = existing_features.copy()

        # For each new feature component, append zeros
        for feature_name, dim_count in new_features:
            zeros = np.zeros(dim_count, dtype=np.float32)

            if len(augmented.shape) == 1:
                augmented = np.concatenate([augmented, zeros])
            else:
                # For time series, add zero columns
                zero_cols = np.zeros((augmented.shape[0], dim_count), dtype=np.float32)
                augmented = np.concatenate([augmented, zero_cols], axis=1)

        return augmented

    def _save_features(self, symbol: str, date: datetime, features: np.ndarray):
        """Save augmented features to training database."""

        # Create metadata tags
        tags = {
            "symbol": symbol,
            "feature_dim": f"{len(features)}D",
            "config_version": f"v{self.config_hash}",
            "backfill": "true",
            "backfill_date": datetime.now().isoformat(),
        }

        # Use InfluxDB saver
        self.influx_saver.save_training_sample(
            symbol=symbol,
            timestamp=date,
            features=features,
            targets=None,  # No targets for historical data
            tags=tags,
        )

    def update_existing_vectors(self):
        """
        Update existing training vectors with zeros for newly added features.

        This is useful when you've already collected some training data
        and then add a new feature. Old vectors can be updated in-place.
        """
        self.logger.info("Updating existing vectors with new features...")

        new_features = self.identify_new_features()
        if not new_features:
            self.logger.info("No new features to add.")
            return

        # Query all existing training data
        existing_data = self.influx_loader.load_all_training_samples()

        if existing_data.empty:
            self.logger.info("No existing training data found.")
            return

        updated_count = 0
        for idx, row in existing_data.iterrows():
            try:
                symbol = row["symbol"]
                timestamp = row["timestamp"]
                existing_vector = row["features"]

                # Augment with zeros
                updated_vector = self._augment_with_zeros(existing_vector, new_features)

                # Update in database
                if not self.dry_run:
                    self._save_features(symbol, timestamp, updated_vector)

                updated_count += 1

            except Exception as e:
                self.logger.error(f"Error updating row {idx}: {e}")

        self.logger.info(f"Updated {updated_count} existing vectors.")

    def _print_summary(self):
        """Print backfill statistics."""
        print("\n" + "=" * 80)
        print("BACKFILL COMPLETE")
        print("=" * 80)
        print(f"Processed days: {self.stats['processed_days']}")
        print(f"Processed symbols: {self.stats['processed_symbols']}")
        print(f"Updated samples: {self.stats['updated_samples']}")
        print(f"Failed samples: {self.stats['failed_samples']}")
        print(f"New dimensions added: {self.stats['new_features_added']}D")
        print(f"Config version: v{self.config_hash}")
        print("=" * 80)

    def validate_backfill(self, sample_size: int = 100):
        """
        Validate that backfill was successful by checking sample records.

        Args:
            sample_size: Number of records to randomly sample and check

        Returns:
            Validation result dictionary
        """
        self.logger.info(f"Validating backfill with {sample_size} samples...")

        # Load random samples
        samples = self.influx_loader.get_random_samples(sample_size, backfill_only=True)

        validation_results = {
            "total_checked": len(samples),
            "correct_dim": 0,
            "has_expected_zeros": 0,
            "errors": [],
        }

        for idx, row in samples.iterrows():
            try:
                symbol = row["symbol"]
                timestamp = row["timestamp"]
                features = row["features"]

                # Check dimension
                expected_dim = DIMS.fincoll_total
                if len(features) == expected_dim:
                    validation_results["correct_dim"] += 1
                else:
                    validation_results["errors"].append(
                        f"{symbol} {timestamp}: Expected {expected_dim}D, got {len(features)}D"
                    )

                # Verify new features are zeros
                new_features = self.identify_new_features()
                for feature_name, dims in new_features:
                    start_idx = self._get_new_feature_start_idx(feature_name)
                    if start_idx >= 0 and start_idx + dims <= len(features):
                        if np.all(features[start_idx : start_idx + dims] == 0):
                            validation_results["has_expected_zeros"] += 1
                        else:
                            validation_results["errors"].append(
                                f"{symbol} {timestamp}: {feature_name} should be zeros but has non-zero values"
                            )

            except Exception as e:
                validation_results["errors"].append(f"Error validating row {idx}: {e}")

        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"Records checked: {validation_results['total_checked']}")
        print(f"Correct dimensions: {validation_results['correct_dim']}")
        print(f"Expected zeros present: {validation_results['has_expected_zeros']}")
        if validation_results["errors"]:
            print(f"Errors found: {len(validation_results['errors'])}")
            for error in validation_results["errors"][:5]:  # Show first 5
                print(f"  - {error}")
        print("=" * 80)

        return validation_results

    def _get_new_feature_start_idx(self, feature_name: str) -> int:
        """Get the starting index for a new feature in the vector."""
        # Calculate based on feature order in dimensions.yaml
        ordered_features = [
            "technical",
            "advanced_technical",
            "velocity",
            "news",
            "fundamentals",
            "cross_asset",
            "sector",
            "options",
            "support_resistance",
            "vwap",
            "senvec",
            "futures",
            "finnhub",
            "early_signal",
            "market_neutral",
            "advanced_risk",
            "momentum_variations",
        ]

        idx = 0
        for feature in ordered_features:
            component_config = self.dim_config["fincoll"]["components"].get(feature, 0)
            if isinstance(component_config, dict):
                dims = component_config.get("dimensions", 0)
            else:
                dims = component_config

            if feature == feature_name:
                return idx

            idx += dims

        return -1


def main():
    """Command-line interface for historical backfill."""
    parser = argparse.ArgumentParser(
        description="Backfill new features into historical training data with zeros"
    )

    parser.add_argument(
        "--symbols",
        "-s",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="Symbols to backfill (default: tech giants)",
    )

    parser.add_argument(
        "--start-date",
        "-start",
        type=lambda d: datetime.strptime(d, "%Y-%m-%d"),
        default=datetime.now() - timedelta(days=365),
        help="Start date (YYYY-MM-DD, default: 1 year ago)",
    )

    parser.add_argument(
        "--end-date",
        "-end",
        type=lambda d: datetime.strptime(d, "%Y-%m-%d"),
        default=datetime.now(),
        help="End date (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update existing vectors in-place with new zero features",
    )

    parser.add_argument(
        "--validate", action="store_true", help="Run validation after backfill"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without saving to database"
    )

    parser.add_argument(
        "--batch-size", type=int, default=50, help="Symbols per batch (default: 50)"
    )

    args = parser.parse_args()

    # Create updater
    updater = HistoricalFeatureUpdater(batch_size=args.batch_size, dry_run=args.dry_run)

    # Identify new features
    print("Detecting new features...")
    new_features = updater.identify_new_features()

    if not new_features:
        print("No new features detected. Exiting.")
        return

    print(f"\nFound {len(new_features)} new features:")
    for name, dims in new_features:
        print(f"  - {name}: {dims} dimensions")

    # Confirm with user
    if not args.dry_run:
        response = input("\nProceed with backfill? (yes/no): ")
        if response.lower() != "yes":
            print("Backfill cancelled.")
            return

    # Perform backfill or update
    if args.update_existing:
        updater.update_existing_vectors()
    else:
        updater.backfill_range(
            symbols=args.symbols, start_date=args.start_date, end_date=args.end_date
        )

    # Validate if requested
    if args.validate:
        updater.validate_backfill()


if __name__ == "__main__":
    main()
