#!/usr/bin/env python3
"""
Complete tests for FeatureLabeler production edge cases

Targets the remaining 4% uncovered lines to achieve 100% coverage
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fincoll.features.feature_labeler import FeatureLabeler
from config.dimensions import DIMS


class TestFeatureLabelerProductionEdgeCases:
    """Test production edge cases for 100% coverage"""

    def test_labeler_handles_extreme_values(self):
        """Labeler should handle extreme price movements"""
        labeler = FeatureLabeler()

        # Simulate extreme volatility with huge swings
        features = np.zeros(DIMS.fincoll_total)
        features[0] = 1000.0  # Extreme RSI (way out of 0-100 range)
        features[1] = 500.0  # Extreme MACD
        features[3] = 200.0  # Extreme ADX

        labeled = labeler.label(features)

        # Should not crash, return valid labels
        assert labeled is not None
        assert "technical_indicators" in labeled
        assert labeled["technical_indicators"]["data"]["momentum"]["rsi_14"] == 1000.0

    def test_labeler_handles_zero_prices(self):
        """Labeler should handle zero or near-zero prices"""
        labeler = FeatureLabeler()

        # All zeros
        zero_features = np.zeros(DIMS.fincoll_total)

        # Should handle without division by zero
        labeled = labeler.label(zero_features)

        assert labeled is not None
        assert "fundamentals" in labeled
        assert labeled["fundamentals"]["data"]["valuation"]["pe_ratio"] == 0.0

    def test_missing_dims_key_raises_error(self):
        """Test that missing DIMS config key raises ValueError"""
        # Create a mock DIMS object that's missing a required key
        mock_dims = MagicMock()
        mock_dims.fincoll_total = 414
        mock_dims.fincoll_technical = 50
        # Missing fincoll_advanced_technical will trigger line 78
        # Use delattr to actually remove the attribute
        delattr(mock_dims, "fincoll_advanced_technical")

        with patch("fincoll.features.feature_labeler.DIMS", mock_dims):
            with pytest.raises(ValueError, match="DIMS config missing required key"):
                FeatureLabeler()

    def test_safe_summary_format_exception(self):
        """Test _safe_summary handles format string exceptions"""
        labeler = FeatureLabeler()

        # Create features that will trigger format exception
        features = np.array([1.0, 2.0, 3.0])

        # Use a format string that will fail (wrong number of placeholders)
        # This should trigger the except block on lines 255-256
        result = labeler._safe_summary(
            features,
            len(features),
            [0, 1],
            "Value {:.1f} {:.1f} {:.1f}",  # 3 placeholders but only 2 indices
            "test",
        )

        # Should return fallback message instead of crashing
        assert "insufficient_test_data" in result

    def test_safe_summary_fallback_to_first_feature(self):
        """Test _safe_summary falls back to first feature when indices too large"""
        labeler = FeatureLabeler()

        # Small array but large index requirement
        features = np.array([42.0])
        n = 1
        indices = [0, 5]  # Index 5 doesn't exist

        # Should trigger line 257-259 (fallback to first feature only)
        result = labeler._safe_summary(features, n, indices, "Value {:.1f}", "fallback")

        # Should fall back to first feature format
        assert "Fallback" in result  # label.title()
        assert "42.00" in result  # features[0]

    def test_safe_summary_with_inf_in_required_indices(self):
        """Test _safe_summary handles inf in required indices"""
        labeler = FeatureLabeler()

        # Features with inf at a required index
        features = np.array([1.0, np.inf, 3.0])
        n = 3
        indices = [0, 1]  # Index 1 is inf

        # Should fail isfinite check and return fallback
        result = labeler._safe_summary(
            features, n, indices, "Values {:.1f} {:.1f}", "inf_test"
        )

        # Should fall back because not all indices are finite
        # The actual behavior is it falls back to first feature only (line 259)
        assert "Inf_Test" in result and "1.00" in result

    def test_safe_summary_with_nan_first_feature(self):
        """Test _safe_summary handles NaN in first feature"""
        labeler = FeatureLabeler()

        # First feature is NaN, indices out of range
        features = np.array([np.nan])
        n = 1
        indices = [0, 10]  # Index 10 doesn't exist

        # Should fail both conditions and return final fallback
        result = labeler._safe_summary(features, n, indices, "Value {:.1f}", "nan_test")

        # Should return final fallback
        assert result == "insufficient_nan_test_data"

    def test_label_with_mixed_finite_infinite_values(self):
        """Test labeling handles mixed finite and infinite values"""
        labeler = FeatureLabeler()

        features = np.random.randn(DIMS.fincoll_total)
        # Inject some inf and nan values
        features[0] = np.inf  # RSI
        features[10] = np.nan  # Some other indicator
        features[100] = -np.inf

        # Should not crash
        labeled = labeler.label(features)

        assert labeled is not None
        assert isinstance(labeled, dict)
        assert "technical_indicators" in labeled

    def test_label_velocity_with_nan(self):
        """Test velocity labeling with NaN values"""
        labeler = FeatureLabeler()

        features = np.full(DIMS.fincoll_total, 0.0)
        # Set velocity section to NaN
        vel_start = DIMS.fincoll_technical + DIMS.fincoll_advanced_technical
        vel_end = vel_start + DIMS.fincoll_velocity
        features[vel_start:vel_end] = np.nan

        labeled = labeler.label(features)

        assert "velocity" in labeled
        assert labeled["velocity"]["summary"] == "no_velocity"

    def test_label_cross_asset_partial_data(self):
        """Test cross-asset labeling with partial horizon data"""
        labeler = FeatureLabeler()

        # Create minimal features (less than all horizons)
        features = np.zeros(5)  # Only enough for 1-day horizon (6 features needed)

        labeled = labeler._label_cross_asset(features)

        # Should handle gracefully with partial data
        assert labeled["dimensions"] == 5
        # Should not have all horizons
        assert "horizon_5d" not in labeled["data"]
        assert "horizon_20d" not in labeled["data"]


class TestFeatureLabelerIntegration:
    """Integration tests with real-world scenarios"""

    def test_full_pipeline_with_realistic_data(self):
        """Test complete labeling pipeline with realistic market data"""
        labeler = FeatureLabeler()

        # Generate realistic-ish features
        features = np.random.randn(DIMS.fincoll_total) * 0.1  # Small random values

        # Set some realistic technical indicators
        features[0] = 55.0  # RSI neutral
        features[1] = 0.01  # MACD slightly bullish
        features[3] = 25.0  # ADX moderate trend

        labeled = labeler.label(features)

        # Verify all expected components present
        expected_components = [
            "technical_indicators",
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

        for component in expected_components:
            assert component in labeled
            assert "dimensions" in labeled[component]
            assert "data" in labeled[component]
            assert "summary" in labeled[component]

    def test_stress_test_rapid_labeling(self):
        """Stress test: label many vectors rapidly"""
        labeler = FeatureLabeler()

        # Label 100 random vectors
        for _ in range(100):
            features = np.random.randn(DIMS.fincoll_total)
            labeled = labeler.label(features)
            assert labeled is not None

    def test_dimension_ranges_sum_to_total(self):
        """Verify dimension ranges are mathematically correct"""
        labeler = FeatureLabeler()
        ranges = labeler._ranges

        # Check no overlaps and complete coverage
        previous_end = 0
        total_size = 0

        for name, info in ranges.items():
            # Check contiguous
            assert info["start"] == previous_end
            # Check size matches
            assert info["end"] - info["start"] == info["size"]
            previous_end = info["end"]
            total_size += info["size"]

        # Check total matches
        assert total_size == DIMS.fincoll_total
