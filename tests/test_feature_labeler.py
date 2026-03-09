#!/usr/bin/env python3
"""
Unit tests for FeatureLabeler

Tests feature labeling, interpretations, and dimension handling
"""

import pytest
import numpy as np
from fincoll.features.feature_labeler import FeatureLabeler
from fincoll.features.interpretations import (
    interpret_rsi,
    interpret_macd_histogram,
    interpret_adx,
    interpret_put_call_ratio,
    interpret_sentiment_score,
)
from config.dimensions import DIMS


class TestInterpretations:
    """Test interpretation functions"""

    def test_rsi_oversold(self):
        assert interpret_rsi(25.0) == "oversold"
        assert interpret_rsi(35.0) == "oversold_warning"

    def test_rsi_overbought(self):
        assert interpret_rsi(75.0) == "overbought"
        assert interpret_rsi(65.0) == "overbought_warning"

    def test_rsi_neutral(self):
        assert interpret_rsi(50.0) == "neutral"

    def test_macd_bullish(self):
        assert interpret_macd_histogram(0.5, 0.3) == "bullish_crossover"
        assert interpret_macd_histogram(0.3, 0.5) == "bullish_weakening"

    def test_macd_bearish(self):
        assert interpret_macd_histogram(-0.5, -0.3) == "bearish_crossover"
        assert interpret_macd_histogram(-0.3, -0.5) == "bearish_weakening"

    def test_macd_choppy(self):
        assert interpret_macd_histogram(0.00001, 0.00002) == "choppy"

    def test_adx_trend_strength(self):
        assert interpret_adx(15.0) == "weak_trend"
        assert interpret_adx(22.0) == "developing_trend"
        assert interpret_adx(35.0) == "strong_trend"
        assert interpret_adx(60.0) == "very_strong_trend"

    def test_put_call_ratio(self):
        assert interpret_put_call_ratio(0.6) == "bullish_sentiment"
        assert interpret_put_call_ratio(0.85) == "neutral_sentiment"
        assert interpret_put_call_ratio(1.15) == "bearish_sentiment"
        assert interpret_put_call_ratio(1.5) == "extreme_bearish_sentiment"

    def test_sentiment_score(self):
        assert interpret_sentiment_score(0.7) == "very_bullish"
        assert interpret_sentiment_score(0.3) == "bullish"
        assert interpret_sentiment_score(-0.7) == "very_bearish"
        assert interpret_sentiment_score(-0.3) == "bearish"
        assert interpret_sentiment_score(0.1) == "neutral"


class TestFeatureLabeler:
    """Test FeatureLabeler class"""

    def test_initialization(self):
        """Test labeler initializes correctly"""
        labeler = FeatureLabeler()
        assert labeler.dims is not None
        assert labeler.dims == DIMS

    def test_dimension_validation(self):
        """Test dimension config validation"""
        labeler = FeatureLabeler()
        # Should not raise if DIMS has all required keys
        labeler._validate_dimensions()

    def test_calculate_dimension_ranges(self):
        """Test dimension range calculation"""
        labeler = FeatureLabeler()
        ranges = labeler._calculate_dimension_ranges()

        # Verify all components present
        assert 'technical' in ranges
        assert 'senvec' in ranges
        assert 'fundamentals' in ranges

        # Verify ranges are contiguous
        assert ranges['technical']['start'] == 0
        assert ranges['technical']['end'] == DIMS.fincoll_technical

        # Verify no gaps
        offset = 0
        for name, info in ranges.items():
            assert info['start'] == offset
            assert info['end'] == offset + info['size']
            offset = info['end']

        # Verify total matches DIMS.fincoll_total
        total = sum(r['size'] for r in ranges.values())
        assert total == DIMS.fincoll_total

    def test_label_with_correct_dimensions(self):
        """Test labeling with correct number of features"""
        labeler = FeatureLabeler()
        features = np.random.randn(DIMS.fincoll_total)

        labeled = labeler.label(features)

        assert isinstance(labeled, dict)
        assert 'technical_indicators' in labeled
        assert 'fundamentals' in labeled
        assert 'senvec' in labeled

    def test_label_with_wrong_dimensions(self):
        """Test labeling fails with wrong number of features"""
        labeler = FeatureLabeler()
        features = np.random.randn(100)  # Wrong size

        with pytest.raises(ValueError, match="Expected .* features"):
            labeler.label(features)

    def test_label_technical_indicators(self):
        """Test technical indicators labeling"""
        labeler = FeatureLabeler()

        # Create features with known values
        features = np.zeros(DIMS.fincoll_technical)
        features[0] = 75.0  # RSI (overbought)
        features[1] = 0.5   # MACD histogram (bullish)
        features[2] = 0.3   # MACD previous
        features[3] = 35.0  # ADX (strong trend)

        labeled = labeler._label_technical(features)

        assert labeled['dimensions'] == DIMS.fincoll_technical
        assert labeled['data']['momentum']['rsi_14'] == 75.0
        assert labeled['data']['momentum']['rsi_interpretation'] == "overbought"
        assert labeled['data']['trend']['adx'] == 35.0
        assert labeled['data']['trend']['adx_interpretation'] == "strong_trend"

    def test_label_fundamentals(self):
        """Test fundamentals labeling"""
        labeler = FeatureLabeler()

        features = np.zeros(DIMS.fincoll_fundamentals)
        features[0] = 15.5  # P/E ratio (fairly valued)
        features[1] = 0.15  # Earnings growth (moderate)

        labeled = labeler._label_fundamentals(features)

        assert labeled['dimensions'] == DIMS.fincoll_fundamentals
        assert labeled['data']['valuation']['pe_ratio'] == 15.5
        assert labeled['data']['valuation']['pe_interpretation'] == "fairly_valued"
        assert labeled['data']['growth']['earnings_growth'] == 0.15
        assert labeled['data']['growth']['growth_interpretation'] == "moderate_growth"

    def test_label_senvec(self):
        """Test SenVec sentiment labeling"""
        labeler = FeatureLabeler()

        features = np.zeros(DIMS.fincoll_senvec)
        features[0] = 0.7  # Social sentiment (very bullish)

        labeled = labeler._label_senvec(features)

        assert labeled['dimensions'] == DIMS.fincoll_senvec
        assert labeled['data']['social_sentiment'] == 0.7
        assert labeled['data']['social_interpretation'] == "very_bullish_social"

    def test_label_with_zero_features(self):
        """Test labeling handles zero features gracefully"""
        labeler = FeatureLabeler()
        features = np.zeros(DIMS.fincoll_total)

        labeled = labeler.label(features)

        # Should not crash, should return default interpretations
        assert isinstance(labeled, dict)
        assert 'technical_indicators' in labeled

    def test_label_all_components_have_dimensions(self):
        """Test all labeled components report their dimensions"""
        labeler = FeatureLabeler()
        features = np.random.randn(DIMS.fincoll_total)

        labeled = labeler.label(features)

        # All components should have 'dimensions' field
        for component_name, component_data in labeled.items():
            assert 'dimensions' in component_data
            assert isinstance(component_data['dimensions'], int)
            assert component_data['dimensions'] > 0

    def test_label_all_components_have_data(self):
        """Test all labeled components have data dict"""
        labeler = FeatureLabeler()
        features = np.random.randn(DIMS.fincoll_total)

        labeled = labeler.label(features)

        for component_name, component_data in labeled.items():
            assert 'data' in component_data
            assert isinstance(component_data['data'], dict)

    def test_label_all_components_have_summary(self):
        """Test all labeled components have summary string"""
        labeler = FeatureLabeler()
        features = np.random.randn(DIMS.fincoll_total)

        labeled = labeler.label(features)

        for component_name, component_data in labeled.items():
            assert 'summary' in component_data
            assert isinstance(component_data['summary'], str)

    def test_label_options_put_call(self):
        """Test options labeling with put/call ratio"""
        labeler = FeatureLabeler()

        features = np.zeros(DIMS.fincoll_options)
        features[0] = 1.2  # Put/call ratio (bearish)

        labeled = labeler._label_options(features)

        assert labeled['data']['put_call_ratio'] == 1.2
        assert labeled['data']['put_call_interpretation'] == "bearish_sentiment"

    def test_label_futures_vix(self):
        """Test futures labeling with VIX"""
        labeler = FeatureLabeler()

        features = np.zeros(DIMS.fincoll_futures)
        features[0] = 25.0  # VIX (elevated fear)

        labeled = labeler._label_futures(features)

        assert labeled['data']['vix'] == 25.0
        assert labeled['data']['vix_interpretation'] == "elevated_fear"

    def test_label_advanced_risk_sharpe(self):
        """Test advanced risk labeling with Sharpe ratio"""
        labeler = FeatureLabeler()

        features = np.zeros(DIMS.fincoll_advanced_risk)
        features[0] = 1.5  # Sharpe ratio (good)
        features[1] = 0.08  # Max drawdown (moderate)

        labeled = labeler._label_advanced_risk(features)

        assert labeled['data']['sharpe_ratio'] == 1.5
        assert labeled['data']['sharpe_interpretation'] == "good_risk_adjusted"
        assert labeled['data']['max_drawdown'] == 0.08
        assert labeled['data']['drawdown_interpretation'] == "moderate_drawdown"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_array(self):
        """Test labeling rejects empty array"""
        labeler = FeatureLabeler()

        with pytest.raises(ValueError):
            labeler.label(np.array([]))

    def test_nan_values(self):
        """Test labeling handles NaN values"""
        labeler = FeatureLabeler()
        features = np.full(DIMS.fincoll_total, np.nan)

        # Should not crash (may have NaN in output)
        labeled = labeler.label(features)
        assert isinstance(labeled, dict)

    def test_inf_values(self):
        """Test labeling handles inf values"""
        labeler = FeatureLabeler()
        features = np.full(DIMS.fincoll_total, np.inf)

        # Should not crash
        labeled = labeler.label(features)
        assert isinstance(labeled, dict)

    def test_negative_values(self):
        """Test labeling handles negative values"""
        labeler = FeatureLabeler()
        features = np.full(DIMS.fincoll_total, -1.0)

        # Should not crash
        labeled = labeler.label(features)
        assert isinstance(labeled, dict)
