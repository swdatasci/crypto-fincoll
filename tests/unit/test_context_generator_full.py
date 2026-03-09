#!/usr/bin/env python3
"""
Complete tests for ContextGenerator edge cases

Tests edge cases not covered in existing tests to reach near-100% coverage
"""

import pytest
from fincoll.features.context_generator import ContextGenerator


class TestContextGeneratorEdgeCases:
    """Test edge cases not covered in existing tests"""

    def test_generator_handles_none_technical_data(self):
        """Generator should handle None technical data gracefully"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": None,
            "fundamentals": {"data": {}},
            "senvec": {"data": {}},
            "options": {"data": {}},
            "advanced_risk": {"data": {}},
        }
        predictions = {}

        result = generator.generate("TEST", labeled_features, predictions)

        assert result is not None
        assert isinstance(result, dict)
        assert "summary" in result
        assert "sentiment" in result

    def test_generator_handles_missing_data_keys(self):
        """Generator should handle missing 'data' keys in labeled features"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {},  # Missing 'data' key
            "fundamentals": {},
            "senvec": {},
            "options": {},
            "advanced_risk": {},
        }
        predictions = {}

        result = generator.generate("TEST", labeled_features, predictions)

        assert result is not None
        assert result["sentiment"] == "neutral"

    def test_generator_handles_none_momentum(self):
        """Generator should handle None momentum in technical data"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {
                "data": {
                    "momentum": None,  # Explicitly None
                    "trend": {},
                }
            },
            "fundamentals": {"data": {}},
            "senvec": {"data": {}},
            "options": {"data": {}},
            "advanced_risk": {"data": {}},
        }
        predictions = {}

        result = generator.generate("TEST", labeled_features, predictions)

        assert result is not None
        assert "TEST" in result["summary"]

    def test_extract_signals_with_macd_weakening(self):
        """Test MACD weakening signals"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {
                    "macd_interpretation": "bullish_weakening",
                },
                "trend": {},
            }
        }

        signals = generator._extract_key_signals(technical, {}, {}, {})

        assert any("MACD weakening" in s for s in signals)
        assert any("⚠️" in s for s in signals)

    def test_extract_signals_with_weak_trend(self):
        """Test weak trend signal extraction"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {},
                "trend": {
                    "adx": 15.0,
                    "adx_interpretation": "weak_trend",
                },
            }
        }

        signals = generator._extract_key_signals(technical, {}, {}, {})

        assert any("Weak trend" in s or "choppy" in s for s in signals)

    def test_extract_signals_with_extreme_bearish_options(self):
        """Test extreme bearish options flow signal"""
        generator = ContextGenerator()

        options = {
            "data": {
                "put_call_ratio": 1.8,
                "put_call_interpretation": "extreme_bearish_sentiment",
            }
        }

        signals = generator._extract_key_signals({}, options, {}, {})

        assert any("Bearish options flow" in s for s in signals)
        assert any("❌" in s for s in signals)

    def test_extract_signals_with_very_bearish_social(self):
        """Test very bearish social sentiment signal"""
        generator = ContextGenerator()

        sentiment = {
            "data": {
                "social_sentiment": -0.8,
                "social_interpretation": "very_bearish_social",
            }
        }

        signals = generator._extract_key_signals({}, {}, sentiment, {})

        assert any("Bearish social sentiment" in s for s in signals)
        assert any("❌" in s for s in signals)

    def test_extract_signals_neutral_prediction(self):
        """Test signal extraction with NEUTRAL direction"""
        generator = ContextGenerator()

        predictions = {
            "best_opportunity": {
                "direction": "NEUTRAL",
                "confidence": 0.75,
            }
        }

        signals = generator._extract_key_signals({}, {}, {}, predictions)

        # Should still generate signals even with NEUTRAL
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_extract_risk_factors_near_bollinger_band(self):
        """Test Bollinger Band risk extraction"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "volatility": {
                    "bollinger_position": 0.95,
                    "bollinger_interpretation": "near_upper_band",
                }
            }
        }

        risks = generator._extract_risk_factors(technical, {}, {})

        assert any("Bollinger Band" in r for r in risks)
        assert any("⚠️" in r for r in risks)

    def test_extract_risk_factors_poor_sharpe(self):
        """Test poor Sharpe ratio risk"""
        generator = ContextGenerator()

        risk = {
            "data": {
                "sharpe_ratio": -0.5,
                "sharpe_interpretation": "negative_risk_adjusted",
            }
        }

        risks = generator._extract_risk_factors({}, risk, {})

        assert any("Poor risk-adjusted returns" in r or "Sharpe" in r for r in risks)

    def test_extract_risk_factors_severe_drawdown(self):
        """Test severe drawdown risk"""
        generator = ContextGenerator()

        risk = {
            "data": {
                "max_drawdown": 0.25,
                "drawdown_interpretation": "severe_drawdown",
            }
        }

        risks = generator._extract_risk_factors({}, risk, {})

        assert any("drawdown" in r.lower() for r in risks)
        assert any("25.0%" in r for r in risks)

    def test_extract_risk_factors_negative_earnings(self):
        """Test negative earnings risk"""
        generator = ContextGenerator()

        fundamentals = {
            "data": {
                "valuation": {
                    "pe_interpretation": "negative_earnings",
                }
            }
        }

        risks = generator._extract_risk_factors({}, {}, fundamentals)

        assert any("Negative earnings" in r for r in risks)

    def test_extract_risk_factors_overvalued(self):
        """Test overvalued risk (not highly overvalued)"""
        generator = ContextGenerator()

        fundamentals = {
            "data": {
                "valuation": {
                    "pe_interpretation": "overvalued",
                }
            }
        }

        risks = generator._extract_risk_factors({}, {}, fundamentals)

        assert any("Overvalued" in r for r in risks)

    def test_determine_sentiment_with_overbought_warning(self):
        """Test sentiment with overbought_warning interpretation"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {
                    "rsi_interpretation": "overbought_warning",
                }
            }
        }

        sentiment = generator._determine_sentiment(technical, {}, {})

        # overbought_warning should contribute to bearish score
        assert sentiment in ["bearish", "neutral"]

    def test_determine_sentiment_with_oversold_warning(self):
        """Test sentiment with oversold_warning interpretation"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {
                    "rsi_interpretation": "oversold_warning",
                }
            }
        }

        sentiment = generator._determine_sentiment(technical, {}, {})

        # oversold_warning should contribute to bullish score
        assert sentiment in ["bullish", "neutral"]

    def test_generate_recommendation_mixed_long(self):
        """Test MIXED LONG recommendation when signals don't align"""
        generator = ContextGenerator()

        best_opportunity = {
            "timeframe": "5min",
            "direction": "LONG",
            "velocity": 0.008,
            "confidence": 0.70,
        }

        key_signals = [
            "❌ Bearish signal 1",
            "❌ Bearish signal 2",
        ]

        risk_factors = ["✅ No significant risks identified"]

        recommendation = generator._generate_recommendation(
            best_opportunity, "neutral", key_signals, risk_factors
        )

        assert "MIXED LONG" in recommendation
        assert "signals mixed" in recommendation

    def test_generate_recommendation_mixed_short(self):
        """Test MIXED SHORT recommendation when signals don't align"""
        generator = ContextGenerator()

        best_opportunity = {
            "timeframe": "15min",
            "direction": "SHORT",
            "velocity": -0.006,
            "confidence": 0.65,
        }

        key_signals = [
            "✅ Bullish signal 1",
            "✅ Bullish signal 2",
        ]

        risk_factors = ["✅ No significant risks identified"]

        recommendation = generator._generate_recommendation(
            best_opportunity, "neutral", key_signals, risk_factors
        )

        assert "MIXED SHORT" in recommendation
        assert "signals mixed" in recommendation

    def test_generate_recommendation_consider_short(self):
        """Test CONSIDER SHORT when signals align"""
        generator = ContextGenerator()

        best_opportunity = {
            "timeframe": "1hour",
            "direction": "SHORT",
            "velocity": -0.007,
            "confidence": 0.75,
        }

        key_signals = [
            "❌ Bearish signal 1",
            "❌ Bearish signal 2",
            "✅ Bullish signal",
        ]

        risk_factors = ["✅ No significant risks identified"]

        recommendation = generator._generate_recommendation(
            best_opportunity, "bearish", key_signals, risk_factors
        )

        assert "CONSIDER SHORT" in recommendation
        assert "Bearish signals align" in recommendation

    def test_generate_summary_with_zero_pe(self):
        """Test summary generation with zero P/E ratio"""
        generator = ContextGenerator()

        summary = generator._generate_summary(
            "TEST",
            {"data": {"momentum": {}}},
            {"data": {"valuation": {"pe_ratio": 0}}},
            {"data": {}},
            {},
        )

        assert "TEST" in summary
        # Zero P/E should result in "Limited fundamental data"
        assert "Limited fundamental data" in summary

    def test_generate_summary_with_negative_pe(self):
        """Test summary generation with negative P/E ratio"""
        generator = ContextGenerator()

        summary = generator._generate_summary(
            "TEST",
            {"data": {"momentum": {}}},
            {"data": {"valuation": {"pe_ratio": -5.0}}},
            {"data": {}},
            {},
        )

        assert "TEST" in summary
        # Negative P/E should be filtered out (pe_ratio > 0 check)
        assert "Limited fundamental data" in summary

    def test_generate_summary_with_low_social_sentiment(self):
        """Test summary with social sentiment below threshold"""
        generator = ContextGenerator()

        summary = generator._generate_summary(
            "TEST",
            {"data": {"momentum": {}}},
            {"data": {}},
            {"data": {"social_sentiment": 0.2}},  # abs(0.2) < 0.3
            {},
        )

        assert "TEST" in summary
        # Should not include social sentiment (below 0.3 threshold)
        assert "sentiment" not in summary or "Limited fundamental data" in summary

    def test_full_integration_with_very_strong_trend(self):
        """Test full integration with very_strong_trend ADX"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {
                "data": {
                    "momentum": {},
                    "trend": {
                        "adx": 55.0,
                        "adx_interpretation": "very_strong_trend",
                    },
                }
            },
            "fundamentals": {"data": {}},
            "senvec": {"data": {}},
            "options": {"data": {}},
            "advanced_risk": {"data": {}},
        }

        predictions = {}

        context = generator.generate("STRONG", labeled_features, predictions)

        assert any("Strong trend" in s and "ADX" in s for s in context["key_signals"])
