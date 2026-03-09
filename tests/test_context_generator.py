#!/usr/bin/env python3
"""
Unit tests for ContextGenerator

Tests context generation, signal extraction, and recommendation logic
"""

import pytest
from fincoll.features.context_generator import ContextGenerator


class TestContextGenerator:
    """Test ContextGenerator class"""

    def test_initialization(self):
        """Test generator initializes correctly"""
        generator = ContextGenerator()
        assert generator is not None

    def test_generate_with_bullish_setup(self):
        """Test context generation with bullish technical setup"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {
                "data": {
                    "momentum": {
                        "rsi_14": 35.0,
                        "rsi_interpretation": "oversold_warning",
                        "macd_histogram": 0.5,
                        "macd_interpretation": "bullish_crossover",
                    },
                    "trend": {
                        "adx": 35.0,
                        "adx_interpretation": "strong_trend",
                    },
                    "volatility": {
                        "bollinger_position": 0.3,
                        "bollinger_interpretation": "lower_half",
                    },
                }
            },
            "fundamentals": {
                "data": {
                    "valuation": {
                        "pe_ratio": 15.5,
                        "pe_interpretation": "fairly_valued",
                    }
                }
            },
            "senvec": {
                "data": {
                    "social_sentiment": 0.6,
                    "social_interpretation": "bullish_social",
                }
            },
            "options": {
                "data": {
                    "put_call_ratio": 0.65,
                    "put_call_interpretation": "bullish_sentiment",
                }
            },
            "advanced_risk": {
                "data": {
                    "sharpe_ratio": 1.5,
                    "sharpe_interpretation": "good_risk_adjusted",
                    "max_drawdown": 0.03,
                    "drawdown_interpretation": "minor_drawdown",
                }
            },
        }

        predictions = {
            "best_opportunity": {
                "timeframe": "5min",
                "direction": "LONG",
                "velocity": 0.008,
                "confidence": 0.82,
            }
        }

        context = generator.generate("AAPL", labeled_features, predictions)

        assert isinstance(context, dict)
        assert "summary" in context
        assert "key_signals" in context
        assert "risk_factors" in context
        assert "sentiment" in context
        assert "recommendation" in context

        # Check sentiment is bullish
        assert context["sentiment"] in ["bullish", "neutral"]  # Should be bullish

        # Check has bullish signals
        bullish_signals = [s for s in context["key_signals"] if "✅" in s]
        assert len(bullish_signals) > 0

        # Check summary mentions AAPL
        assert "AAPL" in context["summary"]

    def test_generate_with_bearish_setup(self):
        """Test context generation with bearish technical setup"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {
                "data": {
                    "momentum": {
                        "rsi_14": 75.0,
                        "rsi_interpretation": "overbought",
                        "macd_histogram": -0.5,
                        "macd_interpretation": "bearish_crossover",
                    },
                    "trend": {
                        "adx": 40.0,
                        "adx_interpretation": "strong_trend",
                    },
                    "volatility": {
                        "bollinger_position": 0.9,
                        "bollinger_interpretation": "near_upper_band",
                    },
                }
            },
            "fundamentals": {"data": {}},
            "senvec": {
                "data": {
                    "social_sentiment": -0.6,
                    "social_interpretation": "bearish_social",
                }
            },
            "options": {
                "data": {
                    "put_call_ratio": 1.3,
                    "put_call_interpretation": "bearish_sentiment",
                }
            },
            "advanced_risk": {"data": {}},
        }

        predictions = {
            "best_opportunity": {
                "timeframe": "15min",
                "direction": "SHORT",
                "velocity": -0.006,
                "confidence": 0.75,
            }
        }

        context = generator.generate("TSLA", labeled_features, predictions)

        assert context["sentiment"] in ["bearish", "neutral"]

        # Check has bearish signals
        bearish_signals = [s for s in context["key_signals"] if "❌" in s]
        assert len(bearish_signals) > 0

    def test_generate_with_minimal_data(self):
        """Test context generation with minimal feature data"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {"data": {}},
            "fundamentals": {"data": {}},
            "senvec": {"data": {}},
            "options": {"data": {}},
            "advanced_risk": {"data": {}},
        }

        predictions = {}

        context = generator.generate("XYZ", labeled_features, predictions)

        assert isinstance(context, dict)
        assert context["sentiment"] == "neutral"
        assert "No clear prediction" in context["summary"]
        assert any("No clear signals" in s or "⚠️" in s for s in context["key_signals"])

    def test_extract_key_signals_rsi_oversold(self):
        """Test RSI oversold signal extraction"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {
                    "rsi_interpretation": "oversold",
                }
            }
        }

        signals = generator._extract_key_signals(technical, {}, {}, {})

        assert any("RSI oversold" in s for s in signals)
        assert any("✅" in s for s in signals)

    def test_extract_key_signals_macd_bullish(self):
        """Test MACD bullish crossover signal"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {
                    "macd_interpretation": "bullish_crossover",
                },
                "trend": {},
            }
        }

        signals = generator._extract_key_signals(technical, {}, {}, {})

        assert any("MACD bullish crossover" in s for s in signals)

    def test_extract_risk_factors_high_drawdown(self):
        """Test high drawdown risk extraction"""
        generator = ContextGenerator()

        risk = {
            "data": {
                "max_drawdown": 0.15,
                "drawdown_interpretation": "significant_drawdown",
            }
        }

        risks = generator._extract_risk_factors({}, risk, {})

        assert any("drawdown" in r.lower() for r in risks)
        assert any("⚠️" in r for r in risks)

    def test_extract_risk_factors_overvalued(self):
        """Test overvaluation risk"""
        generator = ContextGenerator()

        fundamentals = {
            "data": {
                "valuation": {
                    "pe_interpretation": "highly_overvalued",
                }
            }
        }

        risks = generator._extract_risk_factors({}, {}, fundamentals)

        assert any("Overvalued" in r or "P/E" in r for r in risks)

    def test_determine_sentiment_bullish(self):
        """Test bullish sentiment determination"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {
                    "rsi_interpretation": "oversold",
                    "macd_interpretation": "bullish_crossover",
                }
            }
        }

        sentiment_data = {
            "data": {
                "social_interpretation": "bullish_social",
            }
        }

        predictions = {
            "best_opportunity": {
                "direction": "LONG",
            }
        }

        sentiment = generator._determine_sentiment(technical, sentiment_data, predictions)

        assert sentiment == "bullish"

    def test_determine_sentiment_bearish(self):
        """Test bearish sentiment determination"""
        generator = ContextGenerator()

        technical = {
            "data": {
                "momentum": {
                    "rsi_interpretation": "overbought",
                    "macd_interpretation": "bearish_crossover",
                }
            }
        }

        sentiment_data = {
            "data": {
                "social_interpretation": "bearish_social",
            }
        }

        predictions = {
            "best_opportunity": {
                "direction": "SHORT",
            }
        }

        sentiment = generator._determine_sentiment(technical, sentiment_data, predictions)

        assert sentiment == "bearish"

    def test_determine_sentiment_neutral(self):
        """Test neutral sentiment determination"""
        generator = ContextGenerator()

        technical = {"data": {"momentum": {}}}
        sentiment_data = {"data": {}}
        predictions = {}

        sentiment = generator._determine_sentiment(technical, sentiment_data, predictions)

        assert sentiment == "neutral"

    def test_generate_recommendation_high_confidence(self):
        """Test recommendation with high confidence prediction"""
        generator = ContextGenerator()

        best_opportunity = {
            "timeframe": "5min",
            "direction": "LONG",
            "velocity": 0.010,
            "confidence": 0.85,
        }

        key_signals = [
            "✅ RSI oversold",
            "✅ MACD bullish crossover",
            "✅ Bullish social sentiment",
        ]

        risk_factors = ["✅ No significant risks identified"]

        recommendation = generator._generate_recommendation(
            best_opportunity, "bullish", key_signals, risk_factors
        )

        assert "CONSIDER LONG" in recommendation or "LONG" in recommendation
        assert "5min" in recommendation
        assert "85%" in recommendation

    def test_generate_recommendation_low_confidence(self):
        """Test recommendation with low confidence prediction"""
        generator = ContextGenerator()

        best_opportunity = {
            "timeframe": "15min",
            "direction": "SHORT",
            "velocity": -0.005,
            "confidence": 0.35,
        }

        recommendation = generator._generate_recommendation(
            best_opportunity, "bearish", [], []
        )

        assert "HOLD" in recommendation
        assert "Low confidence" in recommendation

    def test_generate_recommendation_high_risk(self):
        """Test recommendation with multiple risk factors"""
        generator = ContextGenerator()

        best_opportunity = {
            "timeframe": "1hour",
            "direction": "LONG",
            "velocity": 0.008,
            "confidence": 0.75,
        }

        risk_factors = [
            "⚠️ High drawdown risk",
            "⚠️ Poor risk-adjusted returns",
            "⚠️ Overvalued (high P/E)",
        ]

        recommendation = generator._generate_recommendation(
            best_opportunity, "bullish", [], risk_factors
        )

        assert "CAUTION" in recommendation
        assert "risk" in recommendation.lower()

    def test_generate_recommendation_no_prediction(self):
        """Test recommendation with no prediction"""
        generator = ContextGenerator()

        recommendation = generator._generate_recommendation({}, "neutral", [], [])

        assert "HOLD" in recommendation
        assert "No clear prediction" in recommendation


class TestContextGeneratorEdgeCases:
    """Test edge cases in context generation"""

    def test_empty_labeled_features(self):
        """Test with completely empty labeled features"""
        generator = ContextGenerator()

        labeled_features = {}
        predictions = {}

        context = generator.generate("TEST", labeled_features, predictions)

        assert isinstance(context, dict)
        assert "summary" in context
        assert "TEST" in context["summary"]

    def test_missing_predictions(self):
        """Test with missing prediction data"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {
                "data": {
                    "momentum": {
                        "rsi_14": 50.0,
                        "rsi_interpretation": "neutral",
                    }
                }
            },
        }

        predictions = {}

        context = generator.generate("ABC", labeled_features, predictions)

        assert "No clear prediction" in context["summary"]
        assert "HOLD" in context["recommendation"]

    def test_all_none_values(self):
        """Test with None values in features"""
        generator = ContextGenerator()

        labeled_features = {
            "technical_indicators": {"data": {"momentum": None}},
        }

        predictions = None

        # Should not crash
        context = generator.generate("NONE", labeled_features, predictions or {})

        assert isinstance(context, dict)
