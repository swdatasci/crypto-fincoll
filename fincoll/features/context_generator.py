#!/usr/bin/env python3
"""
Context Generator - Create LLM-friendly summaries from labeled features

Generates human-readable context strings that help LLM-based RL agents
make informed trading decisions efficiently.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ContextGenerator:
    """
    Generates LLM-friendly context summaries from labeled features

    Output includes:
    - Summary: Concise 2-3 sentence overview
    - Key signals: Actionable trading signals
    - Risk factors: Important warnings
    - Best opportunity: Recommended trade direction
    """

    def __init__(self):
        """Initialize context generator"""
        logger.info("ContextGenerator initialized")

    def generate(
        self, symbol: str, labeled_features: Dict[str, Any], predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate LLM-friendly context from labeled features and predictions

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            labeled_features: Output from FeatureLabeler.label()
            predictions: Velocity predictions from FinVec

        Returns:
            Dict with context for LLM agents:
            {
                "summary": str,              # 2-3 sentence overview
                "key_signals": List[str],    # Actionable signals
                "risk_factors": List[str],   # Warnings
                "sentiment": str,            # Overall sentiment
                "recommendation": str,       # Trade suggestion
            }
        """
        logger.debug(f"Generating context for {symbol}")

        # Extract key information from labeled features
        technical = labeled_features.get("technical_indicators", {}) or {}
        fundamentals = labeled_features.get("fundamentals", {}) or {}
        sentiment_data = labeled_features.get("senvec", {}) or {}
        options_data = labeled_features.get("options", {}) or {}
        risk_data = labeled_features.get("advanced_risk", {}) or {}

        # Extract best prediction
        best_opportunity = (
            predictions.get("best_opportunity", {}) if predictions else {}
        )

        # Generate components
        summary = self._generate_summary(
            symbol, technical, fundamentals, sentiment_data, best_opportunity
        )
        key_signals = self._extract_key_signals(
            technical, options_data, sentiment_data, predictions
        )
        risk_factors = self._extract_risk_factors(technical, risk_data, fundamentals)
        overall_sentiment = self._determine_sentiment(
            technical, sentiment_data, predictions
        )
        recommendation = self._generate_recommendation(
            best_opportunity, overall_sentiment, key_signals, risk_factors
        )

        context = {
            "summary": summary,
            "key_signals": key_signals,
            "risk_factors": risk_factors,
            "sentiment": overall_sentiment,
            "recommendation": recommendation,
        }

        logger.debug(
            f"Context generated for {symbol}: {len(key_signals)} signals, {len(risk_factors)} risks"
        )
        return context

    def _generate_summary(
        self,
        symbol: str,
        technical: Dict,
        fundamentals: Dict,
        sentiment: Dict,
        best_opportunity: Dict,
    ) -> str:
        """
        Generate 2-3 sentence summary

        Format: "[SYMBOL] [sentiment]. [technical indicators]. Best: [prediction]."
        """
        tech_data = technical.get("data", {})
        momentum = tech_data.get("momentum") or {}
        fund_data = fundamentals.get("data", {})
        sent_data = sentiment.get("data", {})

        # Sentence 1: Symbol and momentum
        rsi = momentum.get("rsi_14", 50)
        rsi_interp = momentum.get("rsi_interpretation", "neutral")
        macd_interp = momentum.get("macd_interpretation", "choppy")

        sentiment_str = "bullish" if rsi > 60 else "bearish" if rsi < 40 else "neutral"
        sent1 = f"{symbol} {sentiment_str}, RSI {rsi:.1f} ({rsi_interp}), MACD {macd_interp}."

        # Sentence 2: Key indicators
        pe_ratio = fund_data.get("valuation", {}).get("pe_ratio", 0)
        social_sent = sent_data.get("social_sentiment", 0)

        indicators = []
        if pe_ratio > 0:
            indicators.append(f"P/E {pe_ratio:.1f}")
        if abs(social_sent) > 0.3:
            sent_dir = "positive" if social_sent > 0 else "negative"
            indicators.append(f"{sent_dir} social sentiment")

        sent2 = (
            ", ".join(indicators) + "." if indicators else "Limited fundamental data."
        )

        # Sentence 3: Best opportunity
        if best_opportunity:
            timeframe = best_opportunity.get("timeframe", "unknown")
            direction = best_opportunity.get("direction", "NEUTRAL")
            velocity = best_opportunity.get("velocity", 0)
            confidence = best_opportunity.get("confidence", 0)

            sent3 = f"Best: {timeframe} {direction} {abs(velocity) * 100:.1f}%, {confidence * 100:.0f}% conf."
        else:
            sent3 = "No clear prediction."

        return f"{sent1} {sent2} {sent3}"

    def _extract_key_signals(
        self, technical: Dict, options: Dict, sentiment: Dict, predictions: Dict
    ) -> List[str]:
        """
        Extract actionable trading signals

        Returns list of signals with emoji prefixes:
        - ✅ Bullish signals
        - ❌ Bearish signals
        - ⚠️ Warning signals
        """
        signals = []

        tech_data = technical.get("data", {})
        momentum = tech_data.get("momentum") or {}
        trend = tech_data.get("trend", {})

        # RSI signals
        rsi_interp = momentum.get("rsi_interpretation", "neutral")
        if rsi_interp == "oversold":
            signals.append("✅ RSI oversold (potential bounce)")
        elif rsi_interp == "overbought":
            signals.append("❌ RSI overbought (potential pullback)")

        # MACD signals
        macd_interp = momentum.get("macd_interpretation", "choppy")
        if macd_interp == "bullish_crossover":
            signals.append("✅ MACD bullish crossover")
        elif macd_interp == "bearish_crossover":
            signals.append("❌ MACD bearish crossover")
        elif macd_interp in ["bullish_weakening", "bearish_weakening"]:
            signals.append("⚠️ MACD weakening")

        # Trend signals
        adx = trend.get("adx", 0)
        adx_interp = trend.get("adx_interpretation", "weak_trend")
        if adx_interp in ["strong_trend", "very_strong_trend"]:
            signals.append(f"✅ Strong trend (ADX {adx:.0f})")
        elif adx_interp == "weak_trend":
            signals.append("⚠️ Weak trend (choppy)")

        # Options signals
        opt_data = options.get("data", {})
        pc_ratio = opt_data.get("put_call_ratio", 1.0)
        pc_interp = opt_data.get("put_call_interpretation", "neutral_sentiment")
        if pc_interp == "bullish_sentiment":
            signals.append(f"✅ Bullish options flow (P/C {pc_ratio:.2f})")
        elif pc_interp in ["bearish_sentiment", "extreme_bearish_sentiment"]:
            signals.append(f"❌ Bearish options flow (P/C {pc_ratio:.2f})")

        # Sentiment signals
        sent_data = sentiment.get("data", {})
        social_sent = sent_data.get("social_sentiment", 0)
        social_interp = sent_data.get("social_interpretation", "neutral_social")
        if social_interp in ["very_bullish_social", "bullish_social"]:
            signals.append("✅ Bullish social sentiment")
        elif social_interp in ["very_bearish_social", "bearish_social"]:
            signals.append("❌ Bearish social sentiment")

        # Prediction signals
        best_opp = predictions.get("best_opportunity", {})
        if best_opp:
            direction = best_opp.get("direction", "NEUTRAL")
            confidence = best_opp.get("confidence", 0)
            if confidence > 0.7:
                emoji = (
                    "✅"
                    if direction == "LONG"
                    else "❌"
                    if direction == "SHORT"
                    else "⚠️"
                )
                signals.append(
                    f"{emoji} High confidence {direction.lower()} prediction"
                )

        return signals if signals else ["⚠️ No clear signals"]

    def _extract_risk_factors(
        self, technical: Dict, risk: Dict, fundamentals: Dict
    ) -> List[str]:
        """
        Extract risk factors and warnings

        Returns list of warnings with ⚠️ prefix
        """
        risks = []

        # Volatility risks
        tech_data = technical.get("data", {})
        volatility_data = tech_data.get("volatility", {})
        bb_position = volatility_data.get("bollinger_position", 0.5)
        bb_interp = volatility_data.get("bollinger_interpretation", "mid_range")

        if bb_interp in ["near_upper_band", "near_lower_band"]:
            risks.append(f"⚠️ Near Bollinger Band ({bb_interp.replace('_', ' ')})")

        # Risk metrics
        risk_data = risk.get("data", {})
        sharpe = risk_data.get("sharpe_ratio", 0)
        sharpe_interp = risk_data.get("sharpe_interpretation", "poor_risk_adjusted")
        drawdown = risk_data.get("max_drawdown", 0)
        drawdown_interp = risk_data.get("drawdown_interpretation", "minimal_drawdown")

        if sharpe_interp in ["poor_risk_adjusted", "negative_risk_adjusted"]:
            risks.append(f"⚠️ Poor risk-adjusted returns (Sharpe {sharpe:.2f})")

        if drawdown_interp in ["significant_drawdown", "severe_drawdown"]:
            risks.append(f"⚠️ High drawdown risk ({drawdown * 100:.1f}%)")

        # Fundamental risks
        fund_data = fundamentals.get("data", {})
        valuation = fund_data.get("valuation", {})
        pe_interp = valuation.get("pe_interpretation", "fairly_valued")

        if pe_interp in ["highly_overvalued", "overvalued"]:
            risks.append("⚠️ Overvalued (high P/E)")
        elif pe_interp == "negative_earnings":
            risks.append("⚠️ Negative earnings")

        return risks if risks else ["✅ No significant risks identified"]

    def _determine_sentiment(
        self, technical: Dict, sentiment: Dict, predictions: Dict
    ) -> str:
        """
        Determine overall sentiment: bullish, bearish, or neutral

        Considers technical, sentiment, and prediction data
        """
        bullish_score = 0
        bearish_score = 0

        # Technical sentiment
        tech_data = technical.get("data", {})
        momentum = tech_data.get("momentum") or {}
        rsi_interp = momentum.get("rsi_interpretation", "neutral")
        macd_interp = momentum.get("macd_interpretation", "choppy")

        if rsi_interp in ["oversold", "oversold_warning"]:
            bullish_score += 1
        elif rsi_interp in ["overbought", "overbought_warning"]:
            bearish_score += 1

        if "bullish" in macd_interp:
            bullish_score += 1
        elif "bearish" in macd_interp:
            bearish_score += 1

        # Sentiment data
        sent_data = sentiment.get("data", {})
        social_interp = sent_data.get("social_interpretation", "neutral_social")

        if "bullish" in social_interp:
            bullish_score += 1
        elif "bearish" in social_interp:
            bearish_score += 1

        # Prediction sentiment
        best_opp = predictions.get("best_opportunity", {})
        if best_opp:
            direction = best_opp.get("direction", "NEUTRAL")
            if direction == "LONG":
                bullish_score += 2  # Predictions weighted higher
            elif direction == "SHORT":
                bearish_score += 2

        # Determine overall
        if bullish_score > bearish_score + 1:
            return "bullish"
        elif bearish_score > bullish_score + 1:
            return "bearish"
        else:
            return "neutral"

    def _generate_recommendation(
        self,
        best_opportunity: Dict,
        sentiment: str,
        key_signals: List[str],
        risk_factors: List[str],
    ) -> str:
        """
        Generate trading recommendation

        Format: "[ACTION]: [timeframe] [direction] [velocity]%. [reasoning]."
        """
        if not best_opportunity:
            return "HOLD: No clear prediction available."

        timeframe = best_opportunity.get("timeframe", "unknown")
        direction = best_opportunity.get("direction", "NEUTRAL")
        velocity = best_opportunity.get("velocity", 0)
        confidence = best_opportunity.get("confidence", 0)

        # Count bullish vs bearish signals
        bullish_signals = len([s for s in key_signals if "✅" in s])
        bearish_signals = len([s for s in key_signals if "❌" in s])
        warning_signals = len([s for s in key_signals if "⚠️" in s])

        # Count risks
        high_risks = len([r for r in risk_factors if "⚠️" in r])

        # Determine action
        if confidence < 0.5:
            action = "HOLD"
            reason = "Low confidence prediction"
        elif high_risks > 2:
            action = "CAUTION"
            reason = "Multiple risk factors present"
        elif direction == "LONG" and bullish_signals > bearish_signals:
            action = "CONSIDER LONG"
            reason = f"Bullish signals align with prediction"
        elif direction == "SHORT" and bearish_signals > bullish_signals:
            action = "CONSIDER SHORT"
            reason = f"Bearish signals align with prediction"
        elif direction == "LONG":
            action = "MIXED LONG"
            reason = "Prediction bullish but signals mixed"
        elif direction == "SHORT":
            action = "MIXED SHORT"
            reason = "Prediction bearish but signals mixed"
        else:
            action = "HOLD"
            reason = "No clear direction"

        return f"{action}: {timeframe} {direction} {abs(velocity) * 100:.1f}%, {confidence * 100:.0f}% conf. {reason}."
