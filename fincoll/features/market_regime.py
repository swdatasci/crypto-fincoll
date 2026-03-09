"""
Market Regime Classifier - Sentiment as Context Indicator

KEY INSIGHT: Sentiment tells us WHY prices are behaving unusually!
It provides fear/greed context for otherwise anomalous patterns.

User Quote: "provides us with a fear/greed-index, positivity, and volatility...
context for otherwise unusual behavior changes"

What this does:
- Combines sentiment into regime classification (fear, greed, uncertainty, neutral)
- Uses sentiment to explain anomalies:
  * "Price fell despite positive technicals → check sentiment (bad news?)"
  * "Price spiked → check sentiment (catalyst?)"
- Creates regime context that modifies how model interprets patterns

Agent Delta - Phase 1 FEATURE INTEGRATION stream
"""

import numpy as np
from typing import Dict
from enum import Enum
from dataclasses import dataclass


class MarketRegime(Enum):
    """Market regime classifications based on sentiment and volatility."""
    FEAR = 0           # Negative sentiment + high volatility = panic selling
    GREED = 1          # Positive sentiment + low volatility = euphoria
    UNCERTAINTY = 2    # Mixed sentiment + high volatility = confusion
    NEUTRAL = 3        # Moderate sentiment + moderate volatility = normal
    CAPITULATION = 4   # Extreme negative + extreme volatility = bottom?
    EUPHORIA = 5       # Extreme positive + low volatility = top?


@dataclass
class RegimeContext:
    """
    Market regime context derived from sentiment.

    This tells the model WHY prices might be behaving unusually,
    so it can adjust predictions accordingly.
    """

    # Primary regime
    regime: MarketRegime

    # Regime scores (0-1 for each regime)
    fear_score: float         # How much fear in the market?
    greed_score: float        # How much greed/euphoria?
    uncertainty_score: float  # How much confusion/mixed signals?

    # Sentiment features (raw inputs)
    sentiment_score: float      # -1 to +1
    sentiment_std: float        # Volatility of sentiment
    news_volume: int           # Number of articles
    sentiment_trend: float     # Change in sentiment

    # Context interpretation
    is_high_volatility: bool   # High sentiment volatility
    is_extreme_sentiment: bool # Extreme positive or negative
    is_news_heavy: bool        # Lots of news coverage

    def to_vector(self) -> np.ndarray:
        """
        Convert regime context to numerical vector for model input.

        Returns:
            Array of shape (14,) with:
            [regime_onehot (6), fear_score, greed_score, uncertainty_score,
             sentiment_score, sentiment_std, sentiment_trend,
             is_high_vol, is_extreme, is_news_heavy]
        """
        # One-hot encode regime (6 categories)
        regime_vec = np.zeros(6)
        regime_vec[self.regime.value] = 1.0

        # Continuous scores
        scores = np.array([
            self.fear_score,
            self.greed_score,
            self.uncertainty_score,
            self.sentiment_score,
            self.sentiment_std,
            self.sentiment_trend,
            float(self.is_high_volatility),
            float(self.is_extreme_sentiment),
            float(self.is_news_heavy)
        ])

        return np.concatenate([regime_vec, scores])

    def get_embedding_size(self) -> int:
        """Get the size of the context vector."""
        return 6 + 9  # = 15 features


class MarketRegimeClassifier:
    """
    Classifies market regime from sentiment features.

    This is the "WHY" component of our feature integration:
    - Technicals tell us WHAT is happening (price/volume patterns)
    - Fundamentals tell us WHO the company is (context)
    - Sentiment tells us WHY it's happening (market psychology)
    """

    def __init__(self,
                 fear_threshold: float = -0.3,
                 greed_threshold: float = 0.3,
                 high_vol_threshold: float = 0.5,
                 extreme_threshold: float = 0.6,
                 high_news_threshold: int = 10):
        """
        Initialize the market regime classifier.

        Args:
            fear_threshold: Sentiment score below this = fear
            greed_threshold: Sentiment score above this = greed
            high_vol_threshold: Sentiment std above this = high volatility
            extreme_threshold: Absolute sentiment above this = extreme
            high_news_threshold: News volume above this = news heavy
        """
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold
        self.high_vol_threshold = high_vol_threshold
        self.extreme_threshold = extreme_threshold
        self.high_news_threshold = high_news_threshold

    def classify(self, sentiment_features: Dict[str, float]) -> RegimeContext:
        """
        Classify market regime from sentiment features.

        Args:
            sentiment_features: Dictionary from NewsSentimentAnalyzer.analyze()
                {
                    'sentiment_score': float,    # -1 to +1
                    'sentiment_std': float,      # Volatility
                    'news_volume': int,          # Number of articles
                    'positive_ratio': float,     # % positive
                    'negative_ratio': float,     # % negative
                    'sentiment_trend': float     # Change over time
                }

        Returns:
            RegimeContext with regime classification and scores
        """
        # Extract features
        sentiment_score = sentiment_features.get('sentiment_score', 0.0)
        sentiment_std = sentiment_features.get('sentiment_std', 0.0)
        news_volume = sentiment_features.get('news_volume', 0)
        sentiment_trend = sentiment_features.get('sentiment_trend', 0.0)
        positive_ratio = sentiment_features.get('positive_ratio', 0.0)
        negative_ratio = sentiment_features.get('negative_ratio', 0.0)

        # Classify context flags
        is_high_volatility = sentiment_std > self.high_vol_threshold
        is_extreme_sentiment = abs(sentiment_score) > self.extreme_threshold
        is_news_heavy = news_volume > self.high_news_threshold

        # Calculate regime scores
        fear_score = self._calculate_fear_score(
            sentiment_score, sentiment_std, negative_ratio, sentiment_trend
        )
        greed_score = self._calculate_greed_score(
            sentiment_score, sentiment_std, positive_ratio, sentiment_trend
        )
        uncertainty_score = self._calculate_uncertainty_score(
            sentiment_std, positive_ratio, negative_ratio
        )

        # Classify primary regime
        regime = self._classify_regime(
            sentiment_score, sentiment_std, fear_score, greed_score, uncertainty_score
        )

        return RegimeContext(
            regime=regime,
            fear_score=fear_score,
            greed_score=greed_score,
            uncertainty_score=uncertainty_score,
            sentiment_score=sentiment_score,
            sentiment_std=sentiment_std,
            news_volume=news_volume,
            sentiment_trend=sentiment_trend,
            is_high_volatility=is_high_volatility,
            is_extreme_sentiment=is_extreme_sentiment,
            is_news_heavy=is_news_heavy
        )

    def _calculate_fear_score(self, sentiment_score: float,
                              sentiment_std: float,
                              negative_ratio: float,
                              sentiment_trend: float) -> float:
        """
        Calculate fear score (0-1).

        Fear = negative sentiment + high volatility + worsening trend

        Args:
            sentiment_score: Overall sentiment (-1 to +1)
            sentiment_std: Sentiment volatility
            negative_ratio: % of negative articles
            sentiment_trend: Sentiment change over time

        Returns:
            Fear score (0 = no fear, 1 = extreme fear)
        """
        # Negative sentiment component (invert to 0-1)
        negative_component = np.clip(-sentiment_score, 0, 1)

        # High volatility component (normalize)
        volatility_component = np.clip(sentiment_std / 1.0, 0, 1)

        # Negative ratio component
        negative_ratio_component = negative_ratio

        # Worsening trend component (negative trend = fear)
        trend_component = np.clip(-sentiment_trend, 0, 1)

        # Weighted combination
        fear_score = (
            0.4 * negative_component +
            0.3 * volatility_component +
            0.2 * negative_ratio_component +
            0.1 * trend_component
        )

        return float(fear_score)

    def _calculate_greed_score(self, sentiment_score: float,
                               sentiment_std: float,
                               positive_ratio: float,
                               sentiment_trend: float) -> float:
        """
        Calculate greed score (0-1).

        Greed = positive sentiment + low volatility + improving trend

        Args:
            sentiment_score: Overall sentiment (-1 to +1)
            sentiment_std: Sentiment volatility
            positive_ratio: % of positive articles
            sentiment_trend: Sentiment change over time

        Returns:
            Greed score (0 = no greed, 1 = extreme greed)
        """
        # Positive sentiment component
        positive_component = np.clip(sentiment_score, 0, 1)

        # Low volatility component (invert - greed = complacency)
        stability_component = np.clip(1.0 - sentiment_std / 1.0, 0, 1)

        # Positive ratio component
        positive_ratio_component = positive_ratio

        # Improving trend component
        trend_component = np.clip(sentiment_trend, 0, 1)

        # Weighted combination
        greed_score = (
            0.4 * positive_component +
            0.3 * stability_component +
            0.2 * positive_ratio_component +
            0.1 * trend_component
        )

        return float(greed_score)

    def _calculate_uncertainty_score(self, sentiment_std: float,
                                     positive_ratio: float,
                                     negative_ratio: float) -> float:
        """
        Calculate uncertainty score (0-1).

        Uncertainty = high volatility + mixed signals (both positive and negative)

        Args:
            sentiment_std: Sentiment volatility
            positive_ratio: % of positive articles
            negative_ratio: % of negative articles

        Returns:
            Uncertainty score (0 = clear, 1 = very uncertain)
        """
        # High volatility component
        volatility_component = np.clip(sentiment_std / 1.0, 0, 1)

        # Mixed signals component (both positive and negative present)
        # Highest when both ratios are around 0.4-0.5
        mixed_component = 2 * min(positive_ratio, negative_ratio)

        # Weighted combination
        uncertainty_score = 0.6 * volatility_component + 0.4 * mixed_component

        return float(uncertainty_score)

    def _classify_regime(self, sentiment_score: float,
                        sentiment_std: float,
                        fear_score: float,
                        greed_score: float,
                        uncertainty_score: float) -> MarketRegime:
        """
        Classify primary market regime.

        Args:
            sentiment_score: Overall sentiment
            sentiment_std: Sentiment volatility
            fear_score: Calculated fear score
            greed_score: Calculated greed score
            uncertainty_score: Calculated uncertainty score

        Returns:
            MarketRegime enum
        """
        # Check for extreme regimes first
        if sentiment_score < -0.7 and sentiment_std > 0.7:
            return MarketRegime.CAPITULATION  # Extreme fear + panic

        if sentiment_score > 0.7 and sentiment_std < 0.2:
            return MarketRegime.EUPHORIA  # Extreme greed + complacency

        # Check for dominant regime
        max_score = max(fear_score, greed_score, uncertainty_score)

        if max_score < 0.3:
            return MarketRegime.NEUTRAL  # All scores low = normal market

        if fear_score == max_score:
            return MarketRegime.FEAR

        if greed_score == max_score:
            return MarketRegime.GREED

        if uncertainty_score == max_score:
            return MarketRegime.UNCERTAINTY

        return MarketRegime.NEUTRAL

    def explain_regime(self, context: RegimeContext) -> str:
        """
        Generate human-readable explanation of the market regime.

        Args:
            context: RegimeContext object

        Returns:
            String explanation
        """
        regime_name = context.regime.name.replace('_', ' ').title()

        # Interpret sentiment score
        if context.sentiment_score < -0.5:
            sentiment_desc = "Very Negative"
        elif context.sentiment_score < -0.2:
            sentiment_desc = "Negative"
        elif context.sentiment_score < 0.2:
            sentiment_desc = "Neutral"
        elif context.sentiment_score < 0.5:
            sentiment_desc = "Positive"
        else:
            sentiment_desc = "Very Positive"

        # Interpret volatility
        if context.sentiment_std > 0.7:
            volatility_desc = "Very High (panic/confusion)"
        elif context.sentiment_std > 0.4:
            volatility_desc = "High (mixed signals)"
        elif context.sentiment_std > 0.2:
            volatility_desc = "Moderate"
        else:
            volatility_desc = "Low (consensus)"

        # Interpret trend
        if context.sentiment_trend > 0.3:
            trend_desc = "Improving (becoming more positive)"
        elif context.sentiment_trend > 0.1:
            trend_desc = "Slightly improving"
        elif context.sentiment_trend < -0.3:
            trend_desc = "Deteriorating (becoming more negative)"
        elif context.sentiment_trend < -0.1:
            trend_desc = "Slightly deteriorating"
        else:
            trend_desc = "Stable"

        explanation = f"""
Market Regime Analysis:
======================
Regime: {regime_name}

Sentiment Metrics:
- Overall Sentiment: {sentiment_desc} ({context.sentiment_score:+.3f})
- Sentiment Volatility: {volatility_desc} ({context.sentiment_std:.3f})
- Sentiment Trend: {trend_desc} ({context.sentiment_trend:+.3f})
- News Volume: {context.news_volume} articles

Regime Scores (0-1):
- Fear: {context.fear_score:.3f}
- Greed: {context.greed_score:.3f}
- Uncertainty: {context.uncertainty_score:.3f}

Context Flags:
- High Volatility: {'Yes' if context.is_high_volatility else 'No'}
- Extreme Sentiment: {'Yes' if context.is_extreme_sentiment else 'No'}
- News Heavy: {'Yes' if context.is_news_heavy else 'No'}

Interpretation:
{self._get_regime_interpretation(context.regime)}
"""
        return explanation

    def _get_regime_interpretation(self, regime: MarketRegime) -> str:
        """Get interpretation text for each regime."""
        interpretations = {
            MarketRegime.FEAR: (
                "Market is in a FEAR regime. Negative news dominates with high volatility.\n"
                "Price drops may be overdone due to panic. Look for oversold opportunities."
            ),
            MarketRegime.GREED: (
                "Market is in a GREED regime. Positive sentiment with low volatility.\n"
                "Price rises may be overextended due to euphoria. Watch for overbought conditions."
            ),
            MarketRegime.UNCERTAINTY: (
                "Market is in an UNCERTAINTY regime. Mixed signals with high volatility.\n"
                "Price movements are unpredictable. Reduce position sizes and wait for clarity."
            ),
            MarketRegime.NEUTRAL: (
                "Market is in a NEUTRAL regime. Balanced sentiment with moderate volatility.\n"
                "Normal trading conditions. Technical and fundamental analysis most reliable."
            ),
            MarketRegime.CAPITULATION: (
                "Market is in CAPITULATION. Extreme fear and panic selling.\n"
                "Potential bottom forming. High risk but also high opportunity for contrarians."
            ),
            MarketRegime.EUPHORIA: (
                "Market is in EUPHORIA. Extreme greed and complacency.\n"
                "Potential top forming. High risk of reversal. Consider taking profits."
            )
        }

        return interpretations.get(regime, "Unknown regime")

    def get_regime_adjustment_factor(self, context: RegimeContext) -> float:
        """
        Get adjustment factor for model predictions based on regime.

        In extreme regimes, predictions should be more conservative.

        Args:
            context: RegimeContext object

        Returns:
            Adjustment factor (0-1, where 1 = full confidence, 0.5 = half confidence)
        """
        if context.regime == MarketRegime.CAPITULATION:
            return 0.5  # Very low confidence during panic

        if context.regime == MarketRegime.EUPHORIA:
            return 0.6  # Low confidence during extreme euphoria

        if context.regime == MarketRegime.UNCERTAINTY:
            return 0.7  # Reduced confidence during uncertainty

        if context.regime == MarketRegime.FEAR:
            return 0.8  # Slightly reduced confidence during fear

        if context.regime == MarketRegime.GREED:
            return 0.85  # Slightly reduced confidence during greed

        return 1.0  # Full confidence in neutral regime


if __name__ == "__main__":
    # Quick test
    classifier = MarketRegimeClassifier()

    # Test different sentiment scenarios
    test_cases = [
        {
            'name': 'Fear Regime',
            'sentiment': {
                'sentiment_score': -0.4,
                'sentiment_std': 0.6,
                'news_volume': 15,
                'positive_ratio': 0.2,
                'negative_ratio': 0.7,
                'sentiment_trend': -0.3
            }
        },
        {
            'name': 'Greed Regime',
            'sentiment': {
                'sentiment_score': 0.5,
                'sentiment_std': 0.2,
                'news_volume': 12,
                'positive_ratio': 0.8,
                'negative_ratio': 0.1,
                'sentiment_trend': 0.2
            }
        },
        {
            'name': 'Uncertainty Regime',
            'sentiment': {
                'sentiment_score': 0.0,
                'sentiment_std': 0.8,
                'news_volume': 20,
                'positive_ratio': 0.4,
                'negative_ratio': 0.4,
                'sentiment_trend': 0.0
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test Case: {test_case['name']}")
        print(f"{'='*60}")

        context = classifier.classify(test_case['sentiment'])
        print(classifier.explain_regime(context))

        adj_factor = classifier.get_regime_adjustment_factor(context)
        print(f"\nPrediction Adjustment Factor: {adj_factor:.2f}")
        print(f"Context Vector Shape: {context.to_vector().shape}")
