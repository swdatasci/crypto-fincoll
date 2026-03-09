#!/usr/bin/env python3
"""
Feature interpretation functions for human-readable labels

NO HARDCODED DIMENSIONS - all logic is value-based, not position-based
"""

def interpret_rsi(rsi: float) -> str:
    """Interpret RSI value (0-100)"""
    if rsi < 30:
        return "oversold"
    elif rsi < 40:
        return "oversold_warning"
    elif rsi > 70:
        return "overbought"
    elif rsi > 60:
        return "overbought_warning"
    else:
        return "neutral"


def interpret_macd_histogram(current: float, previous: float) -> str:
    """Interpret MACD histogram trend"""
    if abs(current) < 0.0001:
        return "choppy"

    if current > 0 and current > previous:
        return "bullish_crossover"
    elif current > 0 and current < previous:
        return "bullish_weakening"
    elif current < 0 and current < previous:
        return "bearish_crossover"
    elif current < 0 and current > previous:
        return "bearish_weakening"
    else:
        return "choppy"


def interpret_adx(adx: float) -> str:
    """Interpret ADX trend strength (0-100)"""
    if adx < 20:
        return "weak_trend"
    elif adx < 25:
        return "developing_trend"
    elif adx < 50:
        return "strong_trend"
    else:
        return "very_strong_trend"


def interpret_bollinger_position(position: float) -> str:
    """Interpret price position within Bollinger Bands (0-1)"""
    if position < 0.2:
        return "near_lower_band"
    elif position < 0.4:
        return "lower_half"
    elif position < 0.6:
        return "mid_range"
    elif position < 0.8:
        return "upper_half"
    else:
        return "near_upper_band"


def interpret_put_call_ratio(ratio: float) -> str:
    """Interpret put/call ratio"""
    if ratio < 0.7:
        return "bullish_sentiment"
    elif ratio < 1.0:
        return "neutral_sentiment"
    elif ratio < 1.3:
        return "bearish_sentiment"
    else:
        return "extreme_bearish_sentiment"


def interpret_volume_ratio(ratio: float) -> str:
    """Interpret current volume vs average (ratio)"""
    if ratio > 1.5:
        return "high_volume"
    elif ratio > 1.2:
        return "above_average"
    elif ratio > 0.8:
        return "normal_volume"
    elif ratio > 0.5:
        return "below_average"
    else:
        return "low_volume"


def interpret_momentum(momentum: float) -> str:
    """Interpret momentum score (-1 to 1 normalized)"""
    if momentum > 0.3:
        return "strong_bullish"
    elif momentum > 0.1:
        return "bullish"
    elif momentum < -0.3:
        return "strong_bearish"
    elif momentum < -0.1:
        return "bearish"
    else:
        return "neutral"


def interpret_volatility(volatility: float) -> str:
    """Interpret volatility level (normalized 0-1)"""
    if volatility > 0.7:
        return "extreme_volatility"
    elif volatility > 0.5:
        return "high_volatility"
    elif volatility > 0.3:
        return "moderate_volatility"
    else:
        return "low_volatility"


def interpret_sentiment_score(score: float) -> str:
    """Interpret sentiment score (-1 to 1)"""
    if score > 0.5:
        return "very_bullish"
    elif score > 0.2:
        return "bullish"
    elif score < -0.5:
        return "very_bearish"
    elif score < -0.2:
        return "bearish"
    else:
        return "neutral"


def interpret_price_acceleration(accel: float) -> str:
    """Interpret price acceleration (2nd derivative)"""
    if accel > 0.001:
        return "accelerating_up"
    elif accel > 0:
        return "slightly_accelerating_up"
    elif accel < -0.001:
        return "accelerating_down"
    elif accel < 0:
        return "slightly_accelerating_down"
    else:
        return "constant_velocity"


def interpret_support_resistance_distance(distance: float) -> str:
    """Interpret distance to support/resistance (% from current price)"""
    abs_distance = abs(distance)

    if abs_distance < 0.01:
        return "at_level"
    elif abs_distance < 0.02:
        return "very_near"
    elif abs_distance < 0.05:
        return "near"
    elif abs_distance < 0.10:
        return "approaching"
    else:
        return "far"


def interpret_beta(beta: float) -> str:
    """Interpret beta (market correlation)"""
    if beta > 1.5:
        return "highly_correlated_amplified"
    elif beta > 1.0:
        return "correlated_amplified"
    elif beta > 0.5:
        return "correlated"
    elif beta > -0.5:
        return "uncorrelated"
    elif beta > -1.0:
        return "inverse_correlated"
    else:
        return "highly_inverse_correlated"


def interpret_sharpe_ratio(sharpe: float) -> str:
    """Interpret Sharpe ratio (risk-adjusted return)"""
    if sharpe > 2.0:
        return "excellent_risk_adjusted"
    elif sharpe > 1.0:
        return "good_risk_adjusted"
    elif sharpe > 0.5:
        return "acceptable_risk_adjusted"
    elif sharpe > 0:
        return "poor_risk_adjusted"
    else:
        return "negative_risk_adjusted"


def interpret_drawdown(drawdown: float) -> str:
    """Interpret drawdown percentage (0-1)"""
    if drawdown > 0.2:
        return "severe_drawdown"
    elif drawdown > 0.1:
        return "significant_drawdown"
    elif drawdown > 0.05:
        return "moderate_drawdown"
    elif drawdown > 0.02:
        return "minor_drawdown"
    else:
        return "minimal_drawdown"


def interpret_vix_level(vix: float) -> str:
    """Interpret VIX level (fear index)"""
    if vix > 30:
        return "extreme_fear"
    elif vix > 20:
        return "elevated_fear"
    elif vix > 15:
        return "moderate_fear"
    else:
        return "low_fear"


def interpret_news_sentiment(sentiment: float) -> str:
    """Interpret news sentiment score (-1 to 1)"""
    if sentiment > 0.6:
        return "very_positive_news"
    elif sentiment > 0.3:
        return "positive_news"
    elif sentiment < -0.6:
        return "very_negative_news"
    elif sentiment < -0.3:
        return "negative_news"
    else:
        return "neutral_news"


def interpret_social_sentiment(sentiment: float) -> str:
    """Interpret social media sentiment score (-1 to 1)"""
    if sentiment > 0.6:
        return "very_bullish_social"
    elif sentiment > 0.3:
        return "bullish_social"
    elif sentiment < -0.6:
        return "very_bearish_social"
    elif sentiment < -0.3:
        return "bearish_social"
    else:
        return "neutral_social"


def interpret_pe_ratio(pe: float) -> str:
    """Interpret P/E ratio"""
    if pe < 0:
        return "negative_earnings"
    elif pe < 10:
        return "undervalued"
    elif pe < 20:
        return "fairly_valued"
    elif pe < 30:
        return "overvalued"
    else:
        return "highly_overvalued"


def interpret_earnings_growth(growth: float) -> str:
    """Interpret earnings growth rate (-1 to 1)"""
    if growth > 0.2:
        return "strong_growth"
    elif growth > 0.1:
        return "moderate_growth"
    elif growth > 0:
        return "slight_growth"
    elif growth > -0.1:
        return "slight_decline"
    else:
        return "significant_decline"


# Utility function to get all interpretation functions
def get_all_interpreters():
    """Return dict of all interpretation functions"""
    return {
        'rsi': interpret_rsi,
        'macd_histogram': interpret_macd_histogram,
        'adx': interpret_adx,
        'bollinger_position': interpret_bollinger_position,
        'put_call_ratio': interpret_put_call_ratio,
        'volume_ratio': interpret_volume_ratio,
        'momentum': interpret_momentum,
        'volatility': interpret_volatility,
        'sentiment_score': interpret_sentiment_score,
        'price_acceleration': interpret_price_acceleration,
        'support_resistance_distance': interpret_support_resistance_distance,
        'beta': interpret_beta,
        'sharpe_ratio': interpret_sharpe_ratio,
        'drawdown': interpret_drawdown,
        'vix_level': interpret_vix_level,
        'news_sentiment': interpret_news_sentiment,
        'social_sentiment': interpret_social_sentiment,
        'pe_ratio': interpret_pe_ratio,
        'earnings_growth': interpret_earnings_growth,
    }
