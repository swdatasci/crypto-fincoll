"""
Advanced Risk Metrics Feature Extraction

Computes advanced risk metrics used by top hedge funds:
1. Value at Risk (VaR) - Maximum expected loss at 95% confidence
2. Conditional VaR (CVaR/ES) - Expected loss in worst 5% of cases (tail risk)
3. Downside Deviation - Volatility of negative returns only
4. Upside/Downside Capture Ratio - Performance vs market in up/down periods
5. Skewness - Return distribution asymmetry (want positive = more upside)
6. Kurtosis - Fat tails measure (want low = fewer extreme losses)

Based on research:
- https://medium.com/@akjha22/most-important-risk-metrics-in-quantitative-finance-72747006ef4b
- https://www.ib.barclays/our-insights/3-point-perspective/hedge-fund-outlook-2026.html

Agent Claude - 2026-02-08 - Gap Analysis Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class AdvancedRiskFeatures:
    """
    Extract advanced risk metrics from price history.

    All features computed from rolling 30-day window.
    """

    def __init__(self, lookback_days: int = 30):
        """
        Args:
            lookback_days: Rolling window for risk calculations (default: 30)
        """
        self.lookback_days = lookback_days

    def extract_features(
        self,
        symbol_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Extract all advanced risk metrics.

        Args:
            symbol_data: OHLCV data for symbol (with 'close' column)
            market_data: OHLCV data for market index (SPY) - optional for capture ratios

        Returns:
            Dictionary with 8 advanced risk features:
            - var_95: Value at Risk at 95% confidence
            - cvar_95: Conditional VaR (Expected Shortfall)
            - downside_deviation: Volatility of negative returns only
            - upside_capture: Upside capture ratio vs market
            - downside_capture: Downside capture ratio vs market
            - skewness: Return distribution skewness
            - kurtosis: Return distribution kurtosis (excess)
            - sortino_ratio: Return/downside deviation ratio
        """
        # Normalize column names
        symbol_close = symbol_data.get('close', symbol_data.get('Close', symbol_data.get('close', None)))

        if symbol_close is None:
            return self._zero_features()

        # Compute returns
        symbol_returns = symbol_close.pct_change().fillna(0)

        # Rolling window (last N days)
        returns_window = symbol_returns.tail(self.lookback_days)

        if len(returns_window) < 5:
            return self._zero_features()

        # 1. Value at Risk (95% confidence) - 5th percentile
        var_95 = np.percentile(returns_window, 5)  # Worst 5%

        # 2. Conditional VaR (Expected Shortfall) - average of worst 5%
        cvar_95 = returns_window[returns_window <= var_95].mean() if (returns_window <= var_95).any() else var_95

        # 3. Downside Deviation - volatility of negative returns only
        downside_returns = returns_window[returns_window < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0

        # 4 & 5. Upside/Downside Capture Ratios (if market data available)
        if market_data is not None:
            market_close = market_data.get('close', market_data.get('Close'))
            if market_close is not None:
                market_returns = market_close.pct_change().fillna(0).tail(self.lookback_days)

                # Align lengths and reset index to ensure boolean masking works
                min_len = min(len(returns_window), len(market_returns))
                returns_window = returns_window.tail(min_len).reset_index(drop=True)
                market_returns = market_returns.tail(min_len).reset_index(drop=True)

                # Up days: market positive
                market_up_mask = market_returns > 0
                if market_up_mask.sum() > 0:
                    upside_capture = (
                        returns_window[market_up_mask].mean() /
                        (market_returns[market_up_mask].mean() + 1e-8)
                    )
                else:
                    upside_capture = 1.0

                # Down days: market negative
                market_down_mask = market_returns < 0
                if market_down_mask.sum() > 0:
                    downside_capture = (
                        returns_window[market_down_mask].mean() /
                        (market_returns[market_down_mask].mean() + 1e-8)
                    )
                else:
                    downside_capture = 1.0

                # Clamp to reasonable bounds
                upside_capture = np.clip(upside_capture, -2.0, 3.0)
                downside_capture = np.clip(downside_capture, -2.0, 3.0)
            else:
                upside_capture = 1.0
                downside_capture = 1.0
        else:
            upside_capture = 1.0
            downside_capture = 1.0

        # 6. Skewness - asymmetry (positive = more upside)
        try:
            skewness = returns_window.skew()
            if np.isnan(skewness) or np.isinf(skewness):
                skewness = 0.0
            else:
                skewness = np.clip(skewness, -5.0, 5.0)
        except Exception:
            skewness = 0.0

        # 7. Kurtosis - fat tails (want low = fewer extreme losses)
        try:
            kurtosis = returns_window.kurtosis()  # Excess kurtosis (pandas default)
            if np.isnan(kurtosis) or np.isinf(kurtosis):
                kurtosis = 0.0
            else:
                kurtosis = np.clip(kurtosis, -5.0, 10.0)
        except Exception:
            kurtosis = 0.0

        # 8. Sortino Ratio - return/downside deviation (risk-adjusted return)
        mean_return = returns_window.mean()
        if downside_deviation > 1e-8:
            sortino_ratio = mean_return / downside_deviation
            sortino_ratio = np.clip(sortino_ratio, -5.0, 5.0)
        else:
            sortino_ratio = 0.0

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sortino_ratio': sortino_ratio,
        }

    def _zero_features(self) -> Dict[str, float]:
        """Return zero-filled features if data is missing."""
        return {
            'var_95': 0.0,
            'cvar_95': 0.0,
            'downside_deviation': 0.0,
            'upside_capture': 1.0,
            'downside_capture': 1.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'sortino_ratio': 0.0,
        }

    def to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dict to numpy array (8D)."""
        return np.array([
            features['var_95'],
            features['cvar_95'],
            features['downside_deviation'],
            features['upside_capture'],
            features['downside_capture'],
            features['skewness'],
            features['kurtosis'],
            features['sortino_ratio'],
        ])


# Convenience function for quick extraction
def extract_advanced_risk_features(
    symbol_data: pd.DataFrame,
    market_data: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Quick extraction of advanced risk features as 8D array.

    Args:
        symbol_data: Symbol OHLCV data
        market_data: Market (SPY) OHLCV data (optional)

    Returns:
        8D numpy array with advanced risk features
    """
    extractor = AdvancedRiskFeatures()
    features = extractor.extract_features(symbol_data, market_data)
    return extractor.to_array(features)
