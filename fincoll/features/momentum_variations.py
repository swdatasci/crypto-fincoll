"""
Momentum Variations Feature Extraction

Computes momentum factor variations used by top quant funds:
1. Carhart Momentum (12-1) - 12-month momentum excluding last month (mean reversion)
2. Multi-Horizon Momentum - 3m, 6m, 9m returns
3. Momentum Strength - Win rate and consistency

Based on research:
- https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2018/03/Juhani.pdf
- Carhart (1997) - Four-Factor Model

Key Insight: Different momentum horizons capture different effects:
- Short-term (1m): Mean reversion (negative)
- Intermediate (3-12m): Trend following (positive)
- Long-term (>12m): Can reverse (value effect)

Agent Claude - 2026-02-08 - Gap Analysis Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict


class MomentumVariationsFeatures:
    """
    Extract momentum variations from price history.

    Classic Carhart momentum uses 12-1 (exclude last month to avoid short-term reversal).
    """

    def __init__(self):
        """Initialize momentum extractor."""
        pass

    def extract_features(
        self,
        symbol_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Extract all momentum variation features.

        Args:
            symbol_data: OHLCV data for symbol (with 'close' column)

        Returns:
            Dictionary with 6 momentum variation features:
            - momentum_12_1: Carhart momentum (12-1 months)
            - momentum_3m: 3-month momentum
            - momentum_6m: 6-month momentum
            - momentum_9m: 9-month momentum
            - momentum_win_rate: % of positive months in last 12
            - momentum_consistency: Inverse volatility of monthly returns
        """
        # Normalize column names
        symbol_close = symbol_data.get('close', symbol_data.get('Close', symbol_data.get('close', None)))

        if symbol_close is None or len(symbol_close) < 30:
            return self._zero_features()

        # Compute returns
        daily_returns = symbol_close.pct_change().fillna(0)

        # Approximate monthly returns (21 trading days)
        if len(daily_returns) < 21:
            return self._zero_features()

        # Recent returns for different horizons
        ret_1m = self._compute_period_return(symbol_close, days=21)  # ~1 month
        ret_3m = self._compute_period_return(symbol_close, days=63)  # ~3 months
        ret_6m = self._compute_period_return(symbol_close, days=126)  # ~6 months
        ret_9m = self._compute_period_return(symbol_close, days=189)  # ~9 months
        ret_12m = self._compute_period_return(symbol_close, days=252)  # ~12 months

        # 1. Carhart Momentum (12-1): Exclude last month to avoid short-term reversal
        momentum_12_1 = ret_12m - ret_1m

        # 2-4. Multi-horizon momentum
        momentum_3m = ret_3m
        momentum_6m = ret_6m
        momentum_9m = ret_9m

        # 5. Momentum Win Rate: % of positive months in last 12
        if len(symbol_close) >= 252:
            monthly_returns = []
            for i in range(12):
                start_idx = -(252 - i*21)
                end_idx = -(252 - (i+1)*21) if i < 11 else None
                month_ret = self._compute_period_return(
                    symbol_close.iloc[start_idx:end_idx] if end_idx else symbol_close.iloc[start_idx:],
                    days=21
                )
                monthly_returns.append(month_ret)

            momentum_win_rate = sum(1 for r in monthly_returns if r > 0) / len(monthly_returns)
            momentum_consistency = 1.0 / (np.std(monthly_returns) + 1e-8)  # Inverse volatility
            momentum_consistency = np.clip(momentum_consistency, 0.0, 10.0)
        else:
            momentum_win_rate = 0.5
            momentum_consistency = 0.0

        return {
            'momentum_12_1': momentum_12_1,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'momentum_9m': momentum_9m,
            'momentum_win_rate': momentum_win_rate,
            'momentum_consistency': momentum_consistency,
        }

    def _compute_period_return(self, prices: pd.Series, days: int) -> float:
        """Compute return over last N days."""
        if len(prices) < days:
            days = len(prices)
        if len(prices) < 2:
            return 0.0

        start_price = prices.iloc[-days]
        end_price = prices.iloc[-1]

        if start_price == 0 or np.isnan(start_price) or np.isnan(end_price):
            return 0.0

        return (end_price - start_price) / start_price

    def _zero_features(self) -> Dict[str, float]:
        """Return zero-filled features if data is missing."""
        return {
            'momentum_12_1': 0.0,
            'momentum_3m': 0.0,
            'momentum_6m': 0.0,
            'momentum_9m': 0.0,
            'momentum_win_rate': 0.5,
            'momentum_consistency': 0.0,
        }

    def to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dict to numpy array (6D)."""
        return np.array([
            features['momentum_12_1'],
            features['momentum_3m'],
            features['momentum_6m'],
            features['momentum_9m'],
            features['momentum_win_rate'],
            features['momentum_consistency'],
        ])


# Convenience function for quick extraction
def extract_momentum_variations_features(
    symbol_data: pd.DataFrame,
) -> np.ndarray:
    """
    Quick extraction of momentum variations features as 6D array.

    Args:
        symbol_data: Symbol OHLCV data

    Returns:
        6D numpy array with momentum variations features
    """
    extractor = MomentumVariationsFeatures()
    features = extractor.extract_features(symbol_data)
    return extractor.to_array(features)
