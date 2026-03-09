"""
Market-Neutral Feature Extraction

Separates true alpha from market beta by normalizing against market/sector indices.

CRITICAL INSIGHT: Without market normalization, model confuses:
- "AAPL up 2% because market up 2%" (beta exposure)
- vs "AAPL up 2% while market flat" (true alpha)

This module computes:
1. Beta-adjusted returns (symbol excess return over expected market exposure)
2. Relative strength (outperformance vs market)
3. Sector-relative performance
4. Rolling beta/correlation
5. Market regime context

Agent Claude - 2026-02-08 - Identified as CRITICAL MISSING FEATURE
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class MarketNeutralFeatures:
    """
    Extract market-neutral features to identify true alpha.

    All features normalize by market/sector to remove beta exposure.
    """

    def __init__(
        self,
        market_symbol: str = "SPY",
        lookback_days: int = 30,
    ):
        """
        Args:
            market_symbol: Market index symbol (default: SPY)
            lookback_days: Rolling window for beta/correlation (default: 30)
        """
        self.market_symbol = market_symbol
        self.lookback_days = lookback_days

        # Sector ETF mapping (symbol first letters → sector ETF)
        self.sector_etfs = {
            'tech': 'XLK',      # Technology
            'finance': 'XLF',   # Financials
            'health': 'XLV',    # Healthcare
            'energy': 'XLE',    # Energy
            'consumer': 'XLY',  # Consumer Discretionary
            'staples': 'XLP',   # Consumer Staples
            'industrials': 'XLI',
            'materials': 'XLB',
            'utilities': 'XLU',
            'real_estate': 'XLRE',
        }

    def get_sector_etf(self, symbol: str) -> str:
        """
        Determine sector ETF for a symbol.

        TODO: Use actual sector classification (GICS codes).
        For now, use heuristics based on symbol.
        """
        # Heuristic mapping (should be replaced with actual sector lookup)
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN']
        finance_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']
        health_symbols = ['UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'MRK']

        if symbol in tech_symbols:
            return 'XLK'
        elif symbol in finance_symbols:
            return 'XLF'
        elif symbol in health_symbols:
            return 'XLV'
        else:
            return 'XLK'  # Default to tech (most common in our universe)

    def compute_beta(
        self,
        symbol_returns: pd.Series,
        market_returns: pd.Series,
    ) -> float:
        """
        Compute rolling beta (sensitivity to market movements).

        Beta = Cov(symbol, market) / Var(market)

        Args:
            symbol_returns: Symbol returns series
            market_returns: Market returns series (aligned)

        Returns:
            Beta coefficient (typically 0.5 to 2.0 for stocks)
        """
        if len(symbol_returns) < 5 or len(market_returns) < 5:
            return 1.0  # Neutral beta if insufficient data

        try:
            covariance = np.cov(symbol_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)

            if market_variance < 1e-8:
                return 1.0  # Market not moving, assume neutral

            beta = covariance / market_variance

            # Sanity check: beta should be roughly -2 to +3 for stocks
            if not (-2.0 <= beta <= 3.0):
                return np.clip(beta, -2.0, 3.0)

            return beta
        except Exception:
            return 1.0  # Fallback to market beta

    def compute_correlation(
        self,
        symbol_returns: pd.Series,
        market_returns: pd.Series,
    ) -> float:
        """Compute correlation with market."""
        if len(symbol_returns) < 5 or len(market_returns) < 5:
            return 0.5  # Neutral correlation

        try:
            corr = np.corrcoef(symbol_returns, market_returns)[0, 1]
            return corr if not np.isnan(corr) else 0.5
        except Exception:
            return 0.5

    def extract_features(
        self,
        symbol: str,
        symbol_data: pd.DataFrame,
        market_data: pd.DataFrame,
        sector_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Extract all market-neutral features.

        Args:
            symbol: Stock symbol
            symbol_data: OHLCV data for symbol (with 'close' or 'Close' column)
            market_data: OHLCV data for market index (SPY)
            sector_data: OHLCV data for sector ETF (optional)

        Returns:
            Dictionary with 17 market-neutral features:
            - beta: Rolling 30-day beta
            - correlation: Rolling correlation with market
            - beta_adj_return_1d: 1-day beta-adjusted return
            - beta_adj_return_5d: 5-day beta-adjusted return
            - beta_adj_return_20d: 20-day beta-adjusted return
            - rel_strength_1d: 1-day relative strength vs market
            - rel_strength_5d: 5-day relative strength
            - rel_strength_20d: 20-day relative strength
            - sector_rel_1d: 1-day sector-relative return
            - sector_rel_5d: 5-day sector-relative return
            - sector_rel_20d: 20-day sector-relative return
            - vix: Current VIX level (TODO: integrate)
            - market_trend: Market trend classification (0=down, 0.5=flat, 1=up)
            - market_momentum_5d: Market 5-day momentum
            - market_momentum_20d: Market 20-day momentum
            - market_breadth: % of market above MA200 (TODO: integrate)
            - alpha_ratio: (symbol_return - risk_free) / (market_return - risk_free)
        """
        # Normalize column names
        symbol_close = symbol_data.get('close', symbol_data.get('Close', symbol_data.get('close', None)))
        market_close = market_data.get('close', market_data.get('Close', market_data.get('close', None)))

        if symbol_close is None or market_close is None:
            # Return zeros if data missing
            return self._zero_features()

        # Compute returns
        symbol_returns = symbol_close.pct_change().fillna(0)
        market_returns = market_close.pct_change().fillna(0)

        # Rolling window
        symbol_ret_window = symbol_returns.tail(self.lookback_days)
        market_ret_window = market_returns.tail(self.lookback_days)

        # Beta and correlation
        beta = self.compute_beta(symbol_ret_window, market_ret_window)
        correlation = self.compute_correlation(symbol_ret_window, market_ret_window)

        # Recent returns
        symbol_ret_1d = symbol_returns.iloc[-1] if len(symbol_returns) > 0 else 0.0
        symbol_ret_5d = symbol_returns.iloc[-5:].sum() if len(symbol_returns) >= 5 else 0.0
        symbol_ret_20d = symbol_returns.iloc[-20:].sum() if len(symbol_returns) >= 20 else 0.0

        market_ret_1d = market_returns.iloc[-1] if len(market_returns) > 0 else 0.0
        market_ret_5d = market_returns.iloc[-5:].sum() if len(market_returns) >= 5 else 0.0
        market_ret_20d = market_returns.iloc[-20:].sum() if len(market_returns) >= 20 else 0.0

        # Beta-adjusted returns (CRITICAL: separates alpha from beta)
        beta_adj_1d = symbol_ret_1d - (beta * market_ret_1d)
        beta_adj_5d = symbol_ret_5d - (beta * market_ret_5d)
        beta_adj_20d = symbol_ret_20d - (beta * market_ret_20d)

        # Relative strength (simple outperformance)
        rel_strength_1d = symbol_ret_1d - market_ret_1d
        rel_strength_5d = symbol_ret_5d - market_ret_5d
        rel_strength_20d = symbol_ret_20d - market_ret_20d

        # Sector-relative (if sector data available)
        if sector_data is not None:
            sector_close = sector_data.get('close', sector_data.get('Close'))
            if sector_close is not None:
                sector_returns = sector_close.pct_change().fillna(0)
                sector_ret_1d = sector_returns.iloc[-1] if len(sector_returns) > 0 else 0.0
                sector_ret_5d = sector_returns.iloc[-5:].sum() if len(sector_returns) >= 5 else 0.0
                sector_ret_20d = sector_returns.iloc[-20:].sum() if len(sector_returns) >= 20 else 0.0

                sector_rel_1d = symbol_ret_1d - sector_ret_1d
                sector_rel_5d = symbol_ret_5d - sector_ret_5d
                sector_rel_20d = symbol_ret_20d - sector_ret_20d
            else:
                sector_rel_1d = sector_rel_5d = sector_rel_20d = 0.0
        else:
            sector_rel_1d = sector_rel_5d = sector_rel_20d = 0.0

        # Market trend classification
        if market_ret_20d > 0.05:  # Up 5%+ over 20 days
            market_trend = 1.0
        elif market_ret_20d < -0.05:  # Down 5%+ over 20 days
            market_trend = 0.0
        else:
            market_trend = 0.5  # Flat

        # Alpha ratio (Treynor-style: excess return per unit beta)
        # Simplified: (symbol_return) / (market_return), normalized
        risk_free = 0.0  # Assume 0% risk-free rate (can be updated)
        if abs(market_ret_20d - risk_free) > 0.01:  # Market moved >1%
            alpha_ratio = (symbol_ret_20d - risk_free) / (market_ret_20d - risk_free)
            alpha_ratio = np.clip(alpha_ratio, -2.0, 3.0)  # Sanity bounds
        else:
            alpha_ratio = 1.0  # Neutral if market flat

        return {
            'beta': beta,
            'correlation': correlation,
            'beta_adj_return_1d': beta_adj_1d,
            'beta_adj_return_5d': beta_adj_5d,
            'beta_adj_return_20d': beta_adj_20d,
            'rel_strength_1d': rel_strength_1d,
            'rel_strength_5d': rel_strength_5d,
            'rel_strength_20d': rel_strength_20d,
            'sector_rel_1d': sector_rel_1d,
            'sector_rel_5d': sector_rel_5d,
            'sector_rel_20d': sector_rel_20d,
            'vix': 0.0,  # TODO: Integrate VIX data
            'market_trend': market_trend,
            'market_momentum_5d': market_ret_5d,
            'market_momentum_20d': market_ret_20d,
            'market_breadth': 0.5,  # TODO: Integrate market breadth data
            'alpha_ratio': alpha_ratio,
        }

    def _zero_features(self) -> Dict[str, float]:
        """Return zero-filled features if data is missing."""
        return {
            'beta': 1.0,  # Neutral beta
            'correlation': 0.5,
            'beta_adj_return_1d': 0.0,
            'beta_adj_return_5d': 0.0,
            'beta_adj_return_20d': 0.0,
            'rel_strength_1d': 0.0,
            'rel_strength_5d': 0.0,
            'rel_strength_20d': 0.0,
            'sector_rel_1d': 0.0,
            'sector_rel_5d': 0.0,
            'sector_rel_20d': 0.0,
            'vix': 0.0,
            'market_trend': 0.5,
            'market_momentum_5d': 0.0,
            'market_momentum_20d': 0.0,
            'market_breadth': 0.5,
            'alpha_ratio': 1.0,
        }

    def to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dict to numpy array (17D)."""
        return np.array([
            features['beta'],
            features['correlation'],
            features['beta_adj_return_1d'],
            features['beta_adj_return_5d'],
            features['beta_adj_return_20d'],
            features['rel_strength_1d'],
            features['rel_strength_5d'],
            features['rel_strength_20d'],
            features['sector_rel_1d'],
            features['sector_rel_5d'],
            features['sector_rel_20d'],
            features['vix'],
            features['market_trend'],
            features['market_momentum_5d'],
            features['market_momentum_20d'],
            features['market_breadth'],
            features['alpha_ratio'],
        ])


# Convenience function for quick extraction
def extract_market_neutral_features(
    symbol: str,
    symbol_data: pd.DataFrame,
    market_data: pd.DataFrame,
    sector_data: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Quick extraction of market-neutral features as 17D array.

    Args:
        symbol: Stock symbol
        symbol_data: Symbol OHLCV data
        market_data: Market (SPY) OHLCV data
        sector_data: Sector ETF OHLCV data (optional)

    Returns:
        17D numpy array with market-neutral features
    """
    extractor = MarketNeutralFeatures()
    features = extractor.extract_features(symbol, symbol_data, market_data, sector_data)
    return extractor.to_array(features)
