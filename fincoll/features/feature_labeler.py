#!/usr/bin/env python3
"""
Feature Labeler - Convert 414D raw vector to labeled, interpreted features

CRITICAL: NO HARDCODED DIMENSIONS
All dimension ranges come from config/dimensions.py (DIMS config)
"""

import logging
import numpy as np
from typing import Dict, Any, List
from config.dimensions import DIMS

logger = logging.getLogger(__name__)
from .interpretations import (
    interpret_rsi,
    interpret_macd_histogram,
    interpret_adx,
    interpret_bollinger_position,
    interpret_put_call_ratio,
    interpret_volume_ratio,
    interpret_momentum,
    interpret_volatility,
    interpret_sentiment_score,
    interpret_price_acceleration,
    interpret_support_resistance_distance,
    interpret_beta,
    interpret_sharpe_ratio,
    interpret_drawdown,
    interpret_vix_level,
    interpret_news_sentiment,
    interpret_social_sentiment,
    interpret_pe_ratio,
    interpret_earnings_growth,
)


class FeatureLabeler:
    """
    Converts 414D raw feature vector to labeled, interpreted dict

    NO HARDCODED DIMENSIONS - all ranges from DIMS config
    """

    def __init__(self):
        """Initialize labeler with dimension config"""
        self.dims = DIMS
        self._validate_dimensions()
        # Cache dimension ranges (computed once, reused for all labeling calls)
        self._ranges = self._calculate_dimension_ranges()
        logger.info(f"FeatureLabeler initialized with {self.dims.fincoll_total} total dimensions")

    def _validate_dimensions(self):
        """Validate that DIMS config has all required keys"""
        required_keys = [
            'fincoll_total',
            'fincoll_technical',
            'fincoll_advanced_technical',
            'fincoll_velocity',
            'fincoll_news',
            'fincoll_fundamentals',
            'fincoll_cross_asset',
            'fincoll_sector',
            'fincoll_options',
            'fincoll_support_resistance',
            'fincoll_vwap',
            'fincoll_senvec',
            'fincoll_futures',
            'fincoll_finnhub',
            'fincoll_early_signal',
            'fincoll_market_neutral',
            'fincoll_advanced_risk',
            'fincoll_momentum_variations',
        ]

        for key in required_keys:
            if not hasattr(self.dims, key):
                raise ValueError(f"DIMS config missing required key: {key}")

    def label(self, raw_features: np.ndarray) -> Dict[str, Any]:
        """
        Label 414D raw feature vector

        Args:
            raw_features: numpy array of shape (414,) or (DIMS.fincoll_total,)

        Returns:
            Dict with labeled feature groups
        """
        if len(raw_features) != self.dims.fincoll_total:
            logger.error(f"Dimension mismatch: expected {self.dims.fincoll_total}, got {len(raw_features)}")
            raise ValueError(
                f"Expected {self.dims.fincoll_total} features, got {len(raw_features)}"
            )

        logger.debug(f"Labeling features: shape={raw_features.shape}, total={len(raw_features)}")

        # Use cached dimension ranges
        ranges = self._ranges

        # Label each feature group
        labeled = {
            "technical_indicators": self._label_technical(
                raw_features[ranges['technical']['start']:ranges['technical']['end']]
            ),
            "advanced_technical": self._label_advanced_technical(
                raw_features[ranges['advanced_technical']['start']:ranges['advanced_technical']['end']]
            ),
            "velocity": self._label_velocity(
                raw_features[ranges['velocity']['start']:ranges['velocity']['end']]
            ),
            "news": self._label_news(
                raw_features[ranges['news']['start']:ranges['news']['end']]
            ),
            "fundamentals": self._label_fundamentals(
                raw_features[ranges['fundamentals']['start']:ranges['fundamentals']['end']]
            ),
            "cross_asset": self._label_cross_asset(
                raw_features[ranges['cross_asset']['start']:ranges['cross_asset']['end']]
            ),
            "sector": self._label_sector(
                raw_features[ranges['sector']['start']:ranges['sector']['end']]
            ),
            "options": self._label_options(
                raw_features[ranges['options']['start']:ranges['options']['end']]
            ),
            "support_resistance": self._label_support_resistance(
                raw_features[ranges['support_resistance']['start']:ranges['support_resistance']['end']]
            ),
            "vwap": self._label_vwap(
                raw_features[ranges['vwap']['start']:ranges['vwap']['end']]
            ),
            "senvec": self._label_senvec(
                raw_features[ranges['senvec']['start']:ranges['senvec']['end']]
            ),
            "futures": self._label_futures(
                raw_features[ranges['futures']['start']:ranges['futures']['end']]
            ),
            "finnhub": self._label_finnhub(
                raw_features[ranges['finnhub']['start']:ranges['finnhub']['end']]
            ),
            "early_signal": self._label_early_signal(
                raw_features[ranges['early_signal']['start']:ranges['early_signal']['end']]
            ),
            "market_neutral": self._label_market_neutral(
                raw_features[ranges['market_neutral']['start']:ranges['market_neutral']['end']]
            ),
            "advanced_risk": self._label_advanced_risk(
                raw_features[ranges['advanced_risk']['start']:ranges['advanced_risk']['end']]
            ),
            "momentum_variations": self._label_momentum_variations(
                raw_features[ranges['momentum_variations']['start']:ranges['momentum_variations']['end']]
            ),
        }

        return labeled

    def _calculate_dimension_ranges(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate dimension ranges dynamically from DIMS config

        Returns cumulative ranges for each feature group
        """
        ranges = {}
        offset = 0

        components = [
            ('technical', self.dims.fincoll_technical),
            ('advanced_technical', self.dims.fincoll_advanced_technical),
            ('velocity', self.dims.fincoll_velocity),
            ('news', self.dims.fincoll_news),
            ('fundamentals', self.dims.fincoll_fundamentals),
            ('cross_asset', self.dims.fincoll_cross_asset),
            ('sector', self.dims.fincoll_sector),
            ('options', self.dims.fincoll_options),
            ('support_resistance', self.dims.fincoll_support_resistance),
            ('vwap', self.dims.fincoll_vwap),
            ('senvec', self.dims.fincoll_senvec),
            ('futures', self.dims.fincoll_futures),
            ('finnhub', self.dims.fincoll_finnhub),
            ('early_signal', self.dims.fincoll_early_signal),
            ('market_neutral', self.dims.fincoll_market_neutral),
            ('advanced_risk', self.dims.fincoll_advanced_risk),
            ('momentum_variations', self.dims.fincoll_momentum_variations),
        ]

        for name, size in components:
            ranges[name] = {
                'start': offset,
                'end': offset + size,
                'size': size
            }
            offset += size

        return ranges

    # ========== LABELING METHODS ==========
    # Each method labels one feature group

    def _label_technical(self, features: np.ndarray) -> Dict[str, Any]:
        """Label technical indicators (DIMS.fincoll_technical dimensions)"""
        # Example: RSI, MACD, ADX, Bollinger, etc.
        # TODO: Map actual positions based on FeatureExtractor implementation

        n = len(features)  # Cache length

        # Safe value extraction with NaN/Inf checks
        def safe_float(arr, idx, default=0.0):
            if idx < len(arr) and np.isfinite(arr[idx]):
                return float(arr[idx])
            return default

        return {
            "dimensions": n,
            "data": {
                "momentum": {
                    "rsi_14": safe_float(features, 0),
                    "rsi_interpretation": interpret_rsi(features[0]) if n > 0 and np.isfinite(features[0]) else "neutral",
                    "macd_histogram": safe_float(features, 1),
                    "macd_interpretation": interpret_macd_histogram(
                        features[1], features[2]
                    ) if n > 2 and np.isfinite(features[1]) and np.isfinite(features[2]) else "choppy",
                },
                "trend": {
                    "adx": safe_float(features, 3),
                    "adx_interpretation": interpret_adx(features[3]) if n > 3 and np.isfinite(features[3]) else "weak_trend",
                },
                "volatility": {
                    "bollinger_position": safe_float(features, 4, 0.5),
                    "bollinger_interpretation": interpret_bollinger_position(
                        features[4]
                    ) if n > 4 and np.isfinite(features[4]) else "mid_range",
                },
            },
            "summary": self._safe_summary(features, n, [0, 3], "RSI {:.1f}, ADX {:.1f}", "technical")
        }

    def _safe_summary(self, features: np.ndarray, n: int, indices: List[int], format_str: str, label: str) -> str:
        """
        Safely create summary string with proper bounds checking

        Args:
            features: Feature array
            n: Cached length of features
            indices: List of indices needed for format string
            format_str: Format string with placeholders
            label: Label for insufficient data message

        Returns:
            Formatted summary or fallback message
        """
        if n > max(indices) and all(np.isfinite(features[i]) for i in indices):
            try:
                return format_str.format(*[features[i] for i in indices])
            except (IndexError, ValueError):
                return f"insufficient_{label}_data"
        elif n > 0 and np.isfinite(features[0]):
            # Fallback to first feature only
            return f"{label.title()} [{features[0]:.2f}]"
        return f"insufficient_{label}_data"

    def _label_advanced_technical(self, features: np.ndarray) -> Dict[str, Any]:
        """Label advanced technical indicators (50D)"""
        return {
            "dimensions": len(features),
            "data": {
                "oscillators": {},
                "divergence": {},
            },
            "summary": "Advanced technical indicators"
        }

    def _label_velocity(self, features: np.ndarray) -> Dict[str, Any]:
        """Label velocity features (DIMS.fincoll_velocity dimensions)"""
        n = len(features)

        def safe_float(arr, idx, default=0.0):
            if idx < len(arr) and np.isfinite(arr[idx]):
                return float(arr[idx])
            return default

        return {
            "dimensions": n,
            "data": {
                "price_velocity": safe_float(features, 0),
                "price_acceleration": safe_float(features, 1),
                "acceleration_interpretation": interpret_price_acceleration(
                    features[1]
                ) if n > 1 and np.isfinite(features[1]) else "constant_velocity",
            },
            "summary": f"Velocity {features[0]:.4f}" if n > 0 and np.isfinite(features[0]) else "no_velocity"
        }

    def _label_news(self, features: np.ndarray) -> Dict[str, Any]:
        """Label news features (20D)"""
        return {
            "dimensions": len(features),
            "data": {
                "sentiment": float(features[0]) if len(features) > 0 else 0.0,
                "sentiment_interpretation": interpret_news_sentiment(
                    features[0]
                ) if len(features) > 0 else "neutral_news",
            },
            "summary": f"News sentiment {features[0]:.2f}" if len(features) > 0 else "no_news"
        }

    def _label_fundamentals(self, features: np.ndarray) -> Dict[str, Any]:
        """Label fundamental features (DIMS.fincoll_fundamentals dimensions)"""
        n = len(features)

        def safe_float(arr, idx, default=0.0):
            if idx < len(arr) and np.isfinite(arr[idx]):
                return float(arr[idx])
            return default

        return {
            "dimensions": n,
            "data": {
                "valuation": {
                    "pe_ratio": safe_float(features, 0),
                    "pe_interpretation": interpret_pe_ratio(features[0]) if n > 0 and np.isfinite(features[0]) else "fairly_valued",
                },
                "growth": {
                    "earnings_growth": safe_float(features, 1),
                    "growth_interpretation": interpret_earnings_growth(
                        features[1]
                    ) if n > 1 and np.isfinite(features[1]) else "slight_growth",
                },
            },
            "summary": f"P/E {features[0]:.1f}" if n > 0 and np.isfinite(features[0]) else "no_fundamentals"
        }

    def _label_cross_asset(self, features: np.ndarray) -> Dict[str, Any]:
        """Label cross-asset features (18D) - Beta-Adjusted Residual Momentum"""
        n = len(features)

        def safe_float(arr, idx):
            return float(arr[idx]) if idx < len(arr) and np.isfinite(arr[idx]) else 0.0

        data = {}

        # 1-day horizon (f187-f192)
        if n >= 6:
            data["horizon_1d"] = {
                "spy_return": safe_float(features, 0),
                "residual": safe_float(features, 1),
                "residual_zscore": safe_float(features, 2),
                "residual_velocity": safe_float(features, 3),
                "residual_accel": safe_float(features, 4),
                "residual_jerk": safe_float(features, 5),
            }

        # 5-day horizon (f193-f198)
        if n >= 12:
            data["horizon_5d"] = {
                "spy_return": safe_float(features, 6),
                "residual": safe_float(features, 7),
                "residual_zscore": safe_float(features, 8),
                "residual_velocity": safe_float(features, 9),
                "residual_accel": safe_float(features, 10),
                "residual_jerk": safe_float(features, 11),
            }

        # 20-day horizon (f199-f204)
        if n >= 18:
            data["horizon_20d"] = {
                "spy_return": safe_float(features, 12),
                "residual": safe_float(features, 13),
                "residual_zscore": safe_float(features, 14),
                "residual_velocity": safe_float(features, 15),
                "residual_accel": safe_float(features, 16),
                "residual_jerk": safe_float(features, 17),
            }

        return {
            "dimensions": n,
            "data": data,
            "summary": f"Cross-asset beta-adjusted momentum ({n}D)"
        }

    def _label_sector(self, features: np.ndarray) -> Dict[str, Any]:
        """Label sector features (14D)"""
        return {
            "dimensions": len(features),
            "data": {},
            "summary": "Sector classification"
        }

    def _label_options(self, features: np.ndarray) -> Dict[str, Any]:
        """Label options features (10D)"""
        return {
            "dimensions": len(features),
            "data": {
                "put_call_ratio": float(features[0]) if len(features) > 0 else 1.0,
                "put_call_interpretation": interpret_put_call_ratio(
                    features[0]
                ) if len(features) > 0 else "neutral_sentiment",
            },
            "summary": f"P/C {features[0]:.2f}" if len(features) > 0 else "no_options"
        }

    def _label_support_resistance(self, features: np.ndarray) -> Dict[str, Any]:
        """Label support/resistance features (30D)"""
        return {
            "dimensions": len(features),
            "data": {},
            "summary": "Support/resistance levels"
        }

    def _label_vwap(self, features: np.ndarray) -> Dict[str, Any]:
        """Label VWAP features (5D)"""
        return {
            "dimensions": len(features),
            "data": {},
            "summary": "VWAP multi-timeframe"
        }

    def _label_senvec(self, features: np.ndarray) -> Dict[str, Any]:
        """Label SenVec sentiment features (49D)"""
        return {
            "dimensions": len(features),
            "data": {
                "social_sentiment": float(features[0]) if len(features) > 0 else 0.0,
                "social_interpretation": interpret_social_sentiment(
                    features[0]
                ) if len(features) > 0 else "neutral_social",
            },
            "summary": f"Social sentiment {features[0]:.2f}" if len(features) > 0 else "no_senvec"
        }

    def _label_futures(self, features: np.ndarray) -> Dict[str, Any]:
        """Label futures/macro features (25D)"""
        return {
            "dimensions": len(features),
            "data": {
                "vix": float(features[0]) if len(features) > 0 else 15.0,
                "vix_interpretation": interpret_vix_level(features[0]) if len(features) > 0 else "low_fear",
            },
            "summary": f"VIX {features[0]:.1f}" if len(features) > 0 else "no_futures"
        }

    def _label_finnhub(self, features: np.ndarray) -> Dict[str, Any]:
        """Label Finnhub features (15D)"""
        return {
            "dimensions": len(features),
            "data": {},
            "summary": "Finnhub data"
        }

    def _label_early_signal(self, features: np.ndarray) -> Dict[str, Any]:
        """Label early signal features (30D)"""
        return {
            "dimensions": len(features),
            "data": {},
            "summary": "Early signals"
        }

    def _label_market_neutral(self, features: np.ndarray) -> Dict[str, Any]:
        """Label market neutral features (17D)"""
        return {
            "dimensions": len(features),
            "data": {
                "beta": float(features[0]) if len(features) > 0 else 1.0,
                "beta_interpretation": interpret_beta(features[0]) if len(features) > 0 else "correlated",
            },
            "summary": f"Beta {features[0]:.2f}" if len(features) > 0 else "no_beta"
        }

    def _label_advanced_risk(self, features: np.ndarray) -> Dict[str, Any]:
        """Label advanced risk features (8D)"""
        return {
            "dimensions": len(features),
            "data": {
                "sharpe_ratio": float(features[0]) if len(features) > 0 else 0.0,
                "sharpe_interpretation": interpret_sharpe_ratio(
                    features[0]
                ) if len(features) > 0 else "poor_risk_adjusted",
                "max_drawdown": float(features[1]) if len(features) > 1 else 0.0,
                "drawdown_interpretation": interpret_drawdown(
                    features[1]
                ) if len(features) > 1 else "minimal_drawdown",
            },
            "summary": f"Sharpe {features[0]:.2f}" if len(features) > 0 else "no_risk_metrics"
        }

    def _label_momentum_variations(self, features: np.ndarray) -> Dict[str, Any]:
        """Label momentum variation features (6D)"""
        return {
            "dimensions": len(features),
            "data": {
                "momentum_score": float(features[0]) if len(features) > 0 else 0.0,
                "momentum_interpretation": interpret_momentum(
                    features[0]
                ) if len(features) > 0 else "neutral",
            },
            "summary": f"Momentum {features[0]:.2f}" if len(features) > 0 else "no_momentum"
        }
