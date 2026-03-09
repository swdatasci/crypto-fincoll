"""
FinColl Feature Registry

Defines feature segments based on the central dimension config.
This is the single source of truth for segment indices used in diagnostics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config.dimensions import DIMS


# ---------------------------------------------------------------------------
# Load raw YAML config once so _build_segments() can read datasource /
# update_frequency without any hardcoded strings.
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parents[2] / "config" / "feature_dimensions.yaml"
with open(_CONFIG_PATH) as _f:
    _RAW_CONFIG: Dict[str, Any] = yaml.safe_load(_f)

_FINCOLL_COMPONENTS: Dict[str, Any] = _RAW_CONFIG["fincoll"]["components"]
_SENVEC_COMPONENTS: Dict[str, Any] = _RAW_CONFIG["senvec"]["components"]


def _fc(name: str, field: str, fallback: str = "") -> str:
    """Read a string field from a fincoll component entry."""
    comp = _FINCOLL_COMPONENTS.get(name, {})
    return str(comp.get(field, fallback)) if isinstance(comp, dict) else fallback


def _sv(name: str, field: str, fallback: str = "") -> str:
    """Read a string field from a senvec component entry."""
    comp = _SENVEC_COMPONENTS.get(name, {})
    return str(comp.get(field, fallback)) if isinstance(comp, dict) else fallback


@dataclass
class FeatureSegment:
    """Definition of a feature segment."""

    name: str
    start_idx: int
    end_idx: int
    dimensions: int
    category: str
    description: str
    data_source: str
    provider: str
    update_frequency: str
    sub_groups: Optional[Dict[str, List[int]]] = None


def _build_segments() -> Dict[str, Dict]:
    segments: Dict[str, Dict] = {}
    offset = 0

    def add(
        name: str,
        length: int,
        category: str,
        description: str,
        data_source: str,
        provider: str,
        update_frequency: str,
    ) -> None:
        nonlocal offset
        start = offset
        end = offset + length - 1
        segments[name] = {
            "indices": list(range(start, end + 1)),
            "dimensions": length,
            "category": category,
            "description": description,
            "data_source": data_source,
            "provider": provider,
            "update_frequency": update_frequency,
        }
        offset = end + 1

    add(
        "technical",
        DIMS.fincoll_technical,
        "technical",
        _fc(
            "technical",
            "description",
            "Core technical indicators (RSI, MACD, Bollinger, ATR, etc.)",
        ),
        "OHLCV price data",
        _fc("technical", "datasource", "TradeStation"),
        _fc("technical", "update_frequency", "real-time"),
    )
    add(
        "advanced_technical",
        DIMS.fincoll_advanced_technical,
        "advanced_technical",
        _fc("advanced_technical", "description", "Advanced technical indicators"),
        "OHLCV price data",
        _fc("advanced_technical", "datasource", "TradeStation"),
        _fc("advanced_technical", "update_frequency", "daily"),
    )
    add(
        "velocity",
        DIMS.fincoll_velocity,
        "velocity",
        _fc(
            "velocity",
            "description",
            "Price velocity and acceleration (derived from OHLCV)",
        ),
        "Calculated",
        _fc("velocity", "datasource", "TradeStation"),
        _fc("velocity", "update_frequency", "real-time"),
    )
    add(
        "news",
        DIMS.fincoll_news,
        "news",
        _fc("news", "description", "News sentiment scores"),
        "News feed",
        _fc("news", "datasource", "TradeStation"),
        _fc("news", "update_frequency", "real-time"),
    )
    add(
        "fundamentals",
        DIMS.fincoll_fundamentals,
        "fundamentals",
        _fc(
            "fundamentals",
            "description",
            "Fundamental metrics (PE, ROE, margins, etc.)",
        ),
        "Financial statements",
        _fc("fundamentals", "datasource", "TradeStation"),
        _fc("fundamentals", "update_frequency", "quarterly"),
    )
    add(
        "cross_asset",
        DIMS.fincoll_cross_asset,
        "cross_asset",
        _fc(
            "cross_asset",
            "description",
            "Cross-asset signals (VIX, treasuries, DXY, correlations)",
        ),
        "Macro / ETF prices",
        _fc("cross_asset", "datasource", "TradeStation"),
        _fc("cross_asset", "update_frequency", "daily"),
    )
    add(
        "sector",
        DIMS.fincoll_sector,
        "sector",
        _fc("sector", "description", "Sector relative performance (via sector ETFs)"),
        "Sector ETF prices",
        _fc("sector", "datasource", "TradeStation"),
        _fc("sector", "update_frequency", "daily"),
    )
    add(
        "options",
        DIMS.fincoll_options,
        "options",
        _fc(
            "options",
            "description",
            "Options flow (put/call ratios, OI, skew) — not yet implemented",
        ),
        "Options chain",
        _fc("options", "datasource", "TradeStation"),
        _fc("options", "update_frequency", "daily"),
    )
    add(
        "support_resistance",
        DIMS.fincoll_support_resistance,
        "support_resistance",
        _fc(
            "support_resistance", "description", "Calculated support/resistance levels"
        ),
        "Calculated",
        _fc("support_resistance", "datasource", "TradeStation"),
        _fc("support_resistance", "update_frequency", "daily"),
    )
    add(
        "vwap",
        DIMS.fincoll_vwap,
        "vwap",
        _fc(
            "vwap",
            "description",
            "Multi-timeframe VWAP (calculated from intraday OHLCV)",
        ),
        "Calculated",
        _fc("vwap", "datasource", "TradeStation"),
        _fc("vwap", "update_frequency", "real-time"),
    )
    add(
        "senvec_alphavantage",
        DIMS.senvec_alphavantage,
        "sentiment",
        _sv(
            "alphavantage",
            "description",
            "Cross-asset sentiment (VIX, treasuries, DXY)",
        ),
        "Macro indicators",
        "Alpha Vantage",
        "Hourly",
    )
    add(
        "senvec_social",
        DIMS.senvec_social,
        "sentiment",
        _sv("social", "description", "Social media sentiment"),
        "Twitter, Reddit, StockTwits",
        "Twitter / Reddit / StockTwits",
        "Real-time (5-15 min)",
    )
    add(
        "senvec_news",
        DIMS.senvec_news,
        "sentiment",
        _sv("news", "description", "News sentiment scores"),
        "News articles",
        "FinLight",
        "Real-time",
    )
    add(
        "futures",
        DIMS.fincoll_futures,
        "futures",
        _fc("futures", "description", "Futures contract data (ES, NQ, CL, GC, etc.)"),
        "Futures contracts",
        _fc("futures", "datasource", "TradeStation"),
        _fc("futures", "update_frequency", "daily"),
    )
    add(
        "finnhub",
        DIMS.fincoll_finnhub,
        "finnhub",
        _fc(
            "finnhub",
            "description",
            "Finnhub fundamentals (earnings, insider trades, analyst ratings)",
        ),
        "Financial data",
        _fc("finnhub", "datasource", "Finnhub"),
        _fc("finnhub", "update_frequency", "daily"),
    )
    add(
        "early_signal",
        DIMS.fincoll_early_signal,
        "early_signal",
        _fc(
            "early_signal",
            "description",
            "Momentum/acceleration/volume derivatives for early peak/valley detection",
        ),
        "Calculated",
        _fc("early_signal", "datasource", "TradeStation"),
        _fc("early_signal", "update_frequency", "real-time"),
    )
    add(
        "market_neutral",
        DIMS.fincoll_market_neutral,
        "market_neutral",
        _fc(
            "market_neutral",
            "description",
            "Beta-adjusted returns, relative strength, sector-relative performance",
        ),
        "Calculated",
        _fc("market_neutral", "datasource", "TradeStation"),
        _fc("market_neutral", "update_frequency", "daily"),
    )
    add(
        "advanced_risk",
        DIMS.fincoll_advanced_risk,
        "advanced_risk",
        _fc(
            "advanced_risk",
            "description",
            "VaR, CVaR, downside deviation, upside/downside capture, skewness, kurtosis, Sortino",
        ),
        "Calculated",
        _fc("advanced_risk", "datasource", "TradeStation"),
        _fc("advanced_risk", "update_frequency", "daily"),
    )
    add(
        "momentum_variations",
        DIMS.fincoll_momentum_variations,
        "momentum_variations",
        _fc(
            "momentum_variations",
            "description",
            "Carhart 12-1, multi-horizon (3m, 6m, 9m), win rate, consistency",
        ),
        "Calculated",
        _fc("momentum_variations", "datasource", "TradeStation"),
        _fc("momentum_variations", "update_frequency", "daily"),
    )

    return segments


FINCOLL_FEATURE_GROUPS = _build_segments()

FINCOLL_CATEGORIES = {
    "technical": {
        "groups": ["technical"],
        "description": "Technical analysis indicators",
        "dimensions": DIMS.fincoll_technical,
        "color": "#2196F3",
    },
    "advanced_technical": {
        "groups": ["advanced_technical"],
        "description": "Advanced technical analysis",
        "dimensions": DIMS.fincoll_advanced_technical,
        "color": "#3F51B5",
    },
    "velocity": {
        "groups": ["velocity"],
        "description": "Price velocity and acceleration",
        "dimensions": DIMS.fincoll_velocity,
        "color": "#9C27B0",
    },
    "news": {
        "groups": ["news"],
        "description": "News sentiment",
        "dimensions": DIMS.fincoll_news,
        "color": "#E91E63",
    },
    "fundamentals": {
        "groups": ["fundamentals"],
        "description": "Fundamental analysis",
        "dimensions": DIMS.fincoll_fundamentals,
        "color": "#4CAF50",
    },
    "cross_asset": {
        "groups": ["cross_asset"],
        "description": "Cross-asset signals",
        "dimensions": DIMS.fincoll_cross_asset,
        "color": "#FF9800",
    },
    "sector": {
        "groups": ["sector"],
        "description": "Sector analysis",
        "dimensions": DIMS.fincoll_sector,
        "color": "#795548",
    },
    "options": {
        "groups": ["options"],
        "description": "Options flow",
        "dimensions": DIMS.fincoll_options,
        "color": "#9E9E9E",
    },
    "support_resistance": {
        "groups": ["support_resistance"],
        "description": "Support and resistance levels",
        "dimensions": DIMS.fincoll_support_resistance,
        "color": "#607D8B",
    },
    "vwap": {
        "groups": ["vwap"],
        "description": "VWAP positioning",
        "dimensions": DIMS.fincoll_vwap,
        "color": "#00BCD4",
    },
    "sentiment": {
        "groups": ["senvec_alphavantage", "senvec_social", "senvec_news"],
        "description": "SenVec sentiment features",
        "dimensions": DIMS.senvec_total,
        "color": "#E91E63",
    },
    "futures": {
        "groups": ["futures"],
        "description": "Futures features",
        "dimensions": DIMS.fincoll_futures,
        "color": "#8BC34A",
    },
    "finnhub": {
        "groups": ["finnhub"],
        "description": "Finnhub fundamentals",
        "dimensions": DIMS.fincoll_finnhub,
        "color": "#FFC107",
    },
    "early_signal": {
        "groups": ["early_signal"],
        "description": "Early peak/valley detection signals",
        "dimensions": DIMS.fincoll_early_signal,
        "color": "#FF5722",
    },
    "market_neutral": {
        "groups": ["market_neutral"],
        "description": "Market-neutral / relative performance features",
        "dimensions": DIMS.fincoll_market_neutral,
        "color": "#009688",
    },
    "advanced_risk": {
        "groups": ["advanced_risk"],
        "description": "Advanced risk metrics (VaR, CVaR, Sortino, etc.)",
        "dimensions": DIMS.fincoll_advanced_risk,
        "color": "#F44336",
    },
    "momentum_variations": {
        "groups": ["momentum_variations"],
        "description": "Multi-horizon momentum variations",
        "dimensions": DIMS.fincoll_momentum_variations,
        "color": "#673AB7",
    },
}


def get_segment_info(segment_name: str) -> Optional[Dict]:
    """Get information about a specific feature segment."""
    return FINCOLL_FEATURE_GROUPS.get(segment_name)


def get_all_segments() -> List[Dict]:
    """Get all segments with their info."""
    return [{"name": name, **info} for name, info in FINCOLL_FEATURE_GROUPS.items()]


def get_category_segments(category: str) -> List[str]:
    """Get all segment names in a category."""
    if category in FINCOLL_CATEGORIES:
        return FINCOLL_CATEGORIES[category]["groups"]
    return []


def get_indices_for_category(category: str) -> List[int]:
    """Get all feature indices for a category."""
    indices: List[int] = []
    for group_name in get_category_segments(category):
        if group_name in FINCOLL_FEATURE_GROUPS:
            indices.extend(FINCOLL_FEATURE_GROUPS[group_name]["indices"])
    return sorted(indices)


def get_total_dimensions() -> int:
    """Get total feature dimensions."""
    return sum(info["dimensions"] for info in FINCOLL_FEATURE_GROUPS.values())


def validate_feature_vector(vector_length: int) -> bool:
    """Check if vector length matches expected total."""
    expected = get_total_dimensions()
    return vector_length == expected == DIMS.fincoll_total


def get_feature_summary() -> Dict:
    """Get summary of feature vector structure."""
    return {
        "total_dimensions": get_total_dimensions(),
        "num_groups": len(FINCOLL_FEATURE_GROUPS),
        "num_categories": len(FINCOLL_CATEGORIES),
        "categories": {
            name: {
                "dimensions": info["dimensions"],
                "groups": len(info["groups"]),
                "color": info["color"],
            }
            for name, info in FINCOLL_CATEGORIES.items()
        },
    }
