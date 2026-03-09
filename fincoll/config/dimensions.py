#!/usr/bin/env python3
"""
Central Feature Dimension Configuration Loader

Loads feature dimensions from feature_dimensions.yaml and provides
type-safe access across the entire codebase.

Usage:
    from config.dimensions import DIMS

    # SenVec dimensions
    senvec_dim = DIMS.senvec_total  # 49
    alphavantage_dim = DIMS.senvec_alphavantage  # 18

    # FinColl dimensions
    fincoll_dim = DIMS.fincoll_total  # 353

    # Model dimensions
    input_dim = DIMS.model_input  # 353
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _dim(component) -> int:
    """Extract dimension count from a component that may be an int or a structured dict."""
    if isinstance(component, dict):
        return component["dimensions"]
    return int(component)


def _datasource(component) -> str:
    """Extract datasource string from a structured component dict, or empty string."""
    if isinstance(component, dict):
        return str(component.get("datasource", ""))
    return ""


@dataclass(frozen=True)
class FeatureDimensions:
    """
    Feature dimensions loaded from YAML config.

    All fields are read-only (frozen) to prevent accidental modification.
    """

    # SenVec
    senvec_total: int
    senvec_feature_range: str
    senvec_alphavantage: int
    senvec_alphavantage_range: str
    senvec_social: int
    senvec_social_range: str
    senvec_news: int
    senvec_news_range: str

    # FinColl
    fincoll_total: int
    fincoll_technical: int
    fincoll_advanced_technical: int
    fincoll_velocity: int
    fincoll_news: int
    fincoll_fundamentals: int
    fincoll_cross_asset: int
    fincoll_sector: int
    fincoll_options: int
    fincoll_support_resistance: int
    fincoll_vwap: int
    fincoll_senvec: int
    fincoll_futures: int
    fincoll_finnhub: int
    fincoll_early_signal: int
    fincoll_market_neutral: int
    fincoll_advanced_risk: int
    fincoll_momentum_variations: int

    # Model
    model_input: int
    model_output: int
    model_d_model: int
    model_n_heads: int
    model_n_layers: int
    model_sequence_length: int
    model_dropout: float

    # Validation
    zero_threshold: float
    min_samples: int
    min_symbols: int


def load_dimensions(config_path: Optional[Path] = None) -> FeatureDimensions:
    """
    Load feature dimensions from YAML config file.

    Args:
        config_path: Path to feature_dimensions.yaml (default: auto-detect)

    Returns:
        FeatureDimensions dataclass with all dimension values
    """
    if config_path is None:
        # Auto-detect config path
        config_path = Path(__file__).parent / "feature_dimensions.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return FeatureDimensions(
        # SenVec
        senvec_total=config["senvec"]["total"],
        senvec_feature_range=config["senvec"]["feature_range"],
        senvec_alphavantage=config["senvec"]["components"]["alphavantage"][
            "dimensions"
        ],
        senvec_alphavantage_range=config["senvec"]["components"]["alphavantage"][
            "range"
        ],
        senvec_social=config["senvec"]["components"]["social"]["dimensions"],
        senvec_social_range=config["senvec"]["components"]["social"]["range"],
        senvec_news=config["senvec"]["components"]["news"]["dimensions"],
        senvec_news_range=config["senvec"]["components"]["news"]["range"],
        # FinColl
        fincoll_total=config["fincoll"]["total"],
        fincoll_technical=_dim(config["fincoll"]["components"]["technical"]),
        fincoll_advanced_technical=_dim(
            config["fincoll"]["components"]["advanced_technical"]
        ),
        fincoll_velocity=_dim(config["fincoll"]["components"]["velocity"]),
        fincoll_news=_dim(config["fincoll"]["components"]["news"]),
        fincoll_fundamentals=_dim(config["fincoll"]["components"]["fundamentals"]),
        fincoll_cross_asset=_dim(config["fincoll"]["components"]["cross_asset"]),
        fincoll_sector=_dim(config["fincoll"]["components"]["sector"]),
        fincoll_options=_dim(config["fincoll"]["components"]["options"]),
        fincoll_support_resistance=_dim(
            config["fincoll"]["components"]["support_resistance"]
        ),
        fincoll_vwap=_dim(config["fincoll"]["components"]["vwap"]),
        fincoll_senvec=_dim(config["fincoll"]["components"]["senvec"]),
        fincoll_futures=_dim(config["fincoll"]["components"]["futures"]),
        fincoll_finnhub=_dim(config["fincoll"]["components"]["finnhub"]),
        fincoll_early_signal=_dim(config["fincoll"]["components"]["early_signal"]),
        fincoll_market_neutral=_dim(config["fincoll"]["components"]["market_neutral"]),
        fincoll_advanced_risk=_dim(config["fincoll"]["components"]["advanced_risk"]),
        fincoll_momentum_variations=_dim(
            config["fincoll"]["components"]["momentum_variations"]
        ),
        # Model
        model_input=config["models"]["velocity_transformer"]["input_dim"],
        model_output=config["models"]["velocity_transformer"]["output_dim"],
        model_d_model=config["models"]["velocity_transformer"]["d_model"],
        model_n_heads=config["models"]["velocity_transformer"]["n_heads"],
        model_n_layers=config["models"]["velocity_transformer"]["n_layers"],
        model_sequence_length=config["models"]["velocity_transformer"][
            "sequence_length"
        ],
        model_dropout=config["models"]["velocity_transformer"]["dropout"],
        # Validation
        zero_threshold=config["validation"]["zero_threshold"],
        min_samples=config["validation"]["min_samples"],
        min_symbols=config["validation"]["min_symbols"],
    )


# Global singleton instance
DIMS: FeatureDimensions = load_dimensions()


def get_fincoll_datasources(config_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Return a mapping of fincoll component name → datasource string.

    Values come directly from feature_dimensions.yaml so feature_registry.py
    and diagnostics code never need to hardcode provider names.

    Example::

        {
            "technical": "TradeStation",
            "finnhub":   "Finnhub",
            "senvec":    "SenVec",
            ...
        }
    """
    if config_path is None:
        config_path = Path(__file__).parent / "feature_dimensions.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {
        name: _datasource(component)
        for name, component in config["fincoll"]["components"].items()
    }


def validate_consistency() -> bool:
    """
    Validate that all dimension calculations are consistent.

    Returns:
        True if all validations pass

    Raises:
        ValueError: If dimensions are inconsistent
    """
    # SenVec total should equal sum of components
    senvec_sum = DIMS.senvec_alphavantage + DIMS.senvec_social + DIMS.senvec_news
    if senvec_sum != DIMS.senvec_total:
        raise ValueError(
            f"SenVec dimension mismatch: "
            f"total={DIMS.senvec_total} but components sum to {senvec_sum}"
        )

    # FinColl total should equal sum of components
    fincoll_sum = (
        DIMS.fincoll_technical
        + DIMS.fincoll_advanced_technical
        + DIMS.fincoll_velocity
        + DIMS.fincoll_news
        + DIMS.fincoll_fundamentals
        + DIMS.fincoll_cross_asset
        + DIMS.fincoll_sector
        + DIMS.fincoll_options
        + DIMS.fincoll_support_resistance
        + DIMS.fincoll_vwap
        + DIMS.fincoll_senvec
        + DIMS.fincoll_futures
        + DIMS.fincoll_finnhub
        + DIMS.fincoll_early_signal
        + DIMS.fincoll_market_neutral
        + DIMS.fincoll_advanced_risk
        + DIMS.fincoll_momentum_variations
    )
    if fincoll_sum != DIMS.fincoll_total:
        raise ValueError(
            f"FinColl dimension mismatch: "
            f"total={DIMS.fincoll_total} but components sum to {fincoll_sum}"
        )

    # FinColl SenVec should match SenVec total
    if DIMS.fincoll_senvec != DIMS.senvec_total:
        raise ValueError(
            f"FinColl SenVec dimension ({DIMS.fincoll_senvec}) "
            f"doesn't match SenVec total ({DIMS.senvec_total})"
        )

    # Model input should match FinColl total
    if DIMS.model_input != DIMS.fincoll_total:
        raise ValueError(
            f"Model input dimension ({DIMS.model_input}) "
            f"doesn't match FinColl total ({DIMS.fincoll_total})"
        )

    return True


# Validate on import
validate_consistency()


if __name__ == "__main__":
    print("Feature Dimensions Configuration")
    print("=" * 60)

    print("\nSenVec:")
    print(f"  Total: {DIMS.senvec_total}D ({DIMS.senvec_feature_range})")
    print(
        f"  - AlphaVantage: {DIMS.senvec_alphavantage}D ({DIMS.senvec_alphavantage_range})"
    )
    print(f"  - Social: {DIMS.senvec_social}D ({DIMS.senvec_social_range})")
    print(f"  - News: {DIMS.senvec_news}D ({DIMS.senvec_news_range})")

    print("\nFinColl:")
    print(f"  Total: {DIMS.fincoll_total}D")
    print(f"  Components:")
    print(f"    Technical: {DIMS.fincoll_technical}D")
    print(f"    Advanced Technical: {DIMS.fincoll_advanced_technical}D")
    print(f"    Velocity: {DIMS.fincoll_velocity}D")
    print(f"    News: {DIMS.fincoll_news}D")
    print(f"    Fundamentals: {DIMS.fincoll_fundamentals}D")
    print(f"    Cross-Asset: {DIMS.fincoll_cross_asset}D")
    print(f"    Sector: {DIMS.fincoll_sector}D")
    print(f"    Options: {DIMS.fincoll_options}D")
    print(f"    Support/Resistance: {DIMS.fincoll_support_resistance}D")
    print(f"    VWAP: {DIMS.fincoll_vwap}D")
    print(f"    SenVec: {DIMS.fincoll_senvec}D")
    print(f"    Futures: {DIMS.fincoll_futures}D")
    print(f"    Finnhub: {DIMS.fincoll_finnhub}D")
    print(f"    Early Signal: {DIMS.fincoll_early_signal}D")
    print(f"    Market-Neutral: {DIMS.fincoll_market_neutral}D")
    print(f"    Advanced Risk: {DIMS.fincoll_advanced_risk}D")
    print(f"    Momentum Variations: {DIMS.fincoll_momentum_variations}D")

    print("\nModel:")
    print(f"  Input: {DIMS.model_input}D")
    print(f"  Output: {DIMS.model_output}D")
    print(f"  d_model: {DIMS.model_d_model}")
    print(f"  Heads: {DIMS.model_n_heads}")
    print(f"  Layers: {DIMS.model_n_layers}")
    print(f"  Sequence: {DIMS.model_sequence_length}")
    print(f"  Dropout: {DIMS.model_dropout}")

    print("\nValidation:")
    print(f"  Zero threshold: {DIMS.zero_threshold * 100}%")
    print(f"  Min samples: {DIMS.min_samples}")
    print(f"  Min symbols: {DIMS.min_symbols}")

    print("\n" + "=" * 60)
    print("✅ All dimensions validated successfully")
