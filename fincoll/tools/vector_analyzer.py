"""
Vector Analyzer for FinColl Feature Vectors

Unified analysis tool that works across all feature segments.
Wraps and extends SenVec tools to handle the full configured vector.

Usage:
    from fincoll.tools.vector_analyzer import VectorAnalyzer

    analyzer = VectorAnalyzer(model, X, y)
    results = analyzer.full_analysis()
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import json
import os
import glob
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

from .feature_registry import (
    FINCOLL_FEATURE_GROUPS,
    FINCOLL_CATEGORIES,
    get_total_dimensions,
    get_indices_for_category,
)
from config.dimensions import DIMS as _DIMS


@dataclass
class SegmentAnalysis:
    """Analysis results for a single feature segment."""

    name: str
    category: str
    dimensions: int
    importance: float
    importance_pct: float
    importance_per_dim: float
    rank: int


@dataclass
class CategoryAnalysis:
    """Analysis results for a feature category."""

    name: str
    dimensions: int
    importance: float
    importance_pct: float
    efficiency: float  # importance per dimension
    color: str
    segments: List[str]


@dataclass
class VectorAnalysisResult:
    """Complete vector analysis results."""

    timestamp: str
    total_dimensions: int
    baseline_score: float
    segments: List[SegmentAnalysis]
    categories: List[CategoryAnalysis]
    recommendations: Dict[str, List[str]]
    compression_potential: Dict[str, Any]


class VectorAnalyzer:
    """
    Analyze the full FinColl feature vector (dimension from config).

    Provides segment-level and category-level importance analysis,
    compression recommendations, and optimization suggestions.

    Args:
        model: Sklearn-compatible model (or None for default)
        X: Feature matrix (n_samples, 336)
        y: Target variable
        cv: Cross-validation folds
        scoring: Scoring metric

    Example:
        >>> analyzer = VectorAnalyzer(model, X_train, y_train)
        >>> results = analyzer.full_analysis()
        >>> print(f"Most important: {results.segments[0].name}")
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
    ):
        self.model = model or RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.X = np.asarray(X) if X is not None else None
        self.y = np.asarray(y) if y is not None else None
        self.cv = cv
        self.scoring = scoring

        if self.X is not None:
            expected_dims = get_total_dimensions()
            if self.X.shape[1] != expected_dims:
                logger.warning(
                    f"Feature matrix has {self.X.shape[1]} dims, expected {expected_dims}. "
                    "Some segments may not align correctly."
                )

    def _get_baseline_score(self) -> float:
        """Calculate baseline score with all features."""
        if self.X is None or self.y is None:
            return 0.0
        scores = cross_val_score(
            self.model, self.X, self.y, cv=self.cv, scoring=self.scoring
        )
        return float(scores.mean())

    def _ablate_segment(self, segment_name: str) -> float:
        """Get score with segment removed."""
        if self.X is None or self.y is None:
            return 0.0

        segment = FINCOLL_FEATURE_GROUPS.get(segment_name)
        if not segment:
            return 0.0

        indices = segment["indices"]
        X_ablated = np.delete(self.X, indices, axis=1)

        scores = cross_val_score(
            self.model, X_ablated, self.y, cv=self.cv, scoring=self.scoring
        )
        return float(scores.mean())

    def analyze_segments(self, verbose: bool = True) -> List[SegmentAnalysis]:
        """
        Analyze importance of each feature segment via ablation.

        Returns:
            List of SegmentAnalysis sorted by importance
        """
        if self.X is None or self.y is None:
            logger.warning("No data provided, returning empty analysis")
            return []

        if verbose:
            logger.info("Calculating baseline score...")

        baseline = self._get_baseline_score()

        if verbose:
            logger.info(f"Baseline score: {baseline:.6f}")

        results = []
        total_importance = 0.0

        for segment_name, segment_info in FINCOLL_FEATURE_GROUPS.items():
            if verbose:
                logger.info(f"Analyzing segment: {segment_name}")

            ablated_score = self._ablate_segment(segment_name)
            importance = baseline - ablated_score

            results.append(
                {
                    "name": segment_name,
                    "category": segment_info["category"],
                    "dimensions": segment_info["dimensions"],
                    "importance": importance,
                }
            )
            total_importance += abs(importance)

        # Calculate percentages and rankings
        results.sort(key=lambda x: x["importance"], reverse=True)

        segment_analyses = []
        for rank, r in enumerate(results, 1):
            pct = (
                (r["importance"] / total_importance * 100)
                if total_importance > 0
                else 0
            )
            segment_analyses.append(
                SegmentAnalysis(
                    name=r["name"],
                    category=r["category"],
                    dimensions=r["dimensions"],
                    importance=r["importance"],
                    importance_pct=pct,
                    importance_per_dim=r["importance"] / r["dimensions"]
                    if r["dimensions"] > 0
                    else 0,
                    rank=rank,
                )
            )

        return segment_analyses

    def analyze_categories(
        self, segment_analyses: List[SegmentAnalysis]
    ) -> List[CategoryAnalysis]:
        """
        Aggregate segment analysis by category.

        Args:
            segment_analyses: Results from analyze_segments()

        Returns:
            List of CategoryAnalysis sorted by importance
        """
        category_data = {}

        for seg in segment_analyses:
            cat = seg.category
            if cat not in category_data:
                category_data[cat] = {
                    "importance": 0.0,
                    "dimensions": 0,
                    "segments": [],
                }
            category_data[cat]["importance"] += seg.importance
            category_data[cat]["dimensions"] += seg.dimensions
            category_data[cat]["segments"].append(seg.name)

        total_importance = sum(c["importance"] for c in category_data.values())

        results = []
        for cat_name, data in category_data.items():
            cat_info = FINCOLL_CATEGORIES.get(cat_name, {})
            pct = (
                (data["importance"] / total_importance * 100)
                if total_importance > 0
                else 0
            )
            efficiency = (
                data["importance"] / data["dimensions"] if data["dimensions"] > 0 else 0
            )

            results.append(
                CategoryAnalysis(
                    name=cat_name,
                    dimensions=data["dimensions"],
                    importance=data["importance"],
                    importance_pct=pct,
                    efficiency=efficiency,
                    color=cat_info.get("color", "#999999"),
                    segments=data["segments"],
                )
            )

        results.sort(key=lambda x: x.importance, reverse=True)
        return results

    def get_recommendations(
        self,
        segment_analyses: List[SegmentAnalysis],
        low_threshold_pct: float = 5.0,
        high_threshold_pct: float = 15.0,
    ) -> Dict[str, List[str]]:
        """
        Generate optimization recommendations.

        Args:
            segment_analyses: Results from analyze_segments()
            low_threshold_pct: Below this = candidate for reduction
            high_threshold_pct: Above this = candidate for expansion

        Returns:
            Dict with 'keep', 'reduce', 'expand' lists
        """
        keep = []
        reduce = []
        expand = []

        for seg in segment_analyses:
            if seg.importance_pct < low_threshold_pct:
                reduce.append(seg.name)
            elif seg.importance_pct > high_threshold_pct:
                expand.append(seg.name)
            else:
                keep.append(seg.name)

        return {
            "keep": keep,
            "reduce": reduce,
            "expand": expand,
        }

    def estimate_compression(
        self,
        segment_analyses: List[SegmentAnalysis],
        target_retention: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Estimate potential compression.

        Args:
            segment_analyses: Results from analyze_segments()
            target_retention: Target importance retention

        Returns:
            Compression potential metrics
        """
        total_dims = sum(s.dimensions for s in segment_analyses)
        total_importance = sum(s.importance for s in segment_analyses)

        # Sort by efficiency (importance per dim)
        by_efficiency = sorted(
            segment_analyses, key=lambda x: x.importance_per_dim, reverse=True
        )

        # Greedily select segments until we hit target retention
        selected_dims = 0
        selected_importance = 0.0
        selected_segments = []

        for seg in by_efficiency:
            if selected_importance / total_importance >= target_retention:
                break
            selected_dims += seg.dimensions
            selected_importance += seg.importance
            selected_segments.append(seg.name)

        compression_ratio = selected_dims / total_dims if total_dims > 0 else 1.0

        return {
            "original_dims": total_dims,
            "compressed_dims": selected_dims,
            "compression_ratio": compression_ratio,
            "reduction_pct": (1 - compression_ratio) * 100,
            "importance_retained": selected_importance / total_importance
            if total_importance > 0
            else 0,
            "selected_segments": selected_segments,
            "dropped_segments": [
                s.name for s in segment_analyses if s.name not in selected_segments
            ],
        }

    def full_analysis(self, verbose: bool = True) -> VectorAnalysisResult:
        """
        Run complete vector analysis.

        Returns:
            VectorAnalysisResult with all analysis data
        """
        if verbose:
            logger.info("Starting full feature vector analysis...")

        baseline = self._get_baseline_score() if self.X is not None else 0.0

        segment_analyses = self.analyze_segments(verbose=verbose)
        category_analyses = self.analyze_categories(segment_analyses)
        recommendations = self.get_recommendations(segment_analyses)
        compression = self.estimate_compression(segment_analyses)

        return VectorAnalysisResult(
            timestamp=datetime.now().isoformat(),
            total_dimensions=get_total_dimensions(),
            baseline_score=baseline,
            segments=segment_analyses,
            categories=category_analyses,
            recommendations=recommendations,
            compression_potential=compression,
        )

    def to_json(self, result: VectorAnalysisResult) -> str:
        """Convert analysis result to JSON string."""

        def serialize(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return obj

        data = {
            "timestamp": result.timestamp,
            "total_dimensions": result.total_dimensions,
            "baseline_score": result.baseline_score,
            "segments": [asdict(s) for s in result.segments],
            "categories": [asdict(c) for c in result.categories],
            "recommendations": result.recommendations,
            "compression_potential": result.compression_potential,
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> VectorAnalysisResult:
        """Load analysis result from JSON string."""
        data = json.loads(json_str)
        return VectorAnalysisResult(
            timestamp=data["timestamp"],
            total_dimensions=data["total_dimensions"],
            baseline_score=data["baseline_score"],
            segments=[SegmentAnalysis(**s) for s in data["segments"]],
            categories=[CategoryAnalysis(**c) for c in data["categories"]],
            recommendations=data["recommendations"],
            compression_potential=data["compression_potential"],
        )


# ============================================================================
# Quick analysis functions
# ============================================================================


def quick_segment_analysis(
    X: np.ndarray, y: np.ndarray, verbose: bool = True
) -> List[SegmentAnalysis]:
    """
    Quick segment importance analysis.

    Args:
        X: Feature matrix (n_samples, total_dims)
        y: Target variable
        verbose: Print progress

    Returns:
        List of SegmentAnalysis sorted by importance
    """
    analyzer = VectorAnalyzer(X=X, y=y)
    return analyzer.analyze_segments(verbose=verbose)


def load_npz_data(
    cache_dir: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load feature matrices from NPZ velocity cache files.

    Stacks all available NPZ files into a single (N, 414) matrix and
    derives a self-supervised target from the velocity segment (dims 131-150).

    Args:
        cache_dir: Directory containing *_features.npz files.
                   Defaults to finvec/data/velocity_cache/ relative to this file.

    Returns:
        Tuple (X, y) where X is (N, 414) and y is (N,), or (None, None) if no files found.
    """
    if cache_dir is None:
        # Resolve path relative to this file's location:
        # vector_analyzer.py lives at fincoll/fincoll/tools/
        # velocity_cache is at finvec/data/velocity_cache/
        this_dir = Path(__file__).parent
        cache_dir = str(
            this_dir.parent.parent.parent / "finvec" / "data" / "velocity_cache"
        )

    npz_files = glob.glob(os.path.join(cache_dir, "*_features.npz"))
    if not npz_files:
        logger.warning(f"No NPZ feature files found in {cache_dir}")
        return None, None

    arrays = []
    for fpath in sorted(npz_files):
        try:
            data = np.load(fpath)
            features = data["features"]  # shape (n_samples, 414)
            # Use DIMS.fincoll_total (414) — the full vector including unregistered
            # segments.  get_total_dimensions() only counts the 15 registered
            # segments (353D) and would incorrectly reject real cache files.
            expected = _DIMS.fincoll_total
            if features.ndim == 2 and features.shape[1] == expected:
                arrays.append(features)
                logger.info(f"Loaded {features.shape} from {os.path.basename(fpath)}")
            else:
                logger.warning(
                    f"Skipping {fpath}: unexpected shape {features.shape}, "
                    f"expected (*, {expected})"
                )
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")

    if not arrays:
        logger.warning("No usable NPZ arrays loaded")
        return None, None

    X = np.vstack(arrays)  # (N, 414)

    # Self-supervised target: mean of velocity segment (dims 131-150)
    # This captures directional price momentum, a reasonable proxy for
    # "what the model is ultimately trying to predict"
    velocity_info = FINCOLL_FEATURE_GROUPS.get("velocity", {})
    velocity_indices = velocity_info.get("indices", list(range(131, 151)))
    y = X[:, velocity_indices].mean(axis=1)  # (N,)

    logger.info(f"Loaded feature matrix: X={X.shape}, y={y.shape}")
    return X, y


def get_mock_analysis() -> VectorAnalysisResult:
    """
    Get mock analysis result for testing/demo.

    Returns realistic-looking results without actual data.
    """
    # Generate mock segment analyses
    segments = []
    rank = 1

    # Mock importance values calibrated to real-world expectations
    # Uses the 15 actual registered segment names from feature_registry.py
    mock_importance = {
        "technical": 0.18,
        "advanced_technical": 0.12,
        "senvec_social": 0.14,
        "senvec_alphavantage": 0.09,
        "fundamentals": 0.08,
        "velocity": 0.07,
        "support_resistance": 0.06,
        "futures": 0.05,
        "cross_asset": 0.05,
        "news": 0.04,
        "finnhub": 0.04,
        "sector": 0.03,
        "senvec_news": 0.03,
        "vwap": 0.01,
        "options": 0.01,
    }

    total_importance = sum(mock_importance.values())

    for name, importance in sorted(
        mock_importance.items(), key=lambda x: x[1], reverse=True
    ):
        info = FINCOLL_FEATURE_GROUPS[name]
        segments.append(
            SegmentAnalysis(
                name=name,
                category=info["category"],
                dimensions=info["dimensions"],
                importance=importance,
                importance_pct=importance / total_importance * 100,
                importance_per_dim=importance / info["dimensions"],
                rank=rank,
            )
        )
        rank += 1

    # Build categories from segments
    analyzer = VectorAnalyzer()
    categories = analyzer.analyze_categories(segments)
    recommendations = analyzer.get_recommendations(segments)
    compression = analyzer.estimate_compression(segments)

    return VectorAnalysisResult(
        timestamp=datetime.now().isoformat(),
        total_dimensions=get_total_dimensions(),
        baseline_score=-0.0234,  # Negative MSE
        segments=segments,
        categories=categories,
        recommendations=recommendations,
        compression_potential=compression,
    )
