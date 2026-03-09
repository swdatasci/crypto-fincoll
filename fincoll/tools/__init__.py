"""
FinColl Vector Optimization Tools

Toolkit for analyzing and optimizing the full FinColl feature vector.
Generalizes SenVec tools to work with all feature segments.

Modules:
    - feature_registry: Defines feature segments from config
    - vector_analyzer: Unified analysis across all segments
    - diagnostics_api: FastAPI endpoints for dashboard integration
"""

from .feature_registry import (
    FINCOLL_FEATURE_GROUPS,
    FINCOLL_CATEGORIES,
    get_segment_info,
    get_all_segments,
)

__all__ = [
    'FINCOLL_FEATURE_GROUPS',
    'FINCOLL_CATEGORIES',
    'get_segment_info',
    'get_all_segments',
]

__version__ = '1.0.0'
