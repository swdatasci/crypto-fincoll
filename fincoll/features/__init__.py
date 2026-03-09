"""
Financial Feature Engineering Module

Provides advanced feature calculations including mathematical derivatives
and technical indicators for FinVec V2, plus Phase 1 feature integration.

V2 Features:
- Mathematical derivatives (velocity, acceleration, jerk)
- Technical indicators (RSI, MACD, Bollinger Bands)

Phase 1 Features:
- Fundamentals as Context Embeddings (WHO the company is)
- Historical Fundamental Changes (WHEN context changes)
- Sentiment as Market Regime (WHY behavior changes)
- Unified Feature Combiner (Integration)
"""

from .financial_features import FinancialFeatureCalculator
from .fundamental_embeddings import FundamentalEmbedding, CompanyContext
from .fundamental_history import FundamentalHistoryTracker
from .market_regime import MarketRegimeClassifier, MarketRegime
# NOTE: FeatureCombiner disabled for Phase 3 - requires fundamentals/sentiment/technicals modules
# from .feature_combiner import FeatureCombiner, UnifiedFeatureVector

__all__ = [
    'FinancialFeatureCalculator',
    'FundamentalEmbedding',
    'CompanyContext',
    'FundamentalHistoryTracker',
    'MarketRegimeClassifier',
    'MarketRegime',
    # 'FeatureCombiner',
    # 'UnifiedFeatureVector',
]
