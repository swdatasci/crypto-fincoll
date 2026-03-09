"""
Fundamental Embeddings - Company Context from Fundamentals

KEY INSIGHT: Fundamentals are NOT direct price predictors!
Instead, they provide CONTEXT about what kind of company this is,
so the model knows HOW to interpret price patterns.

User Quote: "fundamentals won't change often, but the context... is valuable
for company/sector/industry... including symbol itself would not provide this
context for change of prediction behaviors"

What this does:
- Maps fundamentals to sector/industry/size/growth context vectors
- Creates embeddings that tell the model "what kind of company is this"
- Growth vs Value (high P/E vs low P/E)
- Tech vs Finance vs Energy (sector patterns)
- Mega-cap vs Small-cap (size effects)
- Debt-heavy vs Cash-rich (capital structure)

Agent Delta - Phase 1 FEATURE INTEGRATION stream
"""

import numpy as np
import yfinance as yf
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SectorType(Enum):
    """Major sector classifications."""

    TECHNOLOGY = 0
    FINANCE = 1
    HEALTHCARE = 2
    ENERGY = 3
    CONSUMER_CYCLICAL = 4
    CONSUMER_DEFENSIVE = 5
    INDUSTRIALS = 6
    MATERIALS = 7
    UTILITIES = 8
    REAL_ESTATE = 9
    COMMUNICATION = 10
    UNKNOWN = 11


class SizeCategory(Enum):
    """Market cap size categories."""

    MEGA_CAP = 0  # > $200B
    LARGE_CAP = 1  # $10B - $200B
    MID_CAP = 2  # $2B - $10B
    SMALL_CAP = 3  # $300M - $2B
    MICRO_CAP = 4  # < $300M


class GrowthStyle(Enum):
    """Growth vs Value classification."""

    HYPER_GROWTH = 0  # High P/E, high growth
    GROWTH = 1  # Above avg P/E, positive growth
    BALANCED = 2  # Moderate P/E and growth
    VALUE = 3  # Low P/E, stable
    DEEP_VALUE = 4  # Very low P/E or distressed


@dataclass
class CompanyContext:
    """
    Company context vector derived from fundamentals.

    This tells the model "what kind of company is this" so it knows
    how to interpret price patterns differently for:
    - A mega-cap tech growth stock (AAPL, MSFT)
    - A large-cap value financial (JPM, BAC)
    - A mid-cap energy company (COP, HAL)
    - A small-cap biotech (high volatility, binary events)
    """

    # Core context dimensions
    sector: SectorType
    size_category: SizeCategory
    growth_style: GrowthStyle

    # Context scores (0-1 normalized)
    growth_score: float  # 0 = deep value, 1 = hyper growth
    quality_score: float  # 0 = low quality, 1 = high quality
    leverage_score: float  # 0 = debt-heavy, 1 = cash-rich
    profitability_score: float  # 0 = unprofitable, 1 = highly profitable

    # Raw fundamentals for reference
    fundamentals: Dict[str, Optional[float]]

    def to_vector(self) -> np.ndarray:
        """
        Convert context to numerical vector for model input.

        Returns:
            Array of shape (11,) with:
            [sector_onehot (12), size (5), growth_style (5),
             growth_score, quality_score, leverage_score, profitability_score]
        """
        # One-hot encode sector (12 categories)
        sector_vec = np.zeros(12)
        sector_vec[self.sector.value] = 1.0

        # One-hot encode size (5 categories)
        size_vec = np.zeros(5)
        size_vec[self.size_category.value] = 1.0

        # One-hot encode growth style (5 categories)
        growth_style_vec = np.zeros(5)
        growth_style_vec[self.growth_style.value] = 1.0

        # Continuous scores
        scores = np.array(
            [
                self.growth_score,
                self.quality_score,
                self.leverage_score,
                self.profitability_score,
            ]
        )

        # Concatenate all features
        return np.concatenate([sector_vec, size_vec, growth_style_vec, scores])

    def get_embedding_size(self) -> int:
        """Get the size of the context vector."""
        return 12 + 5 + 5 + 4  # = 26 features


class FundamentalEmbedding:
    """
    Creates context embeddings from fundamental data.

    This class takes raw fundamental ratios and converts them into
    structured context that tells the model what kind of company it's
    looking at, NOT what the price should be.
    """

    def __init__(self):
        """Initialize the fundamental embedding generator."""
        pass

    def create_context(
        self, fundamentals: Dict[str, Optional[float]], symbol: Optional[str] = None
    ) -> CompanyContext:
        """
        Create company context from fundamental data.

        Args:
            fundamentals: Dictionary of fundamental ratios (from FundamentalCollector)
            symbol: Stock ticker (optional, used to fetch sector info)

        Returns:
            CompanyContext with all context dimensions
        """
        # Classify sector (from yfinance if symbol provided)
        sector = self._classify_sector(symbol) if symbol else SectorType.UNKNOWN

        # Classify size (from market cap if available)
        size_category = self._classify_size(fundamentals)

        # Classify growth style
        growth_style = self._classify_growth_style(fundamentals)

        # Calculate context scores
        growth_score = self._calculate_growth_score(fundamentals)
        quality_score = self._calculate_quality_score(fundamentals)
        leverage_score = self._calculate_leverage_score(fundamentals)
        profitability_score = self._calculate_profitability_score(fundamentals)

        return CompanyContext(
            sector=sector,
            size_category=size_category,
            growth_style=growth_style,
            growth_score=growth_score,
            quality_score=quality_score,
            leverage_score=leverage_score,
            profitability_score=profitability_score,
            fundamentals=fundamentals,
        )

    def _classify_sector(self, symbol: Optional[str]) -> SectorType:
        """
        Classify company sector using yfinance.

        Args:
            symbol: Stock ticker

        Returns:
            SectorType enum
        """
        if symbol is None:
            return SectorType.UNKNOWN

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get("sector", "").lower()

            # Map yfinance sectors to our enum
            sector_map = {
                "technology": SectorType.TECHNOLOGY,
                "financial services": SectorType.FINANCE,
                "healthcare": SectorType.HEALTHCARE,
                "energy": SectorType.ENERGY,
                "consumer cyclical": SectorType.CONSUMER_CYCLICAL,
                "consumer defensive": SectorType.CONSUMER_DEFENSIVE,
                "industrials": SectorType.INDUSTRIALS,
                "basic materials": SectorType.MATERIALS,
                "utilities": SectorType.UTILITIES,
                "real estate": SectorType.REAL_ESTATE,
                "communication services": SectorType.COMMUNICATION,
            }

            for key, value in sector_map.items():
                if key in sector:
                    return value

            return SectorType.UNKNOWN

        except Exception:
            return SectorType.UNKNOWN

    def _classify_size(self, fundamentals: Dict[str, Optional[float]]) -> SizeCategory:
        """
        Classify company size from market cap.

        Note: Market cap is fetched separately as it's not in fundamentals dict.
        For now, we use P/E and other metrics as proxies.
        """
        # TODO: Add market_cap to fundamentals dict
        # For now, use heuristics based on liquidity metrics

        current_ratio = fundamentals.get("current_ratio")
        quick_ratio = fundamentals.get("quick_ratio")

        # Larger companies tend to have more stable ratios
        if current_ratio and quick_ratio:
            if current_ratio > 2.0 and quick_ratio > 1.5:
                return SizeCategory.MEGA_CAP
            elif current_ratio > 1.5 and quick_ratio > 1.0:
                return SizeCategory.LARGE_CAP
            elif current_ratio > 1.0:
                return SizeCategory.MID_CAP
            else:
                return SizeCategory.SMALL_CAP

        # Default to mid-cap if no data
        return SizeCategory.MID_CAP

    def _classify_growth_style(
        self, fundamentals: Dict[str, Optional[float]]
    ) -> GrowthStyle:
        """
        Classify growth vs value style based on P/E and growth rates.

        Args:
            fundamentals: Fundamental ratios

        Returns:
            GrowthStyle enum
        """
        pe_ratio = fundamentals.get("pe_ratio")
        revenue_growth = fundamentals.get("revenue_growth_qoq", 0)
        earnings_growth = fundamentals.get("earnings_growth_qoq", 0)

        # No P/E data - default to balanced
        if pe_ratio is None:
            return GrowthStyle.BALANCED

        # Growth metrics
        avg_growth = ((revenue_growth or 0) + (earnings_growth or 0)) / 2

        # Classify based on P/E and growth
        if pe_ratio > 40 and avg_growth > 20:
            return GrowthStyle.HYPER_GROWTH
        elif pe_ratio > 25 and avg_growth > 10:
            return GrowthStyle.GROWTH
        elif 15 <= pe_ratio <= 25:
            return GrowthStyle.BALANCED
        elif pe_ratio < 15 and avg_growth < 10:
            return GrowthStyle.VALUE
        elif pe_ratio < 10:
            return GrowthStyle.DEEP_VALUE

        return GrowthStyle.BALANCED

    def _calculate_growth_score(
        self, fundamentals: Dict[str, Optional[float]]
    ) -> float:
        """
        Calculate growth score (0 = deep value, 1 = hyper growth).

        Based on:
        - P/E ratio (valuation)
        - Revenue growth QoQ
        - Earnings growth QoQ
        """
        pe_ratio = fundamentals.get("pe_ratio", 20)
        revenue_growth = fundamentals.get("revenue_growth_qoq", 0)
        earnings_growth = fundamentals.get("earnings_growth_qoq", 0)

        # Normalize P/E (0-50 range typical)
        pe_score = np.clip((pe_ratio or 20) / 50, 0, 1) if pe_ratio else 0.5

        # Normalize growth rates (-20% to +40% range)
        revenue_score = np.clip((revenue_growth or 0 + 20) / 60, 0, 1)
        earnings_score = np.clip((earnings_growth or 0 + 20) / 60, 0, 1)

        # Weighted average (P/E is most important for growth/value)
        growth_score = 0.5 * pe_score + 0.25 * revenue_score + 0.25 * earnings_score

        return float(growth_score)

    def _calculate_quality_score(
        self, fundamentals: Dict[str, Optional[float]]
    ) -> float:
        """
        Calculate quality score (0 = low quality, 1 = high quality).

        Based on:
        - ROE (return on equity)
        - ROA (return on assets)
        - Profit margin
        - Current ratio
        """
        roe = fundamentals.get("roe", 0)
        roa = fundamentals.get("roa", 0)
        profit_margin = fundamentals.get("profit_margin", 0)
        current_ratio = fundamentals.get("current_ratio", 1.0)

        # Normalize metrics
        roe_score = np.clip((roe or 0) / 30, 0, 1)  # 30% ROE is excellent
        roa_score = np.clip((roa or 0) / 15, 0, 1)  # 15% ROA is excellent
        margin_score = np.clip(
            (profit_margin or 0) / 25, 0, 1
        )  # 25% margin is excellent
        liquidity_score = np.clip(
            (current_ratio or 0) / 3, 0, 1
        )  # 3.0 current ratio is excellent

        # Weighted average (profitability more important than liquidity)
        quality_score = (
            0.3 * roe_score
            + 0.3 * roa_score
            + 0.3 * margin_score
            + 0.1 * liquidity_score
        )

        return float(quality_score)

    def _calculate_leverage_score(
        self, fundamentals: Dict[str, Optional[float]]
    ) -> float:
        """
        Calculate leverage score (0 = debt-heavy, 1 = cash-rich).

        Based on:
        - Debt-to-equity ratio (lower is better)
        - Current ratio (higher is better)
        - Quick ratio (higher is better)
        """
        debt_to_equity = fundamentals.get("debt_to_equity", 1.0)
        current_ratio = fundamentals.get("current_ratio", 1.0)
        quick_ratio = fundamentals.get("quick_ratio", 0.8)

        # Normalize debt-to-equity (lower is better, invert)
        # 0.0 D/E = score 1.0, 2.0 D/E = score 0.0
        debt_score = np.clip(1.0 - (debt_to_equity or 1.0) / 2.0, 0, 1)

        # Normalize liquidity ratios (higher is better)
        current_score = np.clip((current_ratio or 1.0) / 3.0, 0, 1)
        quick_score = np.clip((quick_ratio or 0.8) / 2.0, 0, 1)

        # Weighted average (debt ratio most important)
        leverage_score = 0.5 * debt_score + 0.25 * current_score + 0.25 * quick_score

        return float(leverage_score)

    def _calculate_profitability_score(
        self, fundamentals: Dict[str, Optional[float]]
    ) -> float:
        """
        Calculate profitability score (0 = unprofitable, 1 = highly profitable).

        Based on:
        - Profit margin
        - ROE
        - FCF yield
        """
        profit_margin = fundamentals.get("profit_margin", 0)
        roe = fundamentals.get("roe", 0)
        fcf_yield = fundamentals.get("fcf_yield", 0)

        # Normalize metrics
        margin_score = np.clip((profit_margin or 0) / 25, 0, 1)
        roe_score = np.clip((roe or 0) / 30, 0, 1)
        fcf_score = np.clip((fcf_yield or 0) / 10, 0, 1)  # 10% FCF yield is excellent

        # Weighted average
        profitability_score = 0.4 * margin_score + 0.4 * roe_score + 0.2 * fcf_score

        return float(profitability_score)

    def describe_context(self, context: CompanyContext) -> str:
        """
        Create human-readable description of company context.

        Args:
            context: CompanyContext object

        Returns:
            String description
        """
        sector_name = context.sector.name.replace("_", " ").title()
        size_name = context.size_category.name.replace("_", "-").title()
        style_name = context.growth_style.name.replace("_", " ").title()

        description = f"""
Company Context Analysis:
========================
Sector: {sector_name}
Size: {size_name}
Style: {style_name}

Context Scores (0-1):
- Growth: {context.growth_score:.3f} ({self._interpret_score(context.growth_score, "growth")})
- Quality: {context.quality_score:.3f} ({self._interpret_score(context.quality_score, "quality")})
- Leverage: {context.leverage_score:.3f} ({self._interpret_score(context.leverage_score, "leverage")})
- Profitability: {context.profitability_score:.3f} ({self._interpret_score(context.profitability_score, "profit")})

Interpretation:
This is a {size_name} {sector_name} company with {style_name} characteristics.
The model will use this context to interpret price patterns appropriately.
"""
        return description

    def _interpret_score(self, score: float, score_type: str) -> str:
        """Interpret a context score as human-readable text."""
        if score_type == "growth":
            if score > 0.7:
                return "High Growth"
            elif score > 0.4:
                return "Moderate Growth"
            else:
                return "Value"
        elif score_type == "quality":
            if score > 0.7:
                return "High Quality"
            elif score > 0.4:
                return "Good Quality"
            else:
                return "Low Quality"
        elif score_type == "leverage":
            if score > 0.7:
                return "Cash Rich"
            elif score > 0.4:
                return "Balanced"
            else:
                return "Debt Heavy"
        elif score_type == "profit":
            if score > 0.7:
                return "Highly Profitable"
            elif score > 0.4:
                return "Profitable"
            else:
                return "Unprofitable"

        return "Unknown"


if __name__ == "__main__":
    # Quick test
    from data.fundamentals.fundamental_collector import FundamentalCollector

    collector = FundamentalCollector()
    embedder = FundamentalEmbedding()

    # Test on AAPL (should be mega-cap tech growth)
    print("Testing on AAPL...")
    fundamentals = collector.collect_fundamentals("AAPL")
    context = embedder.create_context(fundamentals, "AAPL")

    print(embedder.describe_context(context))
    print(f"\nContext vector shape: {context.to_vector().shape}")
    print(f"Context vector (first 10): {context.to_vector()[:10]}")
