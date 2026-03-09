#!/usr/bin/env python3
"""
Market Data API - Real-time quotes for trading execution

Provides bid/ask/last prices needed for trade execution.
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/market-data", tags=["market-data"])

# Global data provider (initialized by server.py startup)
_data_provider = None


def set_provider(data_provider):
    """Set global data provider (called by server.py on startup)"""
    global _data_provider
    _data_provider = data_provider
    logger.info("✅ Market data API: provider configured")


@router.get("/quote/{symbol}")
async def get_quote(symbol: str) -> Dict:
    """
    Get real-time quote for a symbol (bid/ask/last prices).

    Used by PIM trade execution to get current prices before placing orders.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')

    Returns:
        {
            "symbol": "AAPL",
            "price": 175.50,    # Last traded price
            "bid": 175.48,      # Current bid (best buy price)
            "ask": 175.52,      # Current ask (best sell price)
            "volume": 1234567,  # Today's volume
            "timestamp": "2026-03-03T10:30:00",
            "provider": "tradestation"
        }

    Example:
        GET /api/v1/market-data/quote/AAPL
    """
    if _data_provider is None:
        raise HTTPException(
            status_code=503, detail="Data provider not initialized"
        )

    try:
        # Use the MultiProviderFetcher's get_quote method
        # It automatically tries providers in order with fallback
        quote = _data_provider.get_quote(symbol)

        if quote is None:
            raise HTTPException(
                status_code=404,
                detail=f"No quote data available for {symbol}"
            )

        # Ensure price is available (bid/ask/last)
        if (
            (quote.get("price") is None or quote.get("price") <= 0)
            and (quote.get("bid") is None or quote.get("bid") <= 0)
            and (quote.get("ask") is None or quote.get("ask") <= 0)
        ):
            raise HTTPException(
                status_code=404,
                detail=f"No valid price data for {symbol} (all price fields are null/zero)"
            )

        return quote

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch quote for {symbol}: {str(e)}"
        )
