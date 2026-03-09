"""
Training API endpoints for historical data extraction
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from ..utils.api_credentials import APICredentials
from ..providers.alphavantage_client import AlphaVantageClient
from ..features.feature_extractor import FeatureExtractor
from config.dimensions import DIMS  # type: ignore[import]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["training"])

# Global data provider injected by server.py startup (MultiProviderFetcher)
_data_provider = None


def set_provider(data_provider) -> None:
    """Set global data provider (called by server.py on startup)."""
    global _data_provider
    _data_provider = data_provider
    logger.info(
        f"✅ Training router: data provider set to {data_provider.__class__.__name__}"
    )


def _require_provider():
    """Return the global provider or raise 503."""
    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")
    return _data_provider


@router.get("/features/{symbol}")
async def get_training_features(
    symbol: str,
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    provider: str = Query(
        "tradestation",
        description="Data provider (tradestation, yfinance) - defaults to TradeStation, falls back to yfinance only if unavailable",
    ),
    interval: str = Query("1d", description="Bar interval (30s, 1m, 5m, 30m, 1h, 1d)"),
    timeframes: Optional[str] = Query(
        "30s,1m,5m,30m", description="Comma-separated timeframes"
    ),
):
    """
    Extract historical features for training

    Returns feature vectors for each bar in the date range (dimension from config).
    Supports both yfinance (daily) and TradeStation (intraday).
    """
    try:
        logger.info(f"Training feature request: {symbol} from {start} to {end}")

        # Parse dates
        if start:
            start_date = datetime.strptime(start, "%Y-%m-%d")
        else:
            start_date = datetime.now() - timedelta(days=30)

        if end:
            end_date = datetime.strptime(end, "%Y-%m-%d")
        else:
            end_date = datetime.now()

        # Initialize AlphaVantage client for fundamentals/news
        av_client = AlphaVantageClient()

        # Use injected MultiProviderFetcher — routing (TS → yfinance) is handled internally
        data_provider = _require_provider()
        logger.info(
            f"Training data via {data_provider.get_name() if hasattr(data_provider, 'get_name') else data_provider.__class__.__name__}"
            f" (provider hint={provider}, interval={interval})"
        )

        # Initialize feature extractor
        extractor = FeatureExtractor(
            alpha_vantage_client=av_client,
            cache_fundamentals=True,
            cache_news=True,
            enable_senvec=True,  # SenVec integrated in FinColl
            data_provider=data_provider,
        )

        # Fetch OHLCV data — extra 100 days before start_date for technical indicators
        lookback_start = start_date - timedelta(days=100)
        ohlcv_data = data_provider.get_historical_bars(
            symbol=symbol,
            start_date=lookback_start,
            end_date=end_date,
            interval=interval,
        )

        if ohlcv_data is None or ohlcv_data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for {symbol} in date range"
            )

        logger.info(f"Fetched {len(ohlcv_data)} bars for {symbol}")

        # Extract features for each bar
        features_list = []
        for idx, timestamp in enumerate(ohlcv_data.index):
            # Need enough history for technical indicators (use expanding window)
            window_data = ohlcv_data.iloc[: idx + 1]

            if len(window_data) < 50:  # Need minimum history for indicators
                continue

            # Only include timestamps within the requested date range
            # Normalize to tz-naive for comparison (start_date/end_date are always tz-naive)
            if hasattr(timestamp, "to_pydatetime"):
                ts_naive = timestamp.to_pydatetime()
                # Remove timezone if present
                if hasattr(ts_naive, "replace") and ts_naive.tzinfo is not None:
                    ts_naive = ts_naive.replace(tzinfo=None)
            elif hasattr(timestamp, "replace") and hasattr(timestamp, "tzinfo"):
                ts_naive = (
                    timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
                )
            else:
                # Already tz-naive datetime
                ts_naive = timestamp

            if ts_naive < start_date or ts_naive > end_date:
                continue

            try:
                feature_vector = extractor.extract_features(
                    ohlcv_data=window_data, symbol=symbol, timestamp=timestamp
                )
                features_list.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "features": feature_vector.tolist(),
                    }
                )
            except Exception as e:
                logger.warning(f"Feature extraction failed for {timestamp}: {e}")
                continue

        if not features_list:
            raise HTTPException(
                status_code=500, detail="Feature extraction failed for all bars"
            )

        logger.info(f"Extracted {len(features_list)} feature vectors for {symbol}")

        return {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "num_samples": len(features_list),
            "features": features_list,
            "metadata": {
                "feature_dimensions": DIMS.fincoll_total,
                "interval": interval,
                "provider": data_provider.get_name(),
                "senvec_enabled": extractor.enable_senvec,
                "bars_fetched": len(ohlcv_data),
            },
        }

    except Exception as e:
        logger.error(f"Training feature extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def get_batch_training_features(request: dict):
    """
    Batch feature extraction for multiple symbols

    Expects JSON with:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "start": "2024-01-01",
        "end": "2024-12-31",
        "timeframes": "30s,1m,5m,30m"
    }
    """
    try:
        symbols = request.get("symbols", [])
        logger.info(f"Batch training feature request for {len(symbols)} symbols")

        results = []
        for symbol in symbols:
            # TODO: Process each symbol
            results.append({"symbol": symbol, "status": "not_implemented"})

        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "symbols_requested": len(symbols),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Batch training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
