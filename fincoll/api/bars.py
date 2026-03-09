#!/usr/bin/env python3
"""
Intraday Bars API - Multi-timeframe OHLCV data endpoint

Provides 1min, 5min, 15min, 30min, 60min intraday bars for velocity training.
Uses AlphaVantage Premium (150 calls/min) or TradeStation as fallback.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

# Import InfluxDB cache (replaces deprecated SQLite BarsCache)
from ..storage.influxdb_cache import get_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bars", tags=["bars"])

# Global data providers (initialized by server.py startup)
_data_provider = None
_alphavantage_client = None


def _get_influx_cache():
    """Lazy initialization of InfluxDB cache"""
    return get_cache()


def set_providers(data_provider, alphavantage_client=None):
    """Set global data providers (called by server.py on startup)"""
    global _data_provider, _alphavantage_client
    _data_provider = data_provider
    _alphavantage_client = alphavantage_client
    # InfluxDB cache initialized lazily via _get_influx_cache()
    logger.info("✅ Data providers initialized (InfluxDB cache will be used)")


@router.get("/{symbol}/history")
async def get_bars_history(
    symbol: str,
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    interval: str = Query(default="1d", description="Bar interval: 1d"),
) -> Dict:
    """
    Get historical OHLCV bars between two dates.

    Used by the backtesting system to fetch real price data for a date range.
    Returns daily bars (open/high/low/close/volume) with timestamps.

    Example:
        GET /api/v1/bars/AAPL/history?start_date=2024-01-01&end_date=2024-12-31&interval=1d
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format")

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    if _data_provider is None:
        raise HTTPException(status_code=503, detail="Data provider not initialized")

    try:
        df = _data_provider.get_historical_bars(
            symbol=symbol,
            interval=interval,
            start_date=start_dt,
            end_date=end_dt,
        )
    except Exception as e:
        logger.error(f"Error fetching historical bars for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for {symbol} from {start_date} to {end_date}",
        )

    # Normalise column names — providers may return 'timestamp' or 'date'
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)

    return {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "bars_returned": len(df),
        "source": "tradestation",
        "data": df[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(
            orient="records"
        ),
    }


@router.get("/{symbol}")
async def get_bars(
    symbol: str,
    interval: str = Query(
        ...,
        description="Time interval: 1m, 15m, 1h (hour), 1d (daily), 1mo (monthly)",
        pattern="^(1m|15m|1h|1d|1mo)$",
    ),
    bars: int = Query(
        default=100,
        ge=1,
        le=11700,
        description="Number of bars to return (max 11700 for extended history)",
    ),
) -> Dict:
    """
    Get historical OHLCV bars at specified timeframe

    **Premium AlphaVantage**: 150 calls/min for real-time intraday data
    **TradeStation Fallback**: Up to 1 year intraday history

    Examples:
        - GET /api/v1/bars/AAPL?interval=1m&bars=100   # Last 100 1-min bars
        - GET /api/v1/bars/MSFT?interval=15m&bars=500  # Last 500 15-min bars
        - GET /api/v1/bars/GOOGL?interval=1d&bars=252  # Last 252 daily bars
        - GET /api/v1/bars/SPY?interval=1mo&bars=60    # Last 60 monthly bars

    Returns:
        {
            "symbol": "AAPL",
            "interval": "1m",
            "bars_requested": 100,
            "bars_returned": 100,
            "source": "alphavantage" or "tradestation",
            "data": [
                {
                    "timestamp": "2025-12-19T15:59:00",
                    "open": 195.23,
                    "high": 195.45,
                    "low": 195.20,
                    "close": 195.42,
                    "volume": 125430
                },
                ...
            ]
        }
    """
    try:
        # Map interval format to AlphaVantage/TradeStation conventions
        interval_map = {
            "1m": "1min",
            "15m": "15min",
            "1h": "60min",
            "1d": "Daily",
            "1mo": "Monthly",
        }

        av_interval = interval_map.get(interval)
        if not av_interval:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval: {interval}. Must be 1m, 15m, 1h, 1d, or 1mo",
            )

        # STEP 1: Check InfluxDB cache first (avoid API calls!)
        influx_cache = _get_influx_cache()

        if influx_cache.enabled:
            # Calculate lookback window for cache query
            end_date = datetime.now()
            if interval == "1d":
                start_date = end_date - timedelta(
                    days=bars * 2
                )  # Extra margin for daily
            elif interval == "1mo":
                start_date = end_date - timedelta(days=bars * 60)  # ~2 months per bar
            else:
                # For intraday, estimate bars needed
                start_date = end_date - timedelta(
                    days=7
                )  # 1 week lookback for intraday

            cached_df = influx_cache.get_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                source="any",  # Accept any source from cache
            )

            if cached_df is not None and len(cached_df) >= bars:
                cached_df = cached_df.tail(bars)  # Return last N bars
                logger.info(
                    f"InfluxDB cache HIT: {symbol} {interval} - returning {len(cached_df)} cached bars"
                )
                return {
                    "symbol": symbol,
                    "interval": interval,
                    "bars_requested": bars,
                    "bars_returned": len(cached_df),
                    "source": "cached",
                    "data": cached_df.to_dict(orient="records"),
                }

        # STEP 2: Cache miss - fetch from API
        logger.info(f"Cache MISS: {symbol} {interval} - fetching from API")

        # Try AlphaVantage Premium first (best for intraday real-time)
        # Use AlphaVantage for: 1m, 15m, 1h (intraday)
        # Use TradeStation for: 1d, 1mo (daily/monthly)
        if _alphavantage_client and interval in ["1m", "15m", "1h"]:
            try:
                logger.info(
                    f"Fetching {symbol} {interval} bars from AlphaVantage Premium (150/min)"
                )

                # AlphaVantage uses 'full' for extended history, 'compact' for last 100
                outputsize = "full" if bars > 100 else "compact"

                data = _alphavantage_client.get_intraday_bars(
                    symbol=symbol, interval=av_interval, outputsize=outputsize
                )

                # Parse AlphaVantage response
                df = _parse_alphavantage_response(data, av_interval)

                if df is not None and len(df) > 0:
                    # Store in InfluxDB cache before returning
                    if influx_cache.enabled:
                        influx_cache.store_bars(
                            symbol, df, interval=interval, source="alphavantage"
                        )

                    # Return last N bars
                    df = df.tail(bars)

                    return {
                        "symbol": symbol,
                        "interval": interval,
                        "bars_requested": bars,
                        "bars_returned": len(df),
                        "source": "alphavantage",
                        "data": df.to_dict(orient="records"),
                    }
                else:
                    logger.warning(
                        f"AlphaVantage returned empty data for {symbol}, trying TradeStation"
                    )

            except Exception as av_error:
                logger.warning(
                    f"AlphaVantage error for {symbol}: {av_error}, trying TradeStation"
                )

        # Fallback to TradeStation (or primary for daily/monthly)
        if _data_provider:
            logger.info(f"Fetching {symbol} {interval} bars from TradeStation")

            # Calculate date range based on interval
            end_date = datetime.now()
            if interval == "1d":
                # Daily: 5 years back
                start_date = end_date - timedelta(days=365 * 5)
            elif interval == "1mo":
                # Monthly: 10 years back
                start_date = end_date - timedelta(days=365 * 10)
            else:
                # Intraday fallback: 1 year back (TradeStation limit)
                start_date = end_date - timedelta(days=365)

            df = _data_provider.get_historical_bars(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                interval=interval,
            )

            if df is not None and len(df) > 0:
                # Ensure timestamp column
                if "timestamp" not in df.columns:
                    df["timestamp"] = df.index

                # Store in InfluxDB cache before returning
                if influx_cache.enabled:
                    influx_cache.store_bars(
                        symbol, df, interval=interval, source="tradestation"
                    )

                # Return last N bars
                df = df.tail(bars)

                return {
                    "symbol": symbol,
                    "interval": interval,
                    "bars_requested": bars,
                    "bars_returned": len(df),
                    "source": "tradestation",
                    "data": df.to_dict(orient="records"),
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for {symbol} at {interval} interval",
                )

        # No data providers available
        raise HTTPException(
            status_code=503,
            detail="No data providers available (AlphaVantage and TradeStation both unavailable)",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching bars for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _parse_alphavantage_response(data: Dict, interval: str) -> Optional[pd.DataFrame]:
    """
    Parse AlphaVantage TIME_SERIES_INTRADAY response to DataFrame

    Args:
        data: AlphaVantage API response
        interval: Time interval (1min, 5min, etc.)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        # AlphaVantage uses key like "Time Series (1min)"
        time_series_key = f"Time Series ({interval})"

        if time_series_key not in data:
            logger.error(
                f"AlphaVantage response missing '{time_series_key}' key. Keys: {data.keys()}"
            )
            return None

        time_series = data[time_series_key]

        # Convert to DataFrame
        records = []
        required_keys = ["1. open", "2. high", "3. low", "4. close", "5. volume"]

        for timestamp_str, values in time_series.items():
            # Validate expected keys are present to avoid KeyError if AlphaVantage changes format
            if not all(key in values for key in required_keys):
                logger.warning(
                    "Skipping AlphaVantage bar with missing keys at %s. "
                    "Expected keys: %s, received keys: %s",
                    timestamp_str,
                    required_keys,
                    list(values.keys()),
                )
                continue

            try:
                records.append(
                    {
                        "timestamp": pd.to_datetime(timestamp_str),
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"]),
                    }
                )
            except (TypeError, ValueError) as parse_err:
                logger.warning(
                    "Skipping AlphaVantage bar at %s due to parse error: %s. Raw values: %s",
                    timestamp_str,
                    parse_err,
                    values,
                )
                continue

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp")  # Oldest first
        df = df.reset_index(drop=True)

        logger.info(
            f"Parsed {len(df)} bars from AlphaVantage ({df['timestamp'].min()} to {df['timestamp'].max()})"
        )

        return df

    except Exception as e:
        logger.error(f"Error parsing AlphaVantage response: {e}", exc_info=True)
        return None
