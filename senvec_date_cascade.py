#!/usr/bin/env python3
"""
Date-based cascading fallback for SenVec features

Drop-in replacement for senvec_integration.py that implements:
"if I ask for the 8th, use 7th, then 6th, and so on"

To use in training:
1. Copy to /home/rford/caelum/ss/fincoll/fincoll/utils/
2. Rename existing: mv senvec_integration.py senvec_integration_old.py
3. Rename this: mv senvec_date_cascade.py senvec_integration.py
4. Train!
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import logging
import requests

# SenVec cache directory
SENVEC_CACHE_DIR = Path("/home/rford/caelum/data/senvec_features")
SENVEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SENVEC_URL = "http://10.32.3.27:18000"

logger = logging.getLogger(__name__)


def _get_cache_path(symbol: str, date_str: str) -> Path:
    """Get cache file path: /data/senvec_features/YYYYMMDD/SYMBOL.npy"""
    date_dir = SENVEC_CACHE_DIR / date_str
    date_dir.mkdir(exist_ok=True)
    return date_dir / f"{symbol}.npy"


def _save_to_cache(symbol: str, date_str: str, features: np.ndarray):
    """Save 49D features to cache"""
    try:
        cache_path = _get_cache_path(symbol, date_str)
        np.save(cache_path, features)
        logger.debug(f"✓ Cached {symbol} on {date_str}")
    except Exception as e:
        logger.warning(f"✗ Cache save failed for {symbol}: {e}")


def _load_from_cache_cascading(symbol: str, date_str: str, max_lookback_days: int = 30):
    """
    Load features with cascading fallback
    
    If date_str=20250608 not found:
      → Check 20250607
      → Check 20250606
      → ... (up to 30 days back)
    
    Returns: (features, actual_date) or (None, None)
    """
    try:
        target_date = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        logger.error(f"Invalid date: {date_str}")
        return None, None
    
    for days_back in range(max_lookback_days + 1):
        check_date = target_date - timedelta(days=days_back)
        check_date_str = check_date.strftime("%Y%m%d")
        cache_path = _get_cache_path(symbol, check_date_str)
        
        if cache_path.exists():
            try:
                features = np.load(cache_path)
                if features.shape != (49,):
                    logger.warning(f"Wrong shape in cache for {symbol}: {features.shape}")
                    continue
                
                if days_back > 0:
                    logger.info(f"{symbol}: using {check_date_str} (requested {date_str}, {days_back} days back)")
                else:
                    logger.debug(f"{symbol}: exact match {date_str}")
                
                return features, check_date_str
            except Exception as e:
                logger.warning(f"Cache read failed {cache_path}: {e}")
                continue
    
    return None, None


def get_senvec_features(
    symbol: str,
    date: str = None,
    fallback_zeros: bool = True,
    max_lookback_days: int = 30
) -> np.ndarray:
    """
    Get 49D SenVec features with date-based cascading fallback
    
    Workflow:
    1. Try SenVec API for requested date
    2. If success & non-zero → cache it, return
    3. If fail/zero → cascade through cached dates (X-1, X-2, ...)
    4. Last resort → zeros
    
    Args:
        symbol: Stock ticker (AAPL, MSFT, etc.)
        date: YYYY-MM-DD format (default: today)
        fallback_zeros: Return zeros if no data found
        max_lookback_days: How far back to search (default: 30)
    
    Returns:
        49D float32 array
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    date_normalized = date.replace("-", "")
    
    # Try SenVec API
    try:
        response = requests.get(
            f"{SENVEC_URL}/features/{symbol}/compact",
            params={"date": date},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            features_dict = data.get("features", {})
            
            # Convert dict to 49D array
            features = np.zeros(49, dtype=np.float32)
            for key, value in features_dict.items():
                try:
                    idx = int(key.replace("f", "")) - 187  # f187-f235 → 0-48
                    if 0 <= idx < 49:
                        features[idx] = float(value)
                except:
                    continue
            
            # Only cache if non-zero (real data)
            if not np.allclose(features, 0):
                _save_to_cache(symbol, date_normalized, features)
                logger.debug(f"{symbol}: fetched & cached from API")
                return features
            else:
                logger.warning(f"{symbol}: API returned zeros, checking cache")
        else:
            logger.warning(f"{symbol}: API status {response.status_code}")
            
    except Exception as e:
        logger.error(f"{symbol}: API error: {e}")
    
    # API failed or returned zeros → try cache cascade
    cached_features, cached_date = _load_from_cache_cascading(symbol, date_normalized, max_lookback_days)
    
    if cached_features is not None:
        return cached_features.astype(np.float32)
    
    # No cache found → last resort
    if fallback_zeros:
        logger.warning(f"{symbol}: no cache (searched {max_lookback_days} days), returning zeros")
        return np.zeros(49, dtype=np.float32)
    else:
        raise ValueError(f"No data for {symbol} on {date}")


def get_senvec_features_batch(
    symbols: list,
    date: str = None,
    fallback_zeros: bool = True,
    max_lookback_days: int = 30
) -> dict:
    """Get features for multiple symbols with cascading"""
    results = {}
    for symbol in symbols:
        try:
            results[symbol] = get_senvec_features(symbol, date, fallback_zeros, max_lookback_days)
        except Exception as e:
            logger.error(f"{symbol}: {e}")
            if fallback_zeros:
                results[symbol] = np.zeros(49, dtype=np.float32)
    return results


def warm_cache_for_date(symbols: list, date: str = None) -> dict:
    """
    Fetch and cache features for multiple symbols
    
    Run this during market hours via cron to build cache:
    */15 * * * * python senvec_date_cascade.py warm AAPL MSFT GOOGL ...
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    status = {'success': [], 'zeros': [], 'failed': []}
    
    for symbol in symbols:
        try:
            features = get_senvec_features(symbol, date=date, fallback_zeros=False)
            if not np.allclose(features, 0):
                status['success'].append(symbol)
                print(f"✓ {symbol}")
            else:
                status['zeros'].append(symbol)
                print(f"⚠ {symbol} (zeros)")
        except Exception as e:
            status['failed'].append(symbol)
            print(f"✗ {symbol}: {e}")
    
    print(f"\nCached: {len(status['success'])}/{len(symbols)} symbols")
    return status


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == "warm":
        # Warm cache mode
        symbols = sys.argv[2:] if len(sys.argv) > 2 else [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "DIS"
        ]
        warm_cache_for_date(symbols)
    else:
        # Test mode
        print("Testing date-cascading:")
        features = get_senvec_features("AAPL", date="2025-06-08")
        print(f"AAPL features shape: {features.shape}, sum: {features.sum():.2f}")
