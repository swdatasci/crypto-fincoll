# Sector ETF Backfill Guide

**Date**: 2026-03-05  
**Issue**: Insufficient sector ETF historical data causing warnings during feature extraction

## Problem

The feature extractor requires 30 days of historical data for sector ETFs to calculate relative performance features (alpha vs sector). When the system has only been running for a few days, these warnings appear:

```
WARNING - Insufficient sector ETF data for GILD (XLV)
```

## Solution

Two fixes implemented:

### 1. Crypto Symbol Filtering for Finnhub (✅ Completed)

**File**: `fincoll/features/feature_extractor.py:2711`

**Change**: Added crypto symbol detection to skip Finnhub fundamentals API calls for cryptocurrency symbols.

```python
# Skip crypto symbols - they don't have earnings, insider trades, or analyst coverage
# Crypto identifiers: -USD, -USDT, -USDC, etc.
if '-USD' in symbol or symbol.endswith(('USD', 'USDT', 'USDC', 'BTC', 'ETH')):
    return np.zeros(15, dtype=np.float32)
```

**Why**: Cryptocurrencies don't have:
- Earnings reports
- Insider transactions
- Analyst recommendations

**Impact**: 
- Saves Finnhub API rate limit quota
- Reduces log noise
- No change in functionality (was already returning zeros)

### 2. Sector ETF Historical Data Backfill (✅ Script Created)

**Script**: `scripts/backfill_sector_etf_data.py`

**Purpose**: Fetch historical data for all 11 sector ETFs used in feature extraction.

**Sector ETFs**:
```python
XLK   # Technology
XLF   # Financials
XLV   # Healthcare
XLE   # Energy
XLY   # Consumer Cyclical
XLP   # Consumer Defensive
XLI   # Industrials
XLU   # Utilities
XLRE  # Real Estate
XLB   # Materials
XLC   # Communication Services
```

## Usage

### Backfill All Sector ETFs (Recommended)

```bash
cd /home/rford/caelum/caelum-supersystem/fincoll

# Backfill 1 year of data using TradeStation (default)
python scripts/backfill_sector_etf_data.py --days 365

# Backfill 2 years of data
python scripts/backfill_sector_etf_data.py --days 730

# Use Alpaca as provider
python scripts/backfill_sector_etf_data.py --days 365 --provider alpaca
```

### Backfill Specific ETFs

```bash
# Only backfill Healthcare, Technology, and Financials
python scripts/backfill_sector_etf_data.py --etfs XLV XLK XLF --days 180
```

### Expected Output

```
📅 Backfilling 11 sector ETFs from 2025-03-05 to 2026-03-05
📊 Provider: tradestation, Interval: 1d
📈 Fetching XLK...
  ✅ XLK: 252 bars fetched
  📊 Date range: 2025-03-05 to 2026-03-05
  💰 Latest close: $234.56
...

================================================================
📊 BACKFILL SUMMARY
================================================================
✅ Successful: 11/11
❌ Failed: 0/11
📈 Total bars fetched: 2,772

✅ Success: XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLU, XLRE, XLB, XLC

💡 Note: Data is cached in the FeatureExtractor class-level cache
   and will be reused across all feature extraction calls.
```

## How It Works

1. **Script fetches historical data** from TradeStation/Alpaca for each sector ETF
2. **Data is returned to the provider** which may have its own caching
3. **FeatureExtractor caches** sector ETF data at the class level (`_sector_etf_cache`)
4. **Cache is shared** across all FeatureExtractor instances (thread-safe)
5. **TTL**: Cache entries use the same TTL as SPY data (configured in feature_extractor.py)

## Cache Details

**Location**: In-memory class-level cache in `FeatureExtractor`

**Cache Key Format**: `{etf_symbol}_{YYYY-MM-DD-HH}` (hourly buckets)

**Thread Safety**: Protected by `_sector_etf_cache_lock`

**TTL**: Controlled by `self.spy_ttl` (default from config)

## When to Run

- **Initial setup**: Run once when first deploying the system
- **After extended downtime**: If the system was offline for more than the cache TTL
- **Before training**: To ensure full feature coverage for training data
- **After data gaps**: If you notice warnings about insufficient sector ETF data

## Monitoring

Check for these warnings in logs:

```bash
# Check for sector ETF warnings
pm2 logs fincoll | grep "Insufficient sector ETF"

# Check for crypto Finnhub warnings (should be gone after fix)
pm2 logs fincoll | grep "Failed to get Finnhub fundamentals.*-USD"
```

## Next Steps

1. **Run the backfill script** with desired lookback period
2. **Restart fincoll service** to pick up the crypto filtering change:
   ```bash
   pm2 restart fincoll
   ```
3. **Monitor logs** to verify warnings are reduced

## Technical Notes

### Why 30 days?

The `_get_sector_etf_data()` method requests `lookback=30` days plus a 30-day buffer for market holidays:

```python
start_date = timestamp - timedelta(days=lookback + 30)  # 60 days total
```

### Why not database storage?

Sector ETF data is cached in-memory because:
- **Low cardinality**: Only 11 ETFs to track
- **High reuse**: Same ETF data used across many symbols
- **Simple refresh**: Easy to re-fetch if cache expires
- **Performance**: Faster than database queries

### Graceful Degradation

Even without sector ETF data, the system works:
- Returns zero features for relative performance (f216-f218)
- Sector one-hot encoding (f205-f215) still works
- No impact on other feature groups

## References

- Feature Extractor: `fincoll/features/feature_extractor.py:2165` (`_get_sector_etf_data`)
- Sector Features: `fincoll/features/feature_extractor.py:2290` (`_extract_sector_features`)
- Backfill Script: `scripts/backfill_sector_etf_data.py`
