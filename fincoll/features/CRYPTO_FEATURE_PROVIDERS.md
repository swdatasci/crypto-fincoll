# Crypto Feature Providers - Multi-Provider Round-Robin Strategy

## Overview

The crypto feature extraction system uses a **multi-provider round-robin strategy** to maximize combined free rate limits across multiple crypto data APIs.

**Goal**: Achieve 60+ calls/minute using free tiers of 3 providers instead of being limited by a single provider's rate limit.

## Provider Configuration

### 1. CoinGecko (Primary) ✅ ACTIVE
- **Free Tier**: 50 calls/min (Demo plan)
- **Monthly Limit**: 10,000 calls/month
- **API Key**: `CG-p5ZWnSsvU1AArBuf9s2iFf4A`
- **Coverage**: 6,000+ cryptocurrencies
- **Unique Data**: Community engagement, developer activity, liquidity metrics
- **Priority**: 1 (Primary provider)

**Endpoints Used**:
- `/coins/{id}` - Detailed coin data
- `/global` - Global market data
- `/coins/categories` - Category classification
- `/simple/price` - Real-time prices

### 2. CryptoCompare (Backup) ⏳ PENDING
- **Free Tier**: 100,000 calls/month (~138 calls/hour)
- **Coverage**: Wide spectrum of cryptocurrencies
- **Unique Data**: Customizable data streams, order books, liquidity
- **Priority**: 2 (Backup provider)
- **Status**: API key needed (register at cryptocompare.com)

**Why**: Very generous free tier (100k calls/month) makes it ideal for overflow traffic.

### 3. CoinMarketCap (Fallback) ⏳ PENDING
- **Free Tier**: 10,000 calls/month (~13 calls/hour)
- **Coverage**: Thousands of cryptocurrencies
- **Unique Data**: Real-time prices, market cap, volume, historical data
- **Priority**: 3 (Fallback provider)
- **Status**: API key needed (register at coinmarketcap.com)

**Why**: High overlap with CoinGecko makes it perfect for critical fallback.

## Round-Robin Strategy

### Request Flow

```
┌─────────────────────────────────────────┐
│  Feature Extraction Request             │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│  Try Primary: CoinGecko (50 calls/min)  │
└──────────────┬──────────────────────────┘
               │
               ├─ SUCCESS ──> Return data
               │
               v (RATE LIMIT or FAILURE)
┌─────────────────────────────────────────┐
│  Try Backup: CryptoCompare (138/hour)   │
└──────────────┬──────────────────────────┘
               │
               ├─ SUCCESS ──> Return data
               │
               v (RATE LIMIT or FAILURE)
┌─────────────────────────────────────────┐
│  Try Fallback: CoinMarketCap (13/hour)  │
└──────────────┬──────────────────────────┘
               │
               ├─ SUCCESS ──> Return data
               │
               v (ALL FAILED)
┌─────────────────────────────────────────┐
│  Return zeros (graceful degradation)    │
└─────────────────────────────────────────┘
```

### Combined Capacity

| Provider | Rate Limit | Status | Contribution |
|----------|-----------|--------|--------------|
| CoinGecko | 50 calls/min | ✅ Active | Primary load |
| CryptoCompare | 138 calls/hour | ⏳ Pending | Overflow (~2.3/min) |
| CoinMarketCap | 13 calls/hour | ⏳ Pending | Critical fallback (~0.2/min) |
| **TOTAL** | **~52 calls/min** | - | **Combined free** |

**Note**: When all providers are active, effective capacity is ~52 calls/min with intelligent distribution.

## Feature Groups (156D)

### Implemented (Using CoinGecko)

| Feature Group | Dimensions | Provider | Status |
|---------------|-----------|----------|--------|
| Token Metadata | 18D | CoinGecko | ✅ Partial (10D active, 8D placeholder) |
| Crypto Market | 9D | CoinGecko | ✅ Complete |
| Categorization | 31D | CoinGecko | ✅ Partial (10D active, 21D placeholder) |
| Fundamentals | 25D | CoinGecko | ✅ Partial (7D active, 18D placeholder) |

### Placeholder (Awaiting Implementation)

| Feature Group | Dimensions | Suggested Provider | Notes |
|---------------|-----------|-------------------|-------|
| Onchain Pools | 14D | Dune Analytics, Flipside | Requires onchain data APIs |
| Exchanges & Derivatives | 11D | CryptoCompare, CoinMarketCap | Exchange API integrations |
| Public Treasuries | 6D | BitcoinTreasuries.org | MicroStrategy, Tesla holdings |
| Dynamic Clustering | 12D | Computed | Correlation matrix across top 100 coins |
| Macro Factors | 25D | Federal Reserve API, Bloomberg | VIX, DXY, interest rates |
| Sentiment | 5D | Alternative.me, LunarCrush | Fear & Greed Index, social sentiment |

## Implementation Details

### Multi-Provider Fallback

```python
from fincoll.features.crypto_market_features import MultiProviderCryptoFeatureExtractor

# Initialize with all providers
extractor = MultiProviderCryptoFeatureExtractor(
    cache_ttl=300,  # 5 minutes
    enable_cache=True,
)

# Extract features (automatically uses round-robin)
features = extractor.extract("bitcoin")  # 156D vector
```

### Provider Selection Logic

1. **Primary Provider**: Always try CoinGecko first (50 calls/min)
2. **Failure Detection**: Track consecutive failures per provider
3. **Round-Robin State**: Update current provider on success
4. **Fallback Cascade**: Try each provider in priority order
5. **Graceful Degradation**: Return zeros if all providers fail

### Failure Tracking

```python
# Provider failure counts (auto-reset on success)
self.provider_failures = {
    DataProvider.COINGECKO: 0,      # Resets on successful call
    DataProvider.CRYPTOCOMPARE: 0,
    DataProvider.COINMARKETCAP: 0,
}

# After 3 consecutive failures, provider is temporarily skipped
# (could be enhanced with time-based cooldown)
```

## Adding New Providers

### Step 1: Register for API Key

- CryptoCompare: https://www.cryptocompare.com/cryptopian/api-keys
- CoinMarketCap: https://coinmarketcap.com/api/

### Step 2: Update Configuration

Edit `crypto_market_features.py`:

```python
class ProviderConfig:
    CRYPTOCOMPARE = {
        "base_url": "https://min-api.cryptocompare.com/data",
        "api_key": "YOUR_API_KEY_HERE",  # ← Update this
        "rate_limit": 138,
        "priority": 2,
    }
```

### Step 3: Implement Provider-Specific Fetch

```python
def _fetch_cryptocompare(
    self, endpoint: str, params: Optional[Dict]
) -> Optional[Dict]:
    """Fetch from CryptoCompare API."""
    config = ProviderConfig.CRYPTOCOMPARE
    url = f"{config['base_url']}/{endpoint}"
    
    if params is None:
        params = {}
    params["api_key"] = config["api_key"]
    
    response = self.session.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()
```

### Step 4: Test Provider

```python
# Test CryptoCompare connection
from fincoll.features.crypto_market_features import MultiProviderCryptoFeatureExtractor, DataProvider

extractor = MultiProviderCryptoFeatureExtractor()
data = extractor._fetch_with_fallback(
    endpoint="price",
    params={"fsym": "BTC", "tsyms": "USD"},
    provider=DataProvider.CRYPTOCOMPARE,
)
print(data)  # Should return BTC price
```

## Rate Limit Management

### Current Strategy (Simple)

- **No rate limiting enforced** (providers handle their own limits)
- **Graceful fallback** when rate limit hit (429 error)
- **Provider rotation** spreads load naturally

### Future Enhancements

1. **Redis-based rate limiter** (shared across all FinColl instances)
   ```python
   from fincoll.utils.rate_limiter import get_shared_limiter
   
   limiter = get_shared_limiter("coingecko", 50, 60)  # 50 req/min
   ```

2. **Smart provider selection** (choose least-used provider)
   ```python
   def _select_best_provider(self) -> DataProvider:
       """Select provider with most available capacity."""
       # Check Redis for current usage counts
       # Return provider with lowest usage
   ```

3. **Exponential backoff** on repeated failures
   ```python
   if self.provider_failures[provider] >= 3:
       backoff_time = 2 ** self.provider_failures[provider]
       time.sleep(backoff_time)
   ```

## Monitoring & Observability

### Logs

All provider interactions are logged:

```
INFO: Fetching from CoinGecko: /coins/bitcoin
WARNING: Provider coingecko failed: Rate limit exceeded (429)
INFO: Falling back to CryptoCompare
INFO: CryptoCompare success: /coins/bitcoin
```

### Metrics (Future)

Track provider performance:

- `crypto_provider_requests_total{provider="coingecko", status="success"}`
- `crypto_provider_requests_total{provider="coingecko", status="rate_limit"}`
- `crypto_provider_latency_seconds{provider="coingecko"}`

## Cost Optimization

### Current Cost: $0/month

- CoinGecko Demo: FREE (10k calls/month)
- CryptoCompare: FREE (100k calls/month) - when added
- CoinMarketCap: FREE (10k calls/month) - when added

**Combined free capacity: 120,000 calls/month**

### Upgrade Path (If Needed)

Only upgrade if backtests prove 416D features beat 245D baseline:

| Provider | Free Tier | Paid Tier | Cost |
|----------|-----------|-----------|------|
| CoinGecko | 10k/month | 500k/month (Analyst) | $103/month |
| CryptoCompare | 100k/month | Unlimited (Pro) | $50/month |
| CoinMarketCap | 10k/month | 30k/month (Startup) | $29/month |

**Strategy**: Start with all free tiers, only upgrade if ROI is proven.

## Testing

### Manual Test

```bash
cd crypto/fincoll
source .venv/bin/activate

python -c "
from fincoll.features.crypto_market_features import MultiProviderCryptoFeatureExtractor
extractor = MultiProviderCryptoFeatureExtractor()
features = extractor.extract('bitcoin')
print(f'Extracted {len(features)} features')
print(f'First 10: {features[:10]}')
"
```

### Integration Test

```python
import pytest
from fincoll.features.crypto_market_features import MultiProviderCryptoFeatureExtractor

def test_multi_provider_fallback():
    """Test round-robin provider fallback."""
    extractor = MultiProviderCryptoFeatureExtractor()
    
    # Should successfully extract features
    features = extractor.extract("bitcoin")
    assert len(features) == 156
    assert not all(f == 0.0 for f in features)  # At least some features populated
```

## FAQ

**Q: Why not just use CoinGecko Pro?**
A: We want to validate that 416D features improve prediction accuracy before spending $103/month. Multi-provider strategy gives us more free capacity to test.

**Q: What happens if all providers fail?**
A: The extractor returns a 156D vector of zeros (graceful degradation). The prediction pipeline continues but with reduced signal.

**Q: Can we add more providers?**
A: Yes! Follow the "Adding New Providers" section. Candidates: Messari, The Graph, Nomics (if still active).

**Q: How does caching interact with multiple providers?**
A: Cache keys are provider-agnostic (just coin ID + feature type). If CoinGecko fails and CryptoCompare succeeds, the CryptoCompare data is cached for next request.

**Q: Won't different providers return different data formats?**
A: Yes! That's why `_fetch_from_provider()` includes provider-specific adapters to normalize responses to a common format.

---

**Last Updated**: 2026-03-08  
**Status**: CoinGecko active, CryptoCompare/CoinMarketCap pending API keys
