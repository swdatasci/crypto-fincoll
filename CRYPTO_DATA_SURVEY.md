# Crypto Data Provider Survey - Feature Expansion Opportunities

**Purpose**: Survey available data from all crypto providers to identify valuable features for the 391D → ???D expansion

**Current Status**: 391D (245D base + 20D cross-market + 156D crypto-extended - 30D senvec)

---

## Available Providers & API Keys

| Provider | Status | Rate Limit | API Key Status |
|----------|--------|-----------|----------------|
| **CoinGecko** | ✅ Active | 50 calls/min | ✅ `CG-p5ZWnSsvU1AArBuf9s2iFf4A` |
| **CryptoCompare** | 🟡 Ready | 138 calls/hour | ✅ `834706540691a5ab0d4fc00982620aa1331a3920c44017cfb405034c992c8d1d` |
| **CoinMarketCap** | 🟡 Ready | 13 calls/hour | ✅ `431f16c330514634836f517471510b73` |
| **Binance** | 🟡 Ready | 1200 req/min (weight) | ✅ API key + secret available |

**Combined Free Capacity**: ~52 calls/min + Binance (1200 weight/min)

---

## Data Categories & Provider Coverage

### 1. Price & Market Data (Already Implemented - 9D)

**Current Coverage**:
- Real-time price, 24h change
- Market cap, volume, liquidity
- All-time high/low distance

**Provider Coverage**:
| Feature | CoinGecko | CryptoCompare | CoinMarketCap | Binance |
|---------|-----------|---------------|---------------|---------|
| Real-time price | ✅ | ✅ | ✅ | ✅ Best (trading data) |
| 24h volume | ✅ | ✅ | ✅ | ✅ |
| Market cap | ✅ | ✅ | ✅ | ✅ |
| Circulating supply | ✅ | ✅ | ✅ | ✅ |
| Total supply | ✅ | ✅ | ✅ | ✅ |

**Expansion Opportunities**:
- ❌ Already comprehensive
- Consider: Binance order book depth (bid/ask spread, volume at levels)

---

### 2. Token Metadata (Partial - 10D/18D implemented)

**Current Coverage** (10D active):
- Total supply, circulating supply, max supply
- Genesis date, contract address exists
- Is stablecoin, has wrapped version
- Public interest score, Coingecko rank

**Placeholder** (8D):
- Token standard (ERC-20, BEP-20, etc.)
- Multi-chain deployment count
- Layer 1 vs Layer 2 classification

**Provider Coverage**:
| Feature | CoinGecko | CryptoCompare | CoinMarketCap | Binance |
|---------|-----------|---------------|---------------|---------|
| Token standard | ✅ platforms field | ❌ | ✅ | Limited |
| Multi-chain | ✅ platforms count | ❌ | ✅ | ❌ |
| Layer classification | ✅ categories | Partial | ✅ tags | ❌ |
| Contract address | ✅ | ✅ | ✅ | ❌ |
| Wrapped tokens | ✅ | ❌ | Partial | ❌ |

**Expansion Opportunities**:
- ✅ **Implement 8D placeholders** using CoinGecko `/coins/{id}` platforms field
- Consider: Token holder count (requires onchain APIs)

---

### 3. On-Chain Data (Not Implemented - 14D placeholder)

**Current Status**: All zeros (requires onchain APIs)

**Available from Binance**:
| Feature | Binance API | Value |
|---------|-------------|-------|
| Trading pairs | ✅ `/api/v3/exchangeInfo` | Count active pairs |
| Liquidity pools | ❌ (CEX, not DeFi) | N/A |
| Staking APY | ✅ `/sapi/v1/staking/position` | User-specific |
| Lending rates | ✅ `/sapi/v1/lending/project/list` | Available |

**Available from CoinGecko**:
- DeFi locked value (TVL)
- Top liquidity pools

**NOT Available** (requires dedicated onchain APIs):
- Transaction count (24h)
- Unique addresses (24h)
- Average transaction value
- Contract interactions

**Expansion Opportunities**:
- ✅ **Implement Binance lending rates** (3D): Current rate, 7d avg, 30d avg
- ✅ **Implement Binance trading pair count** (1D): Liquidity proxy
- ⚠️ **Onchain metrics** require Dune Analytics or Flipside (paid APIs)
- Consider: CoinGecko TVL for DeFi tokens (2D): Total locked, 24h change

**Recommendation**: Implement Binance features first (4D), defer true onchain (10D) until after backtesting proves value

---

### 4. Exchange & Derivatives Data (Partial - 11D placeholder)

**Current Status**: All zeros

**Available from CryptoCompare**:
| Feature | CryptoCompare API | Endpoint |
|---------|-------------------|----------|
| Exchange volume breakdown | ✅ | `/v2/histoday` with `aggregate` param |
| Top exchanges | ✅ | `/v2/top/exchanges` |
| Exchange count | ✅ | Count from top exchanges |

**Available from Binance**:
| Feature | Binance API | Value |
|---------|-------------|-------|
| **Futures open interest** | ✅ `/fapi/v1/openInterest` | 🔥 CRITICAL for momentum |
| **Futures funding rate** | ✅ `/fapi/v1/fundingRate` | 🔥 Sentiment indicator |
| **Futures long/short ratio** | ✅ `/fapi/v1/globalLongShortAccountRatio` | 🔥 Crowding metric |
| 24h ticker stats | ✅ `/api/v3/ticker/24hr` | Volume, price change |

**Expansion Opportunities**:
- ✅✅✅ **HIGHEST PRIORITY: Binance Futures metrics** (6D):
  1. Open interest (raw + % change 24h)
  2. Funding rate (current + 8h avg)
  3. Long/short ratio (current + trend)
- ✅ **Exchange diversity** (3D): Count of exchanges listing, volume concentration (Herfindahl index), largest exchange % share
- ✅ **CryptoCompare volume breakdown** (2D): Spot vs derivatives ratio, CEX vs DEX volume

**Recommendation**: **IMPLEMENT BINANCE FUTURES FIRST** - these are alpha-generating features for crypto

---

### 5. Sentiment & Social (Partial - 5D placeholder)

**Current Status**: All zeros (no crypto sentiment service)

**Available from CoinGecko**:
| Feature | CoinGecko API | Field |
|---------|---------------|-------|
| Community score | ✅ `/coins/{id}` | `community_data.facebook_likes`, `twitter_followers` |
| Developer activity | ✅ | `developer_data.forks`, `stars`, `subscribers` |
| Sentiment votes | ✅ | `sentiment_votes_up_percentage` |

**Available from CryptoCompare**:
| Feature | CryptoCompare API | Value |
|---------|-------------------|-------|
| News sentiment | ✅ `/v2/news/` | Categorized news with sentiment |
| Social stats | ✅ `/v2/social/coin/latest` | Reddit, Twitter metrics |

**NOT Available** (requires paid APIs):
- Alternative.me Fear & Greed Index (free but separate API)
- LunarCrush social sentiment (paid)

**Expansion Opportunities**:
- ✅ **CoinGecko community** (3D): Twitter followers (normalized), Reddit subscribers, sentiment vote %
- ✅ **CoinGecko developer** (2D): GitHub stars, forks (activity proxy)
- ✅ **CryptoCompare news sentiment** (3D): Positive count (24h), negative count (24h), sentiment ratio
- Consider: Alternative.me Fear & Greed Index (free API, 1D)

**Recommendation**: Implement CoinGecko + CryptoCompare features (8D total), defer paid social APIs

---

### 6. Categorization & Classification (Partial - 10D/31D implemented)

**Current Coverage** (10D active):
- Top 10 category one-hot encoding
- Is DeFi, Is meme coin

**Placeholder** (21D):
- Sector classification (L1, DeFi, NFT, etc.)
- Use case tags
- Ecosystem membership

**Available from CoinGecko**:
- `categories` field: Full list of categories per coin
- DeFi classification built-in

**Available from CoinMarketCap**:
- Detailed tags and categories
- Platform classification (Ethereum, BSC, Polygon, etc.)

**Expansion Opportunities**:
- ✅ **Implement placeholder 21D** using CoinGecko categories
  - Sector: L1 (1D), L2 (1D), DeFi (1D), NFT (1D), Gaming (1D), Metaverse (1D)
  - Use case: Payment (1D), Store of value (1D), Smart contracts (1D), Privacy (1D)
  - Ecosystem: Ethereum (1D), BSC (1D), Solana (1D), Polygon (1D), Avalanche (1D), Cosmos (1D)
  - Extras: CEX token (1D), Oracle (1D), Governance (1D), Wrapped (1D), Stablecoin already covered (1D)

**Recommendation**: Full implementation possible with CoinGecko alone (21D)

---

### 7. Fundamentals (Partial - 7D/25D implemented)

**Current Coverage** (7D active):
- Market cap dominance
- Volume/market cap ratio
- Price change (24h, 7d, 30d)

**Placeholder** (18D):
- ATH/ATL metrics
- Historical volatility
- Correlation with BTC
- Sharpe ratio

**Available from Binance**:
| Feature | Binance API | Value |
|---------|-------------|-------|
| Historical OHLCV | ✅ `/api/v3/klines` | Calculate volatility, Sharpe |
| 24h ticker | ✅ `/api/v3/ticker/24hr` | High, low, volume |

**Available from CoinGecko**:
- ATH date and price
- ATL date and price
- Price change % (multiple timeframes)

**Available from CryptoCompare**:
| Feature | CryptoCompare API | Value |
|---------|-------------------|-------|
| Historical daily | ✅ `/v2/histoday` | OHLCV for volatility |
| Moving averages | ✅ Can compute from histoday | SMA, EMA |

**Expansion Opportunities**:
- ✅ **ATH/ATL metrics** (4D): % from ATH, % from ATL, days since ATH, days since ATL (CoinGecko)
- ✅ **Volatility** (3D): 7d, 30d, 90d realized volatility (Binance klines)
- ✅ **BTC correlation** (2D): 30d rolling correlation, 90d correlation (compute from Binance)
- ✅ **Sharpe ratio** (2D): 30d, 90d Sharpe (compute from Binance)
- ✅ **Moving averages** (4D): Price vs 20D MA, 50D MA, 100D MA, 200D MA (Binance klines)
- ✅ **RSI** (3D): 14d RSI, RSI divergence from BTC, overbought/oversold flag

**Recommendation**: Full 18D implementation possible with Binance historical data + CoinGecko ATH/ATL

---

### 8. Public Treasuries (Not Implemented - 6D placeholder)

**Current Status**: All zeros

**Available Data**:
| Source | Data | API |
|--------|------|-----|
| BitcoinTreasuries.org | MicroStrategy, Tesla holdings | ❌ Web scraping only |
| CoinGecko | Some treasury data | ✅ Limited |

**Expansion Opportunities**:
- ⚠️ **Low priority** - requires web scraping or manual updates
- Consider: Bitcoin treasury holdings (MicroStrategy, Tesla, El Salvador) as static data
- Consider: Ethereum treasury holdings (major protocols)

**Recommendation**: Defer until after backtesting core features

---

### 9. Clustering & Correlations (Not Implemented - 12D placeholder)

**Current Status**: All zeros

**Compute Requirements**:
- Requires correlation matrix across top 100 coins
- Can be computed from Binance historical data
- Computationally expensive (daily batch job)

**Features to Compute**:
1. Cluster assignment (1-10) based on correlation
2. Correlation with BTC (0-1)
3. Correlation with ETH (0-1)
4. Average correlation with top 10 coins
5. Correlation stability (30d std dev)
6. Beta to crypto market (systematic risk)
7. Idiosyncratic volatility (unsystematic risk)
8. Sector correlation (correlation with same-category coins)
9. Leading/lagging indicator (Granger causality flag)
10. Liquidity tier (1-5 based on volume)
11. Market regime sensitivity (correlation during bull/bear)
12. Cross-exchange arbitrage opportunity

**Expansion Opportunities**:
- ✅ **Implement compute pipeline** using Binance klines for top 100 coins
- ⚠️ Requires daily batch job (not real-time)
- Consider: Pre-compute and store in Redis with 24h TTL

**Recommendation**: Medium priority - implement after core Binance features

---

### 10. Macro Factors (Not Implemented - 25D placeholder)

**Current Status**: All zeros (shared with equities cross-market features)

**Available Data**:
| Source | Data | API |
|--------|------|-----|
| Federal Reserve | Interest rates, M2 supply | ✅ FRED API (free) |
| yfinance | VIX, DXY, SPY, gold | ✅ Already implemented for equities |
| Binance | BTC dominance | ✅ Can compute |

**Overlap with Cross-Market Features** (20D):
- SPY, QQQ, DIA, IWM returns (4D)
- VIX level (1D)
- ES, NQ, YM futures (3D)
- Market regime (3D)

**Crypto-Specific Macro** (not in cross-market):
1. BTC dominance (%)
2. BTC dominance 7d change
3. Total crypto market cap
4. Total crypto market cap 7d change
5. DeFi TVL (total locked value)
6. DeFi TVL 7d change
7. Stablecoin market cap (USDT + USDC + BUSD)
8. Stablecoin 7d flow (net minting/burning)

**Expansion Opportunities**:
- ✅ **BTC dominance metrics** (2D): Current %, 7d change (CoinGecko `/global`)
- ✅ **Total crypto market cap** (2D): Current, 7d change (CoinGecko `/global`)
- ✅ **DeFi TVL** (2D): Total, 7d change (CoinGecko `/global/decentralized_finance_defi`)
- ✅ **Stablecoin metrics** (2D): Total market cap, 7d change (CoinGecko `/global`)
- Consider: FRED API for traditional macro (interest rates, M2)

**Recommendation**: Implement crypto-specific macro (8D), leverage existing cross-market features for traditional macro

---

## Summary: Expansion Roadmap

### High Priority (Alpha-Generating) - **28D**

1. **Binance Futures** (6D): Open interest, funding rate, long/short ratio
2. **Fundamentals - Technical** (18D): ATH/ATL, volatility, correlations, Sharpe, MAs, RSI
3. **Exchange Diversity** (3D): Exchange count, concentration, largest %
4. **CryptoCompare Volume** (2D): Spot/derivatives, CEX/DEX ratio

**Total New Features**: 29D (391D → **420D**)

---

### Medium Priority (Valuable Context) - **25D**

5. **Token Metadata Completion** (8D): Token standard, multi-chain, layer classification
6. **Categorization Completion** (21D): Full sector/use case/ecosystem classification
7. **Sentiment** (8D): CoinGecko community + dev, CryptoCompare news
8. **Crypto Macro** (8D): BTC dominance, total cap, DeFi TVL, stablecoin metrics
9. **Binance Lending** (4D): Rates, pair count, TVL proxy

**Total with Medium**: 54D (391D → **445D**)

---

### Low Priority (Defer Until Proven) - **18D**

10. **Clustering** (12D): Correlation matrix, beta, clusters
11. **Onchain** (10D remaining): Transaction counts, unique addresses (requires Dune/Flipside)
12. **Public Treasuries** (6D): MicroStrategy, Tesla holdings (web scraping)

**Total with Low**: 72D (391D → **463D**)

---

## Implementation Phases

### Phase 1: Quick Wins (High Priority) - 1 week
- Create CryptoCompare HTTP service (port 9003)
- Create Binance HTTP service (port 9004)
- Implement Binance futures features (6D)
- Implement fundamentals using Binance klines (18D)
- Implement exchange diversity using CryptoCompare (3D)
- **Deliverable**: 391D → 420D, ready for backtesting

### Phase 2: Context Enrichment (Medium Priority) - 1 week
- Complete token metadata (8D)
- Complete categorization (21D)
- Implement sentiment (8D)
- Implement crypto macro (8D)
- Implement Binance lending (4D)
- **Deliverable**: 420D → 474D

### Phase 3: Advanced Features (Low Priority) - 2 weeks
- Implement clustering pipeline (12D)
- Integrate Dune Analytics for onchain (10D)
- Scrape public treasury data (6D)
- **Deliverable**: 474D → 502D

---

## API Service Architecture

### New HTTP Services Required

| Service | Port | Provider | Status |
|---------|------|----------|--------|
| CryptoCompare | 9003 | CryptoCompare API | 🟡 To create |
| Binance | 9004 | Binance API | 🟡 To create |
| CoinGecko | 9002 | CoinGecko API | 🟡 To create |
| YFinance | 9001 | yfinance | ✅ Created |

**Pattern**: Each provider is a centralized HTTP service with:
- FastAPI + uvicorn
- Redis caching (TTL-based)
- Rate limiting (Redis-backed)
- Round-robin fallback support
- PM2 managed

---

## Cost Analysis

### Current Cost: $0/month

- CoinGecko: FREE (10k calls/month)
- CryptoCompare: FREE (100k calls/month)
- CoinMarketCap: FREE (10k calls/month)
- Binance: FREE (1200 weight/min)

**Combined capacity**: 120k calls/month + Binance (unlimited for spot data)

### If Upgrade Needed (Only After Proven ROI)

| Provider | Upgrade | Cost | Benefit |
|----------|---------|------|---------|
| CoinGecko | Analyst | $103/month | 500k calls/month |
| CryptoCompare | Pro | $50/month | Unlimited |
| Binance | N/A | $0 | Already generous |
| Dune Analytics | Plus | $390/month | Onchain data |

**Strategy**: Stay on free tiers until backtesting proves 420D+ beats 391D baseline

---

## Validation Metrics

Before implementing each phase, validate with:

1. **Backtest Sharpe Ratio**: Does 420D beat 391D?
2. **Feature Importance**: Which new features matter?
3. **Training Time**: Can we still train in reasonable time?
4. **Inference Latency**: Does 420D slow down predictions?
5. **ROI**: Do results justify potential API costs?

**Decision Rule**: Only proceed to next phase if Sharpe improves by ≥10%

---

**Last Updated**: 2026-03-08  
**Status**: Survey complete, ready for Phase 1 implementation
