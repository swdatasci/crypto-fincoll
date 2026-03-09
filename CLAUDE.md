# CLAUDE.md - FinColl Prediction API

This file provides guidance to Claude Code when working with the fincoll repository.

---

## 📚 Documentation via Semantic Search

**IMPORTANT**: Detailed documentation for this repository has been archived and is available via semantic search.

Instead of reading extensive markdown files, use the Caelum semantic search system:

```bash
# CLI search
uv run python scripts/semantic_search.py "your query here"

# MCP tool (in Claude sessions)
search_caelum_knowledge("your query here")
```

**Coverage**: 12,481+ vectors across 1,039+ markdown files from all repos

**Why Use Semantic Search**:
- ✅ Faster than reading multiple docs
- ✅ Finds related context across all repositories
- ✅ Always includes recent changes and updates
- ✅ Answers specific questions directly

**This CLAUDE.md**: Contains essential quick-start info and current status only. For detailed architecture, implementation guides, troubleshooting, and historical context, use semantic search.

---


## MANDATORY: Model Changelog

**Every time a velocity model is trained and evaluated, update `CHANGELOG_MODEL.md` before
swapping the production checkpoint.** Include:
1. Checkpoint filename and date
2. W&B run URL
3. What changed and why
4. A/B metrics table (before vs after) — val loss, per-timeframe MAE, directional accuracy
5. Book notes (plain-language description of the improvement)

See [`CHANGELOG_MODEL.md`](./CHANGELOG_MODEL.md) for the full history.

---

## ✅ TradeStation Rate Limiting (Fixed 2026-01)

**Incident (2025-12-31)**: TradeStation called about API hammering. All three issues have been resolved:

1. ✅ **API call logging** - `_make_request()` logs every request with URL, status, and elapsed ms
2. ✅ **Rate limiting** - Redis-based shared limiter (`fincoll/utils/rate_limiter.py`): 250 req/5min (accounts), 30 req/min (quotes); coordinates across all services
3. ✅ **Exponential backoff** - Retries with 1s, 2s, 4s delays on server errors and rate limit (429) responses

Both `TradeStationProvider` and the deprecated `TradeStationDataCollector` use `get_shared_limiter()`.

**See**: `/home/rford/caelum/caelum-supersystem/PassiveIncomeMaximizer/docs/runbooks/TRADESTATION_API_INCIDENT_2025-12-31.md`

## IMPORTANT: Architecture Overview (Updated 2026-02-17)

FinColl uses a **class-based architecture** - NO training scripts. All logic is in the class hierarchy.

**Output format**: Multi-timeframe profit velocity bests (NOT continuous 1-100 day horizons)
- Per timeframe (1min, 5min, 15min, 1hour, daily):
  - Best long velocity (% return per bar)
  - Best short velocity
  - Bars until peak/valley
- Ranked by |velocity| to find best opportunity

**Key relationship**:
- **finvec** = Model training engine, velocity model, core ML code
- **fincoll** = API server, data providers, feature extraction

## ⚠️ Layer 2 RL Documentation Moved (2026-02-15)

**IMPORTANT**: Layer 2 RL agent documentation has been **relocated to PassiveIncomeMaximizer**.

**Why**: Layer 2 RL agents (MomentumAgent, MacroAgent, RiskAgent, etc.) are **PIM-exclusive features**, not part of fincoll's data service responsibilities.

**New Location**:
```
/home/rford/caelum/caelum-supersystem/PassiveIncomeMaximizer/docs/architecture/layer2/
├── LAYER2_README.md
├── LAYER2_OUTPUT_SPEC.md
├── LAYER2_IMPLEMENTATION_GUIDE.md
└── LAYER2_QUICK_REFERENCE.md
```

**What fincoll provides**: 414D labeled feature vectors + velocity predictions
**What PIM Layer 2 does**: Filters predictions using RL agents before Layer 1 evaluation

**See**: `DOCUMENT_OWNERSHIP_ANALYSIS.md` for architectural reasoning

---

## Class Hierarchy (The ACTUAL Architecture)

### Training & Inference Engine (`finvec/engine/`)

| File | Class | Purpose |
|------|-------|---------|
| `fincoll_engine.py` | `FinCollEngine` | Unified TRAIN/INFER engine - drives batch iteration |
| `fincoll_client.py` | `FinCollClient` | API client for FinColl (fetches 336D features) |
| `velocity_trainer.py` | `VelocityTrainer` | Training loop, validation, checkpointing |

### Velocity Prediction (`finvec/models/heads/`)

| File | Class | Purpose |
|------|-------|---------|
| `velocity_heads.py` | `MultiTimeframeVelocityHead` | Predicts velocities per timeframe |
| `velocity_heads.py` | `VelocityPredictionHeads` | Complete prediction heads module |

### Velocity Targets (`finvec/data/`)

| File | Function/Class | Purpose |
|------|----------------|---------|
| `velocity_targets.py` | `compute_velocity_targets()` | Compute best velocity from future prices |
| `velocity_targets.py` | `VelocityTargetDataset` | Dataset class for training |
| `velocity_dataset.py` | `create_data_loaders()` | Create train/val/test loaders |

### Model (`finvec/models/`)

| File | Class | Purpose |
|------|-------|---------|
| `simple_velocity_model.py` | `SimpleVelocityModel` | MLP for velocity prediction |

## FinCollEngine Usage

```python
from engine.fincoll_engine import FinCollEngine, EngineConfig, EngineMode

# Training mode
config = EngineConfig(
    mode=EngineMode.TRAIN,
    batch_size=10,
    device="cuda",
    timeframes=('1min', '5min', '15min', '1hour', 'daily'),
)
engine = FinCollEngine(
    config=config,
    symbol_scanner=scanner,  # SymbolScanner instance
    model=model,             # VelocityModel
    feature_extractor=extractor,
    data_provider=provider,
    training_db=db,
)

# Run training epoch
results = engine.run_epoch()

# Inference mode
config.mode = EngineMode.INFER
results = engine.process_batch()
```

## API Endpoints

### Current (Legacy - needs migration)
- `POST /api/v1/inference/predict/{symbol}` - Uses old PredictionEngine
- `GET /api/v1/inference/velocity/{symbol}` - Velocity predictions (proxies to velocity server)

### Planned (After Migration)
- `POST /api/v1/predict/{symbol}` - Unified velocity prediction
- `POST /api/v1/batch` - Batch velocity predictions
- `POST /api/v1/training/features/{symbol}` - Get 336D features for training
- `POST /api/v1/training/samples` - Store training samples

## Configuration

### Feature Dimensions
- **361D** feature vectors (technical + fundamental + sentiment + futures)
- Extracted by `FeatureExtractor` class in `fincoll/features/feature_extractor.py`

### Timeframes
```python
STANDARD_TIMEFRAMES = [
    ('1min', 60, 30),      # Up to 30 minutes lookahead
    ('5min', 300, 24),     # Up to 2 hours
    ('15min', 900, 16),    # Up to 4 hours
    ('1hour', 3600, 8),    # Up to 8 hours
    ('daily', 86400, 5),   # Up to 5 days
]
```

### Velocity Target Format
For each timeframe, model predicts:
```python
{
    'long_velocity': float,    # % return per bar (positive)
    'long_bars': int,          # Bars until peak
    'short_velocity': float,   # % return per bar (negative)
    'short_bars': int,         # Bars until valley
    'confidence': float,       # 0-1 confidence score
}
```

## Data Providers

| Provider | File | Notes |
|----------|------|-------|
| TradeStation | `providers/tradestation_provider.py` | Primary, OAuth token at `~/.tradestation_token.json` |
| Alpaca | `providers/alpaca_provider.py` | Fallback |
| yfinance | `providers/yfinance_provider.py` | Being removed — 15-min delayed, unreliable |

## PM2 Deployment (on 10.32.3.27)

Current process name: `fincoll`

```bash
# Restart fincoll (10.32.3.27 is localhost)
pm2 restart fincoll
```

## Port Reference

| Service | Port | Notes |
|---------|------|-------|
| FinColl API | 8001 | Main prediction API |
| Velocity Inference | 5001 | Internal velocity model server |
| SenVec | 18000 | Sentiment service |
| PIM Express | 5000 | PassiveIncomeMaximizer |
| PIM Engine | 5002 | Python PIM service |

## Current Status (2026-02-17)

### Working
- All core modules import successfully
- `FinCollEngine`, `VelocityTrainer`, `FinCollClient` built
- `velocity_dataset`, `simple_velocity_model` exist
- Velocity heads architecture complete
- TradeStation rate limiting, logging, and backoff fully implemented
- Full test suite: **164 passed, 0 failed, 0 skipped**
- All mock provider E2E tests passing (TradeStation, Alpaca, Public)

### Needs Work
1. **FinCollEngine not wired to server** - server.py still uses old PredictionEngine
2. **Training API endpoints missing** - `/api/v1/training/features` not implemented
3. **Deprecated collector** - `futures_features.py` still imports `TradeStationDataCollector`; migrate to `TradeStationTradingProvider`
4. **FastAPI deprecations** - `on_event` → lifespan, `regex` → `pattern` in `bars.py:44`

### Migration Path
1. Wire FinCollEngine to server.py (replace PredictionEngine)
2. Add training endpoints to server
3. Migrate futures_features.py off deprecated TradeStationDataCollector
4. Test training flow
5. Test inference flow

## Dependencies

- **finvec**: Engine, models, training code (at `../finvec/`)
- **TradeStation**: OAuth token for market data
- **CUDA**: GPU for training/inference (optional)

---
**Updated**: 2026-02-17
