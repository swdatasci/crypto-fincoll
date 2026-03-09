# FinColl - Financial Data Collection Service

Centralized microservice for financial data collection serving both FinVec training and PIM inference.

## Purpose

Extract data collection logic from FinVec into a dedicated service to:
- ✅ Avoid duplication (finvec + PIM both need same data)
- ✅ Ensure consistency (training and inference use identical data sources)
- ✅ Provide data provenance (track what data was used for each prediction)
- ✅ Centralize credentials (TradeStation, Alpha Vantage, SenVec)

## Status

**Current**: ✅ Initial repository created, data collection code copied from FinVec

**Next Steps**:
1. Add FastAPI dependencies
2. Create REST API endpoints
3. Test standalone operation
4. Integrate with FinVec (branch + PR)
5. Integrate with PIM

## Copied from FinVec

```
fincoll/
├── providers/       # TradeStation, Alpha Vantage clients
├── collectors/      # Data collection orchestration
├── features/        # Feature extraction (dimension from config)
├── sources/         # Data source adapters
└── utils/           # API credentials, helpers
```

See `~/caelum/ss/PassiveIncomeMaximizer/FINCOLL_SERVICE_DESIGN.md` for complete architecture.
