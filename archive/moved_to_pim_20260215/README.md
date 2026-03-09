# Archived: Layer 2 Documentation Moved to PIM

**Date**: 2026-02-15
**Reason**: Architectural cleanup - Layer 2 is a PIM feature, not fincoll

---

## Files Archived

These 5 files have been **moved to PassiveIncomeMaximizer**:

1. `LAYER2_README.md` → `PassiveIncomeMaximizer/docs/architecture/layer2/LAYER2_README.md`
2. `LAYER2_OUTPUT_SPEC.md` → `PassiveIncomeMaximizer/docs/architecture/layer2/LAYER2_OUTPUT_SPEC.md`
3. `LAYER2_IMPLEMENTATION_GUIDE.md` → `PassiveIncomeMaximizer/docs/architecture/layer2/LAYER2_IMPLEMENTATION_GUIDE.md`
4. `LAYER2_QUICK_REFERENCE.md` → `PassiveIncomeMaximizer/docs/architecture/layer2/LAYER2_QUICK_REFERENCE.md`
5. `QUICKSTART_PIM_INTEGRATION.md` → `PassiveIncomeMaximizer/docs/integrations/FINCOLL_QUICKSTART.md`

---

## Why Moved

**Layer 2 RL agents** (MomentumAgent, MacroAgent, RiskAgent, etc.) are **PIM-exclusive features**.

**Architectural principle**: Documentation should live with implementation code.

**Code verification**: All Layer 2 agent implementations are in `PassiveIncomeMaximizer/engine/pim/agents/`, not in fincoll.

---

## What fincoll Owns

fincoll is a **data service**:
- Extracts 414D feature vectors
- Provides enriched output via `/api/v1/inference/enriched/symbol/{symbol}`
- Returns labeled features + predictions + context

fincoll does **NOT** implement Layer 2 decision-making logic.

---

## For More Details

See meta-repo documentation:
- `DOCUMENT_OWNERSHIP_ANALYSIS.md` - Why documents were moved
- `CODE_ARCHITECTURE_VERIFICATION.md` - Code location verification
- `DOCUMENT_MIGRATION_PHASE2_COMPLETE.md` - Migration process

Also see `fincoll/CLAUDE.md` for redirect notice.
