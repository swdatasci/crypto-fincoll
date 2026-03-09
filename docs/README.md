# FinColl Documentation

**Last Updated**: 2025-12-06

This directory contains documentation for the FinColl financial data collection and ML prediction service.

---

## Documentation Structure

```
docs/
├── README.md                          ← You are here
├── DEPLOYMENT.md                      ← Deployment and systemd setup
├── FEATURE_ABLATION_TESTING.md        ← Feature ablation test procedures
├── SYSTEMD_SERVICE.md                 ← Systemd service configuration
├── TESTING.md                         ← Testing strategy and procedures
├── archive/
│   └── SESSION_2025-11-14_FINCOLL_INTEGRATION.md  ← Integration session notes
└── history/
    ├── PHASE3_PLAN.md                 ← Phase 3 development plan
    ├── PHASE4_PLAN.md                 ← Phase 4 development plan
    ├── V6_*.md (6 files)              ← V6 model development
    ├── V7_*.md (4 files)              ← V7 model development
    └── V8_FEATURE_REDUCTION.md        ← V8 feature reduction experiments
```

---

## Quick Navigation

### Current Documentation

**Operations**:
- **DEPLOYMENT.md** - How to deploy FinColl service
- **SYSTEMD_SERVICE.md** - Configure FinColl as systemd service
- **TESTING.md** - How to test FinColl

**Development**:
- **FEATURE_ABLATION_TESTING.md** - Feature importance testing procedures

### Historical Documentation (history/)

**Development Phases**:
- **PHASE3_PLAN.md** - Phase 3 planning and goals
- **PHASE4_PLAN.md** - Phase 4 planning and goals

**Model Evolution**:
- **V6_*.md** - V6 model (335D features) development history
- **V7_*.md** - V7 model (336D features) development history
- **V8_FEATURE_REDUCTION.md** - V8 experiments (feature reduction)

### Archive (archive/)

**Session Notes**:
- **SESSION_2025-11-14_FINCOLL_INTEGRATION.md** - FinColl integration session

---

## Documentation Guidelines

### When to Create New Docs

**DO create new docs for**:
- Operational procedures (deployment, maintenance)
- Testing procedures and strategies
- Major feature documentation

**DON'T create new docs for**:
- Temporary session notes (use archive/)
- Historical development (use history/)
- Quick fixes or minor changes (update existing docs)

### Where to Put Documentation

**Root docs/ directory**:
- Active operational and development documentation
- Current testing and deployment procedures
- Regularly referenced guides

**history/ subdirectory**:
- Development phase plans (PHASE3, PHASE4, etc.)
- Model version development history (V6, V7, V8, etc.)
- Architectural evolution documentation

**archive/ subdirectory**:
- Session notes and temporary documentation
- One-time analysis or investigation reports
- Documentation that's no longer current but may be referenced

---

## Contributing to Documentation

1. **Update existing docs first** - Don't create new docs if existing ones can be updated
2. **Use clear structure** - Headers, bullet points, code blocks
3. **Date your changes** - Add "Last Updated" timestamps
4. **Link related docs** - Cross-reference when relevant
5. **Archive old content** - Move outdated docs to history/ or archive/

---

## Related Documentation

**Project-Level Docs** (in repository root):
- **CLAUDE.md** - AI assistant instructions and project overview
- **README.md** - Project setup and quick start
- **ARCHITECTURE.md** - System architecture and design
- **SECURITY.md** - Security guidelines

**External Dependencies**:
- **FinVec** (`/home/rford/caelum/ss/finvec/`) - ML models and training
- **SenVec** (`/home/rford/caelum/ss/senvec/`) - Sentiment features
- **PIM** (`/home/rford/caelum/caelum-supersystem/PassiveIncomeMaximizer/`) - Trading system

---

**Maintained by**: FinColl Development Team
**Questions**: See CLAUDE.md for project context
