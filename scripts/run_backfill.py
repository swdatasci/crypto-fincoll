#!/usr/bin/env python3
"""
Historical Feature Backfill Script

Usage:
    python scripts/run_backfill.py --symbols AAPL MSFT GOOGL --start-date 2020-01-01 --end-date 2024-12-31
    python scripts/run_backfill.py --update-existing  # Update existing vectors
    python scripts/run_backfill.py --dry-run  # Preview without saving
"""

import sys
from pathlib import Path

# Add fincoll directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.training.historical_feature_updater import main

if __name__ == "__main__":
    main()
