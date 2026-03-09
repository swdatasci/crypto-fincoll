#!/bin/bash
# Cron-friendly wrapper for Finnhub cache warming
# Run this daily to keep cache warm for active symbols
#
# Example crontab (run daily at 4 AM):
#   0 4 * * * /home/rford/caelum/caelum-supersystem/fincoll/scripts/cron_warm_cache.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINCOLL_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$FINCOLL_ROOT/logs/cache_warming.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Redirect output to log file
exec >> "$LOG_FILE" 2>&1

echo "==================================================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting Finnhub cache warming"
echo "==================================================================="

# Activate virtual environment
cd "$FINCOLL_ROOT"
source .venv/bin/activate

# Run warming script with PIM universe
python scripts/warm_finnhub_cache.py --universe pim --concurrent 5

echo "$(date '+%Y-%m-%d %H:%M:%S') - Cache warming complete"
echo ""
