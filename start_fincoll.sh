#!/bin/bash
# PM2 Enforcement: Check if running under PM2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/pm2-guard.sh" || exit 1

export PYTHONPATH=/home/rford/caelum/caelum-supersystem/finvec:$PYTHONPATH
cd /home/rford/caelum/caelum-supersystem/fincoll
exec uv run uvicorn fincoll.server:app --host 0.0.0.0 --port 8002
