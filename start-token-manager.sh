#!/bin/bash
cd /home/rford/caelum/caelum-supersystem/fincoll
exec /home/rford/.local/bin/uv run python /home/rford/caelum/caelum-supersystem/finvec/utils/tradestation_token_manager.py --daemon
