#!/bin/bash
export PYTHONPATH=/home/rford/caelum/caelum-supersystem/finvec:$PYTHONPATH
cd /home/rford/caelum/caelum-supersystem/fincoll
source .venv/bin/activate
exec uvicorn fincoll.server:app --host 0.0.0.0 --port 8002
