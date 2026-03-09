#!/bin/bash
# Setup FinColl on 10.32.3.27 (run this ON THE SERVER, not over NFS)
# Usage: ssh rford@10.32.3.27 'bash /home/rford/caelum/ss/fincoll/setup_server.sh'

set -e

cd /home/rford/caelum/ss/fincoll

echo "=== Creating venv ==="
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip setuptools wheel

pip install \
  fastapi \
  uvicorn \
  torch \
  yfinance \
  pandas \
  numpy \
  requests \
  aiofiles \
  python-multipart \
  pyarrow

echo "=== Testing imports ==="
python -c "import torch; import fastapi; import yfinance; print('✅ All dependencies installed')"

echo ""
echo "❌ ERROR: This script is DEPRECATED"
echo ""
echo "FinColl is now managed by PM2 to prevent duplicate processes."
echo ""
echo "To start FinColl:"
echo "  cd /home/rford/caelum/caelum-supersystem/fincoll"
echo "  pm2 start ecosystem.config.js"
echo ""
echo "To restart FinColl:"
echo "  pm2 restart fincoll"
echo ""
echo "To view logs:"
echo "  pm2 logs fincoll"
echo ""
echo "To check status:"
echo "  pm2 list"
echo "  curl http://10.32.3.27:8002/health"
echo ""
exit 1
