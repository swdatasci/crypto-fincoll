#!/bin/bash
# Deploy date-cascading SenVec integration

set -e

echo "=== Deploying Date-Cascading SenVec Integration ==="
echo ""

FINCOLL_DIR="/home/rford/caelum/ss/fincoll/fincoll/utils"
SOURCE_FILE="/home/rford/caelum/caelum-supersystem/fincoll/senvec_date_cascade.py"
TARGET_FILE="${FINCOLL_DIR}/senvec_integration.py"
BACKUP_FILE="${FINCOLL_DIR}/senvec_integration.py.before_cascade"

# Backup existing
if [ -f "${TARGET_FILE}" ]; then
    echo "Backing up existing integration..."
    cp "${TARGET_FILE}" "${BACKUP_FILE}"
    echo "✓ Backed up to ${BACKUP_FILE}"
else
    echo "⚠ No existing file found (clean install)"
fi

# Deploy new version
echo "Deploying date-cascading version..."
cp "${SOURCE_FILE}" "${TARGET_FILE}"
echo "✓ Deployed to ${TARGET_FILE}"

# Make executable
chmod +x "${TARGET_FILE}"

# Verify
echo ""
echo "Verifying deployment..."
if python3 -c "from fincoll.utils.senvec_integration import get_senvec_features; print('✓ Import successful')" 2>&1 | grep -q "successful"; then
    echo "✓ Import test passed"
else
    echo "✗ Import test failed"
    echo "Rolling back..."
    cp "${BACKUP_FILE}" "${TARGET_FILE}"
    exit 1
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. Test: python3 ${TARGET_FILE}"
echo "2. Warm cache: python3 ${TARGET_FILE} warm AAPL MSFT GOOGL"
echo "3. Train with: cd finvec && python train_velocity.py --symbol-set diversified --epochs 100"
echo ""
echo "To rollback: cp ${BACKUP_FILE} ${TARGET_FILE}"
