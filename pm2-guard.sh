#!/bin/bash
# PM2 Enforcement Guard for FinColl
# Prevents manual starts that compete with PM2-managed service

# Check if running under PM2
if [ -z "$PM2_HOME" ]; then
  # Check if bypass is allowed (for development)
  if [ "$ALLOW_NON_PM2" != "1" ]; then
    echo ""
    echo "❌ ERROR: FinColl must be started via PM2"
    echo ""
    echo "PM2 Start:"
    echo "  cd /home/rford/caelum/caelum-supersystem/fincoll"
    echo "  pm2 start ecosystem.config.js"
    echo ""
    echo "PM2 Restart:"
    echo "  pm2 restart fincoll"
    echo ""
    echo "Development Bypass (NOT recommended):"
    echo "  ALLOW_NON_PM2=1 ./start_fincoll.sh"
    echo ""
    exit 1
  else
    echo "⚠️  WARNING: Running FinColl outside PM2 (ALLOW_NON_PM2=1)"
    echo "   This is for development only. Production should use PM2."
    echo ""
  fi
fi

# If we get here, either PM2_HOME is set OR ALLOW_NON_PM2=1
exit 0
