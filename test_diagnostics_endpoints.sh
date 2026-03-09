#!/bin/bash
# Test script for FinColl Diagnostics API endpoints
# Run after mounting diagnostics router in fincoll/server.py

BASE_URL="http://localhost:8002"
DIAG_PREFIX="/api/diagnostics"

echo "========================================="
echo "FinColl Diagnostics API Test Suite"
echo "========================================="
echo ""

# Test 1: Diagnostics Health
echo "1. Testing diagnostics health endpoint..."
curl -s "$BASE_URL$DIAG_PREFIX/health" | jq -r '.status' && echo "✅ Diagnostics health OK" || echo "❌ Diagnostics health FAILED"
echo ""

# Test 2: Features Summary
echo "2. Testing features summary endpoint..."
curl -s "$BASE_URL$DIAG_PREFIX/features/summary" | jq -r '.total_dimensions' && echo "✅ Features summary OK" || echo "❌ Features summary FAILED"
echo ""

# Test 3: Feature Segments
echo "3. Testing feature segments endpoint..."
SEGMENTS=$(curl -s "$BASE_URL$DIAG_PREFIX/features/segments" | jq 'length')
echo "   Found $SEGMENTS feature segments"
[ "$SEGMENTS" -gt 0 ] && echo "✅ Feature segments OK" || echo "❌ Feature segments FAILED"
echo ""

# Test 4: Feature Categories
echo "4. Testing feature categories endpoint..."
CATEGORIES=$(curl -s "$BASE_URL$DIAG_PREFIX/features/categories" | jq 'length')
echo "   Found $CATEGORIES feature categories"
[ "$CATEGORIES" -gt 0 ] && echo "✅ Feature categories OK" || echo "❌ Feature categories FAILED"
echo ""

# Test 5: Services Status
echo "5. Testing services status endpoint..."
OVERALL=$(curl -s "$BASE_URL$DIAG_PREFIX/services/status" | jq -r '.overall_status')
echo "   Overall status: $OVERALL"
[ -n "$OVERALL" ] && echo "✅ Services status OK" || echo "❌ Services status FAILED"
echo ""

# Test 6: Data Freshness
echo "6. Testing data freshness endpoint..."
SOURCES=$(curl -s "$BASE_URL$DIAG_PREFIX/data/freshness" | jq '.sources | length')
echo "   Found $SOURCES data sources"
[ "$SOURCES" -gt 0 ] && echo "✅ Data freshness OK" || echo "❌ Data freshness FAILED"
echo ""

# Test 7: Analysis (Mock)
echo "7. Testing analysis endpoint (mock)..."
ANALYSIS=$(curl -s -X POST "$BASE_URL$DIAG_PREFIX/analysis/run" \
  -H "Content-Type: application/json" \
  -d '{"use_mock": true}' | jq -r '.status')
echo "   Analysis status: $ANALYSIS"
[ "$ANALYSIS" = "complete" ] && echo "✅ Analysis endpoint OK" || echo "❌ Analysis endpoint FAILED"
echo ""

# Test 8: Latest Analysis
echo "8. Testing latest analysis endpoint..."
LATEST=$(curl -s "$BASE_URL$DIAG_PREFIX/analysis/latest" | jq -r '.status')
echo "   Latest analysis status: $LATEST"
[ "$LATEST" = "complete" ] && echo "✅ Latest analysis OK" || echo "❌ Latest analysis FAILED"
echo ""

# Test 9: Symbol Features (Mock)
echo "9. Testing symbol features endpoint..."
SYMBOL_DIMS=$(curl -s "$BASE_URL$DIAG_PREFIX/symbol/AAPL/features" | jq -r '.total_dimensions')
echo "   AAPL feature dimensions: $SYMBOL_DIMS"
[ "$SYMBOL_DIMS" -gt 0 ] && echo "✅ Symbol features OK" || echo "❌ Symbol features FAILED"
echo ""

# Summary
echo "========================================="
echo "Test Suite Complete"
echo "========================================="
echo ""
echo "All diagnostics endpoints are accessible at:"
echo "  $BASE_URL$DIAG_PREFIX"
echo ""
echo "Available endpoints:"
curl -s "$BASE_URL/openapi.json" | jq -r '.paths | keys | map(select(contains("diagnostics"))) | .[]' | sort
echo ""
