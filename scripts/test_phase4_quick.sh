#!/usr/bin/env bash
#
# Quick Phase 4 Test Script
# Tests basic functionality of enriched API endpoints
#

set -e

BASE_URL="${FINCOLL_URL:-http://10.32.3.27:8002}"
ENRICHED_ENDPOINT="$BASE_URL/api/v1/inference/enriched"

echo "========================================="
echo "Phase 4: Quick Test Suite"
echo "========================================="
echo ""
echo "Base URL: $BASE_URL"
echo "Testing enriched endpoint..."
echo ""

# Test 1: Health Check
echo "Test 1: API Health Check"
echo "-------------------------"
if curl -s "$BASE_URL/health" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed"
    exit 1
fi
echo ""

# Test 2: Single Symbol Enrichment
echo "Test 2: Single Symbol Enrichment (AAPL)"
echo "----------------------------------------"
RESPONSE=$(curl -s -X POST "$ENRICHED_ENDPOINT/symbol/AAPL?lookback=100" -H "Content-Type: application/json")

if echo "$RESPONSE" | jq -e '.symbol == "AAPL"' > /dev/null 2>&1; then
    echo "✅ Symbol field correct"
else
    echo "❌ Symbol field missing or incorrect"
    echo "Response: $RESPONSE"
    exit 1
fi

if echo "$RESPONSE" | jq -e '.features' > /dev/null 2>&1; then
    echo "✅ Features present"
else
    echo "❌ Features missing"
    exit 1
fi

if echo "$RESPONSE" | jq -e '.context_for_agent' > /dev/null 2>&1; then
    echo "✅ Context present"
else
    echo "❌ Context missing"
    exit 1
fi

if echo "$RESPONSE" | jq -e '.predictions' > /dev/null 2>&1; then
    echo "✅ Predictions present"
else
    echo "❌ Predictions missing"
    exit 1
fi

COMPLETENESS=$(echo "$RESPONSE" | jq -r '.data_quality.feature_completeness // 0')
echo "   Feature completeness: $COMPLETENESS"

CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence.overall // 0')
echo "   Overall confidence: $CONFIDENCE"
echo ""

# Test 3: Batch Processing
echo "Test 3: Batch Processing (5 symbols)"
echo "-------------------------------------"
BATCH_RESPONSE=$(curl -s -X POST "$ENRICHED_ENDPOINT/batch?lookback=100" \
  -H "Content-Type: application/json" \
  -d '["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]')

COMPLETED=$(echo "$BATCH_RESPONSE" | jq -r '.symbols_completed // 0')
FAILED=$(echo "$BATCH_RESPONSE" | jq -r '.symbols_failed // 0')

echo "   Symbols completed: $COMPLETED"
echo "   Symbols failed: $FAILED"

if [ "$COMPLETED" -gt 0 ]; then
    echo "✅ Batch processing working (at least some symbols processed)"
else
    echo "❌ Batch processing failed - no symbols completed"
    exit 1
fi
echo ""

# Test 4: Performance Check
echo "Test 4: Performance Check"
echo "-------------------------"
echo "Testing single symbol response time..."

START_TIME=$(date +%s.%N)
curl -s -X POST "$ENRICHED_ENDPOINT/AAPL?lookback=100" > /dev/null
END_TIME=$(date +%s.%N)

DURATION=$(echo "$END_TIME - $START_TIME" | bc)
echo "   Response time: ${DURATION}s"

if (( $(echo "$DURATION < 5.0" | bc -l) )); then
    echo "✅ Performance acceptable (< 5s)"
else
    echo "⚠️  Performance slow (> 5s) - may need optimization"
fi
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "✅ All basic tests passed"
echo ""
echo "Next steps:"
echo "1. Run full test suite: uv run pytest tests/ -v"
echo "2. Run load tests: bash scripts/test_load.sh"
echo "3. Review PHASE4_TESTING_PLAN.md for complete test coverage"
echo ""
echo "Phase 4 quick test complete! ✅"
