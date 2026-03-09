#!/bin/bash
# Test predictions across diverse symbols to check for model variety

SYMBOLS=("AAPL" "MSFT" "GOOGL" "TSLA" "NVDA" "META" "AMD" "AMZN" "SPY" "QQQ")

echo "Testing prediction diversity across 10 symbols..."
echo "Symbol,Direction,Confidence,ExpectedReturn,CurrentPrice" > prediction_test.csv

for symbol in "${SYMBOLS[@]}"; do
    echo -n "Testing $symbol... "
    result=$(curl -s "http://10.32.3.27:8002/api/v1/inference/predict/$symbol" \
        -X POST \
        -H 'Content-Type: application/json' \
        -d '{"lookback": 100, "provider": "tradestation"}' 2>&1)
    
    if echo "$result" | jq -e . >/dev/null 2>&1; then
        direction=$(echo "$result" | jq -r '.best_opportunity.direction // "NULL"')
        confidence=$(echo "$result" | jq -r '.best_opportunity.confidence // "NULL"')
        expected_return=$(echo "$result" | jq -r '.best_opportunity.expected_return // "NULL"')
        current_price=$(echo "$result" | jq -r '.current_price // "NULL"')
        
        echo "$symbol,$direction,$confidence,$expected_return,$current_price" >> prediction_test.csv
        echo "✓ $direction @ ${confidence}"
    else
        echo "✗ Failed"
        echo "$symbol,ERROR,ERROR,ERROR,ERROR" >> prediction_test.csv
    fi
    
    sleep 2
done

echo ""
echo "Results saved to prediction_test.csv"
echo ""
echo "Summary:"
cat prediction_test.csv | column -t -s,
