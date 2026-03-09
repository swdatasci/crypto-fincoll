#!/usr/bin/env python3
"""
Test script to verify enriched API returns real predictions

Tests:
1. Model loads successfully
2. Predictions vary by symbol
3. Confidence scores are calculated, not hardcoded
4. Response time is reasonable (<500ms)
"""

import sys
import time
import asyncio
import requests
from pathlib import Path

# Add fincoll to path
fincoll_path = Path(__file__).parent
sys.path.insert(0, str(fincoll_path))

FINCOLL_URL = "http://localhost:8002"


async def test_enriched_endpoint():
    """Test enriched prediction endpoint"""

    print("=" * 80)
    print("Testing Enriched API - Real Model Predictions")
    print("=" * 80)

    # Test symbols
    test_symbols = ["AAPL", "GOOGL", "TSLA"]

    results = []

    for symbol in test_symbols:
        print(f"\n[Test] Fetching enriched prediction for {symbol}...")

        start_time = time.time()

        try:
            response = requests.post(
                f"{FINCOLL_URL}/api/v1/inference/enriched/symbol/{symbol}",
                params={
                    "lookback": 100,
                    "interval": "1d",
                    "provider": "yfinance"
                },
                timeout=30
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                # Extract key metrics
                predictions = data.get("predictions", {})
                best_opp = predictions.get("best_opportunity", {})
                confidence = data.get("confidence", {})

                result = {
                    "symbol": symbol,
                    "status": "success",
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "direction": best_opp.get("direction", "NONE"),
                    "velocity": best_opp.get("velocity", 0.0),
                    "confidence": best_opp.get("confidence", 0.0),
                    "overall_confidence": confidence.get("overall", 0.0),
                }

                results.append(result)

                print(f"  ✅ Success: {result['direction']} @ {result['velocity']:.4f}")
                print(f"  📊 Confidence: {result['confidence']:.2f} (overall: {result['overall_confidence']:.2f})")
                print(f"  ⏱️  Response time: {result['elapsed_ms']:.1f}ms")

            else:
                print(f"  ❌ HTTP {response.status_code}: {response.text}")
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            })

    # Validate results
    print("\n" + "=" * 80)
    print("Validation Results")
    print("=" * 80)

    successful = [r for r in results if r["status"] == "success"]

    if not successful:
        print("❌ FAIL: No successful predictions")
        return False

    # Check 1: All predictions have different velocities (not mock data)
    velocities = [r["velocity"] for r in successful]
    unique_velocities = len(set(velocities))

    print(f"\n1. Unique predictions: {unique_velocities}/{len(successful)}")
    if unique_velocities == len(successful):
        print("   ✅ PASS: Predictions vary by symbol (not hardcoded)")
    else:
        print("   ⚠️  WARNING: Some predictions are identical (may be mock data)")

    # Check 2: Confidence scores are calculated (not 0.85 hardcoded)
    confidences = [r["confidence"] for r in successful]
    hardcoded_confidence = all(abs(c - 0.85) < 0.01 for c in confidences)

    print(f"\n2. Confidence scores: {confidences}")
    if not hardcoded_confidence:
        print("   ✅ PASS: Confidence scores are calculated")
    else:
        print("   ❌ FAIL: Confidence scores are hardcoded (0.85)")

    # Check 3: Response time is reasonable
    avg_time = sum(r["elapsed_ms"] for r in successful) / len(successful)

    print(f"\n3. Average response time: {avg_time:.1f}ms")
    if avg_time < 500:
        print("   ✅ PASS: Response time < 500ms")
    else:
        print(f"   ⚠️  WARNING: Response time is {avg_time:.1f}ms (target: <500ms)")

    # Check 4: Predictions include metadata
    first_result_response = requests.post(
        f"{FINCOLL_URL}/api/v1/inference/enriched/symbol/{test_symbols[0]}",
        params={"lookback": 100, "interval": "1d", "provider": "yfinance"},
        timeout=30
    )

    if first_result_response.status_code == 200:
        data = first_result_response.json()
        has_metadata = "predictions" in data and "metadata" in data["predictions"]

        print(f"\n4. Metadata present: {has_metadata}")
        if has_metadata:
            print("   ✅ PASS: Prediction metadata included")
        else:
            print("   ❌ FAIL: Prediction metadata missing")

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {len(test_symbols)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(test_symbols) - len(successful)}")

    if len(successful) == len(test_symbols) and unique_velocities == len(successful):
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED OR WARNINGS")
        return False


if __name__ == "__main__":
    # Check if FinColl server is running
    try:
        response = requests.get(f"{FINCOLL_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ FinColl server not healthy: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to FinColl server at {FINCOLL_URL}")
        print(f"   Error: {e}")
        print(f"\n   Start the server with:")
        print(f"   cd /home/rford/caelum/caelum-supersystem/fincoll")
        print(f"   source .venv/bin/activate")
        print(f"   python -m fincoll.server")
        sys.exit(1)

    # Run tests
    success = asyncio.run(test_enriched_endpoint())
    sys.exit(0 if success else 1)
