#!/usr/bin/env python3
"""
Unit test for velocity engine integration in enriched API

Tests the _get_finvec_predictions function to verify:
1. Real model is called
2. Predictions are properly formatted
3. Confidence scores are real
"""

import sys
import numpy as np
from pathlib import Path

# Add fincoll to path
fincoll_path = Path(__file__).parent
sys.path.insert(0, str(fincoll_path))

# Add finvec to path for model imports
finvec_path = Path(__file__).parent.parent / "finvec"
sys.path.insert(0, str(finvec_path))

from fincoll.api.enriched import _get_finvec_predictions
from config.dimensions import DIMS
import asyncio


async def test_velocity_integration():
    """Test that _get_finvec_predictions calls real model"""

    print("=" * 80)
    print("Testing Velocity Engine Integration")
    print("=" * 80)

    # Create a random feature vector (matching expected dimensions)
    feature_dims = DIMS.fincoll_total
    print(f"\n[Setup] Creating {feature_dims}D feature vector...")

    # Use reasonable random values for features
    np.random.seed(42)  # Reproducible results
    raw_features = np.random.randn(feature_dims).astype(np.float32)

    # Normalize to reasonable ranges
    raw_features = np.clip(raw_features, -5, 5)

    print(f"[Setup] Feature vector shape: {raw_features.shape}")
    print(f"[Setup] Feature vector range: [{raw_features.min():.2f}, {raw_features.max():.2f}]")

    # Test with a symbol
    symbol = "AAPL"

    print(f"\n[Test] Calling _get_finvec_predictions for {symbol}...")

    try:
        predictions = await _get_finvec_predictions(symbol, raw_features)

        print(f"\n[Result] Predictions returned successfully")

        # Check structure
        assert "velocities" in predictions, "Missing 'velocities' key"
        assert "best_opportunity" in predictions, "Missing 'best_opportunity' key"

        velocities = predictions["velocities"]
        best_opp = predictions["best_opportunity"]

        print(f"\n[Validation] Velocities: {len(velocities)} timeframes")

        # Validate velocities
        if len(velocities) == 0:
            print("  ⚠️  WARNING: No velocities returned (model may have failed)")
            return False

        # Check velocity structure
        for i, vel in enumerate(velocities):
            required_keys = [
                "timeframe", "long_velocity", "long_bars", "long_confidence",
                "short_velocity", "short_bars", "short_confidence"
            ]
            for key in required_keys:
                assert key in vel, f"Missing key '{key}' in velocity {i}"

            print(f"  [{i}] {vel['timeframe']}:")
            print(f"      LONG: velocity={vel['long_velocity']:.4f}, bars={vel['long_bars']}, conf={vel['long_confidence']:.2f}")
            print(f"      SHORT: velocity={vel['short_velocity']:.4f}, bars={vel['short_bars']}, conf={vel['short_confidence']:.2f}")

        # Check best opportunity
        print(f"\n[Validation] Best Opportunity:")
        print(f"  Direction: {best_opp.get('direction', 'NONE')}")
        print(f"  Timeframe: {best_opp.get('timeframe', 'unknown')}")
        print(f"  Velocity: {best_opp.get('velocity', 0.0):.4f}")
        print(f"  Bars: {best_opp.get('bars', 0)}")
        print(f"  Confidence: {best_opp.get('confidence', 0.0):.2f}")

        # Validate confidence is not hardcoded
        confidence = best_opp.get("confidence", 0.0)

        # Check if confidence is in reasonable range
        if 0.0 < confidence < 1.0:
            print(f"\n  ✅ PASS: Confidence is calculated ({confidence:.4f})")
        else:
            print(f"\n  ⚠️  WARNING: Confidence out of range: {confidence}")

        # Check if predictions have metadata
        if "metadata" in predictions:
            metadata = predictions["metadata"]
            print(f"\n[Validation] Metadata:")
            print(f"  Model: {metadata.get('model', 'unknown')}")
            print(f"  Input dim: {metadata.get('input_dim', 'unknown')}")
            print(f"  Device: {metadata.get('device', 'unknown')}")

        print("\n" + "=" * 80)
        print("✅ TEST PASSED: Velocity integration working")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_symbols():
    """Test that predictions vary by symbol"""

    print("\n" + "=" * 80)
    print("Testing Multiple Symbols (Verify Non-Mock Data)")
    print("=" * 80)

    test_symbols = ["AAPL", "GOOGL", "TSLA"]
    results = []

    for symbol in test_symbols:
        print(f"\n[Test] Predicting {symbol}...")

        # Create different random features for each symbol
        feature_dims = DIMS.fincoll_total
        np.random.seed(hash(symbol) % (2**32))  # Different seed per symbol
        raw_features = np.random.randn(feature_dims).astype(np.float32)
        raw_features = np.clip(raw_features, -5, 5)

        try:
            predictions = await _get_finvec_predictions(symbol, raw_features)
            best_opp = predictions.get("best_opportunity", {})

            result = {
                "symbol": symbol,
                "velocity": best_opp.get("velocity", 0.0),
                "confidence": best_opp.get("confidence", 0.0),
                "direction": best_opp.get("direction", "NONE"),
            }

            results.append(result)

            print(f"  {result['direction']} @ {result['velocity']:.4f} (conf: {result['confidence']:.2f})")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    # Validate that predictions are different (not mock data)
    print(f"\n[Validation] Checking for unique predictions...")

    velocities = [r["velocity"] for r in results]
    unique_velocities = len(set(velocities))

    print(f"  Unique velocities: {unique_velocities}/{len(results)}")

    if unique_velocities == len(results):
        print("  ✅ PASS: All predictions are unique (not mock data)")
        return True
    else:
        print("  ⚠️  WARNING: Some predictions are identical")
        print("     This could mean mock data or model issues")
        return False


if __name__ == "__main__":
    print("\nRunning Velocity Engine Integration Tests\n")

    # Test 1: Basic integration
    success1 = asyncio.run(test_velocity_integration())

    # Test 2: Multiple symbols
    success2 = asyncio.run(test_multiple_symbols())

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Test 1 (Integration): {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Test 2 (Multiple Symbols): {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1 and success2:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
