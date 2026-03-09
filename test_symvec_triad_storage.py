#!/usr/bin/env python3
"""
Test SymVec Triad Storage Fix

Verifies that complete SymVec triads (input + raw_data + predictions)
are stored together in a single InfluxDB record.
"""

import sys
import numpy as np
from datetime import datetime
from influxdb_client import InfluxDBClient

# Test configuration
INFLUXDB_URL = "http://10.32.3.27:8086"
INFLUXDB_TOKEN = "F0TMD5qmaQ5FJKGYdscRFu9EH80afzlqDTwfj3Kuc3SyjumtD4HqH2oI-Qp9XSKfLI8OWJ4pwN-vuLN4M6V6-g=="
INFLUXDB_ORG = "caelum"
INFLUXDB_BUCKET = "pim_vectors"

def test_triad_storage():
    """Test that triads are stored with all three components"""

    print("=" * 70)
    print("Testing SymVec Triad Storage Fix")
    print("=" * 70)

    # Import storage
    try:
        from fincoll.storage.influxdb_saver import InfluxDBFeatureSaver
        from config.dimensions import DIMS
        print(f"✅ Imports successful (feature dim: {DIMS.fincoll_total})")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Create storage instance
    try:
        saver = InfluxDBFeatureSaver(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
            bucket=INFLUXDB_BUCKET,
        )
        print(f"✅ Storage initialized")
    except Exception as e:
        print(f"❌ Storage init failed: {e}")
        return False

    # Create test data
    test_symbol = "TEST_TRIAD"
    test_timestamp = datetime.utcnow()
    test_features = np.random.randn(DIMS.fincoll_total)
    test_velocities = {
        "1min": 0.0012,
        "15min": 0.0089,
        "1hour": 0.0156,
        "daily": 0.0234,
        "confidence": 0.85,
    }
    test_raw_data = {
        "price": 150.50,
        "volume": 1000000,
        "bid": 150.48,
        "ask": 150.52,
    }

    # Save complete triad
    print(f"\nSaving complete triad for {test_symbol}...")
    try:
        success = saver.save_feature_vector(
            symbol=test_symbol,
            timestamp=test_timestamp,
            features=test_features,
            source="test_script",
            metadata={"test_run": True},
            velocities=test_velocities,
            raw_data=test_raw_data,
        )

        if success:
            print(f"✅ Triad saved successfully")
        else:
            print(f"❌ Save returned False")
            return False

    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify the record was written correctly
    print(f"\nVerifying record in InfluxDB...")
    try:
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        query_api = client.query_api()

        # Query the record we just wrote
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -5m)
          |> filter(fn: (r) => r["_measurement"] == "feature_vectors")
          |> filter(fn: (r) => r["symbol"] == "{test_symbol}")
        '''

        result = query_api.query(query)

        if not result:
            print(f"❌ No data found for {test_symbol}")
            return False

        # Collect all fields
        fields = set()
        for table in result:
            for record in table.records:
                fields.add(record.get_field())

        # Check for expected components
        has_features = any(f.startswith('f') and f[1:].isdigit() for f in fields)
        has_velocities = 'velocity_1m' in fields and 'velocity_15m' in fields
        has_raw_data = 'raw_price' in fields and 'raw_volume' in fields
        has_confidence = 'confidence' in fields

        print(f"\nRecord Components:")
        print(f"  Features (f0-f{DIMS.fincoll_total-1}): {'✅' if has_features else '❌'}")
        print(f"  Velocities: {'✅' if has_velocities else '❌'}")
        print(f"  Confidence: {'✅' if has_confidence else '❌'}")
        print(f"  Raw Data: {'✅' if has_raw_data else '❌'}")
        print(f"\nTotal fields: {len(fields)}")

        if has_features and has_velocities and has_raw_data and has_confidence:
            print(f"\n{'='*70}")
            print(f"✅ SUCCESS: Complete SymVec triad stored correctly!")
            print(f"{'='*70}")
            return True
        else:
            print(f"\n{'='*70}")
            print(f"❌ FAILED: Missing components in stored record")
            print(f"{'='*70}")
            print(f"\nAll fields: {sorted(fields)[:20]}...")
            return False

        client.close()

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        saver.close()


if __name__ == "__main__":
    success = test_triad_storage()
    sys.exit(0 if success else 1)
