#!/usr/bin/env python3
"""Manual test script for BFD plugins (Phase 1).

This script tests:
1. Plugin discovery (BFD adapter, LSTM detector, HMM detector)
2. BFD adapter metric collection
3. LSTM detector training and detection
4. HMM detector training and detection

No pytest required - uses only Python standard library and project dependencies.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocad.plugins.registry import PluginRegistry


def print_test_header(test_name: str):
    """Print test header."""
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)


def print_test_result(passed: bool, message: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {message}" if message else status)


async def test_plugin_discovery():
    """Test 1: Plugin discovery for BFD components."""
    print_test_header("Plugin Discovery - BFD Components")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"

    print(f"Discovering plugins from: {plugin_dir}")
    registry.discover_plugins(plugin_dir)

    # Check BFD adapter
    bfd_adapter = registry.get_protocol_adapter("bfd")
    if bfd_adapter is None:
        print_test_result(False, "BFD adapter not found")
        return False

    print(f"✓ Found BFD adapter: {bfd_adapter.name} v{bfd_adapter.version}")
    print(f"  Supported metrics: {bfd_adapter.supported_metrics}")
    print(f"  Recommended models: {bfd_adapter.get_recommended_models()}")

    # Check LSTM detector
    lstm_detector = registry.get_detector("lstm")
    if lstm_detector is None:
        print_test_result(False, "LSTM detector not found")
        return False

    print(f"✓ Found LSTM detector: {lstm_detector.name} v{lstm_detector.version}")
    print(f"  Supported protocols: {lstm_detector.supported_protocols}")

    # Check HMM detector
    hmm_detector = registry.get_detector("hmm")
    if hmm_detector is None:
        print_test_result(False, "HMM detector not found")
        return False

    print(f"✓ Found HMM detector: {hmm_detector.name} v{hmm_detector.version}")
    print(f"  Supported protocols: {hmm_detector.supported_protocols}")

    # Verify BFD is supported by detectors
    detectors_for_bfd = registry.get_detectors_for_protocol("bfd")
    detector_names = [d.name for d in detectors_for_bfd]

    print(f"\n✓ Detectors supporting BFD: {detector_names}")

    expected_detectors = {"lstm", "hmm"}
    found_detectors = set(detector_names)

    if not expected_detectors.issubset(found_detectors):
        missing = expected_detectors - found_detectors
        print_test_result(False, f"Missing detectors for BFD: {missing}")
        return False

    print_test_result(True, "All BFD components discovered")
    return True


async def test_bfd_adapter_collection():
    """Test 2: BFD adapter metric collection."""
    print_test_header("BFD Adapter Metric Collection")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    bfd_adapter = registry.get_protocol_adapter("bfd")
    if bfd_adapter is None:
        print_test_result(False, "BFD adapter not available")
        return False

    # Configuration for collection
    config = {
        "sessions": [
            {
                "id": "test-bfd-session-1",
                "local_ip": "192.168.1.1",
                "remote_ip": "192.168.1.2",
                "interval_ms": 50,
                "multiplier": 3,
            },
            {
                "id": "test-bfd-session-2",
                "local_ip": "192.168.1.3",
                "remote_ip": "192.168.1.4",
                "interval_ms": 100,
                "multiplier": 5,
            },
        ],
        "interval_sec": 1,  # Collect every 1 second for testing
    }

    # Validate configuration
    try:
        is_valid = bfd_adapter.validate_config(config)
        print(f"✓ Configuration validation: {is_valid}")
    except Exception as e:
        print_test_result(False, f"Configuration validation failed: {e}")
        return False

    # Collect metrics (3 cycles)
    print("\nCollecting BFD metrics (3 cycles)...")
    collected_metrics = []
    expected_metric_names = set(bfd_adapter.supported_metrics)
    seen_metric_names = set()

    try:
        async for metric in bfd_adapter.collect(config):
            collected_metrics.append(metric)
            metric_name = metric.get("metric_name")
            seen_metric_names.add(metric_name)

            print(f"  [{metric['timestamp'].strftime('%H:%M:%S')}] "
                  f"{metric['source_id']}: {metric_name}={metric['value']:.2f}")

            # Stop after collecting 3 full cycles (7 metrics per session × 2 sessions × 3 = 42)
            if len(collected_metrics) >= 42:
                break

    except Exception as e:
        print_test_result(False, f"Collection failed: {e}")
        return False

    # Verify metrics
    print(f"\n✓ Collected {len(collected_metrics)} metrics")
    print(f"✓ Metric types seen: {seen_metric_names}")

    # Check if all expected metrics were seen
    if not expected_metric_names.issubset(seen_metric_names):
        missing = expected_metric_names - seen_metric_names
        print_test_result(False, f"Missing metrics: {missing}")
        return False

    # Verify metric structure
    sample_metric = collected_metrics[0]
    required_fields = {"timestamp", "source_id", "metric_name", "value", "metadata"}
    if not required_fields.issubset(sample_metric.keys()):
        missing = required_fields - sample_metric.keys()
        print_test_result(False, f"Missing fields in metric: {missing}")
        return False

    print_test_result(True, "BFD adapter collection successful")
    return True


async def test_lstm_detector():
    """Test 3: LSTM detector training and detection."""
    print_test_header("LSTM Detector Training and Detection")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    lstm_detector = registry.get_detector("lstm")
    if lstm_detector is None:
        print_test_result(False, "LSTM detector not available")
        return False

    # Check if PyTorch is available
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
    except ImportError:
        print_test_result(False, "PyTorch not installed (required for LSTM)")
        print("  Install with: pip install torch")
        return False

    # Generate synthetic training data (normal BFD detection times)
    print("\nGenerating synthetic training data...")
    np.random.seed(42)
    n_samples = 200
    # Normal BFD detection time: 15-50ms with some noise
    training_data = np.random.uniform(15, 50, n_samples) + np.random.normal(0, 5, n_samples)
    training_data = np.clip(training_data, 10, 60)  # Keep in reasonable range

    print(f"  Training samples: {n_samples}")
    print(f"  Mean: {np.mean(training_data):.2f}ms, Std: {np.std(training_data):.2f}ms")

    # Train model
    print("\nTraining LSTM model...")
    try:
        lstm_detector.train(
            training_data,
            epochs=20,  # Reduced for quick testing
            learning_rate=0.001,
            batch_size=32,
        )
        print(f"✓ Training completed")
        print(f"  Anomaly threshold: {lstm_detector.anomaly_threshold:.4f}")
    except Exception as e:
        print_test_result(False, f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test detection on normal data
    print("\nTesting detection on normal data...")
    normal_history = list(training_data[-10:])  # Last 10 values as history
    normal_value = 30.0  # Normal detection time

    result = lstm_detector.detect({
        "timestamp": datetime.utcnow(),
        "source_id": "test-session",
        "metric_name": "bfd_detection_time_ms",
        "value": normal_value,
        "history": normal_history,
    })

    print(f"  Normal value: {normal_value}ms")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Predicted value: {result['evidence']['predicted_value']:.2f}ms")

    if result["is_anomaly"]:
        print("  ⚠️ Normal data flagged as anomaly (may indicate overfitting)")

    # Test detection on anomalous data
    print("\nTesting detection on anomalous data...")
    anomaly_value = 250.0  # Slow detection time (anomaly)

    result = lstm_detector.detect({
        "timestamp": datetime.utcnow(),
        "source_id": "test-session",
        "metric_name": "bfd_detection_time_ms",
        "value": anomaly_value,
        "history": normal_history,
    })

    print(f"  Anomalous value: {anomaly_value}ms")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Predicted value: {result['evidence']['predicted_value']:.2f}ms")
    print(f"  Prediction error: {result['evidence']['prediction_error']:.4f}")

    if not result["is_anomaly"]:
        print_test_result(False, "Anomalous data not detected")
        return False

    print_test_result(True, "LSTM detector works correctly")
    return True


async def test_hmm_detector():
    """Test 4: HMM detector training and detection."""
    print_test_header("HMM Detector Training and Detection")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    hmm_detector = registry.get_detector("hmm")
    if hmm_detector is None:
        print_test_result(False, "HMM detector not available")
        return False

    # Generate synthetic training data (BFD states: mostly UP=3, rare transitions)
    print("\nGenerating synthetic training data...")
    np.random.seed(42)
    n_samples = 200

    # Normal behavior: mostly state 3 (UP) with rare transitions
    training_data = np.full(n_samples, 3.0)  # Start with all UP
    # Inject few transitions
    for _ in range(5):
        idx = np.random.randint(0, n_samples - 5)
        training_data[idx:idx+2] = 1.0  # Brief DOWN
        training_data[idx+2:idx+3] = 2.0  # INIT
        training_data[idx+3:idx+5] = 3.0  # Back to UP

    print(f"  Training samples: {n_samples}")
    print(f"  State distribution: {np.bincount(training_data.astype(int))}")

    # Train model
    print("\nTraining HMM model...")
    try:
        hmm_detector.train(training_data)
        print(f"✓ Training completed")
        print(f"  Anomaly threshold: {hmm_detector.anomaly_threshold:.4f}")
    except Exception as e:
        print_test_result(False, f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test detection on normal sequence
    print("\nTesting detection on normal sequence...")
    normal_history = [3.0] * 9  # 9 UP states
    normal_value = 3.0  # UP state

    result = hmm_detector.detect({
        "timestamp": datetime.utcnow(),
        "source_id": "test-session",
        "metric_name": "bfd_session_state",
        "value": normal_value,
        "history": normal_history,
    })

    print(f"  Normal sequence: all UP (state 3)")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Log likelihood: {result['evidence']['log_likelihood']:.4f}")

    if result["is_anomaly"]:
        print("  ⚠️ Normal sequence flagged as anomaly")

    # Test detection on anomalous sequence (rapid flapping)
    print("\nTesting detection on anomalous sequence (flapping)...")
    anomaly_history = [3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0]  # Rapid UP/DOWN
    anomaly_value = 1.0  # DOWN

    result = hmm_detector.detect({
        "timestamp": datetime.utcnow(),
        "source_id": "test-session",
        "metric_name": "bfd_session_state",
        "value": anomaly_value,
        "history": anomaly_history,
    })

    print(f"  Anomalous sequence: rapid UP/DOWN flapping")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Log likelihood: {result['evidence']['log_likelihood']:.4f}")

    if not result["is_anomaly"]:
        print_test_result(False, "Flapping sequence not detected")
        return False

    print_test_result(True, "HMM detector works correctly")
    return True


async def main():
    """Run all tests."""
    print("=" * 70)
    print("BFD PLUGINS TEST SUITE (Phase 1)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Plugin Discovery", test_plugin_discovery),
        ("BFD Adapter Collection", test_bfd_adapter_collection),
        ("LSTM Detector", test_lstm_detector),
        ("HMM Detector", test_hmm_detector),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = await test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)")

    if passed_count == total_count:
        print("\n🎉 All tests passed! BFD plugins are ready.")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
