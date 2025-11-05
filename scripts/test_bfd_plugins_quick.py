#!/usr/bin/env python3
"""Quick test script for BFD plugins (no heavy dependencies).

This script tests core functionality without requiring PyTorch:
1. Plugin discovery (BFD adapter, LSTM detector, HMM detector)
2. BFD adapter metric collection
3. Detector interface validation

For full testing including model training, use test_bfd_plugins_manual.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

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

    # Check LSTM detector (optional - requires PyTorch)
    lstm_detector = registry.get_detector("lstm")
    if lstm_detector is None:
        print("⚠️  LSTM detector not found (requires PyTorch)")
    else:
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

    # At least HMM should support BFD
    if "hmm" not in detector_names:
        print_test_result(False, "HMM detector does not support BFD")
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
        import traceback
        traceback.print_exc()
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


async def test_detector_interfaces():
    """Test 3: Detector plugin interfaces."""
    print_test_header("Detector Plugin Interfaces")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    detectors_to_test = []

    # Test LSTM detector interface (optional)
    lstm_detector = registry.get_detector("lstm")
    if lstm_detector is None:
        print("⚠️  LSTM detector not available (requires PyTorch)")
    else:
        print("✓ LSTM detector interface:")
        print(f"  - name: {lstm_detector.name}")
        print(f"  - version: {lstm_detector.version}")
        print(f"  - supported_protocols: {lstm_detector.supported_protocols}")
        print(f"  - can_detect_protocol('bfd'): {lstm_detector.can_detect_protocol('bfd')}")
        print(f"  - description: {lstm_detector.get_description()}")
        detectors_to_test.append(lstm_detector)

    # Test HMM detector interface
    hmm_detector = registry.get_detector("hmm")
    if hmm_detector is None:
        print_test_result(False, "HMM detector not available")
        return False

    print("\n✓ HMM detector interface:")
    print(f"  - name: {hmm_detector.name}")
    print(f"  - version: {hmm_detector.version}")
    print(f"  - supported_protocols: {hmm_detector.supported_protocols}")
    print(f"  - can_detect_protocol('bfd'): {hmm_detector.can_detect_protocol('bfd')}")
    print(f"  - description: {hmm_detector.get_description()}")
    detectors_to_test.append(hmm_detector)

    # Verify methods exist
    required_methods = ["train", "detect", "save_model", "load_model"]
    for detector in detectors_to_test:
        for method in required_methods:
            if not hasattr(detector, method):
                print_test_result(False, f"{detector.name} missing method: {method}")
                return False

    print(f"\n✓ All required methods present: {required_methods}")

    print_test_result(True, "Detector interfaces validated")
    return True


async def main():
    """Run all tests."""
    print("=" * 70)
    print("BFD PLUGINS QUICK TEST (Phase 1)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNote: This is a quick test. For full testing including model training,")
    print("      install PyTorch and run test_bfd_plugins_manual.py")

    tests = [
        ("Plugin Discovery", test_plugin_discovery),
        ("BFD Adapter Collection", test_bfd_adapter_collection),
        ("Detector Interfaces", test_detector_interfaces),
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
        print("\n🎉 All quick tests passed! BFD plugin infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Install PyTorch: pip install torch")
        print("  2. Run full tests: python scripts/test_bfd_plugins_manual.py")
        print("  3. Generate data: python scripts/generate_bfd_data.py --sessions 10 --duration-hours 1")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
