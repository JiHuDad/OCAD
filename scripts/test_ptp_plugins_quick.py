#!/usr/bin/env python3
"""Quick test script for PTP plugins (no heavy dependencies).

This script tests core functionality without requiring PyTorch:
1. Plugin discovery (PTP adapter, TCN detector)
2. PTP adapter metric collection (8 metrics)
3. Detector interface validation

For full testing including model training, use test_ptp_plugins_manual.py
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
    """Test 1: Plugin discovery for PTP components."""
    print_test_header("Plugin Discovery - PTP Components")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"

    print(f"Discovering plugins from: {plugin_dir}")
    registry.discover_plugins(plugin_dir)

    # Check PTP adapter
    ptp_adapter = registry.get_protocol_adapter("ptp")
    if ptp_adapter is None:
        print_test_result(False, "PTP adapter not found")
        return False

    print(f"✓ Found PTP adapter: {ptp_adapter.name} v{ptp_adapter.version}")
    print(f"  Supported metrics: {ptp_adapter.supported_metrics}")
    print(f"  Recommended models: {ptp_adapter.get_recommended_models()}")

    # Verify 8 metrics
    expected_metrics = [
        "ptp_offset_from_master_ns",
        "ptp_mean_path_delay_ns",
        "ptp_clock_drift_ppb",
        "ptp_sync_interval_ms",
        "ptp_delay_req_interval_ms",
        "ptp_announce_interval_ms",
        "ptp_port_state",
        "ptp_steps_removed",
    ]

    if len(ptp_adapter.supported_metrics) != 8:
        print_test_result(False, f"Expected 8 metrics, got {len(ptp_adapter.supported_metrics)}")
        return False

    for metric in expected_metrics:
        if metric not in ptp_adapter.supported_metrics:
            print_test_result(False, f"Missing metric: {metric}")
            return False

    print(f"✓ All 8 expected metrics present")

    # Check TCN detector (optional - requires PyTorch)
    tcn_detector = registry.get_detector("tcn")
    if tcn_detector is None:
        print("⚠️  TCN detector not found (requires PyTorch)")
    else:
        print(f"✓ Found TCN detector: {tcn_detector.name} v{tcn_detector.version}")
        print(f"  Supported protocols: {tcn_detector.supported_protocols}")

        # Verify PTP is supported
        if "ptp" not in tcn_detector.supported_protocols:
            print_test_result(False, "TCN detector does not support PTP")
            return False

    # Verify PTP is in recommended models
    recommended = ptp_adapter.get_recommended_models()
    if "tcn" not in recommended:
        print_test_result(False, "TCN not in recommended models for PTP")
        return False

    print(f"\n✓ TCN is recommended for PTP: {recommended}")

    print_test_result(True, "All PTP components discovered")
    return True


async def test_ptp_adapter_collection():
    """Test 2: PTP adapter metric collection."""
    print_test_header("PTP Adapter Metric Collection")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    ptp_adapter = registry.get_protocol_adapter("ptp")
    if ptp_adapter is None:
        print_test_result(False, "PTP adapter not available")
        return False

    # Configuration for collection
    config = {
        "slaves": [
            {
                "id": "test-ptp-slave-1",
                "clock_id": "00:00:00:00:00:00:01:01",
                "master_clock_id": "00:00:00:00:00:00:00:01",
                "sync_interval_ms": 125,
                "delay_req_interval_ms": 1000,
                "announce_interval_ms": 1000,
                "steps_removed": 1,
            },
            {
                "id": "test-ptp-slave-2",
                "clock_id": "00:00:00:00:00:00:01:02",
                "master_clock_id": "00:00:00:00:00:00:00:01",
                "sync_interval_ms": 250,
                "delay_req_interval_ms": 2000,
                "announce_interval_ms": 2000,
                "steps_removed": 1,
            },
        ],
        "interval_sec": 1,  # Collect every 1 second for testing
    }

    # Validate configuration
    try:
        is_valid = ptp_adapter.validate_config(config)
        print(f"✓ Configuration validation: {is_valid}")
    except Exception as e:
        print_test_result(False, f"Configuration validation failed: {e}")
        return False

    # Collect metrics (3 cycles)
    print("\nCollecting PTP metrics (3 cycles)...")
    collected_metrics = []
    expected_metric_names = set(ptp_adapter.supported_metrics)
    seen_metric_names = set()

    try:
        async for metric in ptp_adapter.collect(config):
            collected_metrics.append(metric)
            metric_name = metric.get("metric_name")
            seen_metric_names.add(metric_name)

            # Format value based on metric type
            value = metric['value']
            if metric_name == "ptp_offset_from_master_ns":
                value_str = f"{value:.2f} ns"
            elif metric_name == "ptp_mean_path_delay_ns":
                value_str = f"{value:.2f} ns ({value/1000:.2f} μs)"
            elif metric_name == "ptp_clock_drift_ppb":
                value_str = f"{value:.2f} ppb"
            elif "interval" in metric_name:
                value_str = f"{value:.0f} ms"
            elif metric_name == "ptp_port_state":
                state_name = metric.get('metadata', {}).get('state_name', 'UNKNOWN')
                value_str = f"{int(value)} ({state_name})"
            elif metric_name == "ptp_steps_removed":
                value_str = f"{int(value)} hops"
            else:
                value_str = f"{value:.2f}"

            print(f"  [{metric['timestamp'].strftime('%H:%M:%S')}] "
                  f"{metric['source_id']}: {metric_name}={value_str}")

            # Stop after collecting 3 full cycles (8 metrics per slave × 2 slaves × 3 = 48)
            if len(collected_metrics) >= 48:
                break

    except Exception as e:
        print_test_result(False, f"Collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify metrics
    print(f"\n✓ Collected {len(collected_metrics)} metrics")
    print(f"✓ Metric types seen: {sorted(seen_metric_names)}")

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

    # Verify PTP-specific metadata
    print(f"\n✓ Sample metric structure:")
    print(f"  timestamp: {sample_metric['timestamp']}")
    print(f"  source_id: {sample_metric['source_id']}")
    print(f"  metric_name: {sample_metric['metric_name']}")
    print(f"  value: {sample_metric['value']}")
    print(f"  metadata: {sample_metric['metadata']}")

    # Check metadata fields
    metadata = sample_metric['metadata']
    if 'protocol' not in metadata or metadata['protocol'] != 'ptp':
        print_test_result(False, "Metadata missing 'protocol' field or not 'ptp'")
        return False

    if 'clock_id' not in metadata:
        print_test_result(False, "Metadata missing 'clock_id' field")
        return False

    print_test_result(True, "PTP adapter collection successful")
    return True


async def test_detector_interfaces():
    """Test 3: Detector plugin interfaces."""
    print_test_header("Detector Plugin Interfaces")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    detectors_to_test = []

    # Test TCN detector interface (optional)
    tcn_detector = registry.get_detector("tcn")
    if tcn_detector is None:
        print("⚠️  TCN detector not available (requires PyTorch)")
        print("   Install with: pip install torch")
        # This is acceptable for quick test
    else:
        print("✓ TCN detector interface:")
        print(f"  - name: {tcn_detector.name}")
        print(f"  - version: {tcn_detector.version}")
        print(f"  - supported_protocols: {tcn_detector.supported_protocols}")
        print(f"  - can_detect_protocol('ptp'): {tcn_detector.can_detect_protocol('ptp')}")
        print(f"  - description: {tcn_detector.get_description()}")
        detectors_to_test.append(tcn_detector)

        # Verify PTP support
        if not tcn_detector.can_detect_protocol('ptp'):
            print_test_result(False, "TCN detector cannot detect PTP protocol")
            return False

    # Check if at least one detector was found
    if len(detectors_to_test) == 0:
        print("\n⚠️  No detectors available for full testing (PyTorch required)")
        print("   This is acceptable for quick test - PTP adapter is working!")
        print_test_result(True, "Detector interfaces validation skipped (no PyTorch)")
        return True

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
    print("PTP PLUGINS QUICK TEST (Phase 3)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNote: This is a quick test. For full testing including model training,")
    print("      install PyTorch and run test_ptp_plugins_manual.py")

    tests = [
        ("Plugin Discovery", test_plugin_discovery),
        ("PTP Adapter Collection", test_ptp_adapter_collection),
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
        print("\n🎉 All quick tests passed! PTP plugin infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Install PyTorch: pip install torch")
        print("  2. Generate data: python scripts/generate_ptp_data.py --slaves 10 --duration-hours 1")
        print("  3. Train models: python scripts/train_tcn_ptp.py --data data/training/ptp/*.parquet")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
