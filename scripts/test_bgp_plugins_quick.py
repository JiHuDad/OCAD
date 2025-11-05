#!/usr/bin/env python3
"""Quick test script for BGP plugins (no heavy dependencies).

This script tests core functionality without requiring PyTorch:
1. Plugin discovery (BGP adapter, GNN detector)
2. BGP adapter metric collection
3. Detector interface validation

For full testing including model training, install PyTorch and NetworkX.
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
    """Test 1: Plugin discovery for BGP components."""
    print_test_header("Plugin Discovery - BGP Components")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"

    print(f"Discovering plugins from: {plugin_dir}")
    registry.discover_plugins(plugin_dir)

    # Check BGP adapter
    bgp_adapter = registry.get_protocol_adapter("bgp")
    if bgp_adapter is None:
        print_test_result(False, "BGP adapter not found")
        return False

    print(f"✓ Found BGP adapter: {bgp_adapter.name} v{bgp_adapter.version}")
    print(f"  Supported metrics: {bgp_adapter.supported_metrics}")
    print(f"  Recommended models: {bgp_adapter.get_recommended_models()}")

    # Verify metric count
    if len(bgp_adapter.supported_metrics) < 7:
        print_test_result(False, f"BGP adapter should support at least 7 metrics, got {len(bgp_adapter.supported_metrics)}")
        return False

    # Check GNN detector (optional - requires PyTorch and NetworkX)
    gnn_detector = registry.get_detector("gnn")
    if gnn_detector is None:
        print("⚠️  GNN detector not found (requires PyTorch and NetworkX)")
    else:
        print(f"✓ Found GNN detector: {gnn_detector.name} v{gnn_detector.version}")
        print(f"  Supported protocols: {gnn_detector.supported_protocols}")

        # Verify BGP is supported
        if "bgp" not in gnn_detector.supported_protocols:
            print_test_result(False, "GNN detector does not support BGP")
            return False

    # Verify BGP is in recommended models
    recommended = bgp_adapter.get_recommended_models()
    if "gnn" not in recommended:
        print_test_result(False, "GNN not in recommended models for BGP")
        return False

    print_test_result(True, "All BGP components discovered")
    return True


async def test_bgp_adapter_collection():
    """Test 2: BGP adapter metric collection."""
    print_test_header("BGP Adapter Metric Collection")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    bgp_adapter = registry.get_protocol_adapter("bgp")
    if bgp_adapter is None:
        print_test_result(False, "BGP adapter not available")
        return False

    # Configuration for collection
    config = {
        "peers": [
            {
                "id": "test-bgp-peer-1",
                "local_asn": 65000,
                "remote_asn": 65001,
                "remote_ip": "192.168.1.1",
            },
            {
                "id": "test-bgp-peer-2",
                "local_asn": 65000,
                "remote_asn": 65002,
                "remote_ip": "192.168.1.2",
            },
        ],
        "interval_sec": 1,  # Collect every 1 second for testing
    }

    # Validate configuration
    try:
        is_valid = bgp_adapter.validate_config(config)
        print(f"✓ Configuration validation: {is_valid}")
    except Exception as e:
        print_test_result(False, f"Configuration validation failed: {e}")
        return False

    # Collect metrics (3 cycles)
    print("\nCollecting BGP metrics (3 cycles)...")
    collected_metrics = []
    expected_metric_names = set(bgp_adapter.supported_metrics)
    seen_metric_names = set()

    try:
        async for metric in bgp_adapter.collect(config):
            collected_metrics.append(metric)
            metric_name = metric.get("metric_name")
            seen_metric_names.add(metric_name)

            print(f"  [{metric['timestamp'].strftime('%H:%M:%S')}] "
                  f"{metric['source_id']}: {metric_name}={metric['value']:.2f}")

            # Stop after collecting 3 full cycles (7 metrics per peer × 2 peers × 3 = 42)
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

    # Verify BGP-specific fields
    if "protocol" not in sample_metric["metadata"]:
        print_test_result(False, "Missing 'protocol' in metadata")
        return False

    if sample_metric["metadata"]["protocol"] != "bgp":
        print_test_result(False, f"Expected protocol='bgp', got '{sample_metric['metadata']['protocol']}'")
        return False

    print_test_result(True, "BGP adapter collection successful")
    return True


async def test_detector_interfaces():
    """Test 3: Detector plugin interfaces."""
    print_test_header("Detector Plugin Interfaces")

    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"
    registry.discover_plugins(plugin_dir)

    detectors_to_test = []

    # Test GNN detector interface (optional)
    gnn_detector = registry.get_detector("gnn")
    if gnn_detector is None:
        print("⚠️  GNN detector not available (requires PyTorch and NetworkX)")
    else:
        print("✓ GNN detector interface:")
        print(f"  - name: {gnn_detector.name}")
        print(f"  - version: {gnn_detector.version}")
        print(f"  - supported_protocols: {gnn_detector.supported_protocols}")
        print(f"  - can_detect_protocol('bgp'): {gnn_detector.can_detect_protocol('bgp')}")
        print(f"  - description: {gnn_detector.get_description()}")
        detectors_to_test.append(gnn_detector)

        # Verify BGP support
        if not gnn_detector.can_detect_protocol("bgp"):
            print_test_result(False, "GNN detector does not support BGP")
            return False

    # Verify methods exist
    required_methods = ["train", "detect", "save_model", "load_model"]
    for detector in detectors_to_test:
        for method in required_methods:
            if not hasattr(detector, method):
                print_test_result(False, f"{detector.name} missing method: {method}")
                return False

    if detectors_to_test:
        print(f"\n✓ All required methods present: {required_methods}")
        print_test_result(True, "Detector interfaces validated")
    else:
        print_test_result(True, "No detectors available for testing (install PyTorch and NetworkX)")

    return True


async def main():
    """Run all tests."""
    print("=" * 70)
    print("BGP PLUGINS QUICK TEST (Phase 2)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNote: This is a quick test. For full testing including model training,")
    print("      install PyTorch and NetworkX, then run test_bgp_plugins_manual.py")

    tests = [
        ("Plugin Discovery", test_plugin_discovery),
        ("BGP Adapter Collection", test_bgp_adapter_collection),
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
        print("\n🎉 All quick tests passed! BGP plugin infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install torch networkx")
        print("  2. Run full tests: python scripts/test_bgp_plugins_manual.py")
        print("  3. Generate data: python scripts/generate_bgp_data.py --peers 10 --duration-hours 1")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
