#!/usr/bin/env python3
"""Manual test script for plugin system (without pytest dependency)."""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.plugins.registry import registry
from ocad.plugins.protocol_adapters.cfm import create_adapter


def test_plugin_registry():
    """Test basic plugin registry functionality."""
    print("=" * 60)
    print("Test 1: Plugin Registry Basic Functionality")
    print("=" * 60)

    try:
        # Test CFM adapter creation
        cfm_adapter = create_adapter()
        print(f"✓ CFM adapter created: {cfm_adapter.name} v{cfm_adapter.version}")

        # Test adapter properties
        assert cfm_adapter.name == "cfm"
        assert cfm_adapter.version == "1.0.0"
        print(f"✓ Adapter properties validated")

        # Test supported metrics
        metrics = cfm_adapter.supported_metrics
        assert len(metrics) == 4
        assert "udp_echo_rtt_ms" in metrics
        print(f"✓ Supported metrics ({len(metrics)}): {', '.join(metrics[:2])}...")

        # Test recommended models
        models = cfm_adapter.get_recommended_models()
        assert len(models) > 0
        print(f"✓ Recommended models ({len(models)}): {', '.join(models)}")

        # Test config validation
        valid_config = {"endpoints": [{"id": "test", "host": "localhost"}]}
        assert cfm_adapter.validate_config(valid_config) is True
        print(f"✓ Config validation passed")

        try:
            invalid_config = {}
            cfm_adapter.validate_config(invalid_config)
            print(f"✗ Config validation should have failed")
            return False
        except ValueError:
            print(f"✓ Config validation correctly rejected invalid config")

        print("\n✅ Test 1 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_discovery():
    """Test plugin discovery mechanism."""
    print("=" * 60)
    print("Test 2: Plugin Discovery")
    print("=" * 60)

    try:
        # Get plugin directory
        plugin_dir = Path(__file__).parent.parent / "ocad" / "plugins"
        print(f"Plugin directory: {plugin_dir}")

        if not plugin_dir.exists():
            print(f"✗ Plugin directory does not exist")
            return False

        # Discover plugins
        print(f"Discovering plugins...")
        registry.discover_plugins(plugin_dir)

        # Check registered adapters
        adapters = registry.list_protocol_adapters()
        print(f"✓ Found {len(adapters)} protocol adapter(s)")

        if "cfm" in adapters:
            cfm_info = adapters["cfm"]
            print(f"  - CFM adapter v{cfm_info['version']}")
            print(f"    Metrics: {', '.join(cfm_info['supported_metrics'][:2])}...")
            print(f"    Models: {', '.join(cfm_info['recommended_models'])}")
        else:
            print(f"✗ CFM adapter not discovered")
            return False

        # Check registered detectors
        detectors = registry.list_detectors()
        print(f"✓ Found {len(detectors)} detector(s)")
        if len(detectors) > 0:
            for name, info in detectors.items():
                print(f"  - {name} v{info['version']}")

        # Test retrieval
        cfm_adapter = registry.get_protocol_adapter("cfm")
        if cfm_adapter is None:
            print(f"✗ Failed to retrieve CFM adapter")
            return False
        print(f"✓ Successfully retrieved CFM adapter")

        print("\n✅ Test 2 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


async def test_cfm_adapter_collection():
    """Test CFM adapter metric collection."""
    print("=" * 60)
    print("Test 3: CFM Adapter Metric Collection")
    print("=" * 60)

    try:
        cfm_adapter = create_adapter()

        # Test configuration
        config = {
            "endpoints": [
                {
                    "id": "test-endpoint-1",
                    "host": "192.168.1.100",
                    "udp_echo": True,
                    "ecpri": True,
                    "lbm": True,
                }
            ],
            "interval_sec": 1,
        }

        print(f"Testing metric collection (1 second)...")

        # Collect metrics for 1 second
        collected_metrics = []
        async for metric in cfm_adapter.collect(config):
            collected_metrics.append(metric)
            print(f"  ✓ Collected: {metric['metric_name']} = {metric['value']} from {metric['source_id']}")

            # Stop after collecting 3 metrics
            if len(collected_metrics) >= 3:
                break

        if len(collected_metrics) == 0:
            print(f"✗ No metrics collected")
            return False

        # Validate collected metrics
        for metric in collected_metrics:
            assert "timestamp" in metric
            assert "source_id" in metric
            assert "metric_name" in metric
            assert "value" in metric
            assert "metadata" in metric

        print(f"✓ Collected {len(collected_metrics)} metrics successfully")
        print(f"✓ All metrics have required fields")

        print("\n✅ Test 3 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_descriptions():
    """Test plugin description generation."""
    print("=" * 60)
    print("Test 4: Plugin Descriptions")
    print("=" * 60)

    try:
        cfm_adapter = create_adapter()

        # Test adapter description
        desc = cfm_adapter.get_description()
        print(f"CFM Adapter Description: {desc}")
        assert "cfm" in desc.lower()
        assert "1.0.0" in desc
        print(f"✓ Adapter description generated correctly")

        print("\n✅ Test 4 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OCAD Plugin System Manual Test Suite")
    print("=" * 60 + "\n")

    results = []

    # Run synchronous tests
    results.append(("Plugin Registry", test_plugin_registry()))
    results.append(("Plugin Discovery", test_plugin_discovery()))
    results.append(("Plugin Descriptions", test_plugin_descriptions()))

    # Run async test
    try:
        result = asyncio.run(test_cfm_adapter_collection())
        results.append(("CFM Adapter Collection", result))
    except Exception as e:
        print(f"❌ Failed to run async test: {e}")
        results.append(("CFM Adapter Collection", False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    # Exit with appropriate code
    if passed == total:
        print("🎉 All tests passed! Plugin system is working correctly.\n")
        sys.exit(0)
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review the errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
