#!/usr/bin/env python3
"""통합 플러그인 테스트 스크립트 (Phase 4).

모든 프로토콜 어댑터와 탐지기를 종합적으로 테스트합니다:

1. 프로토콜 어댑터 테스트:
   - CFM (UDP Echo, eCPRI, LBM, CCM)
   - BFD (session state, flapping)
   - BGP (UPDATE, AS-path) [TODO: Phase 2]
   - PTP (offset, drift) [TODO: Phase 3]

2. 탐지기 테스트:
   - LSTM (BFD, BGP, CFM, PTP)
   - HMM (BFD, BGP)
   - GNN (BGP) [TODO: Phase 2]
   - TCN (PTP, CFM) [TODO: Phase 3]

3. 크로스 프로토콜 테스트:
   - CFM과 BFD 동시 모니터링
   - BGP와 PTP 동시 모니터링 [TODO]

4. 성능 테스트:
   - 100개 엔드포인트 동시 수집
   - 메모리 사용량 확인
   - CPU 사용률 확인

Usage:
    python scripts/test_all_plugins.py
    python scripts/test_all_plugins.py --performance  # 성능 테스트 포함
    python scripts/test_all_plugins.py --quick        # 빠른 테스트만
"""

import sys
import asyncio
import argparse
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocad.plugins.registry import PluginRegistry


# ============================================================================
# Test Utilities
# ============================================================================

class TestStats:
    """테스트 통계 수집."""

    def __init__(self):
        self.protocol_adapters_tested = 0
        self.protocol_adapters_passed = 0
        self.detectors_tested = 0
        self.detectors_passed = 0
        self.integration_tests_passed = 0
        self.integration_tests_total = 0
        self.start_time = None
        self.end_time = None
        self.memory_peak_mb = 0
        self.cpu_usage_percent = []

    def record_memory(self):
        """메모리 사용량 기록."""
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        if mem_mb > self.memory_peak_mb:
            self.memory_peak_mb = mem_mb

    def record_cpu(self):
        """CPU 사용률 기록."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage_percent.append(cpu_percent)

    def get_summary(self) -> str:
        """테스트 요약 반환."""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        avg_cpu = sum(self.cpu_usage_percent) / len(self.cpu_usage_percent) if self.cpu_usage_percent else 0

        return f"""
테스트 통계:
  프로토콜 어댑터: {self.protocol_adapters_passed}/{self.protocol_adapters_tested} 통과
  탐지기: {self.detectors_passed}/{self.detectors_tested} 통과
  통합 테스트: {self.integration_tests_passed}/{self.integration_tests_total} 통과

성능:
  총 소요 시간: {duration:.2f}초
  메모리 피크: {self.memory_peak_mb:.2f} MB
  평균 CPU 사용률: {avg_cpu:.1f}%
"""


def print_test_header(test_name: str):
    """테스트 헤더 출력."""
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)


def print_test_result(passed: bool, message: str = ""):
    """테스트 결과 출력."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {message}" if message else status)


def print_warning(message: str):
    """경고 출력."""
    print(f"⚠️  {message}")


def print_info(message: str):
    """정보 출력."""
    print(f"ℹ️  {message}")


# ============================================================================
# Protocol Adapter Tests
# ============================================================================

async def test_cfm_adapter(registry: PluginRegistry, stats: TestStats) -> bool:
    """CFM 어댑터 테스트."""
    print_test_header("CFM Protocol Adapter")
    stats.protocol_adapters_tested += 1
    stats.record_memory()
    stats.record_cpu()

    cfm_adapter = registry.get_protocol_adapter("cfm")
    if cfm_adapter is None:
        print_test_result(False, "CFM adapter not found")
        return False

    print(f"✓ Found CFM adapter: {cfm_adapter.name} v{cfm_adapter.version}")
    print(f"  Supported metrics: {cfm_adapter.supported_metrics}")
    print(f"  Recommended models: {cfm_adapter.get_recommended_models()}")

    # Configuration validation
    config = {
        "endpoints": [
            {"id": "test-cfm-1", "ip": "192.168.1.1", "port": 50000}
        ],
        "interval_sec": 1,
    }

    try:
        is_valid = cfm_adapter.validate_config(config)
        print(f"✓ Configuration validation: {is_valid}")
    except Exception as e:
        print_test_result(False, f"Configuration validation failed: {e}")
        return False

    # Metric collection (10 samples)
    print("\nCollecting CFM metrics (10 samples)...")
    collected_metrics = []
    try:
        async for metric in cfm_adapter.collect(config):
            collected_metrics.append(metric)
            print(f"  [{metric['timestamp'].strftime('%H:%M:%S')}] "
                  f"{metric['source_id']}: {metric['metric_name']}={metric['value']:.2f}")

            if len(collected_metrics) >= 10:
                break
    except Exception as e:
        print_test_result(False, f"Collection failed: {e}")
        return False

    print(f"\n✓ Collected {len(collected_metrics)} metrics")
    stats.protocol_adapters_passed += 1
    print_test_result(True, "CFM adapter test passed")
    return True


async def test_bfd_adapter(registry: PluginRegistry, stats: TestStats) -> bool:
    """BFD 어댑터 테스트."""
    print_test_header("BFD Protocol Adapter")
    stats.protocol_adapters_tested += 1
    stats.record_memory()
    stats.record_cpu()

    bfd_adapter = registry.get_protocol_adapter("bfd")
    if bfd_adapter is None:
        print_test_result(False, "BFD adapter not found")
        return False

    print(f"✓ Found BFD adapter: {bfd_adapter.name} v{bfd_adapter.version}")
    print(f"  Supported metrics: {bfd_adapter.supported_metrics}")
    print(f"  Recommended models: {bfd_adapter.get_recommended_models()}")

    # Configuration validation
    config = {
        "sessions": [
            {
                "id": "test-bfd-1",
                "local_ip": "192.168.1.1",
                "remote_ip": "192.168.1.2",
                "interval_ms": 50,
                "multiplier": 3,
            }
        ],
        "interval_sec": 1,
    }

    try:
        is_valid = bfd_adapter.validate_config(config)
        print(f"✓ Configuration validation: {is_valid}")
    except Exception as e:
        print_test_result(False, f"Configuration validation failed: {e}")
        return False

    # Metric collection (14 samples = 2 cycles)
    print("\nCollecting BFD metrics (2 cycles)...")
    collected_metrics = []
    try:
        async for metric in bfd_adapter.collect(config):
            collected_metrics.append(metric)
            print(f"  [{metric['timestamp'].strftime('%H:%M:%S')}] "
                  f"{metric['source_id']}: {metric['metric_name']}={metric['value']:.2f}")

            if len(collected_metrics) >= 14:
                break
    except Exception as e:
        print_test_result(False, f"Collection failed: {e}")
        return False

    print(f"\n✓ Collected {len(collected_metrics)} metrics")
    stats.protocol_adapters_passed += 1
    print_test_result(True, "BFD adapter test passed")
    return True


async def test_bgp_adapter(registry: PluginRegistry, stats: TestStats) -> bool:
    """BGP 어댑터 테스트 (TODO: Phase 2)."""
    print_test_header("BGP Protocol Adapter [Phase 2]")
    stats.protocol_adapters_tested += 1

    bgp_adapter = registry.get_protocol_adapter("bgp")
    if bgp_adapter is None:
        print_warning("BGP adapter not implemented yet (Phase 2)")
        print_info("Expected: BGP UPDATE analysis, AS-path monitoring")
        return False

    # TODO: Implement BGP adapter tests
    print_test_result(True, "BGP adapter test (to be implemented)")
    return True


async def test_ptp_adapter(registry: PluginRegistry, stats: TestStats) -> bool:
    """PTP 어댑터 테스트 (TODO: Phase 3)."""
    print_test_header("PTP Protocol Adapter [Phase 3]")
    stats.protocol_adapters_tested += 1

    ptp_adapter = registry.get_protocol_adapter("ptp")
    if ptp_adapter is None:
        print_warning("PTP adapter not implemented yet (Phase 3)")
        print_info("Expected: Time offset monitoring, drift detection")
        return False

    # TODO: Implement PTP adapter tests
    print_test_result(True, "PTP adapter test (to be implemented)")
    return True


# ============================================================================
# Detector Tests
# ============================================================================

async def test_lstm_detector(registry: PluginRegistry, stats: TestStats) -> bool:
    """LSTM 탐지기 테스트."""
    print_test_header("LSTM Detector")
    stats.detectors_tested += 1
    stats.record_memory()
    stats.record_cpu()

    lstm_detector = registry.get_detector("lstm")
    if lstm_detector is None:
        print_warning("LSTM detector not found (requires PyTorch)")
        return False

    print(f"✓ Found LSTM detector: {lstm_detector.name} v{lstm_detector.version}")
    print(f"  Supported protocols: {lstm_detector.supported_protocols}")

    # Verify protocol support
    expected_protocols = ["bfd", "bgp", "cfm", "ptp"]
    for protocol in expected_protocols:
        can_detect = lstm_detector.can_detect_protocol(protocol)
        status = "✓" if can_detect else "✗"
        print(f"  {status} Supports {protocol}: {can_detect}")

    # Test interface methods
    required_methods = ["train", "detect", "save_model", "load_model"]
    for method in required_methods:
        if not hasattr(lstm_detector, method):
            print_test_result(False, f"Missing method: {method}")
            return False

    print(f"✓ All required methods present: {required_methods}")
    stats.detectors_passed += 1
    print_test_result(True, "LSTM detector test passed")
    return True


async def test_hmm_detector(registry: PluginRegistry, stats: TestStats) -> bool:
    """HMM 탐지기 테스트."""
    print_test_header("HMM Detector")
    stats.detectors_tested += 1
    stats.record_memory()
    stats.record_cpu()

    hmm_detector = registry.get_detector("hmm")
    if hmm_detector is None:
        print_test_result(False, "HMM detector not found")
        return False

    print(f"✓ Found HMM detector: {hmm_detector.name} v{hmm_detector.version}")
    print(f"  Supported protocols: {hmm_detector.supported_protocols}")

    # Verify protocol support
    expected_protocols = ["bfd", "bgp"]
    for protocol in expected_protocols:
        can_detect = hmm_detector.can_detect_protocol(protocol)
        status = "✓" if can_detect else "✗"
        print(f"  {status} Supports {protocol}: {can_detect}")

    # Test interface methods
    required_methods = ["train", "detect", "save_model", "load_model"]
    for method in required_methods:
        if not hasattr(hmm_detector, method):
            print_test_result(False, f"Missing method: {method}")
            return False

    print(f"✓ All required methods present: {required_methods}")
    stats.detectors_passed += 1
    print_test_result(True, "HMM detector test passed")
    return True


async def test_gnn_detector(registry: PluginRegistry, stats: TestStats) -> bool:
    """GNN 탐지기 테스트 (TODO: Phase 2)."""
    print_test_header("GNN Detector [Phase 2]")
    stats.detectors_tested += 1

    gnn_detector = registry.get_detector("gnn")
    if gnn_detector is None:
        print_warning("GNN detector not implemented yet (Phase 2)")
        print_info("Expected: BGP graph structure analysis")
        return False

    # TODO: Implement GNN detector tests
    print_test_result(True, "GNN detector test (to be implemented)")
    return True


async def test_tcn_detector(registry: PluginRegistry, stats: TestStats) -> bool:
    """TCN 탐지기 테스트 (TODO: Phase 3)."""
    print_test_header("TCN Detector [Phase 3]")
    stats.detectors_tested += 1

    tcn_detector = registry.get_detector("tcn")
    if tcn_detector is None:
        print_warning("TCN detector not implemented yet (Phase 3)")
        print_info("Expected: PTP time series analysis")
        return False

    # TODO: Implement TCN detector tests
    print_test_result(True, "TCN detector test (to be implemented)")
    return True


# ============================================================================
# Integration Tests
# ============================================================================

async def test_cross_protocol_cfm_bfd(registry: PluginRegistry, stats: TestStats) -> bool:
    """크로스 프로토콜 테스트: CFM과 BFD 동시 모니터링."""
    print_test_header("Cross-Protocol Integration: CFM + BFD")
    stats.integration_tests_total += 1
    stats.record_memory()
    stats.record_cpu()

    cfm_adapter = registry.get_protocol_adapter("cfm")
    bfd_adapter = registry.get_protocol_adapter("bfd")

    if cfm_adapter is None or bfd_adapter is None:
        print_warning("CFM or BFD adapter not available")
        return False

    # Collect from both protocols simultaneously
    cfm_config = {
        "endpoints": [{"id": "test-cfm-1", "ip": "192.168.1.1", "port": 50000}],
        "interval_sec": 1,
    }

    bfd_config = {
        "sessions": [{
            "id": "test-bfd-1",
            "local_ip": "192.168.1.1",
            "remote_ip": "192.168.1.2",
            "interval_ms": 50,
            "multiplier": 3,
        }],
        "interval_sec": 1,
    }

    print("\nCollecting from both CFM and BFD simultaneously (5 samples each)...")
    cfm_metrics = []
    bfd_metrics = []

    try:
        # Create async iterators
        cfm_iterator = cfm_adapter.collect(cfm_config)
        bfd_iterator = bfd_adapter.collect(bfd_config)

        # Collect interleaved
        for _ in range(5):
            try:
                cfm_metric = await cfm_iterator.__anext__()
                cfm_metrics.append(cfm_metric)
                print(f"  CFM: {cfm_metric['metric_name']}={cfm_metric['value']:.2f}")
            except StopAsyncIteration:
                pass

            try:
                bfd_metric = await bfd_iterator.__anext__()
                bfd_metrics.append(bfd_metric)
                print(f"  BFD: {bfd_metric['metric_name']}={bfd_metric['value']:.2f}")
            except StopAsyncIteration:
                pass

    except Exception as e:
        print_test_result(False, f"Cross-protocol collection failed: {e}")
        return False

    print(f"\n✓ Collected {len(cfm_metrics)} CFM metrics and {len(bfd_metrics)} BFD metrics")

    if len(cfm_metrics) > 0 and len(bfd_metrics) > 0:
        stats.integration_tests_passed += 1
        print_test_result(True, "Cross-protocol CFM+BFD test passed")
        return True
    else:
        print_test_result(False, "Insufficient metrics collected")
        return False


async def test_cross_protocol_bgp_ptp(registry: PluginRegistry, stats: TestStats) -> bool:
    """크로스 프로토콜 테스트: BGP와 PTP 동시 모니터링 (TODO: Phase 2-3)."""
    print_test_header("Cross-Protocol Integration: BGP + PTP [Phase 2-3]")
    stats.integration_tests_total += 1

    bgp_adapter = registry.get_protocol_adapter("bgp")
    ptp_adapter = registry.get_protocol_adapter("ptp")

    if bgp_adapter is None or ptp_adapter is None:
        print_warning("BGP or PTP adapter not implemented yet")
        return False

    # TODO: Implement BGP+PTP integration tests
    print_test_result(True, "BGP+PTP integration test (to be implemented)")
    return True


# ============================================================================
# Performance Tests
# ============================================================================

async def test_performance_100_endpoints(registry: PluginRegistry, stats: TestStats) -> bool:
    """성능 테스트: 100개 엔드포인트 동시 수집."""
    print_test_header("Performance Test: 100 Endpoints")
    stats.integration_tests_total += 1

    cfm_adapter = registry.get_protocol_adapter("cfm")
    if cfm_adapter is None:
        print_warning("CFM adapter not available for performance testing")
        return False

    # Configure 100 endpoints
    config = {
        "endpoints": [
            {"id": f"perf-test-{i}", "ip": f"192.168.{i//256}.{i%256}", "port": 50000}
            for i in range(100)
        ],
        "interval_sec": 0.1,  # Fast collection
    }

    print("\nCollecting from 100 endpoints (10 seconds)...")
    start_time = time.time()
    collected_count = 0
    memory_samples = []

    try:
        async for metric in cfm_adapter.collect(config):
            collected_count += 1

            # Sample memory every 100 metrics
            if collected_count % 100 == 0:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(mem_mb)
                stats.record_memory()
                stats.record_cpu()

            # Stop after 10 seconds
            if time.time() - start_time > 10:
                break

    except Exception as e:
        print_test_result(False, f"Performance test failed: {e}")
        return False

    duration = time.time() - start_time
    throughput = collected_count / duration
    avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0

    print(f"\n✓ Performance results:")
    print(f"  - Total metrics collected: {collected_count}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Throughput: {throughput:.2f} metrics/sec")
    print(f"  - Average memory: {avg_memory:.2f} MB")

    # Performance criteria
    if throughput < 10:  # At least 10 metrics/sec
        print_test_result(False, f"Throughput too low: {throughput:.2f} < 10 metrics/sec")
        return False

    if avg_memory > 500:  # Less than 500 MB
        print_test_result(False, f"Memory usage too high: {avg_memory:.2f} > 500 MB")
        return False

    stats.integration_tests_passed += 1
    print_test_result(True, "Performance test passed")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_all_tests(args):
    """모든 테스트 실행."""
    print("=" * 70)
    print("OCAD 플러그인 통합 테스트 (Phase 4)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    stats = TestStats()
    stats.start_time = datetime.now()

    # Initialize plugin registry
    registry = PluginRegistry()
    plugin_dir = project_root / "ocad" / "plugins"

    print(f"\nDiscovering plugins from: {plugin_dir}")
    registry.discover_plugins(plugin_dir)

    # List all discovered plugins
    print("\n" + "=" * 70)
    print("Discovered Plugins")
    print("=" * 70)

    print("\nProtocol Adapters:")
    for adapter_name in registry.list_protocol_adapters():
        adapter = registry.get_protocol_adapter(adapter_name)
        print(f"  - {adapter.name} v{adapter.version}: {adapter.supported_metrics}")

    print("\nDetectors:")
    for detector_name in registry.list_detectors():
        detector = registry.get_detector(detector_name)
        print(f"  - {detector.name} v{detector.version}: {detector.supported_protocols}")

    # Define test suites
    protocol_tests = [
        ("CFM Adapter", test_cfm_adapter),
        ("BFD Adapter", test_bfd_adapter),
    ]

    if not args.quick:
        protocol_tests.extend([
            ("BGP Adapter", test_bgp_adapter),
            ("PTP Adapter", test_ptp_adapter),
        ])

    detector_tests = [
        ("LSTM Detector", test_lstm_detector),
        ("HMM Detector", test_hmm_detector),
    ]

    if not args.quick:
        detector_tests.extend([
            ("GNN Detector", test_gnn_detector),
            ("TCN Detector", test_tcn_detector),
        ])

    integration_tests = [
        ("CFM + BFD Integration", test_cross_protocol_cfm_bfd),
    ]

    if not args.quick:
        integration_tests.append(
            ("BGP + PTP Integration", test_cross_protocol_bgp_ptp)
        )

    if args.performance:
        integration_tests.append(
            ("100 Endpoints Performance", test_performance_100_endpoints)
        )

    # Run all tests
    results = []

    # Protocol adapter tests
    print("\n" + "=" * 70)
    print("PROTOCOL ADAPTER TESTS")
    print("=" * 70)
    for test_name, test_func in protocol_tests:
        try:
            passed = await test_func(registry, stats)
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Detector tests
    print("\n" + "=" * 70)
    print("DETECTOR TESTS")
    print("=" * 70)
    for test_name, test_func in detector_tests:
        try:
            passed = await test_func(registry, stats)
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Integration tests
    print("\n" + "=" * 70)
    print("INTEGRATION TESTS")
    print("=" * 70)
    for test_name, test_func in integration_tests:
        try:
            passed = await test_func(registry, stats)
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Finalize stats
    stats.end_time = datetime.now()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print("\nResults by category:")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n{stats.get_summary()}")

    # Overall result
    success_rate = passed_count / total_count * 100 if total_count > 0 else 0

    if passed_count == total_count:
        print(f"\n🎉 모든 테스트 통과! ({passed_count}/{total_count}, 100%)")
        print("\nNext steps:")
        print("  - Phase 2 완료 시 BGP 어댑터 및 GNN 탐지기 테스트 활성화")
        print("  - Phase 3 완료 시 PTP 어댑터 및 TCN 탐지기 테스트 활성화")
        return 0
    else:
        failed_count = total_count - passed_count
        print(f"\n⚠️ {failed_count}개 테스트 실패 ({passed_count}/{total_count}, {success_rate:.1f}%)")

        # Analyze failures
        failed_tests = [name for name, passed in results if not passed]
        print("\n실패한 테스트:")
        for test_name in failed_tests:
            print(f"  - {test_name}")

        return 1


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="OCAD 플러그인 통합 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_all_plugins.py                 # 전체 테스트
  python scripts/test_all_plugins.py --quick         # 빠른 테스트
  python scripts/test_all_plugins.py --performance   # 성능 테스트 포함
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 테스트만 실행 (구현된 플러그인만)"
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="성능 테스트 포함"
    )

    args = parser.parse_args()

    try:
        exit_code = asyncio.run(run_all_tests(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ 테스트 중단됨 (Ctrl+C)")
        sys.exit(130)


if __name__ == "__main__":
    main()
