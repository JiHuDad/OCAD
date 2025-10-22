#!/usr/bin/env python3
"""추론 성능 테스트 스크립트.

기존 온라인 학습 방식과 사전 훈련 모델 방식의 성능을 비교합니다.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ocad.detectors.residual_v2 import ResidualDetectorV2
from ocad.core.config import Settings
from ocad.core.models import Capabilities, FeatureVector
from ocad.core.logging import get_logger


logger = get_logger(__name__)


def create_sample_features(num_samples: int = 100) -> List[FeatureVector]:
    """테스트용 샘플 피처 벡터를 생성합니다.

    Args:
        num_samples: 생성할 샘플 수

    Returns:
        피처 벡터 리스트
    """
    features_list = []

    for i in range(num_samples):
        # 정상 범위의 값 생성 (간헐적으로 이상 포함)
        is_anomaly = i % 20 == 0  # 5% 이상

        if is_anomaly:
            udp_echo = np.random.uniform(15, 25)  # 이상
        else:
            udp_echo = np.random.uniform(3, 8)  # 정상

        features = FeatureVector(
            endpoint_id=f"test-endpoint-{i % 10}",
            ts_ms=int(time.time() * 1000) + i * 1000,
            window_size_ms=60000,  # 60초 윈도우
            udp_echo_p95=udp_echo,
            udp_echo_p99=udp_echo * 1.1,
            ecpri_p95=np.random.uniform(80, 150),
            ecpri_p99=np.random.uniform(100, 200),
            lbm_rtt_p95=np.random.uniform(5, 10),
            lbm_rtt_p99=np.random.uniform(6, 12),
        )

        features_list.append(features)

    return features_list


def measure_inference_latency(
    detector: ResidualDetectorV2,
    features_list: List[FeatureVector],
    capabilities: Capabilities,
) -> tuple[Dict[str, float], np.ndarray]:
    """추론 지연 시간을 측정합니다.

    Args:
        detector: 탐지기
        features_list: 피처 벡터 리스트
        capabilities: 엔드포인트 기능

    Returns:
        (지연 시간 통계, latencies 배열) 튜플
    """
    latencies = []

    for features in features_list:
        start = time.perf_counter()
        score = detector.detect(features, capabilities)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)

    stats = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
    }

    return stats, latencies


def measure_memory_usage():
    """메모리 사용량을 측정합니다.

    Returns:
        메모리 사용량 (MB)
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # MB


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="추론 성능 테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("ocad/models/tcn"),
        help="사전 훈련 모델 디렉토리",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="테스트 샘플 수",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="추론 디바이스",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("추론 성능 테스트")
    print("=" * 60)

    # 설정 로드
    settings = Settings()

    # 메모리 사용량 (시작)
    mem_before = measure_memory_usage()

    print(f"\n초기 메모리 사용량: {mem_before:.2f} MB")

    # 사전 훈련 모델 방식
    print(f"\n{'='*60}")
    print("사전 훈련 모델 방식 (추론 전용)")
    print(f"{'='*60}")

    detector_pretrained = ResidualDetectorV2(
        config=settings.detection,
        model_dir=args.model_dir,
        use_pretrained=True,
        device=args.device,
    )

    # 로드된 모델 정보
    model_info = detector_pretrained.get_model_info()
    print(f"\n로드된 모델:")
    for metric, info in model_info.items():
        if info.get("loaded"):
            print(f"  - {metric}: ✅ (scaler: {'✅' if info['has_scaler'] else '❌'})")
        else:
            print(f"  - {metric}: ❌")

    # 메모리 사용량 (모델 로드 후)
    mem_after_load = measure_memory_usage()
    mem_model = mem_after_load - mem_before

    print(f"\n모델 로드 후 메모리: {mem_after_load:.2f} MB (+{mem_model:.2f} MB)")

    # 테스트 데이터 생성
    print(f"\n테스트 데이터 생성 중... ({args.num_samples} 샘플)")
    features_list = create_sample_features(args.num_samples)

    capabilities = Capabilities(
        udp_echo=True,
        ecpri_delay=True,
        lbm=True,
        ccm_min=False,
    )

    # 추론 지연 측정
    print(f"\n추론 성능 측정 중...")
    latency_stats, latencies = measure_inference_latency(
        detector_pretrained,
        features_list,
        capabilities,
    )

    print(f"\n{'='*60}")
    print("추론 지연 시간 통계")
    print(f"{'='*60}")
    print(f"평균:     {latency_stats['mean_ms']:.3f} ms")
    print(f"중앙값:   {latency_stats['median_ms']:.3f} ms")
    print(f"P95:      {latency_stats['p95_ms']:.3f} ms")
    print(f"P99:      {latency_stats['p99_ms']:.3f} ms")
    print(f"최소:     {latency_stats['min_ms']:.3f} ms")
    print(f"최대:     {latency_stats['max_ms']:.3f} ms")
    print(f"표준편차: {latency_stats['std_ms']:.3f} ms")

    # 목표 달성 여부
    print(f"\n{'='*60}")
    print("목표 달성 여부")
    print(f"{'='*60}")

    p95_target = 100.0  # ms
    p99_target = 200.0  # ms

    p95_ok = latency_stats['p95_ms'] < p95_target
    p99_ok = latency_stats['p99_ms'] < p99_target

    print(f"P95 < {p95_target} ms: {'✅ 달성' if p95_ok else '❌ 미달성'} "
          f"(실제: {latency_stats['p95_ms']:.3f} ms)")
    print(f"P99 < {p99_target} ms: {'✅ 달성' if p99_ok else '❌ 미달성'} "
          f"(실제: {latency_stats['p99_ms']:.3f} ms)")

    # 메모리 사용량 (최종)
    mem_final = measure_memory_usage()
    mem_inference = mem_final - mem_after_load

    print(f"\n{'='*60}")
    print("메모리 사용량")
    print(f"{'='*60}")
    print(f"시작:         {mem_before:.2f} MB")
    print(f"모델 로드 후: {mem_after_load:.2f} MB (+{mem_model:.2f} MB)")
    print(f"추론 후:      {mem_final:.2f} MB (+{mem_inference:.2f} MB)")
    print(f"총 증가량:    {mem_final - mem_before:.2f} MB")

    # 처리량 계산
    total_time = sum(latencies) / 1000  # seconds
    throughput = args.num_samples / total_time if total_time > 0 else 0

    print(f"\n{'='*60}")
    print("처리량")
    print(f"{'='*60}")
    print(f"총 샘플:      {args.num_samples}")
    print(f"총 시간:      {total_time:.3f} 초")
    print(f"처리량:       {throughput:.2f} 샘플/초")

    # 저장
    import json
    report_path = Path("ocad/models/metadata/inference_performance_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": args.num_samples,
        "device": args.device,
        "latency_statistics": latency_stats,
        "memory_usage_mb": {
            "initial": mem_before,
            "after_model_load": mem_after_load,
            "model_size": mem_model,
            "after_inference": mem_final,
            "inference_overhead": mem_inference,
        },
        "throughput_samples_per_sec": throughput,
        "targets_met": {
            "p95_under_100ms": p95_ok,
            "p99_under_200ms": p99_ok,
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n성능 리포트 저장: {report_path}")
    print(f"\n{'='*60}\n")

    return 0 if (p95_ok and p99_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
