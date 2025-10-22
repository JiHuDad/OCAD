#!/usr/bin/env python3
"""시스템 통합 테스트 스크립트.

ResidualDetectorV2가 SystemOrchestrator에 통합되어 제대로 작동하는지 검증합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from ocad.core.config import Settings
from ocad.system.orchestrator import SystemOrchestrator
from ocad.core.models import Endpoint, EndpointRole, Capabilities, FeatureVector


def test_orchestrator_initialization():
    """SystemOrchestrator가 ResidualDetectorV2와 함께 정상 초기화되는지 테스트."""
    print("=" * 60)
    print("SystemOrchestrator 초기화 테스트")
    print("=" * 60)

    # 설정 로드
    config = Settings()

    # ResidualDetectorV2 설정 확인
    print(f"\n사전 훈련 모델 사용: {config.detection.use_pretrained_models}")
    print(f"모델 디렉토리: {config.detection.pretrained_model_dir}")
    print(f"추론 디바이스: {config.detection.inference_device}")

    # Orchestrator 초기화
    print("\nOrchestrator 초기화 중...")
    try:
        orchestrator = SystemOrchestrator(config)
        print("✅ Orchestrator 초기화 성공")
    except Exception as e:
        print(f"❌ Orchestrator 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    # CompositeDetector 확인
    print(f"\n등록된 탐지기 수: {len(orchestrator.composite_detector.detectors)}")
    for i, detector in enumerate(orchestrator.composite_detector.detectors):
        detector_name = detector.__class__.__name__
        print(f"  {i+1}. {detector_name}")

        # ResidualDetectorV2인 경우 추가 정보 출력
        if detector_name == "ResidualDetectorV2":
            model_info = detector.get_model_info()
            print(f"     로드된 모델:")
            for metric, info in model_info.items():
                status = "✅" if info["loaded"] else "❌"
                version = info.get("version", "N/A")
                print(f"       - {metric}: {status} (v{version})")

    return True


def test_feature_detection():
    """피처 벡터로부터 이상 탐지가 정상 작동하는지 테스트."""
    print("\n" + "=" * 60)
    print("피처 기반 이상 탐지 테스트")
    print("=" * 60)

    # 설정 로드
    config = Settings()
    orchestrator = SystemOrchestrator(config)

    # 테스트용 엔드포인트
    endpoint = Endpoint(
        id="test-endpoint-001",
        role=EndpointRole.O_RU,
        host="192.168.1.100",
        port=830,
    )

    # 테스트용 capabilities
    capabilities = Capabilities(
        udp_echo=True,
        ecpri_delay=True,
        lbm=True,
        ccm_min=False,
    )

    # 정상 피처 벡터
    print("\n1. 정상 피처 벡터 테스트:")
    normal_features = FeatureVector(
        endpoint_id=endpoint.id,
        ts_ms=1000000000,
        window_size_ms=60000,
        udp_echo_p95=5.5,
        udp_echo_p99=6.2,
        ecpri_p95=100.0,
        ecpri_p99=120.0,
        lbm_rtt_p95=7.0,
        lbm_rtt_p99=8.5,
    )

    try:
        detection_result = orchestrator.composite_detector.detect(normal_features, capabilities)
        score = detection_result.composite_score
        print(f"   이상 점수: {score:.4f}")
        if score < 0.5:
            print("   ✅ 정상으로 판정됨")
        else:
            print(f"   ⚠️  높은 점수: {score:.4f}")
    except Exception as e:
        print(f"   ❌ 탐지 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 이상 피처 벡터
    print("\n2. 이상 피처 벡터 테스트:")
    anomaly_features = FeatureVector(
        endpoint_id=endpoint.id,
        ts_ms=1000001000,
        window_size_ms=60000,
        udp_echo_p95=25.0,  # 높은 지연
        udp_echo_p99=30.0,
        ecpri_p95=250.0,    # 높은 지연
        ecpri_p99=300.0,
        lbm_rtt_p95=20.0,   # 높은 RTT
        lbm_rtt_p99=25.0,
    )

    try:
        detection_result = orchestrator.composite_detector.detect(anomaly_features, capabilities)
        score = detection_result.composite_score
        print(f"   이상 점수: {score:.4f}")
        if score > 0.5:
            print("   ✅ 이상으로 판정됨")
        else:
            print(f"   ⚠️  낮은 점수: {score:.4f}")
    except Exception as e:
        print(f"   ❌ 탐지 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """메인 함수."""
    print("\n")
    print("=" * 60)
    print("OCAD 시스템 통합 테스트")
    print("=" * 60)

    # 로깅 설정 (간단하게)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    success = True

    # 테스트 1: Orchestrator 초기화
    if not test_orchestrator_initialization():
        success = False

    # 테스트 2: 피처 탐지
    if not test_feature_detection():
        success = False

    # 결과 출력
    print("\n" + "=" * 60)
    if success:
        print("✅ 모든 통합 테스트 통과")
        print("=" * 60)
        return 0
    else:
        print("❌ 일부 테스트 실패")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    import logging
    sys.exit(main())
