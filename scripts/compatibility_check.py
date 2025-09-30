#!/usr/bin/env python3
"""
OCAD 시스템 호환성 체크 스크립트
다른 플랫폼에서 기본 동작을 검증합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """필수 모듈 import 테스트"""
    print("📦 Import 테스트...")
    try:
        from ocad.core.config import Settings
        from ocad.core.models import Alert, AlertEvidence, FeatureVector, Capabilities, Severity, DetectionScore
        from ocad.alerts.manager import AlertManager
        print("✅ 모든 핵심 모듈 import 성공")
        return True
    except Exception as e:
        print(f"❌ Import 실패: {e}")
        return False

def check_alert_model():
    """Alert 모델 호환성 테스트"""
    print("🔍 Alert 모델 호환성 테스트...")
    try:
        from ocad.core.models import Alert, AlertEvidence, FeatureVector, Capabilities, Severity, DetectionScore
        from datetime import datetime
        
        # 샘플 데이터 생성
        capabilities = Capabilities(
            udp_echo=True,
            ecpri_delay=True,
            lbm=True,
            ccm_min=True,
            lldp=True
        )
        
        evidence = [
            AlertEvidence(
                type="rule",
                value=0.85,
                description="Test rule violation",
                confidence=0.9
            )
        ]
        
        detection_score = DetectionScore(
            endpoint_id="test-endpoint",
            ts_ms=int(datetime.now().timestamp() * 1000),
            rule_score=0.85,
            changepoint_score=0.72,
            residual_score=0.15,
            multivariate_score=0.08,
            composite_score=0.416
        )
        
        alert = Alert(
            id="test-alert-001",
            endpoint_id="test-endpoint",
            ts_ms=int(datetime.now().timestamp() * 1000),
            severity=Severity.WARNING,
            title="테스트 알람",
            description="호환성 테스트용 알람",
            evidence=evidence,
            capabilities_snapshot=capabilities,
            score_snapshot=detection_score
        )
        
        # 속성 접근 테스트
        composite_score = alert.score_snapshot.composite_score if alert.score_snapshot else 0.0
        print(f"✅ Alert 모델 정상 - composite_score: {composite_score}")
        return True
        
    except Exception as e:
        print(f"❌ Alert 모델 테스트 실패: {e}")
        return False

def check_logging_config():
    """로깅 설정 호환성 테스트"""
    print("📝 로깅 설정 테스트...")
    try:
        from ocad.core.logging import configure_logging
        import tempfile
        
        # 임시 디렉토리에 로그 설정 테스트
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_logging(log_level="INFO", enable_json=False, log_dir=temp_dir)
            print("✅ 로깅 설정 정상")
        return True
        
    except Exception as e:
        print(f"❌ 로깅 설정 실패: {e}")
        return False

def check_human_readable_report():
    """사람 친화적 보고서 생성 테스트"""
    print("📖 사람 친화적 보고서 테스트...")
    try:
        from ocad.core.config import Settings
        from ocad.core.models import Alert, AlertEvidence, FeatureVector, Capabilities, Severity, DetectionScore
        from ocad.alerts.manager import AlertManager
        from datetime import datetime
        
        # 설정 및 AlertManager 초기화
        settings = Settings()
        alert_manager = AlertManager(
            alert_config=settings.alert,
            detection_config=settings.detection
        )
        
        # 테스트 데이터 생성
        capabilities = Capabilities(
            udp_echo=True,
            ecpri_delay=True,
            lbm=True,
            ccm_min=True,
            lldp=True
        )
        
        features = FeatureVector(
            endpoint_id="test-endpoint",
            ts_ms=int(datetime.now().timestamp() * 1000),
            window_size_ms=60000,
            sample_count=50,
            udp_echo_p95=15.0,
            udp_echo_p99=18.5,
            lbm_rtt_p95=4.2,
            lbm_rtt_p99=6.8,
            ecpri_p95=3.1,
            ecpri_p99=4.5,
            cusum_udp_echo=5.82,
            cusum_ecpri=3.41,
            cusum_lbm=2.98
        )
        
        evidence = [
            AlertEvidence(
                type="rule",
                value=0.85,
                description="Rule violations: UDP Echo P99 18.5ms > 5.0ms",
                confidence=0.9
            )
        ]
        
        detection_score = DetectionScore(
            endpoint_id="test-endpoint",
            ts_ms=int(datetime.now().timestamp() * 1000),
            rule_score=0.85,
            changepoint_score=0.72,
            residual_score=0.15,
            multivariate_score=0.08,
            composite_score=0.416
        )
        
        alert = Alert(
            id="test-alert-001",
            endpoint_id="test-endpoint",
            ts_ms=int(datetime.now().timestamp() * 1000),
            severity=Severity.WARNING,
            title="테스트 알람",
            description="호환성 테스트용 알람",
            evidence=evidence,
            capabilities_snapshot=capabilities,
            feature_snapshot=features,
            score_snapshot=detection_score
        )
        
        # 사람 친화적 보고서 생성 테스트
        human_report = alert_manager.generate_human_readable_report(alert, features, capabilities)
        
        if "OCAD 이상탐지 알람 분석 보고서" in human_report:
            print("✅ 사람 친화적 보고서 생성 정상")
            return True
        else:
            print("❌ 보고서 형식 오류")
            return False
            
    except Exception as e:
        print(f"❌ 사람 친화적 보고서 테스트 실패: {e}")
        return False

def check_path_handling():
    """경로 처리 호환성 테스트"""
    print("📁 경로 처리 테스트...")
    try:
        from pathlib import Path
        import tempfile
        import os
        
        # 프로젝트 루트 경로 확인
        project_root = Path(__file__).parent.parent
        logs_dir = project_root / "logs"
        
        print(f"프로젝트 루트: {project_root}")
        print(f"로그 디렉토리: {logs_dir}")
        
        # 환경변수 테스트
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ['OCAD_LOG_DIR'] = temp_dir
            custom_log_dir = Path(os.getenv('OCAD_LOG_DIR'))
            print(f"커스텀 로그 디렉토리: {custom_log_dir}")
            
        print("✅ 경로 처리 정상")
        return True
        
    except Exception as e:
        print(f"❌ 경로 처리 실패: {e}")
        return False

def main():
    """메인 호환성 체크"""
    print("🔧 OCAD 시스템 호환성 체크 시작")
    print("=" * 50)
    
    tests = [
        ("기본 Import", check_imports),
        ("Alert 모델", check_alert_model),
        ("로깅 설정", check_logging_config),
        ("사람 친화적 보고서", check_human_readable_report),
        ("경로 처리", check_path_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} 테스트:")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} 테스트에서 문제 발견")
    
    print("\n" + "=" * 50)
    print(f"🎯 결과: {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("✅ 모든 호환성 테스트 통과! OCAD 시스템이 이 플랫폼에서 정상 동작할 것으로 예상됩니다.")
        return True
    else:
        print(f"❌ {total - passed}개 테스트 실패. 해당 기능에서 문제가 발생할 수 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
