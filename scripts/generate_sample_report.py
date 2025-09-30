#!/usr/bin/env python3
"""
OCAD 사람 친화적 보고서 샘플 생성기
실제 알람 없이도 보고서 형식을 확인할 수 있습니다.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.config import Settings
from ocad.core.models import Alert, AlertEvidence, FeatureVector, Capabilities, Severity, EndpointRole, DetectionScore
from ocad.alerts.manager import AlertManager
from rich.console import Console

console = Console()

def create_sample_alert() -> tuple:
    """샘플 알람과 관련 데이터를 생성합니다."""
    
    # 샘플 피처 벡터 (실제 이상 상황을 시뮬레이션)
    features = FeatureVector(
        endpoint_id="sample-o-ru-001",
        ts_ms=int(datetime.now().timestamp() * 1000),
        window_size_ms=60000,
        sample_count=45,
        udp_echo_p95=18.3,
        udp_echo_p99=24.7,  # 임계값 5ms 대비 높음
        lbm_rtt_p95=6.1,
        lbm_rtt_p99=8.9,    # 임계값 5ms 대비 높음
        ecpri_p95=3.2,
        ecpri_p99=4.8,
        cusum_udp_echo=6.45,  # 높은 CUSUM 값
        cusum_ecpri=4.12,
        cusum_lbm=3.87,
        ewma_udp_echo=19.5,
        slope_udp_echo=2.1,
        lbm_fail_runlen=0
    )
    
    # 샘플 기능 설정
    capabilities = Capabilities(
        udp_echo=True,
        ecpri_delay=True,
        lbm=True,
        ccm_min=True,
        lldp=True
    )
    
    # 샘플 증거
    evidence = [
        AlertEvidence(
            type="rule",
            value=0.85,
            description="Rule violations: UDP Echo P99 24.7ms > 5.0ms; LBM RTT P99 8.9ms > 5.0ms (score: 0.85)",
            confidence=0.9
        ),
        AlertEvidence(
            type="spike",
            value=0.72,
            description="Sudden changes: UDP Echo CUSUM 6.45; eCPRI CUSUM 4.12; LBM CUSUM 3.87 (CUSUM score: 0.72)",
            confidence=0.8
        )
    ]
    
    # 샘플 DetectionScore
    detection_score = DetectionScore(
        endpoint_id="sample-o-ru-001",
        ts_ms=int(datetime.now().timestamp() * 1000),
        rule_score=0.85,
        changepoint_score=0.72,
        residual_score=0.15,
        multivariate_score=0.08,
        composite_score=0.416,
        evidence={
            "rule_violations": ["UDP Echo P99 exceeded", "LBM RTT P99 exceeded"],
            "cusum_triggers": ["UDP Echo", "eCPRI", "LBM"]
        }
    )
    
    # 샘플 알람
    alert = Alert(
        id="sample-alert-001",
        endpoint_id="sample-o-ru-001",
        ts_ms=features.ts_ms,
        severity=Severity.WARNING,
        title="O-RU 네트워크 성능 저하 감지",
        description="Anomaly detected with evidence: rule violations, changepoint spike. Performance degradation in O-RU network connectivity detected.",
        evidence=evidence,
        capabilities_snapshot=capabilities,
        feature_snapshot=features,
        score_snapshot=detection_score
    )
    
    return alert, features, capabilities

def main():
    """샘플 보고서를 생성합니다."""
    
    console.print("🧪 OCAD 사람 친화적 보고서 샘플 생성기")
    console.print("=" * 50)
    
    # 설정 및 AlertManager 초기화
    settings = Settings()
    alert_manager = AlertManager(
        alert_config=settings.alert,
        detection_config=settings.detection
    )
    
    # 샘플 데이터 생성
    alert, features, capabilities = create_sample_alert()
    
    # 사람 친화적 보고서 생성
    human_report = alert_manager.generate_human_readable_report(alert, features, capabilities)
    
    # 콘솔에 출력
    console.print("\n📖 생성된 사람 친화적 분석 보고서:")
    console.print("-" * 50)
    print(human_report)
    
    # 파일로 저장 (환경변수 또는 프로젝트 루트 기준)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 환경변수 OCAD_LOG_DIR이 있으면 사용, 없으면 프로젝트 루트의 logs 디렉토리
    if os.getenv('OCAD_LOG_DIR'):
        log_dir = Path(os.getenv('OCAD_LOG_DIR'))
    else:
        project_root = Path(__file__).parent.parent  # scripts의 상위 디렉토리
        log_dir = project_root / "logs"
    
    log_dir.mkdir(exist_ok=True)
    
    report_file = log_dir / f"sample_human_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(human_report)
    
    console.print(f"\n💾 보고서가 {report_file}에 저장되었습니다.")
    
    # 추가 샘플들
    console.print(f"\n🔄 다른 심각도 샘플들을 생성하시겠습니까? (y/N): ", end="")
    
    # Critical 레벨 샘플도 생성
    critical_features = FeatureVector(
        endpoint_id="sample-o-du-002",
        ts_ms=int(datetime.now().timestamp() * 1000),
        window_size_ms=60000,
        sample_count=42,
        udp_echo_p95=45.2,
        udp_echo_p99=67.8,  # 매우 높은 지연
        lbm_rtt_p95=38.1,
        lbm_rtt_p99=52.3,  # 매우 높은 지연
        ecpri_p95=25.4,
        ecpri_p99=41.6,    # 매우 높은 지연
        cusum_udp_echo=15.7,  # 매우 높은 CUSUM
        cusum_ecpri=12.4,
        cusum_lbm=9.8,
        ewma_udp_echo=58.3,
        slope_udp_echo=8.7,
        lbm_fail_runlen=5  # 연속 실패
    )
    
    critical_evidence = [
        AlertEvidence(
            type="rule",
            value=0.95,
            description="Rule violations: UDP Echo P99 67.8ms > 5.0ms; LBM RTT P99 52.3ms > 5.0ms; eCPRI P99 41.6ms > 5.0ms; LBM failures 5 consecutive (score: 0.95)",
            confidence=0.95
        ),
        AlertEvidence(
            type="spike",
            value=0.88,
            description="Sudden changes: UDP Echo CUSUM 15.7; eCPRI CUSUM 12.4; LBM CUSUM 9.8 (CUSUM score: 0.88)",
            confidence=0.85
        )
    ]
    
    # Critical DetectionScore
    critical_detection_score = DetectionScore(
        endpoint_id="sample-o-du-002",
        ts_ms=int(datetime.now().timestamp() * 1000),
        rule_score=0.95,
        changepoint_score=0.88,
        residual_score=0.75,
        multivariate_score=0.82,
        composite_score=0.847,
        evidence={
            "rule_violations": ["All metrics exceeded", "LBM consecutive failures"],
            "cusum_triggers": ["UDP Echo", "eCPRI", "LBM"],
            "severity": "critical"
        }
    )
    
    critical_capabilities = Capabilities(
        udp_echo=True,
        ecpri_delay=True,
        lbm=True,
        ccm_min=True,
        lldp=True
    )
    
    critical_alert = Alert(
        id="sample-critical-001",
        endpoint_id="sample-o-du-002",
        ts_ms=critical_features.ts_ms,
        severity=Severity.CRITICAL,
        title="O-DU 심각한 장애 상황",
        description="Critical anomaly detected: severe performance degradation and connectivity failures in O-DU equipment.",
        evidence=critical_evidence,
        capabilities_snapshot=critical_capabilities,
        feature_snapshot=critical_features,
        score_snapshot=critical_detection_score
    )
    
    # Critical 보고서 생성 및 저장
    critical_report = alert_manager.generate_human_readable_report(critical_alert, critical_features, critical_capabilities)
    critical_file = log_dir / f"sample_critical_report_{timestamp}.txt"
    
    with open(critical_file, 'w', encoding='utf-8') as f:
        f.write(critical_report)
    
    console.print(f"🔴 Critical 레벨 샘플 보고서가 {critical_file}에 저장되었습니다.")
    
    console.print("\n✅ 샘플 보고서 생성 완료!")
    console.print("이 보고서들을 참고하여 실제 알람 발생 시 어떤 정보가 제공되는지 확인하실 수 있습니다.")

if __name__ == "__main__":
    main()
