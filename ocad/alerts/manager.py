"""알람 관리 및 통지 시스템.
스마트한 알람 생성, 중복 제거, 억제, 생명주기 관리를 담당합니다.
"""

import time
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Optional

import structlog

from ..core.config import AlertConfig, DetectionConfig
from ..core.logging import get_logger
from ..core.models import Alert, AlertEvidence, Capabilities, DetectionScore, FeatureVector, Severity


logger = get_logger(__name__)


class AlertManager:
    """알람 생명주기, 중복 제거, 통지를 관리하는 클래스.
    
    주요 기능:
    - 근거 3개 원칙에 따른 스마트 알람 생성
    - Hold-down을 통한 알람 폭주 방지
    - 중복 제거를 통한 노이즈 감소
    - 심각도별 자동 분류 및 관리
    """
    
    def __init__(self, alert_config: AlertConfig, detection_config: DetectionConfig):
        """알람 관리자를 초기화합니다.
        
        Args:
            alert_config: 알람 설정
            detection_config: 탐지 설정
        """
        self.alert_config = alert_config
        self.detection_config = detection_config
        self.logger = logger.bind(component="alert_manager")
        
        # 활성 알람 및 억제 관리
        self.active_alerts: Dict[str, Alert] = {}  # 현재 활성 알람들
        self.suppressed_endpoints: Dict[str, float] = {}  # 엔드포인트별 억제 종료 시간
        self.recent_alerts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))  # 최근 알람 이력
        
        # 중복 제거 추적
        self.last_alert_times: Dict[str, float] = {}  # 엔드포인트별 마지막 알람 시간
        
    def process_detection(
        self, 
        score: DetectionScore, 
        features: FeatureVector, 
        capabilities: Capabilities
    ) -> Optional[Alert]:
        """Process detection score and potentially generate alert.
        
        Args:
            score: Detection score
            features: Feature vector
            capabilities: Endpoint capabilities
            
        Returns:
            Generated alert or None
        """
        endpoint_id = score.endpoint_id
        current_time = time.time()
        
        # Check if endpoint is suppressed
        if self._is_suppressed(endpoint_id, current_time):
            return None
        
        # Calculate severity
        severity = self._calculate_severity(score)
        
        if severity == Severity.INFO:
            return None  # Don't alert on INFO level
        
        # Check deduplication
        if self._is_duplicate(endpoint_id, severity, current_time):
            return None
        
        # Generate evidence
        evidence = self._generate_evidence(score, features, capabilities)
        
        # Debug: always log evidence count
        self.logger.info(
            "Evidence check",
            endpoint_id=endpoint_id,
            evidence_count=len(evidence),
            min_required=self.alert_config.min_evidence_for_alert,
            rule_score=score.rule_score,
            changepoint_score=score.changepoint_score,
            composite_score=score.composite_score,
        )
        
        # Check evidence threshold
        if len(evidence) < self.alert_config.min_evidence_for_alert:
            self.logger.info(
                "Insufficient evidence for alert",
                endpoint_id=endpoint_id,
                evidence_count=len(evidence),
                min_required=self.alert_config.min_evidence_for_alert,
            )
            return None
        
        # Create alert
        alert = self._create_alert(
            endpoint_id=endpoint_id,
            severity=severity,
            score=score,
            features=features,
            capabilities=capabilities,
            evidence=evidence,
        )
        
        # Track for deduplication
        self.last_alert_times[endpoint_id] = current_time
        self.recent_alerts[endpoint_id].append(alert)
        
        # Add to active alerts
        self.active_alerts[alert.id] = alert
        
        # Apply hold-down suppression
        self.suppressed_endpoints[endpoint_id] = current_time + self.detection_config.hold_down_seconds
        
        self.logger.info(
            "Alert generated",
            alert_id=alert.id,
            endpoint_id=endpoint_id,
            severity=severity,
            composite_score=score.composite_score,
            evidence_count=len(evidence),
        )
        
        return alert
    
    def _is_suppressed(self, endpoint_id: str, current_time: float) -> bool:
        """Check if endpoint is currently suppressed.
        
        Args:
            endpoint_id: Endpoint identifier
            current_time: Current timestamp
            
        Returns:
            True if suppressed
        """
        suppression_end = self.suppressed_endpoints.get(endpoint_id, 0)
        return current_time < suppression_end
    
    def _is_duplicate(self, endpoint_id: str, severity: Severity, current_time: float) -> bool:
        """Check if this would be a duplicate alert.
        
        Args:
            endpoint_id: Endpoint identifier
            severity: Alert severity
            current_time: Current timestamp
            
        Returns:
            True if this is a duplicate
        """
        last_alert_time = self.last_alert_times.get(endpoint_id, 0)
        time_since_last = current_time - last_alert_time
        
        # Deduplication window based on severity
        dedup_window = self.detection_config.dedup_window_seconds
        if severity == Severity.CRITICAL:
            dedup_window //= 2  # Shorter window for critical alerts
        
        return time_since_last < dedup_window
    
    def _calculate_severity(self, score: DetectionScore) -> Severity:
        """Calculate alert severity based on composite score.
        
        Args:
            score: Detection score
            
        Returns:
            Alert severity
        """
        composite_score = score.composite_score
        
        if composite_score >= self.alert_config.severity_buckets["critical"]:
            return Severity.CRITICAL
        elif composite_score >= self.alert_config.severity_buckets["warning"]:
            return Severity.WARNING
        else:
            return Severity.INFO
    
    def _generate_evidence(
        self, 
        score: DetectionScore, 
        features: FeatureVector, 
        capabilities: Capabilities
    ) -> List[AlertEvidence]:
        """Generate evidence list for the alert.
        
        Args:
            score: Detection score
            features: Feature vector
            capabilities: Endpoint capabilities
            
        Returns:
            List of evidence items
        """
        evidence = []
        
        # Rule-based evidence with detailed analysis
        if score.rule_score > 0.1:
            rule_details = self._analyze_rule_violations(features, capabilities)
            evidence.append(AlertEvidence(
                type="rule",
                value=score.rule_score,
                description=f"Rule violations: {rule_details} (score: {score.rule_score:.2f})",
                confidence=0.9,
            ))
        
        # Change point evidence with detailed analysis
        if score.changepoint_score > 0.1:
            cusum_details = self._analyze_cusum_changes(features, capabilities)
            evidence.append(AlertEvidence(
                type="spike",
                value=score.changepoint_score,
                description=f"Sudden changes: {cusum_details} (CUSUM score: {score.changepoint_score:.2f})",
                confidence=0.8,
            ))
        
        # Residual evidence (drift)
        if score.residual_score > 0.2:
            evidence.append(AlertEvidence(
                type="drift",
                value=score.residual_score,
                description=f"Performance drift detected (residual score: {score.residual_score:.2f})",
                confidence=0.7,
            ))
        
        # Multivariate evidence (concurrent)
        if score.multivariate_score > 0.2:
            evidence.append(AlertEvidence(
                type="concurrent",
                value=score.multivariate_score,
                description=f"Correlated anomalies detected (multivariate score: {score.multivariate_score:.2f})",
                confidence=0.6,
            ))
        
        # Add specific metric evidence
        if capabilities.udp_echo and features.udp_echo_p99:
            if features.udp_echo_p99 > self.detection_config.rule_p99_threshold_ms:
                evidence.append(AlertEvidence(
                    type="rule",
                    value=features.udp_echo_p99,
                    description=f"UDP echo p99 latency high: {features.udp_echo_p99:.1f}ms",
                    confidence=0.95,
                ))
        
        if capabilities.lbm and features.lbm_fail_runlen:
            if features.lbm_fail_runlen > 3:
                evidence.append(AlertEvidence(
                    type="rule",
                    value=features.lbm_fail_runlen,
                    description=f"LBM consecutive failures: {features.lbm_fail_runlen}",
                    confidence=0.9,
                ))
        
        # Sort by confidence and take top evidence
        evidence.sort(key=lambda x: x.confidence, reverse=True)
        return evidence[:self.alert_config.evidence_count]
    
    def _create_alert(
        self,
        endpoint_id: str,
        severity: Severity,
        score: DetectionScore,
        features: FeatureVector,
        capabilities: Capabilities,
        evidence: List[AlertEvidence],
    ) -> Alert:
        """Create alert object.
        
        Args:
            endpoint_id: Endpoint identifier
            severity: Alert severity
            score: Detection score
            features: Feature vector
            capabilities: Endpoint capabilities
            evidence: Evidence list
            
        Returns:
            Created alert
        """
        alert_id = str(uuid.uuid4())
        
        # Generate title and description
        title = f"{severity.title()} anomaly detected on {endpoint_id}"
        
        evidence_types = [e.type for e in evidence]
        description = f"Anomaly detected with evidence: {', '.join(evidence_types)}"
        
        if severity == Severity.CRITICAL:
            description += ". Immediate attention required."
        elif severity == Severity.WARNING:
            description += ". Performance degradation detected."
        
        return Alert(
            id=alert_id,
            endpoint_id=endpoint_id,
            ts_ms=score.ts_ms,
            severity=severity,
            title=title,
            description=description,
            evidence=evidence,
            capabilities_snapshot=capabilities,
            feature_snapshot=features,
            score_snapshot=score,
        )
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            user: User acknowledging the alert
            
        Returns:
            True if alert was acknowledged
        """
        alert = self.active_alerts.get(alert_id)
        if alert and not alert.acknowledged:
            alert.acknowledged = True
            alert.updated_at = time.time()
            
            self.logger.info(
                "Alert acknowledged",
                alert_id=alert_id,
                endpoint_id=alert.endpoint_id,
                user=user,
            )
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert identifier
            user: User resolving the alert
            
        Returns:
            True if alert was resolved
        """
        alert = self.active_alerts.get(alert_id)
        if alert and not alert.resolved:
            alert.resolved = True
            alert.updated_at = time.time()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(
                "Alert resolved",
                alert_id=alert_id,
                endpoint_id=alert.endpoint_id,
                user=user,
            )
            return True
        
        return False
    
    def suppress_endpoint(self, endpoint_id: str, duration_seconds: int) -> None:
        """Suppress alerts for an endpoint.
        
        Args:
            endpoint_id: Endpoint to suppress
            duration_seconds: Suppression duration
        """
        current_time = time.time()
        self.suppressed_endpoints[endpoint_id] = current_time + duration_seconds
        
        self.logger.info(
            "Endpoint suppressed",
            endpoint_id=endpoint_id,
            duration_seconds=duration_seconds,
        )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts.
        
        Returns:
            List of active alerts
        """
        return list(self.active_alerts.values())
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        current_time = time.time()
        
        # Clean up old suppressions
        expired_suppressions = [
            ep_id for ep_id, end_time in self.suppressed_endpoints.items()
            if current_time > end_time
        ]
        for ep_id in expired_suppressions:
            del self.suppressed_endpoints[ep_id]
        
        return {
            "active_alerts": len(self.active_alerts),
            "suppressed_endpoints": len(self.suppressed_endpoints),
            "total_recent_alerts": sum(len(alerts) for alerts in self.recent_alerts.values()),
        }
    
    def _analyze_rule_violations(self, features: FeatureVector, capabilities: Capabilities) -> str:
        """알람의 룰 위반 원인을 상세 분석합니다.
        
        Args:
            features: 피처 벡터
            capabilities: 엔드포인트 기능
            
        Returns:
            위반 원인 상세 설명
        """
        violations = []
        threshold = self.detection_config.rule_p99_threshold_ms
        
        if capabilities.udp_echo and features.udp_echo_p99 is not None:
            if features.udp_echo_p99 > threshold:
                violations.append(f"UDP Echo P99 {features.udp_echo_p99:.1f}ms > {threshold}ms")
        
        if capabilities.ecpri_delay and features.ecpri_p99 is not None:
            if features.ecpri_p99 > threshold:
                violations.append(f"eCPRI P99 {features.ecpri_p99:.1f}ms > {threshold}ms")
        
        if capabilities.lbm and features.lbm_rtt_p99 is not None:
            if features.lbm_rtt_p99 > threshold:
                violations.append(f"LBM RTT P99 {features.lbm_rtt_p99:.1f}ms > {threshold}ms")
        
        if features.lbm_fail_runlen is not None and features.lbm_fail_runlen > self.detection_config.rule_runlength_threshold:
            violations.append(f"LBM failures {features.lbm_fail_runlen} consecutive")
        
        return "; ".join(violations) if violations else "Unknown rule violations"
    
    def _analyze_cusum_changes(self, features: FeatureVector, capabilities: Capabilities) -> str:
        """CUSUM 변화점 원인을 상세 분석합니다.
        
        Args:
            features: 피처 벡터
            capabilities: 엔드포인트 기능
            
        Returns:
            변화점 원인 상세 설명
        """
        changes = []
        threshold = self.detection_config.cusum_threshold / 50.0  # 스케일링 적용
        
        if capabilities.udp_echo and features.cusum_udp_echo is not None:
            if features.cusum_udp_echo > threshold:
                changes.append(f"UDP Echo CUSUM {features.cusum_udp_echo:.2f}")
        
        if capabilities.ecpri_delay and features.cusum_ecpri is not None:
            if features.cusum_ecpri > threshold:
                changes.append(f"eCPRI CUSUM {features.cusum_ecpri:.2f}")
        
        if capabilities.lbm and features.cusum_lbm is not None:
            if features.cusum_lbm > threshold:
                changes.append(f"LBM CUSUM {features.cusum_lbm:.2f}")
        
        return "; ".join(changes) if changes else "Unknown CUSUM changes"
    
    def generate_human_readable_report(self, alert: "Alert", features: FeatureVector, capabilities: Capabilities) -> str:
        """사람이 이해하기 쉬운 알람 분석 보고서를 생성합니다.
        
        Args:
            alert: 알람 객체
            features: 피처 벡터
            capabilities: 엔드포인트 기능
            
        Returns:
            사람이 읽기 쉬운 분석 보고서
        """
        from datetime import datetime
        
        # 시간 변환
        timestamp = datetime.fromtimestamp(alert.ts_ms / 1000.0)
        
        report = []
        report.append("=" * 80)
        report.append(f"🚨 OCAD 이상탐지 알람 분석 보고서")
        report.append("=" * 80)
        report.append("")
        
        # 기본 정보
        report.append("📍 기본 정보")
        report.append("-" * 40)
        report.append(f"• 엔드포인트: {alert.endpoint_id}")
        report.append(f"• 탐지 시간: {timestamp.strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}")
        report.append(f"• 심각도: {self._get_severity_description(alert.severity)}")
        composite_score = alert.score_snapshot.composite_score if alert.score_snapshot else 0.0
        report.append(f"• 종합 위험도: {composite_score:.1%} (1.0 = 100% 위험)")
        report.append("")
        
        # 문제 요약
        report.append("⚠️ 발견된 문제")
        report.append("-" * 40)
        problem_summary = self._generate_problem_summary(alert, features, capabilities)
        for line in problem_summary:
            report.append(f"• {line}")
        report.append("")
        
        # 상세 기술 분석
        report.append("🔍 상세 기술 분석")
        report.append("-" * 40)
        tech_analysis = self._generate_technical_analysis(alert, features, capabilities)
        for line in tech_analysis:
            report.append(f"  {line}")
        report.append("")
        
        # 영향도 분석
        report.append("📊 영향도 분석")
        report.append("-" * 40)
        impact_analysis = self._generate_impact_analysis(alert, features, capabilities)
        for line in impact_analysis:
            report.append(f"• {line}")
        report.append("")
        
        # 권장 조치사항
        report.append("💡 권장 조치사항")
        report.append("-" * 40)
        recommendations = self._generate_recommendations(alert, features, capabilities)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # 모니터링 포인트
        report.append("👀 지속 모니터링 포인트")
        report.append("-" * 40)
        monitoring_points = self._generate_monitoring_points(alert, features, capabilities)
        for line in monitoring_points:
            report.append(f"• {line}")
        report.append("")
        
        report.append("=" * 80)
        report.append("보고서 생성: OCAD (O-RAN CFM-Lite AI 이상탐지 시스템)")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _get_severity_description(self, severity) -> str:
        """심각도를 사람이 이해하기 쉽게 설명합니다."""
        descriptions = {
            "critical": "🔴 심각 - 즉시 대응 필요",
            "warning": "🟡 경고 - 주의 깊은 모니터링 필요", 
            "info": "🔵 정보 - 참고용"
        }
        return descriptions.get(severity.value.lower(), f"🔘 {severity.value}")
    
    def _generate_problem_summary(self, alert, features: FeatureVector, capabilities: Capabilities) -> list:
        """문제 요약을 생성합니다."""
        problems = []
        
        # 네트워크 지연 문제
        threshold = self.detection_config.rule_p99_threshold_ms
        if capabilities.udp_echo and features.udp_echo_p99 and features.udp_echo_p99 > threshold:
            delay = features.udp_echo_p99
            if delay > threshold * 3:
                problems.append(f"UDP Echo 응답시간이 매우 느림 ({delay:.1f}ms, 정상: {threshold}ms 이하)")
            elif delay > threshold * 2:
                problems.append(f"UDP Echo 응답시간이 느림 ({delay:.1f}ms, 정상: {threshold}ms 이하)")
            else:
                problems.append(f"UDP Echo 응답시간 지연 ({delay:.1f}ms, 정상: {threshold}ms 이하)")
        
        if capabilities.lbm and features.lbm_rtt_p99 and features.lbm_rtt_p99 > threshold:
            delay = features.lbm_rtt_p99
            problems.append(f"LBM 루프백 지연 증가 ({delay:.1f}ms, 정상: {threshold}ms 이하)")
        
        if capabilities.ecpri_delay and features.ecpri_p99 and features.ecpri_p99 > threshold:
            delay = features.ecpri_p99
            problems.append(f"eCPRI 지연 증가 ({delay:.1f}ms, 정상: {threshold}ms 이하)")
        
        # 급격한 변화 감지
        cusum_threshold = self.detection_config.cusum_threshold / 50.0
        if features.cusum_udp_echo and features.cusum_udp_echo > cusum_threshold:
            problems.append("UDP Echo 성능이 급격히 변화함 (CUSUM 이상 탐지)")
        
        if features.cusum_lbm and features.cusum_lbm > cusum_threshold:
            problems.append("LBM 루프백 성능이 급격히 변화함 (CUSUM 이상 탐지)")
        
        return problems if problems else ["알 수 없는 네트워크 이상이 감지됨"]
    
    def _generate_technical_analysis(self, alert, features: FeatureVector, capabilities: Capabilities) -> list:
        """기술적 상세 분석을 생성합니다."""
        analysis = []
        
        # 통계적 분석
        analysis.append("📈 성능 지표 분석:")
        if features.udp_echo_p95 and features.udp_echo_p99:
            analysis.append(f"   UDP Echo - P95: {features.udp_echo_p95:.1f}ms, P99: {features.udp_echo_p99:.1f}ms")
        if features.lbm_rtt_p95 and features.lbm_rtt_p99:
            analysis.append(f"   LBM RTT - P95: {features.lbm_rtt_p95:.1f}ms, P99: {features.lbm_rtt_p99:.1f}ms")
        
        # CUSUM 분석
        if any([features.cusum_udp_echo, features.cusum_ecpri, features.cusum_lbm]):
            analysis.append("")
            analysis.append("📊 변화점 탐지 (CUSUM) 분석:")
            if features.cusum_udp_echo:
                analysis.append(f"   UDP Echo 변화량: {features.cusum_udp_echo:.2f}")
            if features.cusum_ecpri:
                analysis.append(f"   eCPRI 변화량: {features.cusum_ecpri:.2f}")
            if features.cusum_lbm:
                analysis.append(f"   LBM 변화량: {features.cusum_lbm:.2f}")
        
        # 증거 강도 분석
        analysis.append("")
        analysis.append("🔍 탐지 알고리즘별 신뢰도:")
        for evidence in alert.evidence:
            confidence_desc = "매우 높음" if evidence.confidence > 0.8 else "높음" if evidence.confidence > 0.6 else "보통"
            analysis.append(f"   {evidence.type.upper()} 탐지: {evidence.confidence:.0%} ({confidence_desc})")
        
        return analysis
    
    def _generate_impact_analysis(self, alert, features: FeatureVector, capabilities: Capabilities) -> list:
        """영향도 분석을 생성합니다."""
        impact = []
        
        # 서비스 영향도
        if alert.severity.value.lower() == "critical":
            impact.append("🔴 서비스에 심각한 영향을 미칠 수 있음")
            impact.append("사용자 체감 품질 저하 가능성이 높음")
        elif alert.severity.value.lower() == "warning":
            impact.append("🟡 서비스 품질에 영향을 미칠 수 있음")
            impact.append("지속될 경우 사용자 불만이 증가할 수 있음")
        
        # 네트워크 영향도
        threshold = self.detection_config.rule_p99_threshold_ms
        if features.udp_echo_p99 and features.udp_echo_p99 > threshold * 2:
            impact.append("네트워크 연결성에 심각한 문제가 있을 가능성")
        elif features.udp_echo_p99 and features.udp_echo_p99 > threshold:
            impact.append("네트워크 성능 저하가 감지됨")
        
        # O-RAN 특화 영향도
        if alert.endpoint_id.startswith("sim-o-ru"):
            impact.append("O-RU 장비 문제로 무선 접속에 영향 가능")
        elif alert.endpoint_id.startswith("sim-o-du"):
            impact.append("O-DU 장비 문제로 기지국 처리 성능에 영향 가능")
        elif "transport" in alert.endpoint_id:
            impact.append("전송 구간 문제로 전체 네트워크에 영향 가능")
        
        return impact if impact else ["영향도를 정확히 판단하기 어려움"]
    
    def _generate_recommendations(self, alert, features: FeatureVector, capabilities: Capabilities) -> list:
        """권장 조치사항을 생성합니다."""
        recommendations = []
        
        # 즉시 조치사항
        if alert.severity.value.lower() == "critical":
            recommendations.append("즉시 네트워크 관리자에게 알림")
            recommendations.append("해당 엔드포인트의 상세 진단 실시")
        
        # 네트워크 관련 조치
        threshold = self.detection_config.rule_p99_threshold_ms
        if features.udp_echo_p99 and features.udp_echo_p99 > threshold:
            recommendations.append("네트워크 연결 상태 및 대역폭 확인")
            recommendations.append("라우팅 경로 및 홉 수 점검")
        
        if features.lbm_rtt_p99 and features.lbm_rtt_p99 > threshold:
            recommendations.append("CFM(Connectivity Fault Management) 설정 확인")
            recommendations.append("이더넷 링크 상태 점검")
        
        # CUSUM 기반 조치
        if any([features.cusum_udp_echo, features.cusum_ecpri, features.cusum_lbm]):
            recommendations.append("최근 네트워크 설정 변경사항 검토")
            recommendations.append("트래픽 패턴 변화 분석")
        
        # O-RAN 특화 조치
        if alert.endpoint_id.startswith("sim-o-ru"):
            recommendations.append("O-RU 하드웨어 상태 점검")
            recommendations.append("무선 인터페이스 설정 확인")
        elif alert.endpoint_id.startswith("sim-o-du"):
            recommendations.append("O-DU 처리 용량 및 CPU 사용률 확인")
            recommendations.append("베어러 설정 및 QoS 정책 검토")
        
        # 일반적 조치
        recommendations.append("15분 후 상태 재확인")
        recommendations.append("유사한 패턴의 알람이 다른 장비에서 발생하는지 확인")
        
        return recommendations
    
    def _generate_monitoring_points(self, alert, features: FeatureVector, capabilities: Capabilities) -> list:
        """지속 모니터링 포인트를 생성합니다."""
        points = []
        
        # 핵심 메트릭 모니터링
        if capabilities.udp_echo:
            points.append(f"UDP Echo 응답시간이 {self.detection_config.rule_p99_threshold_ms}ms 이하로 회복되는지 확인")
        
        if capabilities.lbm:
            points.append("LBM 루프백 지연이 정상 범위로 돌아오는지 관찰")
        
        # 트렌드 모니터링
        points.append("향후 30분간 동일한 패턴의 이상이 재발하는지 모니터링")
        points.append("다른 엔드포인트에서 유사한 증상이 나타나는지 확인")
        
        # 비즈니스 영향 모니터링
        points.append("사용자 불만 접수 현황 모니터링")
        points.append("전체 네트워크 성능 지표 추이 관찰")
        
        return points
