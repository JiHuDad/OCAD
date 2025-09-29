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
        
        # Check evidence threshold
        if len(evidence) < self.alert_config.min_evidence_for_alert:
            self.logger.debug(
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
        
        # Rule-based evidence
        if score.rule_score > 0.3:
            evidence.append(AlertEvidence(
                type="rule",
                value=score.rule_score,
                description=f"Rule violations detected (score: {score.rule_score:.2f})",
                confidence=0.9,
            ))
        
        # Change point evidence
        if score.changepoint_score > 0.3:
            evidence.append(AlertEvidence(
                type="spike",
                value=score.changepoint_score,
                description=f"Sudden change detected (CUSUM score: {score.changepoint_score:.2f})",
                confidence=0.8,
            ))
        
        # Residual evidence (drift)
        if score.residual_score > 0.3:
            evidence.append(AlertEvidence(
                type="drift",
                value=score.residual_score,
                description=f"Performance drift detected (residual score: {score.residual_score:.2f})",
                confidence=0.7,
            ))
        
        # Multivariate evidence (concurrent)
        if score.multivariate_score > 0.3:
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
