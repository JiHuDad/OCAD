"""Rule-based anomaly detector."""

from typing import Dict

from ..core.models import Capabilities, FeatureVector
from .base import BaseDetector


class RuleBasedDetector(BaseDetector):
    """Rule-based anomaly detector using simple thresholds."""
    
    def can_detect(self, capabilities: Capabilities) -> bool:
        """Rule-based detector can work with any capabilities.
        
        Args:
            capabilities: Endpoint capabilities
            
        Returns:
            Always True
        """
        return True
    
    def detect(self, features: FeatureVector, capabilities: Capabilities) -> float:
        """Detect anomalies using rule-based thresholds.
        
        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities
            
        Returns:
            Anomaly score between 0.0 and 1.0
        """
        score = 0.0
        violations = 0
        total_checks = 0
        
        # UDP Echo rules
        if capabilities.udp_echo and features.udp_echo_p99 is not None:
            total_checks += 1
            if features.udp_echo_p99 > self.config.rule_p99_threshold_ms:
                violations += 1
                self.logger.debug(
                    "UDP echo p99 threshold violated",
                    endpoint_id=features.endpoint_id,
                    p99=features.udp_echo_p99,
                    threshold=self.config.rule_p99_threshold_ms,
                )
        
        # eCPRI delay rules (convert microseconds to milliseconds)
        if capabilities.ecpri_delay and features.ecpri_p99 is not None:
            total_checks += 1
            ecpri_p99_ms = features.ecpri_p99 / 1000.0  # Convert µs to ms
            if ecpri_p99_ms > self.config.rule_p99_threshold_ms:
                violations += 1
                self.logger.debug(
                    "eCPRI delay p99 threshold violated",
                    endpoint_id=features.endpoint_id,
                    p99_ms=ecpri_p99_ms,
                    threshold=self.config.rule_p99_threshold_ms,
                )
        
        # LBM rules
        if capabilities.lbm:
            if features.lbm_rtt_p99 is not None:
                total_checks += 1
                if features.lbm_rtt_p99 > self.config.rule_p99_threshold_ms:
                    violations += 1
                    self.logger.debug(
                        "LBM RTT p99 threshold violated",
                        endpoint_id=features.endpoint_id,
                        p99=features.lbm_rtt_p99,
                        threshold=self.config.rule_p99_threshold_ms,
                    )
            
            # Failure run-length rule
            if features.lbm_fail_runlen is not None:
                total_checks += 1
                if features.lbm_fail_runlen > 3:  # More than 3 consecutive failures
                    violations += 1
                    self.logger.debug(
                        "LBM failure run-length violated",
                        endpoint_id=features.endpoint_id,
                        fail_runlen=features.lbm_fail_runlen,
                    )
        
        # CCM rules
        if capabilities.ccm_min and features.ccm_miss_runlen is not None:
            total_checks += 1
            if features.ccm_miss_runlen > 5:  # More than 5 consecutive misses
                violations += 1
                self.logger.debug(
                    "CCM miss run-length violated",
                    endpoint_id=features.endpoint_id,
                    miss_runlen=features.ccm_miss_runlen,
                )
        
        # Calculate score
        if total_checks > 0:
            score = violations / total_checks
        
        # Debug log for score calculation
        if score > 0:
            self.logger.debug(
                "Rule-based score calculated",
                endpoint_id=features.endpoint_id,
                violations=violations,
                total_checks=total_checks,
                score=score,
            )
        
        return min(1.0, score)
    
    def get_evidence(self, features: FeatureVector) -> Dict[str, float]:
        """Get evidence details for rule violations.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary of evidence details
        """
        evidence = {}
        
        if features.udp_echo_p99 is not None:
            evidence["udp_echo_p99"] = features.udp_echo_p99
        
        if features.ecpri_p99 is not None:
            evidence["ecpri_p99_ms"] = features.ecpri_p99 / 1000.0
        
        if features.lbm_rtt_p99 is not None:
            evidence["lbm_rtt_p99"] = features.lbm_rtt_p99
        
        if features.lbm_fail_runlen is not None:
            evidence["lbm_fail_runlen"] = features.lbm_fail_runlen
        
        if features.ccm_miss_runlen is not None:
            evidence["ccm_miss_runlen"] = features.ccm_miss_runlen
        
        return evidence
