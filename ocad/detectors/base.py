"""Base classes for anomaly detectors."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import structlog

from ..core.config import DetectionConfig
from ..core.logging import get_logger
from ..core.models import Capabilities, DetectionScore, FeatureVector, DetectionResult


logger = get_logger(__name__)


class BaseDetector(ABC):
    """Base class for all anomaly detectors."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize base detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.logger = logger.bind(component=self.__class__.__name__.lower())
    
    @abstractmethod
    def detect(self, features: FeatureVector, capabilities: Capabilities) -> float:
        """Detect anomalies in feature vector (레거시 인터페이스).

        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities

        Returns:
            Anomaly score between 0.0 and 1.0

        Note:
            이 메서드는 하위 호환성을 위해 유지됩니다.
            새로운 코드는 detect_detailed()를 사용하세요.
        """
        pass

    def detect_detailed(self, features: FeatureVector, capabilities: Capabilities) -> DetectionResult:
        """Detect anomalies with detailed information (권장).

        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities

        Returns:
            DetectionResult with detailed information

        Note:
            서브클래스에서 이 메서드를 오버라이드하면 상세 정보를 제공할 수 있습니다.
            오버라이드하지 않으면 기존 detect() 결과를 래핑합니다.
        """
        # 기본 구현: 기존 detect() 호출하고 DetectionResult로 래핑
        score = self.detect(features, capabilities)
        return DetectionResult(
            score=score,
            is_anomaly=score > 0.5,
            detector_name=self.__class__.__name__,
            metric_details={},
        )

    @abstractmethod
    def can_detect(self, capabilities: Capabilities) -> bool:
        """Check if this detector can work with given capabilities.
        
        Args:
            capabilities: Endpoint capabilities
            
        Returns:
            True if detector can work with these capabilities
        """
        pass
    
    def get_evidence(self, features: FeatureVector) -> Dict[str, float]:
        """Get evidence details for the detection.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary of evidence details
        """
        return {}


class CompositeDetector:
    """Composite detector that combines multiple detection algorithms."""
    
    def __init__(self, config: DetectionConfig, detectors: List[BaseDetector]):
        """Initialize composite detector.
        
        Args:
            config: Detection configuration
            detectors: List of individual detectors
        """
        self.config = config
        self.detectors = detectors
        self.logger = logger.bind(component="composite_detector")
    
    def detect(self, features: FeatureVector, capabilities: Capabilities) -> DetectionScore:
        """Run all applicable detectors and combine scores.
        
        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities
            
        Returns:
            Combined detection score
        """
        scores = {
            "rule_score": 0.0,
            "changepoint_score": 0.0,
            "residual_score": 0.0,
            "multivariate_score": 0.0,
        }
        
        evidence = {}
        
        # Run each detector
        for detector in self.detectors:
            if detector.can_detect(capabilities):
                try:
                    score = detector.detect(features, capabilities)
                    detector_type = detector.__class__.__name__.lower()
                    
                    # Map detector type to score field
                    if "rule" in detector_type:
                        scores["rule_score"] = score
                    elif "changepoint" in detector_type or "cusum" in detector_type:
                        scores["changepoint_score"] = score
                    elif "residual" in detector_type or "prediction" in detector_type:
                        scores["residual_score"] = score
                    elif "multivariate" in detector_type:
                        scores["multivariate_score"] = score
                    
                    # Collect evidence
                    detector_evidence = detector.get_evidence(features)
                    if detector_evidence:
                        evidence[detector_type] = detector_evidence
                        
                except Exception as e:
                    self.logger.error(
                        "Detector failed",
                        detector=detector.__class__.__name__,
                        endpoint_id=features.endpoint_id,
                        error=str(e),
                    )
        
        # Calculate composite score
        composite_score = (
            self.config.rule_weight * scores["rule_score"] +
            self.config.changepoint_weight * scores["changepoint_score"] +
            self.config.residual_weight * scores["residual_score"] +
            self.config.multivariate_weight * scores["multivariate_score"]
        )
        
        return DetectionScore(
            endpoint_id=features.endpoint_id,
            ts_ms=features.ts_ms,
            rule_score=scores["rule_score"],
            changepoint_score=scores["changepoint_score"],
            residual_score=scores["residual_score"],
            multivariate_score=scores["multivariate_score"],
            composite_score=min(1.0, composite_score),
            evidence=evidence,
        )
