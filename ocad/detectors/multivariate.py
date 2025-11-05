"""Multivariate anomaly detector using Isolation Forest."""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..core.models import Capabilities, FeatureVector, DetectionResult, MetricDetectionDetail
from .base import BaseDetector


class MultivariateDetector(BaseDetector):
    """Multivariate anomaly detector using Isolation Forest."""
    
    def __init__(self, config):
        """Initialize multivariate detector.

        Args:
            config: Detection configuration
        """
        super().__init__(config)

        # Check if pre-trained models should be used
        self.use_pretrained = getattr(config, 'use_pretrained_models', False)
        self.model_dir = Path(getattr(config, 'multivariate_model_path', 'ocad/models/isolation_forest'))

        # Models for different endpoint groups
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Historical feature vectors for training
        self.feature_history: Dict[str, List[FeatureVector]] = {}

        self.min_training_samples = 50
        self.contamination = 0.1  # Expected anomaly rate

        # Load pre-trained model if enabled
        if self.use_pretrained:
            self._load_pretrained_model()
        
    def can_detect(self, capabilities: Capabilities) -> bool:
        """Check if multivariate detection is possible.
        
        Args:
            capabilities: Endpoint capabilities
            
        Returns:
            True if multiple metrics are available for correlation analysis
        """
        metric_count = sum([
            capabilities.udp_echo,
            capabilities.ecpri_delay,
            capabilities.lbm,
            capabilities.ccm_min,
        ])
        
        return metric_count >= 2  # Need at least 2 metrics for multivariate analysis
    
    def detect(self, features: FeatureVector, capabilities: Capabilities) -> float:
        """Detect multivariate anomalies using Isolation Forest.
        
        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities
            
        Returns:
            Anomaly score between 0.0 and 1.0
        """
        if not self.can_detect(capabilities):
            return 0.0
        
        # Determine group key based on capabilities
        group_key = self._get_group_key(capabilities)
        
        # Extract feature array
        feature_array = self._extract_feature_array(features, capabilities)
        if feature_array is None or len(feature_array) < 2:
            return 0.0
        
        # Add to history
        if group_key not in self.feature_history:
            self.feature_history[group_key] = []
        
        self.feature_history[group_key].append(features)
        
        # Keep only recent history
        if len(self.feature_history[group_key]) > 500:
            self.feature_history[group_key] = self.feature_history[group_key][-250:]
        
        # Train model if we have enough data
        if (group_key not in self.models and 
            len(self.feature_history[group_key]) >= self.min_training_samples):
            self._train_model(group_key, capabilities)
        
        # Make prediction if model is available
        if group_key in self.models:
            try:
                score = self._predict_anomaly(group_key, feature_array)
                return score
            except Exception as e:
                self.logger.debug(
                    "Multivariate prediction failed",
                    group_key=group_key,
                    error=str(e),
                )

        return 0.0

    def detect_detailed(self, features: FeatureVector, capabilities: Capabilities) -> DetectionResult:
        """Detect anomalies with detailed information.

        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities

        Returns:
            DetectionResult with multivariate analysis details
        """
        from datetime import datetime
        import time

        start_time = time.time()

        if not self.can_detect(capabilities):
            return DetectionResult(
                score=0.0,
                is_anomaly=False,
                detector_name="MultivariateDetector",
                metric_details={},
                explanation="다변량 분석에 필요한 메트릭이 부족합니다 (최소 2개 필요).",
                timestamp=datetime.now(),
                processing_time_ms=0.0,
            )

        # Determine group key
        group_key = self._get_group_key(capabilities)

        # Extract features
        feature_array = self._extract_feature_array(features, capabilities)
        if feature_array is None or len(feature_array) < 2:
            return DetectionResult(
                score=0.0,
                is_anomaly=False,
                detector_name="MultivariateDetector",
                metric_details={},
                explanation="유효한 피처를 추출할 수 없습니다.",
                timestamp=datetime.now(),
                processing_time_ms=0.0,
            )

        # Add to history
        if group_key not in self.feature_history:
            self.feature_history[group_key] = []

        self.feature_history[group_key].append(features)

        # Keep only recent history
        if len(self.feature_history[group_key]) > 500:
            self.feature_history[group_key] = self.feature_history[group_key][-250:]

        # Train model if needed
        if (group_key not in self.models and
            len(self.feature_history[group_key]) >= self.min_training_samples):
            self._train_model(group_key, capabilities)

        # Make prediction
        score = 0.0
        anomaly_score_raw = None
        if group_key in self.models:
            try:
                score = self._predict_anomaly(group_key, feature_array)
                # Get raw anomaly score from Isolation Forest
                anomaly_score_raw = self.models[group_key].score_samples(
                    self.scalers[group_key].transform(feature_array.reshape(1, -1))
                )[0]
            except Exception as e:
                self.logger.debug(
                    "Multivariate prediction failed",
                    group_key=group_key,
                    error=str(e),
                )

        # Create metric details
        metric_details = self._create_metric_details(features, capabilities, score, anomaly_score_raw)

        # Generate explanation
        explanation = self._generate_explanation(metric_details, score, len(feature_array))

        processing_time = (time.time() - start_time) * 1000

        return DetectionResult(
            score=score,
            is_anomaly=score > 0.5,
            detector_name="MultivariateDetector",
            metric_details=metric_details,
            dominant_metric="multivariate_pattern" if score > 0.5 else None,
            anomaly_type="multivariate_correlation" if score > 0.5 else None,
            confidence=score if score > 0 else None,
            explanation=explanation,
            timestamp=datetime.now(),
            processing_time_ms=processing_time,
        )

    def _create_metric_details(
        self, features: FeatureVector, capabilities: Capabilities, score: float, anomaly_score_raw: Optional[float]
    ) -> Dict[str, MetricDetectionDetail]:
        """Create metric details for multivariate detection.

        Args:
            features: Feature vector
            capabilities: Capabilities
            score: Normalized anomaly score
            anomaly_score_raw: Raw anomaly score from Isolation Forest

        Returns:
            Dictionary of metric details
        """
        details = {}

        # For multivariate detector, we create a single "pattern" entry
        # since it analyzes correlations rather than individual metrics
        details["multivariate_pattern"] = MetricDetectionDetail(
            metric_name="multivariate_pattern",
            actual_value=anomaly_score_raw if anomaly_score_raw else score,
            score=score,
            is_anomalous=score > 0.5,
            explanation=(
                f"메트릭 조합 패턴이 학습 데이터에서 {score:.1%} isolated (드문 패턴)"
                if score > 0.5
                else "메트릭 조합 패턴이 정상 범위 내"
            ),
        )

        # Also include individual metric values for context
        if capabilities.udp_echo and features.udp_echo_p95:
            details["udp_echo_context"] = MetricDetectionDetail(
                metric_name="udp_echo",
                actual_value=features.udp_echo_p95,
                score=0.0,  # Individual score not applicable
                is_anomalous=False,
                explanation=f"값: {features.udp_echo_p95:.2f} ms (조합 분석용)",
            )

        if capabilities.ecpri_delay and features.ecpri_p95:
            details["ecpri_context"] = MetricDetectionDetail(
                metric_name="ecpri",
                actual_value=features.ecpri_p95 / 1000.0,  # Convert to ms
                score=0.0,
                is_anomalous=False,
                explanation=f"값: {features.ecpri_p95 / 1000.0:.2f} ms (조합 분석용)",
            )

        if capabilities.lbm and features.lbm_rtt_p95:
            details["lbm_context"] = MetricDetectionDetail(
                metric_name="lbm",
                actual_value=features.lbm_rtt_p95,
                score=0.0,
                is_anomalous=False,
                explanation=f"값: {features.lbm_rtt_p95:.2f} ms (조합 분석용)",
            )

        return details

    def _generate_explanation(self, metric_details: Dict[str, MetricDetectionDetail], score: float, num_features: int) -> str:
        """Generate human-readable explanation.

        Args:
            metric_details: Metric detection details
            score: Final score
            num_features: Number of features analyzed

        Returns:
            Explanation string
        """
        if score < 0.3:
            return f"메트릭 조합 패턴이 정상입니다 ({num_features}개 피처 분석)."

        if score < 0.5:
            return f"메트릭 조합이 약간 이례적이지만 (점수: {score:.2f}), 임계값 이하입니다."

        # Extract context values
        udp_val = metric_details.get("udp_echo_context")
        ecpri_val = metric_details.get("ecpri_context")
        lbm_val = metric_details.get("lbm_context")

        parts = ["Isolation Forest가 비정상 패턴 탐지:"]

        if udp_val and ecpri_val:
            parts.append(
                f"UDP RTT={udp_val.actual_value:.2f}ms와 eCPRI Delay={ecpri_val.actual_value:.2f}ms의 조합이 학습 데이터에 없음"
            )
        elif udp_val and lbm_val:
            parts.append(
                f"UDP RTT={udp_val.actual_value:.2f}ms와 LBM RTT={lbm_val.actual_value:.2f}ms의 조합이 학습 데이터에 없음"
            )
        else:
            parts.append("개별 메트릭은 정상이지만, 메트릭 간 상관관계가 비정상")

        parts.append(f"(Isolation Score: {score:.2f}, {num_features}개 피처)")

        return " ".join(parts)

    def _get_group_key(self, capabilities: Capabilities) -> str:
        """Get group key for similar endpoints.
        
        Args:
            capabilities: Endpoint capabilities
            
        Returns:
            Group key string
        """
        # Group endpoints by similar capabilities
        caps_tuple = (
            capabilities.udp_echo,
            capabilities.ecpri_delay,
            capabilities.lbm,
            capabilities.ccm_min,
        )
        return str(caps_tuple)
    
    def _extract_feature_array(self, features: FeatureVector, capabilities: Capabilities) -> Optional[np.ndarray]:
        """Extract numerical feature array from feature vector.
        
        Args:
            features: Feature vector
            capabilities: Endpoint capabilities
            
        Returns:
            Numpy array of features or None
        """
        feature_list = []
        
        # UDP Echo features
        if capabilities.udp_echo:
            if features.udp_echo_p95 is not None:
                feature_list.append(features.udp_echo_p95)
            if features.udp_echo_p99 is not None:
                feature_list.append(features.udp_echo_p99)
            if features.udp_echo_slope is not None:
                feature_list.append(features.udp_echo_slope)
            if features.cusum_udp_echo is not None:
                feature_list.append(features.cusum_udp_echo)
        
        # eCPRI features
        if capabilities.ecpri_delay:
            if features.ecpri_p95 is not None:
                feature_list.append(features.ecpri_p95 / 1000.0)  # Convert to ms
            if features.ecpri_p99 is not None:
                feature_list.append(features.ecpri_p99 / 1000.0)
            if features.ecpri_slope is not None:
                feature_list.append(features.ecpri_slope)
            if features.cusum_ecpri is not None:
                feature_list.append(features.cusum_ecpri)
        
        # LBM features
        if capabilities.lbm:
            if features.lbm_rtt_p95 is not None:
                feature_list.append(features.lbm_rtt_p95)
            if features.lbm_rtt_p99 is not None:
                feature_list.append(features.lbm_rtt_p99)
            if features.lbm_slope is not None:
                feature_list.append(features.lbm_slope)
            if features.cusum_lbm is not None:
                feature_list.append(features.cusum_lbm)
            if features.lbm_fail_runlen is not None:
                feature_list.append(float(features.lbm_fail_runlen))
        
        # CCM features
        if capabilities.ccm_min:
            if features.ccm_miss_runlen is not None:
                feature_list.append(float(features.ccm_miss_runlen))
        
        if len(feature_list) < 2:
            return None
        
        return np.array(feature_list)
    
    def _train_model(self, group_key: str, capabilities: Capabilities) -> None:
        """Train Isolation Forest model for a group.
        
        Args:
            group_key: Group identifier
            capabilities: Endpoint capabilities
        """
        try:
            # Extract training data
            training_data = []
            for features in self.feature_history[group_key]:
                feature_array = self._extract_feature_array(features, capabilities)
                if feature_array is not None:
                    training_data.append(feature_array)
            
            if len(training_data) < self.min_training_samples:
                return
            
            # Convert to numpy array and handle different feature sizes
            max_features = max(len(arr) for arr in training_data)
            training_matrix = []
            
            for arr in training_data:
                # Pad with zeros if needed
                if len(arr) < max_features:
                    padded = np.zeros(max_features)
                    padded[:len(arr)] = arr
                    training_matrix.append(padded)
                else:
                    training_matrix.append(arr[:max_features])
            
            X = np.array(training_matrix)
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            model.fit(X_scaled)
            
            # Store model and scaler
            self.models[group_key] = model
            self.scalers[group_key] = scaler
            
            self.logger.info(
                "Multivariate model trained",
                group_key=group_key,
                training_samples=len(training_data),
                feature_dimension=max_features,
            )
            
        except Exception as e:
            self.logger.error(
                "Multivariate model training failed",
                group_key=group_key,
                error=str(e),
            )
    
    def _predict_anomaly(self, group_key: str, feature_array: np.ndarray) -> float:
        """Predict anomaly score using trained model.
        
        Args:
            group_key: Group identifier
            feature_array: Feature array
            
        Returns:
            Anomaly score between 0.0 and 1.0
        """
        model = self.models[group_key]
        scaler = self.scalers[group_key]
        
        # Get expected feature dimension from scaler
        expected_dim = scaler.n_features_in_
        
        # Pad or truncate feature array to match expected dimension
        if len(feature_array) < expected_dim:
            padded_array = np.zeros(expected_dim)
            padded_array[:len(feature_array)] = feature_array
            feature_array = padded_array
        elif len(feature_array) > expected_dim:
            feature_array = feature_array[:expected_dim]
        
        # Normalize features
        X_scaled = scaler.transform([feature_array])
        
        # Get anomaly score
        # Isolation Forest returns -1 for outliers, 1 for inliers
        anomaly_score = model.decision_function(X_scaled)[0]
        
        # Convert to 0-1 scale (lower scores = more anomalous)
        # decision_function typically returns values around [-0.5, 0.5]
        normalized_score = max(0.0, -anomaly_score)  # Invert and clip
        
        return min(1.0, normalized_score)
    
    def get_evidence(self, features: FeatureVector) -> Dict[str, float]:
        """Get evidence details for multivariate detection.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary of evidence details
        """
        evidence = {}
        
        # Add multivariate feature correlations
        metric_count = 0
        total_deviation = 0.0
        
        if features.udp_echo_p99 is not None:
            metric_count += 1
            # Simple deviation from expected baseline
            baseline_udp = 10.0  # Expected baseline
            deviation = abs(features.udp_echo_p99 - baseline_udp) / baseline_udp
            total_deviation += deviation
        
        if features.ecpri_p99 is not None:
            metric_count += 1
            baseline_ecpri = 100.0  # Expected baseline in microseconds
            deviation = abs(features.ecpri_p99 - baseline_ecpri) / baseline_ecpri
            total_deviation += deviation
        
        if features.lbm_rtt_p99 is not None:
            metric_count += 1
            baseline_lbm = 8.0  # Expected baseline
            deviation = abs(features.lbm_rtt_p99 - baseline_lbm) / baseline_lbm
            total_deviation += deviation
        
        if metric_count > 0:
            evidence["multivariate_deviation"] = total_deviation / metric_count
            evidence["correlated_metrics"] = metric_count

        return evidence

    def _load_pretrained_model(self) -> None:
        """Load pre-trained Isolation Forest model from disk."""
        try:
            model_path = self.model_dir / "isolation_forest_v1.0.0.pkl"
            scaler_path = self.model_dir / "isolation_forest_v1.0.0_scaler.pkl"
            metadata_path = self.model_dir / "isolation_forest_v1.0.0.json"

            if not model_path.exists() or not scaler_path.exists():
                self.logger.warning(
                    "Pre-trained Isolation Forest model not found",
                    model_path=str(model_path)
                )
                return

            # Load model and scaler
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Store as default model
            self.models['default'] = model
            self.scalers['default'] = scaler
            self.feature_names = metadata['metadata']['feature_names']

            self.logger.info(
                "Loaded pre-trained Isolation Forest model",
                version=metadata['metadata']['version'],
                n_features=metadata['metadata']['n_features'],
                contamination=metadata['hyperparameters']['contamination']
            )

        except Exception as e:
            self.logger.error(
                "Failed to load pre-trained Isolation Forest model",
                error=str(e)
            )
