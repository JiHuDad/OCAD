"""Residual-based anomaly detector using predictive models."""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..core.models import Capabilities, FeatureVector, DetectionResult, MetricDetectionDetail
from .base import BaseDetector


class SimpleTCN(nn.Module):
    """Simple Temporal Convolutional Network for time series prediction."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32, output_size: int = 1):
        """Initialize TCN.
        
        Args:
            input_size: Input feature size
            hidden_size: Hidden layer size
            output_size: Output size
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, output_size, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor [batch, features, sequence]
            
        Returns:
            Output tensor
        """
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)
        return x[:, :, -1]  # Return last timestep


class ResidualDetector(BaseDetector):
    """Residual-based detector using predictive models."""
    
    def __init__(self, config):
        """Initialize residual detector.

        Args:
            config: Detection configuration
        """
        super().__init__(config)

        # Check if pre-trained models should be used
        self.use_pretrained = getattr(config, 'use_pretrained_models', False)
        self.model_dir = Path(getattr(config, 'model_path', 'ocad/models/tcn'))

        # Models for different metrics
        self.models = {
            "udp_echo": None,
            "ecpri": None,
            "lbm": None,
        }

        # Data scalers
        self.scalers = {
            "udp_echo": StandardScaler(),
            "ecpri": StandardScaler(),
            "lbm": StandardScaler(),
        }

        # Historical data for training
        self.history = {
            "udp_echo": [],
            "ecpri": [],
            "lbm": [],
        }

        self.sequence_length = 10  # Use last 10 points for prediction
        self.min_training_samples = 50

        # Load pre-trained models if enabled
        if self.use_pretrained:
            self._load_pretrained_models()
        
    def can_detect(self, capabilities: Capabilities) -> bool:
        """Check if residual detection is possible.
        
        Args:
            capabilities: Endpoint capabilities
            
        Returns:
            True if any predictive metric is available
        """
        return any([
            capabilities.udp_echo,
            capabilities.ecpri_delay,
            capabilities.lbm,
        ])
    
    def detect(self, features: FeatureVector, capabilities: Capabilities) -> float:
        """Detect anomalies using prediction residuals.
        
        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities
            
        Returns:
            Anomaly score between 0.0 and 1.0
        """
        residuals = []
        
        # Update history and calculate residuals for each metric
        if capabilities.udp_echo and features.udp_echo_p95 is not None:
            residual = self._calculate_residual(
                "udp_echo", 
                features.udp_echo_p95,
                features.endpoint_id
            )
            if residual is not None:
                residuals.append(residual)
        
        if capabilities.ecpri_delay and features.ecpri_p95 is not None:
            # Convert microseconds to milliseconds for consistency
            ecpri_ms = features.ecpri_p95 / 1000.0
            residual = self._calculate_residual(
                "ecpri",
                ecpri_ms,
                features.endpoint_id
            )
            if residual is not None:
                residuals.append(residual)
        
        if capabilities.lbm and features.lbm_rtt_p95 is not None:
            residual = self._calculate_residual(
                "lbm",
                features.lbm_rtt_p95,
                features.endpoint_id
            )
            if residual is not None:
                residuals.append(residual)
        
        if not residuals:
            return 0.0
        
        # Calculate max normalized residual
        max_residual = max(residuals)
        
        # Normalize score using threshold
        score = min(1.0, max_residual / self.config.residual_threshold)
        
        if score > 0.5:
            self.logger.debug(
                "High prediction residual detected",
                endpoint_id=features.endpoint_id,
                max_residual=max_residual,
                residuals=residuals,
            )

        return score

    def detect_detailed(self, features: FeatureVector, capabilities: Capabilities) -> DetectionResult:
        """Detect anomalies with detailed information.

        Args:
            features: Feature vector to analyze
            capabilities: Endpoint capabilities

        Returns:
            DetectionResult with prediction details
        """
        from datetime import datetime
        start_time = time.time()

        metric_details = {}
        residuals = []

        # UDP Echo
        if capabilities.udp_echo and features.udp_echo_p95 is not None:
            detail = self._calculate_residual_detailed(
                "udp_echo",
                features.udp_echo_p95,
                features.endpoint_id
            )
            if detail:
                metric_details["udp_echo"] = detail
                residuals.append(detail.normalized_error if detail.normalized_error else 0)

        # eCPRI Delay
        if capabilities.ecpri_delay and features.ecpri_p95 is not None:
            ecpri_ms = features.ecpri_p95 / 1000.0
            detail = self._calculate_residual_detailed(
                "ecpri",
                ecpri_ms,
                features.endpoint_id
            )
            if detail:
                metric_details["ecpri"] = detail
                residuals.append(detail.normalized_error if detail.normalized_error else 0)

        # LBM RTT
        if capabilities.lbm and features.lbm_rtt_p95 is not None:
            detail = self._calculate_residual_detailed(
                "lbm",
                features.lbm_rtt_p95,
                features.endpoint_id
            )
            if detail:
                metric_details["lbm"] = detail
                residuals.append(detail.normalized_error if detail.normalized_error else 0)

        # Calculate final score
        if not residuals:
            score = 0.0
            dominant_metric = None
        else:
            max_residual = max(residuals)
            score = min(1.0, max_residual / self.config.residual_threshold)

            # Find dominant metric
            dominant_metric = None
            max_detail_score = 0
            for metric_name, detail in metric_details.items():
                if detail.score > max_detail_score:
                    max_detail_score = detail.score
                    dominant_metric = metric_name

        processing_time = (time.time() - start_time) * 1000

        return DetectionResult(
            score=score,
            is_anomaly=score > 0.5,
            detector_name="ResidualDetector",
            metric_details=metric_details,
            dominant_metric=dominant_metric,
            anomaly_type="prediction_residual" if score > 0.5 else None,
            confidence=min(1.0, score * 1.2) if score > 0 else None,
            explanation=self._generate_explanation(metric_details, score),
            timestamp=datetime.now(),
            processing_time_ms=processing_time,
        )

    def _generate_explanation(self, metric_details: Dict[str, MetricDetectionDetail], score: float) -> str:
        """Generate human-readable explanation.

        Args:
            metric_details: Metric detection details
            score: Final score

        Returns:
            Explanation string
        """
        if score < 0.3:
            return "모든 메트릭이 예측 범위 내에 있습니다."

        anomalous_metrics = [
            name for name, detail in metric_details.items()
            if detail.is_anomalous
        ]

        if not anomalous_metrics:
            return f"예측 오차가 약간 높지만 (점수: {score:.2f}), 임계값 이하입니다."

        metric_names_kr = {
            "udp_echo": "UDP Echo RTT",
            "ecpri": "eCPRI Delay",
            "lbm": "LBM RTT"
        }

        parts = []
        for metric_name in anomalous_metrics:
            detail = metric_details[metric_name]
            kr_name = metric_names_kr.get(metric_name, metric_name)

            if detail.predicted_value and detail.actual_value and detail.error:
                error_pct = abs(detail.error / detail.predicted_value * 100) if detail.predicted_value > 0 else 0
                parts.append(
                    f"{kr_name}: 예측값 {detail.predicted_value:.2f}에 비해 "
                    f"실제값 {detail.actual_value:.2f}이(가) {error_pct:.0f}% 차이남"
                )

        if parts:
            return "TCN 모델 예측 실패: " + ", ".join(parts)
        else:
            return f"{len(anomalous_metrics)}개 메트릭에서 예측 오차 발생"

    def _calculate_residual(self, metric_type: str, value: float, endpoint_id: str) -> Optional[float]:
        """Calculate prediction residual for a metric.
        
        Args:
            metric_type: Type of metric (udp_echo, ecpri, lbm)
            value: Current metric value
            endpoint_id: Endpoint identifier
            
        Returns:
            Normalized residual or None
        """
        # Add to history
        self.history[metric_type].append(value)
        
        # Keep only recent history (for memory efficiency)
        if len(self.history[metric_type]) > 1000:
            self.history[metric_type] = self.history[metric_type][-500:]
        
        # Need enough data for prediction
        if len(self.history[metric_type]) < self.sequence_length + 1:
            return None
        
        try:
            # Train model if needed
            if self.models[metric_type] is None and len(self.history[metric_type]) >= self.min_training_samples:
                self._train_model(metric_type)
            
            # Make prediction if model is available
            if self.models[metric_type] is not None:
                sequence = self.history[metric_type][-self.sequence_length-1:-1]
                prediction = self._predict(metric_type, sequence)

                if prediction is not None:
                    residual = abs(value - prediction)

                    # Normalize residual
                    # 사전 학습 모델 사용 시: 학습 데이터의 std로 정규화
                    # 온라인 학습 시: 최근 데이터의 std로 정규화
                    if hasattr(self, 'residual_normalizers') and metric_type in self.residual_normalizers:
                        # 사전 학습 모델 사용 중
                        std_dev = self.residual_normalizers[metric_type]
                        if std_dev > 0:
                            normalized_residual = residual / std_dev
                            return normalized_residual
                    else:
                        # 온라인 학습 중 - 최근 데이터의 std 사용
                        recent_values = self.history[metric_type][-20:]
                        if len(recent_values) > 3:
                            std_dev = np.std(recent_values)
                            if std_dev > 0:
                                normalized_residual = residual / std_dev
                                return normalized_residual

                    # Fallback: 정규화 불가능한 경우 원본 residual 반환
                    return residual
            
        except Exception as e:
            self.logger.debug(
                "Residual calculation failed",
                metric_type=metric_type,
                endpoint_id=endpoint_id,
                error=str(e),
            )
        
        return None

    def _calculate_residual_detailed(
        self, metric_type: str, value: float, endpoint_id: str
    ) -> Optional[MetricDetectionDetail]:
        """Calculate prediction residual with detailed information.

        Args:
            metric_type: Type of metric (udp_echo, ecpri, lbm)
            value: Current metric value
            endpoint_id: Endpoint identifier

        Returns:
            MetricDetectionDetail with prediction information or None
        """
        # Add to history
        self.history[metric_type].append(value)

        # Keep only recent history
        if len(self.history[metric_type]) > 1000:
            self.history[metric_type] = self.history[metric_type][-500:]

        # Need enough data for prediction
        if len(self.history[metric_type]) < self.sequence_length + 1:
            return None

        try:
            # Train model if needed
            if self.models[metric_type] is None and len(self.history[metric_type]) >= self.min_training_samples:
                self._train_model(metric_type)

            # Make prediction if model is available
            if self.models[metric_type] is not None:
                sequence = self.history[metric_type][-self.sequence_length-1:-1]
                prediction = self._predict(metric_type, sequence)

                if prediction is not None:
                    residual = abs(value - prediction)
                    error = value - prediction  # Signed error

                    # Normalize residual
                    normalized_residual = None
                    if hasattr(self, 'residual_normalizers') and metric_type in self.residual_normalizers:
                        std_dev = self.residual_normalizers[metric_type]
                        if std_dev > 0:
                            normalized_residual = residual / std_dev
                    else:
                        recent_values = self.history[metric_type][-20:]
                        if len(recent_values) > 3:
                            std_dev = np.std(recent_values)
                            if std_dev > 0:
                                normalized_residual = residual / std_dev

                    if normalized_residual is None:
                        normalized_residual = residual

                    # Calculate score
                    score = min(1.0, normalized_residual / self.config.residual_threshold)
                    is_anomalous = score > 0.5

                    # Generate explanation
                    if is_anomalous:
                        error_pct = abs(error / prediction * 100) if prediction > 0 else 0
                        explanation = (
                            f"예측값 {prediction:.2f}에 비해 실제값 {value:.2f}이(가) "
                            f"{error_pct:.0f}% 차이 (오차: {error:+.2f}, {normalized_residual:.1f}σ)"
                        )
                    else:
                        explanation = f"예측 범위 내 (오차: {error:+.2f})"

                    return MetricDetectionDetail(
                        metric_name=metric_type,
                        actual_value=value,
                        predicted_value=prediction,
                        error=error,
                        normalized_error=normalized_residual,
                        score=score,
                        is_anomalous=is_anomalous,
                        explanation=explanation,
                    )

        except Exception as e:
            self.logger.debug(
                "Detailed residual calculation failed",
                metric_type=metric_type,
                endpoint_id=endpoint_id,
                error=str(e),
            )

        return None

    def _train_model(self, metric_type: str) -> None:
        """Train prediction model for a metric type.
        
        Args:
            metric_type: Type of metric to train model for
        """
        try:
            data = np.array(self.history[metric_type])
            
            # Normalize data
            data_scaled = self.scalers[metric_type].fit_transform(data.reshape(-1, 1)).flatten()
            
            # Create training sequences
            X, y = [], []
            for i in range(len(data_scaled) - self.sequence_length):
                X.append(data_scaled[i:i + self.sequence_length])
                y.append(data_scaled[i + self.sequence_length])
            
            if len(X) < 10:  # Not enough sequences
                return
            
            X = np.array(X).reshape(-1, 1, self.sequence_length)
            y = np.array(y)
            
            # Create and train model
            model = SimpleTCN(input_size=1, hidden_size=16, output_size=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Simple training loop
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            model.train()
            for epoch in range(20):  # Quick training
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()
            
            model.eval()
            self.models[metric_type] = model
            
            self.logger.debug(
                "Model trained",
                metric_type=metric_type,
                training_samples=len(X),
                final_loss=loss.item(),
            )
            
        except Exception as e:
            self.logger.error(
                "Model training failed",
                metric_type=metric_type,
                error=str(e),
            )
    
    def _predict(self, metric_type: str, sequence: List[float]) -> Optional[float]:
        """Make prediction using trained model.
        
        Args:
            metric_type: Type of metric
            sequence: Input sequence
            
        Returns:
            Predicted value or None
        """
        try:
            model = self.models[metric_type]
            if model is None:
                return None
            
            # Normalize sequence
            sequence_array = np.array(sequence).reshape(-1, 1)
            sequence_scaled = self.scalers[metric_type].transform(sequence_array).flatten()
            
            # Make prediction
            with torch.no_grad():
                x_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).unsqueeze(0)
                prediction_scaled = model(x_tensor).item()
            
            # Denormalize prediction
            prediction = self.scalers[metric_type].inverse_transform([[prediction_scaled]])[0][0]
            
            return prediction
            
        except Exception as e:
            self.logger.debug(
                "Prediction failed",
                metric_type=metric_type,
                error=str(e),
            )
            return None
    
    def _load_pretrained_models(self) -> None:
        """Load pre-trained TCN models from disk."""
        import joblib

        model_files = {
            "udp_echo": "udp_echo_v2.0.1",
            "ecpri": "ecpri_v2.0.1",
            "lbm": "lbm_v2.0.1",
        }

        for metric_type, model_name in model_files.items():
            try:
                model_path = self.model_dir / f"{model_name}.pth"
                metadata_path = self.model_dir / f"{model_name}.json"
                scaler_path = self.model_dir / f"{model_name}_scaler.pkl"

                if not model_path.exists() or not metadata_path.exists():
                    self.logger.warning(
                        f"Pre-trained model not found for {metric_type}",
                        model_path=str(model_path)
                    )
                    continue

                # Load metadata
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Create model architecture
                model_config = metadata['model_config']
                model = SimpleTCN(
                    input_size=model_config['input_size'],
                    hidden_size=model_config['hidden_size'],
                    output_size=model_config['output_size']
                )

                # Load weights
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(state_dict)
                model.eval()

                self.models[metric_type] = model

                # Load scaler if exists
                if scaler_path.exists():
                    self.scalers[metric_type] = joblib.load(scaler_path)

                    # 학습 데이터의 std를 residual 정규화용으로 저장
                    # (원본 스케일에서의 std)
                    train_std = np.sqrt(self.scalers[metric_type].var_[0])
                    if not hasattr(self, 'residual_normalizers'):
                        self.residual_normalizers = {}
                    self.residual_normalizers[metric_type] = train_std

                    self.logger.info(
                        f"Loaded scaler for {metric_type}",
                        mean=float(self.scalers[metric_type].mean_[0]),
                        std=float(train_std),
                        residual_normalizer=float(train_std)
                    )
                else:
                    self.logger.warning(
                        f"Scaler not found for {metric_type}, using unfitted scaler",
                        scaler_path=str(scaler_path)
                    )

                self.logger.info(
                    f"Loaded pre-trained TCN model for {metric_type}",
                    version=metadata['metadata']['version'],
                    epochs=metadata['performance']['total_epochs']
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to load pre-trained model for {metric_type}",
                    error=str(e)
                )

    def get_evidence(self, features: FeatureVector) -> Dict[str, float]:
        """Get evidence details for residual detection.

        Args:
            features: Feature vector

        Returns:
            Dictionary of evidence details
        """
        evidence = {}

        # Add predicted vs actual values if available
        if features.udp_echo_residual is not None:
            evidence["udp_echo_residual"] = features.udp_echo_residual

        if features.ecpri_residual is not None:
            evidence["ecpri_residual"] = features.ecpri_residual

        if features.lbm_residual is not None:
            evidence["lbm_residual"] = features.lbm_residual

        return evidence
