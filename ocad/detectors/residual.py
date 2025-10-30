"""Residual-based anomaly detector using predictive models."""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..core.models import Capabilities, FeatureVector
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
                    
                    # Normalize residual by recent standard deviation
                    recent_values = self.history[metric_type][-20:]
                    if len(recent_values) > 3:
                        std_dev = np.std(recent_values)
                        if std_dev > 0:
                            normalized_residual = residual / std_dev
                            return normalized_residual
                    
                    return residual
            
        except Exception as e:
            self.logger.debug(
                "Residual calculation failed",
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
        model_files = {
            "udp_echo": "udp_echo_vv2.0.0",
            "ecpri": "ecpri_vv2.0.0",
            "lbm": "lbm_vv2.0.0",
        }

        for metric_type, model_name in model_files.items():
            try:
                model_path = self.model_dir / f"{model_name}.pth"
                metadata_path = self.model_dir / f"{model_name}.json"

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
