"""LSTM detector plugin for sequence-based anomaly detection.

This plugin uses Long Short-Term Memory (LSTM) neural networks to detect
anomalies in time-series sequences, particularly effective for:
- BFD state transitions (flapping detection)
- Sequential patterns in network protocols
- Temporal dependencies in metric sequences

LSTM models learn normal sequence patterns and flag deviations from
expected behavior as anomalies.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import pickle
import logging
import numpy as np

# Try to use structlog if available, otherwise fall back to standard logging
try:
    from ....core.logging import get_logger
    logger = get_logger(__name__)
    HAS_STRUCTLOG = True
except ImportError:
    logger = logging.getLogger(__name__)
    HAS_STRUCTLOG = False

# Try to import PyTorch (optional dependency)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available. LSTM detector will not function properly.")

from ...base import DetectorPlugin


def _log(logger_instance, level: str, msg: str, **kwargs):
    """Helper to log with or without structlog."""
    if HAS_STRUCTLOG or hasattr(logger_instance, "bind"):
        getattr(logger_instance, level)(msg, **kwargs)
    else:
        if kwargs:
            formatted_msg = f"{msg} ({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        else:
            formatted_msg = msg
        getattr(logger_instance, level)(formatted_msg)


class LSTMModel(nn.Module):
    """LSTM neural network for sequence prediction.

    Architecture:
    - Input layer: sequence of metric values
    - LSTM layer(s): captures temporal dependencies
    - Fully connected layer: produces predictions
    - Output: predicted next value(s)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        """Initialize LSTM model.

        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of stacked LSTM layers
            output_size: Number of output predictions
            dropout: Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM forward pass
        # out: (batch, sequence_length, hidden_size)
        # hidden: (num_layers, batch, hidden_size)
        out, (hidden, cell) = self.lstm(x)

        # Use the last output timestep
        # out[:, -1, :] has shape (batch, hidden_size)
        out = self.fc(out[:, -1, :])

        return out


class LSTMDetector(DetectorPlugin):
    """LSTM-based anomaly detector for sequential patterns.

    This detector uses LSTM neural networks to learn normal sequence
    patterns and detect anomalies based on prediction error. It's
    particularly effective for:

    - BFD state transitions (flapping detection)
    - BGP route update sequences
    - Any protocol with temporal dependencies

    The detector works by:
    1. Training: Learn to predict next value(s) from historical sequence
    2. Detection: Compare actual vs predicted values
    3. Scoring: High prediction error indicates anomaly

    Hyperparameters:
    - sequence_length: How many historical points to consider (default: 10)
    - threshold_multiplier: Multiplier for anomaly threshold (default: 3.0)
    - hidden_size: LSTM hidden units (default: 64)
    - num_layers: Number of LSTM layers (default: 2)
    """

    def __init__(
        self,
        sequence_length: int = 10,
        threshold_multiplier: float = 3.0,
        hidden_size: int = 64,
        num_layers: int = 2,
    ):
        """Initialize LSTM detector.

        Args:
            sequence_length: Number of historical points for prediction
            threshold_multiplier: Multiplier for anomaly threshold
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
        """
        if not HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for LSTM detector. "
                "Install with: pip install torch"
            )

        self.sequence_length = sequence_length
        self.threshold_multiplier = threshold_multiplier
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Model components
        self.model: Optional[LSTMModel] = None
        self.scaler_mean: float = 0.0
        self.scaler_std: float = 1.0
        self.anomaly_threshold: float = 0.0

        # Training history
        self.training_losses: List[float] = []
        self.is_trained: bool = False

        # Use bind() if available (structlog), otherwise use base logger
        self.logger = logger.bind(component="lstm_detector") if hasattr(logger, "bind") else logger

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        return ["bfd", "bgp", "cfm", "ptp"]

    def train(self, data: Any, **kwargs) -> None:
        """Train LSTM model on normal behavior sequences.

        Args:
            data: Training data, either:
                - numpy array of shape (n_samples,) - single metric sequence
                - numpy array of shape (n_samples, n_features) - multi-metric
                - list of float values
            **kwargs: Training parameters:
                - epochs (int): Number of training epochs (default: 50)
                - learning_rate (float): Learning rate (default: 0.001)
                - batch_size (int): Batch size (default: 32)

        Raises:
            ValueError: If data format is invalid
        """
        # Parse training parameters
        epochs = kwargs.get("epochs", 50)
        learning_rate = kwargs.get("learning_rate", 0.001)
        batch_size = kwargs.get("batch_size", 32)

        # Convert data to numpy array
        if isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Handle 1D data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape

        if n_samples < self.sequence_length + 1:
            raise ValueError(
                f"Insufficient data: need at least {self.sequence_length + 1} samples, "
                f"got {n_samples}"
            )

        _log(
            self.logger,
            "info",
            "Starting LSTM training",
            n_samples=n_samples,
            n_features=n_features,
            sequence_length=self.sequence_length,
            epochs=epochs,
        )

        # Normalize data
        self.scaler_mean = float(np.mean(data))
        self.scaler_std = float(np.std(data))
        if self.scaler_std < 1e-8:
            self.scaler_std = 1.0

        normalized_data = (data - self.scaler_mean) / self.scaler_std

        # Create sequences for training
        X, y = self._create_sequences(normalized_data)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Initialize model
        self.model = LSTMModel(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=n_features,
        )

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.training_losses = []
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                _log(
                    self.logger,
                    "info",
                    "Training progress",
                    epoch=epoch + 1,
                    loss=avg_loss,
                )

        # Calculate anomaly threshold from training errors
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            errors = torch.abs(predictions - y_tensor)
            mean_error = float(torch.mean(errors))
            std_error = float(torch.std(errors))

        self.anomaly_threshold = mean_error + self.threshold_multiplier * std_error
        self.is_trained = True

        _log(
            self.logger,
            "info",
            "Training completed",
            final_loss=self.training_losses[-1],
            anomaly_threshold=self.anomaly_threshold,
        )

    def _create_sequences(self, data: np.ndarray) -> tuple:
        """Create input-output sequence pairs for training.

        Args:
            data: Normalized data array of shape (n_samples, n_features)

        Returns:
            Tuple of (X, y) where:
                X: Input sequences of shape (n_sequences, sequence_length, n_features)
                y: Target values of shape (n_sequences, n_features)
        """
        n_samples, n_features = data.shape
        n_sequences = n_samples - self.sequence_length

        X = np.zeros((n_sequences, self.sequence_length, n_features))
        y = np.zeros((n_sequences, n_features))

        for i in range(n_sequences):
            X[i] = data[i:i + self.sequence_length]
            y[i] = data[i + self.sequence_length]

        return X, y

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in feature sequence.

        Args:
            features: Feature dictionary containing:
                - timestamp (datetime): Current timestamp
                - source_id (str): Endpoint identifier
                - metric_name (str): Metric name
                - value (float): Current value
                - history (list, optional): Historical values

        Returns:
            Detection result dictionary:
                - score (float): Anomaly score (0.0-1.0)
                - is_anomaly (bool): Binary classification
                - confidence (float): Detection confidence
                - evidence (dict): Evidence details
                - detector_name (str): "lstm"
        """
        if not self.is_trained or self.model is None:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "evidence": {"error": "Model not trained"},
                "detector_name": self.name,
            }

        # Get historical sequence
        history = features.get("history", [])
        current_value = features["value"]

        if len(history) < self.sequence_length:
            # Not enough history yet
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "evidence": {
                    "error": f"Insufficient history: need {self.sequence_length}, got {len(history)}"
                },
                "detector_name": self.name,
            }

        # Use last sequence_length values
        sequence = history[-self.sequence_length:]
        sequence_array = np.array(sequence).reshape(-1, 1)

        # Normalize
        normalized_sequence = (sequence_array - self.scaler_mean) / self.scaler_std

        # Convert to tensor
        X = torch.FloatTensor(normalized_sequence).unsqueeze(0)  # (1, seq_len, 1)

        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
            prediction_value = float(prediction[0, 0])

        # Calculate prediction error
        normalized_actual = (current_value - self.scaler_mean) / self.scaler_std
        prediction_error = abs(normalized_actual - prediction_value)

        # Calculate anomaly score
        if self.anomaly_threshold > 0:
            score = min(1.0, prediction_error / self.anomaly_threshold)
        else:
            score = 0.0

        is_anomaly = prediction_error > self.anomaly_threshold

        # Denormalize for evidence
        predicted_value = prediction_value * self.scaler_std + self.scaler_mean

        return {
            "score": score,
            "is_anomaly": is_anomaly,
            "confidence": min(0.95, score) if is_anomaly else 0.5,
            "evidence": {
                "prediction_error": float(prediction_error),
                "anomaly_threshold": float(self.anomaly_threshold),
                "predicted_value": float(predicted_value),
                "actual_value": float(current_value),
                "sequence_length": self.sequence_length,
            },
            "detector_name": self.name,
        }

    def save_model(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: File path to save model (will create parent directories)

        Raises:
            IOError: If save fails
            ValueError: If model not trained
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained model")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and metadata
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "sequence_length": self.sequence_length,
            "threshold_multiplier": self.threshold_multiplier,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "anomaly_threshold": self.anomaly_threshold,
            "training_losses": self.training_losses,
            "version": self.version,
        }

        torch.save(save_dict, save_path)

        _log(
            self.logger,
            "info",
            "Model saved",
            path=str(save_path),
        )

    def load_model(self, path: str) -> None:
        """Load trained model from disk.

        Args:
            path: File path to load model from

        Raises:
            IOError: If load fails
            ValueError: If model format is invalid
        """
        load_path = Path(path)
        if not load_path.exists():
            raise IOError(f"Model file not found: {path}")

        # Load saved dictionary
        save_dict = torch.load(load_path)

        # Restore hyperparameters
        self.sequence_length = save_dict["sequence_length"]
        self.threshold_multiplier = save_dict["threshold_multiplier"]
        self.hidden_size = save_dict["hidden_size"]
        self.num_layers = save_dict["num_layers"]
        self.scaler_mean = save_dict["scaler_mean"]
        self.scaler_std = save_dict["scaler_std"]
        self.anomaly_threshold = save_dict["anomaly_threshold"]
        self.training_losses = save_dict["training_losses"]

        # Reconstruct model
        # Need to infer input_size from state_dict
        lstm_weight_ih = save_dict["model_state_dict"]["lstm.weight_ih_l0"]
        input_size = lstm_weight_ih.shape[1]

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=input_size,  # Same as input for autoregressive
        )

        self.model.load_state_dict(save_dict["model_state_dict"])
        self.model.eval()
        self.is_trained = True

        _log(
            self.logger,
            "info",
            "Model loaded",
            path=str(load_path),
            version=save_dict.get("version", "unknown"),
        )


def create_detector() -> DetectorPlugin:
    """Plugin entry point.

    Returns:
        LSTM detector instance
    """
    return LSTMDetector()
