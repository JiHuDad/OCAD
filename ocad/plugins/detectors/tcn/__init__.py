"""TCN (Temporal Convolutional Network) detector plugin.

This plugin uses Temporal Convolutional Networks to detect anomalies
in time-series sequences, particularly effective for:
- PTP offset/delay prediction and anomaly detection
- Long-range temporal dependencies (dilated convolutions)
- Efficient parallel training (unlike RNNs)
- CFM latency prediction

TCN models use dilated causal convolutions to capture temporal patterns
over long sequences while maintaining computational efficiency.
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
    from torch.nn.utils import weight_norm
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available. TCN detector will not function properly.")

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


# Only define PyTorch classes if PyTorch is available
if HAS_PYTORCH:
    class Chomp1d(nn.Module):
        """Chomp layer to ensure causal convolution (no future information leakage)."""

        def __init__(self, chomp_size):
            super(Chomp1d, self).__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            """Remove padding from the right side to maintain causality."""
            return x[:, :, :-self.chomp_size].contiguous()


    class TemporalBlock(nn.Module):
        """Temporal block: the basic building block of TCN.

        Components:
        - Dilated causal convolution
        - Weight normalization
        - ReLU activation
        - Dropout
        - Residual connection
        """

        def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.2
        ):
            """Initialize temporal block.

            Args:
                n_inputs: Number of input channels
                n_outputs: Number of output channels
                kernel_size: Convolution kernel size
                stride: Convolution stride
                dilation: Dilation factor
                padding: Padding size (causal)
                dropout: Dropout rate
            """
            super(TemporalBlock, self).__init__()

            # First convolutional layer with dilation
            self.conv1 = weight_norm(nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ))
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            # Second convolutional layer with same dilation
            self.conv2 = weight_norm(nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ))
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            # Sequential network
            self.net = nn.Sequential(
                self.conv1,
                self.chomp1,
                self.relu1,
                self.dropout1,
                self.conv2,
                self.chomp2,
                self.relu2,
                self.dropout2
            )

            # Residual connection (1x1 conv if dimensions don't match)
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()

            self.init_weights()

        def init_weights(self):
            """Initialize weights using normal distribution."""
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

        def forward(self, x):
            """Forward pass with residual connection.

            Args:
                x: Input tensor of shape (batch, channels, seq_len)

            Returns:
                Output tensor of shape (batch, channels, seq_len)
            """
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)


    class TCNModel(nn.Module):
        """Temporal Convolutional Network for sequence prediction.

        Architecture:
        - Multiple temporal blocks with increasing dilation rates
        - Each block: Conv1d (dilated) + ReLU + Dropout + Residual
        - Final linear layer for prediction
        - Receptive field: kernel_size * (2^num_layers - 1)

        Example: kernel_size=3, num_layers=4
        Dilation sequence: 1, 2, 4, 8
        Receptive field: 3 * (2^4 - 1) = 45 timesteps
        """

        def __init__(
            self,
            input_size: int = 1,
            num_channels: List[int] = None,
            kernel_size: int = 3,
            dropout: float = 0.2,
            output_size: int = 1,
        ):
            """Initialize TCN model.

            Args:
                input_size: Number of input features per timestep
                num_channels: List of channel sizes for each layer [25, 25, 25, 25]
                kernel_size: Convolution kernel size (default: 3)
                dropout: Dropout rate for regularization
                output_size: Number of output predictions
            """
            super(TCNModel, self).__init__()

            if num_channels is None:
                num_channels = [25, 25, 25, 25]  # 4 layers, 25 channels each

            layers = []
            num_levels = len(num_channels)

            for i in range(num_levels):
                dilation_size = 2 ** i  # Exponentially increasing dilation: 1, 2, 4, 8, ...
                in_channels = input_size if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]

                layers.append(TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                ))

            self.network = nn.Sequential(*layers)
            self.linear = nn.Linear(num_channels[-1], output_size)

            self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)

        def forward(self, x):
            """Forward pass through the network.

            Args:
                x: Input tensor of shape (batch, input_size, sequence_length)

            Returns:
                Output tensor of shape (batch, output_size)
            """
            # TCN expects (batch, channels, seq_len)
            y = self.network(x)

            # Use the last timestep output
            # y[:, :, -1] has shape (batch, num_channels[-1])
            y = self.linear(y[:, :, -1])

            return y


class TCNDetector(DetectorPlugin):
    """TCN-based anomaly detector for temporal patterns.

    This detector uses Temporal Convolutional Networks to learn normal
    sequence patterns and detect anomalies based on prediction error.
    It's particularly effective for:

    - PTP offset/delay prediction
    - CFM latency prediction
    - Long-range temporal dependencies
    - Efficient parallel processing

    The detector works by:
    1. Training: Learn to predict next value(s) from historical sequence
    2. Detection: Compare actual vs predicted values
    3. Scoring: High prediction error indicates anomaly

    Advantages over LSTM:
    - Parallelizable training (faster)
    - Longer receptive field (captures longer patterns)
    - More stable gradients (no vanishing gradient problem)
    - Better for regularly sampled time series

    Hyperparameters:
    - sequence_length: How many historical points to consider (default: 20)
    - threshold_multiplier: Multiplier for anomaly threshold (default: 3.0)
    - num_channels: Channel sizes for each layer (default: [25, 25, 25, 25])
    - kernel_size: Convolution kernel size (default: 3)
    """

    def __init__(
        self,
        sequence_length: int = 20,
        threshold_multiplier: float = 3.0,
        num_channels: List[int] = None,
        kernel_size: int = 3,
    ):
        """Initialize TCN detector.

        Args:
            sequence_length: Number of historical points for prediction
            threshold_multiplier: Multiplier for anomaly threshold
            num_channels: Channel sizes for each layer
            kernel_size: Convolution kernel size
        """
        if not HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for TCN detector. "
                "Install with: pip install torch"
            )

        self.sequence_length = sequence_length
        self.threshold_multiplier = threshold_multiplier
        self.num_channels = num_channels if num_channels is not None else [25, 25, 25, 25]
        self.kernel_size = kernel_size

        # Model components
        self.model: Optional[TCNModel] = None
        self.scaler_mean: float = 0.0
        self.scaler_std: float = 1.0
        self.anomaly_threshold: float = 0.0

        # Training history
        self.training_losses: List[float] = []
        self.is_trained: bool = False

        # Use bind() if available (structlog), otherwise use base logger
        self.logger = logger.bind(component="tcn_detector") if hasattr(logger, "bind") else logger

    @property
    def name(self) -> str:
        return "tcn"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        return ["ptp", "cfm", "bfd", "bgp"]

    def train(self, data: Any, **kwargs) -> None:
        """Train TCN model on normal behavior sequences.

        Args:
            data: Training data, either:
                - numpy array of shape (n_samples,) - single metric sequence
                - numpy array of shape (n_samples, n_features) - multi-metric
                - list of float values
            **kwargs: Training parameters:
                - epochs (int): Number of training epochs (default: 30)
                - learning_rate (float): Learning rate (default: 0.001)
                - batch_size (int): Batch size (default: 32)

        Raises:
            ValueError: If data format is invalid
        """
        # Parse training parameters
        epochs = kwargs.get("epochs", 30)
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
            "Starting TCN training",
            n_samples=n_samples,
            n_features=n_features,
            sequence_length=self.sequence_length,
            epochs=epochs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
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
        # TCN expects (batch, channels, seq_len)
        X_tensor = torch.FloatTensor(X).permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        y_tensor = torch.FloatTensor(y)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Initialize model
        self.model = TCNModel(
            input_size=n_features,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            output_size=n_features,
        )

        _log(
            self.logger,
            "info",
            "TCN model initialized",
            receptive_field=self.model.receptive_field,
            num_layers=len(self.num_channels),
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

            if (epoch + 1) % 5 == 0:
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
                - detector_name (str): "tcn"
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
        # TCN expects (batch, channels, seq_len)
        X = torch.FloatTensor(normalized_sequence).permute(1, 0).unsqueeze(0)  # (1, 1, seq_len)

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
                "receptive_field": self.model.receptive_field,
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
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "anomaly_threshold": self.anomaly_threshold,
            "training_losses": self.training_losses,
            "receptive_field": self.model.receptive_field,
            "version": self.version,
        }

        torch.save(save_dict, save_path)

        _log(
            self.logger,
            "info",
            "Model saved",
            path=str(save_path),
            receptive_field=self.model.receptive_field,
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
        self.num_channels = save_dict["num_channels"]
        self.kernel_size = save_dict["kernel_size"]
        self.scaler_mean = save_dict["scaler_mean"]
        self.scaler_std = save_dict["scaler_std"]
        self.anomaly_threshold = save_dict["anomaly_threshold"]
        self.training_losses = save_dict["training_losses"]

        # Reconstruct model
        # Need to infer input_size from state_dict
        first_conv_weight = save_dict["model_state_dict"]["network.0.conv1.weight_v"]
        input_size = first_conv_weight.shape[1]

        self.model = TCNModel(
            input_size=input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
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
            receptive_field=save_dict.get("receptive_field", "unknown"),
        )


def create_detector() -> DetectorPlugin:
    """Plugin entry point.

    Returns:
        TCN detector instance
    """
    return TCNDetector()
