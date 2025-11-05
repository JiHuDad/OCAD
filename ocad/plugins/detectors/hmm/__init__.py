"""HMM (Hidden Markov Model) detector plugin for state-based anomaly detection.

This plugin uses Hidden Markov Models to detect anomalies in state-based
protocols, particularly effective for:
- BFD state transitions (ADMIN_DOWN, DOWN, INIT, UP)
- BGP FSM state changes
- Protocol state machines with observable states
- Abnormal state transition patterns

HMM models learn normal state transition probabilities and emission
distributions, flagging sequences with low likelihood as anomalies.
"""

from typing import Any, Dict, List, Optional, Tuple
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

# Try to import hmmlearn (optional dependency)
try:
    from hmmlearn import hmm as hmmlearn_hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    logger.warning("hmmlearn not available. Using simple HMM implementation.")

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


class SimpleGaussianHMM:
    """Simple Gaussian HMM implementation (fallback when hmmlearn unavailable).

    This is a simplified HMM with:
    - Gaussian emission distributions
    - Discrete hidden states
    - Baum-Welch training (EM algorithm)
    - Forward algorithm for likelihood calculation
    """

    def __init__(self, n_components: int = 4):
        """Initialize HMM.

        Args:
            n_components: Number of hidden states
        """
        self.n_components = n_components

        # Model parameters (initialized randomly, refined during training)
        self.startprob_ = np.ones(n_components) / n_components
        self.transmat_ = np.ones((n_components, n_components)) / n_components
        self.means_ = np.zeros((n_components, 1))
        self.covars_ = np.ones((n_components, 1))

        self.is_trained = False

    def fit(self, X: np.ndarray, n_iter: int = 10) -> "SimpleGaussianHMM":
        """Train HMM using Baum-Welch algorithm (simplified).

        Args:
            X: Observations of shape (n_samples, n_features)
            n_iter: Number of EM iterations

        Returns:
            Self
        """
        n_samples, n_features = X.shape

        # Initialize parameters from data
        # Use k-means-like initialization
        indices = np.linspace(0, n_samples - 1, self.n_components, dtype=int)
        self.means_ = X[indices]
        self.covars_ = np.var(X, axis=0, keepdims=True).repeat(self.n_components, axis=0)

        # Simple EM iterations (simplified - not full Baum-Welch)
        for iteration in range(n_iter):
            # E-step: Calculate responsibilities (which state each observation belongs to)
            responsibilities = self._calculate_responsibilities(X)

            # M-step: Update parameters
            state_counts = responsibilities.sum(axis=0) + 1e-10

            # Update means
            self.means_ = (responsibilities.T @ X) / state_counts.reshape(-1, 1)

            # Update covariances
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covars_[k] = (responsibilities[:, k] @ (diff ** 2)) / state_counts[k]

            # Update transition matrix (simplified)
            for i in range(n_samples - 1):
                curr_state = np.argmax(responsibilities[i])
                next_state = np.argmax(responsibilities[i + 1])
                self.transmat_[curr_state, next_state] += 1

            # Normalize transition matrix
            row_sums = self.transmat_.sum(axis=1, keepdims=True) + 1e-10
            self.transmat_ = self.transmat_ / row_sums

        self.is_trained = True
        return self

    def _calculate_responsibilities(self, X: np.ndarray) -> np.ndarray:
        """Calculate soft assignments of observations to states.

        Args:
            X: Observations of shape (n_samples, n_features)

        Returns:
            Responsibilities of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            # Gaussian likelihood
            diff = X - self.means_[k]
            var = self.covars_[k] + 1e-10
            log_prob = -0.5 * np.sum((diff ** 2) / var, axis=1)
            responsibilities[:, k] = np.exp(log_prob)

        # Normalize
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) + 1e-10
        return responsibilities

    def score(self, X: np.ndarray) -> float:
        """Calculate log-likelihood of observations.

        Args:
            X: Observations of shape (n_samples, n_features)

        Returns:
            Log-likelihood (higher = more normal)
        """
        if not self.is_trained:
            return 0.0

        # Forward algorithm (simplified)
        n_samples = X.shape[0]

        # Initial state probabilities
        alpha = self.startprob_.copy()

        log_likelihood = 0.0

        for t in range(n_samples):
            # Emission probabilities
            emission_probs = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff = X[t] - self.means_[k]
                var = self.covars_[k] + 1e-10
                log_prob = -0.5 * np.sum((diff ** 2) / var)
                emission_probs[k] = np.exp(log_prob)

            # Update alpha
            alpha = alpha * emission_probs
            alpha_sum = alpha.sum()

            if alpha_sum > 0:
                log_likelihood += np.log(alpha_sum)
                alpha /= alpha_sum
            else:
                log_likelihood += -1e10

            # Transition
            if t < n_samples - 1:
                alpha = self.transmat_.T @ alpha

        return log_likelihood / n_samples


class HMMDetector(DetectorPlugin):
    """HMM-based anomaly detector for state-based protocols.

    This detector uses Hidden Markov Models to learn normal state
    transition patterns and detect anomalies based on sequence
    likelihood. It's particularly effective for:

    - BFD state transitions (flapping detection)
    - BGP FSM state changes
    - Any protocol with state machines
    - Abnormal state transition sequences

    The detector works by:
    1. Training: Learn state transition probabilities and emission distributions
    2. Detection: Calculate likelihood of observed sequence
    3. Scoring: Low likelihood indicates anomaly

    Hyperparameters:
    - n_components: Number of hidden states (default: 4)
    - sequence_length: Sequence length for likelihood calculation (default: 10)
    - threshold_percentile: Percentile for anomaly threshold (default: 5)
    """

    def __init__(
        self,
        n_components: int = 4,
        sequence_length: int = 10,
        threshold_percentile: float = 5.0,
    ):
        """Initialize HMM detector.

        Args:
            n_components: Number of hidden states
            sequence_length: Sequence length for detection
            threshold_percentile: Percentile threshold for anomalies
        """
        self.n_components = n_components
        self.sequence_length = sequence_length
        self.threshold_percentile = threshold_percentile

        # Model components
        if HAS_HMMLEARN:
            self.model = hmmlearn_hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=100,
            )
        else:
            self.model = SimpleGaussianHMM(n_components=n_components)

        self.anomaly_threshold: float = 0.0
        self.is_trained: bool = False

        # Use bind() if available (structlog), otherwise use base logger
        self.logger = logger.bind(component="hmm_detector") if hasattr(logger, "bind") else logger

    @property
    def name(self) -> str:
        return "hmm"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        return ["bfd", "bgp"]

    def train(self, data: Any, **kwargs) -> None:
        """Train HMM on normal behavior sequences.

        Args:
            data: Training data, either:
                - numpy array of shape (n_samples,) - single metric sequence
                - numpy array of shape (n_samples, n_features) - multi-metric
                - list of float values
            **kwargs: Training parameters (reserved for future use)

        Raises:
            ValueError: If data format is invalid
        """
        # Convert data to numpy array
        if isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Handle 1D data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape

        if n_samples < self.sequence_length:
            raise ValueError(
                f"Insufficient data: need at least {self.sequence_length} samples, "
                f"got {n_samples}"
            )

        _log(
            self.logger,
            "info",
            "Starting HMM training",
            n_samples=n_samples,
            n_features=n_features,
            n_components=self.n_components,
        )

        # Train HMM
        if HAS_HMMLEARN:
            self.model.fit(data)
        else:
            self.model.fit(data, n_iter=20)

        # Calculate anomaly threshold from training data
        # Use sliding windows to get likelihood distribution
        likelihoods = []

        for i in range(n_samples - self.sequence_length + 1):
            sequence = data[i:i + self.sequence_length]
            likelihood = self.model.score(sequence)
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.anomaly_threshold = float(np.percentile(likelihoods, self.threshold_percentile))

        self.is_trained = True

        _log(
            self.logger,
            "info",
            "HMM training completed",
            anomaly_threshold=self.anomaly_threshold,
            mean_likelihood=float(np.mean(likelihoods)),
        )

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in state sequence.

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
                - detector_name (str): "hmm"
        """
        if not self.is_trained:
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

        if len(history) < self.sequence_length - 1:
            # Not enough history yet
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "evidence": {
                    "error": f"Insufficient history: need {self.sequence_length - 1}, got {len(history)}"
                },
                "detector_name": self.name,
            }

        # Build sequence: last (sequence_length - 1) history + current
        sequence_list = history[-(self.sequence_length - 1):] + [current_value]
        sequence = np.array(sequence_list).reshape(-1, 1)

        # Calculate likelihood
        likelihood = self.model.score(sequence)

        # Calculate anomaly score
        # Lower likelihood = higher anomaly score
        if likelihood < self.anomaly_threshold:
            # Map to 0-1 range
            # At threshold: score = 0.5
            # Far below threshold: score → 1.0
            score = 0.5 + 0.5 * min(1.0, (self.anomaly_threshold - likelihood) / abs(self.anomaly_threshold + 1e-10))
        else:
            # Normal: score → 0.0
            score = 0.0

        is_anomaly = likelihood < self.anomaly_threshold

        return {
            "score": score,
            "is_anomaly": is_anomaly,
            "confidence": min(0.9, score) if is_anomaly else 0.5,
            "evidence": {
                "log_likelihood": float(likelihood),
                "anomaly_threshold": float(self.anomaly_threshold),
                "sequence_length": self.sequence_length,
                "n_components": self.n_components,
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
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        save_dict = {
            "model": self.model,
            "n_components": self.n_components,
            "sequence_length": self.sequence_length,
            "threshold_percentile": self.threshold_percentile,
            "anomaly_threshold": self.anomaly_threshold,
            "version": self.version,
            "use_hmmlearn": HAS_HMMLEARN,
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

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

        with open(load_path, "rb") as f:
            save_dict = pickle.load(f)

        # Restore parameters
        self.model = save_dict["model"]
        self.n_components = save_dict["n_components"]
        self.sequence_length = save_dict["sequence_length"]
        self.threshold_percentile = save_dict["threshold_percentile"]
        self.anomaly_threshold = save_dict["anomaly_threshold"]
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
        HMM detector instance
    """
    return HMMDetector()
