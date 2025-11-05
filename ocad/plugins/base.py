"""Base interfaces for plugin system."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
from datetime import datetime
from pathlib import Path

import structlog

from ..core.logging import get_logger


logger = get_logger(__name__)


class ProtocolAdapter(ABC):
    """Base interface for protocol adapters.

    Protocol adapters are responsible for collecting metrics from
    specific network protocols (CFM, BFD, BGP, PTP, etc.).

    Each adapter implements protocol-specific logic for:
    - Data collection via NETCONF, SNMP, or other methods
    - Normalization to a common metric format
    - Configuration validation

    Example:
        >>> class MyProtocolAdapter(ProtocolAdapter):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-protocol"
        ...
        ...     async def collect(self, config: Dict) -> AsyncIterator[Dict]:
        ...         # Collect metrics from protocol
        ...         yield {"timestamp": datetime.utcnow(), "value": 1.0}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol name (e.g., 'cfm', 'bfd', 'bgp', 'ptp').

        Returns:
            Unique protocol identifier
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version (semantic versioning recommended).

        Returns:
            Version string (e.g., '1.0.0')
        """
        pass

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """List of metric names this adapter can collect.

        Returns:
            List of metric names (e.g., ['rtt_ms', 'loss_rate'])
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate adapter configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Collect metrics from the protocol (async generator).

        This method should yield metric dictionaries containing:
        - timestamp (datetime): When the metric was collected
        - source_id (str): Endpoint/device identifier
        - metric_name (str): Name of the metric
        - value (float): Metric value
        - metadata (dict, optional): Additional context

        Args:
            config: Adapter-specific configuration

        Yields:
            Metric dictionaries

        Example:
            >>> async for metric in adapter.collect(config):
            ...     print(metric)
            {'timestamp': datetime(...), 'source_id': 'endpoint-1',
             'metric_name': 'rtt_ms', 'value': 5.2}
        """
        pass

    @abstractmethod
    def get_recommended_models(self) -> List[str]:
        """Get recommended AI models for this protocol.

        Returns:
            List of detector names (e.g., ['tcn', 'lstm', 'rule-based'])
        """
        pass

    def get_description(self) -> str:
        """Get human-readable description of this adapter.

        Returns:
            Description string
        """
        return f"{self.name} protocol adapter v{self.version}"


class DetectorPlugin(ABC):
    """Base interface for detector plugins.

    Detector plugins implement AI models for anomaly detection.
    Each plugin can support one or more protocols.

    Plugins are responsible for:
    - Model training (offline)
    - Model loading/saving
    - Inference (online)

    Example:
        >>> class MyDetector(DetectorPlugin):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-detector"
        ...
        ...     def detect(self, features: Dict) -> Dict:
        ...         # Perform detection
        ...         return {"score": 0.5, "is_anomaly": False}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name (e.g., 'gnn', 'hmm', 'tcn').

        Returns:
            Unique detector identifier
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version (semantic versioning recommended).

        Returns:
            Version string (e.g., '1.0.0')
        """
        pass

    @property
    @abstractmethod
    def supported_protocols(self) -> List[str]:
        """List of protocol names this detector supports.

        Returns:
            List of protocol names (e.g., ['bfd', 'bgp'])
        """
        pass

    @abstractmethod
    def train(self, data: Any, **kwargs) -> None:
        """Train the model (offline).

        Args:
            data: Training data (format depends on detector)
            **kwargs: Additional training parameters

        Raises:
            ValueError: If data format is invalid
        """
        pass

    @abstractmethod
    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies (online inference).

        Args:
            features: Feature dictionary containing:
                - timestamp (datetime): Current timestamp
                - source_id (str): Endpoint identifier
                - metric_name (str): Metric name
                - value (float): Current value
                - Additional protocol-specific features

        Returns:
            Detection result dictionary:
                - score (float): Anomaly score (0.0 = normal, 1.0 = anomalous)
                - is_anomaly (bool): Binary classification
                - confidence (float, optional): Confidence score
                - evidence (dict, optional): Evidence details
                - detector_name (str): Name of this detector

        Example:
            >>> result = detector.detect({'timestamp': datetime.utcnow(),
            ...                           'value': 10.5})
            >>> print(result)
            {'score': 0.8, 'is_anomaly': True, 'confidence': 0.9,
             'detector_name': 'gnn'}
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: File path to save model

        Raises:
            IOError: If save fails
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load trained model from disk.

        Args:
            path: File path to load model from

        Raises:
            IOError: If load fails
            ValueError: If model format is invalid
        """
        pass

    def get_description(self) -> str:
        """Get human-readable description of this detector.

        Returns:
            Description string
        """
        protocols_str = ", ".join(self.supported_protocols)
        return f"{self.name} detector v{self.version} (supports: {protocols_str})"

    def can_detect_protocol(self, protocol_name: str) -> bool:
        """Check if this detector supports a given protocol.

        Args:
            protocol_name: Protocol name to check

        Returns:
            True if protocol is supported
        """
        return protocol_name in self.supported_protocols
