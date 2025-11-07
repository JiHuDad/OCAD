"""Unit tests for plugin system."""

import pytest
from pathlib import Path
from typing import AsyncIterator, Dict, List, Any

from ocad.plugins.base import ProtocolAdapter, DetectorPlugin
from ocad.plugins.registry import PluginRegistry


class MockProtocolAdapter(ProtocolAdapter):
    """Mock protocol adapter for testing."""

    @property
    def name(self) -> str:
        return "mock-protocol"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return ["metric1", "metric2"]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "required_field" in config

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        yield {
            "timestamp": "2025-11-05T00:00:00",
            "source_id": "test",
            "metric_name": "metric1",
            "value": 1.0,
        }

    def get_recommended_models(self) -> List[str]:
        return ["model1", "model2"]


class MockDetectorPlugin(DetectorPlugin):
    """Mock detector plugin for testing."""

    @property
    def name(self) -> str:
        return "mock-detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        return ["mock-protocol", "another-protocol"]

    def train(self, data: Any, **kwargs) -> None:
        pass

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "score": 0.5,
            "is_anomaly": False,
            "detector_name": "mock-detector",
        }

    def save_model(self, path: str) -> None:
        pass

    def load_model(self, path: str) -> None:
        pass


class TestProtocolAdapter:
    """Test ProtocolAdapter interface."""

    def test_adapter_properties(self):
        """Test adapter properties."""
        adapter = MockProtocolAdapter()
        assert adapter.name == "mock-protocol"
        assert adapter.version == "1.0.0"
        assert len(adapter.supported_metrics) == 2
        assert "metric1" in adapter.supported_metrics

    def test_validate_config(self):
        """Test config validation."""
        adapter = MockProtocolAdapter()
        assert adapter.validate_config({"required_field": "value"}) is True
        assert adapter.validate_config({}) is False

    def test_get_recommended_models(self):
        """Test get recommended models."""
        adapter = MockProtocolAdapter()
        models = adapter.get_recommended_models()
        assert len(models) == 2
        assert "model1" in models

    def test_get_description(self):
        """Test get description."""
        adapter = MockProtocolAdapter()
        desc = adapter.get_description()
        assert "mock-protocol" in desc
        assert "1.0.0" in desc

    @pytest.mark.asyncio
    async def test_collect(self):
        """Test metric collection."""
        adapter = MockProtocolAdapter()
        metrics = []
        async for metric in adapter.collect({}):
            metrics.append(metric)
            break  # Only test first metric

        assert len(metrics) == 1
        assert metrics[0]["metric_name"] == "metric1"
        assert metrics[0]["value"] == 1.0


class TestDetectorPlugin:
    """Test DetectorPlugin interface."""

    def test_detector_properties(self):
        """Test detector properties."""
        detector = MockDetectorPlugin()
        assert detector.name == "mock-detector"
        assert detector.version == "1.0.0"
        assert len(detector.supported_protocols) == 2

    def test_can_detect_protocol(self):
        """Test protocol support check."""
        detector = MockDetectorPlugin()
        assert detector.can_detect_protocol("mock-protocol") is True
        assert detector.can_detect_protocol("another-protocol") is True
        assert detector.can_detect_protocol("unknown-protocol") is False

    def test_detect(self):
        """Test detection."""
        detector = MockDetectorPlugin()
        result = detector.detect({"value": 1.0})
        assert "score" in result
        assert "is_anomaly" in result
        assert result["detector_name"] == "mock-detector"

    def test_get_description(self):
        """Test get description."""
        detector = MockDetectorPlugin()
        desc = detector.get_description()
        assert "mock-detector" in desc
        assert "1.0.0" in desc


class TestPluginRegistry:
    """Test PluginRegistry."""

    def test_register_protocol_adapter(self):
        """Test protocol adapter registration."""
        registry = PluginRegistry()
        adapter = MockProtocolAdapter()

        registry.register_protocol_adapter(adapter)

        retrieved = registry.get_protocol_adapter("mock-protocol")
        assert retrieved is not None
        assert retrieved.name == "mock-protocol"

    def test_register_detector(self):
        """Test detector registration."""
        registry = PluginRegistry()
        detector = MockDetectorPlugin()

        registry.register_detector(detector)

        retrieved = registry.get_detector("mock-detector")
        assert retrieved is not None
        assert retrieved.name == "mock-detector"

    def test_list_protocol_adapters(self):
        """Test listing protocol adapters."""
        registry = PluginRegistry()
        adapter = MockProtocolAdapter()
        registry.register_protocol_adapter(adapter)

        adapters = registry.list_protocol_adapters()

        assert "mock-protocol" in adapters
        assert adapters["mock-protocol"]["version"] == "1.0.0"
        assert len(adapters["mock-protocol"]["supported_metrics"]) == 2

    def test_list_detectors(self):
        """Test listing detectors."""
        registry = PluginRegistry()
        detector = MockDetectorPlugin()
        registry.register_detector(detector)

        detectors = registry.list_detectors()

        assert "mock-detector" in detectors
        assert detectors["mock-detector"]["version"] == "1.0.0"
        assert len(detectors["mock-detector"]["supported_protocols"]) == 2

    def test_get_detectors_for_protocol(self):
        """Test getting detectors for a protocol."""
        registry = PluginRegistry()
        detector = MockDetectorPlugin()
        registry.register_detector(detector)

        detectors = registry.get_detectors_for_protocol("mock-protocol")

        assert len(detectors) == 1
        assert detectors[0].name == "mock-detector"

    def test_get_nonexistent_adapter(self):
        """Test getting non-existent adapter."""
        registry = PluginRegistry()
        adapter = registry.get_protocol_adapter("nonexistent")
        assert adapter is None

    def test_get_nonexistent_detector(self):
        """Test getting non-existent detector."""
        registry = PluginRegistry()
        detector = registry.get_detector("nonexistent")
        assert detector is None

    def test_discover_plugins_cfm(self):
        """Test discovering CFM plugin."""
        registry = PluginRegistry()
        plugin_dir = Path(__file__).parent.parent.parent / "ocad" / "plugins"

        registry.discover_plugins(plugin_dir)

        # Check if CFM adapter was discovered
        cfm_adapter = registry.get_protocol_adapter("cfm")
        assert cfm_adapter is not None
        assert cfm_adapter.name == "cfm"
        assert cfm_adapter.version == "1.0.0"
        assert "udp_echo_rtt_ms" in cfm_adapter.supported_metrics

    def test_discover_plugins_empty_dir(self):
        """Test discovering plugins from empty directory."""
        registry = PluginRegistry()
        plugin_dir = Path("/tmp/nonexistent_plugins")

        # Should not raise error
        registry.discover_plugins(plugin_dir)

        assert len(registry.list_protocol_adapters()) == 0
        assert len(registry.list_detectors()) == 0
