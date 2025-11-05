"""Plugin registry for managing protocol adapters and detectors."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import importlib.util
import sys

import structlog

from ..core.logging import get_logger
from .base import ProtocolAdapter, DetectorPlugin


logger = get_logger(__name__)


class PluginRegistry:
    """Central registry for managing plugins.

    The PluginRegistry provides:
    - Registration of protocol adapters and detectors
    - Automatic plugin discovery from directories
    - Plugin lookup and listing
    - Plugin metadata management

    Example:
        >>> registry = PluginRegistry()
        >>> registry.discover_plugins(Path("ocad/plugins"))
        >>> adapter = registry.get_protocol_adapter("bfd")
        >>> print(adapter.name)
        'bfd'
    """

    def __init__(self):
        """Initialize plugin registry."""
        self.protocol_adapters: Dict[str, ProtocolAdapter] = {}
        self.detectors: Dict[str, DetectorPlugin] = {}
        self.logger = logger.bind(component="plugin_registry")

    def register_protocol_adapter(self, adapter: ProtocolAdapter) -> None:
        """Register a protocol adapter.

        Args:
            adapter: Protocol adapter instance

        Raises:
            ValueError: If adapter name is invalid
        """
        name = adapter.name
        if not name:
            raise ValueError("Adapter name cannot be empty")

        if name in self.protocol_adapters:
            self.logger.warning(
                "Protocol adapter already registered, overwriting",
                name=name,
                old_version=self.protocol_adapters[name].version,
                new_version=adapter.version,
            )

        self.protocol_adapters[name] = adapter
        self.logger.info(
            "Protocol adapter registered",
            name=name,
            version=adapter.version,
            supported_metrics=adapter.supported_metrics,
        )

    def register_detector(self, detector: DetectorPlugin) -> None:
        """Register a detector plugin.

        Args:
            detector: Detector plugin instance

        Raises:
            ValueError: If detector name is invalid
        """
        name = detector.name
        if not name:
            raise ValueError("Detector name cannot be empty")

        if name in self.detectors:
            self.logger.warning(
                "Detector already registered, overwriting",
                name=name,
                old_version=self.detectors[name].version,
                new_version=detector.version,
            )

        self.detectors[name] = detector
        self.logger.info(
            "Detector registered",
            name=name,
            version=detector.version,
            supported_protocols=detector.supported_protocols,
        )

    def get_protocol_adapter(self, name: str) -> Optional[ProtocolAdapter]:
        """Get protocol adapter by name.

        Args:
            name: Protocol adapter name

        Returns:
            Protocol adapter instance or None if not found
        """
        adapter = self.protocol_adapters.get(name)
        if adapter is None:
            self.logger.warning("Protocol adapter not found", name=name)
        return adapter

    def get_detector(self, name: str) -> Optional[DetectorPlugin]:
        """Get detector by name.

        Args:
            name: Detector name

        Returns:
            Detector plugin instance or None if not found
        """
        detector = self.detectors.get(name)
        if detector is None:
            self.logger.warning("Detector not found", name=name)
        return detector

    def list_protocol_adapters(self) -> Dict[str, Dict[str, Any]]:
        """List all registered protocol adapters.

        Returns:
            Dictionary mapping adapter names to metadata:
                - version (str): Plugin version
                - supported_metrics (list): Supported metrics
                - recommended_models (list): Recommended AI models
                - description (str): Human-readable description
        """
        return {
            name: {
                "version": adapter.version,
                "supported_metrics": adapter.supported_metrics,
                "recommended_models": adapter.get_recommended_models(),
                "description": adapter.get_description(),
            }
            for name, adapter in self.protocol_adapters.items()
        }

    def list_detectors(self) -> Dict[str, Dict[str, Any]]:
        """List all registered detectors.

        Returns:
            Dictionary mapping detector names to metadata:
                - version (str): Plugin version
                - supported_protocols (list): Supported protocols
                - description (str): Human-readable description
        """
        return {
            name: {
                "version": detector.version,
                "supported_protocols": detector.supported_protocols,
                "description": detector.get_description(),
            }
            for name, detector in self.detectors.items()
        }

    def get_detectors_for_protocol(self, protocol_name: str) -> List[DetectorPlugin]:
        """Get all detectors that support a given protocol.

        Args:
            protocol_name: Protocol name

        Returns:
            List of detector plugins
        """
        return [
            detector
            for detector in self.detectors.values()
            if detector.can_detect_protocol(protocol_name)
        ]

    def discover_plugins(self, plugin_dir: Path) -> None:
        """Automatically discover and load plugins from a directory.

        This method scans the plugin directory for:
        - Protocol adapters in `protocol_adapters/`
        - Detectors in `detectors/`

        Each plugin directory should contain an `__init__.py` with:
        - `create_adapter()` function for protocol adapters
        - `create_detector()` function for detectors

        Args:
            plugin_dir: Path to plugin directory

        Example directory structure:
            plugin_dir/
            ├── protocol_adapters/
            │   ├── bfd/
            │   │   └── __init__.py  # contains create_adapter()
            │   └── bgp/
            │       └── __init__.py
            └── detectors/
                ├── gnn/
                │   └── __init__.py  # contains create_detector()
                └── hmm/
                    └── __init__.py
        """
        if not plugin_dir.exists():
            self.logger.warning(
                "Plugin directory not found",
                path=str(plugin_dir),
            )
            return

        # Discover protocol adapters
        adapters_dir = plugin_dir / "protocol_adapters"
        if adapters_dir.exists():
            self._discover_protocol_adapters(adapters_dir)

        # Discover detectors
        detectors_dir = plugin_dir / "detectors"
        if detectors_dir.exists():
            self._discover_detectors(detectors_dir)

    def _discover_protocol_adapters(self, adapters_dir: Path) -> None:
        """Discover protocol adapters in a directory.

        Args:
            adapters_dir: Path to protocol adapters directory
        """
        for adapter_dir in adapters_dir.iterdir():
            if not adapter_dir.is_dir():
                continue

            if adapter_dir.name.startswith("_"):
                # Skip private directories
                continue

            init_file = adapter_dir / "__init__.py"
            if not init_file.exists():
                self.logger.warning(
                    "Protocol adapter missing __init__.py",
                    path=str(adapter_dir),
                )
                continue

            try:
                adapter = self._load_protocol_adapter(adapter_dir, init_file)
                if adapter:
                    self.register_protocol_adapter(adapter)
            except Exception as e:
                self.logger.error(
                    "Failed to load protocol adapter",
                    path=str(adapter_dir),
                    error=str(e),
                    exc_info=True,
                )

    def _discover_detectors(self, detectors_dir: Path) -> None:
        """Discover detectors in a directory.

        Args:
            detectors_dir: Path to detectors directory
        """
        for detector_dir in detectors_dir.iterdir():
            if not detector_dir.is_dir():
                continue

            if detector_dir.name.startswith("_"):
                # Skip private directories
                continue

            init_file = detector_dir / "__init__.py"
            if not init_file.exists():
                self.logger.warning(
                    "Detector missing __init__.py",
                    path=str(detector_dir),
                )
                continue

            try:
                detector = self._load_detector(detector_dir, init_file)
                if detector:
                    self.register_detector(detector)
            except Exception as e:
                self.logger.error(
                    "Failed to load detector",
                    path=str(detector_dir),
                    error=str(e),
                    exc_info=True,
                )

    def _load_protocol_adapter(
        self, plugin_dir: Path, init_file: Path
    ) -> Optional[ProtocolAdapter]:
        """Load a protocol adapter from a plugin directory.

        Args:
            plugin_dir: Plugin directory path
            init_file: __init__.py file path

        Returns:
            Protocol adapter instance or None
        """
        module_name = f"ocad.plugins.protocol_adapters.{plugin_dir.name}"

        # Dynamic import
        spec = importlib.util.spec_from_file_location(module_name, init_file)
        if spec is None or spec.loader is None:
            self.logger.error(
                "Failed to create module spec",
                path=str(init_file),
            )
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Look for create_adapter() function
        if not hasattr(module, "create_adapter"):
            self.logger.warning(
                "Protocol adapter missing create_adapter()",
                path=str(plugin_dir),
            )
            return None

        # Create adapter instance
        adapter = module.create_adapter()

        # Validate adapter
        if not isinstance(adapter, ProtocolAdapter):
            self.logger.error(
                "create_adapter() did not return ProtocolAdapter",
                path=str(plugin_dir),
                type=type(adapter).__name__,
            )
            return None

        return adapter

    def _load_detector(
        self, plugin_dir: Path, init_file: Path
    ) -> Optional[DetectorPlugin]:
        """Load a detector from a plugin directory.

        Args:
            plugin_dir: Plugin directory path
            init_file: __init__.py file path

        Returns:
            Detector plugin instance or None
        """
        module_name = f"ocad.plugins.detectors.{plugin_dir.name}"

        # Dynamic import
        spec = importlib.util.spec_from_file_location(module_name, init_file)
        if spec is None or spec.loader is None:
            self.logger.error(
                "Failed to create module spec",
                path=str(init_file),
            )
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Look for create_detector() function
        if not hasattr(module, "create_detector"):
            self.logger.warning(
                "Detector missing create_detector()",
                path=str(plugin_dir),
            )
            return None

        # Create detector instance
        detector = module.create_detector()

        # Validate detector
        if not isinstance(detector, DetectorPlugin):
            self.logger.error(
                "create_detector() did not return DetectorPlugin",
                path=str(plugin_dir),
                type=type(detector).__name__,
            )
            return None

        return detector


# Global plugin registry instance
registry = PluginRegistry()
