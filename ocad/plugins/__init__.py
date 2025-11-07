"""Plugin system for OCAD.

This package provides a plugin-based architecture for extending OCAD
with new protocols and detection models.

Main components:
- ProtocolAdapter: Interface for protocol-specific data collection
- DetectorPlugin: Interface for AI model plugins
- PluginRegistry: Central registry for managing plugins
"""

from .base import ProtocolAdapter, DetectorPlugin
from .registry import PluginRegistry, registry

__all__ = [
    "ProtocolAdapter",
    "DetectorPlugin",
    "PluginRegistry",
    "registry",
]
