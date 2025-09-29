"""Unit tests for capability detector."""

import pytest
from unittest.mock import Mock, patch

from ocad.capability.detector import CapabilityDetector, CapabilityRegistry
from ocad.core.config import NetconfConfig
from ocad.core.models import Capabilities, Endpoint, EndpointRole


@pytest.fixture
def netconf_config():
    """Create test NETCONF configuration."""
    return NetconfConfig(
        timeout=10,
        port=830,
        username="test",
        password="test",
        hostkey_verify=False,
    )


@pytest.fixture
def capability_detector(netconf_config):
    """Create capability detector instance."""
    return CapabilityDetector(netconf_config)


@pytest.fixture
def test_endpoint():
    """Create test endpoint."""
    return Endpoint(
        id="test-endpoint",
        host="192.168.1.100",
        port=830,
        role=EndpointRole.O_RU,
    )


class TestCapabilityDetector:
    """Test capability detector functionality."""
    
    def test_parse_capabilities_oran_modules(self, capability_detector):
        """Test parsing of ORAN-specific modules."""
        server_caps = {
            "urn:o-ran:fm:1.0",
            "urn:o-ran:supervision:1.0",
            "urn:o-ran:delay-management:1.0",
        }
        
        capabilities = capability_detector._parse_capabilities(server_caps, EndpointRole.O_RU)
        
        assert capabilities.ccm_min is True
        assert capabilities.udp_echo is True
        assert capabilities.ecpri_delay is True
    
    def test_parse_capabilities_cfm_patterns(self, capability_detector):
        """Test parsing of CFM patterns."""
        server_caps = {
            "urn:ieee:yang:ieee802-dot1ag-cfm",
            "urn:example:cfm-loopback:1.0",
        }
        
        capabilities = capability_detector._parse_capabilities(server_caps, EndpointRole.O_DU)
        
        assert capabilities.lbm is True
    
    def test_parse_capabilities_role_defaults(self, capability_detector):
        """Test role-specific default capabilities."""
        server_caps = set()  # No specific capabilities
        
        # O-RU should get UDP echo as fallback
        o_ru_caps = capability_detector._parse_capabilities(server_caps, EndpointRole.O_RU)
        assert o_ru_caps.udp_echo is True
        
        # O-DU should get eCPRI delay
        o_du_caps = capability_detector._parse_capabilities(server_caps, EndpointRole.O_DU)
        assert o_du_caps.ecpri_delay is True
    
    def test_get_fallback_capabilities(self, capability_detector):
        """Test fallback capabilities for different roles."""
        o_ru_fallback = capability_detector._get_fallback_capabilities(EndpointRole.O_RU)
        assert o_ru_fallback.udp_echo is True
        
        o_du_fallback = capability_detector._get_fallback_capabilities(EndpointRole.O_DU)
        assert o_du_fallback.udp_echo is True
        assert o_du_fallback.ecpri_delay is True
        
        transport_fallback = capability_detector._get_fallback_capabilities(EndpointRole.TRANSPORT)
        assert transport_fallback.lldp is True


class TestCapabilityRegistry:
    """Test capability registry functionality."""
    
    def test_register_endpoint(self):
        """Test endpoint registration."""
        registry = CapabilityRegistry()
        endpoint = Endpoint(
            id="test-1",
            host="192.168.1.1",
            role=EndpointRole.O_RU,
        )
        capabilities = Capabilities(udp_echo=True, lbm=True)
        
        registry.register_endpoint(endpoint, capabilities)
        
        assert registry.get_endpoint("test-1") == endpoint
        assert registry.get_capabilities("test-1") == capabilities
    
    def test_list_endpoints_by_capability(self):
        """Test listing endpoints by capability."""
        registry = CapabilityRegistry()
        
        # Register endpoints with different capabilities
        endpoints = [
            ("ep1", Capabilities(udp_echo=True)),
            ("ep2", Capabilities(lbm=True)),
            ("ep3", Capabilities(udp_echo=True, lbm=True)),
        ]
        
        for ep_id, caps in endpoints:
            endpoint = Endpoint(id=ep_id, host=f"host-{ep_id}", role=EndpointRole.O_RU)
            registry.register_endpoint(endpoint, caps)
        
        # Test filtering
        udp_endpoints = registry.list_endpoints_by_capability("udp_echo")
        assert set(udp_endpoints) == {"ep1", "ep3"}
        
        lbm_endpoints = registry.list_endpoints_by_capability("lbm")
        assert set(lbm_endpoints) == {"ep2", "ep3"}
    
    def test_capability_coverage(self):
        """Test capability coverage calculation."""
        registry = CapabilityRegistry()
        
        # Register endpoints
        endpoints = [
            ("ep1", Capabilities(udp_echo=True)),  # Has capabilities
            ("ep2", Capabilities()),  # No capabilities
            ("ep3", Capabilities(lbm=True)),  # Has capabilities
        ]
        
        for ep_id, caps in endpoints:
            endpoint = Endpoint(id=ep_id, host=f"host-{ep_id}", role=EndpointRole.O_RU)
            registry.register_endpoint(endpoint, caps)
        
        coverage = registry.get_capability_coverage()
        assert coverage == 200.0 / 3.0  # 2 out of 3 endpoints have capabilities
