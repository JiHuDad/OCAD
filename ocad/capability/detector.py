"""NETCONF-based capability detection for ORAN endpoints."""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set

import structlog
from ncclient import manager
from ncclient.operations.errors import RPCError

from ..core.config import NetconfConfig
from ..core.logging import get_logger, log_function_call
from ..core.models import Capabilities, Endpoint, EndpointRole


logger = get_logger(__name__)


class CapabilityDetector:
    """Detects capabilities of ORAN endpoints via NETCONF."""
    
    def __init__(self, config: NetconfConfig):
        """Initialize capability detector.
        
        Args:
            config: NETCONF configuration
        """
        self.config = config
        self.logger = logger.bind(component="capability_detector")
    
    async def detect_capabilities(self, endpoint: Endpoint) -> Capabilities:
        """Detect capabilities of an endpoint.
        
        Args:
            endpoint: Endpoint to probe
            
        Returns:
            Detected capabilities
        """
        self.logger.info(
            "Detecting capabilities",
            **log_function_call("detect_capabilities", endpoint_id=endpoint.id, host=endpoint.host)
        )
        
        try:
            with manager.connect(
                host=endpoint.host,
                port=endpoint.port,
                username=self.config.username,
                password=self.config.password,
                timeout=self.config.timeout,
                hostkey_verify=self.config.hostkey_verify,
                look_for_keys=False,
                allow_agent=False,
            ) as mgr:
                # Get server capabilities
                server_caps = mgr.server_capabilities
                
                # Parse YANG modules and features
                capabilities = self._parse_capabilities(server_caps, endpoint.role)
                
                # Perform specific probes if needed
                capabilities = await self._probe_specific_features(mgr, capabilities)
                
                self.logger.info(
                    "Capabilities detected",
                    endpoint_id=endpoint.id,
                    capabilities=capabilities.dict(),
                )
                
                return capabilities
                
        except Exception as e:
            self.logger.error(
                "Failed to detect capabilities",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            # Return minimal capabilities for fallback
            return self._get_fallback_capabilities(endpoint.role)
    
    def _parse_capabilities(self, server_caps: Set[str], role: EndpointRole) -> Capabilities:
        """Parse server capabilities to determine supported features.
        
        Args:
            server_caps: Set of capability URIs from NETCONF hello
            role: Endpoint role
            
        Returns:
            Parsed capabilities
        """
        capabilities = Capabilities()
        
        # Check for ORAN-specific modules
        oran_modules = {
            "urn:o-ran:fm:1.0": ["ccm_min"],
            "urn:o-ran:supervision:1.0": ["udp_echo"],
            "urn:o-ran:delay-management:1.0": ["ecpri_delay"],
            "urn:o-ran:lbm:1.0": ["lbm"],
            "urn:ietf:yang:ietf-lldp:1.0": ["lldp"],
        }
        
        # IEEE 802.1AG CFM capabilities
        cfm_patterns = [
            r"ieee802-dot1ag-cfm",
            r"cfm.*loopback",
            r"cfm.*continuity",
            r"y\.1731",
        ]
        
        # Check YANG modules
        for cap in server_caps:
            # ORAN modules
            for module, features in oran_modules.items():
                if module in cap:
                    for feature in features:
                        setattr(capabilities, feature, True)
            
            # CFM patterns
            for pattern in cfm_patterns:
                if re.search(pattern, cap, re.IGNORECASE):
                    if "loopback" in cap.lower() or "lbm" in cap.lower():
                        capabilities.lbm = True
                    if "continuity" in cap.lower() or "ccm" in cap.lower():
                        capabilities.ccm_min = True
        
        # Role-specific defaults
        if role == EndpointRole.O_RU:
            # O-RU typically supports basic supervision
            if not any([capabilities.udp_echo, capabilities.ecpri_delay, capabilities.lbm]):
                capabilities.udp_echo = True  # Most basic fallback
        
        elif role == EndpointRole.O_DU:
            # O-DU often has more capabilities
            if not capabilities.ecpri_delay:
                capabilities.ecpri_delay = True
        
        return capabilities
    
    async def _probe_specific_features(self, mgr: manager.Manager, capabilities: Capabilities) -> Capabilities:
        """Probe specific features that require active testing.
        
        Args:
            mgr: NETCONF manager instance
            capabilities: Initial capabilities
            
        Returns:
            Updated capabilities
        """
        # Test UDP echo capability
        if capabilities.udp_echo:
            capabilities.udp_echo = await self._test_udp_echo(mgr)
        
        # Test LBM capability
        if capabilities.lbm:
            capabilities.lbm = await self._test_lbm(mgr)
        
        # Test eCPRI delay capability
        if capabilities.ecpri_delay:
            capabilities.ecpri_delay = await self._test_ecpri_delay(mgr)
        
        return capabilities
    
    async def _test_udp_echo(self, mgr: manager.Manager) -> bool:
        """Test UDP echo capability.
        
        Args:
            mgr: NETCONF manager instance
            
        Returns:
            True if UDP echo is supported
        """
        try:
            # Try to get supervision configuration
            filter_xml = """
            <filter>
                <supervision xmlns="urn:o-ran:supervision:1.0">
                    <cu-plane-monitoring/>
                </supervision>
            </filter>
            """
            
            result = mgr.get_config(source="running", filter=filter_xml)
            return True
            
        except RPCError:
            return False
        except Exception as e:
            self.logger.debug("UDP echo test failed", error=str(e))
            return False
    
    async def _test_lbm(self, mgr: manager.Manager) -> bool:
        """Test LBM capability.
        
        Args:
            mgr: NETCONF manager instance
            
        Returns:
            True if LBM is supported
        """
        try:
            # Try to get LBM configuration or state
            filter_xml = """
            <filter>
                <interfaces xmlns="urn:ietf:yang:ietf-interfaces">
                    <interface>
                        <name>*</name>
                        <dot1ag-cfm xmlns="urn:ieee:yang:ieee802-dot1ag-cfm"/>
                    </interface>
                </interfaces>
            </filter>
            """
            
            result = mgr.get(filter=filter_xml)
            return True
            
        except RPCError:
            return False
        except Exception as e:
            self.logger.debug("LBM test failed", error=str(e))
            return False
    
    async def _test_ecpri_delay(self, mgr: manager.Manager) -> bool:
        """Test eCPRI delay measurement capability.
        
        Args:
            mgr: NETCONF manager instance
            
        Returns:
            True if eCPRI delay is supported
        """
        try:
            # Try to get delay management configuration
            filter_xml = """
            <filter>
                <delay-management xmlns="urn:o-ran:delay-management:1.0">
                    <adaptive-delay-configuration/>
                </delay-management>
            </filter>
            """
            
            result = mgr.get_config(source="running", filter=filter_xml)
            return True
            
        except RPCError:
            return False
        except Exception as e:
            self.logger.debug("eCPRI delay test failed", error=str(e))
            return False
    
    def _get_fallback_capabilities(self, role: EndpointRole) -> Capabilities:
        """Get fallback capabilities when detection fails.
        
        Args:
            role: Endpoint role
            
        Returns:
            Minimal capabilities for the role
        """
        if role == EndpointRole.O_RU:
            return Capabilities(udp_echo=True)
        elif role == EndpointRole.O_DU:
            return Capabilities(udp_echo=True, ecpri_delay=True)
        else:
            return Capabilities(lldp=True)


class CapabilityRegistry:
    """Registry for managing endpoint capabilities."""
    
    def __init__(self):
        """Initialize capability registry."""
        self._capabilities: Dict[str, Capabilities] = {}
        self._endpoints: Dict[str, Endpoint] = {}
        self.logger = get_logger(__name__).bind(component="capability_registry")
    
    def register_endpoint(self, endpoint: Endpoint, capabilities: Capabilities) -> None:
        """Register an endpoint with its capabilities.
        
        Args:
            endpoint: Endpoint information
            capabilities: Detected capabilities
        """
        self._endpoints[endpoint.id] = endpoint
        self._capabilities[endpoint.id] = capabilities
        
        self.logger.info(
            "Endpoint registered",
            endpoint_id=endpoint.id,
            role=endpoint.role,
            capabilities=capabilities.dict(),
        )
    
    def get_capabilities(self, endpoint_id: str) -> Optional[Capabilities]:
        """Get capabilities for an endpoint.
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            Capabilities if found, None otherwise
        """
        return self._capabilities.get(endpoint_id)
    
    def get_endpoint(self, endpoint_id: str) -> Optional[Endpoint]:
        """Get endpoint information.
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            Endpoint if found, None otherwise
        """
        return self._endpoints.get(endpoint_id)
    
    def list_endpoints_by_capability(self, capability: str) -> List[str]:
        """List endpoints that support a specific capability.
        
        Args:
            capability: Capability name (e.g., 'udp_echo', 'lbm')
            
        Returns:
            List of endpoint IDs
        """
        result = []
        for endpoint_id, caps in self._capabilities.items():
            if hasattr(caps, capability) and getattr(caps, capability):
                result.append(endpoint_id)
        
        return result
    
    def get_capability_coverage(self) -> float:
        """Calculate capability coverage percentage.
        
        Returns:
            Percentage of endpoints with detected capabilities
        """
        if not self._endpoints:
            return 0.0
        
        endpoints_with_caps = len([
            ep_id for ep_id, caps in self._capabilities.items()
            if any([caps.udp_echo, caps.lbm, caps.ecpri_delay, caps.ccm_min])
        ])
        
        return (endpoints_with_caps / len(self._endpoints)) * 100.0
