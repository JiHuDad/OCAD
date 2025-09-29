"""LBM (Loopback Message) collector for CFM."""

import time
from typing import Optional
import xml.etree.ElementTree as ET

from ncclient import manager
from ncclient.operations.errors import RPCError

from ..core.models import Capabilities, Endpoint, MetricSample
from .base import BaseCollector


class LbmCollector(BaseCollector):
    """Collector for LBM (CFM Loopback) measurements."""
    
    def can_collect(self, capabilities: Capabilities) -> bool:
        """Check if LBM is supported.
        
        Args:
            capabilities: Endpoint capabilities
            
        Returns:
            True if LBM is supported
        """
        return capabilities.lbm
    
    async def collect(self, endpoint: Endpoint, capabilities: Capabilities) -> Optional[MetricSample]:
        """Collect LBM measurement.
        
        Args:
            endpoint: Endpoint to collect from
            capabilities: Endpoint capabilities
            
        Returns:
            Metric sample with LBM RTT and success status
        """
        if not capabilities.lbm:
            return None
        
        try:
            # Perform LBM test
            lbm_result = await self._perform_lbm_test(endpoint)
            
            if lbm_result is None:
                return None
            
            return MetricSample(
                endpoint_id=endpoint.id,
                ts_ms=int(time.time() * 1000),
                lbm_rtt_ms=lbm_result.get("rtt_ms"),
                lbm_success=lbm_result.get("success", False),
            )
            
        except Exception as e:
            self.logger.error(
                "LBM collection failed",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
    
    async def _perform_lbm_test(self, endpoint: Endpoint) -> Optional[dict]:
        """Perform LBM test on the endpoint.
        
        Args:
            endpoint: Endpoint to test
            
        Returns:
            Dictionary with test results or None
        """
        try:
            with manager.connect(
                host=endpoint.host,
                port=endpoint.port,
                username=self.config.username,
                password=self.config.password,
                timeout=self.config.timeout,
                hostkey_verify=self.config.hostkey_verify,
            ) as mgr:
                
                # First, get the CFM configuration to find MEPs
                mep_info = await self._get_cfm_mep_info(mgr)
                if not mep_info:
                    return None
                
                # Perform loopback test
                lbm_result = await self._execute_lbm_rpc(mgr, mep_info)
                
                return lbm_result
                
        except RPCError as e:
            self.logger.warning(
                "LBM RPC failed",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
        except Exception as e:
            self.logger.error(
                "LBM test failed",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
    
    async def _get_cfm_mep_info(self, mgr: manager.Manager) -> Optional[dict]:
        """Get CFM MEP information.
        
        Args:
            mgr: NETCONF manager
            
        Returns:
            MEP information or None
        """
        try:
            # Get CFM configuration
            filter_xml = """
            <filter>
                <interfaces xmlns="urn:ietf:yang:ietf-interfaces">
                    <interface>
                        <dot1ag-cfm xmlns="urn:ieee:yang:ieee802-dot1ag-cfm">
                            <mep>
                                <mep-id/>
                                <direction/>
                                <md-name/>
                                <ma-name/>
                            </mep>
                        </dot1ag-cfm>
                    </interface>
                </interfaces>
            </filter>
            """
            
            result = mgr.get_config(source="running", filter=filter_xml)
            
            # Parse MEP information
            mep_info = self._parse_mep_config(result.data)
            
            return mep_info
            
        except Exception as e:
            self.logger.debug("Could not get MEP info", error=str(e))
            # Return default MEP info for testing
            return {
                "mep_id": 1,
                "md_name": "default",
                "ma_name": "default",
                "interface": "eth0",
            }
    
    def _parse_mep_config(self, xml_data: str) -> Optional[dict]:
        """Parse MEP configuration from XML.
        
        Args:
            xml_data: XML response
            
        Returns:
            MEP information
        """
        try:
            root = ET.fromstring(xml_data)
            
            namespaces = {
                'if': 'urn:ietf:yang:ietf-interfaces',
                'cfm': 'urn:ieee:yang:ieee802-dot1ag-cfm',
            }
            
            # Find first MEP
            mep_elem = root.find('.//cfm:mep', namespaces)
            if mep_elem is not None:
                mep_id_elem = mep_elem.find('cfm:mep-id', namespaces)
                md_name_elem = mep_elem.find('cfm:md-name', namespaces)
                ma_name_elem = mep_elem.find('cfm:ma-name', namespaces)
                
                return {
                    "mep_id": int(mep_id_elem.text) if mep_id_elem is not None else 1,
                    "md_name": md_name_elem.text if md_name_elem is not None else "default",
                    "ma_name": ma_name_elem.text if ma_name_elem is not None else "default",
                    "interface": "eth0",  # Default interface
                }
            
            # Return default if no MEP found
            return {
                "mep_id": 1,
                "md_name": "default",
                "ma_name": "default", 
                "interface": "eth0",
            }
            
        except ET.ParseError:
            return None
    
    async def _execute_lbm_rpc(self, mgr: manager.Manager, mep_info: dict) -> Optional[dict]:
        """Execute LBM RPC operation.
        
        Args:
            mgr: NETCONF manager
            mep_info: MEP information
            
        Returns:
            LBM test results
        """
        try:
            # Build LBM RPC
            lbm_rpc = f"""
            <loopback-message xmlns="urn:ieee:yang:ieee802-dot1ag-cfm">
                <target-mep-id>{mep_info['mep_id']}</target-mep-id>
                <md-name>{mep_info['md_name']}</md-name>
                <ma-name>{mep_info['ma_name']}</ma-name>
                <number-of-messages>1</number-of-messages>
                <data-tlv>
                    <data-length>64</data-length>
                </data-tlv>
            </loopback-message>
            """
            
            # Record start time
            start_time = time.time()
            
            # Execute RPC
            result = mgr.dispatch(ET.fromstring(lbm_rpc))
            
            # Calculate RTT
            end_time = time.time()
            rtt_ms = (end_time - start_time) * 1000.0
            
            # Parse result
            success = self._parse_lbm_result(result.data)
            
            return {
                "rtt_ms": rtt_ms,
                "success": success,
            }
            
        except Exception as e:
            self.logger.debug("LBM RPC execution failed", error=str(e))
            return {
                "rtt_ms": None,
                "success": False,
            }
    
    def _parse_lbm_result(self, xml_data: str) -> bool:
        """Parse LBM result.
        
        Args:
            xml_data: XML response
            
        Returns:
            True if LBM was successful
        """
        try:
            root = ET.fromstring(xml_data)
            
            # Look for success indicators
            # This depends on the vendor implementation
            success_indicators = [
                "success",
                "ok", 
                "received",
                "reply",
            ]
            
            xml_text = xml_data.lower()
            for indicator in success_indicators:
                if indicator in xml_text:
                    return True
            
            # Check for error indicators
            error_indicators = [
                "error",
                "failed",
                "timeout",
                "unreachable",
            ]
            
            for indicator in error_indicators:
                if indicator in xml_text:
                    return False
            
            # If no clear indicator, assume success if we got a response
            return True
            
        except ET.ParseError:
            return False
        except Exception:
            return False
