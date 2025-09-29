"""UDP Echo collector for O-RAN supervision.
O-RAN supervision 기능을 위한 UDP Echo 수집기
"""

import asyncio
import socket
import time
from typing import Optional

from ncclient import manager
from ncclient.operations.errors import OperationError as RPCError

from ..core.models import Capabilities, Endpoint, MetricSample
from .base import BaseCollector


class UdpEchoCollector(BaseCollector):
    """UDP echo 측정을 위한 수집기.
    
    O-RAN supervision 모듈에서 정의된 UDP echo 기능을 사용하여
    CU-Plane 모니터링을 위한 RTT(Round Trip Time)를 측정합니다.
    """
    
    def can_collect(self, capabilities: Capabilities) -> bool:
        """UDP echo가 지원되는지 확인합니다.
        
        Args:
            capabilities: 엔드포인트 기능 정보
            
        Returns:
            UDP echo가 지원되면 True
        """
        return capabilities.udp_echo
    
    async def collect(self, endpoint: Endpoint, capabilities: Capabilities) -> Optional[MetricSample]:
        """UDP echo RTT 측정을 수집합니다.
        
        Args:
            endpoint: 측정할 엔드포인트
            capabilities: 엔드포인트 기능 정보
            
        Returns:
            UDP echo RTT가 포함된 메트릭 샘플
        """
        if not capabilities.udp_echo:
            return None
        
        try:
            # Get UDP echo configuration from the device
            echo_config = await self._get_echo_config(endpoint)
            if not echo_config:
                return None
            
            # Perform UDP echo measurement
            rtt_ms = await self._perform_udp_echo(
                endpoint.host,
                echo_config.get("port", 7),
                echo_config.get("payload_size", 64)
            )
            
            if rtt_ms is None:
                return None
            
            return MetricSample(
                endpoint_id=endpoint.id,
                ts_ms=int(time.time() * 1000),
                udp_echo_rtt_ms=rtt_ms,
            )
            
        except Exception as e:
            self.logger.error(
                "UDP echo collection failed",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
    
    async def _get_echo_config(self, endpoint: Endpoint) -> Optional[dict]:
        """Get UDP echo configuration from device.
        
        Args:
            endpoint: Endpoint to query
            
        Returns:
            Echo configuration or None
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
                
                # Get supervision configuration
                filter_xml = """
                <filter>
                    <supervision xmlns="urn:o-ran:supervision:1.0">
                        <cu-plane-monitoring>
                            <configured-cu-monitoring-interval/>
                        </cu-plane-monitoring>
                    </supervision>
                </filter>
                """
                
                result = mgr.get_config(source="running", filter=filter_xml)
                
                # Parse the result to extract echo configuration
                # This would depend on the specific YANG model
                # For now, return default configuration
                return {
                    "port": 7,  # Standard echo port
                    "payload_size": 64,
                    "interval": 60,
                }
                
        except RPCError as e:
            self.logger.warning(
                "Could not get echo config",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
        except Exception as e:
            self.logger.error(
                "Failed to get echo config",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
    
    async def _perform_udp_echo(self, host: str, port: int, payload_size: int) -> Optional[float]:
        """Perform UDP echo measurement.
        
        Args:
            host: Target host
            port: UDP port
            payload_size: Payload size in bytes
            
        Returns:
            RTT in milliseconds or None if failed
        """
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5.0)  # 5 second timeout
            
            # Prepare payload
            payload = b'X' * payload_size
            
            # Measure RTT
            start_time = time.time()
            sock.sendto(payload, (host, port))
            
            # Receive echo
            data, addr = sock.recvfrom(1024)
            end_time = time.time()
            
            sock.close()
            
            # Calculate RTT
            rtt_ms = (end_time - start_time) * 1000.0
            
            # Validate response
            if data != payload:
                self.logger.warning(
                    "UDP echo payload mismatch",
                    host=host,
                    expected_size=len(payload),
                    received_size=len(data),
                )
                return None
            
            return rtt_ms
            
        except socket.timeout:
            self.logger.warning("UDP echo timeout", host=host, port=port)
            return None
        except Exception as e:
            self.logger.error(
                "UDP echo failed",
                host=host,
                port=port,
                error=str(e),
            )
            return None
