"""CFM (Connectivity Fault Management) protocol adapter plugin.

This plugin provides data collection for ORAN CFM-Lite metrics:
- UDP Echo RTT
- eCPRI One-way Delay
- LBM (Loopback Message) RTT
- CCM (Continuity Check Message) minimal stats
"""

from typing import AsyncIterator, Dict, List, Any
from datetime import datetime
import asyncio

from ocad.plugins.base import ProtocolAdapter


class CFMAdapter(ProtocolAdapter):
    """CFM protocol adapter for ORAN networks.

    This adapter collects CFM-Lite metrics from ORAN equipment
    (O-RU, O-DU, Transport) via NETCONF/YANG or SNMP.

    Supported metrics:
    - udp_echo_rtt_ms: UDP Echo round-trip time
    - ecpri_delay_us: eCPRI one-way delay
    - lbm_rtt_ms: LBM round-trip time
    - ccm_miss_count: CCM message miss count
    """

    @property
    def name(self) -> str:
        return "cfm"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "udp_echo_rtt_ms",
            "ecpri_delay_us",
            "lbm_rtt_ms",
            "ccm_miss_count",
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate CFM adapter configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if "endpoints" not in config:
            raise ValueError("CFM adapter requires 'endpoints' in config")

        if not isinstance(config["endpoints"], list):
            raise ValueError("'endpoints' must be a list")

        for endpoint in config["endpoints"]:
            if "id" not in endpoint:
                raise ValueError("Each endpoint must have an 'id'")
            if "host" not in endpoint:
                raise ValueError("Each endpoint must have a 'host'")

        return True

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Collect CFM metrics from configured endpoints.

        This is a simplified version for plugin system testing.
        Full implementation should use actual NETCONF/YANG or SNMP collection.

        Args:
            config: Adapter configuration

        Yields:
            Metric dictionaries
        """
        endpoints = config.get("endpoints", [])
        interval = config.get("interval_sec", 10)

        while True:
            for endpoint in endpoints:
                endpoint_id = endpoint["id"]

                # Collect UDP Echo RTT
                if endpoint.get("udp_echo", True):
                    yield {
                        "timestamp": datetime.utcnow(),
                        "source_id": endpoint_id,
                        "metric_name": "udp_echo_rtt_ms",
                        "value": 5.0,  # Simulated value
                        "metadata": {
                            "protocol": "cfm",
                            "host": endpoint["host"],
                        },
                    }

                # Collect eCPRI delay
                if endpoint.get("ecpri", True):
                    yield {
                        "timestamp": datetime.utcnow(),
                        "source_id": endpoint_id,
                        "metric_name": "ecpri_delay_us",
                        "value": 100.0,  # Simulated value
                        "metadata": {
                            "protocol": "cfm",
                            "host": endpoint["host"],
                        },
                    }

                # Collect LBM RTT
                if endpoint.get("lbm", True):
                    yield {
                        "timestamp": datetime.utcnow(),
                        "source_id": endpoint_id,
                        "metric_name": "lbm_rtt_ms",
                        "value": 10.0,  # Simulated value
                        "metadata": {
                            "protocol": "cfm",
                            "host": endpoint["host"],
                        },
                    }

            await asyncio.sleep(interval)

    def get_recommended_models(self) -> List[str]:
        """Get recommended AI models for CFM protocol.

        Returns:
            List of detector names
        """
        return ["tcn", "isolation_forest", "cusum", "rule-based"]


def create_adapter() -> ProtocolAdapter:
    """Plugin entry point.

    Returns:
        CFM protocol adapter instance
    """
    return CFMAdapter()
