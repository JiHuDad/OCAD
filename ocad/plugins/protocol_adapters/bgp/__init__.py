"""BGP (Border Gateway Protocol) protocol adapter plugin.

This plugin provides data collection for BGP sessions:
- BGP FSM (Finite State Machine) state monitoring
- UPDATE/WITHDRAW message counting
- Prefix count tracking
- AS-path length monitoring
- Route flapping detection
- Peer uptime tracking

BGP is the core routing protocol of the Internet, responsible for
exchanging routing information between autonomous systems (AS).
Detecting anomalies like prefix hijacking, route flapping, or
unusual AS-path patterns is critical for network security.
"""

from typing import AsyncIterator, Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import random

from ocad.plugins.base import ProtocolAdapter


class BGPState(Enum):
    """BGP FSM (Finite State Machine) states."""
    IDLE = 0
    CONNECT = 1
    ACTIVE = 2
    OPEN_SENT = 3
    OPEN_CONFIRM = 4
    ESTABLISHED = 5


class BGPAdapter(ProtocolAdapter):
    """BGP protocol adapter for routing anomaly detection.

    This adapter monitors BGP sessions and collects metrics:
    - Session state (BGP FSM: Idle, Connect, Active, OpenSent, OpenConfirm, Established)
    - UPDATE message counts (route advertisements)
    - WITHDRAW message counts (route withdrawals)
    - Prefix counts (number of advertised prefixes)
    - AS-path lengths (routing path complexity)
    - Route flapping counts (rapid route changes)
    - Peer uptime (session stability)

    BGP anomalies can indicate:
    - Prefix hijacking (unauthorized route advertisements)
    - Route flapping (unstable routes)
    - AS-path poisoning (malicious path manipulation)
    - Session instability (peer connection issues)
    """

    @property
    def name(self) -> str:
        return "bgp"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "bgp_session_state",
            "bgp_update_count",
            "bgp_withdraw_count",
            "bgp_prefix_count",
            "bgp_as_path_length",
            "bgp_route_flap_count",
            "bgp_peer_uptime_sec",
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate BGP adapter configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if "peers" not in config:
            raise ValueError("BGP adapter requires 'peers' in config")

        if not isinstance(config["peers"], list):
            raise ValueError("'peers' must be a list")

        for peer in config["peers"]:
            if "id" not in peer:
                raise ValueError("Each BGP peer must have an 'id'")
            if "local_asn" not in peer:
                raise ValueError("Each BGP peer must have a 'local_asn'")
            if "remote_asn" not in peer:
                raise ValueError("Each BGP peer must have a 'remote_asn'")
            if "remote_ip" not in peer:
                raise ValueError("Each BGP peer must have a 'remote_ip'")

        return True

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Collect BGP session metrics.

        This is a simplified simulation for plugin system testing.
        Full implementation should use actual BGP monitoring via:
        - SNMP (BGP4-MIB: .1.3.6.1.2.1.15)
        - NETCONF/YANG (ietf-bgp.yang)
        - BGP monitoring protocol (BMP, RFC 7854)
        - ExaBGP or GoBGP for BGP session monitoring

        Args:
            config: Adapter configuration

        Yields:
            Metric dictionaries containing BGP session data
        """
        peers = config.get("peers", [])
        interval = config.get("interval_sec", 10)  # Check every 10 seconds

        # Simulated peer state for testing
        peer_states = {
            peer["id"]: {
                "state": BGPState.ESTABLISHED,
                "update_count": 0,
                "withdraw_count": 0,
                "prefix_count": random.randint(50, 200),
                "route_flap_count": 0,
                "session_start": datetime.utcnow(),
                "last_update_time": datetime.utcnow(),
            }
            for peer in peers
        }

        while True:
            for peer in peers:
                peer_id = peer["id"]
                state_info = peer_states[peer_id]

                # Simulate occasional BGP activity
                # 1. Route updates (normal: 1-5 updates per interval)
                update_count_delta = random.randint(1, 5)
                state_info["update_count"] += update_count_delta

                # 2. Route withdrawals (occasional: 0-2 per interval)
                if random.random() < 0.3:  # 30% chance
                    withdraw_count_delta = random.randint(1, 2)
                    state_info["withdraw_count"] += withdraw_count_delta
                    state_info["prefix_count"] -= withdraw_count_delta
                    state_info["prefix_count"] = max(10, state_info["prefix_count"])

                # 3. Simulate route flapping (rare: 5% chance)
                if random.random() < 0.05:
                    state_info["route_flap_count"] += 1
                    # Rapid withdraw and re-advertise
                    state_info["withdraw_count"] += 3
                    state_info["update_count"] += 3

                # 4. Simulate session restart (very rare: 1% chance)
                if random.random() < 0.01:
                    # Session goes down and comes back up
                    state_info["state"] = BGPState.IDLE
                    state_info["session_start"] = datetime.utcnow()
                    state_info["route_flap_count"] += 10  # Major disruption
                    await asyncio.sleep(0.5)  # Brief downtime
                    state_info["state"] = BGPState.ESTABLISHED

                # Session state metric
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": peer_id,
                    "metric_name": "bgp_session_state",
                    "value": float(state_info["state"].value),
                    "metadata": {
                        "protocol": "bgp",
                        "local_asn": peer["local_asn"],
                        "remote_asn": peer["remote_asn"],
                        "remote_ip": peer["remote_ip"],
                        "state_name": state_info["state"].name,
                    },
                }

                # UPDATE message count
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": peer_id,
                    "metric_name": "bgp_update_count",
                    "value": float(state_info["update_count"]),
                    "metadata": {
                        "protocol": "bgp",
                        "remote_asn": peer["remote_asn"],
                        "remote_ip": peer["remote_ip"],
                        "delta": update_count_delta,
                    },
                }

                # WITHDRAW message count
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": peer_id,
                    "metric_name": "bgp_withdraw_count",
                    "value": float(state_info["withdraw_count"]),
                    "metadata": {
                        "protocol": "bgp",
                        "remote_asn": peer["remote_asn"],
                        "remote_ip": peer["remote_ip"],
                    },
                }

                # Prefix count
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": peer_id,
                    "metric_name": "bgp_prefix_count",
                    "value": float(state_info["prefix_count"]),
                    "metadata": {
                        "protocol": "bgp",
                        "remote_asn": peer["remote_asn"],
                        "remote_ip": peer["remote_ip"],
                    },
                }

                # AS-path length (simulated: normal 3-7, unusual 10-15)
                if random.random() < 0.1:  # 10% chance of unusual path
                    as_path_length = random.randint(10, 15)  # Unusual
                else:
                    as_path_length = random.randint(3, 7)  # Normal

                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": peer_id,
                    "metric_name": "bgp_as_path_length",
                    "value": float(as_path_length),
                    "metadata": {
                        "protocol": "bgp",
                        "remote_asn": peer["remote_asn"],
                        "remote_ip": peer["remote_ip"],
                    },
                }

                # Route flap count
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": peer_id,
                    "metric_name": "bgp_route_flap_count",
                    "value": float(state_info["route_flap_count"]),
                    "metadata": {
                        "protocol": "bgp",
                        "remote_asn": peer["remote_asn"],
                        "remote_ip": peer["remote_ip"],
                    },
                }

                # Peer uptime (seconds since session established)
                uptime = (datetime.utcnow() - state_info["session_start"]).total_seconds()
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": peer_id,
                    "metric_name": "bgp_peer_uptime_sec",
                    "value": float(uptime),
                    "metadata": {
                        "protocol": "bgp",
                        "remote_asn": peer["remote_asn"],
                        "remote_ip": peer["remote_ip"],
                        "session_start": state_info["session_start"].isoformat(),
                    },
                }

            await asyncio.sleep(interval)

    def get_recommended_models(self) -> List[str]:
        """Get recommended AI models for BGP protocol.

        Returns:
            List of detector names optimized for BGP characteristics
        """
        return ["gnn", "hmm", "lstm", "rule-based"]

    def get_description(self) -> str:
        """Get human-readable description of this adapter.

        Returns:
            Description string
        """
        return (
            f"{self.name} protocol adapter v{self.version} - "
            "Border Gateway Protocol for routing anomaly detection"
        )


def create_adapter() -> ProtocolAdapter:
    """Plugin entry point.

    Returns:
        BGP protocol adapter instance
    """
    return BGPAdapter()
