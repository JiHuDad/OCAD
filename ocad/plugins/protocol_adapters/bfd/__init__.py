"""BFD (Bidirectional Forwarding Detection) protocol adapter plugin.

This plugin provides data collection for BFD sessions:
- Session state monitoring (Up, Down, AdminDown, Init)
- Detection time tracking
- Echo interval monitoring
- Remote state tracking
- Diagnostic code analysis

BFD is designed for fast failure detection (sub-second) between
network nodes. It's commonly used in routing protocols to detect
link failures quickly.
"""

from typing import AsyncIterator, Dict, List, Any
from datetime import datetime
from enum import Enum
import asyncio
import random

from ocad.plugins.base import ProtocolAdapter


class BFDState(Enum):
    """BFD session states."""
    ADMIN_DOWN = 0
    DOWN = 1
    INIT = 2
    UP = 3


class BFDDiagnostic(Enum):
    """BFD diagnostic codes."""
    NO_DIAGNOSTIC = 0
    CONTROL_DETECTION_TIME_EXPIRED = 1
    ECHO_FUNCTION_FAILED = 2
    NEIGHBOR_SIGNALED_SESSION_DOWN = 3
    FORWARDING_PLANE_RESET = 4
    PATH_DOWN = 5
    CONCATENATED_PATH_DOWN = 6
    ADMINISTRATIVELY_DOWN = 7
    REVERSE_CONCATENATED_PATH_DOWN = 8


class BFDAdapter(ProtocolAdapter):
    """BFD protocol adapter for fast failure detection.

    This adapter monitors BFD sessions and collects metrics:
    - Session state changes (state transitions)
    - Detection times (how long it takes to detect failures)
    - Diagnostic codes (failure reasons)
    - Echo intervals and multipliers

    BFD is critical for rapid network failover, so detecting
    anomalies like flapping (rapid Up/Down transitions) or
    slow detection times is important.
    """

    @property
    def name(self) -> str:
        return "bfd"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "bfd_session_state",
            "bfd_detection_time_ms",
            "bfd_echo_interval_ms",
            "bfd_remote_state",
            "bfd_diagnostic_code",
            "bfd_multiplier",
            "bfd_flap_count",  # Number of state changes
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate BFD adapter configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if "sessions" not in config:
            raise ValueError("BFD adapter requires 'sessions' in config")

        if not isinstance(config["sessions"], list):
            raise ValueError("'sessions' must be a list")

        for session in config["sessions"]:
            if "id" not in session:
                raise ValueError("Each BFD session must have an 'id'")
            if "local_ip" not in session:
                raise ValueError("Each BFD session must have a 'local_ip'")
            if "remote_ip" not in session:
                raise ValueError("Each BFD session must have a 'remote_ip'")

        return True

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Collect BFD session metrics.

        This is a simplified simulation for plugin system testing.
        Full implementation should use actual BFD protocol monitoring via:
        - SNMP (BFD-STD-MIB: .1.3.6.1.2.1.222)
        - NETCONF/YANG (ietf-bfd.yang)
        - Native BFD protocol parsing

        Args:
            config: Adapter configuration

        Yields:
            Metric dictionaries containing BFD session data
        """
        sessions = config.get("sessions", [])
        interval = config.get("interval_sec", 5)  # Check every 5 seconds

        # Simulated session state for testing
        session_states = {
            session["id"]: {
                "local_state": BFDState.UP,
                "remote_state": BFDState.UP,
                "flap_count": 0,
                "last_flap_time": datetime.utcnow(),
            }
            for session in sessions
        }

        while True:
            for session in sessions:
                session_id = session["id"]
                state_info = session_states[session_id]

                # Simulate occasional state changes (flapping)
                if random.random() < 0.05:  # 5% chance of state change
                    # Toggle state
                    if state_info["local_state"] == BFDState.UP:
                        state_info["local_state"] = BFDState.DOWN
                        diagnostic = BFDDiagnostic.CONTROL_DETECTION_TIME_EXPIRED
                    else:
                        state_info["local_state"] = BFDState.UP
                        diagnostic = BFDDiagnostic.NO_DIAGNOSTIC

                    state_info["flap_count"] += 1
                    state_info["last_flap_time"] = datetime.utcnow()
                else:
                    diagnostic = BFDDiagnostic.NO_DIAGNOSTIC

                # Session state metric (encoded as integer)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": session_id,
                    "metric_name": "bfd_session_state",
                    "value": float(state_info["local_state"].value),
                    "metadata": {
                        "protocol": "bfd",
                        "local_ip": session["local_ip"],
                        "remote_ip": session["remote_ip"],
                        "state_name": state_info["local_state"].name,
                        "remote_state_name": state_info["remote_state"].name,
                    },
                }

                # Detection time (simulated: normal ~15-50ms, slow during issues)
                if state_info["local_state"] == BFDState.DOWN:
                    detection_time = random.uniform(100, 300)  # Slower during failures
                else:
                    detection_time = random.uniform(15, 50)  # Normal

                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": session_id,
                    "metric_name": "bfd_detection_time_ms",
                    "value": detection_time,
                    "metadata": {
                        "protocol": "bfd",
                        "local_ip": session["local_ip"],
                        "remote_ip": session["remote_ip"],
                    },
                }

                # Remote state
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": session_id,
                    "metric_name": "bfd_remote_state",
                    "value": float(state_info["remote_state"].value),
                    "metadata": {
                        "protocol": "bfd",
                        "local_ip": session["local_ip"],
                        "remote_ip": session["remote_ip"],
                        "remote_state_name": state_info["remote_state"].name,
                    },
                }

                # Echo interval
                echo_interval = session.get("interval_ms", 50)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": session_id,
                    "metric_name": "bfd_echo_interval_ms",
                    "value": float(echo_interval),
                    "metadata": {
                        "protocol": "bfd",
                        "local_ip": session["local_ip"],
                        "remote_ip": session["remote_ip"],
                    },
                }

                # Diagnostic code
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": session_id,
                    "metric_name": "bfd_diagnostic_code",
                    "value": float(diagnostic.value),
                    "metadata": {
                        "protocol": "bfd",
                        "local_ip": session["local_ip"],
                        "remote_ip": session["remote_ip"],
                        "diagnostic_name": diagnostic.name,
                    },
                }

                # Flap count
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": session_id,
                    "metric_name": "bfd_flap_count",
                    "value": float(state_info["flap_count"]),
                    "metadata": {
                        "protocol": "bfd",
                        "local_ip": session["local_ip"],
                        "remote_ip": session["remote_ip"],
                        "last_flap_time": state_info["last_flap_time"].isoformat(),
                    },
                }

                # Multiplier
                multiplier = session.get("multiplier", 3)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": session_id,
                    "metric_name": "bfd_multiplier",
                    "value": float(multiplier),
                    "metadata": {
                        "protocol": "bfd",
                        "local_ip": session["local_ip"],
                        "remote_ip": session["remote_ip"],
                    },
                }

            await asyncio.sleep(interval)

    def get_recommended_models(self) -> List[str]:
        """Get recommended AI models for BFD protocol.

        Returns:
            List of detector names optimized for BFD characteristics
        """
        return ["lstm", "hmm", "cusum", "rule-based"]

    def get_description(self) -> str:
        """Get human-readable description of this adapter.

        Returns:
            Description string
        """
        return (
            f"{self.name} protocol adapter v{self.version} - "
            "Bidirectional Forwarding Detection for fast failure detection"
        )


def create_adapter() -> ProtocolAdapter:
    """Plugin entry point.

    Returns:
        BFD protocol adapter instance
    """
    return BFDAdapter()
