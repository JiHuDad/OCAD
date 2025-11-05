"""PTP (Precision Time Protocol) protocol adapter plugin.

This plugin provides data collection for PTP synchronization:
- Master-Slave hierarchy monitoring
- Offset from master tracking (nanosecond precision)
- Path delay measurement
- Clock drift monitoring (parts per billion)
- Port state tracking
- Sync/Delay/Announce interval monitoring

PTP is designed for sub-microsecond time synchronization in networks.
It's critical for applications requiring precise timing (5G, financial trading, etc).
"""

from typing import AsyncIterator, Dict, List, Any
from datetime import datetime
from enum import Enum
import asyncio
import random

from ocad.plugins.base import ProtocolAdapter


class PTPPortState(Enum):
    """PTP port states (IEEE 1588)."""
    INITIALIZING = 1
    FAULTY = 2
    DISABLED = 3
    LISTENING = 4
    PRE_MASTER = 5
    MASTER = 6
    PASSIVE = 7
    UNCALIBRATED = 8
    SLAVE = 9


class PTPAdapter(ProtocolAdapter):
    """PTP protocol adapter for precision time synchronization monitoring.

    This adapter monitors PTP clock synchronization and collects metrics:
    - Offset from master (nanosecond precision)
    - Mean path delay (network latency)
    - Clock drift (parts per billion)
    - Sync/Delay/Announce message intervals
    - Port state transitions
    - Master clock hierarchy (steps removed)

    PTP is critical for 5G fronthaul timing, so detecting anomalies like:
    - Excessive offset drift
    - Master clock changes
    - Path delay increases
    - Clock instability
    is essential for network reliability.
    """

    @property
    def name(self) -> str:
        return "ptp"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "ptp_offset_from_master_ns",
            "ptp_mean_path_delay_ns",
            "ptp_clock_drift_ppb",
            "ptp_sync_interval_ms",
            "ptp_delay_req_interval_ms",
            "ptp_announce_interval_ms",
            "ptp_port_state",
            "ptp_steps_removed",
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate PTP adapter configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if "slaves" not in config:
            raise ValueError("PTP adapter requires 'slaves' in config")

        if not isinstance(config["slaves"], list):
            raise ValueError("'slaves' must be a list")

        for slave in config["slaves"]:
            if "id" not in slave:
                raise ValueError("Each PTP slave must have an 'id'")
            if "clock_id" not in slave:
                raise ValueError("Each PTP slave must have a 'clock_id'")

        return True

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Collect PTP synchronization metrics.

        This is a simplified simulation for plugin system testing.
        Full implementation should use actual PTP monitoring via:
        - PTP management messages (TLV queries)
        - SNMP (PTP-MIB)
        - NETCONF/YANG (ietf-ptp.yang)
        - Linux ptp4l / phc2sys monitoring

        Args:
            config: Adapter configuration

        Yields:
            Metric dictionaries containing PTP synchronization data
        """
        slaves = config.get("slaves", [])
        interval = config.get("interval_sec", 5)  # Check every 5 seconds

        # Simulated slave state for testing
        slave_states = {
            slave["id"]: {
                "port_state": PTPPortState.SLAVE,
                "offset_from_master_ns": random.uniform(-50, 50),  # Initial offset
                "mean_path_delay_ns": random.uniform(1000, 5000),  # 1-5 microseconds
                "clock_drift_ppb": random.uniform(-10, 10),  # ±10 ppb
                "steps_removed": slave.get("steps_removed", 1),
                "master_clock_id": slave.get("master_clock_id", "00:00:00:00:00:00:00:01"),
            }
            for slave in slaves
        }

        while True:
            for slave in slaves:
                slave_id = slave["id"]
                state_info = slave_states[slave_id]

                # Simulate occasional master changes or synchronization issues
                anomaly_type = random.random()

                if anomaly_type < 0.02:  # 2% chance of master change
                    state_info["port_state"] = PTPPortState.LISTENING
                    state_info["steps_removed"] += 1
                    state_info["offset_from_master_ns"] = random.uniform(-500, 500)  # Large offset
                elif anomaly_type < 0.04:  # 2% chance of clock drift issue
                    state_info["clock_drift_ppb"] = random.uniform(50, 200)  # High drift
                    state_info["offset_from_master_ns"] += random.uniform(100, 500)  # Growing offset
                elif anomaly_type < 0.06:  # 2% chance of path delay increase
                    state_info["mean_path_delay_ns"] = random.uniform(20000, 50000)  # 20-50 microseconds
                else:
                    # Normal operation - small random walk
                    state_info["port_state"] = PTPPortState.SLAVE
                    state_info["offset_from_master_ns"] += random.uniform(-20, 20)
                    state_info["mean_path_delay_ns"] += random.uniform(-100, 100)
                    state_info["clock_drift_ppb"] += random.uniform(-2, 2)

                    # Keep offset and delay within reasonable bounds
                    state_info["offset_from_master_ns"] = max(-100, min(100,
                        state_info["offset_from_master_ns"]))
                    state_info["mean_path_delay_ns"] = max(1000, min(10000,
                        state_info["mean_path_delay_ns"]))
                    state_info["clock_drift_ppb"] = max(-20, min(20,
                        state_info["clock_drift_ppb"]))

                # Offset from master (nanoseconds)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_offset_from_master_ns",
                    "value": float(state_info["offset_from_master_ns"]),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                        "master_clock_id": state_info["master_clock_id"],
                        "port_state": state_info["port_state"].name,
                    },
                }

                # Mean path delay (nanoseconds)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_mean_path_delay_ns",
                    "value": float(state_info["mean_path_delay_ns"]),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                        "master_clock_id": state_info["master_clock_id"],
                    },
                }

                # Clock drift (parts per billion)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_clock_drift_ppb",
                    "value": float(state_info["clock_drift_ppb"]),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                    },
                }

                # Sync interval (milliseconds)
                sync_interval = slave.get("sync_interval_ms", 125)  # Default: 125ms (8 packets/sec)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_sync_interval_ms",
                    "value": float(sync_interval),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                    },
                }

                # Delay_Req interval (milliseconds)
                delay_req_interval = slave.get("delay_req_interval_ms", 1000)  # Default: 1 second
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_delay_req_interval_ms",
                    "value": float(delay_req_interval),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                    },
                }

                # Announce interval (milliseconds)
                announce_interval = slave.get("announce_interval_ms", 1000)  # Default: 1 second
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_announce_interval_ms",
                    "value": float(announce_interval),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                    },
                }

                # Port state (encoded as integer)
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_port_state",
                    "value": float(state_info["port_state"].value),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                        "state_name": state_info["port_state"].name,
                    },
                }

                # Steps removed from Grandmaster
                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": slave_id,
                    "metric_name": "ptp_steps_removed",
                    "value": float(state_info["steps_removed"]),
                    "metadata": {
                        "protocol": "ptp",
                        "clock_id": slave["clock_id"],
                        "master_clock_id": state_info["master_clock_id"],
                    },
                }

            await asyncio.sleep(interval)

    def get_recommended_models(self) -> List[str]:
        """Get recommended AI models for PTP protocol.

        Returns:
            List of detector names optimized for PTP characteristics
        """
        return ["tcn", "lstm", "autoencoder", "cusum"]

    def get_description(self) -> str:
        """Get human-readable description of this adapter.

        Returns:
            Description string
        """
        return (
            f"{self.name} protocol adapter v{self.version} - "
            "Precision Time Protocol for sub-microsecond synchronization"
        )


def create_adapter() -> ProtocolAdapter:
    """Plugin entry point.

    Returns:
        PTP protocol adapter instance
    """
    return PTPAdapter()
