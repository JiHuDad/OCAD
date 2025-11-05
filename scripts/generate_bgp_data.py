#!/usr/bin/env python3
"""Generate synthetic BGP (Border Gateway Protocol) training data.

This script generates realistic BGP session data for training anomaly detection models.
It simulates:
- Normal BGP behavior (stable sessions, regular route updates)
- Anomalies: route flapping, prefix hijacking, AS-path poisoning, session instability

The data is saved in multiple formats for flexibility:
- CSV: Human-readable format
- Parquet: Efficient columnar format for training

Usage:
    # Generate 5 hours of data from 10 BGP peers with 10% anomaly rate
    python scripts/generate_bgp_data.py --peers 10 --duration-hours 5 --anomaly-rate 0.1

    # Quick test data (5 minutes)
    python scripts/generate_bgp_data.py --peers 3 --duration-minutes 5 --anomaly-rate 0.2

    # Large training dataset (24 hours, 50 peers)
    python scripts/generate_bgp_data.py --peers 50 --duration-hours 24 --anomaly-rate 0.05
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd


# BGP FSM state definitions (matching BGPAdapter)
class BGPState:
    IDLE = 0
    CONNECT = 1
    ACTIVE = 2
    OPEN_SENT = 3
    OPEN_CONFIRM = 4
    ESTABLISHED = 5

    NAMES = {
        0: "IDLE",
        1: "CONNECT",
        2: "ACTIVE",
        3: "OPEN_SENT",
        4: "OPEN_CONFIRM",
        5: "ESTABLISHED",
    }


class BGPPeerSimulator:
    """Simulator for a single BGP peer session."""

    def __init__(
        self,
        peer_id: str,
        local_asn: int,
        remote_asn: int,
        remote_ip: str,
    ):
        """Initialize BGP peer simulator.

        Args:
            peer_id: Unique peer identifier
            local_asn: Local AS number
            remote_asn: Remote AS number
            remote_ip: Remote peer IP address
        """
        self.peer_id = peer_id
        self.local_asn = local_asn
        self.remote_asn = remote_asn
        self.remote_ip = remote_ip

        # Session state
        self.state = BGPState.ESTABLISHED
        self.session_start = datetime.utcnow()

        # Counters (cumulative)
        self.update_count = 0
        self.withdraw_count = 0
        self.route_flap_count = 0

        # Current values
        self.prefix_count = random.randint(50, 200)  # Initial prefix count

        # Anomaly flags
        self.is_flapping = False
        self.flap_start_time = None
        self.flap_duration = 0
        self.is_hijacking = False
        self.hijack_duration = 0

    def collect_metrics(self, timestamp: datetime, is_anomaly_period: bool) -> Dict[str, Any]:
        """Collect metrics for current timestamp.

        Args:
            timestamp: Current timestamp
            is_anomaly_period: Whether this is an anomaly period

        Returns:
            Dictionary of metrics
        """
        # Normal BGP behavior: periodic route updates
        normal_update_rate = random.randint(1, 5)  # 1-5 updates per interval
        normal_withdraw_rate = random.randint(0, 2)  # 0-2 withdrawals per interval

        # Initialize per-interval metrics
        update_delta = normal_update_rate
        withdraw_delta = normal_withdraw_rate

        if is_anomaly_period:
            # Anomaly scenarios
            anomaly_type = random.choice(["flapping", "hijacking", "session_reset"])

            if anomaly_type == "flapping":
                # Route flapping: excessive route changes
                if not self.is_flapping and random.random() < 0.3:
                    # Start flapping
                    self.is_flapping = True
                    self.flap_start_time = timestamp
                    self.flap_duration = random.randint(5, 30)  # Flap for 5-30 intervals

                if self.is_flapping:
                    # Rapid route changes
                    update_delta = random.randint(20, 50)  # Excessive updates
                    withdraw_delta = random.randint(10, 30)  # Excessive withdrawals
                    self.route_flap_count += random.randint(5, 15)

                    # Prefix count fluctuates wildly
                    self.prefix_count += random.randint(-30, -10)
                    self.prefix_count = max(10, self.prefix_count)

                    # End flapping after duration
                    self.flap_duration -= 1
                    if self.flap_duration <= 0:
                        self.is_flapping = False
                        self.prefix_count = random.randint(50, 200)  # Stabilize

            elif anomaly_type == "hijacking":
                # Prefix hijacking: sudden prefix count spike
                if not self.is_hijacking and random.random() < 0.2:
                    self.is_hijacking = True
                    self.hijack_duration = random.randint(10, 50)
                    # Hijacked: many new prefixes advertised
                    self.prefix_count += random.randint(100, 300)
                    update_delta = random.randint(50, 100)

                if self.is_hijacking:
                    self.hijack_duration -= 1
                    if self.hijack_duration <= 0:
                        self.is_hijacking = False
                        self.prefix_count = random.randint(50, 200)  # Return to normal

            elif anomaly_type == "session_reset":
                # Session instability: state goes down and restarts
                if random.random() < 0.05:
                    self.state = BGPState.IDLE
                    self.session_start = timestamp
                    # Session reset causes route withdrawals
                    withdraw_delta = self.prefix_count
                    self.prefix_count = 0
                    self.route_flap_count += 20  # Major disruption

        else:
            # Normal behavior: stable session
            if self.state != BGPState.ESTABLISHED:
                # Session comes back up
                self.state = BGPState.ESTABLISHED
                # Re-advertise routes
                self.prefix_count = random.randint(50, 200)
                update_delta = self.prefix_count

            # Small natural variations
            prefix_change = random.randint(-2, 3)
            self.prefix_count += prefix_change
            self.prefix_count = max(10, self.prefix_count)

        # Update counters
        self.update_count += update_delta
        self.withdraw_count += withdraw_delta

        # AS-path length (normal: 3-7, unusual: 10-15 during anomalies)
        if is_anomaly_period and random.random() < 0.3:
            as_path_length = random.randint(10, 15)  # AS-path poisoning
        else:
            as_path_length = random.randint(3, 7)  # Normal

        # Peer uptime
        uptime_sec = (timestamp - self.session_start).total_seconds()

        return {
            "timestamp": timestamp,
            "source_id": self.peer_id,
            "local_asn": self.local_asn,
            "remote_asn": self.remote_asn,
            "remote_ip": self.remote_ip,
            "session_state": self.state,
            "session_state_name": BGPState.NAMES[self.state],
            "update_count": self.update_count,
            "withdraw_count": self.withdraw_count,
            "prefix_count": self.prefix_count,
            "as_path_length": as_path_length,
            "route_flap_count": self.route_flap_count,
            "peer_uptime_sec": uptime_sec,
            "is_anomaly": is_anomaly_period,
            "update_delta": update_delta,
            "withdraw_delta": withdraw_delta,
        }


def generate_bgp_data(
    n_peers: int,
    duration_seconds: int,
    collection_interval: int,
    anomaly_rate: float,
    output_dir: Path,
) -> None:
    """Generate BGP training data.

    Args:
        n_peers: Number of BGP peers to simulate
        duration_seconds: Duration of data collection in seconds
        collection_interval: Collection interval in seconds
        anomaly_rate: Fraction of time with anomalies (0.0-1.0)
        output_dir: Output directory for data files
    """
    print(f"Generating BGP data...")
    print(f"  Peers: {n_peers}")
    print(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Anomaly rate: {anomaly_rate * 100:.1f}%")

    # Create simulators for each peer
    simulators = []
    for i in range(n_peers):
        peer_id = f"bgp-peer-{i+1}"
        local_asn = 65000  # Private ASN
        remote_asn = 64500 + (i % 100)  # Varied remote ASNs
        remote_ip = f"10.{i // 254}.{(i % 254) + 1}.1"

        simulator = BGPPeerSimulator(
            peer_id=peer_id,
            local_asn=local_asn,
            remote_asn=remote_asn,
            remote_ip=remote_ip,
        )
        simulators.append(simulator)

    # Generate time series
    start_time = datetime.utcnow()
    n_collections = duration_seconds // collection_interval

    all_metrics = []

    print(f"\nGenerating {n_collections} collection cycles...")

    for cycle in range(n_collections):
        timestamp = start_time + timedelta(seconds=cycle * collection_interval)

        # Determine if this is an anomaly period
        is_anomaly_period = random.random() < anomaly_rate

        # Collect metrics from all peers
        for simulator in simulators:
            metrics = simulator.collect_metrics(timestamp, is_anomaly_period)
            all_metrics.append(metrics)

        # Progress indicator
        if (cycle + 1) % 100 == 0:
            progress = (cycle + 1) / n_collections * 100
            print(f"  Progress: {progress:.1f}% ({cycle + 1}/{n_collections} cycles)")

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Add derived features for ML
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Statistics
    n_total = len(df)
    n_anomalies = df["is_anomaly"].sum()
    anomaly_pct = n_anomalies / n_total * 100

    print(f"\nGenerated {n_total:,} records")
    print(f"  Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)")
    print(f"  Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)")

    # Flapping statistics
    n_flaps = df.groupby("source_id")["route_flap_count"].max().sum()
    print(f"  Total route flaps: {n_flaps}")

    # Update/Withdraw statistics
    total_updates = df["update_count"].max()
    total_withdraws = df["withdraw_count"].max()
    print(f"  Total updates: {total_updates:,}")
    print(f"  Total withdrawals: {total_withdraws:,}")

    # Save to files
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV (human-readable)
    csv_path = output_dir / f"bgp_data_{timestamp_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Parquet (efficient for training)
    parquet_path = output_dir / f"bgp_data_{timestamp_str}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")

    # Save summary statistics
    summary_path = output_dir / f"bgp_data_{timestamp_str}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("BGP Training Data Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Peers: {n_peers}\n")
        f.write(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m\n")
        f.write(f"  Collection interval: {collection_interval}s\n")
        f.write(f"  Anomaly rate: {anomaly_rate * 100:.1f}%\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Total records: {n_total:,}\n")
        f.write(f"  Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)\n")
        f.write(f"  Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)\n")
        f.write(f"  Total route flaps: {n_flaps}\n")
        f.write(f"  Total updates: {total_updates:,}\n")
        f.write(f"  Total withdrawals: {total_withdraws:,}\n\n")
        f.write(f"Files:\n")
        f.write(f"  CSV: {csv_path.name}\n")
        f.write(f"  Parquet: {parquet_path.name}\n")

    print(f"Saved summary: {summary_path}")

    print("\n✅ Data generation completed successfully!")
    print(f"\nUse this data for training:")
    print(f"  python scripts/train_gnn_bgp.py --data {parquet_path}")
    print(f"  python scripts/train_lstm_bgp.py --data {parquet_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic BGP training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--peers",
        type=int,
        default=10,
        help="Number of BGP peers to simulate (default: 10)",
    )

    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument(
        "--duration-hours",
        type=float,
        help="Duration in hours",
    )
    duration_group.add_argument(
        "--duration-minutes",
        type=float,
        help="Duration in minutes",
    )
    duration_group.add_argument(
        "--duration-seconds",
        type=int,
        default=3600,
        help="Duration in seconds (default: 3600 = 1 hour)",
    )

    parser.add_argument(
        "--collection-interval",
        type=int,
        default=10,
        help="Collection interval in seconds (default: 10)",
    )

    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.1,
        help="Fraction of time with anomalies, 0.0-1.0 (default: 0.1 = 10%%)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/bgp"),
        help="Output directory (default: data/training/bgp)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Calculate duration
    if args.duration_hours:
        duration_seconds = int(args.duration_hours * 3600)
    elif args.duration_minutes:
        duration_seconds = int(args.duration_minutes * 60)
    else:
        duration_seconds = args.duration_seconds

    # Validate parameters
    if args.peers <= 0:
        parser.error("--peers must be positive")
    if duration_seconds <= 0:
        parser.error("Duration must be positive")
    if args.collection_interval <= 0:
        parser.error("--collection-interval must be positive")
    if not (0.0 <= args.anomaly_rate <= 1.0):
        parser.error("--anomaly-rate must be between 0.0 and 1.0")

    # Generate data
    generate_bgp_data(
        n_peers=args.peers,
        duration_seconds=duration_seconds,
        collection_interval=args.collection_interval,
        anomaly_rate=args.anomaly_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
