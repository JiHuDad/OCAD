#!/usr/bin/env python3
"""Generate BGP ML training/validation datasets.

This script generates three types of datasets:
1. train/ - Normal data for training (anomaly_rate=0.0, large volume)
2. val_normal/ - Normal data for validation (anomaly_rate=0.0, medium size)
3. val_anomaly/ - Anomalous data for validation (anomaly_rate=0.8-1.0, medium size)

Usage:
    python scripts/generate_bgp_ml_data.py --output data/bgp

    # Custom configuration
    python scripts/generate_bgp_ml_data.py --peers 20 --train-hours 5 --val-hours 1 --output data/bgp
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import sys


# BGP FSM state definitions
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
        self.peer_id = peer_id
        self.local_asn = local_asn
        self.remote_asn = remote_asn
        self.remote_ip = remote_ip

        # Session state
        self.state = BGPState.ESTABLISHED
        self.session_start = datetime.utcnow()

        # Counters
        self.update_count = 0
        self.withdraw_count = 0
        self.route_flap_count = 0

        # Current values
        self.prefix_count = random.randint(50, 200)

        # Anomaly flags
        self.is_flapping = False
        self.flap_duration = 0
        self.is_hijacking = False
        self.hijack_duration = 0

    def collect_metrics(self, timestamp: datetime, is_anomaly_period: bool) -> Dict[str, Any]:
        """Collect metrics for current timestamp."""
        # Normal behavior
        normal_update_rate = random.randint(1, 5)
        normal_withdraw_rate = random.randint(0, 2)

        update_delta = normal_update_rate
        withdraw_delta = normal_withdraw_rate

        if is_anomaly_period:
            # Anomaly scenarios
            anomaly_type = random.choice(["flapping", "hijacking", "session_reset"])

            if anomaly_type == "flapping":
                if not self.is_flapping and random.random() < 0.3:
                    self.is_flapping = True
                    self.flap_duration = random.randint(5, 30)

                if self.is_flapping:
                    update_delta = random.randint(20, 50)
                    withdraw_delta = random.randint(10, 30)
                    self.route_flap_count += random.randint(5, 15)

                    self.prefix_count += random.randint(-30, -10)
                    self.prefix_count = max(10, self.prefix_count)

                    self.flap_duration -= 1
                    if self.flap_duration <= 0:
                        self.is_flapping = False
                        self.prefix_count = random.randint(50, 200)

            elif anomaly_type == "hijacking":
                if not self.is_hijacking and random.random() < 0.2:
                    self.is_hijacking = True
                    self.hijack_duration = random.randint(10, 50)
                    self.prefix_count += random.randint(100, 300)
                    update_delta = random.randint(50, 100)

                if self.is_hijacking:
                    self.hijack_duration -= 1
                    if self.hijack_duration <= 0:
                        self.is_hijacking = False
                        self.prefix_count = random.randint(50, 200)

            elif anomaly_type == "session_reset":
                if random.random() < 0.05:
                    self.state = BGPState.IDLE
                    self.session_start = timestamp
                    withdraw_delta = self.prefix_count
                    self.prefix_count = 0
                    self.route_flap_count += 20

        else:
            # Normal behavior
            if self.state != BGPState.ESTABLISHED:
                self.state = BGPState.ESTABLISHED
                self.prefix_count = random.randint(50, 200)
                update_delta = self.prefix_count

            prefix_change = random.randint(-2, 3)
            self.prefix_count += prefix_change
            self.prefix_count = max(10, self.prefix_count)

        # Update counters
        self.update_count += update_delta
        self.withdraw_count += withdraw_delta

        # AS-path length
        if is_anomaly_period and random.random() < 0.3:
            as_path_length = random.randint(10, 15)
        else:
            as_path_length = random.randint(3, 7)

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


def generate_dataset(
    dataset_name: str,
    n_peers: int,
    duration_seconds: int,
    collection_interval: int,
    anomaly_rate: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a single dataset (train/val_normal/val_anomaly)."""
    print(f"\n{'='*60}")
    print(f"Generating dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Peers: {n_peers}")
    print(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Anomaly rate: {anomaly_rate * 100:.1f}%")

    # Create simulators
    simulators = []
    for i in range(n_peers):
        peer_id = f"bgp-peer-{i+1}"
        local_asn = 65000
        remote_asn = 64500 + (i % 100)
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

    for cycle in range(n_collections):
        timestamp = start_time + timedelta(seconds=cycle * collection_interval)

        is_anomaly_period = random.random() < anomaly_rate

        for simulator in simulators:
            metrics = simulator.collect_metrics(timestamp, is_anomaly_period)
            all_metrics.append(metrics)

        if (cycle + 1) % 100 == 0:
            progress = (cycle + 1) / n_collections * 100
            print(f"  Progress: {progress:.1f}% ({cycle + 1}/{n_collections} cycles)")

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Add derived features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Statistics
    n_total = len(df)
    n_anomalies = df["is_anomaly"].sum()
    anomaly_pct = n_anomalies / n_total * 100

    print(f"\nGenerated {n_total:,} records")
    print(f"  Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)")
    print(f"  Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)")

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    parquet_path = output_dir / f"{dataset_name}_{timestamp_str}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved: {parquet_path}")

    return {
        "dataset_name": dataset_name,
        "path": str(parquet_path),
        "n_records": n_total,
        "n_anomalies": n_anomalies,
        "anomaly_pct": anomaly_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate BGP ML training/validation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--peers",
        type=int,
        default=10,
        help="Number of BGP peers to simulate (default: 10)",
    )

    parser.add_argument(
        "--train-hours",
        type=float,
        default=2.0,
        help="Training data duration in hours (default: 2.0)",
    )

    parser.add_argument(
        "--val-hours",
        type=float,
        default=0.5,
        help="Validation data duration in hours (default: 0.5)",
    )

    parser.add_argument(
        "--collection-interval",
        type=int,
        default=10,
        help="Collection interval in seconds (default: 10)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bgp"),
        help="Output directory (default: data/bgp)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Calculate durations
    train_seconds = int(args.train_hours * 3600)
    val_seconds = int(args.val_hours * 3600)

    # Generate datasets
    results = []

    # 1. Training data (normal only)
    train_dir = args.output / "train"
    result = generate_dataset(
        dataset_name="train",
        n_peers=args.peers,
        duration_seconds=train_seconds,
        collection_interval=args.collection_interval,
        anomaly_rate=0.0,  # Normal only
        output_dir=train_dir,
    )
    results.append(result)

    # 2. Validation normal
    val_normal_dir = args.output / "val_normal"
    result = generate_dataset(
        dataset_name="val_normal",
        n_peers=args.peers,
        duration_seconds=val_seconds,
        collection_interval=args.collection_interval,
        anomaly_rate=0.0,  # Normal only
        output_dir=val_normal_dir,
    )
    results.append(result)

    # 3. Validation anomaly
    val_anomaly_dir = args.output / "val_anomaly"
    result = generate_dataset(
        dataset_name="val_anomaly",
        n_peers=args.peers,
        duration_seconds=val_seconds,
        collection_interval=args.collection_interval,
        anomaly_rate=0.9,  # 90% anomalies
        output_dir=val_anomaly_dir,
    )
    results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['dataset_name']}:")
        print(f"  Path: {result['path']}")
        print(f"  Records: {result['n_records']:,}")
        print(f"  Anomalies: {result['n_anomalies']:,} ({result['anomaly_pct']:.1f}%)")

    print("\n✅ All datasets generated successfully!")
    print(f"\nNext steps:")
    print(f"  1. Train: python scripts/train_bgp_gnn.py --data {train_dir}")
    print(f"  2. Infer: python scripts/infer_bgp.py --model models/bgp/gnn_v1.0.0.pth --data {val_normal_dir} {val_anomaly_dir}")
    print(f"  3. Report: python scripts/report_bgp.py --predictions results/bgp/predictions.csv")


if __name__ == "__main__":
    main()
