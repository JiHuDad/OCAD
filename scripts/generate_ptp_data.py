#!/usr/bin/env python3
"""Generate synthetic PTP (Precision Time Protocol) training data.

This script generates realistic PTP synchronization data for training anomaly detection models.
It simulates:
- Normal PTP behavior (stable offset, low drift, consistent synchronization)
- Anomalies: clock drift, master changes, path delay increases, synchronization failures

The data is saved in multiple formats for flexibility:
- CSV: Human-readable format
- Parquet: Efficient columnar format for training

Usage:
    # Generate 5 hours of data from 10 PTP slaves with 10% anomaly rate
    python scripts/generate_ptp_data.py --slaves 10 --duration-hours 5 --anomaly-rate 0.1

    # Quick test data (5 minutes)
    python scripts/generate_ptp_data.py --slaves 3 --duration-minutes 5 --anomaly-rate 0.2

    # Large training dataset (24 hours, 50 slaves)
    python scripts/generate_ptp_data.py --slaves 50 --duration-hours 24 --anomaly-rate 0.05
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd


# PTP port state definitions (matching PTPAdapter)
class PTPPortState:
    INITIALIZING = 1
    FAULTY = 2
    DISABLED = 3
    LISTENING = 4
    PRE_MASTER = 5
    MASTER = 6
    PASSIVE = 7
    UNCALIBRATED = 8
    SLAVE = 9

    NAMES = {
        1: "INITIALIZING",
        2: "FAULTY",
        3: "DISABLED",
        4: "LISTENING",
        5: "PRE_MASTER",
        6: "MASTER",
        7: "PASSIVE",
        8: "UNCALIBRATED",
        9: "SLAVE",
    }


class PTPSlaveSimulator:
    """Simulator for a single PTP slave clock."""

    def __init__(
        self,
        slave_id: str,
        clock_id: str,
        master_clock_id: str = "00:00:00:00:00:00:00:01",
        sync_interval_ms: int = 125,
        delay_req_interval_ms: int = 1000,
        announce_interval_ms: int = 1000,
    ):
        """Initialize PTP slave simulator.

        Args:
            slave_id: Unique slave identifier
            clock_id: Slave clock ID (MAC address format)
            master_clock_id: Master clock ID
            sync_interval_ms: Sync message interval
            delay_req_interval_ms: Delay_Req message interval
            announce_interval_ms: Announce message interval
        """
        self.slave_id = slave_id
        self.clock_id = clock_id
        self.master_clock_id = master_clock_id
        self.sync_interval_ms = sync_interval_ms
        self.delay_req_interval_ms = delay_req_interval_ms
        self.announce_interval_ms = announce_interval_ms

        # Synchronization state
        self.port_state = PTPPortState.SLAVE
        self.offset_from_master_ns = random.uniform(-50, 50)  # Initial offset (nanoseconds)
        self.mean_path_delay_ns = random.uniform(1000, 5000)  # 1-5 microseconds
        self.clock_drift_ppb = random.uniform(-10, 10)  # ±10 ppb (parts per billion)
        self.steps_removed = 1  # Hops from grandmaster

        # Anomaly tracking
        self.anomaly_type = None
        self.anomaly_duration = 0

    def collect_metrics(self, timestamp: datetime, is_anomaly_period: bool) -> Dict[str, Any]:
        """Collect metrics for current timestamp.

        Args:
            timestamp: Current timestamp
            is_anomaly_period: Whether this is an anomaly period

        Returns:
            Dictionary of metrics
        """
        if is_anomaly_period:
            # Determine anomaly type (if not already in anomaly)
            if self.anomaly_type is None:
                anomaly_types = [
                    "clock_drift",
                    "master_change",
                    "path_delay_increase",
                    "sync_failure",
                ]
                self.anomaly_type = random.choice(anomaly_types)
                self.anomaly_duration = random.randint(5, 30)  # Anomaly lasts 5-30 cycles

            # Apply anomaly effects
            if self.anomaly_type == "clock_drift":
                # Clock drift anomaly: high drift rate, growing offset
                self.clock_drift_ppb = random.uniform(50, 200)
                self.offset_from_master_ns += random.uniform(50, 200)

            elif self.anomaly_type == "master_change":
                # Master change: port state transition, offset spike, steps_removed increase
                self.port_state = PTPPortState.LISTENING
                self.steps_removed += 1
                self.offset_from_master_ns = random.uniform(-500, 500)  # Large offset spike
                self.master_clock_id = f"00:00:00:00:00:00:00:{random.randint(2, 9):02d}"

            elif self.anomaly_type == "path_delay_increase":
                # Network delay increase: path delay grows
                self.mean_path_delay_ns = random.uniform(20000, 50000)  # 20-50 microseconds
                self.offset_from_master_ns += random.uniform(20, 100)

            elif self.anomaly_type == "sync_failure":
                # Synchronization failure: offset continues to grow
                self.port_state = PTPPortState.UNCALIBRATED
                self.offset_from_master_ns += random.uniform(100, 500)
                self.clock_drift_ppb = random.uniform(-50, 50)

            # Decrement anomaly duration
            self.anomaly_duration -= 1
            if self.anomaly_duration <= 0:
                # End anomaly, recover to normal state
                self.anomaly_type = None
                self.port_state = PTPPortState.SLAVE
                self.offset_from_master_ns = random.uniform(-50, 50)
                self.mean_path_delay_ns = random.uniform(1000, 5000)
                self.clock_drift_ppb = random.uniform(-10, 10)

        else:
            # Normal operation: small random walk
            self.port_state = PTPPortState.SLAVE

            # Offset random walk (stays within ±100ns)
            self.offset_from_master_ns += random.uniform(-20, 20)
            self.offset_from_master_ns = max(-100, min(100, self.offset_from_master_ns))

            # Path delay random walk (stays within 1-10μs)
            self.mean_path_delay_ns += random.uniform(-100, 100)
            self.mean_path_delay_ns = max(1000, min(10000, self.mean_path_delay_ns))

            # Clock drift random walk (stays within ±20ppb)
            self.clock_drift_ppb += random.uniform(-2, 2)
            self.clock_drift_ppb = max(-20, min(20, self.clock_drift_ppb))

        return {
            "timestamp": timestamp,
            "source_id": self.slave_id,
            "clock_id": self.clock_id,
            "master_clock_id": self.master_clock_id,
            "port_state": self.port_state,
            "port_state_name": PTPPortState.NAMES[self.port_state],
            "offset_from_master_ns": self.offset_from_master_ns,
            "mean_path_delay_ns": self.mean_path_delay_ns,
            "clock_drift_ppb": self.clock_drift_ppb,
            "sync_interval_ms": self.sync_interval_ms,
            "delay_req_interval_ms": self.delay_req_interval_ms,
            "announce_interval_ms": self.announce_interval_ms,
            "steps_removed": self.steps_removed,
            "is_anomaly": is_anomaly_period,
            "anomaly_type": self.anomaly_type if is_anomaly_period else None,
        }


def generate_ptp_data(
    n_slaves: int,
    duration_seconds: int,
    collection_interval: int,
    anomaly_rate: float,
    output_dir: Path,
) -> None:
    """Generate PTP training data.

    Args:
        n_slaves: Number of PTP slaves to simulate
        duration_seconds: Duration of data collection in seconds
        collection_interval: Collection interval in seconds
        anomaly_rate: Fraction of time with anomalies (0.0-1.0)
        output_dir: Output directory for data files
    """
    print(f"Generating PTP data...")
    print(f"  Slaves: {n_slaves}")
    print(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Anomaly rate: {anomaly_rate * 100:.1f}%")

    # Create simulators for each slave
    simulators = []
    for i in range(n_slaves):
        slave_id = f"ptp-slave-{i+1}"
        clock_id = f"00:00:00:00:00:{i // 256:02x}:{(i % 256):02x}:{random.randint(0, 255):02x}"

        simulator = PTPSlaveSimulator(
            slave_id=slave_id,
            clock_id=clock_id,
            master_clock_id="00:00:00:00:00:00:00:01",
            sync_interval_ms=random.choice([125, 250]),  # 8 or 4 packets/sec
            delay_req_interval_ms=random.choice([1000, 2000]),  # 1 or 0.5 packets/sec
            announce_interval_ms=random.choice([1000, 2000]),  # 1 or 0.5 packets/sec
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

        # Collect metrics from all slaves
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

    # Compute absolute values for easy filtering
    df["abs_offset_ns"] = df["offset_from_master_ns"].abs()
    df["abs_drift_ppb"] = df["clock_drift_ppb"].abs()

    # Statistics
    n_total = len(df)
    n_anomalies = df["is_anomaly"].sum()
    anomaly_pct = n_anomalies / n_total * 100

    print(f"\nGenerated {n_total:,} records")
    print(f"  Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)")
    print(f"  Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)")

    # Anomaly type breakdown
    if n_anomalies > 0:
        anomaly_types = df[df["is_anomaly"]]["anomaly_type"].value_counts()
        print(f"\n  Anomaly breakdown:")
        for anomaly_type, count in anomaly_types.items():
            print(f"    {anomaly_type}: {count} ({count / n_anomalies * 100:.1f}%)")

    # Offset statistics
    print(f"\n  Offset statistics:")
    print(f"    Mean: {df['offset_from_master_ns'].mean():.2f} ns")
    print(f"    Std: {df['offset_from_master_ns'].std():.2f} ns")
    print(f"    Max absolute: {df['abs_offset_ns'].max():.2f} ns")

    # Path delay statistics
    print(f"\n  Path delay statistics:")
    print(f"    Mean: {df['mean_path_delay_ns'].mean():.2f} ns ({df['mean_path_delay_ns'].mean() / 1000:.2f} μs)")
    print(f"    Std: {df['mean_path_delay_ns'].std():.2f} ns")
    print(f"    Max: {df['mean_path_delay_ns'].max():.2f} ns ({df['mean_path_delay_ns'].max() / 1000:.2f} μs)")

    # Clock drift statistics
    print(f"\n  Clock drift statistics:")
    print(f"    Mean: {df['clock_drift_ppb'].mean():.2f} ppb")
    print(f"    Std: {df['clock_drift_ppb'].std():.2f} ppb")
    print(f"    Max absolute: {df['abs_drift_ppb'].max():.2f} ppb")

    # Save to files
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV (human-readable)
    csv_path = output_dir / f"ptp_data_{timestamp_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Parquet (efficient for training)
    parquet_path = output_dir / f"ptp_data_{timestamp_str}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")

    # Save summary statistics
    summary_path = output_dir / f"ptp_data_{timestamp_str}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PTP Training Data Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Slaves: {n_slaves}\n")
        f.write(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m\n")
        f.write(f"  Collection interval: {collection_interval}s\n")
        f.write(f"  Anomaly rate: {anomaly_rate * 100:.1f}%\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Total records: {n_total:,}\n")
        f.write(f"  Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)\n")
        f.write(f"  Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)\n\n")

        if n_anomalies > 0:
            f.write(f"Anomaly breakdown:\n")
            for anomaly_type, count in anomaly_types.items():
                f.write(f"  {anomaly_type}: {count} ({count / n_anomalies * 100:.1f}%)\n")
            f.write("\n")

        f.write(f"Offset statistics:\n")
        f.write(f"  Mean: {df['offset_from_master_ns'].mean():.2f} ns\n")
        f.write(f"  Std: {df['offset_from_master_ns'].std():.2f} ns\n")
        f.write(f"  Max absolute: {df['abs_offset_ns'].max():.2f} ns\n\n")

        f.write(f"Path delay statistics:\n")
        f.write(f"  Mean: {df['mean_path_delay_ns'].mean():.2f} ns\n")
        f.write(f"  Std: {df['mean_path_delay_ns'].std():.2f} ns\n")
        f.write(f"  Max: {df['mean_path_delay_ns'].max():.2f} ns\n\n")

        f.write(f"Clock drift statistics:\n")
        f.write(f"  Mean: {df['clock_drift_ppb'].mean():.2f} ppb\n")
        f.write(f"  Std: {df['clock_drift_ppb'].std():.2f} ppb\n")
        f.write(f"  Max absolute: {df['abs_drift_ppb'].max():.2f} ppb\n\n")

        f.write(f"Files:\n")
        f.write(f"  CSV: {csv_path.name}\n")
        f.write(f"  Parquet: {parquet_path.name}\n")

    print(f"Saved summary: {summary_path}")

    print("\n✅ Data generation completed successfully!")
    print(f"\nUse this data for training:")
    print(f"  python scripts/train_tcn_ptp.py --data {parquet_path}")
    print(f"  python scripts/train_lstm_ptp.py --data {parquet_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic PTP training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--slaves",
        type=int,
        default=10,
        help="Number of PTP slaves to simulate (default: 10)",
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
        default=5,
        help="Collection interval in seconds (default: 5)",
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
        default=Path("data/training/ptp"),
        help="Output directory (default: data/training/ptp)",
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
    if args.slaves <= 0:
        parser.error("--slaves must be positive")
    if duration_seconds <= 0:
        parser.error("Duration must be positive")
    if args.collection_interval <= 0:
        parser.error("--collection-interval must be positive")
    if not (0.0 <= args.anomaly_rate <= 1.0):
        parser.error("--anomaly-rate must be between 0.0 and 1.0")

    # Generate data
    generate_ptp_data(
        n_slaves=args.slaves,
        duration_seconds=duration_seconds,
        collection_interval=args.collection_interval,
        anomaly_rate=args.anomaly_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
