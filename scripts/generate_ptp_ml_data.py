#!/usr/bin/env python3
"""Generate PTP ML training and validation datasets.

This script generates three separate datasets for machine learning pipeline:
1. Train dataset: Normal data only (anomaly_rate=0.0) - Large volume
2. Validation normal: Normal data only (anomaly_rate=0.0) - Medium volume
3. Validation anomaly: Anomaly-heavy data (anomaly_rate=0.8-1.0) - Medium volume

The data is saved in Parquet format for efficient ML training.

Usage:
    # Generate default datasets
    python scripts/generate_ptp_ml_data.py --output data/ptp

    # Custom configuration
    python scripts/generate_ptp_ml_data.py --slaves 20 --train-hours 5 --val-hours 1 --output data/ptp
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd


# PTP port state definitions
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
        """Initialize PTP slave simulator."""
        self.slave_id = slave_id
        self.clock_id = clock_id
        self.master_clock_id = master_clock_id
        self.sync_interval_ms = sync_interval_ms
        self.delay_req_interval_ms = delay_req_interval_ms
        self.announce_interval_ms = announce_interval_ms

        # Synchronization state
        self.port_state = PTPPortState.SLAVE
        self.offset_from_master_ns = random.uniform(-50, 50)
        self.mean_path_delay_ns = random.uniform(1000, 5000)
        self.clock_drift_ppb = random.uniform(-10, 10)
        self.steps_removed = 1

        # Anomaly tracking
        self.anomaly_type = None
        self.anomaly_duration = 0

    def collect_metrics(self, timestamp: datetime, is_anomaly_period: bool) -> Dict[str, Any]:
        """Collect metrics for current timestamp."""
        if is_anomaly_period:
            # Determine anomaly type
            if self.anomaly_type is None:
                anomaly_types = [
                    "clock_drift",
                    "master_change",
                    "path_delay_increase",
                    "sync_failure",
                ]
                self.anomaly_type = random.choice(anomaly_types)
                self.anomaly_duration = random.randint(5, 30)

            # Apply anomaly effects
            if self.anomaly_type == "clock_drift":
                self.clock_drift_ppb = random.uniform(50, 200)
                self.offset_from_master_ns += random.uniform(50, 200)

            elif self.anomaly_type == "master_change":
                self.port_state = PTPPortState.LISTENING
                self.steps_removed += 1
                self.offset_from_master_ns = random.uniform(-500, 500)
                self.master_clock_id = f"00:00:00:00:00:00:00:{random.randint(2, 9):02d}"

            elif self.anomaly_type == "path_delay_increase":
                self.mean_path_delay_ns = random.uniform(20000, 50000)
                self.offset_from_master_ns += random.uniform(20, 100)

            elif self.anomaly_type == "sync_failure":
                self.port_state = PTPPortState.UNCALIBRATED
                self.offset_from_master_ns += random.uniform(100, 500)
                self.clock_drift_ppb = random.uniform(-50, 50)

            # Decrement anomaly duration
            self.anomaly_duration -= 1
            if self.anomaly_duration <= 0:
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


def generate_dataset(
    dataset_type: str,
    n_slaves: int,
    duration_seconds: int,
    collection_interval: int,
    anomaly_rate: float,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate a single dataset.

    Args:
        dataset_type: "train", "val_normal", or "val_anomaly"
        n_slaves: Number of PTP slaves to simulate
        duration_seconds: Duration of data collection in seconds
        collection_interval: Collection interval in seconds
        anomaly_rate: Fraction of time with anomalies (0.0-1.0)
        output_dir: Output directory for data files

    Returns:
        DataFrame containing the generated data
    """
    print(f"\nGenerating {dataset_type} dataset...")
    print(f"  Slaves: {n_slaves}")
    print(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Anomaly rate: {anomaly_rate * 100:.1f}%")

    # Create simulators
    simulators = []
    for i in range(n_slaves):
        slave_id = f"ptp-slave-{i+1}"
        clock_id = f"00:00:00:00:00:{i // 256:02x}:{(i % 256):02x}:{random.randint(0, 255):02x}"

        simulator = PTPSlaveSimulator(
            slave_id=slave_id,
            clock_id=clock_id,
            master_clock_id="00:00:00:00:00:00:00:01",
            sync_interval_ms=random.choice([125, 250]),
            delay_req_interval_ms=random.choice([1000, 2000]),
            announce_interval_ms=random.choice([1000, 2000]),
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
    df["abs_offset_ns"] = df["offset_from_master_ns"].abs()
    df["abs_drift_ppb"] = df["clock_drift_ppb"].abs()

    # Statistics
    n_total = len(df)
    n_anomalies = df["is_anomaly"].sum()
    anomaly_pct = n_anomalies / n_total * 100

    print(f"  Generated {n_total:,} records")
    print(f"    Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)")
    print(f"    Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)")

    if n_anomalies > 0:
        anomaly_types = df[df["is_anomaly"]]["anomaly_type"].value_counts()
        print(f"    Anomaly breakdown:")
        for anomaly_type, count in anomaly_types.items():
            print(f"      {anomaly_type}: {count} ({count / n_anomalies * 100:.1f}%)")

    # Save to Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = output_dir / f"ptp_{dataset_type}_{timestamp_str}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved: {parquet_path}")

    return df


def generate_all_datasets(
    n_slaves: int,
    train_hours: float,
    val_hours: float,
    collection_interval: int,
    output_base_dir: Path,
    seed: int = None,
) -> None:
    """Generate all three datasets for ML pipeline.

    Args:
        n_slaves: Number of PTP slaves to simulate
        train_hours: Duration of training dataset in hours
        val_hours: Duration of each validation dataset in hours
        collection_interval: Collection interval in seconds
        output_base_dir: Base output directory (will create subdirs)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Random seed: {seed}")

    print("=" * 70)
    print("PTP ML Data Generation Pipeline")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Slaves: {n_slaves}")
    print(f"  Training duration: {train_hours}h")
    print(f"  Validation duration: {val_hours}h each")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Output directory: {output_base_dir}")

    # 1. Generate training dataset (normal data only)
    train_dir = output_base_dir / "train"
    train_df = generate_dataset(
        dataset_type="train",
        n_slaves=n_slaves,
        duration_seconds=int(train_hours * 3600),
        collection_interval=collection_interval,
        anomaly_rate=0.0,  # No anomalies
        output_dir=train_dir,
    )

    # 2. Generate validation normal dataset
    val_normal_dir = output_base_dir / "val_normal"
    val_normal_df = generate_dataset(
        dataset_type="val_normal",
        n_slaves=n_slaves,
        duration_seconds=int(val_hours * 3600),
        collection_interval=collection_interval,
        anomaly_rate=0.0,  # No anomalies
        output_dir=val_normal_dir,
    )

    # 3. Generate validation anomaly dataset
    val_anomaly_dir = output_base_dir / "val_anomaly"
    val_anomaly_df = generate_dataset(
        dataset_type="val_anomaly",
        n_slaves=n_slaves,
        duration_seconds=int(val_hours * 3600),
        collection_interval=collection_interval,
        anomaly_rate=0.9,  # Heavy anomalies (90%)
        output_dir=val_anomaly_dir,
    )

    # Generate summary
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"\nDataset Summary:")
    print(f"  Train: {len(train_df):,} records (100% normal)")
    print(f"  Val Normal: {len(val_normal_df):,} records (100% normal)")
    print(f"  Val Anomaly: {len(val_anomaly_df):,} records ({val_anomaly_df['is_anomaly'].mean() * 100:.1f}% anomalies)")
    print(f"\nTotal records: {len(train_df) + len(val_normal_df) + len(val_anomaly_df):,}")

    print(f"\nNext steps:")
    print(f"  1. Train TCN model:")
    print(f"     python scripts/train_ptp_tcn.py --data {train_dir}")
    print(f"  2. Run inference:")
    print(f"     python scripts/infer_ptp.py --model models/ptp/tcn_v1.0.0.pth --data {val_normal_dir}")
    print(f"  3. Generate report:")
    print(f"     python scripts/report_ptp.py --predictions results/ptp/predictions.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PTP ML training and validation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--slaves",
        type=int,
        default=10,
        help="Number of PTP slaves to simulate (default: 10)",
    )

    parser.add_argument(
        "--train-hours",
        type=float,
        default=2.0,
        help="Duration of training dataset in hours (default: 2.0)",
    )

    parser.add_argument(
        "--val-hours",
        type=float,
        default=0.5,
        help="Duration of each validation dataset in hours (default: 0.5)",
    )

    parser.add_argument(
        "--collection-interval",
        type=int,
        default=5,
        help="Collection interval in seconds (default: 5)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ptp"),
        help="Output base directory (default: data/ptp)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate parameters
    if args.slaves <= 0:
        parser.error("--slaves must be positive")
    if args.train_hours <= 0:
        parser.error("--train-hours must be positive")
    if args.val_hours <= 0:
        parser.error("--val-hours must be positive")
    if args.collection_interval <= 0:
        parser.error("--collection-interval must be positive")

    # Generate datasets
    generate_all_datasets(
        n_slaves=args.slaves,
        train_hours=args.train_hours,
        val_hours=args.val_hours,
        collection_interval=args.collection_interval,
        output_base_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
