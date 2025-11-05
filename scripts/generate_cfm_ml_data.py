#!/usr/bin/env python3
"""Generate synthetic CFM (Connectivity Fault Management) ML training/validation data.

This script generates realistic CFM-Lite metrics data for training and validating
anomaly detection models. It creates 3 separate datasets:

1. train/ - Normal behavior only (for training models)
2. val_normal/ - Normal behavior (for validation)
3. val_anomaly/ - Anomalous behavior (for validation)

CFM-Lite metrics simulated:
- UDP Echo RTT (ms)
- eCPRI One-way Delay (us)
- LBM RTT (ms)
- CCM Miss Count (counter)

Anomaly scenarios:
- Latency spike: Sudden RTT/delay increase
- Packet loss: CCM miss count increase
- Network instability: High jitter, unstable delays
- Sustained degradation: Gradual performance decline

Usage:
    # Generate 1 hour training + validation data from 10 endpoints
    python scripts/generate_cfm_ml_data.py --endpoints 10 --duration-hours 1 --output data/cfm

    # Quick test (5 minutes)
    python scripts/generate_cfm_ml_data.py --endpoints 3 --duration-minutes 5 --anomaly-rate 0.3

    # Large dataset (24 hours, 50 endpoints)
    python scripts/generate_cfm_ml_data.py --endpoints 50 --duration-hours 24
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd


class CFMEndpointSimulator:
    """Simulator for a single CFM-monitored endpoint."""

    def __init__(
        self,
        endpoint_id: str,
        endpoint_ip: str,
        base_udp_rtt: float = 5.0,
        base_ecpri_delay: float = 100.0,
        base_lbm_rtt: float = 10.0,
    ):
        """Initialize CFM endpoint simulator.

        Args:
            endpoint_id: Unique endpoint identifier
            endpoint_ip: Endpoint IP address
            base_udp_rtt: Baseline UDP Echo RTT in ms
            base_ecpri_delay: Baseline eCPRI delay in us
            base_lbm_rtt: Baseline LBM RTT in ms
        """
        self.endpoint_id = endpoint_id
        self.endpoint_ip = endpoint_ip
        self.base_udp_rtt = base_udp_rtt
        self.base_ecpri_delay = base_ecpri_delay
        self.base_lbm_rtt = base_lbm_rtt

        # CCM state
        self.ccm_miss_count = 0
        self.ccm_consecutive_misses = 0

        # Anomaly state
        self.anomaly_type = None
        self.anomaly_start_time = None
        self.anomaly_duration = 0

    def collect_metrics(self, timestamp: datetime, is_anomaly_period: bool) -> Dict[str, Any]:
        """Collect metrics for current timestamp.

        Args:
            timestamp: Current timestamp
            is_anomaly_period: Whether this is an anomaly period

        Returns:
            Dictionary of metrics
        """
        # Determine anomaly scenario
        if is_anomaly_period and self.anomaly_type is None:
            # Start new anomaly
            self.anomaly_type = random.choice([
                "latency_spike",
                "packet_loss",
                "network_instability",
                "sustained_degradation",
            ])
            self.anomaly_start_time = timestamp
            self.anomaly_duration = random.randint(10, 60)  # 10-60 collection cycles

        if self.anomaly_type is not None:
            self.anomaly_duration -= 1
            if self.anomaly_duration <= 0:
                # End anomaly
                self.anomaly_type = None
                self.ccm_consecutive_misses = 0

        # Generate metrics based on current state
        if self.anomaly_type == "latency_spike":
            # Sudden latency increase
            udp_rtt = self.base_udp_rtt * random.uniform(3.0, 10.0)  # 3-10x increase
            ecpri_delay = self.base_ecpri_delay * random.uniform(3.0, 10.0)
            lbm_rtt = self.base_lbm_rtt * random.uniform(3.0, 10.0)
            ccm_miss = 1 if random.random() < 0.3 else 0  # 30% chance

        elif self.anomaly_type == "packet_loss":
            # High packet loss (CCM misses)
            udp_rtt = self.base_udp_rtt * random.uniform(1.5, 3.0)  # Moderate RTT increase
            ecpri_delay = self.base_ecpri_delay * random.uniform(1.5, 3.0)
            lbm_rtt = self.base_lbm_rtt * random.uniform(1.5, 3.0)
            ccm_miss = 1 if random.random() < 0.7 else 0  # 70% miss rate

        elif self.anomaly_type == "network_instability":
            # High jitter, unstable delays
            jitter = random.uniform(0.5, 3.0)
            udp_rtt = self.base_udp_rtt * jitter
            ecpri_delay = self.base_ecpri_delay * jitter
            lbm_rtt = self.base_lbm_rtt * jitter
            ccm_miss = 1 if random.random() < 0.2 else 0  # 20% miss rate

        elif self.anomaly_type == "sustained_degradation":
            # Gradual degradation over time
            degradation_factor = 1.0 + (self.anomaly_start_time is not None and
                                       (timestamp - self.anomaly_start_time).total_seconds() / 60.0 * 0.1)
            udp_rtt = self.base_udp_rtt * degradation_factor
            ecpri_delay = self.base_ecpri_delay * degradation_factor
            lbm_rtt = self.base_lbm_rtt * degradation_factor
            ccm_miss = 1 if random.random() < 0.15 else 0  # 15% miss rate

        else:
            # Normal behavior
            # Small random variations around baseline
            udp_rtt = self.base_udp_rtt * random.uniform(0.9, 1.1)
            ecpri_delay = self.base_ecpri_delay * random.uniform(0.9, 1.1)
            lbm_rtt = self.base_lbm_rtt * random.uniform(0.9, 1.1)
            ccm_miss = 1 if random.random() < 0.01 else 0  # 1% rare miss

        # Update CCM miss count
        if ccm_miss:
            self.ccm_miss_count += 1
            self.ccm_consecutive_misses += 1
        else:
            self.ccm_consecutive_misses = 0

        return {
            "timestamp": timestamp,
            "source_id": self.endpoint_id,
            "endpoint_ip": self.endpoint_ip,
            "udp_echo_rtt_ms": udp_rtt,
            "ecpri_delay_us": ecpri_delay,
            "lbm_rtt_ms": lbm_rtt,
            "ccm_miss_count": self.ccm_miss_count,
            "ccm_consecutive_misses": self.ccm_consecutive_misses,
            "is_anomaly": is_anomaly_period,
            "anomaly_type": self.anomaly_type if is_anomaly_period else "normal",
        }


def generate_dataset(
    n_endpoints: int,
    duration_seconds: int,
    collection_interval: int,
    anomaly_rate: float,
    dataset_name: str,
) -> pd.DataFrame:
    """Generate a single dataset.

    Args:
        n_endpoints: Number of endpoints to simulate
        duration_seconds: Duration of data collection in seconds
        collection_interval: Collection interval in seconds
        anomaly_rate: Fraction of time with anomalies (0.0-1.0)
        dataset_name: Name of dataset (for logging)

    Returns:
        DataFrame with collected metrics
    """
    print(f"\nGenerating {dataset_name} dataset...")
    print(f"  Endpoints: {n_endpoints}")
    print(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Anomaly rate: {anomaly_rate * 100:.1f}%")

    # Create simulators for each endpoint
    simulators = []
    for i in range(n_endpoints):
        endpoint_id = f"cfm-endpoint-{i+1}"
        endpoint_ip = f"192.168.{i // 254}.{(i % 254) + 1}"

        simulator = CFMEndpointSimulator(
            endpoint_id=endpoint_id,
            endpoint_ip=endpoint_ip,
            base_udp_rtt=random.uniform(3.0, 8.0),  # Varied baselines
            base_ecpri_delay=random.uniform(50.0, 200.0),
            base_lbm_rtt=random.uniform(8.0, 15.0),
        )
        simulators.append(simulator)

    # Generate time series
    start_time = datetime.utcnow()
    n_collections = duration_seconds // collection_interval

    all_metrics = []

    for cycle in range(n_collections):
        timestamp = start_time + timedelta(seconds=cycle * collection_interval)

        # Determine if this is an anomaly period
        is_anomaly_period = random.random() < anomaly_rate

        # Collect metrics from all endpoints
        for simulator in simulators:
            metrics = simulator.collect_metrics(timestamp, is_anomaly_period)
            all_metrics.append(metrics)

        # Progress indicator
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
    anomaly_pct = n_anomalies / n_total * 100 if n_total > 0 else 0.0

    print(f"  Generated {n_total:,} records")
    print(f"    Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)")
    print(f"    Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)")

    # Anomaly type distribution
    if n_anomalies > 0:
        anomaly_types = df[df["is_anomaly"]]["anomaly_type"].value_counts()
        print(f"  Anomaly types:")
        for anom_type, count in anomaly_types.items():
            print(f"    {anom_type}: {count} ({count/n_anomalies*100:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate CFM ML training/validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--endpoints",
        type=int,
        default=10,
        help="Number of endpoints to simulate (default: 10)",
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
        default=0.2,
        help="Anomaly rate for validation anomaly dataset, 0.0-1.0 (default: 0.2 = 20%%)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cfm"),
        help="Output directory (default: data/cfm)",
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
    if args.endpoints <= 0:
        parser.error("--endpoints must be positive")
    if duration_seconds <= 0:
        parser.error("Duration must be positive")
    if args.collection_interval <= 0:
        parser.error("--collection-interval must be positive")
    if not (0.0 <= args.anomaly_rate <= 1.0):
        parser.error("--anomaly-rate must be between 0.0 and 1.0")

    print("="*70)
    print("CFM ML Data Generation")
    print("="*70)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Generate training dataset (normal only)
    print("\n[1/3] Training Dataset (Normal Only)")
    print("-"*70)
    train_df = generate_dataset(
        n_endpoints=args.endpoints,
        duration_seconds=duration_seconds,
        collection_interval=args.collection_interval,
        anomaly_rate=0.0,  # No anomalies in training data
        dataset_name="train",
    )

    train_dir = args.output / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    train_csv = train_dir / f"cfm_train_{timestamp_str}.csv"
    train_parquet = train_dir / f"cfm_train_{timestamp_str}.parquet"

    train_df.to_csv(train_csv, index=False)
    train_df.to_parquet(train_parquet, index=False)

    print(f"  ✅ Saved: {train_csv}")
    print(f"  ✅ Saved: {train_parquet}")

    # 2. Generate validation normal dataset
    print("\n[2/3] Validation Normal Dataset")
    print("-"*70)
    val_normal_df = generate_dataset(
        n_endpoints=args.endpoints,
        duration_seconds=duration_seconds // 2,  # Half duration for validation
        collection_interval=args.collection_interval,
        anomaly_rate=0.0,  # Normal only
        dataset_name="val_normal",
    )

    val_normal_dir = args.output / "val_normal"
    val_normal_dir.mkdir(parents=True, exist_ok=True)

    val_normal_csv = val_normal_dir / f"cfm_val_normal_{timestamp_str}.csv"
    val_normal_parquet = val_normal_dir / f"cfm_val_normal_{timestamp_str}.parquet"

    val_normal_df.to_csv(val_normal_csv, index=False)
    val_normal_df.to_parquet(val_normal_parquet, index=False)

    print(f"  ✅ Saved: {val_normal_csv}")
    print(f"  ✅ Saved: {val_normal_parquet}")

    # 3. Generate validation anomaly dataset
    print("\n[3/3] Validation Anomaly Dataset")
    print("-"*70)
    val_anomaly_df = generate_dataset(
        n_endpoints=args.endpoints,
        duration_seconds=duration_seconds // 2,  # Half duration for validation
        collection_interval=args.collection_interval,
        anomaly_rate=args.anomaly_rate,  # User-specified anomaly rate
        dataset_name="val_anomaly",
    )

    val_anomaly_dir = args.output / "val_anomaly"
    val_anomaly_dir.mkdir(parents=True, exist_ok=True)

    val_anomaly_csv = val_anomaly_dir / f"cfm_val_anomaly_{timestamp_str}.csv"
    val_anomaly_parquet = val_anomaly_dir / f"cfm_val_anomaly_{timestamp_str}.parquet"

    val_anomaly_df.to_csv(val_anomaly_csv, index=False)
    val_anomaly_df.to_parquet(val_anomaly_parquet, index=False)

    print(f"  ✅ Saved: {val_anomaly_csv}")
    print(f"  ✅ Saved: {val_anomaly_parquet}")

    # Save summary
    summary_path = args.output / f"cfm_data_{timestamp_str}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("CFM ML Data Generation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Endpoints: {args.endpoints}\n")
        f.write(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m\n")
        f.write(f"  Collection interval: {args.collection_interval}s\n")
        f.write(f"  Validation anomaly rate: {args.anomaly_rate * 100:.1f}%\n\n")
        f.write(f"Datasets:\n")
        f.write(f"  1. Training (Normal Only):\n")
        f.write(f"     Records: {len(train_df):,}\n")
        f.write(f"     Files: {train_csv.name}, {train_parquet.name}\n\n")
        f.write(f"  2. Validation Normal:\n")
        f.write(f"     Records: {len(val_normal_df):,}\n")
        f.write(f"     Files: {val_normal_csv.name}, {val_normal_parquet.name}\n\n")
        f.write(f"  3. Validation Anomaly:\n")
        f.write(f"     Records: {len(val_anomaly_df):,}\n")
        f.write(f"     Anomalies: {val_anomaly_df['is_anomaly'].sum():,} ({val_anomaly_df['is_anomaly'].sum()/len(val_anomaly_df)*100:.1f}%)\n")
        f.write(f"     Files: {val_anomaly_csv.name}, {val_anomaly_parquet.name}\n\n")
        f.write(f"CFM Metrics:\n")
        f.write(f"  - udp_echo_rtt_ms: UDP Echo RTT (ms)\n")
        f.write(f"  - ecpri_delay_us: eCPRI One-way Delay (us)\n")
        f.write(f"  - lbm_rtt_ms: LBM RTT (ms)\n")
        f.write(f"  - ccm_miss_count: CCM Miss Count\n\n")
        f.write(f"Next Steps:\n")
        f.write(f"  1. Train models:\n")
        f.write(f"     python scripts/train_cfm_tcn.py --train-data {train_parquet}\n\n")
        f.write(f"  2. Run inference:\n")
        f.write(f"     python scripts/infer_cfm.py --val-normal {val_normal_parquet} --val-anomaly {val_anomaly_parquet}\n\n")
        f.write(f"  3. Generate report:\n")
        f.write(f"     python scripts/report_cfm.py --predictions results/cfm/predictions.csv\n")

    print(f"\n{'='*70}")
    print(f"✅ Data generation completed successfully!")
    print(f"{'='*70}")
    print(f"Summary: {summary_path}")
    print(f"\nNext steps:")
    print(f"  1. Train TCN model:")
    print(f"     python scripts/train_cfm_tcn.py --metric-type udp_echo --train-data {train_parquet}")
    print(f"  2. Run inference:")
    print(f"     python scripts/infer_cfm.py --val-normal {val_normal_parquet} --val-anomaly {val_anomaly_parquet}")
    print(f"  3. Generate report:")
    print(f"     python scripts/report_cfm.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
