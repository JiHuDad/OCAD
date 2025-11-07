#!/usr/bin/env python3
"""Generate synthetic BFD (Bidirectional Forwarding Detection) training data.

This script generates realistic BFD session data for training anomaly detection models.
It simulates:
- Normal BFD behavior (stable sessions, fast detection)
- Anomalies: flapping (rapid Up/Down transitions), slow detection, sustained failures

The data is saved in multiple formats for flexibility:
- CSV: Human-readable format
- Parquet: Efficient columnar format for training

Usage:
    # Generate 5 hours of data from 10 BFD sessions with 10% anomaly rate
    python scripts/generate_bfd_data.py --sessions 10 --duration-hours 5 --anomaly-rate 0.1

    # Quick test data (5 minutes)
    python scripts/generate_bfd_data.py --sessions 3 --duration-minutes 5 --anomaly-rate 0.2

    # Large training dataset (24 hours, 50 sessions)
    python scripts/generate_bfd_data.py --sessions 50 --duration-hours 24 --anomaly-rate 0.05
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd


# BFD state definitions (matching BFDAdapter)
class BFDState:
    ADMIN_DOWN = 0
    DOWN = 1
    INIT = 2
    UP = 3

    NAMES = {
        0: "ADMIN_DOWN",
        1: "DOWN",
        2: "INIT",
        3: "UP",
    }


# BFD diagnostic codes (matching BFDAdapter)
class BFDDiagnostic:
    NO_DIAGNOSTIC = 0
    CONTROL_DETECTION_TIME_EXPIRED = 1
    ECHO_FUNCTION_FAILED = 2
    NEIGHBOR_SIGNALED_SESSION_DOWN = 3
    FORWARDING_PLANE_RESET = 4
    PATH_DOWN = 5
    CONCATENATED_PATH_DOWN = 6
    ADMINISTRATIVELY_DOWN = 7
    REVERSE_CONCATENATED_PATH_DOWN = 8

    NAMES = {
        0: "NO_DIAGNOSTIC",
        1: "CONTROL_DETECTION_TIME_EXPIRED",
        2: "ECHO_FUNCTION_FAILED",
        3: "NEIGHBOR_SIGNALED_SESSION_DOWN",
        4: "FORWARDING_PLANE_RESET",
        5: "PATH_DOWN",
        6: "CONCATENATED_PATH_DOWN",
        7: "ADMINISTRATIVELY_DOWN",
        8: "REVERSE_CONCATENATED_PATH_DOWN",
    }


class BFDSessionSimulator:
    """Simulator for a single BFD session."""

    def __init__(
        self,
        session_id: str,
        local_ip: str,
        remote_ip: str,
        interval_ms: int = 50,
        multiplier: int = 3,
    ):
        """Initialize BFD session simulator.

        Args:
            session_id: Unique session identifier
            local_ip: Local IP address
            remote_ip: Remote IP address
            interval_ms: Echo interval in milliseconds
            multiplier: Detection multiplier
        """
        self.session_id = session_id
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.interval_ms = interval_ms
        self.multiplier = multiplier

        # Session state
        self.local_state = BFDState.UP
        self.remote_state = BFDState.UP
        self.flap_count = 0

        # Anomaly flags
        self.is_flapping = False
        self.flap_start_time = None
        self.flap_duration = 0

    def collect_metrics(self, timestamp: datetime, is_anomaly_period: bool) -> Dict[str, Any]:
        """Collect metrics for current timestamp.

        Args:
            timestamp: Current timestamp
            is_anomaly_period: Whether this is an anomaly period

        Returns:
            Dictionary of metrics
        """
        # Determine if state change occurs
        state_changed = False

        if is_anomaly_period:
            # Anomaly scenarios
            if not self.is_flapping and random.random() < 0.3:
                # Start flapping
                self.is_flapping = True
                self.flap_start_time = timestamp
                self.flap_duration = random.randint(5, 30)  # Flap for 5-30 collection cycles

            if self.is_flapping:
                # Flapping: 50% chance of state change per cycle
                if random.random() < 0.5:
                    state_changed = True
                    if self.local_state == BFDState.UP:
                        self.local_state = BFDState.DOWN
                        diagnostic = BFDDiagnostic.CONTROL_DETECTION_TIME_EXPIRED
                    else:
                        self.local_state = BFDState.UP
                        diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
                    self.flap_count += 1

                # End flapping after duration
                self.flap_duration -= 1
                if self.flap_duration <= 0:
                    self.is_flapping = False
                    self.local_state = BFDState.UP  # Return to UP
            else:
                # Non-flapping anomaly: sustained DOWN state
                if random.random() < 0.1:
                    state_changed = True
                    self.local_state = BFDState.DOWN
                    diagnostic = BFDDiagnostic.PATH_DOWN
                    self.flap_count += 1
        else:
            # Normal behavior: maintain stable UP state
            # Auto-recover to UP if in abnormal state during normal periods
            if self.local_state != BFDState.UP:
                # Recovery to normal UP state
                self.local_state = BFDState.UP
                state_changed = True
                diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
            else:
                # Already in UP state, no change
                state_changed = False
                diagnostic = BFDDiagnostic.NO_DIAGNOSTIC

        if not state_changed:
            diagnostic = BFDDiagnostic.NO_DIAGNOSTIC

        # Detection time (normal: 15-50ms, slow during failures: 100-300ms)
        if self.local_state == BFDState.DOWN or is_anomaly_period:
            detection_time_ms = random.uniform(100, 300)  # Slow
        else:
            detection_time_ms = random.uniform(15, 50)  # Normal

        # Echo interval (usually stable, but can drift during anomalies)
        if is_anomaly_period:
            echo_interval_ms = self.interval_ms + random.uniform(-10, 10)
        else:
            echo_interval_ms = float(self.interval_ms)

        # Determine actual anomaly status
        # Anomaly if: period flag OR abnormal state OR excessive flapping
        is_actual_anomaly = (
            is_anomaly_period  # Explicitly marked anomaly period
            or self.local_state != BFDState.UP  # Not in UP state
            or self.flap_count > 5  # Excessive flapping
        )

        return {
            "timestamp": timestamp,
            "source_id": self.session_id,
            "local_ip": self.local_ip,
            "remote_ip": self.remote_ip,
            "local_state": self.local_state,
            "local_state_name": BFDState.NAMES[self.local_state],
            "remote_state": self.remote_state,
            "remote_state_name": BFDState.NAMES[self.remote_state],
            "detection_time_ms": detection_time_ms,
            "echo_interval_ms": echo_interval_ms,
            "diagnostic_code": diagnostic,
            "diagnostic_name": BFDDiagnostic.NAMES[diagnostic],
            "multiplier": self.multiplier,
            "flap_count": self.flap_count,
            "is_anomaly": is_actual_anomaly,
        }


def generate_bfd_data(
    n_sessions: int,
    duration_seconds: int,
    collection_interval: int,
    anomaly_rate: float,
    output_dir: Path,
) -> None:
    """Generate BFD training data.

    Args:
        n_sessions: Number of BFD sessions to simulate
        duration_seconds: Duration of data collection in seconds
        collection_interval: Collection interval in seconds
        anomaly_rate: Fraction of time with anomalies (0.0-1.0)
        output_dir: Output directory for data files
    """
    print(f"Generating BFD data...")
    print(f"  Sessions: {n_sessions}")
    print(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Anomaly rate: {anomaly_rate * 100:.1f}%")

    # Create simulators for each session
    simulators = []
    for i in range(n_sessions):
        session_id = f"bfd-session-{i+1}"
        local_ip = f"192.168.{i // 254}.{(i % 254) + 1}"
        remote_ip = f"192.168.{i // 254}.{((i + 1) % 254) + 1}"

        simulator = BFDSessionSimulator(
            session_id=session_id,
            local_ip=local_ip,
            remote_ip=remote_ip,
            interval_ms=random.choice([50, 100, 200]),  # Varied intervals
            multiplier=random.choice([3, 5, 10]),  # Varied multipliers
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

        # Collect metrics from all sessions
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
    n_flaps = df.groupby("source_id")["flap_count"].max().sum()
    print(f"  Total flaps: {n_flaps}")

    # Save to files
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV (human-readable)
    csv_path = output_dir / f"bfd_data_{timestamp_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Parquet (efficient for training)
    parquet_path = output_dir / f"bfd_data_{timestamp_str}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")

    # Save summary statistics
    summary_path = output_dir / f"bfd_data_{timestamp_str}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("BFD Training Data Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Sessions: {n_sessions}\n")
        f.write(f"  Duration: {duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m\n")
        f.write(f"  Collection interval: {collection_interval}s\n")
        f.write(f"  Anomaly rate: {anomaly_rate * 100:.1f}%\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Total records: {n_total:,}\n")
        f.write(f"  Normal: {n_total - n_anomalies:,} ({100 - anomaly_pct:.1f}%)\n")
        f.write(f"  Anomalies: {n_anomalies:,} ({anomaly_pct:.1f}%)\n")
        f.write(f"  Total flaps: {n_flaps}\n\n")
        f.write(f"Files:\n")
        f.write(f"  CSV: {csv_path.name}\n")
        f.write(f"  Parquet: {parquet_path.name}\n")

    print(f"Saved summary: {summary_path}")

    print("\n✅ Data generation completed successfully!")
    print(f"\nUse this data for training:")
    print(f"  python scripts/train_lstm_bfd.py --data {parquet_path}")
    print(f"  python scripts/train_hmm_bfd.py --data {parquet_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic BFD training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Number of BFD sessions to simulate (default: 10)",
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
        default=Path("data/training/bfd"),
        help="Output directory (default: data/training/bfd)",
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
    if args.sessions <= 0:
        parser.error("--sessions must be positive")
    if duration_seconds <= 0:
        parser.error("Duration must be positive")
    if args.collection_interval <= 0:
        parser.error("--collection-interval must be positive")
    if not (0.0 <= args.anomaly_rate <= 1.0):
        parser.error("--anomaly-rate must be between 0.0 and 1.0")

    # Generate data
    generate_bfd_data(
        n_sessions=args.sessions,
        duration_seconds=duration_seconds,
        collection_interval=args.collection_interval,
        anomaly_rate=args.anomaly_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
