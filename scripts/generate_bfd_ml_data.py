#!/usr/bin/env python3
"""Generate BFD ML training and validation datasets.

This script generates 3 separate datasets:
1. train/ - Large training dataset (normal data only)
2. val_normal/ - Validation dataset (normal data only)
3. val_anomaly/ - Validation dataset (high anomaly rate)

Usage:
    # Generate all datasets
    python scripts/generate_bfd_ml_data.py --output data/bfd

    # Quick test (small datasets)
    python scripts/generate_bfd_ml_data.py --output data/bfd --quick

    # Custom sizes
    python scripts/generate_bfd_ml_data.py --output data/bfd \
        --train-hours 24 --val-hours 2 --sessions 50
"""

import argparse
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_bfd_data import generate_bfd_data


def generate_ml_datasets(
    output_dir: Path,
    n_sessions: int,
    train_hours: float,
    val_hours: float,
    collection_interval: int,
) -> None:
    """Generate all 3 ML datasets.

    Args:
        output_dir: Base output directory
        n_sessions: Number of BFD sessions
        train_hours: Training dataset duration in hours
        val_hours: Validation dataset duration in hours
        collection_interval: Collection interval in seconds
    """
    print("=" * 80)
    print("BFD ML Dataset Generation")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Sessions: {n_sessions}")
    print(f"Training duration: {train_hours}h")
    print(f"Validation duration: {val_hours}h each")
    print(f"Collection interval: {collection_interval}s")
    print()

    # 1. Training data (normal only)
    print("\n" + "=" * 80)
    print("1/3: Generating TRAINING dataset (normal data only)")
    print("=" * 80)
    train_dir = output_dir / "train"
    generate_bfd_data(
        n_sessions=n_sessions,
        duration_seconds=int(train_hours * 3600),
        collection_interval=collection_interval,
        anomaly_rate=0.0,  # No anomalies
        output_dir=train_dir,
    )

    # 2. Validation normal data
    print("\n" + "=" * 80)
    print("2/3: Generating VALIDATION NORMAL dataset")
    print("=" * 80)
    val_normal_dir = output_dir / "val_normal"
    generate_bfd_data(
        n_sessions=n_sessions,
        duration_seconds=int(val_hours * 3600),
        collection_interval=collection_interval,
        anomaly_rate=0.0,  # No anomalies
        output_dir=val_normal_dir,
    )

    # 3. Validation anomaly data
    print("\n" + "=" * 80)
    print("3/3: Generating VALIDATION ANOMALY dataset")
    print("=" * 80)
    val_anomaly_dir = output_dir / "val_anomaly"
    generate_bfd_data(
        n_sessions=n_sessions,
        duration_seconds=int(val_hours * 3600),
        collection_interval=collection_interval,
        anomaly_rate=0.9,  # High anomaly rate
        output_dir=val_anomaly_dir,
    )

    # Summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nDatasets saved to: {output_dir}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/       - Training data (normal, {train_hours}h)")
    print(f"  ├── val_normal/  - Validation normal ({val_hours}h)")
    print(f"  └── val_anomaly/ - Validation anomaly ({val_hours}h, 90% anomalies)")
    print("\nNext steps:")
    print(f"  # Train LSTM model")
    print(f"  python scripts/train_bfd_lstm.py --data {train_dir}")
    print()
    print(f"  # Train HMM model")
    print(f"  python scripts/train_bfd_hmm.py --data {train_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BFD ML datasets (train, val_normal, val_anomaly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bfd"),
        help="Base output directory (default: data/bfd)",
    )

    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Number of BFD sessions (default: 10)",
    )

    parser.add_argument(
        "--train-hours",
        type=float,
        default=4.0,
        help="Training dataset duration in hours (default: 4.0)",
    )

    parser.add_argument(
        "--val-hours",
        type=float,
        default=1.0,
        help="Validation dataset duration in hours (default: 1.0)",
    )

    parser.add_argument(
        "--collection-interval",
        type=int,
        default=5,
        help="Collection interval in seconds (default: 5)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (small datasets: 5min train, 2min val, 3 sessions)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        print("⚡ QUICK MODE: Generating small test datasets")
        args.sessions = 3
        args.train_hours = 5 / 60  # 5 minutes
        args.val_hours = 2 / 60  # 2 minutes

    # Set random seed if provided
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Generate datasets
    generate_ml_datasets(
        output_dir=args.output,
        n_sessions=args.sessions,
        train_hours=args.train_hours,
        val_hours=args.val_hours,
        collection_interval=args.collection_interval,
    )


if __name__ == "__main__":
    main()
