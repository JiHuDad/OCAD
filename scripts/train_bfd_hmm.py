#!/usr/bin/env python3
"""Train HMM detector for BFD protocol anomaly detection.

This script trains a Hidden Markov Model to learn BFD state transition
patterns and detect anomalies based on sequence likelihood.

Usage:
    # Train on single file
    python scripts/train_bfd_hmm.py --data data/bfd/train/bfd_data_*.parquet

    # Train on directory
    python scripts/train_bfd_hmm.py --data data/bfd/train

    # Custom hyperparameters
    python scripts/train_bfd_hmm.py --data data/bfd/train \
        --n-components 6 --sequence-length 15
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.plugins.detectors.hmm import HMMDetector


def load_training_data(data_path: Path, metric_name: str = "local_state") -> np.ndarray:
    """Load training data from parquet file(s).

    Args:
        data_path: Path to parquet file or directory containing parquet files
        metric_name: Metric column to use for training

    Returns:
        NumPy array of training values
    """
    print(f"Loading training data from: {data_path}")

    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = sorted(data_path.glob("*.parquet"))
    else:
        raise ValueError(f"Path not found: {data_path}")

    if not files:
        raise ValueError(f"No parquet files found in {data_path}")

    print(f"Found {len(files)} parquet file(s)")

    # Load all files
    dfs = []
    for file in files:
        df = pd.read_parquet(file)
        dfs.append(df)
        print(f"  - {file.name}: {len(df):,} records")

    # Combine
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records: {len(df):,}")

    # Filter for normal data only
    if "is_anomaly" in df.columns:
        normal_df = df[df["is_anomaly"] == False]
        print(f"Normal records: {len(normal_df):,} ({len(normal_df)/len(df)*100:.1f}%)")
        df = normal_df

    # Extract metric
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in data. Available columns: {df.columns.tolist()}")

    training_data = df[metric_name].values
    print(f"Training data shape: {training_data.shape}")

    # Show state distribution for state-based metrics
    if metric_name in ["local_state", "remote_state"]:
        unique, counts = np.unique(training_data, return_counts=True)
        print(f"\nState distribution:")
        state_names = {0: "ADMIN_DOWN", 1: "DOWN", 2: "INIT", 3: "UP"}
        for state, count in zip(unique, counts):
            state_name = state_names.get(int(state), f"State_{int(state)}")
            print(f"  {state_name}: {count:,} ({count/len(training_data)*100:.1f}%)")
    else:
        print(f"Value range: [{training_data.min():.2f}, {training_data.max():.2f}]")
        print(f"Mean: {training_data.mean():.2f}, Std: {training_data.std():.2f}")

    return training_data


def train_hmm_model(
    data: np.ndarray,
    n_components: int,
    sequence_length: int,
    threshold_percentile: float,
) -> HMMDetector:
    """Train HMM detector.

    Args:
        data: Training data array
        n_components: Number of hidden states
        sequence_length: Sequence length
        threshold_percentile: Anomaly threshold percentile

    Returns:
        Trained HMM detector
    """
    print("\n" + "=" * 80)
    print("Training HMM Detector")
    print("=" * 80)
    print(f"Hyperparameters:")
    print(f"  Hidden states (n_components): {n_components}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Threshold percentile: {threshold_percentile}")
    print()

    # Create detector
    detector = HMMDetector(
        n_components=n_components,
        sequence_length=sequence_length,
        threshold_percentile=threshold_percentile,
    )

    # Train
    start_time = datetime.now()
    detector.train(data=data)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ Training completed in {elapsed:.1f}s")
    print(f"Anomaly threshold: {detector.anomaly_threshold:.6f}")

    return detector


def main():
    parser = argparse.ArgumentParser(
        description="Train HMM detector for BFD protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data (parquet file or directory)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="local_state",
        help="Metric name to train on (default: local_state)",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="Number of hidden states (default: 4)",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length (default: 10)",
    )

    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=5.0,
        help="Anomaly threshold percentile (default: 5.0)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/bfd/hmm_v1.0.0.pkl"),
        help="Output model path (default: models/bfd/hmm_v1.0.0.pkl)",
    )

    args = parser.parse_args()

    # Load data
    training_data = load_training_data(args.data, args.metric)

    # Train model
    detector = train_hmm_model(
        data=training_data,
        n_components=args.n_components,
        sequence_length=args.sequence_length,
        threshold_percentile=args.threshold_percentile,
    )

    # Save model
    print(f"\nSaving model to: {args.output}")
    detector.save_model(str(args.output))

    # Save training metadata
    metadata_path = args.output.with_suffix(".txt")
    with open(metadata_path, "w") as f:
        f.write("BFD HMM Model Training Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Trained: {datetime.now()}\n\n")
        f.write("Data:\n")
        f.write(f"  Source: {args.data}\n")
        f.write(f"  Metric: {args.metric}\n")
        f.write(f"  Samples: {len(training_data):,}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Hidden states: {args.n_components}\n")
        f.write(f"  Sequence length: {args.sequence_length}\n")
        f.write(f"  Threshold percentile: {args.threshold_percentile}\n\n")
        f.write("Training Results:\n")
        f.write(f"  Anomaly threshold: {detector.anomaly_threshold:.6f}\n")

    print(f"Saved metadata to: {metadata_path}")

    print("\n✅ HMM training completed successfully!")
    print(f"\nModel saved to: {args.output}")
    print("\nNext steps:")
    print(f"  # Run inference")
    print(f"  python scripts/infer_bfd.py --model {args.output} --detector hmm")


if __name__ == "__main__":
    main()
