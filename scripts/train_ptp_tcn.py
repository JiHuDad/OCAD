#!/usr/bin/env python3
"""Train TCN model for PTP anomaly detection.

This script trains a Temporal Convolutional Network (TCN) to predict PTP offset values
and detect anomalies based on prediction error.

Usage:
    # Train on specific parquet file
    python scripts/train_ptp_tcn.py --data data/ptp/train/ptp_train_20251105_120000.parquet

    # Train on all files in directory
    python scripts/train_ptp_tcn.py --data data/ptp/train/

    # Custom hyperparameters
    python scripts/train_ptp_tcn.py --data data/ptp/train/ --epochs 100 --batch-size 64 --sequence-length 30
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.plugins.detectors.tcn import TCNDetector


def load_training_data(data_path: Path, metric_name: str = "offset_from_master_ns") -> np.ndarray:
    """Load training data from parquet file(s).

    Args:
        data_path: Path to parquet file or directory containing parquet files
        metric_name: Metric to extract for training

    Returns:
        Numpy array of training data
    """
    if data_path.is_file():
        print(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        parquet_files = [data_path]
    elif data_path.is_dir():
        parquet_files = sorted(data_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in: {data_path}")
        print(f"Loading data from {len(parquet_files)} parquet file(s):")
        for pf in parquet_files:
            print(f"  - {pf.name}")
        df = pd.concat([pd.read_parquet(pf) for pf in parquet_files], ignore_index=True)
    else:
        raise ValueError(f"Invalid path: {data_path}")

    print(f"Loaded {len(df):,} records")

    # Check for anomalies (should be 0 for training data)
    if "is_anomaly" in df.columns:
        n_anomalies = df["is_anomaly"].sum()
        if n_anomalies > 0:
            print(f"WARNING: Training data contains {n_anomalies} anomalies ({n_anomalies / len(df) * 100:.1f}%)")
            print("  Filtering out anomalies for training...")
            df = df[~df["is_anomaly"]]
            print(f"  Filtered to {len(df):,} normal records")

    # Extract metric
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in data. Available: {df.columns.tolist()}")

    # Sort by timestamp and source_id to ensure temporal order
    df = df.sort_values(["source_id", "timestamp"])

    training_data = df[metric_name].values

    print(f"Training data shape: {training_data.shape}")
    print(f"Training data statistics:")
    print(f"  Mean: {training_data.mean():.2f} ns")
    print(f"  Std: {training_data.std():.2f} ns")
    print(f"  Min: {training_data.min():.2f} ns")
    print(f"  Max: {training_data.max():.2f} ns")
    print(f"  Median: {np.median(training_data):.2f} ns")

    return training_data


def train_model(
    training_data: np.ndarray,
    sequence_length: int = 20,
    threshold_multiplier: float = 3.0,
    num_channels: list = None,
    kernel_size: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> TCNDetector:
    """Train TCN model.

    Args:
        training_data: Training data array
        sequence_length: Number of historical points for prediction
        threshold_multiplier: Multiplier for anomaly threshold
        num_channels: Channel sizes for each layer
        kernel_size: Convolution kernel size
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Trained TCNDetector instance
    """
    if num_channels is None:
        num_channels = [25, 25, 25, 25]

    print("\n" + "=" * 70)
    print("Training TCN Model")
    print("=" * 70)
    print(f"Hyperparameters:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Threshold multiplier: {threshold_multiplier}")
    print(f"  Num channels: {num_channels}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    # Initialize detector
    detector = TCNDetector(
        sequence_length=sequence_length,
        threshold_multiplier=threshold_multiplier,
        num_channels=num_channels,
        kernel_size=kernel_size,
    )

    # Train
    start_time = datetime.now()
    detector.train(
        training_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    end_time = datetime.now()

    training_duration = (end_time - start_time).total_seconds()
    print(f"\nTraining completed in {training_duration:.2f} seconds")
    print(f"  Final loss: {detector.training_losses[-1]:.6f}")
    print(f"  Anomaly threshold: {detector.anomaly_threshold:.6f}")
    print(f"  Receptive field: {detector.model.receptive_field} timesteps")

    return detector


def save_model(detector: TCNDetector, output_path: Path, metadata: dict = None) -> None:
    """Save trained model with metadata.

    Args:
        detector: Trained TCNDetector instance
        output_path: Path to save model
        metadata: Optional metadata dictionary
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    detector.save_model(str(output_path))

    # Save metadata
    if metadata is not None:
        metadata_path = output_path.with_suffix(".metadata.txt")
        with open(metadata_path, "w") as f:
            f.write("PTP TCN Model Metadata\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Trained: {datetime.now()}\n")
            f.write(f"Model path: {output_path}\n\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved metadata: {metadata_path}")

    print(f"Model saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train TCN model for PTP anomaly detection",
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
        default="offset_from_master_ns",
        help="Metric to train on (default: offset_from_master_ns)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ptp/tcn_v1.0.0.pth"),
        help="Output model path (default: models/ptp/tcn_v1.0.0.pth)",
    )

    # Hyperparameters
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Number of historical points for prediction (default: 20)",
    )

    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=3.0,
        help="Multiplier for anomaly threshold (default: 3.0)",
    )

    parser.add_argument(
        "--num-channels",
        type=int,
        nargs="+",
        default=[25, 25, 25, 25],
        help="Channel sizes for each layer (default: 25 25 25 25)",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Convolution kernel size (default: 3)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    args = parser.parse_args()

    # Validate
    if not args.data.exists():
        parser.error(f"Data path does not exist: {args.data}")

    print("=" * 70)
    print("PTP TCN Training Pipeline")
    print("=" * 70)

    # Load data
    training_data = load_training_data(args.data, args.metric)

    # Train model
    detector = train_model(
        training_data=training_data,
        sequence_length=args.sequence_length,
        threshold_multiplier=args.threshold_multiplier,
        num_channels=args.num_channels,
        kernel_size=args.kernel_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Save model
    metadata = {
        "metric": args.metric,
        "n_samples": len(training_data),
        "sequence_length": args.sequence_length,
        "threshold_multiplier": args.threshold_multiplier,
        "num_channels": args.num_channels,
        "kernel_size": args.kernel_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_loss": detector.training_losses[-1],
        "anomaly_threshold": detector.anomaly_threshold,
        "receptive_field": detector.model.receptive_field,
    }

    save_model(detector, args.output, metadata)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Run inference on validation data:")
    print(f"     python scripts/infer_ptp.py --model {args.output} --data data/ptp/val_normal/")
    print(f"  2. Generate performance report:")
    print(f"     python scripts/report_ptp.py --predictions results/ptp/predictions.csv")


if __name__ == "__main__":
    main()
