#!/usr/bin/env python3
"""Train LSTM detector for BFD protocol anomaly detection.

This script trains an LSTM model to predict BFD detection times and
detect anomalies based on prediction error.

Usage:
    # Train on single file
    python scripts/train_bfd_lstm.py --data data/bfd/train/bfd_data_*.parquet

    # Train on directory
    python scripts/train_bfd_lstm.py --data data/bfd/train

    # Custom hyperparameters
    python scripts/train_bfd_lstm.py --data data/bfd/train \
        --epochs 100 --batch-size 64 --learning-rate 0.001
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.plugins.detectors.lstm import LSTMDetector


def load_training_data(data_path: Path, metric_name: str = "detection_time_ms") -> np.ndarray:
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
    print(f"Value range: [{training_data.min():.2f}, {training_data.max():.2f}]")
    print(f"Mean: {training_data.mean():.2f}, Std: {training_data.std():.2f}")

    return training_data


def train_lstm_model(
    data: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
) -> LSTMDetector:
    """Train LSTM detector.

    Args:
        data: Training data array
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sequence_length: Sequence length
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers

    Returns:
        Trained LSTM detector
    """
    print("\n" + "=" * 80)
    print("Training LSTM Detector")
    print("=" * 80)
    print(f"Hyperparameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print()

    # Create detector
    detector = LSTMDetector(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    # Train
    start_time = datetime.now()
    detector.train(
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ Training completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Final loss: {detector.training_losses[-1]:.6f}")
    print(f"Anomaly threshold: {detector.anomaly_threshold:.6f}")

    return detector


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM detector for BFD protocol",
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
        default="detection_time_ms",
        help="Metric name to train on (default: detection_time_ms)",
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

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length (default: 10)",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden size (default: 64)",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/bfd/lstm_v1.0.0.pth"),
        help="Output model path (default: models/bfd/lstm_v1.0.0.pth)",
    )

    args = parser.parse_args()

    # Load data
    training_data = load_training_data(args.data, args.metric)

    # Train model
    detector = train_lstm_model(
        data=training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    # Save model
    print(f"\nSaving model to: {args.output}")
    detector.save_model(str(args.output))

    # Save training metadata
    metadata_path = args.output.with_suffix(".txt")
    with open(metadata_path, "w") as f:
        f.write("BFD LSTM Model Training Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Trained: {datetime.now()}\n\n")
        f.write("Data:\n")
        f.write(f"  Source: {args.data}\n")
        f.write(f"  Metric: {args.metric}\n")
        f.write(f"  Samples: {len(training_data):,}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Sequence length: {args.sequence_length}\n")
        f.write(f"  Hidden size: {args.hidden_size}\n")
        f.write(f"  Num layers: {args.num_layers}\n\n")
        f.write("Training Results:\n")
        f.write(f"  Final loss: {detector.training_losses[-1]:.6f}\n")
        f.write(f"  Anomaly threshold: {detector.anomaly_threshold:.6f}\n\n")
        f.write("Loss history (last 10 epochs):\n")
        for i, loss in enumerate(detector.training_losses[-10:], 1):
            epoch_num = len(detector.training_losses) - 10 + i
            f.write(f"  Epoch {epoch_num:3d}: {loss:.6f}\n")

    print(f"Saved metadata to: {metadata_path}")

    print("\n✅ LSTM training completed successfully!")
    print(f"\nModel saved to: {args.output}")
    print("\nNext steps:")
    print(f"  # Run inference")
    print(f"  python scripts/infer_bfd.py --model {args.output} --detector lstm")


if __name__ == "__main__":
    main()
