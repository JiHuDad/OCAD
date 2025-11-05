#!/usr/bin/env python3
"""Train TCN models for CFM metrics.

This script trains TCN (Temporal Convolutional Network) models for each CFM metric:
- udp_echo_rtt_ms: UDP Echo RTT
- ecpri_delay_us: eCPRI One-way Delay
- lbm_rtt_ms: LBM RTT

Each metric is trained separately to learn normal behavior patterns.

Usage:
    # Train all CFM metrics
    python scripts/train_cfm.py --train-data data/cfm/train/cfm_train_*.parquet

    # Train specific metric
    python scripts/train_cfm.py --train-data data/cfm/train/cfm_train_*.parquet --metric udp_echo_rtt_ms

    # Custom training parameters
    python scripts/train_cfm.py --train-data data/cfm/train/cfm_train_*.parquet --epochs 100 --batch-size 64
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Add OCAD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TCN detector for training
from ocad.plugins.detectors.tcn import TCNDetector


# CFM metrics to train
CFM_METRICS = [
    "udp_echo_rtt_ms",
    "ecpri_delay_us",
    "lbm_rtt_ms",
]


def load_training_data(data_path: Path, metric_name: str) -> np.ndarray:
    """Load training data for a specific metric.

    Args:
        data_path: Path to training data file (parquet or csv)
        metric_name: Name of metric column to extract

    Returns:
        1D numpy array of metric values

    Raises:
        ValueError: If file or column not found
    """
    if not data_path.exists():
        raise ValueError(f"Training data not found: {data_path}")

    # Load data
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in data. Available: {df.columns.tolist()}")

    # Extract metric values
    values = df[metric_name].values

    print(f"  Loaded {len(values):,} samples for {metric_name}")
    print(f"  Range: [{values.min():.2f}, {values.max():.2f}]")
    print(f"  Mean: {values.mean():.2f}, Std: {values.std():.2f}")

    return values


def train_metric_model(
    metric_name: str,
    train_data: np.ndarray,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    sequence_length: int = 20,
    version: str = "1.0.0",
) -> TCNDetector:
    """Train TCN model for a single metric.

    Args:
        metric_name: Name of metric
        train_data: Training data array
        output_dir: Output directory for models
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sequence_length: Sequence length for TCN
        version: Model version

    Returns:
        Trained TCN detector
    """
    print(f"\n{'='*70}")
    print(f"Training TCN for {metric_name}")
    print(f"{'='*70}")

    # Initialize detector
    detector = TCNDetector(
        sequence_length=sequence_length,
        threshold_multiplier=3.0,  # 3-sigma threshold
        num_channels=[32, 32, 32, 32],  # 4 layers, 32 channels each
        kernel_size=3,
    )

    # Train model
    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Num channels: {detector.num_channels}")

    detector.train(
        data=train_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"cfm_{metric_name}_v{version}.pth"

    detector.save_model(str(model_path))

    print(f"\n✅ Model saved: {model_path}")
    print(f"  Anomaly threshold: {detector.anomaly_threshold:.4f}")
    print(f"  Receptive field: {detector.model.receptive_field} timesteps")
    print(f"  Final training loss: {detector.training_losses[-1]:.6f}")

    return detector


def main():
    parser = argparse.ArgumentParser(
        description="Train TCN models for CFM metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training data file (parquet or csv)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=CFM_METRICS + ["all"],
        default="all",
        help="Metric to train (default: all)",
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
        default=20,
        help="Sequence length for TCN (default: 20)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/cfm"),
        help="Output directory for models (default: models/cfm)",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Model version (default: 1.0.0)",
    )

    args = parser.parse_args()

    print("="*70)
    print("CFM TCN Model Training")
    print("="*70)

    # Determine which metrics to train
    if args.metric == "all":
        metrics_to_train = CFM_METRICS
    else:
        metrics_to_train = [args.metric]

    print(f"\nMetrics to train: {', '.join(metrics_to_train)}")
    print(f"Training data: {args.train_data}")

    # Train each metric
    trained_models = {}

    for metric in metrics_to_train:
        try:
            # Load data
            train_data = load_training_data(args.train_data, metric)

            # Train model
            detector = train_metric_model(
                metric_name=metric,
                train_data=train_data,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                sequence_length=args.sequence_length,
                version=args.version,
            )

            trained_models[metric] = detector

        except Exception as e:
            print(f"\n❌ Error training {metric}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    print(f"Successfully trained {len(trained_models)}/{len(metrics_to_train)} models:")
    for metric, detector in trained_models.items():
        model_path = args.output_dir / f"cfm_{metric}_v{args.version}.pth"
        print(f"  ✅ {metric}: {model_path}")
        print(f"     Threshold: {detector.anomaly_threshold:.4f}")
        print(f"     Final loss: {detector.training_losses[-1]:.6f}")

    if len(trained_models) < len(metrics_to_train):
        failed = set(metrics_to_train) - set(trained_models.keys())
        print(f"\n❌ Failed to train: {', '.join(failed)}")
        return 1

    print(f"\n✅ All models trained successfully!")
    print(f"\nNext steps:")
    print(f"  1. Run inference:")
    print(f"     python scripts/infer_cfm.py")
    print(f"  2. Generate report:")
    print(f"     python scripts/report_cfm.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
