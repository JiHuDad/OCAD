#!/usr/bin/env python3
"""Train Isolation Forest models for CFM metrics.

This script trains Isolation Forest models for CFM anomaly detection.
Isolation Forest is simpler and doesn't require deep learning frameworks.

Usage:
    # Train all CFM metrics
    python scripts/train_cfm_isoforest.py --train-data data/cfm/train/cfm_train_*.parquet

    # Train specific metric
    python scripts/train_cfm_isoforest.py --train-data data/cfm/train/cfm_train_*.parquet --metric udp_echo_rtt_ms
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

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
    contamination: float = 0.05,
    n_estimators: int = 100,
    version: str = "1.0.0",
) -> tuple:
    """Train Isolation Forest model for a single metric.

    Args:
        metric_name: Name of metric
        train_data: Training data array
        output_dir: Output directory for models
        contamination: Expected proportion of outliers
        n_estimators: Number of trees
        version: Model version

    Returns:
        Tuple of (model, scaler)
    """
    print(f"\n{'='*70}")
    print(f"Training Isolation Forest for {metric_name}")
    print(f"{'='*70}")

    # Reshape for sklearn
    X = train_data.reshape(-1, 1)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    print(f"\nTraining parameters:")
    print(f"  N estimators: {n_estimators}")
    print(f"  Contamination: {contamination}")
    print(f"  Samples: {len(X)}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # Calculate anomaly scores on training data
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    n_anomalies = (predictions == -1).sum()
    print(f"\nTraining complete:")
    print(f"  Detected anomalies in training: {n_anomalies} ({n_anomalies/len(X)*100:.2f}%)")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"cfm_{metric_name}_v{version}.pkl"
    scaler_path = output_dir / f"cfm_{metric_name}_v{version}_scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")

    return model, scaler


def main():
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest models for CFM metrics",
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
        "--contamination",
        type=float,
        default=0.05,
        help="Expected proportion of outliers (default: 0.05 = 5%%)",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees (default: 100)",
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
    print("CFM Isolation Forest Model Training")
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
            model, scaler = train_metric_model(
                metric_name=metric,
                train_data=train_data,
                output_dir=args.output_dir,
                contamination=args.contamination,
                n_estimators=args.n_estimators,
                version=args.version,
            )

            trained_models[metric] = (model, scaler)

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
    for metric in trained_models.keys():
        model_path = args.output_dir / f"cfm_{metric}_v{args.version}.pkl"
        print(f"  ✅ {metric}: {model_path}")

    if len(trained_models) < len(metrics_to_train):
        failed = set(metrics_to_train) - set(trained_models.keys())
        print(f"\n❌ Failed to train: {', '.join(failed)}")
        return 1

    print(f"\n✅ All models trained successfully!")
    print(f"\nNext steps:")
    print(f"  1. Run inference:")
    print(f"     python scripts/infer_cfm_isoforest.py")
    print(f"  2. Generate report:")
    print(f"     python scripts/report_cfm.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
