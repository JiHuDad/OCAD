#!/usr/bin/env python3
"""Run inference on BFD validation data using trained models.

This script loads a trained model (LSTM or HMM) and runs inference
on validation datasets, producing predictions and performance metrics.

Usage:
    # LSTM inference on normal data
    python scripts/infer_bfd.py --model models/bfd/lstm_v1.0.0.pth \
        --detector lstm --data data/bfd/val_normal

    # HMM inference on anomaly data
    python scripts/infer_bfd.py --model models/bfd/hmm_v1.0.0.pkl \
        --detector hmm --data data/bfd/val_anomaly

    # Inference on both normal and anomaly data
    python scripts/infer_bfd.py --model models/bfd/lstm_v1.0.0.pth \
        --detector lstm --data data/bfd/val_normal data/bfd/val_anomaly
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_path: Path, detector_type: str):
    """Load trained model.

    Args:
        model_path: Path to model file
        detector_type: Type of detector ('lstm' or 'hmm')

    Returns:
        Loaded detector instance
    """
    print(f"Loading {detector_type.upper()} model from: {model_path}")

    if detector_type == "lstm":
        from ocad.plugins.detectors.lstm import LSTMDetector
        detector = LSTMDetector()
    elif detector_type == "hmm":
        from ocad.plugins.detectors.hmm import HMMDetector
        detector = HMMDetector()
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

    detector.load_model(str(model_path))
    print(f"✅ Model loaded successfully")
    print(f"   Sequence length: {detector.sequence_length}")
    print(f"   Anomaly threshold: {detector.anomaly_threshold:.6f}")

    return detector


def load_validation_data(data_paths: List[Path], metric_name: str) -> pd.DataFrame:
    """Load validation data from parquet file(s).

    Args:
        data_paths: List of paths to parquet files or directories
        metric_name: Metric column to use

    Returns:
        Combined DataFrame
    """
    print(f"\nLoading validation data...")

    all_files = []
    for data_path in data_paths:
        if data_path.is_file():
            all_files.append(data_path)
        elif data_path.is_dir():
            all_files.extend(sorted(data_path.glob("*.parquet")))

    if not all_files:
        raise ValueError(f"No parquet files found in {data_paths}")

    print(f"Found {len(all_files)} parquet file(s)")

    # Load all files
    dfs = []
    for file in all_files:
        df = pd.read_parquet(file)
        dfs.append(df)
        print(f"  - {file.parent.name}/{file.name}: {len(df):,} records")

    # Combine
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records: {len(df):,}")

    # Check if metric exists
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found. Available: {df.columns.tolist()}")

    # Show data statistics
    if "is_anomaly" in df.columns:
        n_anomalies = df["is_anomaly"].sum()
        print(f"  Normal: {len(df) - n_anomalies:,} ({(len(df) - n_anomalies)/len(df)*100:.1f}%)")
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(df)*100:.1f}%)")

    return df


def run_inference(detector, df: pd.DataFrame, metric_name: str, sequence_length: int) -> pd.DataFrame:
    """Run inference on validation data.

    Args:
        detector: Trained detector
        df: Validation DataFrame
        metric_name: Metric column name
        sequence_length: Sequence length for detector

    Returns:
        DataFrame with predictions
    """
    print(f"\n" + "=" * 80)
    print("Running Inference")
    print("=" * 80)

    results = []

    # Group by source_id to maintain sequence history
    for source_id, group in df.groupby("source_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        values = group[metric_name].values

        for i in range(len(group)):
            row = group.iloc[i]

            # Build history
            if i < sequence_length:
                history = values[:i].tolist()
            else:
                history = values[i-sequence_length:i].tolist()

            # Detect
            current_value = values[i]
            result = detector.detect({
                "timestamp": row["timestamp"],
                "source_id": source_id,
                "metric_name": metric_name,
                "value": current_value,
                "history": history,
            })

            # Store result
            results.append({
                "timestamp": row["timestamp"],
                "source_id": source_id,
                "metric_name": metric_name,
                "value": current_value,
                "true_label": row.get("is_anomaly", False),
                "predicted_label": result["is_anomaly"],
                "anomaly_score": result["score"],
                "confidence": result["confidence"],
                "evidence": str(result.get("evidence", {})),
            })

        # Progress
        if len(results) % 1000 == 0:
            print(f"  Processed: {len(results):,} records")

    print(f"✅ Inference completed: {len(results):,} predictions")

    return pd.DataFrame(results)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate performance metrics.

    Args:
        df: DataFrame with true_label and predicted_label

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values

    # Handle case where all labels are same class
    if len(np.unique(y_true)) == 1:
        print("\n⚠️  Warning: All true labels are the same class")
        print(f"   This is expected for single-class validation sets (e.g., val_normal)")
        print(f"   Metrics like precision/recall may not be meaningful.")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on BFD validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model file",
    )

    parser.add_argument(
        "--detector",
        choices=["lstm", "hmm"],
        required=True,
        help="Detector type",
    )

    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to validation data (parquet files or directories)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        help="Metric name (auto-detect based on detector if not specified)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bfd/predictions.csv"),
        help="Output predictions file (default: results/bfd/predictions.csv)",
    )

    args = parser.parse_args()

    # Auto-detect metric if not specified
    if args.metric is None:
        if args.detector == "lstm":
            args.metric = "detection_time_ms"
        elif args.detector == "hmm":
            args.metric = "local_state"
        print(f"Auto-detected metric: {args.metric}")

    # Load model
    detector = load_model(args.model, args.detector)

    # Load validation data
    df = load_validation_data(args.data, args.metric)

    # Run inference
    predictions_df = run_inference(detector, df, args.metric, detector.sequence_length)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("Performance Metrics")
    print("=" * 80)
    metrics = calculate_metrics(predictions_df)

    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  [[TN={cm[0,0]:5d}, FP={cm[0,1]:5d}]")
    print(f"   [FN={cm[1,0]:5d}, TP={cm[1,1]:5d}]]")

    # Save predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(args.output, index=False)
    print(f"\n✅ Predictions saved to: {args.output}")

    # Save metrics
    metrics_path = args.output.with_suffix(".metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("BFD Inference Metrics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Detector: {args.detector}\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"Data: {', '.join(str(p) for p in args.data)}\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-score:  {metrics['f1_score']:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  [[TN={cm[0,0]:5d}, FP={cm[0,1]:5d}]\n")
        f.write(f"   [FN={cm[1,0]:5d}, TP={cm[1,1]:5d}]]\n")

    print(f"Saved metrics to: {metrics_path}")

    print("\n✅ Inference completed successfully!")
    print(f"\nNext steps:")
    print(f"  # Generate detailed report")
    print(f"  python scripts/report_bfd.py --predictions {args.output}")


if __name__ == "__main__":
    main()
