#!/usr/bin/env python3
"""Run inference on PTP validation data using trained TCN model.

This script loads a trained TCN model and runs inference on validation data,
producing predictions and anomaly scores.

Usage:
    # Run on validation normal data
    python scripts/infer_ptp.py --model models/ptp/tcn_v1.0.0.pth --data data/ptp/val_normal/

    # Run on validation anomaly data
    python scripts/infer_ptp.py --model models/ptp/tcn_v1.0.0.pth --data data/ptp/val_anomaly/

    # Custom output path
    python scripts/infer_ptp.py --model models/ptp/tcn_v1.0.0.pth --data data/ptp/val_normal/ --output results/ptp/normal_predictions.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.plugins.detectors.tcn import TCNDetector


def load_validation_data(data_path: Path, metric_name: str = "offset_from_master_ns") -> pd.DataFrame:
    """Load validation data from parquet file(s).

    Args:
        data_path: Path to parquet file or directory containing parquet files
        metric_name: Metric to extract for inference

    Returns:
        DataFrame with validation data
    """
    if data_path.is_file():
        print(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
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

    # Check for metric
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in data. Available: {df.columns.tolist()}")

    # Sort by timestamp and source_id
    df = df.sort_values(["source_id", "timestamp"])

    # Statistics
    if "is_anomaly" in df.columns:
        n_anomalies = df["is_anomaly"].sum()
        print(f"  Normal: {len(df) - n_anomalies:,} ({(len(df) - n_anomalies) / len(df) * 100:.1f}%)")
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies / len(df) * 100:.1f}%)")

    print(f"\nMetric '{metric_name}' statistics:")
    print(f"  Mean: {df[metric_name].mean():.2f} ns")
    print(f"  Std: {df[metric_name].std():.2f} ns")
    print(f"  Min: {df[metric_name].min():.2f} ns")
    print(f"  Max: {df[metric_name].max():.2f} ns")

    return df


def run_inference(detector: TCNDetector, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Run inference on validation data.

    Args:
        detector: Trained TCNDetector instance
        df: Validation data
        metric_name: Metric name to predict

    Returns:
        DataFrame with predictions and anomaly scores
    """
    print("\n" + "=" * 70)
    print("Running Inference")
    print("=" * 70)
    print(f"Sequence length: {detector.sequence_length}")
    print(f"Anomaly threshold: {detector.anomaly_threshold:.6f}")

    results = []

    # Group by source_id to maintain temporal sequences
    grouped = df.groupby("source_id")

    for source_id, group in tqdm(grouped, desc="Processing sources"):
        group = group.sort_values("timestamp")
        values = group[metric_name].values

        # Build history for each point
        for i in range(len(values)):
            if i < detector.sequence_length:
                # Not enough history yet
                results.append({
                    "timestamp": group.iloc[i]["timestamp"],
                    "source_id": source_id,
                    "metric_name": metric_name,
                    "actual_value": values[i],
                    "predicted_value": np.nan,
                    "prediction_error": np.nan,
                    "anomaly_score": 0.0,
                    "is_anomaly_predicted": False,
                    "is_anomaly_actual": group.iloc[i].get("is_anomaly", False),
                    "anomaly_type_actual": group.iloc[i].get("anomaly_type", None),
                    "confidence": 0.0,
                })
            else:
                # Have enough history, run inference
                history = values[i - detector.sequence_length:i].tolist()

                features = {
                    "timestamp": group.iloc[i]["timestamp"],
                    "source_id": source_id,
                    "metric_name": metric_name,
                    "value": values[i],
                    "history": history,
                }

                detection_result = detector.detect(features)

                results.append({
                    "timestamp": group.iloc[i]["timestamp"],
                    "source_id": source_id,
                    "metric_name": metric_name,
                    "actual_value": values[i],
                    "predicted_value": detection_result["evidence"].get("predicted_value", np.nan),
                    "prediction_error": detection_result["evidence"].get("prediction_error", np.nan),
                    "anomaly_score": detection_result["score"],
                    "is_anomaly_predicted": detection_result["is_anomaly"],
                    "is_anomaly_actual": group.iloc[i].get("is_anomaly", False),
                    "anomaly_type_actual": group.iloc[i].get("anomaly_type", None),
                    "confidence": detection_result["confidence"],
                })

    results_df = pd.DataFrame(results)

    print(f"\nInference completed: {len(results_df):,} predictions")

    # Calculate metrics for records with predictions
    valid_predictions = results_df[results_df["predicted_value"].notna()]
    print(f"  Valid predictions: {len(valid_predictions):,}")
    print(f"  Skipped (insufficient history): {len(results_df) - len(valid_predictions):,}")

    if len(valid_predictions) > 0:
        mean_error = valid_predictions["prediction_error"].mean()
        print(f"\n  Mean prediction error: {mean_error:.6f}")
        print(f"  Max prediction error: {valid_predictions['prediction_error'].max():.6f}")

        n_predicted_anomalies = valid_predictions["is_anomaly_predicted"].sum()
        print(f"\n  Predicted anomalies: {n_predicted_anomalies} ({n_predicted_anomalies / len(valid_predictions) * 100:.2f}%)")

        if "is_anomaly_actual" in valid_predictions.columns:
            n_actual_anomalies = valid_predictions["is_anomaly_actual"].sum()
            print(f"  Actual anomalies: {n_actual_anomalies} ({n_actual_anomalies / len(valid_predictions) * 100:.2f}%)")

            # Confusion matrix
            tp = ((valid_predictions["is_anomaly_predicted"] == True) & (valid_predictions["is_anomaly_actual"] == True)).sum()
            tn = ((valid_predictions["is_anomaly_predicted"] == False) & (valid_predictions["is_anomaly_actual"] == False)).sum()
            fp = ((valid_predictions["is_anomaly_predicted"] == True) & (valid_predictions["is_anomaly_actual"] == False)).sum()
            fn = ((valid_predictions["is_anomaly_predicted"] == False) & (valid_predictions["is_anomaly_actual"] == True)).sum()

            print(f"\n  Confusion Matrix:")
            print(f"    True Positives: {tp}")
            print(f"    True Negatives: {tn}")
            print(f"    False Positives: {fp}")
            print(f"    False Negatives: {fn}")

            if tp + fp > 0:
                precision = tp / (tp + fp)
                print(f"    Precision: {precision:.4f}")
            else:
                precision = 0.0
                print(f"    Precision: N/A (no positive predictions)")

            if tp + fn > 0:
                recall = tp / (tp + fn)
                print(f"    Recall: {recall:.4f}")
            else:
                recall = 0.0
                print(f"    Recall: N/A (no actual anomalies)")

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"    F1-Score: {f1:.4f}")

            if tp + tn + fp + fn > 0:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                print(f"    Accuracy: {accuracy:.4f}")

    return results_df


def save_predictions(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save predictions to CSV.

    Args:
        results_df: DataFrame with predictions
        output_path: Path to save predictions
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on PTP validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.pth file)",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to validation data (parquet file or directory)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="offset_from_master_ns",
        help="Metric to predict (default: offset_from_master_ns)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ptp/predictions.csv"),
        help="Output path for predictions (default: results/ptp/predictions.csv)",
    )

    args = parser.parse_args()

    # Validate
    if not args.model.exists():
        parser.error(f"Model file does not exist: {args.model}")
    if not args.data.exists():
        parser.error(f"Data path does not exist: {args.data}")

    print("=" * 70)
    print("PTP TCN Inference Pipeline")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {args.model}")
    detector = TCNDetector()
    detector.load_model(str(args.model))
    print(f"Model loaded successfully")
    print(f"  Sequence length: {detector.sequence_length}")
    print(f"  Receptive field: {detector.model.receptive_field}")
    print(f"  Anomaly threshold: {detector.anomaly_threshold:.6f}")

    # Load data
    df = load_validation_data(args.data, args.metric)

    # Run inference
    results_df = run_inference(detector, df, args.metric)

    # Save predictions
    save_predictions(results_df, args.output)

    print("\n" + "=" * 70)
    print("Inference Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  Generate performance report:")
    print(f"    python scripts/report_ptp.py --predictions {args.output}")


if __name__ == "__main__":
    main()
