#!/usr/bin/env python3
"""Run inference on CFM validation data using trained TCN models.

This script loads trained TCN models and runs inference on validation data
to evaluate anomaly detection performance.

Usage:
    # Run inference on validation datasets
    python scripts/infer_cfm.py \
        --val-normal data/cfm/val_normal/cfm_val_normal_*.parquet \
        --val-anomaly data/cfm/val_anomaly/cfm_val_anomaly_*.parquet \
        --models-dir models/cfm

    # Custom output directory
    python scripts/infer_cfm.py \
        --val-normal data/cfm/val_normal/cfm_val_normal_*.parquet \
        --val-anomaly data/cfm/val_anomaly/cfm_val_anomaly_*.parquet \
        --models-dir models/cfm \
        --output results/cfm
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import glob

# Add OCAD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TCN detector
from ocad.plugins.detectors.tcn import TCNDetector


# CFM metrics
CFM_METRICS = [
    "udp_echo_rtt_ms",
    "ecpri_delay_us",
    "lbm_rtt_ms",
]


def load_models(models_dir: Path, version: str = "1.0.0") -> Dict[str, TCNDetector]:
    """Load trained TCN models for all CFM metrics.

    Args:
        models_dir: Directory containing trained models
        version: Model version to load

    Returns:
        Dictionary mapping metric names to loaded detectors

    Raises:
        ValueError: If models not found
    """
    models = {}

    print(f"Loading models from {models_dir}...")

    for metric in CFM_METRICS:
        model_path = models_dir / f"cfm_{metric}_v{version}.pth"

        if not model_path.exists():
            print(f"  ⚠️  Model not found: {model_path}")
            continue

        try:
            detector = TCNDetector()
            detector.load_model(str(model_path))
            models[metric] = detector
            print(f"  ✅ Loaded {metric}: threshold={detector.anomaly_threshold:.4f}")
        except Exception as e:
            print(f"  ❌ Error loading {metric}: {e}")

    if not models:
        raise ValueError(f"No models found in {models_dir}")

    return models


def run_inference_on_dataset(
    df: pd.DataFrame,
    models: Dict[str, TCNDetector],
    dataset_name: str,
    sequence_length: int = 20,
) -> pd.DataFrame:
    """Run inference on a dataset.

    Args:
        df: Input dataframe with CFM metrics
        models: Dictionary of trained models
        dataset_name: Name of dataset (for logging)
        sequence_length: Sequence length for TCN

    Returns:
        DataFrame with predictions added
    """
    print(f"\n{'='*70}")
    print(f"Running inference on {dataset_name}")
    print(f"{'='*70}")
    print(f"Records: {len(df):,}")

    # Sort by source_id and timestamp to ensure correct sequence
    df = df.sort_values(["source_id", "timestamp"]).reset_index(drop=True)

    # Add prediction columns
    for metric in models.keys():
        df[f"{metric}_pred_score"] = 0.0
        df[f"{metric}_pred_anomaly"] = False

    # Group by endpoint for sequence-based inference
    endpoints = df["source_id"].unique()
    print(f"Endpoints: {len(endpoints)}")

    for endpoint_id in endpoints:
        endpoint_mask = df["source_id"] == endpoint_id
        endpoint_df = df[endpoint_mask].copy()

        # For each metric, run inference
        for metric, detector in models.items():
            values = endpoint_df[metric].values

            # Build history and run inference
            for i in range(len(values)):
                if i < sequence_length:
                    # Not enough history yet
                    continue

                # Get historical sequence
                history = values[i - sequence_length:i].tolist()
                current_value = values[i]

                # Create features dictionary
                features = {
                    "timestamp": endpoint_df.iloc[i]["timestamp"],
                    "source_id": endpoint_id,
                    "metric_name": metric,
                    "value": current_value,
                    "history": history,
                }

                # Run detection
                result = detector.detect(features)

                # Store results
                idx = endpoint_df.index[i]
                df.loc[idx, f"{metric}_pred_score"] = result["score"]
                df.loc[idx, f"{metric}_pred_anomaly"] = result["is_anomaly"]

    # Calculate ensemble score (max across metrics)
    score_columns = [f"{m}_pred_score" for m in models.keys()]
    df["ensemble_score"] = df[score_columns].max(axis=1)
    df["ensemble_anomaly"] = df["ensemble_score"] > 0.5

    print(f"\nInference complete!")
    print(f"  Records with sufficient history: {(df['ensemble_score'] > 0).sum():,}")
    print(f"  Detected anomalies: {df['ensemble_anomaly'].sum():,}")

    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics.

    Args:
        df: DataFrame with predictions and ground truth

    Returns:
        Dictionary of performance metrics
    """
    # Only evaluate records with predictions (sufficient history)
    df_eval = df[df["ensemble_score"] > 0].copy()

    if len(df_eval) == 0:
        return {}

    # Ground truth
    y_true = df_eval["is_anomaly"].values
    y_pred = df_eval["ensemble_anomaly"].values

    # Calculate metrics
    tp = ((y_true == True) & (y_pred == True)).sum()
    fp = ((y_true == False) & (y_pred == True)).sum()
    tn = ((y_true == False) & (y_pred == False)).sum()
    fn = ((y_true == True) & (y_pred == False)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_evaluated": len(df_eval),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on CFM validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--val-normal",
        type=str,
        required=True,
        help="Path to validation normal data (supports wildcards)",
    )

    parser.add_argument(
        "--val-anomaly",
        type=str,
        required=True,
        help="Path to validation anomaly data (supports wildcards)",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/cfm"),
        help="Directory containing trained models (default: models/cfm)",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Model version to load (default: 1.0.0)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cfm"),
        help="Output directory for predictions (default: results/cfm)",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Sequence length for TCN (default: 20)",
    )

    args = parser.parse_args()

    print("="*70)
    print("CFM Inference")
    print("="*70)

    # Load models
    try:
        models = load_models(args.models_dir, args.version)
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        return 1

    print(f"\nLoaded {len(models)} models: {', '.join(models.keys())}")

    # Find data files
    val_normal_files = glob.glob(args.val_normal)
    val_anomaly_files = glob.glob(args.val_anomaly)

    if not val_normal_files:
        print(f"❌ No validation normal files found: {args.val_normal}")
        return 1

    if not val_anomaly_files:
        print(f"❌ No validation anomaly files found: {args.val_anomaly}")
        return 1

    print(f"\nValidation normal files: {len(val_normal_files)}")
    for f in val_normal_files:
        print(f"  {f}")

    print(f"\nValidation anomaly files: {len(val_anomaly_files)}")
    for f in val_anomaly_files:
        print(f"  {f}")

    # Load validation data
    print(f"\nLoading validation data...")

    df_normal = pd.concat([pd.read_parquet(f) for f in val_normal_files], ignore_index=True)
    df_anomaly = pd.concat([pd.read_parquet(f) for f in val_anomaly_files], ignore_index=True)

    print(f"  Normal: {len(df_normal):,} records")
    print(f"  Anomaly: {len(df_anomaly):,} records")

    # Run inference
    df_normal_pred = run_inference_on_dataset(df_normal, models, "Validation Normal", args.sequence_length)
    df_anomaly_pred = run_inference_on_dataset(df_anomaly, models, "Validation Anomaly", args.sequence_length)

    # Calculate metrics for each dataset
    print(f"\n{'='*70}")
    print("Performance Metrics")
    print(f"{'='*70}")

    metrics_normal = calculate_metrics(df_normal_pred)
    metrics_anomaly = calculate_metrics(df_anomaly_pred)

    if metrics_normal:
        print(f"\nValidation Normal (should detect few anomalies):")
        print(f"  Accuracy:  {metrics_normal['accuracy']*100:.2f}%")
        print(f"  False Positives: {metrics_normal['false_positives']} / {metrics_normal['total_evaluated']}")
        print(f"  False Positive Rate: {metrics_normal['false_positives']/metrics_normal['total_evaluated']*100:.2f}%")

    if metrics_anomaly:
        print(f"\nValidation Anomaly (should detect anomalies):")
        print(f"  Accuracy:  {metrics_anomaly['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics_anomaly['precision']*100:.2f}%")
        print(f"  Recall:    {metrics_anomaly['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics_anomaly['f1_score']*100:.2f}%")

    # Combine datasets
    df_combined = pd.concat([df_normal_pred, df_anomaly_pred], ignore_index=True)
    metrics_combined = calculate_metrics(df_combined)

    if metrics_combined:
        print(f"\nCombined (Overall Performance):")
        print(f"  Accuracy:  {metrics_combined['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics_combined['precision']*100:.2f}%")
        print(f"  Recall:    {metrics_combined['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics_combined['f1_score']*100:.2f}%")

    # Save predictions
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    predictions_csv = args.output / f"predictions_{timestamp}.csv"
    df_combined.to_csv(predictions_csv, index=False)
    print(f"\n✅ Predictions saved: {predictions_csv}")

    # Save metrics
    metrics_csv = args.output / f"metrics_{timestamp}.csv"
    metrics_df = pd.DataFrame([
        {"dataset": "normal", **metrics_normal},
        {"dataset": "anomaly", **metrics_anomaly},
        {"dataset": "combined", **metrics_combined},
    ])
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✅ Metrics saved: {metrics_csv}")

    print(f"\n{'='*70}")
    print("✅ Inference completed successfully!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  python scripts/report_cfm.py --predictions {predictions_csv} --metrics {metrics_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
