#!/usr/bin/env python3
"""Run inference on CFM data using trained Isolation Forest models.

Usage:
    # Validation mode (with ground truth)
    python scripts/infer_cfm_isoforest.py \
        --model models/cfm \
        --data data/cfm/val \
        --output results/cfm/predictions.csv

    # Production mode (without ground truth)
    python scripts/infer_cfm_isoforest.py \
        --model models/cfm \
        --data data/cfm/production \
        --output results/cfm/predictions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import joblib


# CFM metrics
CFM_METRICS = [
    "udp_echo_rtt_ms",
    "ecpri_delay_us",
    "lbm_rtt_ms",
]


def load_models(models_dir: Path, version: str = "1.0.0") -> Dict[str, tuple]:
    """Load trained Isolation Forest models for all CFM metrics.

    Args:
        models_dir: Directory containing model files
        version: Model version to load

    Returns:
        Dictionary mapping metric names to (model, scaler) tuples
    """
    models = {}

    print(f"Loading models from {models_dir}...")

    for metric in CFM_METRICS:
        model_path = models_dir / f"cfm_{metric}_v{version}.pkl"
        scaler_path = models_dir / f"cfm_{metric}_v{version}_scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            print(f"  ⚠️  Model or scaler not found for {metric}")
            continue

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            models[metric] = (model, scaler)
            print(f"  ✅ Loaded {metric}")
        except Exception as e:
            print(f"  ❌ Error loading {metric}: {e}")

    if not models:
        raise ValueError(f"No models found in {models_dir}")

    return models


def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from file or directory.

    Args:
        data_path: Path to data file or directory

    Returns:
        Combined DataFrame
    """
    if data_path.is_file():
        # Single file
        if data_path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    elif data_path.is_dir():
        # Directory - find all parquet files
        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            # Try csv
            csv_files = list(data_path.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No parquet or csv files found in {data_path}")
            return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    else:
        raise ValueError(f"Data path not found: {data_path}")


def run_inference(df: pd.DataFrame, models: Dict[str, tuple]) -> pd.DataFrame:
    """Run inference on a dataset.

    Args:
        df: Input dataframe with CFM metrics
        models: Dictionary of (model, scaler) tuples

    Returns:
        DataFrame with predictions added
    """
    print(f"\n{'='*70}")
    print(f"Running inference")
    print(f"{'='*70}")
    print(f"Records: {len(df):,}")

    # Check for required metrics
    missing_metrics = [m for m in models.keys() if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metrics in data: {missing_metrics}")

    # Add prediction columns
    for metric in models.keys():
        df[f"{metric}_pred_score"] = 0.0
        df[f"{metric}_pred_anomaly"] = False

    # For each metric, run inference
    for metric, (model, scaler) in models.items():
        values = df[metric].values.reshape(-1, 1)

        # Normalize
        values_scaled = scaler.transform(values)

        # Predict
        scores = model.decision_function(values_scaled)
        predictions = model.predict(values_scaled)

        # Convert to 0-1 score (higher = more anomalous)
        # Isolation Forest: negative scores = anomalies
        # We'll convert so that score > 0.5 means anomaly
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
            # Invert so anomalies have high scores
            normalized_scores = 1.0 - normalized_scores
        else:
            normalized_scores = np.zeros_like(scores)

        # Store results
        df[f"{metric}_pred_score"] = normalized_scores
        df[f"{metric}_pred_anomaly"] = (predictions == -1)  # -1 means anomaly in sklearn

    # Calculate ensemble score (max across metrics)
    score_columns = [f"{m}_pred_score" for m in models.keys()]
    df["ensemble_score"] = df[score_columns].max(axis=1)
    df["ensemble_anomaly"] = df[[f"{m}_pred_anomaly" for m in models.keys()]].any(axis=1)

    print(f"\nInference complete!")
    print(f"  Detected anomalies: {df['ensemble_anomaly'].sum():,} ({df['ensemble_anomaly'].sum()/len(df)*100:.1f}%)")

    return df


def calculate_metrics(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Calculate performance metrics if ground truth is available.

    Args:
        df: DataFrame with predictions

    Returns:
        Metrics dictionary if is_anomaly column exists, None otherwise
    """
    if "is_anomaly" not in df.columns:
        return None

    # Ground truth
    y_true = df["is_anomaly"].values
    y_pred = df["ensemble_anomaly"].values

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
        "total_evaluated": len(df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on CFM data using Isolation Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model directory containing trained models",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data file or directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cfm/predictions.csv"),
        help="Output predictions file (default: results/cfm/predictions.csv)",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Model version to load (default: 1.0.0)",
    )

    args = parser.parse_args()

    print("="*70)
    print("CFM Inference (Isolation Forest)")
    print("="*70)

    # Validate paths
    if not args.model.exists():
        print(f"❌ Model directory not found: {args.model}")
        return 1

    if not args.data.exists():
        print(f"❌ Data path not found: {args.data}")
        return 1

    # Load models
    try:
        models = load_models(args.model, args.version)
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\nLoaded {len(models)} models: {', '.join(models.keys())}")

    # Load data
    try:
        print(f"\nLoading data from {args.data}...")
        df = load_data(args.data)
        print(f"  Loaded {len(df):,} records")

        # Check if validation mode (has ground truth)
        has_ground_truth = "is_anomaly" in df.columns
        if has_ground_truth:
            print(f"  ✅ Validation mode: Ground truth available (is_anomaly column found)")
            n_anomalies = df["is_anomaly"].sum()
            print(f"  Ground truth anomalies: {n_anomalies:,} ({n_anomalies/len(df)*100:.1f}%)")
        else:
            print(f"  ℹ️  Production mode: No ground truth (is_anomaly column not found)")

    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    try:
        df_pred = run_inference(df, models)
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Calculate metrics if ground truth available
    if has_ground_truth:
        print(f"\n{'='*70}")
        print("Performance Metrics")
        print(f"{'='*70}")

        metrics = calculate_metrics(df_pred)
        if metrics:
            print(f"\n  Accuracy:  {metrics['accuracy']*100:.2f}%")
            print(f"  Precision: {metrics['precision']*100:.2f}%")
            print(f"  Recall:    {metrics['recall']*100:.2f}%")
            print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
            print(f"\n  True Positives:  {metrics['true_positives']}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  True Negatives:  {metrics['true_negatives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")

    # Save predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(args.output, index=False)
    print(f"\n✅ Predictions saved: {args.output}")

    print(f"\n{'='*70}")
    print("✅ Inference completed successfully!")
    print(f"{'='*70}")

    if has_ground_truth:
        print(f"\nNext steps:")
        print(f"  python scripts/report_cfm.py --predictions {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
