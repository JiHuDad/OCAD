#!/usr/bin/env python3
"""Run inference on BGP data using trained GNN model.

This script loads a trained GNN model and runs inference on validation data.

Usage:
    python scripts/infer_bgp.py --model models/bgp/gnn_v1.0.0.pth --data data/bgp/val_normal data/bgp/val_anomaly

    # Single dataset
    python scripts/infer_bgp.py --model models/bgp/gnn_v1.0.0.pth --data data/bgp/val_normal
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add OCAD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX not installed. Install with: pip install networkx")
    sys.exit(1)

from ocad.plugins.detectors.gnn import GNNDetector


def create_as_path_graph(row: pd.Series) -> nx.Graph:
    """Create an AS-path graph from a BGP session row."""
    graph = nx.Graph()

    local_asn = row["local_asn"]
    remote_asn = row["remote_asn"]

    # Add nodes with features
    update_rate = row["update_delta"] / 100.0
    prefix_count = row["prefix_count"] / 200.0

    graph.add_node(local_asn, features=[
        local_asn / 70000.0,
        update_rate,
        prefix_count,
    ])

    graph.add_node(remote_asn, features=[
        remote_asn / 70000.0,
        update_rate,
        prefix_count,
    ])

    # Add intermediate AS hops
    path_length = int(row["as_path_length"])

    prev_as = local_asn
    for i in range(1, path_length):
        if i == path_length - 1:
            intermediate_as = remote_asn
        else:
            intermediate_as = 64000 + (local_asn + remote_asn + i) % 1000

            if intermediate_as not in graph:
                graph.add_node(intermediate_as, features=[
                    intermediate_as / 70000.0,
                    update_rate * 0.5,
                    prefix_count * 0.5,
                ])

        graph.add_edge(prev_as, intermediate_as)
        prev_as = intermediate_as

    return graph


def load_data(data_paths: List[Path]) -> pd.DataFrame:
    """Load data from multiple paths."""
    print("Loading data...")

    all_dfs = []
    for data_path in data_paths:
        if data_path.is_dir():
            parquet_files = list(data_path.glob("*.parquet"))
            print(f"  {data_path.name}: {len(parquet_files)} file(s)")

            for pf in parquet_files:
                df = pd.read_parquet(pf)
                all_dfs.append(df)
        else:
            df = pd.read_parquet(data_path)
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total records: {len(df)}")

    return df


def run_inference(detector: GNNDetector, df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on DataFrame."""
    print("\nRunning inference...")

    results = []

    for idx, row in df.iterrows():
        # Create graph
        graph = create_as_path_graph(row)

        # Detect
        features = {
            "timestamp": row["timestamp"],
            "source_id": row["source_id"],
            "graph": graph,
        }

        try:
            detection_result = detector.detect(features)

            results.append({
                "timestamp": row["timestamp"],
                "source_id": row["source_id"],
                "ground_truth": row["is_anomaly"],
                "predicted_score": detection_result["score"],
                "predicted_anomaly": detection_result["is_anomaly"],
                "confidence": detection_result["confidence"],
                "min_distance": detection_result["evidence"].get("min_distance_to_normal", 0.0),
                # Additional context
                "session_state": row["session_state"],
                "update_delta": row["update_delta"],
                "withdraw_delta": row["withdraw_delta"],
                "prefix_count": row["prefix_count"],
                "as_path_length": row["as_path_length"],
                "route_flap_count": row["route_flap_count"],
            })

        except Exception as e:
            print(f"Warning: Inference failed for row {idx}: {e}")
            results.append({
                "timestamp": row["timestamp"],
                "source_id": row["source_id"],
                "ground_truth": row["is_anomaly"],
                "predicted_score": 0.0,
                "predicted_anomaly": False,
                "confidence": 0.0,
                "min_distance": 0.0,
                "error": str(e),
            })

        if (len(results)) % 100 == 0:
            print(f"  Processed {len(results)}/{len(df)} records")

    results_df = pd.DataFrame(results)
    print(f"Inference completed: {len(results_df)} records")

    return results_df


def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate performance metrics."""
    print("\nCalculating metrics...")

    # Ground truth and predictions
    y_true = results_df["ground_truth"].astype(int).values
    y_pred = results_df["predicted_anomaly"].astype(int).values

    # Confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on BGP data using trained GNN model",
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
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to data directory or parquet file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bgp/predictions.csv"),
        help="Output predictions file (default: results/bgp/predictions.csv)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    for data_path in args.data:
        if not data_path.exists():
            print(f"ERROR: Data path not found: {data_path}")
            sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    detector = GNNDetector()

    try:
        detector.load_model(str(args.model))
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    print(f"Model loaded successfully")
    print(f"  Normal embeddings: {len(detector.normal_embeddings)}")
    print(f"  Anomaly threshold: {detector.anomaly_threshold:.4f}")

    # Load data
    try:
        df = load_data(args.data)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        sys.exit(1)

    # Run inference
    try:
        results_df = run_inference(detector, df)
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Calculate metrics
    metrics = calculate_metrics(results_df)

    # Print metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)

    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {cm['tp']:>6}")
    print(f"  True Negatives:  {cm['tn']:>6}")
    print(f"  False Positives: {cm['fp']:>6}")
    print(f"  False Negatives: {cm['fn']:>6}")

    print("\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']*100:>6.2f}%")
    print(f"  Precision: {metrics['precision']*100:>6.2f}%")
    print(f"  Recall:    {metrics['recall']*100:>6.2f}%")
    print(f"  F1-score:  {metrics['f1_score']*100:>6.2f}%")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to: {args.output}")

    # Save metrics
    metrics_path = args.output.parent / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("BGP GNN Detection Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Data: {', '.join(str(p) for p in args.data)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {cm['tp']}, TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}\n\n")

        f.write("Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {metrics['precision']*100:.2f}%\n")
        f.write(f"  Recall:    {metrics['recall']*100:.2f}%\n")
        f.write(f"  F1-score:  {metrics['f1_score']*100:.2f}%\n")

    print(f"Saved metrics to: {metrics_path}")

    print("\n✅ Inference completed successfully!")
    print(f"\nNext step:")
    print(f"  Generate report: python scripts/report_bgp.py --predictions {args.output}")


if __name__ == "__main__":
    main()
