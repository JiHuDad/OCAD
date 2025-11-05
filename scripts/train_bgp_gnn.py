#!/usr/bin/env python3
"""Train GNN model for BGP anomaly detection.

This script trains a Graph Neural Network (GNN) to detect BGP anomalies
by learning normal AS-path patterns.

Usage:
    python scripts/train_bgp_gnn.py --data data/bgp/train

    # Custom configuration
    python scripts/train_bgp_gnn.py --data data/bgp/train --epochs 200 --output models/bgp/gnn_v1.0.0.pth
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List

# Add OCAD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX not installed. Install with: pip install networkx")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

from ocad.plugins.detectors.gnn import GNNDetector


def create_as_path_graph(row: pd.Series) -> nx.Graph:
    """Create an AS-path graph from a BGP session row.

    This creates a simple graph where:
    - Nodes represent AS numbers
    - Edges represent AS-path connections
    - Node features include AS number, update count, prefix count

    Args:
        row: DataFrame row with BGP metrics

    Returns:
        NetworkX graph representing the AS-path
    """
    graph = nx.Graph()

    local_asn = row["local_asn"]
    remote_asn = row["remote_asn"]

    # Add nodes with features
    # Features: [normalized_asn, update_rate, prefix_count]
    update_rate = row["update_delta"] / 100.0  # Normalize to 0-1
    prefix_count = row["prefix_count"] / 200.0  # Normalize to 0-1

    graph.add_node(local_asn, features=[
        local_asn / 70000.0,  # Normalize AS number
        update_rate,
        prefix_count,
    ])

    graph.add_node(remote_asn, features=[
        remote_asn / 70000.0,
        update_rate,
        prefix_count,
    ])

    # Add intermediate AS hops (simulated from as_path_length)
    path_length = int(row["as_path_length"])

    # Create path: local -> intermediate_1 -> ... -> remote
    prev_as = local_asn
    for i in range(1, path_length):
        if i == path_length - 1:
            # Last hop is remote_asn
            intermediate_as = remote_asn
        else:
            # Intermediate AS (simulated)
            intermediate_as = 64000 + (local_asn + remote_asn + i) % 1000

            if intermediate_as not in graph:
                graph.add_node(intermediate_as, features=[
                    intermediate_as / 70000.0,
                    update_rate * 0.5,  # Lower update rate for intermediate
                    prefix_count * 0.5,
                ])

        # Add edge
        graph.add_edge(prev_as, intermediate_as)
        prev_as = intermediate_as

    return graph


def load_training_data(data_path: Path) -> List[nx.Graph]:
    """Load training data and convert to AS-path graphs.

    Args:
        data_path: Path to training data directory or parquet file

    Returns:
        List of NetworkX graphs
    """
    print(f"Loading training data from: {data_path}")

    # Load parquet files
    if data_path.is_dir():
        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_path}")

        print(f"Found {len(parquet_files)} parquet file(s)")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_parquet(data_path)

    print(f"Loaded {len(df)} records")

    # Filter normal data only (for training)
    df_normal = df[df["is_anomaly"] == False].copy()
    print(f"Using {len(df_normal)} normal records for training")

    # Sample to reduce size if too large (GNN training is expensive)
    max_graphs = 1000
    if len(df_normal) > max_graphs:
        print(f"Sampling {max_graphs} records to reduce training time...")
        df_normal = df_normal.sample(n=max_graphs, random_state=42)

    # Convert to graphs
    print("Converting to AS-path graphs...")
    graphs = []
    for idx, row in df_normal.iterrows():
        graph = create_as_path_graph(row)
        graphs.append(graph)

        if (len(graphs)) % 100 == 0:
            print(f"  Converted {len(graphs)}/{len(df_normal)} graphs")

    print(f"Created {len(graphs)} AS-path graphs")
    return graphs


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN model for BGP anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data directory or parquet file",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden layer dimension (default: 64)",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GNN layers (default: 2)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/bgp/gnn_v1.0.0.pth"),
        help="Output model path (default: models/bgp/gnn_v1.0.0.pth)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.data.exists():
        print(f"ERROR: Data path not found: {args.data}")
        sys.exit(1)

    # Load data
    try:
        graphs = load_training_data(args.data)
    except Exception as e:
        print(f"ERROR: Failed to load training data: {e}")
        sys.exit(1)

    if len(graphs) == 0:
        print("ERROR: No graphs to train on")
        sys.exit(1)

    # Create detector
    print("\nInitializing GNN detector...")
    detector = GNNDetector(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    # Train
    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")

    start_time = datetime.now()

    try:
        detector.train(
            data=graphs,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\nTraining completed in {duration:.1f}s")

    # Save model
    print(f"\nSaving model to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        detector.save_model(str(args.output))
    except Exception as e:
        print(f"ERROR: Failed to save model: {e}")
        sys.exit(1)

    # Model info
    print("\n" + "="*60)
    print("MODEL INFO")
    print("="*60)
    print(f"Model: GNN v{detector.version}")
    print(f"Path: {args.output}")
    print(f"Training graphs: {len(graphs)}")
    print(f"Normal embeddings: {len(detector.normal_embeddings)}")
    print(f"Anomaly threshold: {detector.anomaly_threshold:.4f}")
    print(f"Training time: {duration:.1f}s")

    print("\n✅ Training completed successfully!")
    print(f"\nNext steps:")
    print(f"  1. Test inference: python scripts/infer_bgp.py --model {args.output} --data data/bgp/val_normal")
    print(f"  2. Full evaluation: python scripts/infer_bgp.py --model {args.output} --data data/bgp/val_normal data/bgp/val_anomaly")


if __name__ == "__main__":
    main()
