"""GNN (Graph Neural Network) detector plugin for graph-based anomaly detection.

This plugin uses Graph Neural Networks to detect anomalies in network
topology and routing graphs, particularly effective for:
- BGP AS-path analysis (prefix hijacking detection)
- Network topology anomalies
- Unusual routing patterns
- Graph-level feature extraction

GNN models learn normal graph patterns (e.g., typical AS-paths) and
flag deviations such as unusual paths or hijacked prefixes.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pickle
import logging
import numpy as np

# Try to use structlog if available, otherwise fall back to standard logging
try:
    from ....core.logging import get_logger
    logger = get_logger(__name__)
    HAS_STRUCTLOG = True
except ImportError:
    logger = logging.getLogger(__name__)
    HAS_STRUCTLOG = False

# Try to import PyTorch (optional dependency)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available. GNN detector will use fallback mode.")

# NetworkX for graph structure management
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("NetworkX not available. GNN detector will not function properly.")

from ...base import DetectorPlugin


def _log(logger_instance, level: str, msg: str, **kwargs):
    """Helper to log with or without structlog."""
    if HAS_STRUCTLOG or hasattr(logger_instance, "bind"):
        getattr(logger_instance, level)(msg, **kwargs)
    else:
        if kwargs:
            formatted_msg = f"{msg} ({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        else:
            formatted_msg = msg
        getattr(logger_instance, level)(formatted_msg)


class SimpleGraphAttention(nn.Module):
    """Simplified Graph Attention Network layer.

    This is a lightweight implementation that doesn't require PyTorch Geometric.
    It uses basic PyTorch operations to implement graph attention.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        """Initialize graph attention layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout rate
        """
        super(SimpleGraphAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Linear transformation for node features
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Attention mechanism (self-attention on concatenated node pairs)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph attention layer.

        Args:
            node_features: Node feature matrix (n_nodes, in_features)
            adj_matrix: Adjacency matrix (n_nodes, n_nodes)

        Returns:
            Updated node features (n_nodes, out_features)
        """
        n_nodes = node_features.size(0)

        # Linear transformation: (n_nodes, out_features)
        h = self.W(node_features)

        # Compute attention coefficients
        # Create all pairs: (n_nodes, n_nodes, 2*out_features)
        h_i = h.unsqueeze(1).repeat(1, n_nodes, 1)  # (n_nodes, n_nodes, out_features)
        h_j = h.unsqueeze(0).repeat(n_nodes, 1, 1)  # (n_nodes, n_nodes, out_features)
        concat = torch.cat([h_i, h_j], dim=2)  # (n_nodes, n_nodes, 2*out_features)

        # Attention scores: (n_nodes, n_nodes)
        e = self.leaky_relu(self.a(concat).squeeze(2))

        # Mask attention scores with adjacency matrix
        # Set non-edges to -inf so softmax makes them 0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)

        # Softmax normalization
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Aggregate neighbor features
        h_prime = torch.matmul(attention, h)

        return h_prime


class GNNModel(nn.Module):
    """Graph Neural Network for AS-path anomaly detection.

    Architecture:
    - Graph attention layers: learn node representations
    - Global pooling: aggregate graph-level features
    - Fully connected layers: classification/anomaly scoring
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize GNN model.

        Args:
            input_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of graph attention layers
            dropout: Dropout rate
        """
        super(GNNModel, self).__init__()

        self.num_layers = num_layers

        # Graph attention layers
        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(SimpleGraphAttention(input_dim, hidden_dim, dropout))

        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(SimpleGraphAttention(hidden_dim, hidden_dim, dropout))

        # Last layer
        if num_layers > 1:
            self.gat_layers.append(SimpleGraphAttention(hidden_dim, output_dim, dropout))

        # Output layer (for graph-level embedding)
        self.fc = nn.Linear(output_dim if num_layers > 1 else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN.

        Args:
            node_features: Node feature matrix (n_nodes, input_dim)
            adj_matrix: Adjacency matrix (n_nodes, n_nodes)

        Returns:
            Graph-level embedding (output_dim,)
        """
        x = node_features

        # Apply graph attention layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, adj_matrix)
            if i < len(self.gat_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        # Global mean pooling (graph-level)
        graph_embedding = torch.mean(x, dim=0)

        # Final transformation
        output = self.fc(graph_embedding)

        return output


class GNNDetector(DetectorPlugin):
    """GNN-based anomaly detector for graph patterns.

    This detector uses Graph Neural Networks to learn normal graph
    patterns and detect anomalies. It's particularly effective for:

    - BGP AS-path analysis (prefix hijacking)
    - Unusual routing paths
    - Network topology changes
    - Graph-level anomaly detection

    The detector works by:
    1. Training: Learn embeddings of normal AS-path graphs
    2. Detection: Compare new graphs to learned normal patterns
    3. Scoring: Distance from normal pattern indicates anomaly

    Hyperparameters:
    - hidden_dim: Hidden layer dimension (default: 64)
    - num_layers: Number of graph layers (default: 2)
    - threshold_multiplier: Multiplier for anomaly threshold (default: 3.0)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        threshold_multiplier: float = 3.0,
    ):
        """Initialize GNN detector.

        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of graph attention layers
            threshold_multiplier: Multiplier for anomaly threshold
        """
        if not HAS_PYTORCH:
            logger.warning(
                "PyTorch not available. GNN detector will use fallback mode. "
                "Install with: pip install torch"
            )

        if not HAS_NETWORKX:
            raise ImportError(
                "NetworkX is required for GNN detector. "
                "Install with: pip install networkx"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.threshold_multiplier = threshold_multiplier

        # Model components
        self.model: Optional[GNNModel] = None
        self.normal_embeddings: List[np.ndarray] = []  # Store normal graph embeddings
        self.anomaly_threshold: float = 0.0

        # Training history
        self.is_trained: bool = False

        # Use bind() if available (structlog), otherwise use base logger
        self.logger = logger.bind(component="gnn_detector") if hasattr(logger, "bind") else logger

    @property
    def name(self) -> str:
        return "gnn"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        return ["bgp"]

    def _graph_to_tensors(
        self, graph: nx.Graph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert NetworkX graph to PyTorch tensors.

        Args:
            graph: NetworkX graph with node features

        Returns:
            Tuple of (node_features, adjacency_matrix)
        """
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch is required for GNN operations")

        # Get node features (assumes 'features' attribute on nodes)
        node_features = []
        for node in graph.nodes():
            features = graph.nodes[node].get('features', [0.0, 0.0, 0.0])
            if not isinstance(features, list):
                features = [float(features), 0.0, 0.0]
            node_features.append(features)

        node_features_tensor = torch.FloatTensor(node_features)

        # Get adjacency matrix
        adj_matrix = nx.to_numpy_array(graph)
        adj_matrix_tensor = torch.FloatTensor(adj_matrix)

        return node_features_tensor, adj_matrix_tensor

    def train(self, data: Any, **kwargs) -> None:
        """Train GNN model on normal graph patterns.

        Args:
            data: Training data, either:
                - List of NetworkX graphs
                - List of tuples (node_features, adj_matrix)
            **kwargs: Training parameters:
                - epochs (int): Number of training epochs (default: 100)
                - learning_rate (float): Learning rate (default: 0.001)

        Raises:
            ValueError: If data format is invalid
        """
        if not HAS_PYTORCH:
            _log(
                self.logger,
                "warning",
                "PyTorch not available. Using fallback mode (no training).",
            )
            self.is_trained = True
            self.anomaly_threshold = 1.0
            return

        epochs = kwargs.get("epochs", 100)
        learning_rate = kwargs.get("learning_rate", 0.001)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Training data must be a non-empty list of graphs")

        _log(
            self.logger,
            "info",
            "Starting GNN training",
            n_graphs=len(data),
            epochs=epochs,
        )

        # Convert graphs to tensors if needed
        graph_data = []
        for item in data:
            if isinstance(item, nx.Graph):
                node_features, adj_matrix = self._graph_to_tensors(item)
                graph_data.append((node_features, adj_matrix))
            elif isinstance(item, tuple) and len(item) == 2:
                graph_data.append(item)
            else:
                raise ValueError(f"Unsupported data type: {type(item)}")

        # Infer input dimension from first graph
        first_node_features, _ = graph_data[0]
        input_dim = first_node_features.shape[1]

        # Initialize model
        self.model = GNNModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=32,  # Fixed output dimension
            num_layers=self.num_layers,
        )

        # Contrastive learning: minimize distance between normal graph embeddings
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0

            # Sample pairs of graphs for contrastive learning
            for i in range(len(graph_data)):
                node_features, adj_matrix = graph_data[i]

                # Get embedding
                embedding = self.model(node_features, adj_matrix)

                # Sample another graph for comparison
                j = (i + 1) % len(graph_data)
                node_features_j, adj_matrix_j = graph_data[j]
                embedding_j = self.model(node_features_j, adj_matrix_j)

                # Contrastive loss: minimize distance between normal graphs
                loss = F.mse_loss(embedding, embedding_j)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(graph_data)

            if (epoch + 1) % 20 == 0:
                _log(
                    self.logger,
                    "info",
                    "Training progress",
                    epoch=epoch + 1,
                    loss=avg_loss,
                )

        # Store normal embeddings for anomaly detection
        self.model.eval()
        self.normal_embeddings = []

        with torch.no_grad():
            for node_features, adj_matrix in graph_data:
                embedding = self.model(node_features, adj_matrix)
                self.normal_embeddings.append(embedding.numpy())

        # Calculate anomaly threshold (mean + threshold_multiplier * std of pairwise distances)
        distances = []
        for i in range(len(self.normal_embeddings)):
            for j in range(i + 1, len(self.normal_embeddings)):
                dist = np.linalg.norm(self.normal_embeddings[i] - self.normal_embeddings[j])
                distances.append(dist)

        if distances:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            self.anomaly_threshold = mean_dist + self.threshold_multiplier * std_dist
        else:
            self.anomaly_threshold = 1.0

        self.is_trained = True

        _log(
            self.logger,
            "info",
            "Training completed",
            n_normal_embeddings=len(self.normal_embeddings),
            anomaly_threshold=self.anomaly_threshold,
        )

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in graph features.

        Args:
            features: Feature dictionary containing:
                - timestamp (datetime): Current timestamp
                - source_id (str): Endpoint identifier
                - graph (nx.Graph): NetworkX graph to analyze
                OR
                - node_features (torch.Tensor): Node features
                - adj_matrix (torch.Tensor): Adjacency matrix

        Returns:
            Detection result dictionary:
                - score (float): Anomaly score (0.0-1.0)
                - is_anomaly (bool): Binary classification
                - confidence (float): Detection confidence
                - evidence (dict): Evidence details
                - detector_name (str): "gnn"
        """
        if not self.is_trained:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "evidence": {"error": "Model not trained"},
                "detector_name": self.name,
            }

        if not HAS_PYTORCH:
            # Fallback mode: simple heuristic
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "evidence": {"warning": "PyTorch not available, using fallback mode"},
                "detector_name": self.name,
            }

        # Get graph data
        if "graph" in features:
            node_features, adj_matrix = self._graph_to_tensors(features["graph"])
        elif "node_features" in features and "adj_matrix" in features:
            node_features = features["node_features"]
            adj_matrix = features["adj_matrix"]
        else:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "evidence": {"error": "Missing graph data"},
                "detector_name": self.name,
            }

        # Get embedding
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(node_features, adj_matrix)
            embedding_np = embedding.numpy()

        # Calculate minimum distance to normal embeddings
        if not self.normal_embeddings:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "evidence": {"error": "No normal embeddings stored"},
                "detector_name": self.name,
            }

        min_distance = min(
            np.linalg.norm(embedding_np - normal_emb)
            for normal_emb in self.normal_embeddings
        )

        # Calculate anomaly score
        if self.anomaly_threshold > 0:
            score = min(1.0, min_distance / self.anomaly_threshold)
        else:
            score = 0.0

        is_anomaly = min_distance > self.anomaly_threshold

        return {
            "score": score,
            "is_anomaly": is_anomaly,
            "confidence": min(0.95, score) if is_anomaly else 0.5,
            "evidence": {
                "min_distance_to_normal": float(min_distance),
                "anomaly_threshold": float(self.anomaly_threshold),
                "n_normal_patterns": len(self.normal_embeddings),
            },
            "detector_name": self.name,
        }

    def save_model(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: File path to save model (will create parent directories)

        Raises:
            IOError: If save fails
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and metadata
        save_dict = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "threshold_multiplier": self.threshold_multiplier,
            "normal_embeddings": self.normal_embeddings,
            "anomaly_threshold": self.anomaly_threshold,
            "version": self.version,
        }

        if HAS_PYTORCH and self.model is not None:
            save_dict["model_state_dict"] = self.model.state_dict()

        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

        _log(
            self.logger,
            "info",
            "Model saved",
            path=str(save_path),
        )

    def load_model(self, path: str) -> None:
        """Load trained model from disk.

        Args:
            path: File path to load model from

        Raises:
            IOError: If load fails
            ValueError: If model format is invalid
        """
        load_path = Path(path)
        if not load_path.exists():
            raise IOError(f"Model file not found: {path}")

        with open(load_path, 'rb') as f:
            save_dict = pickle.load(f)

        # Restore hyperparameters
        self.hidden_dim = save_dict["hidden_dim"]
        self.num_layers = save_dict["num_layers"]
        self.threshold_multiplier = save_dict["threshold_multiplier"]
        self.normal_embeddings = save_dict["normal_embeddings"]
        self.anomaly_threshold = save_dict["anomaly_threshold"]

        # Reconstruct model if PyTorch is available
        if HAS_PYTORCH and "model_state_dict" in save_dict:
            # Infer input_dim from first normal embedding
            if self.normal_embeddings:
                output_dim = len(self.normal_embeddings[0])
            else:
                output_dim = 32

            self.model = GNNModel(
                input_dim=3,  # Default BGP features
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                num_layers=self.num_layers,
            )

            self.model.load_state_dict(save_dict["model_state_dict"])
            self.model.eval()

        self.is_trained = True

        _log(
            self.logger,
            "info",
            "Model loaded",
            path=str(load_path),
            version=save_dict.get("version", "unknown"),
        )


def create_detector() -> DetectorPlugin:
    """Plugin entry point.

    Returns:
        GNN detector instance
    """
    return GNNDetector()
