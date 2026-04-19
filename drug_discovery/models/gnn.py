"""
Graph Neural Network Models for Molecular Property Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing, global_max_pool, global_mean_pool


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction
    Uses Graph Attention Networks (GAT) for state-of-the-art performance
    """

    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        output_dim: int = 1,
        pooling: str = "mean",  # 'mean', 'max', or 'attention'
        input_dim: int | None = None,
        **_: object,
    ):
        """
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_dim: Output dimension
            pooling: Graph pooling method
        """
        super().__init__()

        resolved_nodes = input_dim if input_dim is not None else node_features
        self.node_features = resolved_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling

        # Initial node embedding
        self.node_encoder = nn.Linear(resolved_nodes, hidden_dim)

        # Edge embedding
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=hidden_dim)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=hidden_dim)
                )

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Attention pooling
        if pooling == "attention":
            self.attention_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass

        Args:
            data: PyTorch Geometric batch

        Returns:
            Predictions
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encode nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x_new = gat_layer(x, edge_index, edge_attr)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Graph pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "attention":
            # Attention-based pooling
            attention_weights = self.attention_pool(x)
            attention_weights = torch.softmax(attention_weights, dim=0)
            x = global_mean_pool(x * attention_weights, batch)

        # MLP for final prediction
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network Layer
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim)
        )

        self.update_net = nn.Sequential(nn.Linear(2 * node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target nodes, x_j: source nodes
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(message_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class MolecularMPNN(nn.Module):
    """
    Message Passing Neural Network for molecular graphs
    """

    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        self.mpnn_layers = nn.ModuleList([MPNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for i, mpnn_layer in enumerate(self.mpnn_layers):
            x = mpnn_layer(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
