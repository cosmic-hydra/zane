"""
E(3)-Equivariant Graph Neural Networks for 3D Molecular Modeling
Implements rotation and translation equivariant architectures
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool

logger = logging.getLogger(__name__)


class E3EquivariantGNN(nn.Module):
    """
    E(3)-Equivariant Graph Neural Network for 3D molecular structures
    Maintains equivariance under rotations and translations
    """

    def __init__(
        self,
        node_features: int = 15,
        edge_features: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        """
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node embedding
        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # Edge embedding
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        # E(3)-equivariant layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(E3EquivariantConv(hidden_dim, hidden_dim))

        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        """
        Forward pass

        Args:
            data: PyTorch Geometric data with pos (3D coordinates)

        Returns:
            Predictions
        """
        x, edge_index, edge_attr, pos, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.pos if hasattr(data, "pos") else None,
            data.batch,
        )

        # Encode features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Apply E(3)-equivariant layers
        for i, conv in enumerate(self.conv_layers):
            x_new = conv(x, edge_index, edge_attr, pos)
            x_new = self.layer_norms[i](x_new)
            x_new = F.relu(x_new)

            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Global pooling
        x = global_mean_pool(x, batch)

        # Output prediction
        out = self.mlp(x)

        return out


class E3EquivariantConv(MessagePassing):
    """
    E(3)-Equivariant Message Passing Layer
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),  # +1 for distance
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            pos: 3D positions [num_nodes, 3]

        Returns:
            Updated node features
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        pos_i: torch.Tensor | None = None,
        pos_j: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Construct messages

        Args:
            x_i: Target node features
            x_j: Source node features
            edge_attr: Edge features
            pos_i: Target positions
            pos_j: Source positions

        Returns:
            Messages
        """
        # Calculate distance if positions available
        if pos_i is not None and pos_j is not None:
            dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        else:
            dist = torch.ones(x_i.size(0), 1, device=x_i.device)

        # Concatenate features
        message_input = torch.cat([x_i, x_j, dist], dim=-1)

        # Generate message
        message = self.message_net(message_input)

        return message

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update node features

        Args:
            aggr_out: Aggregated messages
            x: Current node features

        Returns:
            Updated features
        """
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class ProteinLigandCoDesignModel(nn.Module):
    """
    3D Protein-Ligand Co-Design Model
    Jointly models protein binding pocket and ligand for optimization
    """

    def __init__(
        self,
        ligand_features: int = 15,
        protein_features: int = 20,
        hidden_dim: int = 256,
        num_layers: int = 6,
        output_dim: int = 1,
    ):
        """
        Args:
            ligand_features: Number of ligand atom features
            protein_features: Number of protein residue features
            hidden_dim: Hidden dimension
            num_layers: Number of interaction layers
            output_dim: Output dimension (e.g., binding affinity)
        """
        super().__init__()

        # Separate encoders for ligand and protein
        self.ligand_encoder = E3EquivariantGNN(
            node_features=ligand_features, hidden_dim=hidden_dim, num_layers=num_layers // 2, output_dim=hidden_dim
        )

        self.protein_encoder = E3EquivariantGNN(
            node_features=protein_features, hidden_dim=hidden_dim, num_layers=num_layers // 2, output_dim=hidden_dim
        )

        # Interaction layers
        self.interaction_layers = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers // 2)])

        # Output prediction
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, ligand_data, protein_data):
        """
        Forward pass for protein-ligand complex

        Args:
            ligand_data: Ligand graph data
            protein_data: Protein graph data

        Returns:
            Binding affinity prediction
        """
        # Encode ligand and protein separately
        ligand_features = self.ligand_encoder(ligand_data)
        protein_features = self.protein_encoder(protein_data)

        # Interaction modeling
        combined = torch.cat([ligand_features, protein_features], dim=-1)

        for layer in self.interaction_layers:
            combined = F.relu(layer(combined))

        # Predict binding affinity
        output = self.output_mlp(combined)

        return output
