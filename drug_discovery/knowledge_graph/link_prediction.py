"""
Link Prediction for Knowledge Graph

Uses GNNs to predict missing links in the drug discovery knowledge graph.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

logger = logging.getLogger(__name__)


class LinkPredictorGNN(nn.Module):
    """GNN model for link prediction in knowledge graphs."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """Predicts link existence between node pairs."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)


class LinkPredictionService:
    """Service to handle link prediction tasks."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        self.model = LinkPredictorGNN(in_dim, hidden_dim, out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_step(self, data: Data, optimizer: torch.optim.Optimizer):
        self.model.train()
        optimizer.zero_grad()

        z = self.model(data.x.to(self.device), data.edge_index.to(self.device))

        # Positive edges
        pos_edge_index = data.edge_index
        pos_out = self.model.decode(z, pos_edge_index)
        pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()

        # Negative edges (random sampling for simplicity here)
        neg_edge_index = torch.randint(0, data.num_nodes, pos_edge_index.size(), device=self.device)
        neg_out = self.model.decode(z, neg_edge_index)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict_links(self, x: torch.Tensor, edge_index: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of links between given node pairs.

        Args:
            x: Node features
            edge_index: Existing graph edges
            pairs: Node pairs to predict (2, num_pairs)
        """
        self.model.eval()
        with torch.no_grad():
            z = self.model(x.to(self.device), edge_index.to(self.device))
            logits = self.model.decode(z, pairs.to(self.device))
            probs = torch.sigmoid(logits)
        return probs
