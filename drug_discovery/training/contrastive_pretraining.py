"""Contrastive Self-Supervised Pretraining for ZANE.
2D-3D molecular contrastive learning (GraphMVP/MolCLR-inspired).
Ref: Liu et al. "Pre-training Molecular Graph Representation with 3D Geometry" (2022)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    hidden_dim: int = 256
    proj_dim: int = 128
    temperature: float = 0.1
    augment_node_drop: float = 0.1
    augment_edge_drop: float = 0.1
    augment_coord_noise: float = 0.05
    learning_rate: float = 1e-3


class GraphAugmentor:
    """Stochastic graph augmentations for contrastive views."""

    def __init__(self, config):
        self.config = config

    def augment(self, h, pos, edge_index):
        h = self._node_drop(h)
        edge_index = self._edge_drop(edge_index)
        pos = self._noise(pos)
        return h, pos, edge_index

    def _node_drop(self, h):
        if self.config.augment_node_drop > 0:
            return h * (torch.rand(h.size(0), 1, device=h.device) > self.config.augment_node_drop).float()
        return h

    def _edge_drop(self, ei):
        if self.config.augment_edge_drop > 0 and ei.size(1) > 0:
            return ei[:, torch.rand(ei.size(1), device=ei.device) > self.config.augment_edge_drop]
        return ei

    def _noise(self, pos):
        return (
            pos + torch.randn_like(pos) * self.config.augment_coord_noise
            if self.config.augment_coord_noise > 0
            else pos
        )


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (InfoNCE)."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i, z_j = F.normalize(z_i, -1), F.normalize(z_j, -1)
        z = torch.cat([z_i, z_j], 0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, -1e9)
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
        return F.cross_entropy(sim, labels)


class PretrainEncoder(nn.Module):
    """GNN encoder for pretraining."""

    def __init__(self, hidden_dim, num_layers=4, max_atomic=100):
        super().__init__()
        self.embed = nn.Embedding(max_atomic, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "msg": nn.Sequential(
                            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
                        ),
                        "upd": nn.Sequential(
                            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
                        ),
                        "ln": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, z, pos, edge_index):
        h = self.embed(z)
        if edge_index.size(1) > 0:
            row, col = edge_index
            for layer in self.layers:
                dist = torch.sqrt(((pos[row] - pos[col]) ** 2).sum(-1, keepdim=True) + 1e-8)
                m = layer["msg"](torch.cat([h[row], h[col], dist], -1))
                agg = torch.zeros_like(h)
                agg.index_add_(0, row, m)
                h = layer["ln"](h + layer["upd"](torch.cat([h, agg], -1)))
        return h.mean(0)


class ContrastivePretrainer(nn.Module):
    """2D-3D contrastive pretraining. Pretrain, then extract encoder for finetuning.
    Example:
        pt = ContrastivePretrainer(ContrastiveConfig(hidden_dim=256))
        loss = pt(z, pos, edge_index)  # pretrain step
        encoder = pt.get_pretrained_encoder()  # for downstream
    """

    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.encoder = PretrainEncoder(config.hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, config.proj_dim)
        )
        self.augmentor = GraphAugmentor(config)
        self.loss_fn = NTXentLoss(config.temperature)

    def forward(self, z, pos, edge_index):
        h1, p1, e1 = self.augmentor.augment(self.encoder.embed(z), pos, edge_index)
        h2, p2, e2 = self.augmentor.augment(self.encoder.embed(z), pos, edge_index)
        z1 = self._encode(h1, p1, e1)
        z2 = self._encode(h2, p2, e2)
        return self.loss_fn(self.projector(z1.unsqueeze(0)), self.projector(z2.unsqueeze(0)))

    def _encode(self, h, pos, edge_index):
        if edge_index.size(1) == 0:
            return h.mean(0)
        row, col = edge_index
        for layer in self.encoder.layers:
            dist = torch.sqrt(((pos[row] - pos[col]) ** 2).sum(-1, keepdim=True) + 1e-8)
            m = layer["msg"](torch.cat([h[row], h[col], dist], -1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, m)
            h = layer["ln"](h + layer["upd"](torch.cat([h, agg], -1)))
        return h.mean(0)

    def get_pretrained_encoder(self):
        return self.encoder
