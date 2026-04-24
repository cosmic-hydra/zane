"""D-MPNN (Directed Message Passing Neural Network) for ZANE.
Chemprop-style architecture: bond-centric directed messages, attentive readout.
Ref: Yang et al. "Analyzing Learned Molecular Representations" (JCIM 2019)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DMPNNConfig:
    atom_fdim: int = 133
    bond_fdim: int = 14
    hidden_dim: int = 300
    depth: int = 8
    dropout: float = 0.0
    output_dim: int = 1
    num_tasks: int = 1
    aggregation: str = "mean"
    bias: bool = False


class DMPNNEncoder(nn.Module):
    """Bond-centric directed message passing encoder."""

    def __init__(self, config: DMPNNConfig):
        super().__init__()
        c = config
        self.config = c
        self.W_i = nn.Linear(c.atom_fdim + c.bond_fdim, c.hidden_dim, bias=c.bias)
        self.W_h = nn.Linear(c.hidden_dim, c.hidden_dim, bias=c.bias)
        self.W_o = nn.Linear(c.atom_fdim + c.hidden_dim, c.hidden_dim)
        self.depth = c.depth
        self.dropout = nn.Dropout(c.dropout)
        if c.aggregation == "attention":
            self.attn_w = nn.Linear(c.hidden_dim, 1)

    def forward(self, atom_feats, bond_feats, a2b, b2a, b2revb, batch=None):
        input_feats = torch.cat([atom_feats[b2a], bond_feats], dim=-1)
        message = F.relu(self.W_i(input_feats))
        for _ in range(self.depth - 1):
            nei_msg = self._aggregate_neighbors(message, a2b, b2a, b2revb)
            message = F.relu(self.W_h(nei_msg) + self.W_i(input_feats))
            message = self.dropout(message)
        atom_hidden = self._bond_to_atom(message, a2b, atom_feats.size(0))
        atom_out = F.relu(self.W_o(torch.cat([atom_feats, atom_hidden], -1)))
        return self._readout(atom_out, batch)

    def _aggregate_neighbors(self, message, a2b, b2a, b2revb):
        nei = torch.zeros_like(message)
        for i in range(message.size(0)):
            src = b2a[i].item()
            bonds = a2b[src]
            valid = bonds[(bonds >= 0) & (bonds != b2revb[i])]
            if len(valid) > 0:
                nei[i] = message[valid].sum(0)
        return nei

    def _bond_to_atom(self, message, a2b, n_atoms):
        h = torch.zeros(n_atoms, message.size(-1), device=message.device)
        for a in range(n_atoms):
            valid = a2b[a][a2b[a] >= 0]
            if len(valid) > 0:
                h[a] = message[valid].sum(0)
        return h

    def _readout(self, atom_out, batch):
        if batch is None:
            return atom_out.mean(0, keepdim=True)
        ng = batch.max().item() + 1
        pooled = torch.zeros(ng, atom_out.size(-1), device=atom_out.device)
        counts = torch.zeros(ng, 1, device=atom_out.device)
        pooled.index_add_(0, batch, atom_out)
        counts.index_add_(0, batch, torch.ones(atom_out.size(0), 1, device=atom_out.device))
        return pooled / counts.clamp(min=1) if self.config.aggregation == "mean" else pooled


class DMPNN(nn.Module):
    """Full D-MPNN model. Example: model = DMPNN(DMPNNConfig(depth=8))"""

    def __init__(self, config: DMPNNConfig):
        super().__init__()
        self.encoder = DMPNNEncoder(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim * config.num_tasks),
        )

    def forward(self, atom_feats, bond_feats, a2b, b2a, b2revb, batch=None):
        return self.ffn(self.encoder(atom_feats, bond_feats, a2b, b2a, b2revb, batch))
