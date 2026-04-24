"""
GFlowNet-Based Molecular Generator for ZANE.
Generative Flow Networks for diverse, reward-proportional molecular sampling.
References: Bengio et al. "GFlowNet Foundations" (JMLR 2023)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GFlowNetConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    max_atoms: int = 38
    atom_vocab: int = 10
    bond_types: int = 4
    temperature: float = 1.0
    reward_exponent: float = 2.0
    learning_rate: float = 5e-4


class GFlowNetPolicy(nn.Module):
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        h = config.hidden_dim
        self.atom_embed = nn.Embedding(config.atom_vocab + 1, h)
        self.graph_encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "msg": nn.Sequential(nn.Linear(2 * h + 1, h), nn.SiLU(), nn.Linear(h, h)),
                        "upd": nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h)),
                        "ln": nn.LayerNorm(h),
                    }
                )
                for _ in range(config.num_layers)
            ]
        )
        self.add_atom_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, config.atom_vocab))
        self.stop_head = nn.Sequential(nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1))
        self.config = config

    def forward(self, state):
        z, edge_index = state["atoms"], state["edge_index"]
        h = self.atom_embed(z)
        if edge_index.numel() > 0:
            row, col = edge_index
            for layer in self.graph_encoder:
                m = layer["msg"](torch.cat([h[row], h[col], torch.ones(row.size(0), 1, device=h.device)], -1))
                agg = torch.zeros_like(h)
                agg.index_add_(0, row, m)
                h = layer["ln"](h + layer["upd"](torch.cat([h, agg], -1)))
        g = h.mean(0) if h.size(0) > 0 else torch.zeros(self.config.hidden_dim, device=h.device)
        return {
            "atom_logits": self.add_atom_head(g) / self.config.temperature,
            "stop_logit": self.stop_head(g),
            "graph_embedding": g,
        }


class GFlowNetBackwardPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        self.net = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, g):
        return self.net(g)


class GFlowNetTrainer:
    """Trajectory Balance training for GFlowNet molecular generation.
    Example: trainer = GFlowNetTrainer(config, reward_fn=fn, device="cuda")
    """

    def __init__(self, config, reward_fn=None, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.forward_policy = GFlowNetPolicy(config).to(self.device)
        self.backward_policy = GFlowNetBackwardPolicy(config).to(self.device)
        self.log_Z = nn.Parameter(torch.zeros(1, device=self.device))
        self.reward_fn = reward_fn or (lambda m: 1.0)
        self.optimizer = torch.optim.Adam(
            list(self.forward_policy.parameters()) + list(self.backward_policy.parameters()) + [self.log_Z],
            lr=config.learning_rate,
        )

    def train_step(self):
        self.optimizer.zero_grad()
        traj, log_pf, log_pb = self._sample_trajectory()
        r = self.reward_fn(traj)
        loss = (self.log_Z + log_pf - log_pb - math.log(max(r, 1e-8)) * self.config.reward_exponent) ** 2
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _sample_trajectory(self):
        atoms = torch.zeros(0, dtype=torch.long, device=self.device)
        edges = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        log_pf = torch.tensor(0.0, device=self.device)
        log_pb = torch.tensor(0.0, device=self.device)
        for step in range(self.config.max_atoms):
            out = self.forward_policy({"atoms": atoms, "edge_index": edges})
            sp = torch.sigmoid(out["stop_logit"])
            if step > 0 and torch.rand(1, device=self.device) < sp:
                log_pf = log_pf + torch.log(sp + 1e-8)
                break
            if step > 0:
                log_pf = log_pf + torch.log(1 - sp + 1e-8)
            probs = F.softmax(out["atom_logits"], -1)
            atom = torch.multinomial(probs, 1)
            log_pf = log_pf + torch.log(probs[atom] + 1e-8)
            atoms = torch.cat([atoms, atom.view(-1)])
            if atoms.size(0) > 1:
                ni = atoms.size(0) - 1
                edges = torch.cat([edges, torch.tensor([[ni, ni - 1], [ni - 1, ni]], device=self.device)], 1)
            log_pb = log_pb + self.backward_policy(out["graph_embedding"]).squeeze()
        return {"atoms": atoms, "edges": edges, "num_atoms": atoms.size(0)}, log_pf, log_pb

    @torch.no_grad()
    def sample(self, n=10):
        self.forward_policy.eval()
        results = []
        for _ in range(n):
            t, _, _ = self._sample_trajectory()
            results.append({"atoms": t["atoms"].cpu().numpy(), "num_atoms": t["num_atoms"]})
        self.forward_policy.train()
        return results
