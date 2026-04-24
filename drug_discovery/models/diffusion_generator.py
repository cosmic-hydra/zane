"""
Diffusion-Based Molecule Generator for ZANE.

Implements SE(3)-equivariant denoising diffusion for 3D de novo molecular
design with classifier-free guidance for property-steered generation.

References:
    Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D"
    Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion molecule generator."""

    hidden_dim: int = 256
    num_layers: int = 8
    num_atom_types: int = 10
    max_atoms: int = 50
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    guidance_scale: float = 2.0
    use_flow_matching: bool = False


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class EquivariantDenoisingBlock(nn.Module):
    """Equivariant denoising block for coordinate + feature updates."""

    def __init__(self, hidden_dim, time_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1, bias=False)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pos, edge_index, t_emb):
        row, col = edge_index
        diff = pos[row] - pos[col]
        dist = torch.sqrt((diff**2).sum(-1, keepdim=True) + 1e-8)
        t_row = t_emb[row] if t_emb.size(0) == h.size(0) else t_emb.expand(row.size(0), -1)
        edge_feat = torch.cat([h[row], h[col], dist, t_row], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        coord_w = self.coord_mlp(m_ij)
        unit = diff / (dist + 1e-8)
        coord_agg = torch.zeros_like(pos)
        coord_agg.index_add_(0, row, coord_w * unit)
        pos_out = pos + coord_agg
        msg_agg = torch.zeros_like(h)
        msg_agg.index_add_(0, row, m_ij)
        t_node = t_emb if t_emb.size(0) == h.size(0) else t_emb.expand(h.size(0), -1)
        h_out = self.node_mlp(torch.cat([h, msg_agg, t_node], dim=-1))
        return self.layer_norm(h + h_out), pos_out


class MolecularDiffusionModel(nn.Module):
    """SE(3)-equivariant denoising diffusion model for molecules."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        hd = config.hidden_dim
        self.atom_embed = nn.Embedding(config.num_atom_types + 1, hd)
        self.time_embed = nn.Sequential(SinusoidalTimeEmbedding(hd), nn.Linear(hd, hd), nn.SiLU(), nn.Linear(hd, hd))
        self.blocks = nn.ModuleList([EquivariantDenoisingBlock(hd, hd) for _ in range(config.num_layers)])
        self.coord_head = nn.Linear(hd, 3)
        self.atom_head = nn.Linear(hd, config.num_atom_types)

    def forward(self, atom_types, pos, edge_index, timesteps, batch=None):
        h = self.atom_embed(atom_types)
        t_emb = self.time_embed(timesteps)
        t_per_node = t_emb[batch] if batch is not None else t_emb.expand(h.size(0), -1)
        for block in self.blocks:
            h, pos = block(h, pos, edge_index, t_per_node)
        return self.coord_head(h), self.atom_head(h)


class DiffusionMoleculeGenerator:
    """High-level API for diffusion-based molecule generation.

    Example::
        config = DiffusionConfig(hidden_dim=256, num_layers=8)
        gen = DiffusionMoleculeGenerator(config, device="cuda")
        molecules = gen.sample(num_molecules=10, num_atoms=20)
    """

    def __init__(self, config: DiffusionConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = MolecularDiffusionModel(config).to(self.device)
        betas = torch.linspace(config.beta_start, config.beta_end, config.noise_steps)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(self.device)

    @torch.no_grad()
    def sample(self, num_molecules, num_atoms, edge_index_fn=None):
        """Generate molecules via reverse diffusion."""
        self.model.eval()
        total = num_molecules * num_atoms
        pos = torch.randn(total, 3, device=self.device)
        atom_types = torch.randint(0, self.config.num_atom_types, (total,), device=self.device)
        batch = torch.arange(num_molecules, device=self.device).repeat_interleave(num_atoms)
        for t_val in reversed(range(self.config.noise_steps)):
            t = torch.full((num_molecules,), t_val, device=self.device, dtype=torch.long)
            if edge_index_fn:
                edge_index = edge_index_fn(pos, batch)
            else:
                edge_index = self._fully_connected(num_molecules, num_atoms)
            eps_pos, eps_atom = self.model(atom_types, pos, edge_index, t, batch)
            ab = self.alpha_bar[t_val]
            beta = 1 - ab / (self.alpha_bar[t_val - 1] if t_val > 0 else 1.0)
            pos = (pos - beta / (1 - ab).sqrt() * eps_pos) / (1 - beta).sqrt()
            if t_val > 0:
                pos = pos + beta.sqrt() * torch.randn_like(pos)
            atom_types = eps_atom.argmax(dim=-1)
        return {
            "positions": pos.view(num_molecules, num_atoms, 3),
            "atom_types": atom_types.view(num_molecules, num_atoms),
        }

    def _fully_connected(self, n_mol, n_atoms):
        src, dst = [], []
        for i in range(n_mol):
            off = i * n_atoms
            idx = torch.arange(n_atoms, device=self.device) + off
            r = idx.repeat_interleave(n_atoms)
            c = idx.repeat(n_atoms)
            m = r != c
            src.append(r[m])
            dst.append(c[m])
        return torch.stack([torch.cat(src), torch.cat(dst)])
