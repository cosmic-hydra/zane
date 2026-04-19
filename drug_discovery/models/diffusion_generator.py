"""
Diffusion-Based Molecule Generator for ZANE.

Implements SE(3)-equivariant denoising diffusion for 3D de novo molecular
design with classifier-free guidance for property-steered generation.

References:
    Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D"
    Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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
    protein_dim: int = 256
    pocket_dim: int = 256
    cross_attention_heads: int = 4
    use_film_conditioning: bool = True
    uncond_dropout_prob: float = 0.1


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
            nn.Linear(2 * hidden_dim + 1 + time_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False))
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pos, edge_index, t_emb):
        row, col = edge_index
        diff = pos[row] - pos[col]
        dist = torch.sqrt((diff ** 2).sum(-1, keepdim=True) + 1e-8)
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


def _dense_from_batch(features, batch):
    """Convert ragged batch to dense tensor with padding mask."""
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    counts = torch.bincount(batch, minlength=batch_size)
    max_nodes = int(counts.max().item()) if counts.numel() > 0 else 0
    dense = features.new_zeros((batch_size, max_nodes, features.size(-1)))
    padding_mask = torch.ones((batch_size, max_nodes), device=features.device, dtype=torch.bool)
    cursor = torch.zeros((batch_size,), device=features.device, dtype=torch.long)
    for idx, b in enumerate(batch):
        pos = cursor[b].item()
        dense[b, pos] = features[idx]
        padding_mask[b, pos] = False
        cursor[b] += 1
    return dense, padding_mask


def _flatten_from_dense(dense, padding_mask):
    """Flatten dense batch back to ragged representation."""
    keep = ~padding_mask
    flat = dense[keep]
    batch = []
    for b_idx in range(dense.size(0)):
        n = int(keep[b_idx].sum().item())
        batch.extend([b_idx] * n)
    batch_tensor = torch.tensor(batch, device=dense.device, dtype=torch.long)
    return flat, batch_tensor


class ProteinConditioning(nn.Module):
    """Protein/pocket conditioning via cross-attention + FiLM."""

    def __init__(self, hidden_dim: int, protein_dim: int, heads: int = 4, use_film: bool = True):
        super().__init__()
        self.use_film = use_film
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU()) if use_film else None

    def forward(self, ligand_h, protein_context, ligand_mask=None, protein_mask=None):
        protein_ctx = self.protein_proj(protein_context)
        attn_out, _ = self.attn(
            ligand_h, protein_ctx, protein_ctx, key_padding_mask=protein_mask, need_weights=False
        )
        h = self.norm(ligand_h + attn_out)
        if self.use_film:
            pooled = torch.mean(protein_ctx.masked_fill(protein_mask.unsqueeze(-1), 0.0), dim=1)
            film_params = self.film(pooled)
            gamma, beta = film_params.chunk(2, dim=-1)
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return h


class MolecularDiffusionModel(nn.Module):
    """SE(3)-equivariant denoising diffusion model for molecules."""
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        hd = config.hidden_dim
        self.atom_embed = nn.Embedding(config.num_atom_types + 1, hd)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hd), nn.Linear(hd, hd), nn.SiLU(), nn.Linear(hd, hd))
        self.blocks = nn.ModuleList([EquivariantDenoisingBlock(hd, hd) for _ in range(config.num_layers)])
        self.coord_head = nn.Linear(hd, 3)
        self.atom_head = nn.Linear(hd, config.num_atom_types)
        self.conditioner = ProteinConditioning(
            hidden_dim=hd,
            protein_dim=config.protein_dim,
            heads=config.cross_attention_heads,
            use_film=config.use_film_conditioning,
        )

    def forward(self, atom_types, pos, edge_index, timesteps, batch=None, protein_context=None, protein_mask=None):
        h = self.atom_embed(atom_types)
        t_emb = self.time_embed(timesteps)
        t_per_node = t_emb[batch] if batch is not None else t_emb.expand(h.size(0), -1)

        if protein_context is not None and batch is not None:
            dense_h, padding_mask = _dense_from_batch(h, batch)
            dense_pos, _ = _dense_from_batch(pos, batch)
            conditioned = self.conditioner(
                dense_h,
                protein_context,
                ligand_mask=padding_mask,
                protein_mask=protein_mask if protein_mask is not None else torch.zeros(
                    protein_context.size()[:2], dtype=torch.bool, device=protein_context.device
                ),
            )
            h, batch = _flatten_from_dense(conditioned, padding_mask)
            pos, _ = _flatten_from_dense(dense_pos, padding_mask)
            t_per_node = t_emb[batch]

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
        self._uncond_prob = float(config.uncond_dropout_prob)

    def _prepare_condition(self, protein_context, num_molecules: int):
        if protein_context is None:
            return None, None
        ctx = torch.as_tensor(protein_context, device=self.device, dtype=torch.float32)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(num_molecules, -1, -1)
        elif ctx.dim() == 3 and ctx.size(0) != num_molecules:
            ctx = ctx.expand(num_molecules, -1, -1)
        protein_mask = torch.zeros(ctx.shape[:2], dtype=torch.bool, device=self.device)
        return ctx, protein_mask

    @torch.no_grad()
    def sample(self, num_molecules, num_atoms, edge_index_fn=None, protein_context=None, protein_mask=None, guidance_scale: float | None = None):
        """Generate molecules via reverse diffusion."""
        self.model.eval()
        total = num_molecules * num_atoms
        pos = torch.randn(total, 3, device=self.device)
        atom_types = torch.randint(0, self.config.num_atom_types, (total,), device=self.device)
        batch = torch.arange(num_molecules, device=self.device).repeat_interleave(num_atoms)
        cond_ctx, cond_mask = self._prepare_condition(protein_context, num_molecules)
        if protein_mask is not None:
            cond_mask = protein_mask.to(self.device)
        g_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        for t_val in reversed(range(self.config.noise_steps)):
            t = torch.full((num_molecules,), t_val, device=self.device, dtype=torch.long)
            if edge_index_fn:
                edge_index = edge_index_fn(pos, batch)
            else:
                edge_index = self._fully_connected(num_molecules, num_atoms)
            cond_pos, cond_atom = self.model(
                atom_types,
                pos,
                edge_index,
                t,
                batch,
                protein_context=cond_ctx,
                protein_mask=cond_mask,
            )
            if cond_ctx is not None:
                with torch.no_grad():
                    if torch.rand(1, device=self.device).item() < self._uncond_prob:
                        eff_cond_ctx = None
                        eff_cond_mask = None
                    else:
                        eff_cond_ctx = cond_ctx
                        eff_cond_mask = cond_mask
                uncond_pos, uncond_atom = self.model(
                    atom_types,
                    pos,
                    edge_index,
                    t,
                    batch,
                    protein_context=eff_cond_ctx,
                    protein_mask=eff_cond_mask,
                )
                eps_pos = uncond_pos + g_scale * (cond_pos - uncond_pos)
                eps_atom = uncond_atom + g_scale * (cond_atom - uncond_atom)
            else:
                eps_pos, eps_atom = cond_pos, cond_atom
            ab = self.alpha_bar[t_val]
            beta = 1 - ab / (self.alpha_bar[t_val - 1] if t_val > 0 else 1.0)
            pos = (pos - beta / (1 - ab).sqrt() * eps_pos) / (1 - beta).sqrt()
            if t_val > 0:
                pos = pos + beta.sqrt() * torch.randn_like(pos)
            atom_types = eps_atom.argmax(dim=-1)
        return {"positions": pos.view(num_molecules, num_atoms, 3),
                "atom_types": atom_types.view(num_molecules, num_atoms)}

    def _fully_connected(self, n_mol, n_atoms):
        src, dst = [], []
        for i in range(n_mol):
            off = i * n_atoms
            idx = torch.arange(n_atoms, device=self.device) + off
            r = idx.repeat_interleave(n_atoms)
            c = idx.repeat(n_atoms)
            m = r != c
            src.append(r[m]); dst.append(c[m])
        return torch.stack([torch.cat(src), torch.cat(dst)])
