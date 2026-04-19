"""
Protein–ligand interaction modeling with joint embeddings and cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProteinLigandConfig:
    ligand_feat_dim: int = 64
    residue_feat_dim: int = 64
    hidden_dim: int = 256
    heads: int = 4
    dropout: float = 0.1


class ProteinLigandInteractionModel(nn.Module):
    """
    Joint encoder producing binding affinity and a residue–atom contact map.
    """

    def __init__(self, config: ProteinLigandConfig):
        super().__init__()
        self.config = config
        hd = config.hidden_dim
        self.ligand_proj = nn.Linear(config.ligand_feat_dim, hd)
        self.residue_proj = nn.Linear(config.residue_feat_dim, hd)
        self.cross_attn = nn.MultiheadAttention(hd, num_heads=config.heads, batch_first=True, dropout=config.dropout)
        self.ligand_norm = nn.LayerNorm(hd)
        self.residue_norm = nn.LayerNorm(hd)
        self.fusion = nn.Sequential(nn.Linear(hd, hd), nn.SiLU(), nn.Dropout(config.dropout))
        self.affinity_head = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, hd // 2), nn.SiLU(), nn.Linear(hd // 2, 1))
        self.contact_proj = nn.Linear(hd, hd, bias=False)

    def forward(
        self,
        ligand_feats: torch.Tensor,
        residue_feats: torch.Tensor,
        ligand_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ligand_feats: (B, N_atoms, D_lig)
            residue_feats: (B, N_res, D_res)
            ligand_mask: optional boolean mask where True marks padding atoms.
            residue_mask: optional boolean mask where True marks padding residues.
        """
        lig = self.ligand_proj(ligand_feats)
        res = self.residue_proj(residue_feats)
        lig = self.ligand_norm(lig)
        res = self.residue_norm(res)

        attn_out, _ = self.cross_attn(lig, res, res, key_padding_mask=residue_mask)
        lig_fused = self.fusion(lig + attn_out)

        # Binding affinity from pooled ligand representation.
        if ligand_mask is not None:
            valid_counts = (~ligand_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (lig_fused.masked_fill(ligand_mask.unsqueeze(-1), 0.0).sum(dim=1)) / valid_counts
        else:
            pooled = lig_fused.mean(dim=1)
        affinity = self.affinity_head(pooled).squeeze(-1)

        # Contact map via bilinear projection.
        res_proj = self.contact_proj(res)
        contact = torch.einsum("bid,bjd->bij", lig_fused, res_proj)
        if ligand_mask is not None:
            contact = contact.masked_fill(ligand_mask.unsqueeze(-1), float("-inf"))
        if residue_mask is not None:
            contact = contact.masked_fill(residue_mask.unsqueeze(1), float("-inf"))
        contact = torch.sigmoid(contact)

        return {"affinity": affinity, "contact_map": contact}
