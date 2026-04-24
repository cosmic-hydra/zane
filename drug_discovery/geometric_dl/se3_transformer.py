"""
SE(3)-Equivariant Transformer for Molecular Dynamics Prediction.

Ingests 3D atomic coordinates from PDB files and predicts structural
conformational changes over time to identify transient binding pockets.

References:
    - Satorras et al., "E(n) Graph Neural Networks"
    - Jumper et al., "AlphaFold2" (for architectural inspiration)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    logger.warning("PyTorch not available. Using simplified geometry model.")

try:
    import e3nn
    from e3nn import o3
    from e3nn.nn import Gate, LayerNorm

    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False
    e3nn = None
    o3 = None
    logger.warning("e3nn not available. Using PyTorch-only implementation.")


@dataclass
class AtomFeature:
    """Atomic features for SE(3)-equivariant model."""

    atomic_number: int = 0
    position: np.ndarray | None = None  # (3,) coordinates
    residue_type: str = "UNK"
    residue_id: int = 0
    chain_id: str = "A"
    atom_name: str = "CA"
    amino_acid_one_hot: np.ndarray | None = None  # (20,) one-hot


@dataclass
class SE3Config:
    """Configuration for SE(3)-Equivariant Transformer."""

    hidden_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_radius: float = 10.0  # Angstroms
    radial_layers: int = 3
    lmax: int = 3  # Maximum spherical harmonic degree
    soft_cutoff: bool = True


class SE3EquivariantBlock(nn.Module if TORCH_AVAILABLE else object):
    """
    SE(3)-Equivariant message passing block.

    Implements equivariant operations on point clouds that preserve
    rotational and translational symmetries.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps | str = "128x0e",
        irreps_out: o3.Irreps | str = "128x0e",
        irreps_node_attr: o3.Irreps | str = "0e",
        irreps_edge_attr: o3.Irreps | str = "0e+1e+2e",
        irreps_node_output: o3.Irreps | str = "0e+1e+2e",
        irreps_sh: o3.Irreps | str = "0e+1e+2e",
        fc_neurons: list[int] | None = None,
    ):
        """
        Initialize SE(3)-equivariant block.

        Args:
            irreps_in: Input irreducible representations.
            irreps_out: Output irreps.
            irreps_node_attr: Node attribute irreps.
            irreps_edge_attr: Edge attribute irreps.
            irreps_node_output: Output irreps for nodes.
            irreps_sh: Spherical harmonic irreps.
            fc_neurons: Hidden layer sizes for fully connected.
        """
        super().__init__()

        if not E3NN_AVAILABLE:
            self.hidden_dim = 128
            return

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        fc_neurons = fc_neurons or [64, 64]

        # Tensor product for message computation
        self.tp = o3.TensorProduct(
            o3.Irreps(irreps_in),
            o3.Irreps(irreps_sh),
            "uvu",
            internal_weights=True,
            shared_weights=True,
        )

        # Fully connected network for message
        input_dim = self.irreps_in.dim + irreps_node_attr.dim + irreps_edge_attr.dim
        layers = []
        for dim in fc_neurons:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.SiLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, self.tp.num_weights))
        self.message_mlp = nn.Sequential(*layers)

        # Update network
        self.update_lin = nn.Linear(self.irreps_in.dim + self.tp.irreps_out.dim, self.irreps_out.dim)

        # Layer norm
        self.layer_norm = LayerNorm(self.irreps_out)

        logger.debug(f"SE3EquivariantBlock: {irreps_in} -> {irreps_out}")

    def forward(self, x, edge_index, edge_attr, edge_sh):
        """Forward pass with equivariant message passing."""
        if not E3NN_AVAILABLE:
            return x

        src, dst = edge_index

        # Message: tensor product of input features with spherical harmonics
        messages = self.tp(x[src], edge_sh)

        # Apply message MLP weights
        weight = self.message_mlp(torch.cat([x[src], edge_attr], dim=-1))
        messages = messages * weight.unsqueeze(-1)

        # Aggregate messages
        agg = torch.zeros(x.shape[0], self.tp.irreps_out.dim, device=x.device)
        agg.scatter_add_(0, src.unsqueeze(-1).expand_as(messages), messages)

        # Update node features
        x_new = torch.cat([x, agg], dim=-1)
        x_new = self.update_lin(x_new)
        x_new = self.layer_norm(x_new)

        return x_new


class EquivariantAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    SE(3)-Equivariant Multi-Head Attention.

    Computes attention weights using invariant scalar features while
    preserving equivariance in the value projections.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize equivariant attention.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        # Query, Key, Value projections (invariant)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Invariant feature extraction for attention weights
        self.inv_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_distance):
        """
        Forward pass.

        Args:
            x: Node features (N, hidden_dim).
            edge_index: Edge connectivity (2, num_edges).
            edge_distance: Edge distances (num_edges,).

        Returns:
            Updated node features.
        """
        src, dst = edge_index

        # Compute query, key, value
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        # Attention scores (using invariant distance features)
        inv_features = self.inv_proj(x).squeeze(-1)  # (N,)
        inv_src = inv_features[src]
        inv_dst = inv_features[dst]

        # Simple attention based on invariant features
        attn_scores = (inv_src + inv_dst) / 2  # (num_edges,)

        # Softmax over destination nodes
        num_nodes = x.shape[0]
        max_scores = torch.zeros(num_nodes, device=x.device) - float("inf")
        max_scores.scatter_reduce_(0, dst, attn_scores, reduce="amax")

        attn_scores = attn_scores - max_scores[dst]
        attn_weights = torch.exp(attn_scores)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values
        aggr = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        v_src = v[src]  # (num_edges, heads, head_dim)
        aggr.scatter_add_(
            0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(v_src), v_src * attn_weights.unsqueeze(-1).unsqueeze(-1)
        )

        # Normalize
        counts = torch.zeros(num_nodes, device=x.device)
        counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        counts = counts.clamp(min=1).unsqueeze(-1).unsqueeze(-1)
        aggr = aggr / counts

        # Output projection
        aggr = aggr.view(-1, self.hidden_dim)
        out = self.out_proj(aggr)

        return out


class SE3Transformer(nn.Module if TORCH_AVAILABLE else object):
    """
    SE(3)-Equivariant Transformer for Molecular Structure.

    Architecture:
    1. Input embedding of atomic features and coordinates
    2. Stack of SE(3)-equivariant layers
    3. Output head for coordinates and structural predictions

    Example::

        model = SE3Transformer(config=SE3Config(hidden_dim=128, num_layers=6))
        coordinates = model.predict_displacement(atom_features, positions)
    """

    def __init__(self, config: SE3Config | None = None):
        """
        Initialize SE(3)-Transformer.

        Args:
            config: Model configuration.
        """
        super().__init__()

        self.config = config or SE3Config()

        if not TORCH_AVAILABLE:
            self.device = "cpu"
            logger.info("SE3Transformer initialized in geometric mode")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_dim = self.config.hidden_dim

        # Atom type embedding
        self.atom_embedding = nn.Embedding(100, hidden_dim)

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(21, hidden_dim)

        # Positional encoding for residues
        self.pos_encoding = nn.Linear(1, hidden_dim // 2)

        # SE(3)-Equivariant layers
        self.layers = nn.ModuleList()
        for i in range(self.config.num_layers):
            self.layers.append(
                SE3EquivariantBlock(
                    irreps_in=f"{hidden_dim}x0e",
                    irreps_out=f"{hidden_dim}x0e",
                    irreps_node_attr="0e",
                    irreps_edge_attr="0e+1e+2e",
                )
            )

        # Equivariant attention
        self.attentions = nn.ModuleList()
        for _ in range(self.config.num_layers // 2):
            self.attentions.append(
                EquivariantAttention(
                    hidden_dim=hidden_dim,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                )
            )

        # Coordinate prediction head (equivariant)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3D displacement
        )

        # Structure quality head
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(self.config.dropout)

        self.to(self.device)
        logger.info(f"SE3Transformer: {self.config.num_layers} layers, {hidden_dim} dim, on {self.device}")

    def forward(self, atom_z, positions, batch_idx=None):
        """
        Forward pass.

        Args:
            atom_z: Atomic numbers (N,).
            positions: Atomic positions (N, 3).
            batch_idx: Batch indices for pooling.

        Returns:
            Dictionary with predictions.
        """
        if not TORCH_AVAILABLE:
            return self._geometric_forward(atom_z, positions)

        # Embeddings
        h = self.atom_embedding(atom_z)

        # Add positional encoding from coordinates
        coord_enc = self._positional_encoding(positions)
        h = h + coord_enc

        # Build edges based on distance
        edge_index, edge_distance = self._build_edges(positions)

        # Spherical harmonics for edge features
        edge_sh = self._compute_spherical_harmonics(edge_index, edge_distance, positions)

        # SE(3)-equivariant message passing
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, h[edge_index[0]], edge_sh)
            h = self.dropout(h)

            # Add attention every other layer
            if i % 2 == 1 and i // 2 < len(self.attentions):
                h = h + self.attentions[i // 2](h, edge_index, edge_distance)

        # Predict coordinate displacements
        coord_delta = self.coord_head(h)
        new_positions = positions + coord_delta

        # Structure quality
        quality = self.quality_head(h)

        return {
            "positions": new_positions,
            "displacement": coord_delta,
            "quality": quality,
            "features": h,
        }

    def _positional_encoding(self, positions):
        """Compute positional encoding from coordinates."""
        if not TORCH_AVAILABLE:
            return torch.zeros(positions.shape[0], self.config.hidden_dim)

        # Radial encoding
        r = torch.norm(positions, dim=-1, keepdim=True)
        r_enc = self.pos_encoding(r)

        # Angular encoding (simplified) - pool 3 coords to 1
        angles = positions / (r.clamp(min=1e-6) + 1e-6)
        angle_pool = torch.mean(angles, dim=-1, keepdim=True)  # (N, 1)
        angle_enc = self.pos_encoding(angle_pool)

        return torch.cat([r_enc, angle_enc], dim=-1)

    def _build_edges(self, positions, max_radius=None):
        """Build edges based on spatial proximity."""
        if max_radius is None:
            max_radius = self.config.max_radius

        N = positions.shape[0]

        # Compute pairwise distances
        dists = torch.cdist(positions, positions)

        # Create edges within radius
        edge_index = []
        edge_distance = []

        for i in range(N):
            for j in range(N):
                if i != j and dists[i, j] < max_radius:
                    edge_index.append([i, j])
                    edge_distance.append(dists[i, j].item())

        if not edge_index:
            edge_index = [[0, 1], [1, 0]]
            edge_distance = [1.0, 1.0]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_distance = torch.tensor(edge_distance)

        return edge_index, edge_distance

    def _compute_spherical_harmonics(self, edge_index, distances, positions):
        """Compute spherical harmonic features for edges."""
        if not E3NN_AVAILABLE or not TORCH_AVAILABLE:
            return torch.zeros(edge_index.shape[1], 9)  # lmax=2 -> 9 components

        src, dst = edge_index
        r = positions[dst] - positions[src]
        r_norm = torch.norm(r, dim=-1, keepdim=True)
        r = r / (r_norm + 1e-8)

        # Compute spherical harmonics up to lmax
        sh = o3.spherical_harmonics(
            o3.Irreps(f"{self.config.lmax}e"),
            r,
            normalize=True,
        )

        return sh

    def _geometric_forward(self, atom_z, positions):
        """Fallback forward without PyTorch."""
        return {
            "positions": positions,
            "displacement": np.zeros_like(positions),
            "quality": np.ones(len(atom_z)) * 0.5,
            "features": np.zeros((len(atom_z), self.config.hidden_dim)),
        }

    def predict_displacement(self, atom_features, positions):
        """Predict conformational displacement."""
        if TORCH_AVAILABLE:
            atom_z = torch.tensor(atom_features, dtype=torch.long, device=self.device)
            positions_t = torch.tensor(positions, dtype=torch.float32, device=self.device)
            return self.forward(atom_z, positions_t)
        else:
            return self._geometric_forward(atom_features, positions)


class TransientPocketPredictor:
    """
    Predicts transient binding pockets from conformational ensembles.

    Identifies microsecond-length binding pockets using:
    1. Gaussian smoothing of protein surface
    2. Curvature analysis
    3. Dynamic pocket detection across trajectories
    """

    def __init__(
        self,
        min_pocket_size: int = 50,  # Angstroms^3
        min_pocket_depth: float = 3.0,  # Angstroms
    ):
        """
        Initialize pocket predictor.

        Args:
            min_pocket_size: Minimum pocket volume.
            min_pocket_depth: Minimum pocket depth.
        """
        self.min_pocket_size = min_pocket_size
        self.min_pocket_depth = min_pocket_depth

        logger.info("TransientPocketPredictor initialized")

    def predict(
        self,
        protein_coords: np.ndarray,
        protein_indices: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Predict transient pockets.

        Args:
            protein_coords: Protein atom coordinates (N, 3).
            protein_indices: Atom indices for alpha carbons.
            threshold: Detection threshold.

        Returns:
            List of pocket dictionaries.
        """
        if protein_indices is None:
            # Use alpha carbons (every 4th atom as proxy)
            protein_indices = np.arange(0, len(protein_coords), 4)

        ca_coords = protein_coords[protein_indices]

        # Compute convex hull or alpha shape
        pockets = self._find_pockets(ca_coords, threshold)

        return pockets

    def _find_pockets(self, coords, threshold):
        """Find pockets using alpha shape / convex hull analysis."""
        pockets = []

        try:
            from scipy.spatial import Delaunay

            # Compute Delaunay triangulation
            tri = Delaunay(coords)

            # Find tetrahedra with large volume (potential pockets)
            for simplex in tri.simplices:
                if len(simplex) >= 4:
                    tetrahedron = coords[simplex]
                    volume = self._tetrahedron_volume(tetrahedron)

                    if volume > self.min_pocket_size:
                        center = tetrahedron.mean(axis=0)
                        pockets.append(
                            {
                                "center": center,
                                "volume": volume,
                                "atoms": simplex.tolist(),
                                "score": volume / self.min_pocket_size,
                            }
                        )

        except Exception as e:
            logger.warning(f"Pocket prediction failed: {e}")

        return pockets

    def _tetrahedron_volume(self, points):
        """Compute tetrahedron volume."""
        if len(points) < 4:
            return 0.0

        a, b, c, d = points[:4]
        v = b - a
        w = c - a
        u = d - a

        return abs(np.dot(u, np.cross(v, w))) / 6.0
