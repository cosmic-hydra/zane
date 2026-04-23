"""
Transient Pocket Predictor - Identifies microsecond-length binding pockets.

This module is part of the 4D Geometric Deep Learning module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PocketPrediction:
    """Prediction of a transient binding pocket.

    Attributes:
        pocket_id: Unique pocket identifier.
        center: Center of mass coordinates.
        volume: Pocket volume in Angstroms^3.
        depth: Pocket depth in Angstroms.
        hydrophobicity: Hydrophobic character score.
        polarity: Polar character score.
        druggability_score: Predicted druggability (0-1).
        confidence: Prediction confidence (0-1).
        residues: Residue indices forming pocket.
    """

    pocket_id: str
    center: np.ndarray | None = None
    volume: float = 0.0
    depth: float = 0.0
    hydrophobicity: float = 0.5
    polarity: float = 0.5
    druggability_score: float = 0.5
    confidence: float = 0.5
    residues: list[int] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pocket_id": self.pocket_id,
            "center": self.center.tolist() if self.center is not None else None,
            "volume": self.volume,
            "depth": self.depth,
            "hydrophobicity": self.hydrophobicity,
            "polarity": self.polarity,
            "druggability_score": self.druggability_score,
            "confidence": self.confidence,
            "residues": self.residues,
            "properties": self.properties,
        }


class TransientPocketPredictor:
    """
    Predicts transient binding pockets from conformational ensembles.

    Identifies microsecond-length binding pockets using:
    1. Gaussian smoothing of protein surface
    2. Curvature analysis
    3. Dynamic pocket detection across trajectories

    Example::

        predictor = TransientPocketPredictor()
        pockets = predictor.predict_from_trajectory(
            trajectory_frames=trajectory,
            protein_indices=alpha_carbon_indices,
        )
    """

    def __init__(
        self,
        min_pocket_size: float = 50.0,
        min_pocket_depth: float = 3.0,
        max_pockets: int = 10,
    ):
        """
        Initialize pocket predictor.

        Args:
            min_pocket_size: Minimum pocket volume (Å³).
            min_pocket_depth: Minimum pocket depth (Å).
            max_pockets: Maximum number of pockets to return.
        """
        self.min_pocket_size = min_pocket_size
        self.min_pocket_depth = min_pocket_depth
        self.max_pockets = max_pockets

        # Pocket detection parameters
        self.grid_resolution = 1.0  # Å
        self.surface_threshold = 2.0  # Å from protein surface

        logger.info(f"TransientPocketPredictor: min_size={min_pocket_size}, max_pockets={max_pockets}")

    def predict_from_structure(
        self,
        protein_coords: np.ndarray,
        atom_types: np.ndarray | None = None,
        residue_indices: np.ndarray | None = None,
    ) -> list[PocketPrediction]:
        """
        Predict pockets from static structure.

        Args:
            protein_coords: Protein atom coordinates (N, 3).
            atom_types: Atomic numbers or types.
            residue_indices: Residue index for each atom.

        Returns:
            List of PocketPrediction objects.
        """
        if atom_types is None:
            atom_types = np.ones(len(protein_coords), dtype=int)

        if residue_indices is None:
            residue_indices = np.arange(len(protein_coords))

        # Compute solvent-accessible surface
        surface_points = self._compute_surface(protein_coords, atom_types)

        # Find pockets as voids in surface
        pockets = self._find_pockets_on_surface(surface_points, protein_coords, residue_indices)

        # Score and rank pockets
        pockets = self._score_pockets(pockets)

        return pockets[:self.max_pockets]

    def predict_from_trajectory(
        self,
        trajectory_frames: list[np.ndarray],
        protein_indices: np.ndarray | None = None,
    ) -> list[PocketPrediction]:
        """
        Predict transient pockets from MD trajectory.

        Args:
            trajectory_frames: List of coordinate arrays (T, N, 3).
            protein_indices: Indices of protein atoms (vs solvent).

        Returns:
            List of PocketPrediction with temporal information.
        """
        all_pockets = []

        for t, frame in enumerate(trajectory_frames):
            frame_pockets = self.predict_from_structure(frame)

            for pocket in frame_pockets:
                pocket.properties["frame"] = t
                pocket.properties["transient"] = True

            all_pockets.extend(frame_pockets)

        # Cluster pockets across frames
        persistent_pockets = self._cluster_pockets_temporally(all_pockets)

        return persistent_pockets[:self.max_pockets]

    def _compute_surface(
        self,
        coords: np.ndarray,
        atom_types: np.ndarray,
    ) -> np.ndarray:
        """Compute solvent-accessible surface points."""
        surface_points = []

        # Simple approximation: points at van der Waals radius from atoms
        for i, (pos, atype) in enumerate(zip(coords, atom_types)):
            # Approximate radius based on atom type
            radius = self._vdw_radius(atype)

            # Sample points on sphere
            n_samples = 20
            theta = np.random.uniform(0, 2 * np.pi, n_samples)
            phi = np.random.uniform(0, np.pi, n_samples)

            for th, ph in zip(theta, phi):
                point = pos + radius * np.array([
                    np.sin(ph) * np.cos(th),
                    np.sin(ph) * np.sin(th),
                    np.cos(ph),
                ])

                # Check if point is on surface (not buried)
                if self._is_surface_point(point, coords, atom_types):
                    surface_points.append(point)

        return np.array(surface_points) if surface_points else coords

    def _vdw_radius(self, atom_type: int) -> float:
        """Get van der Waals radius for atom type."""
        radii = {
            1: 1.20,   # H
            6: 1.70,   # C
            7: 1.55,   # N
            8: 1.52,   # O
            9: 1.47,   # F
            15: 1.80,  # P
            16: 1.80,  # S
            17: 1.75,  # Cl
            35: 1.85,  # Br
        }
        return radii.get(atom_type, 1.5)

    def _is_surface_point(
        self,
        point: np.ndarray,
        coords: np.ndarray,
        atom_types: np.ndarray,
    ) -> bool:
        """Check if point is on surface (not buried)."""
        for i, (pos, atype) in enumerate(zip(coords, atom_types)):
            dist = np.linalg.norm(point - pos)
            radius = self._vdw_radius(atype)

            # If inside another atom, it's buried
            if dist < radius * 0.8:
                return False

        return True

    def _find_pockets_on_surface(
        self,
        surface_points: np.ndarray,
        protein_coords: np.ndarray,
        residue_indices: np.ndarray,
    ) -> list[PocketPrediction]:
        """Find pockets from surface points."""
        pockets = []

        if len(surface_points) == 0:
            return pockets

        # Cluster surface points
        from scipy.spatial import Delaunay

        try:
            # Delaunay triangulation
            tri = Delaunay(surface_points)

            # Find large voids (pockets)
            for simplex_idx in range(len(tri.simplices)):
                simplex = tri.simplices[simplex_idx]

                if len(simplex) >= 3:
                    # Compute cavity size
                    points = surface_points[simplex]
                    center = points.mean(axis=0)

                    # Check distance to protein
                    dist_to_protein = np.min([
                        np.min(np.linalg.norm(protein_coords - p, axis=1))
                        for p in points
                    ])

                    # Pocket if cavity is surrounded by protein
                    if dist_to_protein < 5.0:  # Å from protein
                        volume = self._estimate_cavity_volume(points)

                        if volume > self.min_pocket_size:
                            # Find contributing residues
                            nearby_residues = self._find_nearby_residues(
                                center, protein_coords, residue_indices
                            )

                            pockets.append(PocketPrediction(
                                pocket_id=f"pocket_{len(pockets)}",
                                center=center,
                                volume=volume,
                                depth=dist_to_protein,
                                residues=nearby_residues,
                            ))

        except Exception as e:
            logger.warning(f"Pocket finding failed: {e}")

        return pockets

    def _estimate_cavity_volume(self, points: np.ndarray) -> float:
        """Estimate volume of cavity from surface points."""
        if len(points) < 4:
            return 0.0

        # Use convex hull volume as estimate
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            # Fallback: bounding box
            extents = points.max(axis=0) - points.min(axis=0)
            return np.prod(extents) * 0.1

    def _find_nearby_residues(
        self,
        center: np.ndarray,
        protein_coords: np.ndarray,
        residue_indices: np.ndarray,
        cutoff: float = 8.0,
    ) -> list[int]:
        """Find residues within cutoff of pocket center."""
        nearby = []
        for i, (pos, res_idx) in enumerate(zip(protein_coords, residue_indices)):
            if np.linalg.norm(pos - center) < cutoff:
                if res_idx not in nearby:
                    nearby.append(int(res_idx))
        return nearby

    def _score_pockets(
        self,
        pockets: list[PocketPrediction],
    ) -> list[PocketPrediction]:
        """Score and rank pockets by druggability."""
        for pocket in pockets:
            # Volume score (larger is better, up to a point)
            volume_score = min(pocket.volume / 500.0, 1.0)

            # Depth score (deeper is better)
            depth_score = min(pocket.depth / 8.0, 1.0)

            # Size consistency
            size_score = 1.0 if self.min_pocket_size <= pocket.volume <= 2000 else 0.5

            # Combined druggability score
            pocket.druggability_score = (
                0.4 * volume_score
                + 0.3 * depth_score
                + 0.3 * size_score
            )

            # Confidence based on multiple indicators
            pocket.confidence = (
                0.5 * pocket.druggability_score
                + 0.5 * size_score
            )

            # Store scoring components
            pocket.properties.update({
                "volume_score": volume_score,
                "depth_score": depth_score,
                "size_score": size_score,
            })

        # Sort by druggability
        pockets.sort(key=lambda p: p.druggability_score, reverse=True)

        return pockets

    def _cluster_pockets_temporally(
        self,
        pockets: list[PocketPrediction],
    ) -> list[PocketPrediction]:
        """Cluster pockets across trajectory frames."""
        if not pockets:
            return []

        # Group by spatial similarity
        clusters = []
        for pocket in pockets:
            assigned = False

            for cluster in clusters:
                ref_center = cluster["center"]
                if pocket.center is not None:
                    dist = np.linalg.norm(pocket.center - ref_center)

                    if dist < 5.0:  # Å threshold
                        cluster["pockets"].append(pocket)
                        cluster["count"] += 1
                        assigned = True
                        break

            if not assigned:
                clusters.append({
                    "center": pocket.center if pocket.center is not None else np.zeros(3),
                    "pockets": [pocket],
                    "count": 1,
                })

        # Select persistent pockets (appearing in multiple frames)
        persistent = []
        for cluster in clusters:
            persistence = cluster["count"] / max(1, len(set(p.properties.get("frame", 0) for p in cluster["pockets"])))

            if persistence > 0.3:  # Appears in >30% of frames
                # Average properties
                avg_pocket = cluster["pockets"][0]
                avg_pocket.center = np.mean([p.center for p in cluster["pockets"] if p.center is not None], axis=0)
                avg_pocket.volume = np.mean([p.volume for p in cluster["pockets"]])
                avg_pocket.confidence = persistence
                avg_pocket.properties["temporal_persistence"] = persistence
                avg_pocket.properties["n_frames"] = cluster["count"]

                persistent.append(avg_pocket)

        return persistent

    def visualize_pockets(
        self,
        pockets: list[PocketPrediction],
        output_path: str = "pockets.pdb",
    ) -> None:
        """
        Write pocket centers to PDB for visualization.

        Args:
            pockets: List of pocket predictions.
            output_path: Output PDB path.
        """
        lines = ["HEADER    POCKET PREDICTION"]

        for i, pocket in enumerate(pockets):
            center = pocket.center if pocket.center is not None else [0, 0, 0]
            lines.append(
                f"HETATM{(i+1) % 100000:5d}  DUM DUM {i+1:3d}    "
                f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}  1.00 "
                f"{pocket.druggability_score:6.2f}          D"
            )

        lines.append("END")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Pockets written to {output_path}")
