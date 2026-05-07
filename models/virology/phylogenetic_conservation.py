"""Phylogenetic conservation analysis for pan-viral target selection."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from Bio import AlignIO
except ImportError:  # pragma: no cover - optional dependency
    AlignIO = None

try:
    from scipy.stats import entropy as scipy_entropy
except ImportError:  # pragma: no cover - optional dependency
    scipy_entropy = None


@dataclass
class ConservedPocketConstraint:
    residues: list[int]
    coordinates: list[list[float]]
    center: list[float]
    entropy_threshold: float
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "residues": self.residues,
            "coordinates": self.coordinates,
            "center": self.center,
            "entropy_threshold": self.entropy_threshold,
            "rationale": self.rationale,
        }


class UltraConservedSiteFinder:
    """Find low-entropy, mutation-resistant antiviral target coordinates."""

    def __init__(self, entropy_threshold: float = 0.15, min_sequences: int = 3):
        self.entropy_threshold = float(max(0.0, entropy_threshold))
        self.min_sequences = int(max(2, min_sequences))

    @staticmethod
    def _infer_alignment_format(msa_path: Path) -> str:
        suffix = msa_path.suffix.lower()
        if suffix in {".fasta", ".fa", ".faa"}:
            return "fasta"
        if suffix in {".aln", ".clustal"}:
            return "clustal"
        if suffix in {".sto", ".stockholm"}:
            return "stockholm"
        return "fasta"

    @staticmethod
    def _shannon_entropy(values: list[str]) -> float:
        filtered = [v for v in values if v not in {"-", "X", "?", "N"}]
        if not filtered:
            return 0.0
        counts = np.array(list(Counter(filtered).values()), dtype=float)
        probs = counts / counts.sum()
        if scipy_entropy is not None:
            return float(scipy_entropy(probs, base=2))
        return float(-np.sum([p * math.log2(p) for p in probs if p > 0]))

    def calculate_shannon_entropy(self, msa_file: str) -> dict[str, Any]:
        """Calculate per-position Shannon entropy for a viral MSA."""
        msa_path = Path(msa_file)
        if not msa_path.exists():
            raise FileNotFoundError(f"MSA file not found: {msa_file}")
        if AlignIO is None:
            raise ImportError("Biopython AlignIO is required for MSA parsing.")

        alignment = AlignIO.read(str(msa_path), self._infer_alignment_format(msa_path))
        if len(alignment) < self.min_sequences:
            raise ValueError(f"Expected at least {self.min_sequences} aligned sequences, got {len(alignment)}.")

        alignment_length = alignment.get_alignment_length()
        entropy_by_position: dict[int, float] = {}
        for idx in range(alignment_length):
            column = [str(record.seq[idx]) for record in alignment]
            entropy_by_position[idx + 1] = self._shannon_entropy(column)

        conserved_positions = [
            pos for pos, ent in entropy_by_position.items() if ent <= self.entropy_threshold
        ]
        return {
            "num_sequences": int(len(alignment)),
            "alignment_length": int(alignment_length),
            "entropy_by_position": entropy_by_position,
            "entropy_threshold": self.entropy_threshold,
            "conserved_positions": conserved_positions,
        }

    @staticmethod
    def _extract_residue_coordinates(pdb_path: Path, chain_id: str | None = None) -> dict[int, list[float]]:
        residue_coords: dict[int, list[float]] = {}
        with pdb_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                chain = line[21].strip() or None
                if chain_id is not None and chain != chain_id:
                    continue
                try:
                    residue_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue
                residue_coords[residue_id] = [x, y, z]
        return residue_coords

    def extract_rigid_pockets(
        self,
        target_structure_pdb: str,
        entropy_profile: dict[str, Any],
        chain_id: str | None = None,
    ) -> dict[str, Any]:
        """Map low-entropy residues to immutable docking pocket constraints."""
        pdb_path = Path(target_structure_pdb)
        if not pdb_path.exists():
            raise FileNotFoundError(f"Target structure not found: {target_structure_pdb}")
        if "conserved_positions" not in entropy_profile:
            raise ValueError("entropy_profile must include 'conserved_positions'.")

        residue_coords = self._extract_residue_coordinates(pdb_path, chain_id=chain_id)
        conserved = [int(pos) for pos in entropy_profile["conserved_positions"]]
        rigid_residues = [res for res in conserved if res in residue_coords]
        if not rigid_residues:
            raise ValueError("No conserved residues mapped to target structure coordinates.")

        coords = np.array([residue_coords[res] for res in rigid_residues], dtype=float)
        center = coords.mean(axis=0).tolist()
        pocket = ConservedPocketConstraint(
            residues=rigid_residues,
            coordinates=[residue_coords[r] for r in rigid_residues],
            center=[float(v) for v in center],
            entropy_threshold=float(entropy_profile.get("entropy_threshold", self.entropy_threshold)),
            rationale="Docking restricted to ultra-conserved, low-entropy residues to suppress viral escape.",
        )
        return {
            "immutable_pocket": pocket.as_dict(),
            "docking_constraints": {
                "allowed_residue_ids": rigid_residues,
                "binding_site_center": pocket.center,
                "enforce_conserved_site_only": True,
            },
        }
