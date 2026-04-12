"""DiffDock-based molecular docking adapter.

Wraps the DiffDock diffusion model
(https://github.com/gcorso/DiffDock) for protein–ligand 3D pose prediction
with a graceful fallback to the classical Vina-based
:class:`~drug_discovery.physics.docking.DockingEngine` when DiffDock is not
installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class DockingPose:
    """A single predicted protein–ligand docking pose."""

    ligand_smiles: str
    protein_id: str
    confidence: float
    binding_energy: float | None = None
    pose_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ligand_smiles": self.ligand_smiles,
            "protein_id": self.protein_id,
            "confidence": self.confidence,
            "binding_energy": self.binding_energy,
            "pose_index": self.pose_index,
            "metadata": self.metadata,
        }


class DiffDockAdapter:
    """High-level interface for DiffDock-based docking predictions.

    When the DiffDock submodule is not initialised the adapter automatically
    falls back to the classical AutoDock-Vina engine so the pipeline can still
    produce results.

    Usage::

        adapter = DiffDockAdapter()
        poses = adapter.dock("CCO", "1ABC", num_poses=5)
        best = adapter.rank_poses(poses)
    """

    def __init__(
        self,
        num_inference_steps: int = 20,
        num_poses: int = 10,
        fallback_to_vina: bool = True,
    ):
        """
        Args:
            num_inference_steps: Number of diffusion reverse-process steps.
            num_poses: Number of binding poses to generate.
            fallback_to_vina: When *True*, fall back to Vina if DiffDock is
                unavailable.
        """
        self.num_inference_steps = num_inference_steps
        self.num_poses = num_poses
        self.fallback_to_vina = fallback_to_vina

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return *True* if the DiffDock submodule can be imported."""
        ensure_local_checkout_on_path("diffdock")
        try:
            import diffdock  # type: ignore[import]  # noqa: F401

            return True
        except Exception:
            return False

    def dock(
        self,
        ligand_smiles: str,
        protein_id: str,
        protein_pdb: str | None = None,
        **kwargs: Any,
    ) -> list[DockingPose]:
        """Predict binding poses for *ligand_smiles* against *protein_id*.

        Args:
            ligand_smiles: Ligand SMILES string.
            protein_id: Identifier for the target protein (PDB ID or custom name).
            protein_pdb: Optional path to a PDB file or PDB string.
            **kwargs: Extra arguments forwarded to the underlying engine.

        Returns:
            List of :class:`DockingPose` objects, sorted by confidence
            (descending).
        """
        if self.is_available():
            return self._run_diffdock(ligand_smiles, protein_id, protein_pdb, **kwargs)

        logger.warning(
            "DiffDock not available (submodule not initialised). "
            "Run: git submodule update --init external/DiffDock"
        )

        if self.fallback_to_vina and protein_pdb:
            return self._run_vina_fallback(ligand_smiles, protein_id, protein_pdb, **kwargs)

        return []

    def rank_poses(self, poses: list[DockingPose]) -> list[DockingPose]:
        """Return *poses* sorted by confidence score (highest first)."""
        return sorted(poses, key=lambda p: p.confidence, reverse=True)

    def screen_library(
        self,
        smiles_list: list[str],
        protein_id: str,
        protein_pdb: str | None = None,
        top_n: int = 10,
    ) -> list[DockingPose]:
        """Screen a library of compounds and return the top *top_n* poses.

        Args:
            smiles_list: List of ligand SMILES to screen.
            protein_id: Target protein identifier.
            protein_pdb: Optional path to a PDB file.
            top_n: Number of top-ranked poses to return across the library.

        Returns:
            Sorted list of the best :class:`DockingPose` instances.
        """
        all_poses: list[DockingPose] = []
        for smi in smiles_list:
            poses = self.dock(smi, protein_id, protein_pdb)
            all_poses.extend(poses)

        return self.rank_poses(all_poses)[:top_n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_diffdock(
        self,
        ligand_smiles: str,
        protein_id: str,
        protein_pdb: str | None,
        **kwargs: Any,
    ) -> list[DockingPose]:
        """Invoke the DiffDock model (requires the submodule to be present)."""
        try:
            import diffdock  # type: ignore[import]

            raw = diffdock.predict(
                ligand=ligand_smiles,
                protein=protein_pdb or protein_id,
                num_poses=self.num_poses,
                inference_steps=self.num_inference_steps,
                **kwargs,
            )
            poses = []
            for i, entry in enumerate(raw or []):
                poses.append(
                    DockingPose(
                        ligand_smiles=ligand_smiles,
                        protein_id=protein_id,
                        confidence=float(entry.get("confidence", 0.0)),
                        binding_energy=entry.get("energy"),
                        pose_index=i,
                        metadata={"source": "diffdock"},
                    )
                )
            return poses
        except Exception as exc:
            logger.error("DiffDock prediction failed: %s", exc)
            return []

    def _run_vina_fallback(
        self,
        ligand_smiles: str,
        protein_id: str,
        protein_pdb: str,
        **kwargs: Any,
    ) -> list[DockingPose]:
        """Use the existing AutoDock-Vina engine as a fallback."""
        try:
            from drug_discovery.physics.docking import DockingEngine

            engine = DockingEngine()
            center: tuple[float, float, float] = kwargs.pop("center", (0.0, 0.0, 0.0))
            result = engine.dock_ligand(ligand_smiles, protein_pdb, center=center)
            if result.get("success"):
                return [
                    DockingPose(
                        ligand_smiles=ligand_smiles,
                        protein_id=protein_id,
                        confidence=1.0,
                        binding_energy=result.get("binding_affinity"),
                        pose_index=0,
                        metadata={"source": "vina_fallback"},
                    )
                ]
        except Exception as exc:
            logger.error("Vina fallback failed: %s", exc)

        return []

    @staticmethod
    def integration_status() -> dict[str, Any]:
        """Return the current integration status for DiffDock."""
        return get_integration_status("diffdock").as_dict()
