"""OpenFold protein structure prediction adapter.

Wraps OpenFold (https://github.com/aqlaboratory/openfold) for predicting
protein 3D structures from amino-acid sequences.  Falls back to minimal
metadata when the library is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class ProteinStructure:
    """Predicted or cached protein structure information."""

    protein_id: str
    sequence: str
    pdb_string: str | None = None
    plddt_score: float | None = None
    num_residues: int = 0
    backend: str = "openfold"
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Return *True* if a PDB string is present (structure was predicted)."""
        return self.pdb_string is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "protein_id": self.protein_id,
            "sequence": self.sequence,
            "pdb_available": self.pdb_string is not None,
            "plddt_score": self.plddt_score,
            "num_residues": self.num_residues,
            "backend": self.backend,
            "metadata": self.metadata,
        }


class ProteinStructurePredictor:
    """Predict protein 3D structures using OpenFold.

    When the OpenFold submodule is not installed the class returns a
    :class:`ProteinStructure` with ``pdb_string=None`` and ``backend``
    set to ``'unavailable'``, allowing downstream docking steps to skip
    or substitute a cached structure.

    Usage::

        predictor = ProteinStructurePredictor()
        structure = predictor.predict("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ...")
        print(structure.plddt_score)
    """

    def __init__(self, config_preset: str = "model_1", use_gpu: bool = False):
        """
        Args:
            config_preset: OpenFold configuration preset name.
            use_gpu: Whether to run inference on GPU.
        """
        self.config_preset = config_preset
        self.use_gpu = use_gpu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return *True* if OpenFold can be imported."""
        ensure_local_checkout_on_path("openfold")
        try:
            import openfold  # type: ignore[import]  # noqa: F401

            return True
        except Exception:
            return False

    def predict(
        self,
        sequence: str,
        protein_id: str = "unknown",
        **kwargs: Any,
    ) -> ProteinStructure:
        """Predict the 3D structure for an amino-acid *sequence*.

        Args:
            sequence: Single-letter amino-acid sequence.
            protein_id: Human-readable identifier for the protein.
            **kwargs: Additional arguments forwarded to OpenFold.

        Returns:
            :class:`ProteinStructure` instance.
        """
        if self.is_available():
            return self._run_openfold(sequence, protein_id, **kwargs)

        logger.warning(
            "OpenFold not available (submodule not initialised). "
            "Run: git submodule update --init external/openfold"
        )
        return ProteinStructure(
            protein_id=protein_id,
            sequence=sequence,
            num_residues=len(sequence),
            backend="unavailable",
            metadata={"note": "OpenFold submodule required"},
        )

    def predict_batch(
        self,
        sequences: list[str],
        protein_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[ProteinStructure]:
        """Predict structures for multiple sequences.

        Args:
            sequences: List of amino-acid sequence strings.
            protein_ids: Optional list of protein identifiers (same length as
                *sequences*).  Defaults to ``'protein_0'``, ``'protein_1'``…
            **kwargs: Forwarded to :meth:`predict`.

        Returns:
            List of :class:`ProteinStructure` instances.
        """
        if protein_ids is None:
            protein_ids = [f"protein_{i}" for i in range(len(sequences))]

        return [self.predict(seq, pid, **kwargs) for seq, pid in zip(sequences, protein_ids)]

    def compare_structures(
        self,
        predicted: ProteinStructure,
        reference_pdb: str,
    ) -> dict[str, Any]:
        """Compute structural comparison metrics between *predicted* and a
        reference PDB structure.

        Args:
            predicted: The :class:`ProteinStructure` to evaluate.
            reference_pdb: Path to a reference PDB file.

        Returns:
            Dictionary with metrics (e.g. TM-score, RMSD).  Returns empty
            dict when neither structure data nor comparison tools are
            available.
        """
        if not predicted.is_available():
            return {"available": False, "note": "Predicted structure not available"}

        try:
            import os

            if not os.path.exists(reference_pdb):
                return {"available": False, "note": f"Reference PDB not found: {reference_pdb}"}

            # Placeholder: compute Cα RMSD via RDKit / BioPython when available.
            try:
                from Bio import PDB  # type: ignore[import]

                parser = PDB.PDBParser(QUIET=True)
                _ = parser.get_structure("ref", reference_pdb)
                return {
                    "available": True,
                    "note": "BioPython available; detailed RMSD computation not yet implemented.",
                    "num_residues_predicted": predicted.num_residues,
                }
            except Exception:
                pass

            return {
                "available": False,
                "note": "Install biopython for structural comparison.",
            }
        except Exception as exc:
            logger.error("Structure comparison failed: %s", exc)
            return {"available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_openfold(
        self,
        sequence: str,
        protein_id: str,
        **kwargs: Any,
    ) -> ProteinStructure:
        """Run OpenFold inference (requires the submodule)."""
        try:
            from drug_discovery.external_tooling import openfold_predict_structure

            meta = openfold_predict_structure(sequence, **kwargs)
            return ProteinStructure(
                protein_id=protein_id,
                sequence=sequence,
                pdb_string=meta.get("pdb_string"),
                plddt_score=meta.get("plddt_mean"),
                num_residues=meta.get("num_residues", len(sequence)),
                backend="openfold",
                metadata=meta,
            )
        except Exception as exc:
            logger.error("OpenFold prediction failed: %s", exc)
            return ProteinStructure(
                protein_id=protein_id,
                sequence=sequence,
                num_residues=len(sequence),
                backend="openfold_error",
                metadata={"error": str(exc)},
            )

    @staticmethod
    def integration_status() -> dict[str, Any]:
        """Return the current integration status for OpenFold."""
        return get_integration_status("openfold").as_dict()
