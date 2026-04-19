"""OpenFold adapter for protein 3D structure prediction.

Wraps the ``aqlaboratory/openfold`` external submodule.  When OpenFold is not
installed the adapter returns a failed result rather than raising, so that
pipeline code that uses protein structures can handle unavailability gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class StructurePrediction:
    """Result of an OpenFold protein structure prediction."""

    sequence: str
    pdb_string: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "pdb_string": self.pdb_string,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
        }


class OpenFoldAdapter:
    """Adapter for OpenFold protein structure prediction.

    Wraps ``aqlaboratory/openfold``.  All heavy imports are deferred to call
    time so that the rest of ZANE works without OpenFold installed.

    Example::

        adapter = OpenFoldAdapter()
        result = adapter.predict_structure("ACDEFGHIKLMNPQRSTVWY")
        print(result.pdb_string[:100])
    """

    def __init__(self, use_templates: bool = False):
        """
        Args:
            use_templates: Whether to use structure templates during prediction.
                Requires additional template databases.
        """
        self.use_templates = use_templates

    def is_available(self) -> bool:
        """Return ``True`` when the ``openfold`` package is importable."""
        return get_integration_status("openfold").available

    def predict_structure(self, sequence: str) -> StructurePrediction:
        """Predict the 3D structure of a protein from its amino-acid sequence.

        Args:
            sequence: Amino-acid sequence (single-letter codes, uppercase).

        Returns:
            :class:`StructurePrediction` containing the PDB-format structure
            string and confidence score.  ``success`` will be ``False`` when
            OpenFold is unavailable or prediction fails.
        """
        if not sequence:
            return StructurePrediction(sequence=sequence, error="Empty sequence")

        ensure_local_checkout_on_path("openfold")

        try:
            from openfold.np import protein as of_protein
            from openfold.utils.script_utils import run_model

            output = run_model(sequence, use_templates=self.use_templates)
            if isinstance(output, str):
                pdb_string = output
                confidence = None
            elif isinstance(output, dict):
                pdb_string = output.get("pdb_string") or of_protein.to_pdb(output.get("protein"))
                confidence = output.get("mean_plddt") or output.get("confidence")
            else:
                pdb_string = of_protein.to_pdb(output)
                confidence = None

            return StructurePrediction(
                sequence=sequence,
                pdb_string=pdb_string,
                confidence=float(confidence) if confidence is not None else None,
                success=True,
            )
        except Exception as exc:
            logger.warning("OpenFold structure prediction failed: %s", exc)
            return StructurePrediction(sequence=sequence, error=str(exc))

    def batch_predict(self, sequences: list[str]) -> list[StructurePrediction]:
        """Predict structures for multiple sequences.

        Args:
            sequences: List of amino-acid sequence strings.

        Returns:
            List of :class:`StructurePrediction` objects in input order.
        """
        return [self.predict_structure(seq) for seq in sequences]


# ---------------------------------------------------------------------------
# Compatibility helpers expected by higher-level orchestration/tests.


def parse_pdb(pdb_string: str | None) -> dict[str, Any]:
    """Lightweight PDB parser stub returning metadata dictionary."""
    if not pdb_string:
        return {"atoms": 0, "chains": 0, "residues": 0}
    return {"atoms": pdb_string.count("ATOM"), "chains": pdb_string.count("CHAIN"), "residues": pdb_string.count("RES")}


class ProteinGraph:
    """Minimal protein graph container used in tests."""

    def __init__(self, nodes: int = 0, edges: int = 0):
        self.nodes = nodes
        self.edges = edges
