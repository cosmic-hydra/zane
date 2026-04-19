"""MolecularTransformer adapter for reaction outcome prediction.

This module provides a thin, failure-safe wrapper around the MolecularTransformer
external submodule (``external/MolecularTransformer``).  All heavy imports are
deferred to call time so that the rest of ZANE continues to work even when the
optional dependency is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class ReactionPrediction:
    """Outcome of a single reaction prediction call."""

    reactants: str
    predictions: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    success: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reactants": self.reactants,
            "predictions": self.predictions,
            "scores": self.scores,
            "success": self.success,
            "error": self.error,
        }


class MolecularTransformerAdapter:
    """Adapter for MolecularTransformer reaction prediction.

    Wraps the ``pschwllr/MolecularTransformer`` external submodule.  Falls back
    gracefully when the package is not installed or the submodule has not been
    checked out.

    Example::

        adapter = MolecularTransformerAdapter(beam_size=5)
        result = adapter.predict("CC(=O)O.CCN")
        print(result.predictions)
    """

    def __init__(self, beam_size: int = 5):
        """
        Args:
            beam_size: Beam width for the transformer decoding step.
        """
        self.beam_size = beam_size

    def is_available(self) -> bool:
        """Return ``True`` when the MolecularTransformer package is importable."""
        return get_integration_status("molecular_transformer").available

    def predict(self, reactants_smiles: str) -> ReactionPrediction:
        """Predict reaction product(s) from reactant SMILES.

        Args:
            reactants_smiles: Reactant SMILES (dot-separated for multiple species).

        Returns:
            :class:`ReactionPrediction` with ranked product SMILES and confidence
            scores.  When the dependency is unavailable the ``success`` field will
            be ``False`` and ``error`` will describe the reason.
        """
        if not reactants_smiles:
            return ReactionPrediction(reactants=reactants_smiles, error="Empty reactants SMILES")

        ensure_local_checkout_on_path("molecular_transformer")

        try:
            import molecular_transformer as mt

            raw = mt.predict(reactants_smiles, beam_size=self.beam_size)
            predictions: list[str] = []
            scores: list[float] = []
            for item in raw:
                if isinstance(item, tuple):
                    smiles, score = item[0], float(item[1])
                else:
                    smiles, score = str(item), 0.0
                predictions.append(smiles)
                scores.append(score)

            return ReactionPrediction(
                reactants=reactants_smiles,
                predictions=predictions,
                scores=scores,
                success=True,
            )
        except Exception as exc:
            logger.warning("MolecularTransformer prediction failed: %s", exc)
            return ReactionPrediction(
                reactants=reactants_smiles,
                error=str(exc),
            )

    def batch_predict(self, reactants_list: list[str]) -> list[ReactionPrediction]:
        """Predict reaction outcomes for a list of reactant SMILES.

        Args:
            reactants_list: List of reactant SMILES strings.

        Returns:
            List of :class:`ReactionPrediction` objects in input order.
        """
        return [self.predict(r) for r in reactants_list]


# Compatibility alias expected by tests
class ReactionPredictor(ReactionPrediction):
    """Alias for ReactionPrediction."""

    pass
