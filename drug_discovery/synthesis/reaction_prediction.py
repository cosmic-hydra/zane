"""Reaction outcome prediction adapter backed by MolecularTransformer.

This module wraps the MolecularTransformer model
(https://github.com/pschwllr/MolecularTransformer) with a graceful fallback
when the external submodule is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class ReactionPrediction:
    """Result of a single reaction-outcome prediction."""

    reactants: str
    products: list[str]
    confidence_scores: list[float] = field(default_factory=list)
    backend: str = "molecular_transformer"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_product(self) -> str | None:
        """Return the highest-confidence predicted product, if any."""
        return self.products[0] if self.products else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reactants": self.reactants,
            "products": self.products,
            "confidence_scores": self.confidence_scores,
            "top_product": self.top_product,
            "backend": self.backend,
            "metadata": self.metadata,
        }


class ReactionPredictor:
    """Predict chemical reaction outcomes using MolecularTransformer.

    When the MolecularTransformer submodule is not installed, the class
    degrades gracefully and returns empty predictions with a descriptive
    warning, so the broader pipeline keeps running.

    Usage::

        predictor = ReactionPredictor()
        result = predictor.predict("CCO.CC(=O)O")
        print(result.top_product)
    """

    def __init__(self, beam_size: int = 5, model_path: str | None = None):
        """
        Args:
            beam_size: Number of beam-search hypotheses (top-N products).
            model_path: Optional path to a fine-tuned model checkpoint.
        """
        self.beam_size = beam_size
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, reactants_smiles: str, **kwargs: Any) -> ReactionPrediction:
        """Predict reaction products for the given reactant SMILES.

        Args:
            reactants_smiles: Reactant SMILES string.  Use '.' to separate
                multiple reactants (e.g. ``"CCO.CC(=O)O"``).
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            :class:`ReactionPrediction` instance.
        """
        from drug_discovery.external_tooling import predict_reaction

        products = predict_reaction(reactants_smiles, beam_size=self.beam_size, **kwargs)

        if not products:
            logger.warning(
                "MolecularTransformer unavailable or returned no products for '%s'. "
                "Install the submodule: git submodule update --init external/MolecularTransformer",
                reactants_smiles,
            )

        return ReactionPrediction(
            reactants=reactants_smiles,
            products=products,
            confidence_scores=[],
            backend="molecular_transformer",
            metadata={"beam_size": self.beam_size},
        )

    def predict_batch(self, reactants_list: list[str], **kwargs: Any) -> list[ReactionPrediction]:
        """Predict reaction outcomes for multiple reactant SMILES strings.

        Args:
            reactants_list: List of reactant SMILES strings.
            **kwargs: Additional keyword arguments forwarded to :meth:`predict`.

        Returns:
            List of :class:`ReactionPrediction` instances (same order as input).
        """
        return [self.predict(smi, **kwargs) for smi in reactants_list]

    def validate_reaction(self, reactants_smiles: str, expected_product: str) -> dict[str, Any]:
        """Check whether *expected_product* appears among the predicted products.

        Args:
            reactants_smiles: Reactant SMILES.
            expected_product: Known or hypothetical product SMILES to validate.

        Returns:
            Dictionary with keys ``matched``, ``rank``, and ``predicted_products``.
        """
        result = self.predict(reactants_smiles)

        try:
            from rdkit import Chem

            def _canon(smi: str) -> str | None:
                mol = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(mol) if mol is not None else None

            canon_expected = _canon(expected_product)
            canon_predicted = [_canon(p) for p in result.products]
        except Exception:
            # If RDKit is unavailable, fall back to exact string comparison.
            canon_expected = expected_product
            canon_predicted = list(result.products)

        rank: int | None = None
        if canon_expected:
            for i, pred in enumerate(canon_predicted):
                if pred == canon_expected:
                    rank = i
                    break

        return {
            "matched": rank is not None,
            "rank": rank,
            "predicted_products": result.products,
        }

    @staticmethod
    def integration_status() -> dict[str, Any]:
        """Return the current integration status for MolecularTransformer."""
        return get_integration_status("molecular_transformer").as_dict()
