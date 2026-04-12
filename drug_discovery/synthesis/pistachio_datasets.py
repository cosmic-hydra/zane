"""Pistachio reaction-dataset tools adapter.

Wraps the Pistachio toolkit (https://github.com/CASPistachio/pistachio) for
loading patent-scale chemical reaction datasets.  Falls back to a simple
JSON-lines reader when the library is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class ReactionRecord:
    """A single chemical reaction record from a Pistachio-style dataset."""

    reaction_id: str
    reactants: list[str]
    products: list[str]
    reagents: list[str] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)
    source: str = "pistachio"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reactants_smiles(self) -> str:
        """Return reactants as a dot-joined SMILES string."""
        return ".".join(self.reactants)

    @property
    def top_product(self) -> str | None:
        """Return the first listed product, if any."""
        return self.products[0] if self.products else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "reactants": self.reactants,
            "products": self.products,
            "reagents": self.reagents,
            "conditions": self.conditions,
            "reactants_smiles": self.reactants_smiles,
            "top_product": self.top_product,
            "source": self.source,
            "metadata": self.metadata,
        }


class PistachioDatasets:
    """Load and filter patent-scale reaction datasets using Pistachio tools.

    Usage::

        ds = PistachioDatasets()
        records = ds.load("path/to/reactions.jsonl", limit=500)
        drug_like = ds.filter_drug_like(records)
    """

    def __init__(self, limit: int = 1000):
        """
        Args:
            limit: Default maximum number of records to load.
        """
        self.limit = limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return *True* if the Pistachio library can be imported."""
        ensure_local_checkout_on_path("pistachio")
        try:
            import pistachio  # type: ignore[import]  # noqa: F401

            return True
        except Exception:
            return False

    def load(self, data_path: str, limit: int | None = None) -> list[ReactionRecord]:
        """Load reaction records from a Pistachio-format dataset file.

        Args:
            data_path: Path to the dataset file (JSON-lines or Pistachio native).
            limit: Maximum number of records to return (overrides instance
                default when provided).

        Returns:
            List of :class:`ReactionRecord` instances.
        """
        effective_limit = limit if limit is not None else self.limit
        from drug_discovery.external_tooling import pistachio_load_reactions

        raw_records = pistachio_load_reactions(data_path, limit=effective_limit)
        return [self._parse_record(r) for r in raw_records]

    def filter_drug_like(
        self,
        records: list[ReactionRecord],
        max_mol_weight: float = 600.0,
        max_hbd: int = 5,
        max_hba: int = 10,
    ) -> list[ReactionRecord]:
        """Filter reactions to retain those whose products are drug-like.

        Applies a subset of Lipinski's Rule of Five to the top product of
        each record.

        Args:
            records: Input reaction records.
            max_mol_weight: Maximum molecular weight of products.
            max_hbd: Maximum hydrogen-bond donors.
            max_hba: Maximum hydrogen-bond acceptors.

        Returns:
            Filtered list of :class:`ReactionRecord` instances.
        """
        filtered: list[ReactionRecord] = []
        for rec in records:
            product = rec.top_product
            if product is None:
                continue
            if self._is_drug_like(product, max_mol_weight, max_hbd, max_hba):
                filtered.append(rec)
        logger.info(
            "Drug-like filter: %d / %d records retained.", len(filtered), len(records)
        )
        return filtered

    def build_training_dataset(
        self,
        records: list[ReactionRecord],
    ) -> list[dict[str, str]]:
        """Convert reaction records to a SMILES-pair training format.

        Args:
            records: Source reaction records.

        Returns:
            List of ``{"src": "reactants", "tgt": "product"}`` dicts suitable
            for training sequence-to-sequence reaction models.
        """
        rows: list[dict[str, str]] = []
        for rec in records:
            src = rec.reactants_smiles
            tgt = rec.top_product
            if src and tgt:
                rows.append({"src": src, "tgt": tgt})
        return rows

    def statistics(self, records: list[ReactionRecord]) -> dict[str, Any]:
        """Compute basic statistics over a set of reaction records.

        Args:
            records: List of :class:`ReactionRecord` instances.

        Returns:
            Dictionary with record count, average reactants/products per
            reaction, and source breakdown.
        """
        if not records:
            return {"count": 0}

        avg_reactants = sum(len(r.reactants) for r in records) / len(records)
        avg_products = sum(len(r.products) for r in records) / len(records)

        sources: dict[str, int] = {}
        for r in records:
            sources[r.source] = sources.get(r.source, 0) + 1

        return {
            "count": len(records),
            "avg_reactants_per_reaction": round(avg_reactants, 2),
            "avg_products_per_reaction": round(avg_products, 2),
            "sources": sources,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_record(self, raw: dict[str, Any]) -> ReactionRecord:
        """Convert a raw dictionary to a :class:`ReactionRecord`."""
        reaction_id = str(raw.get("id", raw.get("reaction_id", "")))
        reactants_raw = raw.get("reactants", raw.get("reactant_smiles", []))
        products_raw = raw.get("products", raw.get("product_smiles", []))
        reagents_raw = raw.get("reagents", raw.get("reagent_smiles", []))

        def _to_list(val: Any) -> list[str]:
            if isinstance(val, list):
                return [str(v) for v in val]
            if isinstance(val, str):
                return [s.strip() for s in val.split(".") if s.strip()]
            return []

        return ReactionRecord(
            reaction_id=reaction_id,
            reactants=_to_list(reactants_raw),
            products=_to_list(products_raw),
            reagents=_to_list(reagents_raw),
            conditions=raw.get("conditions", {}),
            source=raw.get("source", "pistachio"),
            metadata={k: v for k, v in raw.items() if k not in ("reactants", "products", "reagents", "conditions")},
        )

    @staticmethod
    def _is_drug_like(
        smiles: str,
        max_mol_weight: float,
        max_hbd: int,
        max_hba: int,
    ) -> bool:
        """Check basic drug-likeness using RDKit when available."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            mw = Descriptors.MolWt(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            return mw <= max_mol_weight and hbd <= max_hbd and hba <= max_hba
        except Exception:
            return True  # Accept if RDKit is unavailable

    @staticmethod
    def integration_status() -> dict[str, Any]:
        """Return the current integration status for Pistachio."""
        return get_integration_status("pistachio").as_dict()
