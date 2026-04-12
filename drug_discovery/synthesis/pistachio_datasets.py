"""Pistachio reaction dataset loader.

Wraps the ``CASPistachio/pistachio`` external submodule to load patent-level
chemical reaction data for training and validating synthesis models.  Falls
back gracefully when the optional dependency is absent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class ReactionRecord:
    """A single reaction entry loaded from a Pistachio dataset."""

    reactants: str | list[str]
    products: str | list[str] | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reactants": self.reactants,
            "products": self.products,
            "conditions": self.conditions,
            "metadata": self.metadata,
        }


@dataclass
class PistachioDatasetResult:
    """Outcome of a Pistachio dataset load operation."""

    dataset_path: str
    reactions: list[ReactionRecord] = field(default_factory=list)
    count: int = 0
    success: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "reactions": [r.as_dict() for r in self.reactions],
            "count": self.count,
            "success": self.success,
            "error": self.error,
        }


class PistachioDatasets:
    """Load and filter reaction datasets from the Pistachio corpus.

    Wraps ``CASPistachio/pistachio``.  When the package is unavailable every
    call returns a failed :class:`PistachioDatasetResult` rather than raising,
    so that the rest of the pipeline can handle missing data gracefully.

    Example::

        loader = PistachioDatasets(limit=500)
        result = loader.load("/data/pistachio_reactions.json")
        for rxn in result.reactions:
            print(rxn.reactants, "→", rxn.products)
    """

    def __init__(self, limit: int = 1000, filter_drug_like: bool = False):
        """
        Args:
            limit: Maximum number of reactions to load.
            filter_drug_like: When ``True`` filter reactions to those involving
                drug-like molecules (requires RDKit).
        """
        self.limit = limit
        self.filter_drug_like = filter_drug_like

    def is_available(self) -> bool:
        """Return ``True`` when the ``pistachio`` package is importable."""
        return get_integration_status("pistachio").available

    def load(self, dataset_path: str) -> PistachioDatasetResult:
        """Load reactions from *dataset_path*.

        Args:
            dataset_path: Path to a Pistachio reaction dataset file.

        Returns:
            :class:`PistachioDatasetResult` with parsed reaction records.
        """
        if not dataset_path:
            return PistachioDatasetResult(dataset_path=dataset_path, error="Empty dataset_path")

        ensure_local_checkout_on_path("pistachio")

        try:
            import pistachio

            reactions = list(self._iter_reactions(pistachio, dataset_path))
            if self.filter_drug_like:
                reactions = [r for r in reactions if self._is_drug_like(r)]

            return PistachioDatasetResult(
                dataset_path=dataset_path,
                reactions=reactions[: self.limit],
                count=len(reactions[: self.limit]),
                success=True,
            )
        except Exception as exc:
            logger.warning("Pistachio dataset load failed for %s: %s", dataset_path, exc)
            return PistachioDatasetResult(dataset_path=dataset_path, error=str(exc))

    def _iter_reactions(self, pistachio_module: Any, dataset_path: str) -> Iterator[ReactionRecord]:
        """Yield :class:`ReactionRecord` objects from the pistachio module."""
        import itertools

        loader_fn = getattr(pistachio_module, "load", None) or getattr(pistachio_module, "load_reactions", None)
        if loader_fn is None:
            raise AttributeError("pistachio module has no 'load' or 'load_reactions' function")

        for item in itertools.islice(loader_fn(dataset_path), self.limit):
            if isinstance(item, dict):
                yield ReactionRecord(
                    reactants=item.get("reactants", ""),
                    products=item.get("products"),
                    conditions=item.get("conditions", {}),
                    metadata={k: v for k, v in item.items() if k not in ("reactants", "products", "conditions")},
                )
            else:
                yield ReactionRecord(
                    reactants=getattr(item, "reactants", str(item)),
                    products=getattr(item, "products", None),
                    conditions=getattr(item, "conditions", {}),
                )

    @staticmethod
    def _is_drug_like(record: ReactionRecord) -> bool:
        """Return ``True`` if the reaction involves at least one drug-like molecule."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors

            smiles_candidates: list[str] = []
            reactants = record.reactants
            if isinstance(reactants, str):
                smiles_candidates.extend(reactants.split("."))
            elif isinstance(reactants, list):
                smiles_candidates.extend(reactants)

            for smi in smiles_candidates:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mw = Descriptors.MolWt(mol)
                hba = rdMolDescriptors.CalcNumHBA(mol)
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                if 150 <= mw <= 600 and hba <= 10 and hbd <= 5:
                    return True
        except Exception:
            pass
        return False

    def load_from_directory(self, directory: str) -> list[PistachioDatasetResult]:
        """Load reactions from all dataset files found in *directory*.

        Args:
            directory: Path to a directory containing Pistachio dataset files.

        Returns:
            List of :class:`PistachioDatasetResult` objects, one per file.
        """
        results = []
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return [PistachioDatasetResult(dataset_path=directory, error=f"Not a directory: {directory}")]
        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file():
                results.append(self.load(str(file_path)))
        return results
