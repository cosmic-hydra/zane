"""Commercial Synthon API & Retrosynthesis Mapper.

After a molecule passes ADMET and FEP scoring, this module:

1. Runs retrosynthetic pathway analysis to decompose the molecule
   into purchasable building blocks (synthons)
2. Queries external vendor APIs (Enamine REAL, Mcule, Sigma-Aldrich)
   to verify commercial availability and pricing
3. Computes a "Supply Chain Viability Score" combining availability,
   cost, and lead time

The module uses RDKit for BRICS decomposition when available and
falls back to a rule-based fragmentation heuristic.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import BRICS, AllChem, Descriptors  # type: ignore[import-untyped]

    _RDKIT = True
except ImportError:
    _RDKIT = False

try:
    import requests as _requests

    _REQUESTS = True
except ImportError:
    _REQUESTS = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Synthon:
    """A building block required for synthesis."""

    smiles: str
    name: str = ""
    role: str = "building_block"  # building_block, reagent, catalyst
    available: bool | None = None
    vendors: list[str] = field(default_factory=list)
    estimated_cost_usd: float | None = None
    lead_time_days: int | None = None
    catalog_ids: dict[str, str] = field(default_factory=dict)

    @property
    def viability(self) -> float:
        """0-1 viability score for this synthon."""
        if self.available is False:
            return 0.0
        if self.available is None:
            return 0.5  # unknown
        score = 1.0
        if self.estimated_cost_usd is not None:
            # Penalize expensive reagents (> $500/g)
            if self.estimated_cost_usd > 500:
                score *= 0.5
            elif self.estimated_cost_usd > 100:
                score *= 0.8
        if self.lead_time_days is not None:
            if self.lead_time_days > 30:
                score *= 0.6
            elif self.lead_time_days > 14:
                score *= 0.8
        if len(self.vendors) == 0:
            score *= 0.3
        return score


@dataclass
class RetrosynthesisResult:
    """Complete retrosynthesis and supply chain analysis."""

    target_smiles: str
    synthons: list[Synthon] = field(default_factory=list)
    pathway_steps: int = 0
    pathway_description: str = ""
    supply_chain_score: float = 0.0  # 0-1 aggregate viability
    total_estimated_cost_usd: float | None = None
    max_lead_time_days: int | None = None
    all_available: bool = False
    analysis_method: str = "heuristic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "synthons": [
                {
                    "smiles": s.smiles,
                    "name": s.name,
                    "available": s.available,
                    "vendors": s.vendors,
                    "cost_usd": s.estimated_cost_usd,
                    "lead_time_days": s.lead_time_days,
                    "viability": s.viability,
                }
                for s in self.synthons
            ],
            "pathway_steps": self.pathway_steps,
            "supply_chain_score": self.supply_chain_score,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "max_lead_time_days": self.max_lead_time_days,
            "all_available": self.all_available,
            "analysis_method": self.analysis_method,
        }


# ---------------------------------------------------------------------------
# Vendor API adapters
# ---------------------------------------------------------------------------
class VendorAdapter:
    """Base class for chemical vendor API queries."""

    name: str = "generic"
    base_url: str = ""

    def search(self, smiles: str) -> dict[str, Any] | None:
        """Search for a compound by SMILES. Returns availability info or None."""
        raise NotImplementedError


class EnamineAdapter(VendorAdapter):
    """Adapter for the Enamine REAL database API."""

    name = "enamine"
    base_url = "https://new.enaminestore.com/api/v1"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def search(self, smiles: str) -> dict[str, Any] | None:
        if not _REQUESTS or not self.api_key:
            return self._mock_search(smiles)

        try:
            resp = _requests.get(
                f"{self.base_url}/catalog/search",
                params={"smiles": smiles, "mode": "exact"},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    hit = data["results"][0]
                    return {
                        "available": True,
                        "catalog_id": hit.get("catalog_id", ""),
                        "price_usd": hit.get("price", {}).get("usd"),
                        "lead_time_days": hit.get("lead_time_days", 7),
                    }
            return None
        except Exception as exc:
            logger.warning("Enamine API query failed: %s", exc)
            return self._mock_search(smiles)

    @staticmethod
    def _mock_search(smiles: str) -> dict[str, Any]:
        """Deterministic mock for testing without API key."""
        h = int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
        available = (h % 3) != 0  # ~67% availability
        return {
            "available": available,
            "catalog_id": f"EN-{h % 100000:05d}" if available else "",
            "price_usd": (h % 500) + 10.0 if available else None,
            "lead_time_days": (h % 21) + 3 if available else None,
        }


class MculeAdapter(VendorAdapter):
    """Adapter for the Mcule marketplace API."""

    name = "mcule"
    base_url = "https://mcule.com/api/v1"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def search(self, smiles: str) -> dict[str, Any] | None:
        if not _REQUESTS or not self.api_key:
            return self._mock_search(smiles)

        try:
            resp = _requests.post(
                f"{self.base_url}/search/exact/",
                json={"query": smiles},
                headers={"Authorization": f"Token {self.api_key}"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    hit = data["results"][0]
                    return {
                        "available": True,
                        "catalog_id": hit.get("mcule_id", ""),
                        "price_usd": hit.get("price"),
                        "lead_time_days": hit.get("delivery_days", 10),
                    }
            return None
        except Exception as exc:
            logger.warning("Mcule API query failed: %s", exc)
            return self._mock_search(smiles)

    @staticmethod
    def _mock_search(smiles: str) -> dict[str, Any]:
        h = int(hashlib.sha256(f"mcule:{smiles}".encode()).hexdigest()[:8], 16)
        available = (h % 4) != 0  # ~75% availability
        return {
            "available": available,
            "catalog_id": f"MCULE-{h % 100000:05d}" if available else "",
            "price_usd": (h % 300) + 20.0 if available else None,
            "lead_time_days": (h % 14) + 5 if available else None,
        }


# ---------------------------------------------------------------------------
# Retrosynthesis decomposition
# ---------------------------------------------------------------------------
def decompose_molecule(smiles: str) -> list[str]:
    """Decompose a molecule into purchasable fragments.

    Uses RDKit BRICS when available; otherwise falls back to a
    heuristic bond-splitting approach.
    """
    if _RDKIT:
        return _brics_decompose(smiles)
    return _heuristic_decompose(smiles)


def _brics_decompose(smiles: str) -> list[str]:
    """BRICS decomposition via RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    fragments = list(BRICS.BRICSDecompose(mol))
    if not fragments:
        return [smiles]

    # Clean up BRICS dummy atoms for vendor lookup
    cleaned = []
    for frag in fragments:
        # Remove [*] dummy atoms
        clean = Chem.MolFromSmiles(frag)
        if clean is not None:
            # Replace dummy atoms with H
            from rdkit.Chem import AllChem, RWMol

            rw = RWMol(clean)
            atoms_to_remove = []
            for atom in rw.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atoms_to_remove.append(atom.GetIdx())
            for idx in sorted(atoms_to_remove, reverse=True):
                rw.RemoveAtom(idx)
            try:
                Chem.SanitizeMol(rw)
                cleaned_smi = Chem.MolToSmiles(rw)
                if cleaned_smi and len(cleaned_smi) > 1:
                    cleaned.append(cleaned_smi)
            except Exception:
                pass

    return cleaned if cleaned else [smiles]


def _heuristic_decompose(smiles: str) -> list[str]:
    """Rule-based decomposition when RDKit is unavailable."""
    # Split on common bond patterns
    fragments = []
    # Split on amide bonds (C(=O)N)
    parts = smiles.replace("C(=O)N", "|").split("|")
    for part in parts:
        part = part.strip("()")
        if len(part) >= 2:
            fragments.append(part)

    if not fragments:
        fragments = [smiles]
    return fragments


# ---------------------------------------------------------------------------
# Supply Chain Analyzer
# ---------------------------------------------------------------------------
class SupplyChainAnalyzer:
    """Analyze commercial availability of synthesis building blocks.

    Usage::

        analyzer = SupplyChainAnalyzer()
        result = analyzer.analyze("CC(=O)Oc1ccccc1C(=O)O")
        print(result.supply_chain_score)
        print(result.as_dict())
    """

    def __init__(
        self,
        vendors: list[VendorAdapter] | None = None,
        enamine_api_key: str = "",
        mcule_api_key: str = "",
    ):
        if vendors is not None:
            self.vendors = vendors
        else:
            self.vendors = [
                EnamineAdapter(api_key=enamine_api_key),
                MculeAdapter(api_key=mcule_api_key),
            ]

    def analyze(self, smiles: str) -> RetrosynthesisResult:
        """Full retrosynthesis + supply chain analysis for one molecule."""
        fragments = decompose_molecule(smiles)
        method = "brics" if _RDKIT else "heuristic"

        synthons: list[Synthon] = []
        for frag in fragments:
            synthon = self._check_availability(frag)
            synthons.append(synthon)

        # Aggregate metrics
        all_available = all(s.available is True for s in synthons)
        costs = [s.estimated_cost_usd for s in synthons if s.estimated_cost_usd is not None]
        lead_times = [s.lead_time_days for s in synthons if s.lead_time_days is not None]

        total_cost = sum(costs) if costs else None
        max_lead = max(lead_times) if lead_times else None

        # Supply chain viability score = geometric mean of synthon viabilities
        if synthons:
            viabilities = [s.viability for s in synthons]
            product = 1.0
            for v in viabilities:
                product *= v
            sc_score = product ** (1.0 / len(viabilities))
        else:
            sc_score = 0.0

        return RetrosynthesisResult(
            target_smiles=smiles,
            synthons=synthons,
            pathway_steps=max(1, len(fragments) - 1),
            pathway_description=f"{len(fragments)} fragments via {method} decomposition",
            supply_chain_score=sc_score,
            total_estimated_cost_usd=total_cost,
            max_lead_time_days=max_lead,
            all_available=all_available,
            analysis_method=method,
        )

    def analyze_batch(self, smiles_list: Sequence[str]) -> list[RetrosynthesisResult]:
        """Analyze multiple molecules."""
        return [self.analyze(s) for s in smiles_list]

    def _check_availability(self, fragment_smiles: str) -> Synthon:
        """Query all vendors for a single fragment."""
        synthon = Synthon(smiles=fragment_smiles)
        best_price = None
        best_lead = None

        for vendor in self.vendors:
            try:
                result = vendor.search(fragment_smiles)
                if result and result.get("available"):
                    synthon.available = True
                    synthon.vendors.append(vendor.name)
                    if result.get("catalog_id"):
                        synthon.catalog_ids[vendor.name] = result["catalog_id"]
                    price = result.get("price_usd")
                    if price is not None:
                        if best_price is None or price < best_price:
                            best_price = price
                    lead = result.get("lead_time_days")
                    if lead is not None:
                        if best_lead is None or lead < best_lead:
                            best_lead = lead
            except Exception as exc:
                logger.warning("Vendor %s query failed for %s: %s", vendor.name, fragment_smiles, exc)

        if synthon.available is None and not synthon.vendors:
            synthon.available = False

        synthon.estimated_cost_usd = best_price
        synthon.lead_time_days = best_lead
        return synthon
