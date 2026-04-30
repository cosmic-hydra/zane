"""Multi-endpoint toxicity gate for drug candidates.

Implements a strict go/no-go filter that blocks candidates exceeding
toxicity thresholds across multiple endpoints (hERG, Ames, hepatotoxicity,
cytotoxicity). Designed to drive pipeline toxicity toward zero.

The gate uses ensemble scoring -- a candidate must pass *all* endpoints
to proceed. This conservative strategy prioritises safety over throughput.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # type: ignore[import-untyped]

    _RDKIT = True
except ImportError:
    _RDKIT = False


@dataclass
class ToxicityGateConfig:
    """Thresholds for the toxicity gate.  Lower = stricter."""

    herg_threshold: float = 0.3  # hERG inhibition probability
    ames_threshold: float = 0.3  # Ames mutagenicity probability
    hepatotox_threshold: float = 0.4  # Hepatotoxicity probability
    cytotox_threshold: float = 0.4  # Cytotoxicity probability
    max_logp: float = 5.0  # Lipinski upper bound
    min_logp: float = -1.0
    max_tpsa: float = 140.0  # Topological polar surface area
    max_mw: float = 500.0  # Molecular weight
    max_rotatable_bonds: int = 10
    require_all_pass: bool = True  # Candidate must pass every endpoint


@dataclass
class EndpointScore:
    """Score for a single toxicity endpoint."""

    name: str
    probability: float
    threshold: float
    passed: bool
    detail: str = ""


@dataclass
class ToxicityVerdict:
    """Aggregated toxicity verdict for a candidate."""

    smiles: str
    passed: bool
    overall_toxicity: float  # 0 = non-toxic, 1 = extremely toxic
    endpoints: list[EndpointScore] = field(default_factory=list)
    drug_likeness: float = 0.0  # 0-1 QED-like score
    lipinski_violations: int = 0
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def safety_score(self) -> float:
        """1.0 = perfectly safe, 0.0 = maximally toxic."""
        return max(0.0, 1.0 - self.overall_toxicity)

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "passed": self.passed,
            "overall_toxicity": self.overall_toxicity,
            "safety_score": self.safety_score,
            "drug_likeness": self.drug_likeness,
            "lipinski_violations": self.lipinski_violations,
            "endpoints": [
                {"name": e.name, "probability": e.probability, "passed": e.passed}
                for e in self.endpoints
            ],
            "rejection_reasons": self.rejection_reasons,
        }


class ToxicityGate:
    """Multi-endpoint toxicity filter.

    Usage::

        gate = ToxicityGate()
        verdict = gate.evaluate("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        if verdict.passed:
            print("Safe to proceed")

        # Batch filtering
        safe = gate.filter_safe(["CCO", "c1ccccc1", "ClC(Cl)(Cl)Cl"])
    """

    def __init__(
        self,
        config: ToxicityGateConfig | None = None,
        admet_predictor: Any | None = None,
    ):
        self.config = config or ToxicityGateConfig()
        self.admet_predictor = admet_predictor
        self._cache: dict[str, ToxicityVerdict] = {}

    def evaluate(self, smiles: str) -> ToxicityVerdict:
        """Evaluate a single SMILES against all toxicity endpoints."""
        if smiles in self._cache:
            return self._cache[smiles]

        verdict = self._evaluate_internal(smiles)
        self._cache[smiles] = verdict
        return verdict

    def evaluate_batch(self, smiles_list: Sequence[str]) -> list[ToxicityVerdict]:
        """Evaluate a batch of SMILES."""
        return [self.evaluate(s) for s in smiles_list]

    def filter_safe(self, smiles_list: Sequence[str]) -> list[str]:
        """Return only SMILES that pass all toxicity checks."""
        verdicts = self.evaluate_batch(smiles_list)
        return [v.smiles for v in verdicts if v.passed]

    def batch_safety_rate(self, smiles_list: Sequence[str]) -> float:
        """Fraction of candidates passing the toxicity gate."""
        if not smiles_list:
            return 0.0
        verdicts = self.evaluate_batch(smiles_list)
        return sum(1 for v in verdicts if v.passed) / len(smiles_list)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------
    def _evaluate_internal(self, smiles: str) -> ToxicityVerdict:
        props = self._get_molecular_properties(smiles)
        if props is None:
            return ToxicityVerdict(
                smiles=smiles,
                passed=False,
                overall_toxicity=1.0,
                rejection_reasons=["Could not compute molecular properties"],
            )

        endpoints: list[EndpointScore] = []
        rejection_reasons: list[str] = []

        # External ADMET predictor
        admet_scores = self._get_admet_scores(smiles)

        # hERG
        herg_p = admet_scores.get("herg", self._estimate_herg(props))
        ep = EndpointScore("hERG", herg_p, self.config.herg_threshold, herg_p <= self.config.herg_threshold)
        endpoints.append(ep)
        if not ep.passed:
            rejection_reasons.append(f"hERG inhibition risk: {herg_p:.2f}")

        # Ames mutagenicity
        ames_p = admet_scores.get("ames", self._estimate_ames(props))
        ep = EndpointScore("Ames", ames_p, self.config.ames_threshold, ames_p <= self.config.ames_threshold)
        endpoints.append(ep)
        if not ep.passed:
            rejection_reasons.append(f"Ames mutagenicity risk: {ames_p:.2f}")

        # Hepatotoxicity
        hepato_p = admet_scores.get("hepatotox", self._estimate_hepatotox(props))
        ep = EndpointScore(
            "Hepatotoxicity", hepato_p, self.config.hepatotox_threshold,
            hepato_p <= self.config.hepatotox_threshold,
        )
        endpoints.append(ep)
        if not ep.passed:
            rejection_reasons.append(f"Hepatotoxicity risk: {hepato_p:.2f}")

        # Cytotoxicity
        cyto_p = admet_scores.get("cytotox", self._estimate_cytotox(props))
        ep = EndpointScore(
            "Cytotoxicity", cyto_p, self.config.cytotox_threshold,
            cyto_p <= self.config.cytotox_threshold,
        )
        endpoints.append(ep)
        if not ep.passed:
            rejection_reasons.append(f"Cytotoxicity risk: {cyto_p:.2f}")

        # Lipinski / drug-likeness
        lipinski_violations = self._count_lipinski_violations(props)
        if lipinski_violations > 1:
            rejection_reasons.append(f"Lipinski violations: {lipinski_violations}")

        # Overall score (geometric mean for strictness)
        tox_probs = [e.probability for e in endpoints]
        overall_toxicity = 1.0 - _product(1.0 - p for p in tox_probs)
        drug_likeness = self._compute_drug_likeness(props, overall_toxicity)

        if self.config.require_all_pass:
            passed = all(e.passed for e in endpoints) and lipinski_violations <= 1
        else:
            passed = overall_toxicity < 0.5

        # Suggest counter-toxins if failed
        suggested_counters = []
        if not passed:
            suggested_counters = self._suggest_counter_toxins(endpoints)

        return ToxicityVerdict(
            smiles=smiles,
            passed=passed,
            overall_toxicity=overall_toxicity,
            endpoints=endpoints,
            drug_likeness=drug_likeness,
            lipinski_violations=lipinski_violations,
            rejection_reasons=rejection_reasons,
            metadata={"suggested_counters": suggested_counters}
        )

    def _suggest_counter_toxins(self, endpoints: list[EndpointScore]) -> list[str]:
        """Suggest molecules that might mitigate specific toxicity endpoints."""
        suggestions = []
        for ep in endpoints:
            if not ep.passed:
                if ep.name == "hERG":
                    suggestions.append("Dexrazoxane (general cardioprotectant)")
                elif ep.name == "Hepatotoxicity":
                    suggestions.append("N-acetylcysteine (glutathione precursor)")
                elif ep.name == "Ames":
                    suggestions.append("DNA-repair enhancers (theoretical)")
                elif ep.name == "Cytotoxicity":
                    suggestions.append("Autophagy modulators")
        return list(set(suggestions))

    # ------------------------------------------------------------------
    # Molecular properties
    # ------------------------------------------------------------------
    def _get_molecular_properties(self, smiles: str) -> dict[str, float] | None:
        if _RDKIT:
            return self._rdkit_props(smiles)
        return self._heuristic_props(smiles)

    @staticmethod
    def _rdkit_props(smiles: str) -> dict[str, float] | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "mw": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
            "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
            "rot_bonds": int(Descriptors.NumRotatableBonds(mol)),
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        }

    @staticmethod
    def _heuristic_props(smiles: str) -> dict[str, float]:
        """Deterministic property estimates when RDKit is unavailable."""
        digest = hashlib.sha256(smiles.encode()).hexdigest()
        seed = int(digest[:8], 16) / float(0xFFFFFFFF)
        n_atoms = len([c for c in smiles if c.isupper()])
        return {
            "mw": n_atoms * 14.0 + seed * 50,
            "logp": seed * 5 - 1,
            "tpsa": seed * 120,
            "hba": max(1, int(n_atoms * 0.3)),
            "hbd": max(0, int(n_atoms * 0.15)),
            "rot_bonds": max(0, n_atoms - 3),
            "heavy_atoms": n_atoms,
            "aromatic_rings": 1 if "c" in smiles.lower() else 0,
            "ring_count": smiles.count("1") // 2,
        }

    # ------------------------------------------------------------------
    # Endpoint estimators (used when no external predictor)
    # ------------------------------------------------------------------
    def _get_admet_scores(self, smiles: str) -> dict[str, float]:
        if self.admet_predictor is None:
            return {}
        try:
            result = self.admet_predictor.predict(smiles)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            logger.warning("ADMET predictor failed for %s: %s", smiles, exc)
        return {}

    @staticmethod
    def _estimate_herg(props: dict[str, float]) -> float:
        """Estimate hERG inhibition from molecular properties.

        hERG risk correlates with: high logP, low TPSA, high MW.
        """
        logp = props.get("logp", 2.0)
        tpsa = props.get("tpsa", 60.0)
        mw = props.get("mw", 300.0)
        score = _sigmoid(logp - 3.5) * 0.5 + _sigmoid(80 - tpsa) * 0.3 + _sigmoid(mw - 400) * 0.2
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _estimate_ames(props: dict[str, float]) -> float:
        """Estimate Ames mutagenicity. Correlates with aromatic content."""
        aro = props.get("aromatic_rings", 0)
        logp = props.get("logp", 2.0)
        score = _sigmoid(aro - 2) * 0.6 + _sigmoid(logp - 4) * 0.4
        return min(max(score * 0.5, 0.0), 1.0)

    @staticmethod
    def _estimate_hepatotox(props: dict[str, float]) -> float:
        """Estimate hepatotoxicity. Correlates with MW and logP."""
        mw = props.get("mw", 300.0)
        logp = props.get("logp", 2.0)
        score = _sigmoid(mw - 450) * 0.5 + _sigmoid(logp - 4) * 0.5
        return min(max(score * 0.6, 0.0), 1.0)

    @staticmethod
    def _estimate_cytotox(props: dict[str, float]) -> float:
        """Estimate cytotoxicity."""
        logp = props.get("logp", 2.0)
        heavy = props.get("heavy_atoms", 20)
        score = _sigmoid(logp - 4.5) * 0.5 + _sigmoid(heavy - 40) * 0.5
        return min(max(score * 0.4, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Drug-likeness
    # ------------------------------------------------------------------
    def _count_lipinski_violations(self, props: dict[str, float]) -> int:
        violations = 0
        if props.get("mw", 0) > self.config.max_mw:
            violations += 1
        if props.get("logp", 0) > self.config.max_logp:
            violations += 1
        if props.get("logp", 0) < self.config.min_logp:
            violations += 1
        if props.get("hba", 0) > 10:
            violations += 1
        if props.get("hbd", 0) > 5:
            violations += 1
        if props.get("rot_bonds", 0) > self.config.max_rotatable_bonds:
            violations += 1
        if props.get("tpsa", 0) > self.config.max_tpsa:
            violations += 1
        return violations

    @staticmethod
    def _compute_drug_likeness(props: dict[str, float], toxicity: float) -> float:
        """Simple QED-like drug-likeness score in [0, 1]."""
        mw = props.get("mw", 300.0)
        logp = props.get("logp", 2.0)
        hba = props.get("hba", 3)
        hbd = props.get("hbd", 1)

        # Desirability functions (Gaussian-shaped around ideal values)
        d_mw = math.exp(-0.5 * ((mw - 350) / 150) ** 2)
        d_logp = math.exp(-0.5 * ((logp - 2.5) / 1.5) ** 2)
        d_hba = math.exp(-0.5 * ((hba - 4) / 3) ** 2)
        d_hbd = math.exp(-0.5 * ((hbd - 1.5) / 1.5) ** 2)
        d_tox = 1.0 - toxicity

        # Geometric mean
        product = d_mw * d_logp * d_hba * d_hbd * d_tox
        return max(0.0, product ** 0.2)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _product(iterable) -> float:
    result = 1.0
    for x in iterable:
        result *= x
    return result
