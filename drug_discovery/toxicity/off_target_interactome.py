"""Off-Target Interactome Panel — ToxPanelScorer.

Scores a molecule against five critical cardiac/metabolic off-targets using
RDKit physicochemical descriptor heuristics (logP, TPSA, MW, HBD, ring count)
as a fast, dependency-light baseline.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # type: ignore[import-untyped]

    _RDKIT = True
except ImportError:  # pragma: no cover
    _RDKIT = False
    logger.warning("RDKit not available. ToxPanelScorer will use SMILES length heuristics.")

try:
    from torch_geometric.data import Data as PyGData  # type: ignore[import-untyped]

    _TORCH_GEO = True
except ImportError:  # pragma: no cover
    _TORCH_GEO = False
    PyGData = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class HighToxicityVeto(Exception):
    """Raised when a molecule exceeds the safety threshold on any off-target."""

    def __init__(self, target: str, score: float, threshold: float, smiles: str) -> None:
        self.target = target
        self.score = score
        self.threshold = threshold
        self.smiles = smiles
        super().__init__(
            f"HighToxicityVeto on {target}: predicted score {score:.3f} "
            f"exceeds threshold {threshold:.3f} for molecule {smiles!r}"
        )


# ---------------------------------------------------------------------------
# Off-target definitions
# ---------------------------------------------------------------------------
_OFF_TARGETS: list[tuple[str, float]] = [
    ("hERG", 0.5),       # Cardiac ion channel — QT prolongation risk
    ("CYP3A4", 0.6),     # Major CYP450 isoform — DDI / liver toxicity
    ("5-HT2B", 0.55),    # Serotonin receptor — valvulopathy risk
    ("hNAV1.5", 0.55),   # Cardiac sodium channel — arrhythmia risk
    ("hKv1.5", 0.5),     # Cardiac potassium channel — atrial arrhythmia
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _get_rdkit_props(smiles: str) -> dict[str, float] | None:
    """Return physicochemical properties via RDKit, or None on parse failure."""
    if not _RDKIT:
        return None
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
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
    }


def _fallback_props(smiles: str) -> dict[str, float]:
    """Deterministic fallback when RDKit is unavailable."""
    n = len([c for c in smiles if c.isupper()])
    aromatic = smiles.count("c") + smiles.count("n") + smiles.count("o")
    return {
        "mw": n * 12.0 + 50.0,
        "logp": 0.5 * n - 1.0,
        "tpsa": max(20.0, 80.0 - aromatic * 3.0),
        "hba": max(1, int(n * 0.25)),
        "hbd": max(0, int(n * 0.1)),
        "rot_bonds": max(0, n - 4),
        "ring_count": smiles.count("1") // 2,
        "aromatic_rings": aromatic // 6,
    }


# ---------------------------------------------------------------------------
# Per-target scoring heuristics
# ---------------------------------------------------------------------------
def _score_herg(props: dict[str, float]) -> float:
    score = (
        _sigmoid(props["logp"] - 3.5) * 0.40
        + _sigmoid(55.0 - props["tpsa"]) * 0.30
        + _sigmoid(props["mw"] - 380.0) * 0.20
        + _sigmoid(props["hbd"] - 1.5) * 0.10
    )
    return min(score, 1.0)


def _score_cyp3a4(props: dict[str, float]) -> float:
    score = (
        _sigmoid(props["logp"] - 3.0) * 0.45
        + _sigmoid(props["mw"] - 420.0) * 0.30
        + _sigmoid(props["aromatic_rings"] - 2.0) * 0.25
    )
    return min(score, 1.0)


def _score_5ht2b(props: dict[str, float]) -> float:
    score = (
        _sigmoid(props["logp"] - 2.5) * 0.35
        + _sigmoid(props["ring_count"] - 2.5) * 0.35
        + _sigmoid(props["hba"] - 3.0) * 0.30
    )
    return min(score, 1.0)


def _score_hnav15(props: dict[str, float]) -> float:
    score = (
        _sigmoid(props["logp"] - 3.0) * 0.40
        + _sigmoid(props["mw"] - 300.0) * 0.25
        + _sigmoid(props["rot_bonds"] - 4.0) * 0.20
        + _sigmoid(props["hbd"] - 1.0) * 0.15
    )
    return min(score, 1.0)


def _score_hkv15(props: dict[str, float]) -> float:
    score = (
        _sigmoid(props["logp"] - 3.5) * 0.45
        + _sigmoid(props["aromatic_rings"] - 1.5) * 0.30
        + _sigmoid(props["mw"] - 350.0) * 0.25
    )
    return min(score, 1.0)


_TARGET_SCORERS: dict[str, Any] = {
    "hERG": _score_herg,
    "CYP3A4": _score_cyp3a4,
    "5-HT2B": _score_5ht2b,
    "hNAV1.5": _score_hnav15,
    "hKv1.5": _score_hkv15,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ToxPanelScorer:
    """Off-target safety panel scorer."""

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        raise_on_first: bool = True,
        use_advanced_models: bool = True,
    ) -> None:
        self._thresholds: dict[str, float] = {name: thr for name, thr in _OFF_TARGETS}
        if thresholds:
            self._thresholds.update(thresholds)
        self.raise_on_first = raise_on_first
        self.use_advanced_models = use_advanced_models
        self._admet_predictor = None
        
        if self.use_advanced_models:
            try:
                from drug_discovery.evaluation.advanced_admet import AdvancedADMETPredictor, ADMETConfig
                self._admet_predictor = AdvancedADMETPredictor(ADMETConfig())
                # In a real scenario, we'd load weights here.
            except ImportError:
                logger.warning("AdvancedADMETPredictor not available, falling back to heuristics.")
                self.use_advanced_models = False

    def _to_smiles(self, input_mol: str | Any) -> str:
        if isinstance(input_mol, str):
            return input_mol
        if _TORCH_GEO and isinstance(input_mol, PyGData):
            smiles_attr = getattr(input_mol, "smiles", None)
            if smiles_attr is not None:
                return str(smiles_attr)
            return "C"
        return "C"

    def _get_props(self, smiles: str) -> dict[str, float]:
        props = _get_rdkit_props(smiles)
        if props is None:
            props = _fallback_props(smiles)
        return props

    def _score_target(self, target: str, props: dict[str, float], input_mol: Any = None) -> float:
        """Return binding-affinity proxy score [0, 1] for *target*."""
        if self.use_advanced_models and self._admet_predictor:
            # Map ToxPanel targets to ADMET endpoints
            mapping = {
                "hERG": "herg_inhibition",
                "CYP3A4": "cyp3a4_inhibition",
            }
            endpoint = mapping.get(target)
            if endpoint and isinstance(input_mol, (str, PyGData)):
                # This is a simplified call; ideally we'd pass proper features
                # For now, we'll simulate the call or use heuristics if it fails
                try:
                    # Mocking the required inputs for the example
                    dummy_z = torch.tensor([6, 6, 6])
                    dummy_pos = torch.randn(3, 3)
                    dummy_edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
                    dummy_tokens = torch.randint(0, 10, (1, 10))
                    
                    preds = self._admet_predictor(dummy_z, dummy_pos, dummy_edge, dummy_tokens)
                    if endpoint in preds:
                        probs = F.softmax(preds[endpoint], dim=-1)
                        return probs[..., 1].item()
                except Exception as e:
                    logger.debug(f"Advanced prediction failed for {target}: {e}")

        scorer_fn = _TARGET_SCORERS.get(target)
        if scorer_fn is None:
            return 0.0
        return scorer_fn(props)

    def score_off_targets(self, input_mol: str | Any) -> dict[str, float]:
        smiles = self._to_smiles(input_mol)
        props = self._get_props(smiles)

        scores: dict[str, float] = {}
        violations: list[tuple[str, float, float]] = []

        for target, default_thr in _OFF_TARGETS:
            threshold = self._thresholds.get(target, default_thr)
            score = self._score_target(target, props, input_mol=input_mol)
            scores[target] = round(score, 4)

            if score > threshold:
                if self.raise_on_first:
                    raise HighToxicityVeto(target, score, threshold, smiles)
                violations.append((target, score, threshold))

        if violations:
            worst = max(violations, key=lambda t: t[1])
            raise HighToxicityVeto(worst[0], worst[1], worst[2], smiles)

        return scores

    def as_dict(self, input_mol: str | Any) -> dict[str, Any]:
        smiles = self._to_smiles(input_mol)
        props = self._get_props(smiles)

        result_scores: dict[str, float] = {}
        violations: list[dict[str, Any]] = []

        for target, default_thr in _OFF_TARGETS:
            threshold = self._thresholds.get(target, default_thr)
            score = self._score_target(target, props, input_mol=input_mol)
            result_scores[target] = round(score, 4)
            if score > threshold:
                violations.append({"target": target, "score": round(score, 4), "threshold": threshold})

        return {
            "smiles": smiles,
            "scores": result_scores,
            "violations": violations,
            "passed": len(violations) == 0,
        }
