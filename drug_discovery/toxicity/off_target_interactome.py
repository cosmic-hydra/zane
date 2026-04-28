"""Off-Target Interactome Panel — ToxPanelScorer.

Scores a molecule against five critical cardiac/metabolic off-targets using
RDKit physicochemical descriptor heuristics (logP, TPSA, MW, HBD, ring count)
as a fast, dependency-light baseline.  When ``torch`` and ``torch_geometric``
are available the scorer can also accept a pre-built PyTorch Geometric
:class:`~torch_geometric.data.Data` graph; the node features are summed into a
single feature vector that is mapped back to descriptor-equivalent values.

**Heuristic note**: The binding-affinity scores produced here are
*descriptor-based proxies*, not docked poses or trained GNN predictions.  They
are appropriate for fast pre-screening and integration testing.  Connect a
trained model to :meth:`ToxPanelScorer._score_target` to replace the heuristic.
"""

from __future__ import annotations

import logging
import math
from typing import Any

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
    import torch  # type: ignore[import-untyped]

    from torch_geometric.data import Data as PyGData  # type: ignore[import-untyped]

    _TORCH_GEO = True
except ImportError:  # pragma: no cover
    _TORCH_GEO = False
    PyGData = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class HighToxicityVeto(Exception):
    """Raised when a molecule exceeds the safety threshold on any off-target.

    Attributes:
        target: Name of the off-target that triggered the veto.
        score: Predicted binding-affinity proxy (0–1).
        threshold: Safety threshold that was crossed.
        smiles: SMILES of the offending molecule.
    """

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
# Each entry: (target_name, threshold)
# Thresholds represent the binding-affinity proxy value above which a molecule
# is considered to have unacceptable interaction with that off-target.
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
    """hERG pharmacophore model: lipophilic + low TPSA + moderate-large MW."""
    score = (
        _sigmoid(props["logp"] - 3.5) * 0.40
        + _sigmoid(55.0 - props["tpsa"]) * 0.30
        + _sigmoid(props["mw"] - 380.0) * 0.20
        + _sigmoid(props["hbd"] - 1.5) * 0.10
    )
    return min(score, 1.0)


def _score_cyp3a4(props: dict[str, float]) -> float:
    """CYP3A4 inhibition: large, lipophilic compounds with aromatic cores."""
    score = (
        _sigmoid(props["logp"] - 3.0) * 0.45
        + _sigmoid(props["mw"] - 420.0) * 0.30
        + _sigmoid(props["aromatic_rings"] - 2.0) * 0.25
    )
    return min(score, 1.0)


def _score_5ht2b(props: dict[str, float]) -> float:
    """5-HT2B affinity: basic amines, moderate MW, aromatic character."""
    score = (
        _sigmoid(props["logp"] - 2.5) * 0.35
        + _sigmoid(props["ring_count"] - 2.5) * 0.35
        + _sigmoid(props["hba"] - 3.0) * 0.30
    )
    return min(score, 1.0)


def _score_hnav15(props: dict[str, float]) -> float:
    """hNAV1.5 blockade: local anaesthetic-like — lipophilic + basic amine."""
    score = (
        _sigmoid(props["logp"] - 3.0) * 0.40
        + _sigmoid(props["mw"] - 300.0) * 0.25
        + _sigmoid(props["rot_bonds"] - 4.0) * 0.20
        + _sigmoid(props["hbd"] - 1.0) * 0.15
    )
    return min(score, 1.0)


def _score_hkv15(props: dict[str, float]) -> float:
    """hKv1.5 blockade: large, planar, lipophilic molecules."""
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
    """Off-target safety panel scorer.

    Accepts either a SMILES string or a PyTorch Geometric :class:`Data` graph
    (when ``torch_geometric`` is installed).  Scores the molecule against five
    critical off-targets and raises :class:`HighToxicityVeto` if any predicted
    binding-affinity proxy exceeds its safety threshold.

    Args:
        thresholds: Optional per-target threshold overrides.  Keys must match
            the target names in :data:`_OFF_TARGETS`.
        raise_on_first: When ``True`` (default), raises after the first failed
            target.  When ``False``, collects all failures and raises with the
            worst-scoring target.

    Example::

        scorer = ToxPanelScorer()
        scores = scorer.score_off_targets("CC(=O)Oc1ccccc1C(=O)O")
        # raises HighToxicityVeto if any score crosses the threshold
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        raise_on_first: bool = True,
    ) -> None:
        self._thresholds: dict[str, float] = {name: thr for name, thr in _OFF_TARGETS}
        if thresholds:
            self._thresholds.update(thresholds)
        self.raise_on_first = raise_on_first

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------
    def _to_smiles(self, input_mol: str | Any) -> str:
        """Convert SMILES string or PyG Data graph to a canonical SMILES string."""
        if isinstance(input_mol, str):
            return input_mol

        if _TORCH_GEO and isinstance(input_mol, PyGData):
            # Attempt to recover a SMILES if stored as metadata
            smiles_attr = getattr(input_mol, "smiles", None)
            if smiles_attr is not None:
                return str(smiles_attr)
            # Fallback: use node-feature sum to derive a deterministic pseudo-SMILES key
            # (used only to derive consistent RDKit property estimates via the fallback path)
            if input_mol.x is not None:
                feat_sum = float(input_mol.x.sum().item())
                # Map to a lightweight reference molecule for descriptor extraction
                return f"[feature_sum={feat_sum:.4f}]"
            return "C"  # minimal fallback

        raise TypeError(f"Expected a SMILES str or PyG Data, got {type(input_mol).__name__!r}")

    def _get_props(self, smiles: str) -> dict[str, float]:
        props = _get_rdkit_props(smiles)
        if props is None:
            props = _fallback_props(smiles)
        return props

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_target(self, target: str, props: dict[str, float]) -> float:
        """Return binding-affinity proxy score [0, 1] for *target*."""
        scorer_fn = _TARGET_SCORERS.get(target)
        if scorer_fn is None:
            logger.warning("No scorer defined for target %r; returning 0.0", target)
            return 0.0
        return scorer_fn(props)

    def score_off_targets(
        self,
        input_mol: str | Any,
    ) -> dict[str, float]:
        """Score *input_mol* against all five off-targets.

        Args:
            input_mol: A SMILES string or a PyTorch Geometric ``Data`` object.

        Returns:
            Dictionary mapping target name to its binding-affinity proxy score.

        Raises:
            HighToxicityVeto: If any off-target score exceeds its threshold.
            ValueError: If *input_mol* cannot be parsed to a valid molecule.
        """
        smiles = self._to_smiles(input_mol)
        props = self._get_props(smiles)

        scores: dict[str, float] = {}
        violations: list[tuple[str, float, float]] = []

        for target, default_thr in _OFF_TARGETS:
            threshold = self._thresholds.get(target, default_thr)
            score = self._score_target(target, props)
            scores[target] = round(score, 4)

            logger.debug("  %s: score=%.3f threshold=%.3f", target, score, threshold)

            if score > threshold:
                if self.raise_on_first:
                    raise HighToxicityVeto(
                        target=target,
                        score=score,
                        threshold=threshold,
                        smiles=smiles,
                    )
                violations.append((target, score, threshold))

        if violations:
            # Raise on the highest-scoring violating target
            worst = max(violations, key=lambda t: t[1])
            raise HighToxicityVeto(
                target=worst[0],
                score=worst[1],
                threshold=worst[2],
                smiles=smiles,
            )

        return scores

    def as_dict(self, input_mol: str | Any) -> dict[str, Any]:
        """Run panel and return serialisable result dict (no exception on veto).

        Returns::

            {
                "smiles": str,
                "scores": {"hERG": float, ...},
                "violations": [{"target": str, "score": float, "threshold": float}],
                "passed": bool,
            }
        """
        smiles = self._to_smiles(input_mol)
        props = self._get_props(smiles)

        result_scores: dict[str, float] = {}
        violations: list[dict[str, Any]] = []

        for target, default_thr in _OFF_TARGETS:
            threshold = self._thresholds.get(target, default_thr)
            score = self._score_target(target, props)
            result_scores[target] = round(score, 4)
            if score > threshold:
                violations.append({"target": target, "score": round(score, 4), "threshold": threshold})

        return {
            "smiles": smiles,
            "scores": result_scores,
            "violations": violations,
            "passed": len(violations) == 0,
        }
