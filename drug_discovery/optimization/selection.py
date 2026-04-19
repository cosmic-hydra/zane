"""Candidate selection utilities combining uncertainty, EHVI, and diversity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _tanimoto_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    inter = float(np.logical_and(fp_a, fp_b).sum())
    union = float(np.logical_or(fp_a, fp_b).sum())
    return 1.0 - (inter / union) if union > 0 else 1.0


def _fingerprint(smiles: str, n_bits: int = 512) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid smiles")
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=bool)
        for bit in fp.GetOnBits():
            arr[int(bit)] = True
        return arr
    except Exception:
        rng = np.random.default_rng(abs(hash(smiles)) % (2**32))
        return rng.integers(0, 2, size=n_bits, dtype=bool)


def expected_hypervolume_improvement(values: np.ndarray, reference_point: Sequence[float]) -> np.ndarray:
    ref = np.asarray(reference_point, dtype=np.float32)
    improvements = np.maximum(ref - values, 0.0)
    return np.prod(improvements, axis=1)


@dataclass
class CandidateSelectionConfig:
    top_k: int = 10
    high_uncertainty: int = 3
    diversity_weight: float = 0.2
    uncertainty_weight: float = 0.5
    ehvi_weight: float = 0.3
    reference_point: Sequence[float] = (0.0, 1.0)


class CandidateSelector:
    """Combine EHVI, MC-dropout uncertainty, and diversity to rank candidates."""

    def __init__(self, config: CandidateSelectionConfig | None = None):
        self.config = config or CandidateSelectionConfig()

    def _diversity_scores(self, smiles: Sequence[str]) -> List[float]:
        fps = [_fingerprint(smi) for smi in smiles]
        scores: List[float] = []
        for i, fp in enumerate(fps):
            others = [fps[j] for j in range(len(fps)) if j != i]
            if not others:
                scores.append(1.0)
                continue
            dists = [_tanimoto_distance(fp, o) for o in others]
            scores.append(float(np.mean(dists)))
        return scores

    def select(
        self,
        candidates: List[dict],
        metric_fn: Callable[[dict], float] | None = None,
    ) -> List[dict]:
        if not candidates:
            return []
        diversity = self._diversity_scores([c.get("smiles", f"c{i}") for i, c in enumerate(candidates)])
        scores = []
        metric_values = []
        for idx, cand in enumerate(candidates):
            base_metric = metric_fn(cand) if metric_fn else float(cand.get("predicted_property", 0.0))
            metric_values.append([base_metric, float(cand.get("qed_score", 0.0))])
            unc = float(cand.get("uncertainty", 0.0))
            div = diversity[idx]
            scores.append(
                self.config.ehvi_weight * base_metric + self.config.uncertainty_weight * unc + self.config.diversity_weight * div
            )
        metric_arr = np.asarray(metric_values, dtype=np.float32)
        ehvi = expected_hypervolume_improvement(metric_arr, reference_point=self.config.reference_point)
        ranked = []
        for idx, score in enumerate(scores):
            combined = float(score + 0.1 * ehvi[idx])
            ranked.append((combined, candidates[idx]))
            candidates[idx]["diversity"] = diversity[idx]
            candidates[idx]["ehvi"] = float(ehvi[idx])
        ranked.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in ranked[: self.config.top_k]]
        high_unc = sorted(candidates, key=lambda c: c.get("uncertainty", 0.0), reverse=True)[: self.config.high_uncertainty]
        merged = {id(c): c for c in top}
        for cand in high_unc:
            merged[id(cand)] = cand
        return list(merged.values())
