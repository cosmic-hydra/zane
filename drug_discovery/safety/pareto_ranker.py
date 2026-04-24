"""Multi-objective Pareto-optimal candidate ranker.

Given a set of drug candidates scored on multiple objectives (binding
affinity, toxicity, synthetic accessibility, drug-likeness), this module
identifies the Pareto front and ranks candidates by a weighted
scalarization with configurable objective weights.

The ranker is designed to sit between the surrogate filter and the final
selection step, ensuring that the pipeline delivers candidates that are
simultaneously high-affinity, low-toxicity, and synthesizable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveSpec:
    """Specification for a single optimisation objective."""

    name: str
    weight: float = 1.0
    minimize: bool = True  # True for delta_g, toxicity; False for QED
    ideal: float | None = None  # Optional ideal value for normalisation
    nadir: float | None = None  # Optional worst acceptable value


DEFAULT_OBJECTIVES = [
    ObjectiveSpec("delta_g", weight=2.0, minimize=True, ideal=-12.0, nadir=0.0),
    ObjectiveSpec("toxicity", weight=1.5, minimize=True, ideal=0.0, nadir=1.0),
    ObjectiveSpec("drug_likeness", weight=1.0, minimize=False, ideal=1.0, nadir=0.0),
    ObjectiveSpec("sa_score", weight=0.8, minimize=True, ideal=1.0, nadir=10.0),
]


@dataclass
class RankedCandidate:
    """A candidate with its multi-objective scores and rank."""

    smiles: str
    scores: dict[str, float] = field(default_factory=dict)
    scalarized_score: float = 0.0
    pareto_rank: int = 0  # 0 = Pareto-optimal front
    pareto_optimal: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "scores": self.scores,
            "scalarized_score": self.scalarized_score,
            "pareto_rank": self.pareto_rank,
            "pareto_optimal": self.pareto_optimal,
        }


class ParetoRanker:
    """Multi-objective Pareto ranking and weighted scalarization.

    Usage::

        ranker = ParetoRanker()
        candidates = [
            {"smiles": "CCO", "delta_g": -8.0, "toxicity": 0.1, "drug_likeness": 0.8, "sa_score": 2.0},
            {"smiles": "c1ccccc1", "delta_g": -6.0, "toxicity": 0.3, "drug_likeness": 0.7, "sa_score": 1.5},
        ]
        ranked = ranker.rank(candidates)
        top = ranker.select_top(candidates, k=5)
    """

    def __init__(self, objectives: list[ObjectiveSpec] | None = None):
        self.objectives = objectives or list(DEFAULT_OBJECTIVES)
        self._obj_names = {o.name for o in self.objectives}

    def rank(self, candidates: Sequence[dict[str, Any]]) -> list[RankedCandidate]:
        """Rank candidates by Pareto dominance and scalarized score.

        Args:
            candidates: List of dicts, each with ``"smiles"`` and
                objective values keyed by objective name.

        Returns:
            List of :class:`RankedCandidate` sorted by scalarized score
            (best first).
        """
        if not candidates:
            return []

        # Extract objective matrix
        n = len(candidates)
        obj_matrix = self._build_objective_matrix(candidates)

        # Normalise to [0, 1] where 0 is best
        norm_matrix = self._normalise(obj_matrix)

        # Compute Pareto ranks
        pareto_ranks = self._compute_pareto_ranks(norm_matrix)

        # Scalarize
        weights = np.array([o.weight for o in self.objectives])
        weights = weights / weights.sum()
        scalarized = (norm_matrix * weights).sum(axis=1)

        # Build results
        results = []
        for i, cand in enumerate(candidates):
            rc = RankedCandidate(
                smiles=cand.get("smiles", ""),
                scores={o.name: float(obj_matrix[i, j]) for j, o in enumerate(self.objectives)},
                scalarized_score=float(scalarized[i]),
                pareto_rank=int(pareto_ranks[i]),
                pareto_optimal=pareto_ranks[i] == 0,
            )
            results.append(rc)

        # Sort: Pareto rank first, then scalarized score (lower is better)
        results.sort(key=lambda r: (r.pareto_rank, r.scalarized_score))
        return results

    def select_top(
        self,
        candidates: Sequence[dict[str, Any]],
        k: int = 10,
        pareto_only: bool = False,
    ) -> list[RankedCandidate]:
        """Select the top-k candidates.

        Args:
            candidates: Candidate dicts.
            k: Number to select.
            pareto_only: If True, only select from the Pareto front.
        """
        ranked = self.rank(candidates)
        if pareto_only:
            ranked = [r for r in ranked if r.pareto_optimal]
        return ranked[:k]

    def pareto_front(self, candidates: Sequence[dict[str, Any]]) -> list[RankedCandidate]:
        """Return only Pareto-optimal candidates."""
        return self.select_top(candidates, k=len(candidates), pareto_only=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_objective_matrix(self, candidates: Sequence[dict[str, Any]]) -> np.ndarray:
        """Build (n_candidates, n_objectives) matrix of raw scores."""
        n = len(candidates)
        m = len(self.objectives)
        matrix = np.zeros((n, m))
        for i, cand in enumerate(candidates):
            for j, obj in enumerate(self.objectives):
                val = cand.get(obj.name, 0.0)
                matrix[i, j] = float(val) if val is not None else 0.0
        return matrix

    def _normalise(self, matrix: np.ndarray) -> np.ndarray:
        """Normalise objectives to [0, 1] where 0 = best."""
        norm = np.zeros_like(matrix)
        for j, obj in enumerate(self.objectives):
            col = matrix[:, j]
            if obj.ideal is not None and obj.nadir is not None:
                lo, hi = obj.ideal, obj.nadir
            else:
                lo, hi = col.min(), col.max()

            rng = hi - lo if hi != lo else 1.0

            if obj.minimize:
                norm[:, j] = np.clip((col - lo) / rng, 0.0, 1.0)
            else:
                norm[:, j] = np.clip((hi - col) / rng, 0.0, 1.0)

        return norm

    @staticmethod
    def _compute_pareto_ranks(norm_matrix: np.ndarray) -> np.ndarray:
        """Assign Pareto ranks (0 = front, 1 = second front, etc.)."""
        n = norm_matrix.shape[0]
        ranks = np.full(n, -1, dtype=int)
        remaining = set(range(n))
        current_rank = 0

        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(norm_matrix[j] <= norm_matrix[i]) and np.any(norm_matrix[j] < norm_matrix[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)

            for i in front:
                ranks[i] = current_rank
                remaining.discard(i)
            current_rank += 1

            if not front:
                # Safety valve: assign remaining to current rank
                for i in remaining:
                    ranks[i] = current_rank
                break

        return ranks
