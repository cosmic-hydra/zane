"""Active learning utilities: uncertainty-based sampling and batch selection."""

from __future__ import annotations

import heapq
from collections.abc import Sequence


class UncertaintySampler:
    """Selects examples with highest uncertainty and simple diversity heuristics.

    Usage:
        sampler = UncertaintySampler()
        selected = sampler.select_batch(smiles_list, uncertainties, batch_size=10)
    """

    def __init__(self, diversity_dedup: bool = True):
        self.diversity_dedup = diversity_dedup

    def _dedupe(self, items: Sequence[str], max_keep: int) -> list[str]:
        seen = set()
        out = []
        for s in items:
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max_keep:
                break
        return out

    def select_batch(
        self,
        candidates: Sequence[str],
        uncertainties: Sequence[float],
        batch_size: int = 16,
    ) -> list[str]:
        """Return batch of SMILES selected by highest uncertainty.

        candidates and uncertainties must be same length.
        """
        if len(candidates) != len(uncertainties):
            raise ValueError("candidates and uncertainties must be same length")
        # Use a max-heap of uncertainties
        heap: list[tuple[float, int]] = []
        for i, u in enumerate(uncertainties):
            heap.append((-float(u), i))
        heapq.heapify(heap)

        selected_indices = []
        while heap and len(selected_indices) < batch_size:
            _, idx = heapq.heappop(heap)
            selected_indices.append(idx)

        selected = [candidates[i] for i in selected_indices]
        if self.diversity_dedup:
            selected = self._dedupe(selected, batch_size)
        return selected

    def select_top_k_by_entropy(self, probs: Sequence[Sequence[float]], k: int) -> list[int]:
        """Select top-k indices by predictive entropy.

        probs: list of probability vectors per candidate
        returns list of indices
        """
        entropies = []
        for p in probs:
            import math

            e = 0.0
            for pi in p:
                if pi > 0:
                    e -= pi * math.log(pi)
            entropies.append(e)
        idxs = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)
        return idxs[:k]
