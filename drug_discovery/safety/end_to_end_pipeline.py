"""End-to-end drug generation pipeline with safety guarantees.

Connects all components in a single workflow:

1. **Generate** candidates via GFlowNet (or any generator)
2. **Validate** SMILES via :class:`SmilesValidator` (target: 99.9 % success)
3. **Surrogate filter** via :class:`SurrogateModel` Expected Improvement
4. **Physics Oracle** FEP scoring for top candidates
5. **Toxicity gate** via :class:`ToxicityGate` (multi-endpoint)
6. **Pareto ranking** via :class:`ParetoRanker` (multi-objective)
7. **Return** ranked, safe, high-affinity candidates

Usage::

    pipeline = SafeGenerationPipeline(
        protein_pdb_path="target.pdb",
    )
    results = pipeline.run(num_candidates=1000, top_k=10)
    for r in results:
        print(r.smiles, r.scores)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the end-to-end pipeline."""

    num_candidates: int = 1000
    surrogate_top_fraction: float = 0.01  # top 1 % sent to Oracle
    oracle_top_k: int = 50
    final_top_k: int = 10
    validation_max_mw: float = 800.0
    validation_max_heavy_atoms: int = 80
    toxicity_require_all_pass: bool = True


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    candidates_generated: int = 0
    candidates_valid: int = 0
    validation_rate: float = 0.0
    candidates_surrogate_selected: int = 0
    candidates_oracle_scored: int = 0
    candidates_safe: int = 0
    safety_rate: float = 0.0
    final_candidates: list[dict[str, Any]] = field(default_factory=list)
    pareto_front_size: int = 0
    elapsed_seconds: float = 0.0
    best_delta_g: float | None = None
    best_drug_likeness: float | None = None

    def summary(self) -> str:
        lines = [
            f"Generated:          {self.candidates_generated}",
            f"Valid SMILES:        {self.candidates_valid} ({self.validation_rate:.1%})",
            f"Surrogate-selected:  {self.candidates_surrogate_selected}",
            f"Oracle-scored:       {self.candidates_oracle_scored}",
            f"Passed safety gate:  {self.candidates_safe} ({self.safety_rate:.1%})",
            f"Pareto front size:   {self.pareto_front_size}",
            f"Final candidates:    {len(self.final_candidates)}",
            f"Best delta-G:        {self.best_delta_g}",
            f"Best drug-likeness:  {self.best_drug_likeness}",
            f"Elapsed:             {self.elapsed_seconds:.2f}s",
        ]
        return "\n".join(lines)


class SafeGenerationPipeline:
    """End-to-end pipeline ensuring high-affinity, low-toxicity candidates.

    Chains validation, surrogate filtering, physics scoring, toxicity
    gating, and Pareto ranking into a single ``run()`` call.
    """

    def __init__(
        self,
        protein_pdb_path: str = "target.pdb",
        config: PipelineConfig | None = None,
        generator_fn: Callable[[int], list[str]] | None = None,
        physics_oracle: Any | None = None,
        surrogate: Any | None = None,
    ):
        self.protein_pdb_path = protein_pdb_path
        self.config = config or PipelineConfig()
        self._generator_fn = generator_fn
        self._physics_oracle = physics_oracle
        self._surrogate = surrogate

    def run(
        self,
        num_candidates: int | None = None,
        top_k: int | None = None,
        seed_smiles: list[str] | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            num_candidates: Override config.num_candidates.
            top_k: Override config.final_top_k.
            seed_smiles: Provide SMILES directly instead of generating.

        Returns:
            :class:`PipelineResult` with ranked safe candidates.
        """
        t0 = time.monotonic()
        n = num_candidates or self.config.num_candidates
        k = top_k or self.config.final_top_k
        result = PipelineResult()

        # ----------------------------------------------------------
        # Step 1: Generate
        # ----------------------------------------------------------
        if seed_smiles:
            raw_smiles = list(seed_smiles)
        else:
            raw_smiles = self._generate(n)
        result.candidates_generated = len(raw_smiles)
        logger.info("Step 1: Generated %d candidates", len(raw_smiles))

        # ----------------------------------------------------------
        # Step 2: Validate
        # ----------------------------------------------------------
        valid_smiles = self._validate(raw_smiles)
        result.candidates_valid = len(valid_smiles)
        result.validation_rate = len(valid_smiles) / max(len(raw_smiles), 1)
        logger.info("Step 2: %d valid (%.1f%%)", len(valid_smiles), result.validation_rate * 100)

        if not valid_smiles:
            result.elapsed_seconds = time.monotonic() - t0
            return result

        # ----------------------------------------------------------
        # Step 3: Surrogate filter
        # ----------------------------------------------------------
        selected = self._surrogate_filter(valid_smiles)
        result.candidates_surrogate_selected = len(selected)
        logger.info("Step 3: %d surrogate-selected", len(selected))

        # ----------------------------------------------------------
        # Step 4: Physics Oracle scoring
        # ----------------------------------------------------------
        scored = self._oracle_score(selected)
        result.candidates_oracle_scored = len(scored)
        logger.info("Step 4: %d oracle-scored", len(scored))

        # ----------------------------------------------------------
        # Step 5: Toxicity gate
        # ----------------------------------------------------------
        safe_candidates = self._toxicity_filter(scored)
        result.candidates_safe = len(safe_candidates)
        result.safety_rate = len(safe_candidates) / max(len(scored), 1)
        logger.info("Step 5: %d passed safety (%.1f%%)", len(safe_candidates), result.safety_rate * 100)

        if not safe_candidates:
            result.elapsed_seconds = time.monotonic() - t0
            return result

        # ----------------------------------------------------------
        # Step 6: Pareto ranking
        # ----------------------------------------------------------
        ranked = self._pareto_rank(safe_candidates, k)
        result.final_candidates = [r.as_dict() for r in ranked]
        result.pareto_front_size = sum(1 for r in ranked if r.pareto_optimal)

        if ranked:
            scores = [r.scores for r in ranked]
            dgs = [s.get("delta_g", 0) for s in scores if s.get("delta_g") is not None]
            dls = [s.get("drug_likeness", 0) for s in scores]
            result.best_delta_g = min(dgs) if dgs else None
            result.best_drug_likeness = max(dls) if dls else None

        result.elapsed_seconds = time.monotonic() - t0
        logger.info("Pipeline complete in %.2fs. %d final candidates.", result.elapsed_seconds, len(ranked))
        return result

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------
    def _generate(self, n: int) -> list[str]:
        if self._generator_fn:
            return self._generator_fn(n)

        # Default: diverse drug-like seed pool
        pool = [
            "CCO", "c1ccccc1", "CC(=O)O",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
            "OC(=O)c1ccccc1O",
            "CC(=O)NC1=CC=C(C=C1)O",
            "C1CCCCC1", "c1ccncc1",
            "C1=CC=C(C=C1)C(=O)O",
            "OC1=CC=CC=C1",
            "c1ccc(cc1)N",
            "CC(C)(C)c1ccc(cc1)O",
            "c1ccc2[nH]ccc2c1",
        ]
        return [pool[i % len(pool)] for i in range(n)]

    def _validate(self, smiles_list: list[str]) -> list[str]:
        from drug_discovery.safety.smiles_validator import SmilesValidator

        validator = SmilesValidator(
            max_molecular_weight=self.config.validation_max_mw,
            max_heavy_atoms=self.config.validation_max_heavy_atoms,
        )
        return validator.filter_valid(smiles_list)

    def _surrogate_filter(self, smiles_list: list[str]) -> list[str]:
        if self._surrogate is not None and self._surrogate.n_observations >= 2:
            self._surrogate.fit()
            return self._surrogate.select_top_candidates(
                smiles_list,
                top_fraction=self.config.surrogate_top_fraction,
                min_candidates=min(self.config.oracle_top_k, len(smiles_list)),
            )
        # No surrogate data yet -- pass all through (capped)
        return smiles_list[: self.config.oracle_top_k]

    def _oracle_score(self, smiles_list: list[str]) -> list[dict[str, Any]]:
        if self._physics_oracle is not None:
            results = self._physics_oracle.score_batch_sync(smiles_list)
            return [
                {"smiles": r.smiles, "delta_g": r.delta_g or 0.0}
                for r in results
            ]
        # Mock scoring
        import hashlib

        scored = []
        for s in smiles_list:
            h = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            scored.append({"smiles": s, "delta_g": -12.0 * h})
        return scored

    def _toxicity_filter(self, scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from drug_discovery.safety.toxicity_gate import ToxicityGate, ToxicityGateConfig

        gate_cfg = ToxicityGateConfig(require_all_pass=self.config.toxicity_require_all_pass)
        gate = ToxicityGate(config=gate_cfg)

        safe = []
        for cand in scored:
            verdict = gate.evaluate(cand["smiles"])
            if verdict.passed:
                cand["toxicity"] = verdict.overall_toxicity
                cand["drug_likeness"] = verdict.drug_likeness
                cand["sa_score"] = 3.0  # placeholder
                safe.append(cand)
        return safe

    def _pareto_rank(self, candidates: list[dict[str, Any]], k: int):
        from drug_discovery.safety.pareto_ranker import ParetoRanker

        ranker = ParetoRanker()
        return ranker.select_top(candidates, k=k)
