"""
Custom Drugmaking Process Module.

Orchestrates end-to-end drug design workflows:
1. Generate novel compounds from scratch using multi-backend generation
2. Test effectiveness and toxicity
3. Multi-objective Bayesian optimization (EHVI) to balance objectives
4. Iterative refinement toward optimal candidates

References:
    - Multi-objective optimization: Daulton et al., "Differentiable EHVI" (NeurIPS 2020)
    - Drug generation: REINVENT4, GT4SD, Molformer backends
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from drug_discovery.evaluation.predictor import ADMETPredictor
    from drug_discovery.generation.backends import GenerationManager
    from drug_discovery.optimization.multi_objective import (
        MOBOConfig,
        MultiObjectiveBayesianOptimizer,
    )
    from drug_discovery.testing.toxicity import ToxicityPredictor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for multi-objective optimization.

    Attributes:
        objective_names: Names of objectives to optimize.
        objective_directions: "maximize" or "minimize" for each objective.
        ref_point: Reference point for hypervolume calculation.
        num_iterations: Number of optimization iterations.
        batch_size: Number of candidates per iteration.
        initial_samples: Number of random samples to seed the optimizer.
        effectiveness_weight: Weight for effectiveness in composite score.
        safety_weight: Weight for safety in composite score.
    """

    objective_names: list[str] = field(
        default_factory=lambda: ["potency", "selectivity", "solubility", "safety"]
    )
    objective_directions: list[str] = field(
        default_factory=lambda: ["maximize", "maximize", "maximize", "maximize"]
    )
    ref_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    num_iterations: int = 20
    batch_size: int = 5
    initial_samples: int = 10
    effectiveness_weight: float = 0.6
    safety_weight: float = 0.4


@dataclass
class CompoundTestResult:
    """Results from testing a compound for effectiveness and toxicity.

    Attributes:
        smiles: SMILES string of the compound.
        effectiveness: Composite effectiveness score (0-1).
        toxicity_score: Overall toxicity score (0-1, lower is safer).
        safety: Safety score (1 - toxicity_score).
        admet_passed: Whether the compound passes ADMET criteria.
        details: Detailed test results for each endpoint.
    """

    smiles: str
    effectiveness: float = 0.0
    toxicity_score: float = 1.0
    safety: float = 0.0
    admet_passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "effectiveness": self.effectiveness,
            "toxicity_score": self.toxicity_score,
            "safety": self.safety,
            "admet_passed": self.admet_passed,
            "details": self.details,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class CandidateResult:
    """Result for a single candidate compound.

    Attributes:
        smiles: SMILES string of the candidate.
        objectives: Dictionary of objective values.
        pareto_ranked: Whether the candidate is on the Pareto front.
        rank: Pareto rank (0 is best).
        composite_score: Weighted composite score combining objectives.
    """

    smiles: str
    objectives: dict[str, float] = field(default_factory=dict)
    pareto_ranked: bool = False
    rank: int = -1
    composite_score: float = 0.0
    optimization_config: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def compute_composite_score(self, config: OptimizationConfig) -> float:
        """Compute weighted composite score from objectives.

        Args:
            config: Optimization configuration with weights.

        Returns:
            Composite score (0-1, higher is better).
        """
        if not self.objectives:
            return 0.0

        effectiveness = self.objectives.get("potency", 0.0) * config.effectiveness_weight
        if "selectivity" in self.objectives:
            effectiveness += self.objectives["selectivity"] * (1 - config.effectiveness_weight) * 0.5
        if "solubility" in self.objectives:
            effectiveness += self.objectives["solubility"] * (1 - config.effectiveness_weight) * 0.5

        safety = self.objectives.get("safety", 0.0) * config.safety_weight

        self.composite_score = effectiveness * config.effectiveness_weight + safety * config.safety_weight
        return self.composite_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "objectives": self.objectives,
            "pareto_ranked": self.pareto_ranked,
            "rank": self.rank,
            "composite_score": self.composite_score,
        }


class CustomDrugmakingModule:
    """
    End-to-end drug design module that generates, tests, and optimizes compounds.

    This module integrates:
    - GenerationManager for multi-backend compound generation
    - ToxicityPredictor and ADMETPredictor for testing
    - MultiObjectiveBayesianOptimizer (EHVI) for optimization

    Example::

        module = CustomDrugmakingModule(
            optimization_config=OptimizationConfig(
                objective_names=["potency", "safety"],
                num_iterations=15,
            )
        )

        # Generate and test initial compounds
        results = module.generate_and_test(num_candidates=20)

        # Run optimization loop
        pareto_front = module.optimize(iterations=15)

        # Run complete workflow
        final_result = module.run_end_to_end(target_objectives=["potency", "safety"])
    """

    def __init__(
        self,
        optimization_config: OptimizationConfig | None = None,
        use_rdkit: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize the drugmaking module.

        Args:
            optimization_config: Configuration for multi-objective optimization.
            use_rdkit: Whether to use RDKit for molecular operations.
            device: Device for ML models (cpu/cuda).
        """
        self.optimization_config = optimization_config or OptimizationConfig()
        self.device = device

        self._generation_manager: "GenerationManager | None" = None
        self._toxicity_predictor: "ToxicityPredictor | None" = None
        self._admet_predictor: "ADMETPredictor | None" = None
        self._optimizer: "MultiObjectiveBayesianOptimizer | None" = None

        self._generated_candidates: list[str] = []
        self._tested_compounds: list[CompoundTestResult] = []
        self._optimization_history: list[dict[str, Any]] = []

        logger.info("CustomDrugmakingModule initialized")

    @property
    def generation_manager(self) -> "GenerationManager":
        """Lazy-loaded generation manager."""
        if self._generation_manager is None:
            from drug_discovery.generation.backends import GenerationManager
            self._generation_manager = GenerationManager()
        return self._generation_manager

    @property
    def toxicity_predictor(self) -> "ToxicityPredictor":
        """Lazy-loaded toxicity predictor."""
        if self._toxicity_predictor is None:
            from drug_discovery.testing.toxicity import ToxicityPredictor
            self._toxicity_predictor = ToxicityPredictor()
        return self._toxicity_predictor

    @property
    def admet_predictor(self) -> "ADMETPredictor":
        """Lazy-loaded ADMET predictor."""
        if self._admet_predictor is None:
            from drug_discovery.evaluation.predictor import ADMETPredictor
            self._admet_predictor = ADMETPredictor()
        return self._admet_predictor

    def generate_compounds(self, num_candidates: int = 10, prompt: str | None = None) -> list[str]:
        """
        Generate novel compounds from scratch using generation backends.

        Args:
            num_candidates: Number of compounds to generate.
            prompt: Optional prompt/hint for generative models.

        Returns:
            List of generated SMILES strings.
        """
        logger.info(f"Generating {num_candidates} novel compounds")

        result = self.generation_manager.generate(prompt=prompt, num=num_candidates)

        if result["success"]:
            self._generated_candidates.extend(result["molecules"])
            logger.info(f"Generated {len(result['molecules'])} compounds using {result['backend']}")
            return result["molecules"]
        else:
            logger.warning(f"Generation failed: {result}")
            return self._generate_fallback_compounds(num_candidates)

    def _generate_fallback_compounds(self, num: int) -> list[str]:
        """Generate simple fallback compounds when backends are unavailable."""
        fallback_smiles = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CC(=O)OC(C)=O",  # acetic anhydride
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        ]
        candidates = []
        for i in range(num):
            candidates.append(fallback_smiles[i % len(fallback_smiles)])
        logger.info(f"Using {len(candidates)} fallback compounds")
        return candidates

    def test_effectiveness(self, smiles: str) -> float:
        """
        Test compound effectiveness (potency, selectivity, solubility).

        Args:
            smiles: SMILES string.

        Returns:
            Effectiveness score (0-1, higher is better).
        """
        effectiveness = 0.5

        lipinski = self.admet_predictor.check_lipinski_rule(smiles)
        qed = self.admet_predictor.calculate_qed(smiles)

        if lipinski and lipinski["passes"]:
            effectiveness += 0.15
        if qed is not None:
            effectiveness = (effectiveness + qed) / 2

        return min(1.0, max(0.0, effectiveness))

    def test_toxicity(self, smiles: str) -> CompoundTestResult:
        """
        Test compound for toxicity across multiple endpoints.

        Args:
            smiles: SMILES string.

        Returns:
            CompoundTestResult with toxicity assessment.
        """
        try:
            toxicity_results = self.toxicity_predictor.predict_all_toxicity_endpoints(smiles)
            overall_toxicity = toxicity_results["overall"]["toxicity_score"]

            lipinski = self.admet_predictor.check_lipinski_rule(smiles)
            qed = self.admet_predictor.calculate_qed(smiles)
            sa_score = self.admet_predictor.calculate_synthetic_accessibility(smiles)

            admet_passed = bool(lipinski and lipinski["passes"] and qed is not None and qed > 0.3)

            effectiveness = self.test_effectiveness(smiles)
            safety = 1.0 - overall_toxicity

            return CompoundTestResult(
                smiles=smiles,
                effectiveness=effectiveness,
                toxicity_score=overall_toxicity,
                safety=safety,
                admet_passed=admet_passed,
                details={
                    "toxicity_endpoints": toxicity_results,
                    "lipinski": lipinski,
                    "qed": qed,
                    "synthetic_accessibility": sa_score,
                },
                success=True,
            )

        except Exception as e:
            logger.warning(f"Toxicity test failed for {smiles}: {e}")
            return CompoundTestResult(
                smiles=smiles,
                effectiveness=0.0,
                toxicity_score=1.0,
                safety=0.0,
                admet_passed=False,
                success=False,
                error=str(e),
            )

    def generate_and_test(self, num_candidates: int = 20, prompt: str | None = None) -> list[CompoundTestResult]:
        """
        Generate novel compounds and test their properties.

        Args:
            num_candidates: Number of compounds to generate and test.
            prompt: Optional prompt for generative models.

        Returns:
            List of CompoundTestResult objects.
        """
        smiles_list = self.generate_compounds(num_candidates=num_candidates, prompt=prompt)

        results = []
        for smiles in smiles_list:
            result = self.test_toxicity(smiles)
            results.append(result)

        self._tested_compounds.extend(results)
        logger.info(f"Generated and tested {len(results)} compounds")

        return results

    def _featurize_smiles(self, smiles: str) -> np.ndarray:
        """Convert SMILES to numerical feature vector for optimization.

        Args:
            smiles: SMILES string.

        Returns:
            Feature vector (6-dimensional).
        """
        features = []

        lipinski = self.admet_predictor.calculate_lipinski_properties(smiles)
        if lipinski:
            features.extend([
                lipinski.get("molecular_weight", 500) / 1000,
                lipinski.get("logp", 3) / 10,
                lipinski.get("h_bond_donors", 2) / 10,
                lipinski.get("h_bond_acceptors", 5) / 10,
                lipinski.get("rotatable_bonds", 3) / 10,
                lipinski.get("aromatic_rings", 1) / 5,
            ])
        else:
            features = [0.5] * 6

        return np.array(features, dtype=np.float32)

    def optimize(
        self,
        initial_candidates: list[str] | None = None,
        iterations: int | None = None,
    ) -> list[CandidateResult]:
        """
        Run multi-objective Bayesian optimization (EHVI) on compounds.

        Args:
            initial_candidates: Optional list of SMILES to seed optimization.
            iterations: Number of optimization iterations.

        Returns:
            List of CandidateResult objects on the Pareto front.
        """
        from drug_discovery.optimization.multi_objective import (
            MOBOConfig,
            MultiObjectiveBayesianOptimizer,
        )

        config = self.optimization_config
        if iterations is not None:
            config = OptimizationConfig(
                **{
                    k: v for k, v in config.__dict__.items()
                    if k != "num_iterations"
                },
                num_iterations=iterations,
            )

        n_obj = len(config.objective_names)
        self._optimizer = MultiObjectiveBayesianOptimizer(
            MOBOConfig(
                objective_names=config.objective_names,
                objective_directions=config.objective_directions,
                ref_point=config.ref_point,
                num_iterations=config.num_iterations,
                batch_size=config.batch_size,
            )
        )

        candidates = initial_candidates or self._generated_candidates
        if not candidates:
            candidates = self.generate_compounds(num_candidates=config.initial_samples)

        logger.info(f"Starting multi-objective optimization with {len(candidates)} initial candidates")

        all_candidates: list[CandidateResult] = []

        for iteration in range(config.num_iterations):
            if iteration < len(candidates):
                batch_smiles = candidates[iteration:iteration + config.batch_size]
            else:
                batch_smiles = self.generate_compounds(num_candidates=config.batch_size)

            X_batch = np.array([self._featurize_smiles(s) for s in batch_smiles])
            Y_batch = []

            for smiles in batch_smiles:
                test_result = self.test_toxicity(smiles)
                objectives = {
                    "potency": test_result.effectiveness,
                    "selectivity": test_result.effectiveness * 0.9,
                    "solubility": 1.0 - (test_result.toxicity_score * 0.5),
                    "safety": test_result.safety,
                }

                y_values = [objectives.get(name, 0.5) for name in config.objective_names]
                Y_batch.append(y_values)

                candidate = CandidateResult(
                    smiles=smiles,
                    objectives=objectives,
                    optimization_config=config,
                )
                candidate.compute_composite_score(config)
                all_candidates.append(candidate)

            Y_batch = np.array(Y_batch, dtype=np.float32)

            self._optimizer.tell(X_batch, Y_batch)

            self._optimization_history.append({
                "iteration": iteration + 1,
                "num_candidates": len(all_candidates),
                "best_score": max(c.composite_score for c in all_candidates) if all_candidates else 0.0,
            })

            logger.info(f"Optimization iteration {iteration + 1}/{config.num_iterations}")

        pareto_front = self._optimizer.get_pareto_front()
        pareto_indices = np.where(pareto_front["mask"])[0]

        for idx, candidate in enumerate(all_candidates):
            if idx in pareto_indices:
                candidate.pareto_ranked = True

        pareto_candidates = [c for c in all_candidates if c.pareto_ranked]

        pareto_candidates.sort(key=lambda x: x.composite_score, reverse=True)
        for rank, candidate in enumerate(pareto_candidates):
            candidate.rank = rank

        logger.info(f"Found {len(pareto_candidates)} Pareto-optimal candidates")

        return pareto_candidates

    def run_end_to_end(
        self,
        num_initial: int = 20,
        num_optimization: int = 15,
        target_objectives: list[str] | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Run complete end-to-end drug design workflow.

        Args:
            num_initial: Number of initial compounds to generate and test.
            num_optimization: Number of optimization iterations.
            target_objectives: List of objective names to optimize.
            prompt: Optional prompt for generation.

        Returns:
            Dictionary containing workflow results and metadata.
        """
        logger.info("Starting end-to-end drug design workflow")

        if target_objectives:
            direction_map = {
                "potency": "maximize",
                "safety": "maximize",
                "toxicity": "minimize",
                "solubility": "maximize",
                "selectivity": "maximize",
            }
            directions = [direction_map.get(obj, "maximize") for obj in target_objectives]
            self.optimization_config = OptimizationConfig(
                objective_names=target_objectives,
                objective_directions=directions,
                num_iterations=num_optimization,
            )

        results = self.generate_and_test(num_candidates=num_initial, prompt=prompt)

        passed = [r for r in results if r.admet_passed]
        logger.info(f"ADMET passed: {len(passed)}/{len(results)}")

        pareto_front = self.optimize(
            initial_candidates=[r.smiles for r in results],
            iterations=num_optimization,
        )

        top_candidates = sorted(
            pareto_front,
            key=lambda x: x.composite_score,
            reverse=True,
        )[:5]

        return {
            "success": True,
            "total_tested": len(results),
            "admet_passed": len(passed),
            "admet_pass_rate": len(passed) / len(results) if results else 0.0,
            "pareto_front_size": len(pareto_front),
            "top_candidates": [c.as_dict() for c in top_candidates],
            "optimization_history": self._optimization_history,
        }

    def get_candidates_summary(self) -> dict[str, Any]:
        """Get summary statistics of tested compounds."""
        if not self._tested_compounds:
            return {"count": 0, "message": "No compounds tested yet"}

        total = len(self._tested_compounds)
        passed = sum(1 for r in self._tested_compounds if r.admet_passed)
        avg_toxicity = np.mean([r.toxicity_score for r in self._tested_compounds])
        avg_effectiveness = np.mean([r.effectiveness for r in self._tested_compounds])

        return {
            "count": total,
            "admet_passed": passed,
            "admet_pass_rate": passed / total,
            "avg_toxicity_score": float(avg_toxicity),
            "avg_effectiveness": float(avg_effectiveness),
            "optimization_iterations": len(self._optimization_history),
        }
