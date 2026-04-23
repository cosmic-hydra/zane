"""
Counter-Substance Finder for Risk Mitigation.

Identifies molecules that can counteract or neutralize drug effects by predicting
antagonistic interactions. Uses drug combination testing to find compounds with
negative synergy scores relative to a target drug.

This module supports:
- Finding antagonistic molecules for a given drug
- Predicting counter-substance effectiveness
- Scoring potential antidotes based on multiple criteria
- Batch screening of compound libraries

References:
    - Bliss independence model: Bliss (1939) Annals Applied Biology
    - Loewe additivity: Loewe (1928) Ergeb Physiol
    - Drug synergy/antagonism: Greco et al. (1995) Pharmacological Reviews
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from drug_discovery.testing.drug_combinations import DrugCombinationTester
    from drug_discovery.testing.toxicity import ToxicityPredictor

logger = logging.getLogger(__name__)


@dataclass
class CounterSubstanceResult:
    """Result for a counter-substance candidate.

    Attributes:
        smiles: SMILES string of the counter-substance.
        antagonism_score: Negative synergy score indicating antagonism (lower is more antagonistic).
        interaction_type: Type of interaction (antagonistic, additive, synergistic).
        safety_score: Safety score of the counter-substance (1 - toxicity).
        efficacy_score: Predicted efficacy as a counter-substance (0-1).
        combined_score: Combined score balancing antagonism and safety.
        details: Additional details about the prediction.
        success: Whether the prediction succeeded.
        error: Error message if prediction failed.
    """

    smiles: str
    antagonism_score: float = 0.0
    interaction_type: str = "unknown"
    safety_score: float = 0.5
    efficacy_score: float = 0.0
    combined_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    def compute_combined_score(
        self,
        antagonism_weight: float = 0.5,
        safety_weight: float = 0.3,
        efficacy_weight: float = 0.2,
    ) -> float:
        """
        Compute combined score for counter-substance ranking.

        Args:
            antagonism_weight: Weight for antagonism score (higher antagonism = better).
            safety_weight: Weight for safety score.
            efficacy_weight: Weight for efficacy score.

        Returns:
            Combined score (0-1, higher is better).
        """
        normalized_antagonism = min(1.0, max(0.0, -self.antagonism_score))

        self.combined_score = (
            normalized_antagonism * antagonism_weight
            + self.safety_score * safety_weight
            + self.efficacy_score * efficacy_weight
        )
        return self.combined_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "antagonism_score": self.antagonism_score,
            "interaction_type": self.interaction_type,
            "safety_score": self.safety_score,
            "efficacy_score": self.efficacy_score,
            "combined_score": self.combined_score,
            "details": self.details,
            "success": self.success,
            "error": self.error,
        }


class CounterSubstanceFinder:
    """
    Finds counter-substances that can mitigate risks from target drugs.

    Uses drug combination testing to predict antagonistic interactions,
    then ranks candidates by combined score balancing antagonism, safety,
    and predicted efficacy.

    Example::

        finder = CounterSubstanceFinder()

        # Find counter-substances for a drug
        results = finder.find_counter_substances(
            drug_smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            candidate_pool=["CCO", "c1ccccc1", "CC(=O)O"],
            min_count=5,
        )

        for result in results:
            print(f"{result.smiles}: combined_score={result.combined_score:.3f}")
    """

    def __init__(
        self,
        use_ml_models: bool = True,
        antagonism_threshold: float = -0.05,
        safety_threshold: float = 0.3,
    ):
        """
        Initialize the counter-substance finder.

        Args:
            use_ml_models: Whether to use ML models for synergy prediction.
            antagonism_threshold: Minimum antagonism score to consider a counter-substance.
            safety_threshold: Minimum safety score for a valid counter-substance.
        """
        self._combination_tester: "DrugCombinationTester | None" = None
        self._toxicity_predictor: "ToxicityPredictor | None" = None

        self.use_ml_models = use_ml_models
        self.antagonism_threshold = antagonism_threshold
        self.safety_threshold = safety_threshold

        self._default_candidates = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "O",  # water
            "CC(=O)OC(C)=O",  # acetic anhydride
            "c1ccccc1",  # benzene
            "CC(=O)OC(=O)C",  # acetyl chloride precursor
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
            "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@H](O)[C@H]1O",  # ATP-like
            "CC(C)C[C@@H](C(=O)O)CHNH2",  # leucine
            "O=C(O)C=CC(=O)O",  # maleic acid
        ]

        logger.info("CounterSubstanceFinder initialized")

    @property
    def combination_tester(self) -> "DrugCombinationTester":
        """Lazy-loaded drug combination tester."""
        if self._combination_tester is None:
            from drug_discovery.testing.drug_combinations import DrugCombinationTester
            self._combination_tester = DrugCombinationTester(use_ml_models=self.use_ml_models)
        return self._combination_tester

    @property
    def toxicity_predictor(self) -> "ToxicityPredictor":
        """Lazy-loaded toxicity predictor."""
        if self._toxicity_predictor is None:
            from drug_discovery.testing.toxicity import ToxicityPredictor
            self._toxicity_predictor = ToxicityPredictor()
        return self._toxicity_predictor

    def _compute_safety_score(self, smiles: str) -> float:
        """Compute safety score from toxicity prediction.

        Args:
            smiles: SMILES string.

        Returns:
            Safety score (0-1, higher is safer).
        """
        try:
            toxicity_results = self.toxicity_predictor.predict_all_toxicity_endpoints(smiles)
            overall_toxicity = toxicity_results["overall"]["toxicity_score"]
            return 1.0 - overall_toxicity
        except Exception as e:
            logger.warning(f"Toxicity prediction failed for {smiles}: {e}")
            return 0.5

    def _compute_efficacy_score(
        self,
        smiles: str,
        drug_smiles: str,
        antagonism_score: float,
    ) -> float:
        """
        Compute efficacy score for counter-substance.

        Combines structural diversity (different from drug) with antagonism strength.

        Args:
            smiles: Counter-substance SMILES.
            drug_smiles: Target drug SMILES.
            antagonism_score: Antagonism score from combination test.

        Returns:
            Efficacy score (0-1).
        """
        try:
            result = self.combination_tester.test_combination(
                smiles, drug_smiles, method="bliss", effect1=0.5, effect2=0.5
            )

            similarity = result.get("similarity", 0.5)

            diversity_score = 1.0 - similarity

            antagonism_contribution = min(1.0, max(0.0, -antagonism_score))

            efficacy = (
                diversity_score * 0.3
                + antagonism_contribution * 0.7
            )

            return min(1.0, max(0.0, efficacy))

        except Exception as e:
            logger.warning(f"Efficacy computation failed: {e}")
            return 0.3

    def test_counter_candidate(
        self,
        counter_smiles: str,
        drug_smiles: str,
    ) -> CounterSubstanceResult:
        """
        Test a single candidate as a counter-substance.

        Args:
            counter_smiles: SMILES of the candidate counter-substance.
            drug_smiles: SMILES of the target drug.

        Returns:
            CounterSubstanceResult with prediction details.
        """
        try:
            result = self.combination_tester.test_combination(
                counter_smiles, drug_smiles, method="ml"
            )

            antagonism_score = result.get("synergy_score", 0.0)
            interaction_type = result.get("interaction_type", "unknown")

            safety_score = self._compute_safety_score(counter_smiles)

            efficacy_score = self._compute_efficacy_score(
                counter_smiles, drug_smiles, antagonism_score
            )

            counter_result = CounterSubstanceResult(
                smiles=counter_smiles,
                antagonism_score=antagonism_score,
                interaction_type=interaction_type,
                safety_score=safety_score,
                efficacy_score=efficacy_score,
                details={
                    "combination_result": result,
                    "drug_smiles": drug_smiles,
                },
                success=True,
            )

            counter_result.compute_combined_score()

            return counter_result

        except Exception as e:
            logger.warning(f"Counter-substance test failed for {counter_smiles}: {e}")
            return CounterSubstanceResult(
                smiles=counter_smiles,
                success=False,
                error=str(e),
            )

    def find_counter_substances(
        self,
        drug_smiles: str,
        candidate_pool: list[str] | None = None,
        min_count: int = 5,
        max_count: int | None = None,
        use_default_pool: bool = True,
    ) -> list[CounterSubstanceResult]:
        """
        Find counter-substances for a given drug.

        Args:
            drug_smiles: SMILES string of the target drug.
            candidate_pool: Optional list of SMILES to test as counter-substances.
            min_count: Minimum number of counter-substances to find.
            max_count: Maximum number of counter-substances to return.
            use_default_pool: Whether to use the default candidate pool if pool is empty.

        Returns:
            List of CounterSubstanceResult sorted by combined score (best first).
        """
        candidates = candidate_pool or []

        if not candidates and use_default_pool:
            candidates = self._default_candidates.copy()
            logger.info(f"Using default candidate pool of {len(candidates)} molecules")

        if not candidates:
            logger.warning("No candidates provided and default pool disabled")
            return []

        logger.info(f"Testing {len(candidates)} candidates as counter-substances for drug")

        results: list[CounterSubstanceResult] = []

        for counter_smiles in candidates:
            result = self.test_counter_candidate(counter_smiles, drug_smiles)
            if result.success:
                results.append(result)

        results.sort(key=lambda x: x.combined_score, reverse=True)

        valid_results = [
            r for r in results
            if r.antagonism_score <= self.antagonism_threshold or r.safety_score >= self.safety_threshold
        ]

        if len(valid_results) < min_count:
            valid_results = results[:max(min_count, len(results))]
            logger.warning(
                f"Only found {len(valid_results)} candidates meeting strict criteria, "
                "returning top matches"
            )

        if max_count is not None:
            valid_results = valid_results[:max_count]

        antagonistic_count = sum(1 for r in valid_results if r.interaction_type == "antagonistic")

        logger.info(
            f"Found {len(valid_results)} counter-substances, "
            f"{antagonistic_count} with antagonistic interaction"
        )

        return valid_results

    def screen_library(
        self,
        drug_smiles: str,
        library_smiles: list[str],
        top_k: int = 10,
    ) -> list[CounterSubstanceResult]:
        """
        Screen a library of compounds to find counter-substances.

        Args:
            drug_smiles: SMILES of the target drug.
            library_smiles: List of SMILES to screen.
            top_k: Number of top candidates to return.

        Returns:
            List of top CounterSubstanceResult sorted by combined score.
        """
        logger.info(f"Screening library of {len(library_smiles)} compounds")

        results = self.find_counter_substances(
            drug_smiles=drug_smiles,
            candidate_pool=library_smiles,
            min_count=1,
            max_count=top_k,
            use_default_pool=False,
        )

        return results[:top_k]

    def get_counter_substance_summary(
        self,
        results: list[CounterSubstanceResult],
    ) -> dict[str, Any]:
        """
        Generate summary statistics for counter-substance search results.

        Args:
            results: List of CounterSubstanceResult from find_counter_substances.

        Returns:
            Dictionary with summary statistics.
        """
        if not results:
            return {
                "count": 0,
                "message": "No results to summarize",
            }

        antagonistic = [r for r in results if r.interaction_type == "antagonistic"]
        synergistic = [r for r in results if r.interaction_type == "synergistic"]
        additive = [r for r in results if r.interaction_type == "additive"]

        return {
            "total_candidates": len(results),
            "antagonistic_count": len(antagonistic),
            "synergistic_count": len(synergistic),
            "additive_count": len(additive),
            "avg_combined_score": float(np.mean([r.combined_score for r in results])),
            "best_candidate": results[0].smiles if results else None,
            "best_combined_score": float(results[0].combined_score) if results else 0.0,
            "avg_safety_score": float(np.mean([r.safety_score for r in results])),
            "avg_antagonism_score": float(np.mean([r.antagonism_score for r in results])),
        }
