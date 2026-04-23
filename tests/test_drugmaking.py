"""
Tests for the drugmaking module.
"""

import pytest


class TestDrugmakingImports:
    """Test module imports and basic structure."""

    def test_imports(self):
        """Test that all expected classes can be imported."""
        from drug_discovery.drugmaking import (
            CustomDrugmakingModule,
            CompoundTestResult,
            CandidateResult,
            OptimizationConfig,
            CounterSubstanceFinder,
            CounterSubstanceResult,
        )
        assert CustomDrugmakingModule is not None
        assert CompoundTestResult is not None
        assert CandidateResult is not None
        assert OptimizationConfig is not None
        assert CounterSubstanceFinder is not None
        assert CounterSubstanceResult is not None

    def test_dataclass_instantiation(self):
        """Test that dataclasses can be instantiated."""
        from drug_discovery.drugmaking import (
            CompoundTestResult,
            CandidateResult,
            OptimizationConfig,
            CounterSubstanceResult,
        )

        opt_config = OptimizationConfig(
            objective_names=["potency", "safety"],
            num_iterations=5,
        )
        assert opt_config.objective_names == ["potency", "safety"]
        assert opt_config.num_iterations == 5

        test_result = CompoundTestResult(
            smiles="CCO",
            effectiveness=0.8,
            safety=0.9,
        )
        assert test_result.smiles == "CCO"
        assert test_result.effectiveness == 0.8
        assert test_result.safety == 0.9

        candidate = CandidateResult(
            smiles="CCO",
            objectives={"potency": 0.8, "safety": 0.9},
        )
        assert candidate.smiles == "CCO"
        assert candidate.objectives["potency"] == 0.8

        counter_result = CounterSubstanceResult(
            smiles="CCO",
            antagonism_score=-0.2,
            interaction_type="antagonistic",
        )
        assert counter_result.smiles == "CCO"
        assert counter_result.antagonism_score == -0.2
        assert counter_result.interaction_type == "antagonistic"


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from drug_discovery.drugmaking import OptimizationConfig

        config = OptimizationConfig()
        assert config.objective_names == ["potency", "selectivity", "solubility", "safety"]
        assert config.objective_directions == ["maximize", "maximize", "maximize", "maximize"]
        assert config.num_iterations == 20
        assert config.batch_size == 5

    def test_custom_config(self):
        """Test custom configuration."""
        from drug_discovery.drugmaking import OptimizationConfig

        config = OptimizationConfig(
            objective_names=["potency", "toxicity"],
            objective_directions=["maximize", "minimize"],
            ref_point=[0.0, 1.0],
            num_iterations=10,
        )
        assert len(config.objective_names) == 2
        assert config.objective_directions == ["maximize", "minimize"]
        assert config.ref_point == [0.0, 1.0]


class TestCompoundTestResult:
    """Test CompoundTestResult dataclass."""

    def test_as_dict(self):
        """Test as_dict method."""
        from drug_discovery.drugmaking import CompoundTestResult

        result = CompoundTestResult(
            smiles="CCO",
            effectiveness=0.8,
            toxicity_score=0.2,
            safety=0.8,
            admet_passed=True,
        )
        result_dict = result.as_dict()

        assert result_dict["smiles"] == "CCO"
        assert result_dict["effectiveness"] == 0.8
        assert result_dict["toxicity_score"] == 0.2
        assert result_dict["safety"] == 0.8
        assert result_dict["admet_passed"] is True


class TestCandidateResult:
    """Test CandidateResult dataclass."""

    def test_compute_composite_score(self):
        """Test composite score computation."""
        from drug_discovery.drugmaking import CandidateResult, OptimizationConfig

        candidate = CandidateResult(
            smiles="CCO",
            objectives={"potency": 0.8, "safety": 0.9},
        )
        config = OptimizationConfig(
            effectiveness_weight=0.6,
            safety_weight=0.4,
        )

        score = candidate.compute_composite_score(config)
        assert 0.0 <= score <= 1.0

    def test_as_dict(self):
        """Test as_dict method."""
        from drug_discovery.drugmaking import CandidateResult

        candidate = CandidateResult(
            smiles="CCO",
            objectives={"potency": 0.8},
            pareto_ranked=True,
            rank=0,
        )
        result_dict = candidate.as_dict()

        assert result_dict["smiles"] == "CCO"
        assert result_dict["pareto_ranked"] is True
        assert result_dict["rank"] == 0


class TestCounterSubstanceResult:
    """Test CounterSubstanceResult dataclass."""

    def test_compute_combined_score(self):
        """Test combined score computation."""
        from drug_discovery.drugmaking import CounterSubstanceResult

        result = CounterSubstanceResult(
            smiles="CCO",
            antagonism_score=-0.3,
            safety_score=0.9,
            efficacy_score=0.7,
        )

        score = result.compute_combined_score(
            antagonism_weight=0.5,
            safety_weight=0.3,
            efficacy_weight=0.2,
        )
        assert 0.0 <= score <= 1.0

    def test_as_dict(self):
        """Test as_dict method."""
        from drug_discovery.drugmaking import CounterSubstanceResult

        result = CounterSubstanceResult(
            smiles="CCO",
            antagonism_score=-0.3,
            interaction_type="antagonistic",
        )
        result_dict = result.as_dict()

        assert result_dict["smiles"] == "CCO"
        assert result_dict["antagonism_score"] == -0.3
        assert result_dict["interaction_type"] == "antagonistic"


class TestCustomDrugmakingModule:
    """Test CustomDrugmakingModule class."""

    def test_instantiation(self):
        """Test module instantiation."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        assert module is not None

    def test_instantiation_with_config(self):
        """Test module instantiation with config."""
        from drug_discovery.drugmaking import CustomDrugmakingModule, OptimizationConfig

        config = OptimizationConfig(num_iterations=5)
        module = CustomDrugmakingModule(optimization_config=config)
        assert module.optimization_config.num_iterations == 5

    def test_get_candidates_summary(self):
        """Test get_candidates_summary method."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        summary = module.get_candidates_summary()

        assert "count" in summary
        assert summary["count"] == 0


class TestCounterSubstanceFinder:
    """Test CounterSubstanceFinder class."""

    def test_instantiation(self):
        """Test finder instantiation."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()
        assert finder is not None

    def test_instantiation_with_params(self):
        """Test finder instantiation with parameters."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder(
            antagonism_threshold=-0.1,
            safety_threshold=0.5,
        )
        assert finder.antagonism_threshold == -0.1
        assert finder.safety_threshold == 0.5

    def test_get_counter_substance_summary(self):
        """Test get_counter_substance_summary method."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()
        summary = finder.get_counter_substance_summary(results=[])

        assert "count" in summary
        assert summary["count"] == 0
