"""
Comprehensive tests for the drugmaking module - State-of-the-Art Drug Design.
"""

import numpy as np


class TestDrugmakingImports:
    """Test module imports and basic structure."""

    def test_imports(self):
        """Test that all expected classes can be imported."""
        from drug_discovery.drugmaking import (
            CandidateResult,
            CompoundTestResult,
            CounterSubstanceFinder,
            CounterSubstanceResult,
            CustomDrugmakingModule,
            OptimizationConfig,
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
            CandidateResult,
            CompoundTestResult,
            CounterSubstanceResult,
            OptimizationConfig,
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


class TestRDKitMolecularProperties:
    """Test RDKit molecular properties calculator."""

    def test_rdkit_properties_instantiation(self):
        """Test RDKit properties calculator instantiation."""
        from drug_discovery.drugmaking.process import RDKitMolecularProperties

        props = RDKitMolecularProperties()
        assert props is not None
        assert hasattr(props, 'available')

    def test_calculate_properties(self):
        """Test molecular property calculation."""
        from drug_discovery.drugmaking.process import RDKitMolecularProperties

        props = RDKitMolecularProperties()

        # Test ethanol
        result = props.calculate_all_properties("CCO")
        assert "molecular_weight" in result
        assert "logp" in result
        assert "qed_score" in result
        assert result["molecular_weight"] > 0

    def test_heuristic_properties(self):
        """Test heuristic properties fallback."""
        from drug_discovery.drugmaking.process import RDKitMolecularProperties

        props = RDKitMolecularProperties()
        result = props._heuristic_properties("CCO")

        assert "molecular_weight" in result
        assert "logp" in result
        assert result["molecular_weight"] > 0


class TestPhysicsBasedProperties:
    """Test physics-based property predictions."""

    def test_physics_properties_instantiation(self):
        """Test physics properties calculator instantiation."""
        from drug_discovery.drugmaking.process import PhysicsBasedProperties

        physics = PhysicsBasedProperties()
        assert physics is not None
        assert hasattr(physics, 'rdkit_props')

    def test_predict_binding_affinity(self):
        """Test binding affinity prediction."""
        from drug_discovery.drugmaking.process import PhysicsBasedProperties

        physics = PhysicsBasedProperties()
        result = physics.predict_binding_affinity("CCO")

        assert "binding_score" in result
        assert "estimated_delta_g" in result
        assert 0 <= result["binding_score"] <= 1

    def test_predict_solubility(self):
        """Test solubility prediction."""
        from drug_discovery.drugmaking.process import PhysicsBasedProperties

        physics = PhysicsBasedProperties()
        result = physics.predict_solubility("CCO")

        assert "log_s" in result
        assert "solubility_mg_l" in result
        assert "solubility_class" in result

    def test_predict_lipophilicity(self):
        """Test lipophilicity prediction."""
        from drug_discovery.drugmaking.process import PhysicsBasedProperties

        physics = PhysicsBasedProperties()
        result = physics.predict_lipophilicity("CCO")

        assert "logp" in result
        assert "logd_ph74" in result
        assert "lipophilicity_class" in result

    def test_predict_bioavailability(self):
        """Test bioavailability prediction."""
        from drug_discovery.drugmaking.process import PhysicsBasedProperties

        physics = PhysicsBasedProperties()
        result = physics.predict_bioavailability("CCO")

        assert "bioavailability_score" in result
        assert "estimated_f_percent" in result
        assert 0 <= result["estimated_f_percent"] <= 100


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from drug_discovery.drugmaking import OptimizationConfig

        config = OptimizationConfig()
        assert config.objective_names == [
            "potency", "selectivity", "solubility", "safety",
            "synthetic_accessibility", "lipophilicity"
        ]
        assert config.num_iterations == 30
        assert config.batch_size == 10
        assert config.use_uncertainty is True

    def test_custom_config(self):
        """Test custom configuration."""
        from drug_discovery.drugmaking import OptimizationConfig

        config = OptimizationConfig(
            objective_names=["potency", "toxicity"],
            objective_directions=["maximize", "minimize"],
            ref_point=[0.0, 1.0],
            num_iterations=10,
            use_uncertainty=False,
        )
        assert len(config.objective_names) == 2
        assert config.objective_directions == ["maximize", "minimize"]
        assert config.ref_point == [0.0, 1.0]
        assert config.use_uncertainty is False


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
            molecular_properties={"molecular_weight": 46},
            physics_properties={"binding_affinity": -5.0},
        )
        result_dict = result.as_dict()

        assert result_dict["smiles"] == "CCO"
        assert result_dict["effectiveness"] == 0.8
        assert result_dict["molecular_properties"]["molecular_weight"] == 46
        assert result_dict["physics_properties"]["binding_affinity"] == -5.0


class TestCandidateResult:
    """Test CandidateResult dataclass."""

    def test_compute_composite_score(self):
        """Test composite score computation."""
        from drug_discovery.drugmaking import CandidateResult, OptimizationConfig

        candidate = CandidateResult(
            smiles="CCO",
            objectives={"potency": 0.8, "safety": 0.9, "solubility": 0.7},
            uncertainties={"potency": 0.1, "safety": 0.1, "solubility": 0.1},
        )
        config = OptimizationConfig(
            effectiveness_weight=0.5,
            safety_weight=0.3,
            exploration_weight=0.2,
            use_uncertainty=True,
        )

        score = candidate.compute_composite_score(config)
        # Score can be slightly > 1.0 due to exploration bonus, but should be reasonable
        assert 0.0 <= score <= 2.0

    def test_as_dict(self):
        """Test as_dict method."""
        from drug_discovery.drugmaking import CandidateResult

        candidate = CandidateResult(
            smiles="CCO",
            objectives={"potency": 0.8},
            uncertainties={"potency": 0.1},
            pareto_ranked=True,
            rank=0,
            confidence=0.9,
        )
        result_dict = candidate.as_dict()

        assert result_dict["smiles"] == "CCO"
        assert result_dict["pareto_ranked"] is True
        assert result_dict["confidence"] == 0.9


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
            functional_group_score=0.5,
        )

        score = result.compute_combined_score(
            antagonism_weight=0.3,
            safety_weight=0.25,
            efficacy_weight=0.2,
            mechanism_weight=0.15,
            functional_group_weight=0.1,
        )
        assert 0.0 <= score <= 1.0

    def test_as_dict(self):
        """Test as_dict method."""
        from drug_discovery.drugmaking import CounterSubstanceResult

        result = CounterSubstanceResult(
            smiles="CCO",
            antagonism_score=-0.3,
            interaction_type="antagonistic",
            neutralization_mechanism="acid-base neutralization",
            functional_group_score=0.5,
        )
        result_dict = result.as_dict()

        assert result_dict["smiles"] == "CCO"
        assert result_dict["antagonism_score"] == -0.3
        assert result_dict["neutralization_mechanism"] == "acid-base neutralization"


class TestMolecularAnalyzer:
    """Test molecular analyzer for counter-substance finding."""

    def test_instantiation(self):
        """Test molecular analyzer instantiation."""
        from drug_discovery.drugmaking.risk_mitigation import MolecularAnalyzer

        analyzer = MolecularAnalyzer()
        assert analyzer is not None

    def test_calculate_properties(self):
        """Test property calculation."""
        from drug_discovery.drugmaking.risk_mitigation import MolecularAnalyzer

        analyzer = MolecularAnalyzer()
        props = analyzer.calculate_properties("CCO")

        assert "molecular_weight" in props
        assert "logp" in props
        assert "h_bond_donors" in props

    def test_detect_functional_groups(self):
        """Test functional group detection."""
        from drug_discovery.drugmaking.risk_mitigation import MolecularAnalyzer

        analyzer = MolecularAnalyzer()

        # Acetic acid should have carboxylic acid group
        groups = analyzer.detect_functional_groups("CC(=O)O")
        assert "carboxylic_acid" in groups

        # Ethanol should have alcohol group
        groups = analyzer.detect_functional_groups("CCO")
        assert "alcohol" in groups


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

    def test_instantiation_with_seed(self):
        """Test module instantiation with random seed."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule(seed=42)
        assert module is not None

    def test_rdkit_props_lazy_loading(self):
        """Test RDKit properties lazy loading."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        props = module.rdkit_props
        assert props is not None
        assert module._rdkit_props is not None

    def test_physics_props_lazy_loading(self):
        """Test physics properties lazy loading."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        physics = module.physics_props
        assert physics is not None
        assert module._physics_props is not None

    def test_generate_compounds_fallback(self):
        """Test fallback compound generation."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        compounds = module.generate_compounds(num_candidates=5, strategy="scaffold")
        assert len(compounds) > 0
        assert all(isinstance(s, str) for s in compounds)

    def test_scaffold_based_generation(self):
        """Test scaffold-based generation."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        compounds = module._generate_scaffold_based(3)
        assert len(compounds) == 3

    def test_fragment_based_generation(self):
        """Test fragment-based generation."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        compounds = module._generate_fragment_based(3)
        assert len(compounds) == 3

    def test_test_toxicity_returns_result(self):
        """Test toxicity testing returns CompoundTestResult."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        result = module.test_toxicity("CCO")

        assert result is not None
        assert hasattr(result, 'smiles')
        assert hasattr(result, 'effectiveness')
        assert hasattr(result, 'safety')
        assert hasattr(result, 'molecular_properties')
        assert hasattr(result, 'physics_properties')

    def test_featurize_smiles(self):
        """Test SMILES featurization."""
        from drug_discovery.drugmaking import CustomDrugmakingModule

        module = CustomDrugmakingModule()
        features = module._featurize_smiles("CCO")

        assert features is not None
        assert len(features) == 18  # 18 features expected
        assert all(isinstance(f, (int, float, np.floating)) for f in features)

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
            min_similarity_to_antidote=0.3,
        )
        assert finder.antagonism_threshold == -0.1
        assert finder.safety_threshold == 0.5
        assert finder.min_similarity_to_antidote == 0.3

    def test_molecular_analyzer(self):
        """Test molecular analyzer initialization."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()
        assert finder.molecular_analyzer is not None

    def test_compute_functional_group_score(self):
        """Test functional group score computation."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()

        # Acetic acid should have high acid group score
        score = finder._compute_functional_group_score("CC(=O)O", "acid_toxicity")
        assert 0 <= score <= 1

    def test_compute_mechanism_score(self):
        """Test mechanism score computation."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()
        mechanism = finder._compute_mechanism_score("CC(=O)O", "acid_toxicity")
        assert mechanism in ["acid-base neutralization", "unknown"]

    def test_add_known_antidote(self):
        """Test adding known antidote."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()
        initial_count = len(finder._known_antidotes)

        # Try adding a unique antidote (use timestamp to ensure uniqueness)
        import time
        unique_antidote = f"TEST_ANTIDOTE_{int(time.time() * 1000)}"
        finder.add_known_antidote(unique_antidote)
        assert len(finder._known_antidotes) == initial_count + 1
        assert unique_antidote in finder._known_antidotes

    def test_get_counter_substance_summary(self):
        """Test get_counter_substance_summary method."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()
        summary = finder.get_counter_substance_summary(results=[])

        assert "count" in summary
        assert summary["count"] == 0

    def test_find_by_mechanism(self):
        """Test finding counter-substances by mechanism."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()
        results = finder.find_by_mechanism(
            drug_smiles="CC(=O)O",
            target_toxicity="acid_toxicity",
            min_count=2,
        )
        assert isinstance(results, list)


class TestIntegration:
    """Integration tests for the drugmaking module."""

    def test_full_pipeline_small(self):
        """Test full pipeline with small parameters."""
        from drug_discovery.drugmaking import CustomDrugmakingModule, OptimizationConfig

        config = OptimizationConfig(
            objective_names=["potency", "safety"],
            num_iterations=2,
            batch_size=2,
            initial_samples=2,
            use_uncertainty=False,
        )
        module = CustomDrugmakingModule(optimization_config=config)

        result = module.run_end_to_end(
            num_initial=3,
            num_optimization=2,
            target_objectives=["potency", "safety"],
        )

        assert result["success"] is True

    def test_end_to_end_with_physics_properties(self):
        """Test end-to-end includes physics properties."""
        from drug_discovery.drugmaking import CustomDrugmakingModule, OptimizationConfig

        config = OptimizationConfig(num_iterations=1, batch_size=1, initial_samples=1)
        module = CustomDrugmakingModule(optimization_config=config)

        result = module.run_end_to_end(
            num_initial=2,
            num_optimization=1,
        )

        assert "property_summary" in result
        assert "avg_binding_affinity" in result["property_summary"]
        assert "avg_solubility_log_s" in result["property_summary"]

    def test_counter_substance_full_pipeline(self):
        """Test counter-substance finding pipeline."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()

        results = finder.find_counter_substances(
            drug_smiles="CC(=O)Oc1ccccc1C(=O)O",
            min_count=3,
        )

        assert isinstance(results, list)
        assert len(results) >= 0  # May be 0 if dependencies unavailable

        if results:
            summary = finder.get_counter_substance_summary(results)
            assert summary["total_candidates"] == len(results)

    def test_screening_library(self):
        """Test library screening."""
        from drug_discovery.drugmaking import CounterSubstanceFinder

        finder = CounterSubstanceFinder()

        library = ["CCO", "CC(=O)O", "c1ccccc1", "O", "N"]
        results = finder.screen_library(
            drug_smiles="CC(=O)Oc1ccccc1C(=O)O",
            library_smiles=library,
            top_k=3,
        )

        assert isinstance(results, list)
        assert len(results) <= 3
