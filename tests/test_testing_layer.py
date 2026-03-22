"""
Test Suite for Drug Discovery Testing Layer

Tests for:
- Toxicity prediction (cytotoxicity, hepatotoxicity, cardiotoxicity, mutagenicity)
- Drug combination testing
- Robustness testing
- Uncertainty estimation
"""

import pytest
import numpy as np
import pandas as pd
from drug_discovery.testing.toxicity import ToxicityPredictor
from drug_discovery.testing.drug_combinations import DrugCombinationTester
from drug_discovery.testing.robustness import RobustnessTester
from drug_discovery.testing.uncertainty import UncertaintyEstimator


class TestToxicityPredictor:
    """Test ToxicityPredictor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.predictor = ToxicityPredictor(use_ensemble=False)
        self.test_smiles = [
            "CCO",  # Ethanol
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]

    def test_predict_cytotoxicity(self):
        """Test cytotoxicity prediction."""
        result = self.predictor.predict_cytotoxicity(self.test_smiles[0])

        assert "toxic" in result
        assert "confidence" in result
        assert 0 <= result["toxic"] <= 1
        assert 0 <= result["confidence"] <= 1

    def test_predict_hepatotoxicity(self):
        """Test hepatotoxicity prediction."""
        result = self.predictor.predict_hepatotoxicity(self.test_smiles[1])

        assert "toxic" in result
        assert "toxic_class" in result
        assert result["toxic_class"] in ["hepatotoxic", "non-hepatotoxic"]

    def test_predict_cardiotoxicity(self):
        """Test cardiotoxicity prediction."""
        result = self.predictor.predict_cardiotoxicity(
            self.test_smiles[2],
            include_herg=True,
        )

        assert "toxic" in result
        assert "herg_inhibitor" in result
        assert "herg_risk" in result

    def test_predict_mutagenicity(self):
        """Test mutagenicity prediction."""
        result = self.predictor.predict_mutagenicity(self.test_smiles[0])

        assert "mutagenic" in result
        assert "ames_positive" in result
        assert isinstance(result["ames_positive"], bool)

    def test_predict_all_endpoints(self):
        """Test prediction of all toxicity endpoints."""
        result = self.predictor.predict_all_toxicity_endpoints(self.test_smiles[1])

        assert "cytotoxicity" in result
        assert "hepatotoxicity" in result
        assert "cardiotoxicity" in result
        assert "mutagenicity" in result
        assert "overall" in result

    def test_batch_predict(self):
        """Test batch prediction."""
        df = self.predictor.batch_predict(self.test_smiles)

        assert len(df) == len(self.test_smiles)
        assert "smiles" in df.columns
        assert "cytotoxicity" in df.columns
        assert "overall_toxicity" in df.columns

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        result = self.predictor.predict_cytotoxicity("INVALID")

        assert result["toxic"] == 0.5  # Default value
        assert result["confidence"] == 0.0


class TestDrugCombinationTester:
    """Test DrugCombinationTester."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tester = DrugCombinationTester(use_ml_models=False)
        self.smiles1 = "CCO"
        self.smiles2 = "CC(=O)OC1=CC=CC=C1C(=O)O"

    def test_predict_synergy_bliss(self):
        """Test Bliss independence synergy prediction."""
        result = self.tester.predict_synergy_bliss(
            self.smiles1,
            self.smiles2,
            effect1=0.5,
            effect2=0.5,
        )

        assert "expected_effect" in result
        assert "synergy_score" in result
        assert "interaction_type" in result

    def test_predict_synergy_loewe(self):
        """Test Loewe additivity synergy prediction."""
        result = self.tester.predict_synergy_loewe(
            self.smiles1,
            self.smiles2,
            dose1=1.0,
            dose2=1.0,
            ic50_1=1.0,
            ic50_2=1.0,
        )

        assert "combination_index" in result
        assert "interaction_type" in result

    def test_test_combination(self):
        """Test drug combination testing."""
        result = self.tester.test_combination(
            self.smiles1,
            self.smiles2,
            method="bliss",
            effect1=0.3,
            effect2=0.4,
        )

        assert result is not None
        assert "interaction_type" in result

    def test_batch_test_combinations(self):
        """Test batch combination testing."""
        combinations = [
            (self.smiles1, self.smiles2),
            (self.smiles2, self.smiles1),
        ]

        df = self.tester.batch_test_combinations(combinations, method="ml")

        assert len(df) == len(combinations)
        assert "interaction_type" in df.columns


class TestRobustnessTester:
    """Test RobustnessTester."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tester = RobustnessTester(random_state=42)

    def test_smiles_perturbation_robustness(self):
        """Test SMILES perturbation robustness."""
        smiles_list = ["CCO", "CC(C)O"]

        def dummy_model(smiles):
            return 0.5

        result = self.tester.test_smiles_perturbation_robustness(
            dummy_model,
            smiles_list,
            tolerance=0.1,
        )

        assert "total_tested" in result
        assert "robustness_rate" in result
        assert 0 <= result["robustness_rate"] <= 1

    def test_distribution_shift(self):
        """Test distribution shift detection."""
        train_smiles = ["CCO", "CC(C)O"]
        test_smiles = ["CCCO", "CC(C)CO"]

        def dummy_model(smiles):
            return 0.5

        result = self.tester.test_distribution_shift(
            dummy_model,
            train_smiles,
            test_smiles,
        )

        assert "distribution_shift" in result
        assert "train_pred_mean" in result
        assert "test_pred_mean" in result


class TestUncertaintyEstimator:
    """Test UncertaintyEstimator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.estimator = UncertaintyEstimator()

    def test_estimate_ensemble_uncertainty(self):
        """Test ensemble uncertainty estimation."""
        predictions = [0.5, 0.6, 0.55, 0.52]

        result = self.estimator.estimate_ensemble_uncertainty(predictions)

        assert "mean_prediction" in result
        assert "std_prediction" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_estimate_bayesian_uncertainty(self):
        """Test Bayesian uncertainty estimation."""
        posterior_samples = np.random.randn(100, 5)

        result = self.estimator.estimate_bayesian_uncertainty(posterior_samples)

        assert "posterior_mean" in result
        assert "total_uncertainty" in result
        assert "credible_interval_95" in result

    def test_compute_prediction_confidence(self):
        """Test confidence computation."""
        confidence = self.estimator.compute_prediction_confidence(
            prediction=0.8,
            uncertainty=0.1,
            method="exponential",
        )

        assert 0 <= confidence <= 1

    def test_batch_uncertainty_estimation(self):
        """Test batch uncertainty estimation."""
        def dummy_model1(smiles):
            return 0.5

        def dummy_model2(smiles):
            return 0.6

        ensemble_models = [dummy_model1, dummy_model2]
        smiles_list = ["CCO", "CC(C)O"]

        df = self.estimator.batch_uncertainty_estimation(
            ensemble_models,
            smiles_list,
        )

        assert len(df) > 0
        assert "mean_prediction" in df.columns
        assert "epistemic_uncertainty" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
