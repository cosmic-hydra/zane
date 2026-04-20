"""
Tests for ADMET Predictor
"""

import numpy as np

from drug_discovery.evaluation import ADMETPredictor, ModelEvaluator


class TestADMETPredictor:
    """Test ADMET prediction functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.predictor = ADMETPredictor()
        self.aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.invalid_smiles = "INVALID"

    def test_lipinski_properties(self):
        """Test Lipinski properties calculation"""
        props = self.predictor.calculate_lipinski_properties(self.aspirin)

        assert props is not None
        assert "molecular_weight" in props
        assert "logp" in props
        assert "h_bond_donors" in props
        assert "h_bond_acceptors" in props

        # Aspirin should have reasonable values
        assert 100 < props["molecular_weight"] < 300
        assert props["h_bond_donors"] >= 0
        assert props["h_bond_acceptors"] >= 0

    def test_lipinski_rule(self):
        """Test Lipinski's Rule of Five"""
        result = self.predictor.check_lipinski_rule(self.aspirin)

        assert result is not None
        assert "passes" in result
        assert "violations" in result
        assert "properties" in result
        assert isinstance(result["passes"], bool)

    def test_qed_calculation(self):
        """Test QED calculation"""
        qed = self.predictor.calculate_qed(self.aspirin)

        assert qed is not None
        assert 0 <= qed <= 1

    def test_toxicity_flags(self):
        """Test toxicity flag prediction"""
        flags = self.predictor.predict_toxicity_flags(self.aspirin)

        assert flags is not None
        assert "contains_reactive_groups" in flags
        assert "potential_pains" in flags
        assert isinstance(flags["contains_reactive_groups"], bool)
        assert isinstance(flags["potential_pains"], bool)

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES"""
        props = self.predictor.calculate_lipinski_properties(self.invalid_smiles)
        assert props is None

        qed = self.predictor.calculate_qed(self.invalid_smiles)
        assert qed is None


class TestModelEvaluatorCalibration:
    """Test uncertainty calibration metrics for regression."""

    def setup_method(self):
        self.evaluator = ModelEvaluator()

    def test_expected_calibration_error_regression(self):
        y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        y_pred = np.array([0.1, 1.2, 1.8, 2.9], dtype=float)
        y_unc = np.array([0.05, 0.2, 0.15, 0.3], dtype=float)

        ece = self.evaluator.expected_calibration_error_regression(y_true, y_pred, y_unc, n_bins=5)
        assert 0.0 <= ece <= 1.0

    def test_prediction_interval_coverage(self):
        y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        y_pred = np.array([0.1, 1.0, 2.1, 3.0], dtype=float)
        y_std = np.array([0.2, 0.2, 0.2, 0.2], dtype=float)

        coverage = self.evaluator.prediction_interval_coverage(y_true, y_pred, y_std)
        assert 0.0 <= coverage <= 1.0


class TestScientificBenchmark:
    """Validate scientific benchmark integrity."""

    def test_r2_baseline_integrity(self):
        """Ensure benchmarked models beat a simple mean baseline."""
        from drug_discovery.benchmarking.moleculenet_eval import run_benchmark
        # Run a small-scale benchmark (1 epoch, 1 seed) to check pipeline integrity
        # In a real test, we might use a mock, but here we verify the logic flow.
        result = run_benchmark("bace", seeds=(42,), nb_epoch=1)
        
        assert result is not None
        assert "r2_mean" in result
        # Critical assertion: Model should be better than predicting the mean
        assert result["r2_mean"] > 0.0, "Model performance R² is below baseline; check data pipeline."

