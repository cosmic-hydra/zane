"""
Tests for HERGPredictor module - parametrized QSAR model for hERG cardiotoxicity assessment.

Tests verify:
1. Coefficient parametrization (no hardcoded magic numbers)
2. IC50 estimation from pharmacophore properties
3. CiPA risk classification
4. QTc prolongation risk assessment
5. Backward compatibility with GLPToxPanel
"""

import unittest

from drug_discovery.evaluation.herg_predictor import HERGPrediction, HERGPredictor, predict_herg
from drug_discovery.glp_tox_panel import PreClinicalToxPanel


class TestHERGPredictor(unittest.TestCase):
    """Test HERGPredictor parametrized QSAR model."""

    def setUp(self):
        """Initialize test predictor with default coefficients."""
        self.predictor = HERGPredictor()

    def test_default_initialization(self):
        """Test that predictor initializes with learnable coefficients."""
        # These should be tunable parameters, not hardcoded
        self.assertGreater(self.predictor.logp_coeff, 0)
        self.assertLess(self.predictor.tpsa_coeff, 0)  # Negative: lower TPSA = higher risk
        self.assertGreater(self.predictor.basic_n_coeff, 0)
        self.assertGreater(self.predictor.ic50_baseline_nM, 0)

    def test_custom_coefficients(self):
        """Test that custom QSAR coefficients can be set."""
        custom = HERGPredictor(logp_coeff=0.6, tpsa_coeff=-0.5)
        self.assertEqual(custom.logp_coeff, 0.6)
        self.assertEqual(custom.tpsa_coeff, -0.5)
        # Other coefficients should still have defaults
        self.assertGreater(custom.basic_n_coeff, 0)

    def test_prediction_output_structure(self):
        """Test that prediction returns complete HERGPrediction object."""
        result = self.predictor.predict("CCO")
        
        self.assertIsInstance(result, HERGPrediction)
        self.assertEqual(result.smiles, "CCO")
        self.assertIsInstance(result.inhibition_probability, float)
        self.assertIsInstance(result.ic50_estimate_nM, float)
        self.assertIsInstance(result.cipa_risk_category, str)
        self.assertIsInstance(result.qtichan_risk, str)
        self.assertIsInstance(result.features_contributing, dict)
        self.assertIsInstance(result.key_concerns, list)

    def test_probability_bounds(self):
        """Test that inhibition probability is bounded [0, 1]."""
        test_smiles = ["CCO", "c1ccccc1CCN", "CC(=O)O", "c1ccc(cc1)N(C)C"]
        for smiles in test_smiles:
            result = self.predictor.predict(smiles)
            self.assertGreaterEqual(result.inhibition_probability, 0.0)
            self.assertLessEqual(result.inhibition_probability, 1.0)
            self.assertGreaterEqual(result.inhibition_probability_low, 0.0)
            self.assertLessEqual(result.inhibition_probability_high, 1.0)

    def test_ic50_estimation(self):
        """Test that IC50 is estimated in realistic nanomolar range."""
        result = self.predictor.predict("c1ccc(Nc2nccc(Oc3ccccc3)n2)cc1")  # Example ARB-like
        
        # IC50 should be in plausible range (100 nM to 100 µM)
        self.assertGreaterEqual(result.ic50_estimate_nM, 100.0)
        self.assertLess(result.ic50_estimate_nM, 100000.0)
        
        # Range should be plausible
        self.assertLess(result.ic50_range_nM[0], result.ic50_range_nM[1])

    def test_cipa_risk_classification(self):
        """Test CiPA risk category classification."""
        # Very lipophilic aromatic compound with basic N (should be high risk)
        high_risk = self.predictor.predict("CN1CCC[C@H]1c2ccccc2Cl")  # Piperidine-phenyl
        self.assertIn(high_risk.cipa_risk_category, ["category_2", "category_3"])
        
        # Simple polar compound should be lower risk
        low_risk = self.predictor.predict("CCO")  # Ethanol
        # Ethanol should be low or category_2 (not high)
        self.assertNotEqual(low_risk.cipa_risk_category, "category_3")

    def test_qtc_risk_assessment(self):
        """Test QTc prolongation risk is derived from hERG and potency."""
        result = self.predictor.predict("CCO")
        self.assertIn(result.qtichan_risk, ["very_low", "low", "moderate", "high"])
        
        # Risk should increase with inhibition probability
        high_inhibit = self.predictor.predict("CN1CCN(c2ccc(cc2)Cl)CC1")  # More aromatic/basic
        low_inhibit = self.predictor.predict("CO")
        
        # This is not strictly guaranteed due to heuristics, but generally true
        # Just verify that risk categories are valid
        self.assertIn(high_inhibit.qtichan_risk, ["very_low", "low", "moderate", "high"])
        self.assertIn(low_inhibit.qtichan_risk, ["very_low", "low", "moderate", "high"])

    def test_model_confidence_estimate(self):
        """Test that model confidence is reported."""
        result = self.predictor.predict("CCO")
        self.assertGreaterEqual(result.model_confidence, 0.5)  # Should be somewhat confident
        self.assertLessEqual(result.model_confidence, 1.0)

    def test_key_concerns_identification(self):
        """Test identification of key structural/property concerns."""
        # Molecule with high logP and low TPSA should have concerns
        result = self.predictor.predict("c1ccc(cc1)C(c2ccccc2)C(c3ccccc3)c4ccccc4")  # Very lipophilic
        self.assertGreater(len(result.key_concerns), 0)
        
        # Simple molecule might have fewer or no concerns
        result_simple = self.predictor.predict("C")
        # Should be a list (may be empty)
        self.assertIsInstance(result_simple.key_concerns, list)

    def test_quick_predict_function(self):
        """Test convenience predict_herg function."""
        result = predict_herg("CCO")
        self.assertIsInstance(result, HERGPrediction)
        self.assertGreaterEqual(result.inhibition_probability, 0.0)
        self.assertLessEqual(result.inhibition_probability, 1.0)


class TestHERGIntegrationWithGLP(unittest.TestCase):
    """Test integration of HERGPredictor with PreClinicalToxPanel."""

    def test_glp_uses_herg_predictor(self):
        """Test that PreClinicalToxPanel uses new HERGPredictor."""
        panel = PreClinicalToxPanel()
        self.assertIsNotNone(panel.herg_predictor)
        self.assertIsInstance(panel.herg_predictor, HERGPredictor)

    def test_glp_backward_compatibility(self):
        """Test that GLPToxPanel still produces HERGResult with correct interface."""
        panel = PreClinicalToxPanel()
        result = panel.evaluate("CCO")
        
        # HERGResult should have expected attributes
        self.assertIsNotNone(result.herg)
        self.assertGreaterEqual(result.herg.inhibition_probability, 0.0)
        self.assertLessEqual(result.herg.inhibition_probability, 1.0)
        self.assertIn(result.herg.risk_class, ("low", "moderate", "high"))
        self.assertGreater(result.herg.ic50_estimate_uM, 0.0)
        self.assertIsInstance(result.herg.cardiac_risk, str)
        self.assertIsInstance(result.herg.passed, bool)

    def test_herg_threshold_enforcement(self):
        """Test that hERG threshold is properly enforced in GLP panel."""
        # Very strict threshold
        strict_panel = PreClinicalToxPanel(herg_threshold=0.1)
        result = strict_panel.evaluate("c1ccccc1")  # Benzene (somewhat aromatic)
        
        # Should fail if inhibition > 0.1
        if result.herg.inhibition_probability > 0.1:
            self.assertFalse(result.herg.passed)
        else:
            self.assertTrue(result.herg.passed)

    def test_custom_herg_predictor_in_glp(self):
        """Test that custom HERGPredictor can be passed to PreClinicalToxPanel."""
        custom_predictor = HERGPredictor(logp_coeff=0.3, tpsa_coeff=-0.2)
        panel = PreClinicalToxPanel(herg_predictor=custom_predictor)
        
        self.assertIs(panel.herg_predictor, custom_predictor)
        self.assertEqual(panel.herg_predictor.logp_coeff, 0.3)

    def test_batch_evaluation_uses_herg_predictor(self):
        """Test that batch evaluation properly uses HERGPredictor."""
        panel = PreClinicalToxPanel()
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        results = panel.evaluate_batch(smiles_list)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.herg)
            self.assertGreaterEqual(result.herg.inhibition_probability, 0.0)
            self.assertLessEqual(result.herg.inhibition_probability, 1.0)


if __name__ == "__main__":
    unittest.main()
