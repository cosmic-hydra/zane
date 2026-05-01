"""Tests for strict compliance and parametrized quality gates.

Comprehensive test suite covering:
- Strict compliance evaluation 
- Quality tier classification
- Data integrity verification
- Risk factor identification
- Parametrized threshold flexibility
- Regulatory tier switching
- Batch evaluation
"""

import unittest
from datetime import datetime

from drug_discovery.safety.parametrized_toxicity_gate import (
    ParametrizedToxicityGate,
    ToxicityThresholdConfig,
)
from drug_discovery.safety.strict_compliance_gate import (
    ComplianceLevel,
    QualityTier,
    RiskFactor,
    StrictComplianceGate,
    evaluate_batch_with_strict_compliance,
)


class TestToxicityThresholdConfig(unittest.TestCase):
    """Test parametrized threshold configuration."""

    def test_default_config(self):
        """Test default thresholds are initialized."""
        config = ToxicityThresholdConfig()
        self.assertEqual(config.herg_threshold, 0.3)
        self.assertEqual(config.ames_threshold, 0.3)
        self.assertEqual(config.hepatotox_threshold, 0.4)
        self.assertEqual(config.logp_max, 5.0)

    def test_config_validation(self):
        """Test invalid thresholds raise ValueError."""
        with self.assertRaises(ValueError):
            ToxicityThresholdConfig(herg_threshold=1.5)
        
        with self.assertRaises(ValueError):
            ToxicityThresholdConfig(logp_min=10, logp_max=5)

    def test_regulatory_tier_discovery(self):
        """Test discovery tier uses relaxed thresholds."""
        config = ToxicityThresholdConfig.from_regulatory_tier("discovery")
        self.assertEqual(config.herg_threshold, 0.5)
        self.assertEqual(config.ames_threshold, 0.4)
        self.assertEqual(config.min_prediction_confidence, 0.3)

    def test_regulatory_tier_ind(self):
        """Test IND tier uses strict thresholds."""
        config = ToxicityThresholdConfig.from_regulatory_tier("ind")
        self.assertEqual(config.herg_threshold, 0.25)
        self.assertEqual(config.ames_threshold, 0.15)
        self.assertEqual(config.hepatotox_threshold, 0.2)
        self.assertEqual(config.min_prediction_confidence, 0.6)

    def test_regulatory_tier_nda(self):
        """Test NDA tier uses strictest thresholds."""
        config = ToxicityThresholdConfig.from_regulatory_tier("nda")
        self.assertEqual(config.herg_threshold, 0.15)
        self.assertEqual(config.ames_threshold, 0.1)
        self.assertEqual(config.min_prediction_confidence, 0.75)

    def test_unknown_tier_raises_error(self):
        """Test unknown regulatory tier raises ValueError."""
        with self.assertRaises(ValueError):
            ToxicityThresholdConfig.from_regulatory_tier("unknown")


class TestParametrizedToxicityGate(unittest.TestCase):
    """Test parametrized toxicity gate behavior."""

    def test_default_gate_accepts_good_molecule(self):
        """Test default gate accepts molecule within thresholds."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(
            herg_prob=0.1,
            ames_prob=0.1,
            hepatotox_prob=0.2,
            logp=2.0,
            tpsa=80.0,
            mw=300.0,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["reasons"]), 0)

    def test_gate_rejects_high_herg(self):
        """Test gate rejects high hERG inhibition."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(herg_prob=0.5)
        self.assertFalse(result["passed"])
        self.assertTrue(any("hERG" in r for r in result["reasons"]))

    def test_gate_rejects_high_ames(self):
        """Test gate rejects high Ames mutagenicity."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(ames_prob=0.6)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Ames" in r for r in result["reasons"]))

    def test_gate_warns_on_elevated_herg(self):
        """Test gate warns when hERG is elevated but below threshold."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(herg_prob=0.25)
        self.assertTrue(result["passed"])
        self.assertTrue(any("warning" in w.lower() for w in result["warnings"]))

    def test_custom_thresholds(self):
        """Test gate respects custom threshold configuration."""
        config = ToxicityThresholdConfig(herg_threshold=0.1)
        gate = ParametrizedToxicityGate(config)
        result = gate.evaluate(herg_prob=0.15)
        self.assertFalse(result["passed"])
        self.assertEqual(result["active_thresholds"]["herg"], 0.1)

    def test_update_thresholds_at_runtime(self):
        """Test updating thresholds without recreating gate."""
        gate = ParametrizedToxicityGate()
        gate.update_thresholds(herg_threshold=0.1)
        result = gate.evaluate(herg_prob=0.15)
        self.assertFalse(result["passed"])

    def test_logp_range_check(self):
        """Test LogP is within configured range."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(logp=6.0)  # Above default max of 5.0
        self.assertFalse(result["passed"])
        self.assertTrue(any("LogP" in r for r in result["reasons"]))

    def test_mw_bounds_check(self):
        """Test molecular weight respects bounds."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(mw=600.0)  # Above default max of 500
        self.assertFalse(result["passed"])

    def test_tpsa_range_check(self):
        """Test TPSA within configured range."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(tpsa=150.0)  # Above default max of 140
        self.assertFalse(result["passed"])

    def test_confidence_requirement(self):
        """Test minimum prediction confidence."""
        config = ToxicityThresholdConfig(min_prediction_confidence=0.7)
        gate = ParametrizedToxicityGate(config)
        result = gate.evaluate(confidence=0.5)
        self.assertFalse(result["passed"])
        self.assertTrue(any("confidence" in r.lower() for r in result["reasons"]))

    def test_multiple_failures_consolidated(self):
        """Test multiple threshold violations are all reported."""
        gate = ParametrizedToxicityGate()
        result = gate.evaluate(
            herg_prob=0.5,
            ames_prob=0.6,
            logp=6.0,
        )
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(len(result["reasons"]), 3)


class TestStrictComplianceGate(unittest.TestCase):
    """Test strict compliance evaluation."""

    def test_invalid_smiles_rejected(self):
        """Test invalid SMILES strings are rejected."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate("invalid_smiles")
        self.assertFalse(assessment.overall_passed)
        self.assertEqual(assessment.quality_tier, QualityTier.REJECTED)

    def test_empty_smiles_rejected(self):
        """Test empty SMILES string is rejected."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate("")
        self.assertFalse(assessment.overall_passed)

    def test_valid_simple_molecule(self):
        """Test valid simple molecule like aspirin."""
        gate = StrictComplianceGate(compliance_level=ComplianceLevel.RELAXED)
        # Aspirin SMILES: CC(=O)Oc1ccccc1C(=O)O
        assessment = gate.evaluate("CC(=O)Oc1ccccc1C(=O)O")
        self.assertIsNotNone(assessment.audit_id)
        self.assertGreater(len(assessment.compliance_checks), 0)

    def test_quality_tier_classification_no_risk(self):
        """Test TIER_1 classification for clean molecules."""
        gate = StrictComplianceGate(compliance_level=ComplianceLevel.RELAXED)
        assessment = gate.evaluate("CC(=O)O")  # Acetic acid
        # Should be Tier 1 or 2 (no risk factors)
        self.assertIn(
            assessment.quality_tier,
            [QualityTier.TIER_1, QualityTier.TIER_2],
        )

    def test_risk_factor_identification(self):
        """Test risk factors are identified."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        # Aspirin has aromatic ring
        self.assertGreater(len(assessment.risk_factors), 0)

    def test_data_integrity_verification(self):
        """Test data integrity report is generated."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate("CC(=O)O")
        self.assertIsNotNone(assessment.data_integrity.checksum)
        self.assertTrue(assessment.data_integrity.integrity_verified)
        self.assertFalse(assessment.data_integrity.tampering_detected)

    def test_audit_id_uniqueness(self):
        """Test different SMILES generate different audit IDs."""
        gate = StrictComplianceGate()
        assessment1 = gate.evaluate("CC(=O)O")
        assessment2 = gate.evaluate("CC(=O)N")
        self.assertNotEqual(assessment1.audit_id, assessment2.audit_id)

    def test_compliance_check_result_formatting(self):
        """Test compliance checks are properly formatted."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate("CC(=O)O")
        for check in assessment.compliance_checks:
            self.assertIsNotNone(check.check_name)
            self.assertIsNotNone(check.message)
            self.assertTrue(isinstance(check.passed, bool))
            self.assertIn(check.severity, ["critical", "major", "minor", "info"])

    def test_recommendation_text_generated(self):
        """Test recommendation text is generated."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate("CC(=O)O")
        self.assertIsNotNone(assessment.recommendation)
        self.assertGreater(len(assessment.recommendation), 0)

    def test_hardened_compliance_stricter_than_relaxed(self):
        """Test hardened compliance is stricter than relaxed."""
        relaxed = StrictComplianceGate(compliance_level=ComplianceLevel.RELAXED)
        hardened = StrictComplianceGate(
            compliance_level=ComplianceLevel.HARDENED,
            strict_herg_threshold=0.1,
        )
        
        # Hardened should have lower thresholds
        self.assertLess(hardened.strict_herg_threshold, relaxed.strict_herg_threshold)

    def test_toxicity_prob_integration(self):
        """Test toxicity probabilities are evaluated."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate(
            "CC(=O)O",
            toxicity_probs={
                "herg": 0.1,
                "ames": 0.05,
                "hepatotox": 0.2,
            },
        )
        # Should have toxicity-related checks
        toxicity_checks = [
            c for c in assessment.compliance_checks
            if any(x in c.check_name for x in ["hERG", "Ames", "Hepatotoxicity"])
        ]
        self.assertGreater(len(toxicity_checks), 0)

    def test_as_dict_serialization(self):
        """Test assessment can be serialized to dict."""
        gate = StrictComplianceGate()
        assessment = gate.evaluate("CC(=O)O")
        d = assessment.as_dict()
        self.assertIn("smiles", d)
        self.assertIn("quality_tier", d)
        self.assertIn("overall_passed", d)
        self.assertIn("compliance_checks", d)


class TestBatchEvaluation(unittest.TestCase):
    """Test batch evaluation functionality."""

    def test_batch_evaluation(self):
        """Test batch evaluation of multiple SMILES."""
        smiles_list = [
            "CC(=O)O",  # Acetic acid
            "CC(=O)N",  # Acetamide
            "CCO",  # Ethanol
        ]
        result = evaluate_batch_with_strict_compliance(
            smiles_list,
            compliance_level=ComplianceLevel.RELAXED,
        )
        
        self.assertEqual(result["total_evaluated"], 3)
        self.assertEqual(result["passed"] + result["rejected"], 3)
        self.assertGreaterEqual(result["pass_rate"], 0)
        self.assertLessEqual(result["pass_rate"], 1)

    def test_batch_includes_all_assessments(self):
        """Test batch result includes all individual assessments."""
        smiles_list = ["CC(=O)O", "CC(=O)N"]
        result = evaluate_batch_with_strict_compliance(smiles_list)
        
        self.assertEqual(len(result["assessments"]), 2)
        for smiles in smiles_list:
            self.assertIn(smiles, result["assessments"])

    def test_batch_critical_issues_tracked(self):
        """Test batch evaluation tracks critical issues."""
        smiles_list = ["CC(=O)O", "invalid"]  # One valid, one invalid
        result = evaluate_batch_with_strict_compliance(smiles_list)
        
        # Should have at least one critical issue (invalid SMILES)
        self.assertGreater(len(result["critical_issues"]), 0)


class TestComplianceLevelConfiguration(unittest.TestCase):
    """Test compliance level configuration."""

    def test_relaxed_vs_strict_herg_thresholds(self):
        """Test compliance level affects hERG thresholds."""
        relaxed = StrictComplianceGate(compliance_level=ComplianceLevel.RELAXED)
        strict = StrictComplianceGate(compliance_level=ComplianceLevel.STRICT)
        
        # Strict should have lower (more restrictive) hERG threshold
        self.assertLess(
            strict.strict_herg_threshold,
            relaxed.strict_herg_threshold,
        )

    def test_hardened_is_most_restrictive(self):
        """Test hardened compliance level is most restrictive."""
        relaxed = StrictComplianceGate(compliance_level=ComplianceLevel.RELAXED)
        hardened = StrictComplianceGate(compliance_level=ComplianceLevel.HARDENED)
        
        # Hardened should have strictest threshold
        self.assertLess(
            hardened.strict_herg_threshold,
            relaxed.strict_herg_threshold,
        )
        self.assertLess(
            hardened.strict_ames_threshold,
            relaxed.strict_ames_threshold,
        )


class TestPropertyCalculation(unittest.TestCase):
    """Test molecular property calculations."""

    def test_simple_property_heuristic(self):
        """Test property calculation heuristic."""
        gate = StrictComplianceGate()
        # Simple SMILES: acetic acid
        props = gate._calculate_properties("CC(=O)O")
        
        self.assertIn("mw", props)
        self.assertIn("logp", props)
        self.assertIn("tpsa", props)
        self.assertGreater(props["mw"], 0)


def run_test_log_output() -> None:
    """Run tests with logging."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
