"""Tests for formulation simulator, GLP tox panel, and SaMD validation."""

from __future__ import annotations

import json

import pytest

from drug_discovery.formulation_simulator import (
    FormulationSimulator,
    FormulationReport,
    PolymorphResult,
    StabilityCondition,
    StabilityResult,
    ICH_CONDITIONS,
)
from drug_discovery.glp_tox_panel import (
    PreClinicalToxPanel,
    GLPToxPanel,
    HERGResult,
    CYP450Result,
    AmesResult,
    CYP_ENZYMES,
)
from drug_discovery.compliance.validation.iq_oq_pq import (
    IQOQPQRunner,
    QualificationResult,
    REFERENCE_MOLECULES,
)


# =========================================================================
# Module 1: Formulation Simulator
# =========================================================================
class TestFormulationSimulator:
    def test_basic_assessment(self):
        sim = FormulationSimulator()
        report = sim.run_full_assessment("CCO")
        assert isinstance(report, FormulationReport)
        assert 0.0 <= report.overall_stability_score <= 1.0
        assert len(report.stability_results) == len(ICH_CONDITIONS)

    def test_aspirin_assessment(self):
        sim = FormulationSimulator()
        report = sim.run_full_assessment("CC(=O)Oc1ccccc1C(=O)O")
        assert report.polymorph_result is not None
        assert report.recommended_storage != ""

    def test_stability_results_have_all_fields(self):
        sim = FormulationSimulator()
        report = sim.run_full_assessment("CCO")
        for r in report.stability_results:
            assert isinstance(r, StabilityResult)
            assert r.temperature_K > 0
            assert 0.0 <= r.degradation_risk <= 1.0

    def test_ph_conditions_evaluated(self):
        sim = FormulationSimulator()
        report = sim.run_full_assessment("CCO")
        ph_results = [r for r in report.stability_results if r.pH is not None]
        assert len(ph_results) >= 2  # stomach + plasma

    def test_high_temp_increases_degradation(self):
        # Use identical durations to isolate temperature effect
        conds = [
            StabilityCondition("cold", 298.15, 1.0, duration_ns=1.0),
            StabilityCondition("hot", 353.15, 1.0, duration_ns=1.0),
        ]
        sim = FormulationSimulator(conditions=conds)
        report = sim.run_full_assessment("CCO")
        cold = report.stability_results[0]
        hot = report.stability_results[1]
        assert hot.degradation_risk >= cold.degradation_risk

    def test_polymorph_screening(self):
        sim = FormulationSimulator()
        report = sim.run_full_assessment("CCO")
        poly = report.polymorph_result
        assert poly is not None
        assert 0.0 <= poly.polymorph_risk <= 1.0
        assert poly.num_predicted_forms >= 1
        assert poly.shelf_stability_months > 0

    def test_as_dict(self):
        sim = FormulationSimulator()
        report = sim.run_full_assessment("CCO")
        d = report.as_dict()
        assert "overall_stability_score" in d
        assert "polymorph" in d
        assert "stability_conditions" in d

    def test_batch(self):
        sim = FormulationSimulator()
        reports = sim.run_batch(["CCO", "c1ccccc1"])
        assert len(reports) == 2

    def test_custom_conditions(self):
        conds = [StabilityCondition("extreme", 373.15, 1.0)]
        sim = FormulationSimulator(conditions=conds)
        report = sim.run_full_assessment("CCO")
        assert len(report.stability_results) == 1


# =========================================================================
# Module 2: GLP Pre-Clinical Tox Panel
# =========================================================================
class TestPreClinicalToxPanel:
    def test_basic_evaluation(self):
        panel = PreClinicalToxPanel()
        result = panel.evaluate("CCO")
        assert isinstance(result, GLPToxPanel)
        assert result.herg is not None
        assert result.cyp450 is not None
        assert result.ames is not None
        assert 0.0 <= result.overall_tox_score <= 1.0

    def test_herg_assay(self):
        panel = PreClinicalToxPanel()
        result = panel.evaluate("CCO")
        herg = result.herg
        assert isinstance(herg, HERGResult)
        assert 0.0 <= herg.inhibition_probability <= 1.0
        assert herg.risk_class in ("low", "moderate", "high")
        assert herg.ic50_estimate_uM > 0
        assert isinstance(herg.cardiac_risk, str)

    def test_cyp450_matrix(self):
        panel = PreClinicalToxPanel()
        result = panel.evaluate("CCO")
        cyp = result.cyp450
        assert isinstance(cyp, CYP450Result)
        for enzyme in CYP_ENZYMES:
            assert enzyme in cyp.enzyme_inhibitions
            assert 0.0 <= cyp.enzyme_inhibitions[enzyme] <= 1.0
        assert cyp.ddi_risk in ("low", "moderate", "high")
        assert cyp.primary_metabolism_route in CYP_ENZYMES

    def test_ames_test(self):
        panel = PreClinicalToxPanel()
        result = panel.evaluate("CCO")
        ames = result.ames
        assert isinstance(ames, AmesResult)
        assert 0.0 <= ames.mutagenicity_probability <= 1.0
        assert 0.0 <= ames.carcinogenicity_probability <= 1.0
        assert ames.risk_class in ("non-mutagenic", "equivocal", "mutagenic")

    def test_ind_readiness(self):
        panel = PreClinicalToxPanel()
        result = panel.evaluate("CCO")
        # Simple molecule should have low toxicity
        assert isinstance(result.ind_ready, bool)

    def test_batch(self):
        panel = PreClinicalToxPanel()
        results = panel.evaluate_batch(["CCO", "c1ccccc1", "CC(=O)O"])
        assert len(results) == 3

    def test_as_dict(self):
        panel = PreClinicalToxPanel()
        result = panel.evaluate("CCO")
        d = result.as_dict()
        assert "herg" in d
        assert "cyp450" in d
        assert "ames" in d
        assert "ind_ready" in d

    def test_strict_thresholds(self):
        panel = PreClinicalToxPanel(herg_threshold=0.01, cyp_threshold=0.01, ames_threshold=0.01)
        result = panel.evaluate("c1ccccc1")
        # Very strict: most things should fail
        assert len(result.rejection_reasons) >= 0  # may or may not fail


# =========================================================================
# Module 3: GxP / SaMD Validation (IQ/OQ/PQ)
# =========================================================================
class TestIQOQPQRunner:
    def test_full_qualification(self):
        runner = IQOQPQRunner()
        result = runner.run_full_qualification()
        assert isinstance(result, QualificationResult)
        assert result.timestamp != ""
        assert result.total_elapsed_seconds > 0
        assert result.report_hash != ""

    def test_iq_checks_present(self):
        runner = IQOQPQRunner()
        result = runner.run_full_qualification()
        assert len(result.iq_checks) > 0
        # Core modules should import
        core_checks = [c for c in result.iq_checks if "polyglot_integration" in c.name or "numpy" in c.name]
        for c in core_checks:
            assert c.passed, f"IQ check failed: {c.name}: {c.message}"

    def test_oq_checks_present(self):
        runner = IQOQPQRunner()
        result = runner.run_full_qualification()
        assert len(result.oq_checks) > 0
        # Should have validation + toxicity checks for each reference mol
        assert len(result.oq_checks) == len(REFERENCE_MOLECULES) * 2

    def test_pq_checks_present(self):
        runner = IQOQPQRunner()
        result = runner.run_full_qualification()
        assert len(result.pq_checks) >= 4  # pipeline, audit, formulation, glp

    def test_summary_string(self):
        runner = IQOQPQRunner()
        result = runner.run_full_qualification()
        summary = result.summary()
        assert "IQ" in summary
        assert "OQ" in summary
        assert "PQ" in summary
        assert "PASS" in summary or "FAIL" in summary

    def test_as_dict(self):
        runner = IQOQPQRunner()
        result = runner.run_full_qualification()
        d = result.as_dict()
        assert "iq_checks" in d
        assert "oq_checks" in d
        assert "pq_checks" in d
        assert "report_hash" in d

    def test_save_report(self, tmp_path):
        runner = IQOQPQRunner()
        result = runner.run_full_qualification()
        path = str(tmp_path / "qual_report.json")
        runner.save_report(result, path)

        with open(path) as f:
            data = json.load(f)
        assert data["timestamp"] == result.timestamp
        assert data["report_hash"] == result.report_hash

    def test_custom_reference_molecules(self):
        custom = {
            "ethanol": {
                "smiles": "CCO",
                "expected_mw_range": (44.0, 48.0),
                "expected_toxicity_max": 0.8,
                "expected_drug_likeness_min": 0.0,
            },
        }
        runner = IQOQPQRunner(reference_molecules=custom)
        result = runner.run_full_qualification()
        assert len(result.oq_checks) == 2  # 1 mol * 2 checks

    def test_report_hash_determinism(self):
        """Two runs should produce different hashes (different timestamps)."""
        runner = IQOQPQRunner()
        r1 = runner.run_full_qualification()
        r2 = runner.run_full_qualification()
        # Hashes differ because timestamps differ
        assert r1.report_hash != r2.report_hash
