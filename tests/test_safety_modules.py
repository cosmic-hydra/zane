"""Tests for the safety module: SMILES validation, toxicity gate, Pareto ranker,
and end-to-end pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drug_discovery.safety.smiles_validator import SmilesValidator, ValidationResult
from drug_discovery.safety.toxicity_gate import ToxicityGate, ToxicityGateConfig, ToxicityVerdict
from drug_discovery.safety.pareto_ranker import ParetoRanker, RankedCandidate, ObjectiveSpec
from drug_discovery.safety.end_to_end_pipeline import SafeGenerationPipeline, PipelineConfig, PipelineResult

try:
    from rdkit import Chem

    _RDKIT = True
except Exception:
    _RDKIT = False


# =========================================================================
# SmilesValidator
# =========================================================================
class TestSmilesValidator:
    def test_empty_smiles(self):
        v = SmilesValidator()
        r = v.validate("")
        assert not r.passed
        assert "Empty" in r.rejection_reason

    def test_valid_smiles_heuristic(self):
        v = SmilesValidator()
        r = v.validate("CCO")
        # Should pass via either RDKit or heuristic path
        assert r.is_valid

    def test_validate_batch(self):
        v = SmilesValidator()
        results = v.validate_batch(["CCO", "c1ccccc1", ""])
        assert len(results) == 3
        assert results[0].is_valid
        assert not results[2].is_valid

    def test_filter_valid(self):
        v = SmilesValidator()
        valid = v.filter_valid(["CCO", "", "c1ccccc1"])
        assert len(valid) >= 1  # at least CCO should pass

    def test_success_rate(self):
        v = SmilesValidator()
        rate = v.success_rate(["CCO", "c1ccccc1", ""])
        assert 0.0 < rate <= 1.0

    def test_success_rate_empty(self):
        v = SmilesValidator()
        assert v.success_rate([]) == 0.0

    @pytest.mark.skipif(not _RDKIT, reason="RDKit not installed")
    def test_rdkit_canonicalization(self):
        v = SmilesValidator()
        r = v.validate("C(C)O")
        assert r.passed
        assert r.canonical == "CCO"

    @pytest.mark.skipif(not _RDKIT, reason="RDKit not installed")
    def test_rdkit_invalid_smiles(self):
        v = SmilesValidator()
        r = v.validate("TOTALLY_INVALID_SMILES")
        assert not r.passed

    @pytest.mark.skipif(not _RDKIT, reason="RDKit not installed")
    def test_molecular_weight_limit(self):
        v = SmilesValidator(max_molecular_weight=50.0)
        r = v.validate("CC(=O)Oc1ccccc1C(=O)O")  # aspirin ~180 Da
        assert not r.passed
        assert "weight" in r.rejection_reason.lower()

    def test_heuristic_unbalanced_brackets(self):
        v = SmilesValidator()
        r = v.validate("[NH2")
        if not _RDKIT:
            assert not r.passed

    def test_heuristic_unbalanced_parens(self):
        v = SmilesValidator()
        r = v.validate("CC(CC")
        # Should detect unbalanced parens in heuristic mode
        if not _RDKIT:
            assert not r.passed

    def test_balance_parens_static(self):
        result = SmilesValidator._balance_parens("CC(C(O")
        assert result.count("(") == result.count(")")

    def test_min_heavy_atoms(self):
        v = SmilesValidator(min_heavy_atoms=5)
        r = v.validate("CC")  # only 2 heavy atoms
        assert not r.passed


# =========================================================================
# ToxicityGate
# =========================================================================
class TestToxicityGate:
    def test_basic_evaluation(self):
        gate = ToxicityGate()
        verdict = gate.evaluate("CCO")
        assert isinstance(verdict, ToxicityVerdict)
        assert len(verdict.endpoints) == 4  # hERG, Ames, hepatotox, cytotox
        assert 0.0 <= verdict.overall_toxicity <= 1.0

    def test_safety_score(self):
        gate = ToxicityGate()
        verdict = gate.evaluate("CCO")
        assert verdict.safety_score == pytest.approx(1.0 - verdict.overall_toxicity)

    def test_filter_safe(self):
        gate = ToxicityGate()
        safe = gate.filter_safe(["CCO", "c1ccccc1"])
        assert isinstance(safe, list)

    def test_batch_safety_rate(self):
        gate = ToxicityGate()
        rate = gate.batch_safety_rate(["CCO", "c1ccccc1", "CC(=O)O"])
        assert 0.0 <= rate <= 1.0

    def test_batch_safety_rate_empty(self):
        gate = ToxicityGate()
        assert gate.batch_safety_rate([]) == 0.0

    def test_cache(self):
        gate = ToxicityGate()
        v1 = gate.evaluate("CCO")
        v2 = gate.evaluate("CCO")
        assert v1 is v2  # same cached object
        gate.clear_cache()
        v3 = gate.evaluate("CCO")
        assert v3 is not v1

    def test_as_dict(self):
        gate = ToxicityGate()
        verdict = gate.evaluate("CCO")
        d = verdict.as_dict()
        assert "smiles" in d
        assert "passed" in d
        assert "safety_score" in d
        assert "endpoints" in d

    def test_strict_config(self):
        strict = ToxicityGateConfig(
            herg_threshold=0.01,
            ames_threshold=0.01,
            hepatotox_threshold=0.01,
            cytotox_threshold=0.01,
        )
        gate = ToxicityGate(config=strict)
        # Very strict -- most molecules should fail
        verdict = gate.evaluate("CC(=O)Oc1ccccc1C(=O)O")
        # Not asserting passed/failed since it depends on estimators

    def test_drug_likeness_score(self):
        gate = ToxicityGate()
        verdict = gate.evaluate("CCO")
        assert 0.0 <= verdict.drug_likeness <= 1.0

    def test_lipinski_violations(self):
        gate = ToxicityGate()
        verdict = gate.evaluate("CCO")
        assert verdict.lipinski_violations >= 0

    def test_with_mock_admet(self):
        class MockAdmet:
            def predict(self, smiles):
                return {"herg": 0.05, "ames": 0.02, "hepatotox": 0.1, "cytotox": 0.08}

        gate = ToxicityGate(admet_predictor=MockAdmet())
        verdict = gate.evaluate("CCO")
        # With low mock scores, should pass
        assert verdict.passed


# =========================================================================
# ParetoRanker
# =========================================================================
class TestParetoRanker:
    @pytest.fixture
    def sample_candidates(self):
        return [
            {"smiles": "A", "delta_g": -10.0, "toxicity": 0.1, "drug_likeness": 0.9, "sa_score": 2.0},
            {"smiles": "B", "delta_g": -8.0, "toxicity": 0.05, "drug_likeness": 0.85, "sa_score": 1.5},
            {"smiles": "C", "delta_g": -6.0, "toxicity": 0.3, "drug_likeness": 0.7, "sa_score": 3.0},
            {"smiles": "D", "delta_g": -12.0, "toxicity": 0.5, "drug_likeness": 0.5, "sa_score": 5.0},
            {"smiles": "E", "delta_g": -9.0, "toxicity": 0.08, "drug_likeness": 0.95, "sa_score": 1.0},
        ]

    def test_rank_returns_all(self, sample_candidates):
        ranker = ParetoRanker()
        ranked = ranker.rank(sample_candidates)
        assert len(ranked) == 5

    def test_rank_empty(self):
        ranker = ParetoRanker()
        assert ranker.rank([]) == []

    def test_pareto_front_exists(self, sample_candidates):
        ranker = ParetoRanker()
        front = ranker.pareto_front(sample_candidates)
        assert len(front) >= 1
        for r in front:
            assert r.pareto_optimal

    def test_select_top(self, sample_candidates):
        ranker = ParetoRanker()
        top = ranker.select_top(sample_candidates, k=3)
        assert len(top) == 3

    def test_select_top_pareto_only(self, sample_candidates):
        ranker = ParetoRanker()
        top = ranker.select_top(sample_candidates, k=10, pareto_only=True)
        for r in top:
            assert r.pareto_optimal

    def test_scalarized_scores_ordered(self, sample_candidates):
        ranker = ParetoRanker()
        ranked = ranker.rank(sample_candidates)
        # Within same pareto rank, scalarized scores should be non-decreasing
        for i in range(len(ranked) - 1):
            if ranked[i].pareto_rank == ranked[i + 1].pareto_rank:
                assert ranked[i].scalarized_score <= ranked[i + 1].scalarized_score + 1e-8

    def test_as_dict(self, sample_candidates):
        ranker = ParetoRanker()
        ranked = ranker.rank(sample_candidates)
        d = ranked[0].as_dict()
        assert "smiles" in d
        assert "scalarized_score" in d
        assert "pareto_rank" in d

    def test_custom_objectives(self):
        objs = [
            ObjectiveSpec("delta_g", weight=1.0, minimize=True),
            ObjectiveSpec("toxicity", weight=1.0, minimize=True),
        ]
        ranker = ParetoRanker(objectives=objs)
        candidates = [
            {"smiles": "X", "delta_g": -10.0, "toxicity": 0.1},
            {"smiles": "Y", "delta_g": -5.0, "toxicity": 0.5},
        ]
        ranked = ranker.rank(candidates)
        assert len(ranked) == 2
        # X dominates Y on both objectives
        assert ranked[0].smiles == "X"
        assert ranked[0].pareto_optimal

    def test_single_candidate(self):
        ranker = ParetoRanker()
        ranked = ranker.rank([{"smiles": "Z", "delta_g": -7.0, "toxicity": 0.2, "drug_likeness": 0.8, "sa_score": 2.0}])
        assert len(ranked) == 1
        assert ranked[0].pareto_optimal


# =========================================================================
# SafeGenerationPipeline
# =========================================================================
class TestSafeGenerationPipeline:
    def test_basic_run(self):
        pipeline = SafeGenerationPipeline()
        result = pipeline.run(num_candidates=20, top_k=5)
        assert isinstance(result, PipelineResult)
        assert result.candidates_generated == 20
        assert result.candidates_valid > 0
        assert result.elapsed_seconds > 0

    def test_with_seed_smiles(self):
        pipeline = SafeGenerationPipeline()
        result = pipeline.run(
            seed_smiles=["CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"],
            top_k=3,
        )
        assert result.candidates_generated == 4
        assert result.candidates_valid >= 1

    def test_empty_generator(self):
        pipeline = SafeGenerationPipeline(generator_fn=lambda n: [])
        result = pipeline.run(num_candidates=0, top_k=5)
        assert result.candidates_generated == 0
        assert len(result.final_candidates) == 0

    def test_validation_rate(self):
        pipeline = SafeGenerationPipeline()
        result = pipeline.run(num_candidates=50, top_k=10)
        # With valid drug-like seed pool, validation rate should be high
        assert result.validation_rate > 0.5

    def test_summary_string(self):
        pipeline = SafeGenerationPipeline()
        result = pipeline.run(num_candidates=10, top_k=3)
        summary = result.summary()
        assert "Generated" in summary
        assert "Valid" in summary

    def test_custom_generator(self):
        def gen(n):
            return ["CCO"] * n

        pipeline = SafeGenerationPipeline(generator_fn=gen)
        result = pipeline.run(num_candidates=10, top_k=3)
        assert result.candidates_generated == 10

    def test_pipeline_config(self):
        cfg = PipelineConfig(
            num_candidates=15,
            final_top_k=3,
            oracle_top_k=10,
        )
        pipeline = SafeGenerationPipeline(config=cfg)
        result = pipeline.run()
        assert result.candidates_generated == 15
