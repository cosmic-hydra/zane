"""Tests for the Physics Oracle, Bayesian surrogate, and GFlowNet reward rewiring.

Covers:
- ``drug_discovery.polyglot_integration`` (PhysicsOracle, FEPResult)
- ``drug_discovery.training.closed_loop`` (SurrogateModel, ClosedLoopLearner)
- ``drug_discovery.models.gflownet`` (PhysicsRewardFunction, GFlowNetTrainer)

Imports use direct module paths to avoid triggering the package __init__
which may pull in optional dependencies (flwr, etc.).
"""

from __future__ import annotations

import asyncio
import importlib
import math
import sys

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: import directly from a module file, bypassing package __init__
# ---------------------------------------------------------------------------
def _import_module(dotted: str):
    """Import a module by its dotted path, skipping package __init__ triggers."""
    return importlib.import_module(dotted)


# We import the modules directly to avoid drug_discovery.training.__init__
# pulling in flwr and other optional deps.
_polyglot = _import_module("drug_discovery.polyglot_integration")
_gflownet = _import_module("drug_discovery.models.gflownet")

# For closed_loop we must avoid the training package __init__
# so we import the module file directly.
_closed_loop_spec = importlib.util.spec_from_file_location(
    "drug_discovery.training.closed_loop",
    "drug_discovery/training/closed_loop.py",
)
_closed_loop = importlib.util.module_from_spec(_closed_loop_spec)
sys.modules["drug_discovery.training.closed_loop"] = _closed_loop
_closed_loop_spec.loader.exec_module(_closed_loop)

FEPResult = _polyglot.FEPResult
_run_single_fep = _polyglot._run_single_fep
PhysicsOracle = _polyglot.PhysicsOracle

SurrogateModel = _closed_loop.SurrogateModel
smiles_to_fingerprint = _closed_loop.smiles_to_fingerprint
ClosedLoopLearner = _closed_loop.ClosedLoopLearner
_normal_pdf = _closed_loop._normal_pdf
_normal_cdf = _closed_loop._normal_cdf

PhysicsRewardFunction = _gflownet.PhysicsRewardFunction
PhysicsRewardConfig = _gflownet.PhysicsRewardConfig
GFlowNetConfig = _gflownet.GFlowNetConfig
GFlowNetTrainer = _gflownet.GFlowNetTrainer
GFlowNetPolicy = _gflownet.GFlowNetPolicy


# Check if RDKit is available (the fallback FEP path needs it)
try:
    from rdkit import Chem  # type: ignore[import-untyped]

    _RDKIT = True
except Exception:
    _RDKIT = False


# =========================================================================
# polyglot_integration -- PhysicsOracle & FEPResult
# =========================================================================
class TestFEPResult:
    def test_as_dict_round_trip(self):
        r = FEPResult(smiles="CCO", delta_g=-8.3, success=True, converged=True)
        d = r.as_dict()
        assert d["smiles"] == "CCO"
        assert d["delta_g"] == pytest.approx(-8.3)
        assert d["success"] is True

    def test_default_fields(self):
        r = FEPResult(smiles="C")
        assert r.delta_g is None
        assert r.success is False
        assert r.error is None


class TestRunSingleFep:
    def test_empty_smiles(self):
        r = _run_single_fep("", "target.pdb")
        assert not r.success
        assert "Empty" in (r.error or "")

    def test_no_protein(self):
        r = _run_single_fep("CCO", "")
        assert not r.success
        assert "protein" in (r.error or "").lower() or "No protein" in (r.error or "")

    @pytest.mark.skipif(not _RDKIT, reason="RDKit not installed; fallback MD sim requires it")
    def test_fallback_produces_result(self):
        r = _run_single_fep("CCO", "target.pdb")
        assert r.success
        assert r.delta_g is not None


class TestPhysicsOracle:
    def test_score_batch_empty(self):
        oracle = PhysicsOracle()
        results = oracle.score_batch_sync([])
        assert results == []

    @pytest.mark.skipif(not _RDKIT, reason="RDKit not installed; fallback MD sim requires it")
    def test_score_batch_local_fallback(self):
        oracle = PhysicsOracle(protein_pdb_path="target.pdb")
        results = oracle.score_batch_sync(["CCO", "c1ccccc1"])
        assert len(results) == 2
        for r in results:
            assert r.success
            assert r.delta_g is not None

    @pytest.mark.skipif(not _RDKIT, reason="RDKit not installed; fallback MD sim requires it")
    def test_score_batch_async(self):
        oracle = PhysicsOracle(protein_pdb_path="target.pdb")
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(oracle.score_batch(["CCO"]))
        finally:
            loop.close()
        assert len(results) == 1
        assert results[0].success

    def test_dict_to_result(self):
        d = {"smiles": "CCO", "delta_g": -7.0, "success": True, "converged": True}
        r = PhysicsOracle._dict_to_result(d)
        assert r.smiles == "CCO"
        assert r.delta_g == -7.0


# =========================================================================
# training/closed_loop -- SurrogateModel
# =========================================================================
class TestSurrogateModel:
    def test_observe_and_count(self):
        sm = SurrogateModel(fp_dim=64)
        assert sm.n_observations == 0
        sm.observe(np.random.randn(64), -8.0)
        assert sm.n_observations == 1

    def test_fit_requires_two_observations(self):
        sm = SurrogateModel(fp_dim=64)
        sm.observe(np.random.randn(64), -8.0)
        sm.fit()  # should warn but not raise

    def test_fit_and_predict(self):
        sm = SurrogateModel(fp_dim=32)
        rng = np.random.default_rng(42)
        for _ in range(10):
            sm.observe(rng.standard_normal(32), rng.uniform(-10, 0))
        sm.fit()

        test_fps = rng.standard_normal((5, 32))
        means, stds = sm.predict(test_fps)
        assert means.shape == (5,)
        assert stds.shape == (5,)

    def test_expected_improvement(self):
        sm = SurrogateModel(fp_dim=16)
        rng = np.random.default_rng(0)
        for _ in range(5):
            sm.observe(rng.standard_normal(16), rng.uniform(-10, 0))
        sm.fit()

        fps = rng.standard_normal((20, 16))
        ei = sm.expected_improvement(fps)
        assert ei.shape == (20,)
        assert np.all(ei >= 0)

    def test_select_top_candidates(self):
        sm = SurrogateModel(fp_dim=16)
        rng = np.random.default_rng(1)
        for _ in range(10):
            sm.observe(rng.standard_normal(16), rng.uniform(-10, 0))
        sm.fit()

        pool = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCC"]
        selected = sm.select_top_candidates(pool, top_fraction=0.5, min_candidates=1)
        assert 1 <= len(selected) <= len(pool)
        for s in selected:
            assert s in pool


class TestSmilesToFingerprint:
    def test_deterministic(self):
        fp1 = smiles_to_fingerprint("CCO", nbits=128)
        fp2 = smiles_to_fingerprint("CCO", nbits=128)
        np.testing.assert_array_equal(fp1, fp2)

    def test_different_smiles_differ(self):
        fp1 = smiles_to_fingerprint("CCO", nbits=128)
        fp2 = smiles_to_fingerprint("c1ccccc1", nbits=128)
        assert not np.array_equal(fp1, fp2)

    def test_shape(self):
        fp = smiles_to_fingerprint("CCO", nbits=256)
        assert fp.shape == (256,)


class TestClosedLoopLearner:
    def test_basic_loop(self):
        learner = ClosedLoopLearner()
        results = learner.run_closed_loop(
            target_protein="test_protein",
            num_iterations=2,
            candidates_per_iteration=10,
            top_k_for_training=3,
        )
        assert len(results) == 2
        assert results[0]["iteration"] == 1
        assert results[0]["num_generated"] == 10

    def test_surrogate_observations_grow(self):
        learner = ClosedLoopLearner()
        results = learner.run_closed_loop(
            target_protein="test",
            num_iterations=3,
            candidates_per_iteration=5,
        )
        # Surrogate should accumulate observations
        assert results[-1]["surrogate_observations"] >= results[0]["surrogate_observations"]


# =========================================================================
# models/gflownet -- PhysicsRewardFunction & GFlowNetTrainer
# =========================================================================
class TestPhysicsRewardFunction:
    def test_default_reward_positive(self):
        fn = PhysicsRewardFunction()
        traj = {"atoms": torch.tensor([0, 1, 2]), "num_atoms": 3}
        r = fn(traj)
        assert r > 0

    def test_reward_with_mock_oracle(self):
        class MockOracle:
            def score_batch_sync(self, smiles_list):
                return [FEPResult(smiles=s, delta_g=-9.0, success=True) for s in smiles_list]

        fn = PhysicsRewardFunction(oracle=MockOracle())
        traj = {"atoms": torch.tensor([0, 1, 2]), "num_atoms": 3}
        r = fn(traj)
        assert r > 0
        # Strong binding should produce high reward
        assert r > 1.0  # exp(-1 * -9) = exp(9) >> 1

    def test_toxicity_penalty(self):
        class MockAdmet:
            def predict(self, smiles):
                return {"toxicity": 0.9}

        cfg = PhysicsRewardConfig(toxicity_threshold=0.5)
        fn_toxic = PhysicsRewardFunction(admet_predictor=MockAdmet(), config=cfg)
        fn_clean = PhysicsRewardFunction()

        traj = {"atoms": torch.tensor([0, 1]), "num_atoms": 2}
        r_toxic = fn_toxic(traj)
        r_clean = fn_clean(traj)
        # Toxic molecule should get a lower reward
        assert r_toxic < r_clean

    def test_combine_negative_delta_g_rewarded(self):
        fn = PhysicsRewardFunction()
        r_strong = fn._combine(delta_g=-10.0, toxicity=0.0)
        r_weak = fn._combine(delta_g=-2.0, toxicity=0.0)
        assert r_strong > r_weak

    def test_min_reward_floor(self):
        cfg = PhysicsRewardConfig(min_reward=0.001)
        fn = PhysicsRewardFunction(config=cfg)
        assert fn({"atoms": torch.tensor([0]), "num_atoms": 1}) >= cfg.min_reward

    def test_default_atoms_to_smiles(self):
        s = PhysicsRewardFunction._default_atoms_to_smiles(
            {"atoms": torch.tensor([0, 1, 2]), "num_atoms": 3}
        )
        assert isinstance(s, str)
        assert len(s) > 0

    def test_atoms_to_smiles_empty(self):
        s = PhysicsRewardFunction._default_atoms_to_smiles(
            {"atoms": torch.tensor([], dtype=torch.long), "num_atoms": 0}
        )
        assert s == "C"


class TestGFlowNetTrainer:
    def test_train_step_default(self):
        cfg = GFlowNetConfig(hidden_dim=32, num_layers=1, max_atoms=5)
        trainer = GFlowNetTrainer(cfg)
        loss = trainer.train_step()
        assert isinstance(loss, float)
        assert not math.isnan(loss)

    def test_train_step_with_physics_reward(self):
        cfg = GFlowNetConfig(hidden_dim=32, num_layers=1, max_atoms=5)
        reward_fn = PhysicsRewardFunction()
        trainer = GFlowNetTrainer(cfg, reward_fn=reward_fn)
        loss = trainer.train_step()
        assert isinstance(loss, float)
        assert not math.isnan(loss)

    def test_sample(self):
        cfg = GFlowNetConfig(hidden_dim=32, num_layers=1, max_atoms=5)
        trainer = GFlowNetTrainer(cfg)
        samples = trainer.sample(n=3)
        assert len(samples) == 3
        for s in samples:
            assert "atoms" in s
            assert "num_atoms" in s


class TestGFlowNetPolicy:
    def test_forward_empty_graph(self):
        cfg = GFlowNetConfig(hidden_dim=32, num_layers=2)
        policy = GFlowNetPolicy(cfg)
        state = {
            "atoms": torch.tensor([0, 1, 2]),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
        }
        out = policy(state)
        assert "atom_logits" in out
        assert "stop_logit" in out
        assert "graph_embedding" in out


# =========================================================================
# Normal distribution helpers
# =========================================================================
class TestNormalHelpers:
    def test_normal_pdf_at_zero(self):
        val = _normal_pdf(np.array([0.0]))
        expected = 1.0 / math.sqrt(2 * math.pi)
        assert val[0] == pytest.approx(expected, rel=1e-5)

    def test_normal_cdf_symmetry(self):
        assert _normal_cdf(np.array([0.0]))[0] == pytest.approx(0.5, abs=1e-5)
        assert _normal_cdf(np.array([3.0]))[0] > 0.99
        assert _normal_cdf(np.array([-3.0]))[0] < 0.01


# =========================================================================
# Improvement tests: caching, warm-start, diverse candidates
# =========================================================================
class TestPhysicsOracleCache:
    def test_cache_enabled_by_default(self):
        oracle = PhysicsOracle()
        stats = oracle.cache_stats
        assert stats["enabled"] is True
        assert stats["size"] == 0

    def test_cache_disabled(self):
        oracle = PhysicsOracle(enable_cache=False)
        assert oracle.cache_stats["enabled"] is False

    def test_clear_cache(self):
        oracle = PhysicsOracle()
        oracle._cache["test"] = FEPResult(smiles="test")
        assert oracle.cache_stats["size"] == 1
        oracle.clear_cache()
        assert oracle.cache_stats["size"] == 0


class TestSurrogateModelSerialization:
    def test_save_and_load(self, tmp_path):
        sm = SurrogateModel(fp_dim=32)
        rng = np.random.default_rng(42)
        for _ in range(5):
            sm.observe(rng.standard_normal(32), rng.uniform(-10, 0))
        sm.fit()

        path = str(tmp_path / "surrogate.npz")
        sm.save(path)

        loaded = SurrogateModel.load(path)
        assert loaded.n_observations == sm.n_observations
        assert loaded.fp_dim == sm.fp_dim

        # Predictions should match
        test_fp = rng.standard_normal((3, 32))
        m1, s1 = sm.predict(test_fp)
        m2, s2 = loaded.predict(test_fp)
        np.testing.assert_allclose(m1, m2, atol=1e-6)
        np.testing.assert_allclose(s1, s2, atol=1e-6)

    def test_save_empty_model(self, tmp_path):
        sm = SurrogateModel(fp_dim=16)
        path = str(tmp_path / "empty.npz")
        sm.save(path)
        loaded = SurrogateModel.load(path)
        assert loaded.n_observations == 0

    def test_fit_count_persisted(self, tmp_path):
        sm = SurrogateModel(fp_dim=16)
        rng = np.random.default_rng(0)
        for _ in range(3):
            sm.observe(rng.standard_normal(16), rng.uniform(-5, 0))
        sm.fit()
        sm.fit()
        assert sm._fit_count == 2

        path = str(tmp_path / "fitted.npz")
        sm.save(path)
        loaded = SurrogateModel.load(path)
        # load auto-fits, so fit_count should be original + 1
        assert loaded._fit_count >= 2


class TestDiverseCandidateGeneration:
    def test_candidates_are_diverse(self):
        learner = ClosedLoopLearner()
        candidates = learner._generate_candidates("protein_x", 20)
        smiles_set = {c["smiles"] for c in candidates}
        # With 20 candidates and 16 seed SMILES, we should have > 1 unique
        assert len(smiles_set) > 1

    def test_candidates_cycle_through_pool(self):
        learner = ClosedLoopLearner()
        candidates = learner._generate_candidates("protein_x", 32)
        assert len(candidates) == 32
        # First and 17th should be the same (16-element pool)
        assert candidates[0]["smiles"] == candidates[16]["smiles"]
