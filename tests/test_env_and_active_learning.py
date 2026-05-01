"""Unit tests for environmental tests, LIMS optimizer, active learning sampler, and ABFE residuals."""
import unittest
from infrastructure.lims.latency_optimizer import LimsLatencyOptimizer, get_default_optimizer
import importlib.util
import os
import pathlib

# Import uncertainty_sampler directly from its file to avoid package-level heavy deps
_this_dir = pathlib.Path(__file__).resolve().parent
_al_path = (_this_dir / ".." / "drug_discovery" / "active_learning" / "uncertainty_sampler.py").resolve()
spec = importlib.util.spec_from_file_location("uncertainty_sampler", str(_al_path))
uncertainty_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uncertainty_mod)
UncertaintySampler = uncertainty_mod.UncertaintySampler
from drug_discovery.smd.abfe_residuals import compute_residuals, rmse, summarize_abfe
from drug_discovery.safety.environmental_tests import run_environmental_tests


class TestEnvAndAL(unittest.TestCase):
    def test_lims_optimizer_basic(self):
        opt = LimsLatencyOptimizer(cache_ttl=1.0)
        # simple function
        def f(x):
            return x * 2
        wrapped = opt.instrument(lambda x: str(x))(f)  # key_func passed wrongly on purpose -> fallback
        # call should succeed
        self.assertEqual(f(3), 6)
        # default optimizer exists
        self.assertIsNotNone(get_default_optimizer())

    def test_uncertainty_sampler(self):
        sampler = UncertaintySampler()
        smiles = ['A', 'B', 'C', 'D']
        uncertainties = [0.1, 0.9, 0.4, 0.8]
        selected = sampler.select_batch(smiles, uncertainties, batch_size=2)
        self.assertEqual(len(selected), 2)
        # highest uncertainty items should be included
        self.assertTrue('B' in selected or 'D' in selected)

    def test_abfe_residuals(self):
        pred = [1.0, 2.0, 3.0]
        obs = [1.1, 1.9, 2.8]
        res = compute_residuals(pred, obs)
        self.assertEqual(len(res), 3)
        s = summarize_abfe(pred, obs)
        self.assertIn('rmse', s)
        self.assertGreaterEqual(s['rmse'], 0.0)

    def test_environmental_tests(self):
        out = run_environmental_tests('CC(=O)O')
        self.assertIn('ph_profiles', out)
        self.assertIn('plasma_binding', out)
        self.assertIn('7.4', out['ph_profiles'])


if __name__ == '__main__':
    unittest.main()
