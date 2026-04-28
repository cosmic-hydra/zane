"""Test Suite for ZANE 2026 Q2 Upgrade Modules."""
import numpy as np
import pytest


class TestScientificValidation:
    def test_metrics_regression(self):
        from drug_discovery.validation.scientific_validation import compute_metrics
        y_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_p = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        m = compute_metrics(y_t, y_p, "regression")
        assert m["rmse"] < 0.3 and m["r2"] > 0.9

    def test_scaffold_split(self):
        from drug_discovery.validation.scientific_validation import scaffold_split
        smi = ["CCO","CCN","c1ccccc1","CC(=O)O","CCCC","CCC","CC=O","c1ccncc1","CC(C)O","CCCN"]
        tr, va, te = scaffold_split(smi, 0.7, 0.15, seed=42)
        assert len(tr)+len(va)+len(te) == len(smi)
        assert len(set(tr)&set(va)) == 0

    def test_scaffold_kfold(self):
        from drug_discovery.validation.scientific_validation import scaffold_kfold
        folds = scaffold_kfold(["CCO","CCN","c1ccccc1","CC(=O)O","CCCC"]*4, n_folds=3)
        assert len(folds) == 3

    def test_bootstrap_ci(self):
        from drug_discovery.validation.scientific_validation import bootstrap_ci
        ci = bootstrap_ci(np.array([0.85,0.87,0.82,0.89,0.86]), n_bootstrap=1000)
        assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]

    def test_config_hash(self):
        from drug_discovery.validation.scientific_validation import config_hash
        assert config_hash({"lr":0.001,"epochs":100}) == config_hash({"epochs":100,"lr":0.001})

    def test_experiment_report(self):
        from drug_discovery.validation.scientific_validation import ExperimentReport
        r = ExperimentReport(model_name="egnn")
        r.fold_metrics = [{"rmse": 0.5}, {"rmse": 0.4}, {"rmse": 0.45}]
        r.compute_aggregates()
        assert r.aggregate_metrics["rmse"]["mean"] == pytest.approx(0.45, abs=0.01)

class TestDataPipeline:
    def test_smiles_validation(self):
        from drug_discovery.data.pipeline import is_valid_smiles_fast
        assert is_valid_smiles_fast("CCO") and not is_valid_smiles_fast("") and not is_valid_smiles_fast("CC(")

    def test_validate_batch(self):
        from drug_discovery.data.pipeline import validate_batch
        r = validate_batch(["CCO","CCN","INVALID!!!!","c1ccccc1"])
        assert r["total"] == 4 and r["valid"] >= 2

    def test_lipinski(self):
        from drug_discovery.data.pipeline import lipinski_filter
        assert lipinski_filter({"mol_weight":300,"logp":2.0,"hbd":1,"hba":3})["passes"]

    def test_tanimoto(self):
        from drug_discovery.data.pipeline import tanimoto_similarity
        fp = np.array([1,0,1,1,0], dtype=np.float32)
        assert tanimoto_similarity(fp, fp) == pytest.approx(1.0)

    def test_dataset(self):
        from drug_discovery.data.pipeline import MolecularDataset
        assert MolecularDataset(smiles=["CCO","CCN"]).size == 2

class TestUncertainty:
    def test_conformal(self):
        from drug_discovery.evaluation.uncertainty import ConformalPredictor
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(np.array([1.,2.,3.,4.,5.]), np.array([1.1,1.9,3.2,3.8,5.1]))
        lo, hi = cp.predict_interval(np.array([1.,2.,3.]))
        assert all(lo < hi)

    def test_ece(self):
        from drug_discovery.evaluation.uncertainty import expected_calibration_error
        assert 0 <= expected_calibration_error(np.array([.9,.8,.3,.1,.7]), np.array([1,1,0,0,1]), 5) <= 1

class TestMultiObjective:
    def test_pareto(self):
        from drug_discovery.optimization.multi_objective import is_pareto_efficient
        assert is_pareto_efficient(np.array([[1,4],[2,3],[3,2],[4,1],[2,2]])).sum() >= 1

    def test_hypervolume(self):
        from drug_discovery.optimization.multi_objective import hypervolume_indicator
        assert hypervolume_indicator(np.array([[2.,3.],[3.,2.]]), np.array([0.,0.])) > 0

    def test_gp(self):
        from drug_discovery.optimization.multi_objective import GaussianProcessSurrogate
        gp = GaussianProcessSurrogate()
        gp.fit(np.random.rand(10, 3), np.random.rand(10))
        m, v = gp.predict(np.random.rand(2, 3))
        assert m.shape == (2,) and all(v > 0)

    def test_mobo(self):
        from drug_discovery.optimization.multi_objective import MOBOConfig, MultiObjectiveBayesianOptimizer
        opt = MultiObjectiveBayesianOptimizer(MOBOConfig(objective_names=["a","b"],
            objective_directions=["maximize","maximize"], ref_point=[0.,0.], num_mc_samples=10))
        opt.tell(np.random.rand(5,3), np.random.rand(5,2))
        assert opt.summary()["total_observations"] == 5
