
import pytest
from drug_discovery.benchmarking.moleculenet_eval import run_benchmark, DeepChemModelWrapper

class TestMoleculeNetBenchmarking:
    """Validate scientific benchmark suite integrity."""

    def test_run_benchmark_regression_integrity(self):
        """Ensure BACE regression benchmark beats mean baseline and is consistent."""
        # Use minimal seeds and epochs for fast verification
        result = run_benchmark("bace", DeepChemModelWrapper, seeds=(42, 123), epochs=1)
        
        assert result is not None
        assert result["dataset"] == "bace"
        assert "r2_mean" in result
        assert "r2_std" in result
        
        # Core assertions from plan:
        assert result["r2_mean"] > 0.0, f"BACE R\u00b2 {result['r2_mean']:.3f} failed to beat mean baseline."
        assert result["r2_std"] < 0.15, f"BACE R\u00b2 standard deviation {result['r2_std']:.3f} is too high."
        
    def test_run_benchmark_classification_integrity(self):
        """Ensure BBBP classification benchmark returns valid AUC."""
        result = run_benchmark("bbbp", DeepChemModelWrapper, seeds=(42,), epochs=1)
        
        assert result is not None
        assert result["dataset"] == "bbbp"
        assert "auc_mean" in result
        
        # AUC should be between 0.5 (random) and 1.0 (perfect)
        assert result["auc_mean"] >= 0.5, f"BBBP AUC {result['auc_mean']:.3f} is worse than random."
        assert result["auc_mean"] <= 1.0

    def test_invalid_dataset(self):
        """Ensure proper error for unknown datasets."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            run_benchmark("unknown_ds", DeepChemModelWrapper)
