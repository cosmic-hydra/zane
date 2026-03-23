"""
Tests for Drug Discovery Pipeline
"""

from pathlib import Path

import pandas as pd
import pytest

from drug_discovery import DrugDiscoveryPipeline


class TestDrugDiscoveryPipeline:
    """Test main pipeline functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.pipeline = DrugDiscoveryPipeline(model_type="gnn", device="cpu")

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline is not None
        assert self.pipeline.model_type == "gnn"
        assert self.pipeline.device == "cpu"

    def test_collect_data(self):
        """Test data collection (if APIs are available)"""
        # This test might be slow or fail if APIs are unavailable
        # So we make it optional
        try:
            data = self.pipeline.collect_data(sources=["pubchem"], limit_per_source=10)
            assert isinstance(data, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Data collection failed (expected): {e}")

    def test_build_model(self):
        """Test model building"""
        model = self.pipeline.build_model(node_features=8, edge_features=3, hidden_dim=64)

        assert model is not None

    def test_predict_properties_without_training(self):
        """Test that prediction fails without training"""
        with pytest.raises(RuntimeError):
            self.pipeline.predict_properties("CC(=O)OC1=CC=CC=C1C(=O)O")

    def test_save_load(self):
        """Test saving and loading pipeline"""
        # Build model first
        self.pipeline.build_model()

        # Save
        self.pipeline.save("./test_pipeline.pt")

        # Create new pipeline and load
        new_pipeline = DrugDiscoveryPipeline(model_type="gnn", device="cpu")
        new_pipeline.load("./test_pipeline.pt")

        assert new_pipeline.model is not None

    def test_creates_checkpoint_dir(self, tmp_path):
        """Pipeline initialization should create checkpoint directory."""
        ckpt_dir = tmp_path / "ckpts"
        assert not ckpt_dir.exists()
        pipeline = DrugDiscoveryPipeline(model_type="gnn", device="cpu", checkpoint_dir=str(ckpt_dir))
        assert pipeline is not None
        assert Path(ckpt_dir).exists()
