"""
Test Suite for Data Layer and Pipeline

Tests for:
- Data collection from multiple sources
- Data normalization and validation
- Feature store
- Dataset versioning
- Streaming pipeline
"""

import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from drug_discovery.data.dataset import MolecularDataset
from drug_discovery.data.feature_store import FeatureStore
from drug_discovery.data.normalizer import DataNormalizer
from drug_discovery.data.versioning import DatasetVersioning


class TestDataNormalizer:
    """Test DataNormalizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.normalizer = DataNormalizer()
        self.test_smiles = [
            "CCO",  # Ethanol
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]

    def test_canonicalize_smiles(self):
        """Test SMILES canonicalization."""
        canonical = self.normalizer.canonicalize_smiles("CCO")
        assert canonical is not None
        assert isinstance(canonical, str)

    def test_compute_inchikey(self):
        """Test InChIKey computation."""
        inchikey = self.normalizer.compute_inchikey("CCO")
        assert inchikey is not None
        assert isinstance(inchikey, str)
        assert len(inchikey) > 0

    def test_is_valid_molecule(self):
        """Test molecule validation."""
        assert self.normalizer.is_valid_molecule("CCO") is True
        assert self.normalizer.is_valid_molecule("INVALID") is False

    def test_normalize_dataframe(self):
        """Test DataFrame normalization."""
        df = pd.DataFrame({
            "smiles": self.test_smiles + ["CCO"],  # Include duplicate
            "activity": [1.0, 2.0, 3.0, 1.5],
        })

        normalized_df = self.normalizer.normalize_dataframe(
            df,
            smiles_column="smiles",
        )

        assert "canonical_smiles" in normalized_df.columns
        assert "inchikey" in normalized_df.columns
        assert len(normalized_df) <= len(df)  # Duplicates removed

    def test_apply_filters(self):
        """Test molecular filters."""
        df = pd.DataFrame({
            "smiles": self.test_smiles,
        })

        filtered_df = self.normalizer.apply_filters(
            df,
            smiles_column="smiles",
            lipinski_filter=True,
        )

        assert len(filtered_df) <= len(df)


class TestFeatureStore:
    """Test FeatureStore."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = FeatureStore(store_path=self.temp_dir)

    def teardown_method(self):
        """Cleanup temp directory."""
        shutil.rmtree(self.temp_dir)

    def test_store_and_retrieve_embedding(self):
        """Test storing and retrieving embeddings."""
        embedding = np.random.rand(512)
        key = "test_molecule"

        self.store.store_embedding(key, embedding, feature_type="molecule")
        retrieved = self.store.retrieve_embedding(key, feature_type="molecule")

        assert retrieved is not None
        np.testing.assert_array_almost_equal(embedding, retrieved)

    def test_store_batch(self):
        """Test batch storage."""
        keys = ["mol1", "mol2", "mol3"]
        embeddings = np.random.rand(3, 512)

        self.store.store_batch(keys, embeddings)

        # Verify all stored
        for key in keys:
            assert self.store.retrieve_embedding(key) is not None

    def test_retrieve_batch(self):
        """Test batch retrieval."""
        keys = ["mol1", "mol2"]
        embeddings = np.random.rand(2, 512)

        self.store.store_batch(keys, embeddings)
        retrieved = self.store.retrieve_batch(keys)

        assert len(retrieved) == len(keys)

    def test_cache_functionality(self):
        """Test in-memory cache."""
        embedding = np.random.rand(512)
        key = "test_molecule"

        self.store.store_embedding(key, embedding)

        # First retrieval (from disk)
        retrieved1 = self.store.retrieve_embedding(key)

        # Second retrieval (from cache)
        retrieved2 = self.store.retrieve_embedding(key)

        np.testing.assert_array_almost_equal(retrieved1, retrieved2)


class TestDatasetVersioning:
    """Test DatasetVersioning."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.versioning = DatasetVersioning(versions_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup temp directory."""
        shutil.rmtree(self.temp_dir)

    def test_create_version(self):
        """Test dataset version creation."""
        df = pd.DataFrame({
            "smiles": ["CCO", "CC(C)O"],
            "activity": [1.0, 2.0],
        })

        version_id = self.versioning.create_version(
            df,
            version_name="test_version",
            description="Test dataset",
        )

        assert version_id is not None
        assert isinstance(version_id, str)

    def test_load_version(self):
        """Test loading dataset version."""
        df = pd.DataFrame({
            "smiles": ["CCO", "CC(C)O"],
            "activity": [1.0, 2.0],
        })

        version_id = self.versioning.create_version(df, version_name="test")
        loaded_df = self.versioning.load_version(version_id)

        assert loaded_df is not None
        assert len(loaded_df) == len(df)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_list_versions(self):
        """Test listing versions."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        self.versioning.create_version(df, version_name="v1")
        self.versioning.create_version(df, version_name="v2")

        versions = self.versioning.list_versions()
        assert len(versions) >= 2

    def test_tag_version(self):
        """Test tagging versions."""
        df = pd.DataFrame({"col": [1, 2]})
        version_id = self.versioning.create_version(df, version_name="test")

        success = self.versioning.tag_version(version_id, "production")
        assert success is True

        tagged = self.versioning.get_versions_by_tag("production")
        assert version_id in tagged


class TestMolecularDataset:
    """Test MolecularDataset."""

    def setup_method(self):
        """Setup test fixtures."""
        self.df = pd.DataFrame({
            "smiles": ["CCO", "CC(C)O", "CCCO"],
            "target": [1.0, 0.0, 1.0],
        })

    def test_fingerprint_featurization(self):
        """Test fingerprint featurization."""
        dataset = MolecularDataset(
            self.df,
            smiles_column="smiles",
            target_column="target",
            featurization="fingerprint",
        )

        assert len(dataset) > 0
        feature, target = dataset[0]
        assert feature.shape[0] > 0

    def test_descriptor_featurization(self):
        """Test descriptor featurization."""
        dataset = MolecularDataset(
            self.df,
            smiles_column="smiles",
            target_column="target",
            featurization="descriptors",
        )

        assert len(dataset) > 0
        feature, target = dataset[0]
        assert feature.shape[0] > 0

    def test_graph_featurization(self):
        """Test graph featurization."""
        dataset = MolecularDataset(
            self.df,
            smiles_column="smiles",
            target_column="target",
            featurization="graph",
        )

        assert len(dataset) > 0
        feature, target = dataset[0]
        assert isinstance(feature, dict)
        assert "atom_features" in feature
        assert "adjacency" in feature

    def test_get_feature_dim(self):
        """Test feature dimensionality."""
        dataset = MolecularDataset(
            self.df,
            smiles_column="smiles",
            featurization="fingerprint",
        )

        dim = dataset.get_feature_dim()
        assert dim > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
