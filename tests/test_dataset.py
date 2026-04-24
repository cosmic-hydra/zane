"""
Tests for Molecular Dataset and Featurization
"""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from drug_discovery.data import (
    MolecularDataset,
    MolecularFeaturizer,
    murcko_scaffold_kfold_split_molecular,
    murcko_scaffold_split_molecular,
    train_test_split_molecular,
)


class TestMolecularFeaturizer:
    """Test molecular featurization"""

    def setup_method(self):
        """Setup test fixtures"""
        self.featurizer = MolecularFeaturizer()

    def test_smiles_to_graph(self):
        """Test SMILES to graph conversion"""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        graph = self.featurizer.smiles_to_graph(smiles)

        assert graph is not None
        assert hasattr(graph, "x")  # Node features
        assert hasattr(graph, "edge_index")  # Edge indices
        assert hasattr(graph, "edge_attr")  # Edge features

    def test_invalid_smiles_to_graph(self):
        """Test invalid SMILES handling"""
        invalid_smiles = "INVALID_SMILES"
        graph = self.featurizer.smiles_to_graph(invalid_smiles)

        assert graph is None

    def test_smiles_to_fingerprint(self):
        """Test SMILES to fingerprint conversion"""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        fp = self.featurizer.smiles_to_fingerprint(smiles, n_bits=2048)

        assert fp is not None
        assert len(fp) == 2048
        assert fp.dtype == torch.float32 or fp.dtype == "float32"

    def test_compute_descriptors(self):
        """Test molecular descriptor computation"""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        descriptors = self.featurizer.compute_molecular_descriptors(smiles)

        assert descriptors is not None
        assert len(descriptors) > 0


class TestMolecularDataset:
    """Test MolecularDataset"""

    def setup_method(self):
        """Setup test fixtures"""
        self.data = pd.DataFrame(
            {
                "smiles": [
                    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
                    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
                    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
                ],
                "property": [1.0, 2.0, 3.0],
            }
        )

    def test_dataset_creation_graph(self):
        """Test dataset creation with graph featurization"""
        dataset = MolecularDataset(self.data, smiles_col="smiles", target_col="property", featurization="graph")

        assert len(dataset) > 0
        assert len(dataset) <= len(self.data)

    def test_dataset_creation_fingerprint(self):
        """Test dataset creation with fingerprint featurization"""
        dataset = MolecularDataset(self.data, smiles_col="smiles", target_col="property", featurization="fingerprint")

        assert len(dataset) > 0

    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        dataset = MolecularDataset(self.data, smiles_col="smiles", featurization="fingerprint")

        item = dataset[0]
        assert item is not None

    def test_seeded_split_is_reproducible(self):
        """Train/test splits should be stable for the same seed."""
        dataset = MolecularDataset(self.data, smiles_col="smiles", target_col="property", featurization="fingerprint")
        train_a, test_a = train_test_split_molecular(dataset, test_size=0.34, seed=123)
        train_b, test_b = train_test_split_molecular(dataset, test_size=0.34, seed=123)

        assert list(train_a.indices) == list(train_b.indices)
        assert list(test_a.indices) == list(test_b.indices)

    def test_scaffold_split_is_reproducible(self):
        """Scaffold split should be stable for the same seed."""
        dataset = MolecularDataset(self.data, smiles_col="smiles", target_col="property", featurization="fingerprint")
        train_a, test_a = murcko_scaffold_split_molecular(dataset, test_size=0.34, seed=123)
        train_b, test_b = murcko_scaffold_split_molecular(dataset, test_size=0.34, seed=123)

        assert list(train_a.indices) == list(train_b.indices)
        assert list(test_a.indices) == list(test_b.indices)

    def test_scaffold_kfold_reproducible(self):
        """Scaffold k-fold splits should be reproducible for the same seed."""
        dataset = MolecularDataset(self.data, smiles_col="smiles", target_col="property", featurization="fingerprint")
        folds_a = murcko_scaffold_kfold_split_molecular(dataset, n_splits=3, seed=7)
        folds_b = murcko_scaffold_kfold_split_molecular(dataset, n_splits=3, seed=7)

        assert len(folds_a) == len(folds_b)
        for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b):
            assert list(tr_a.indices) == list(tr_b.indices)
            assert list(te_a.indices) == list(te_b.indices)

    def test_scaffold_kfold_covers_all_samples(self):
        """Across k folds, each sample should appear in a test fold exactly once."""
        dataset = MolecularDataset(self.data, smiles_col="smiles", target_col="property", featurization="fingerprint")
        folds = murcko_scaffold_kfold_split_molecular(dataset, n_splits=3, seed=9)

        seen_test = []
        for _, test_fold in folds:
            seen_test.extend(list(test_fold.indices))

        assert sorted(seen_test) == list(range(len(dataset)))
