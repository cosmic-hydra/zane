"""
Tests for Models
"""

import pytest

torch = pytest.importorskip("torch")
from drug_discovery.models import EnsembleModel, MolecularGNN, MolecularTransformer
from torch_geometric.data import Data


class TestMolecularGNN:
    """Test Graph Neural Network model"""

    def test_model_creation(self):
        """Test model creation"""
        model = MolecularGNN(node_features=8, edge_features=3, hidden_dim=64, num_layers=2)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass"""
        model = MolecularGNN(node_features=8, edge_features=3, hidden_dim=64, num_layers=2)

        # Create dummy graph data
        x = torch.randn(10, 8)  # 10 nodes, 8 features
        edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
        edge_attr = torch.randn(20, 3)  # 3 edge features
        batch = torch.zeros(10, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        output = model(data)

        assert output is not None
        assert output.shape[1] == 1  # Output dimension


class TestMolecularTransformer:
    """Test Transformer model"""

    def test_model_creation(self):
        """Test model creation"""
        model = MolecularTransformer(input_dim=2048, hidden_dim=256, num_layers=3)

        assert model is not None

    def test_forward_pass(self):
        """Test forward pass"""
        model = MolecularTransformer(input_dim=2048, hidden_dim=256, num_layers=3)

        # Dummy fingerprint data
        x = torch.randn(4, 2048)  # Batch of 4 fingerprints

        output = model(x)

        assert output is not None
        assert output.shape[0] == 4


class TestEnsembleModel:
    """Test Ensemble model"""

    def test_ensemble_creation(self):
        """Test ensemble creation"""
        model1 = torch.nn.Linear(10, 1)
        model2 = torch.nn.Linear(10, 1)

        ensemble = EnsembleModel([model1, model2])

        assert ensemble is not None
        assert len(ensemble.models) == 2

    def test_ensemble_forward(self):
        """Test ensemble forward pass"""
        model1 = torch.nn.Linear(10, 1)
        model2 = torch.nn.Linear(10, 1)

        ensemble = EnsembleModel([model1, model2])

        x = torch.randn(4, 10)
        output = ensemble(x)

        assert output is not None
        assert output.shape == (4, 1)
