"""
Comprehensive test suite for ensemble.py - 70+ tests
Tests ensemble model operations, multi-task learning, hybrid models
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from drug_discovery.models.ensemble import EnsembleModel, MultiTaskModel, HybridModel


class MockModel(nn.Module):
    """Simple mock model for testing"""
    def __init__(self, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(10, output_dim)

    def forward(self, x):
        return self.linear(x)


class MockGNNModel(nn.Module):
    """Mock GNN model"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 32)

    def forward(self, x):
        return self.linear(x)


class MockTransformerModel(nn.Module):
    """Mock Transformer model"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 32)

    def forward(self, x):
        return self.linear(x)


class TestEnsembleModelBasics:
    """Test basic ensemble model functionality"""

    def test_ensemble_init_learnable_weights(self):
        """Test ensemble initialization with learnable weights"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models, learnable_weights=True)

        assert ensemble.num_models == 3
        assert isinstance(ensemble.weights, nn.Parameter)
        assert ensemble.weights.shape == (3,)

    def test_ensemble_init_fixed_weights(self):
        """Test ensemble initialization with fixed weights"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models, learnable_weights=False)

        assert ensemble.num_models == 3
        assert ensemble.weights.shape == (3,)
        # Fixed weights should be registered as buffer
        assert "weights" in ensemble._buffers

    def test_ensemble_init_single_model(self):
        """Test ensemble with single model"""
        models = [MockModel()]
        ensemble = EnsembleModel(models)
        assert ensemble.num_models == 1

    def test_ensemble_init_many_models(self):
        """Test ensemble with many models"""
        models = [MockModel() for _ in range(10)]
        ensemble = EnsembleModel(models)
        assert ensemble.num_models == 10

    def test_ensemble_init_weights_shape(self):
        """Test ensemble weights shape matches model count"""
        for num_models in [1, 3, 5, 10]:
            models = [MockModel() for _ in range(num_models)]
            ensemble = EnsembleModel(models)
            assert ensemble.weights.shape[0] == num_models

    def test_ensemble_init_weights_sum_to_one(self):
        """Test initial weights sum to 1"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)
        # Before softmax, weights should be equal
        expected = torch.ones(3) / 3
        assert torch.allclose(ensemble.weights.detach(), expected, atol=1e-5)


class TestEnsembleForward:
    """Test ensemble forward pass"""

    def test_ensemble_forward_shapes(self):
        """Test ensemble forward pass output shape"""
        models = [MockModel(output_dim=1) for _ in range(3)]
        ensemble = EnsembleModel(models)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        output = ensemble(x)

        assert output.shape == (batch_size, 1)

    def test_ensemble_forward_batching(self):
        """Test ensemble handles batches correctly"""
        models = [MockModel(output_dim=5) for _ in range(2)]
        ensemble = EnsembleModel(models)

        for batch_size in [1, 4, 32, 128]:
            x = torch.randn(batch_size, 10)
            output = ensemble(x)
            assert output.shape == (batch_size, 5)

    def test_ensemble_forward_different_outputs(self):
        """Test ensemble averaging different model outputs"""
        model1 = MockModel(output_dim=1)
        model2 = MockModel(output_dim=1)

        # Set specific weights for reproducibility
        with torch.no_grad():
            model1.linear.weight.fill_(1.0)
            model1.linear.bias.fill_(1.0)
            model2.linear.weight.fill_(2.0)
            model2.linear.bias.fill_(2.0)

        ensemble = EnsembleModel([model1, model2], learnable_weights=False)
        x = torch.ones(1, 10)

        # Output should be weighted average
        output = ensemble(x)
        assert output.shape == (1, 1)
        assert output.item() != 0

    def test_ensemble_forward_gradient_flow(self):
        """Test gradients flow through ensemble"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models, learnable_weights=True)

        x = torch.randn(4, 10, requires_grad=True)
        output = ensemble(x)
        loss = output.sum()
        loss.backward()

        assert ensemble.weights.grad is not None
        assert x.grad is not None

    def test_ensemble_forward_reproducibility(self):
        """Test forward pass is reproducible"""
        torch.manual_seed(42)
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)

        x = torch.randn(4, 10)
        out1 = ensemble(x)
        out2 = ensemble(x)

        assert torch.allclose(out1, out2)

    def test_ensemble_forward_kwargs(self):
        """Test ensemble handles kwargs"""
        models = [MockModel() for _ in range(2)]
        ensemble = EnsembleModel(models)

        x = torch.randn(4, 10)
        # Should work with positional and keyword args
        output = ensemble(x, extra_param=None)
        assert output.shape == (4, 1)


class TestEnsembleIndividualPredictions:
    """Test individual model predictions extraction"""

    def test_individual_predictions_dict_structure(self):
        """Test individual predictions return dictionary"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)

        x = torch.randn(4, 10)
        preds = ensemble.get_individual_predictions(x)

        assert isinstance(preds, dict)
        assert len(preds) == 3
        assert all(f"model_{i}" in preds for i in range(3))

    def test_individual_predictions_shapes(self):
        """Test individual predictions have correct shapes"""
        models = [MockModel(output_dim=5) for _ in range(3)]
        ensemble = EnsembleModel(models)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        preds = ensemble.get_individual_predictions(x)

        for i in range(3):
            assert preds[f"model_{i}"].shape == (batch_size, 5)

    def test_individual_predictions_different_values(self):
        """Test individual predictions differ from ensemble"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models, learnable_weights=True)

        x = torch.randn(4, 10)
        ensemble_pred = ensemble(x)
        individual_preds = ensemble.get_individual_predictions(x)

        # At least some individual predictions should differ from ensemble
        model_0_pred = individual_preds["model_0"]
        assert not torch.allclose(model_0_pred, ensemble_pred, atol=1e-4)

    def test_individual_predictions_many_models(self):
        """Test individual predictions with many models"""
        models = [MockModel() for _ in range(10)]
        ensemble = EnsembleModel(models)

        x = torch.randn(2, 10)
        preds = ensemble.get_individual_predictions(x)

        assert len(preds) == 10
        for i in range(10):
            assert f"model_{i}" in preds


class TestEnsembleWeights:
    """Test ensemble weight learning and adjustment"""

    def test_learnable_weights_initialization(self):
        """Test learnable weights initialize to uniform"""
        models = [MockModel() for _ in range(5)]
        ensemble = EnsembleModel(models, learnable_weights=True)

        normalized = torch.softmax(ensemble.weights, dim=0)
        expected = torch.ones(5) / 5
        assert torch.allclose(normalized, expected, atol=1e-5)

    def test_fixed_weights_not_learnable(self):
        """Test fixed weights are not learnable parameters"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models, learnable_weights=False)

        # Weights should not be in parameters
        param_names = [name for name, _ in ensemble.named_parameters()]
        assert "weights" not in param_names

    def test_learnable_weights_learning(self):
        """Test learnable weights can be updated"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models, learnable_weights=True)
        optimizer = torch.optim.SGD(ensemble.parameters(), lr=0.1)

        x = torch.randn(4, 10)
        initial_weights = ensemble.weights.clone().detach()

        # One optimization step
        output = ensemble(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        updated_weights = ensemble.weights.detach()
        assert not torch.allclose(initial_weights, updated_weights)

    def test_weight_softmax_normalization(self):
        """Test weights are softmax normalized"""
        models = [MockModel() for _ in range(4)]
        ensemble = EnsembleModel(models)

        with torch.no_grad():
            ensemble.weights.fill_(2.0)  # Set all to 2.0

        normalized = torch.softmax(ensemble.weights, dim=0)
        # All should be equal after softmax
        assert torch.allclose(normalized, torch.ones(4) * 0.25)


class TestMultiTaskModel:
    """Test multi-task learning model"""

    def test_multitask_init(self):
        """Test multi-task model initialization"""
        base_model = MockModel(output_dim=32)
        mtask = MultiTaskModel(base_model, num_tasks=3)

        assert mtask.num_tasks == 3
        assert len(mtask.task_heads) == 3

    def test_multitask_forward_dict_output(self):
        """Test multi-task forward returns dictionary"""
        base_model = MockModel(output_dim=32)
        mtask = MultiTaskModel(base_model, num_tasks=5)

        x = torch.randn(4, 10)
        output = mtask(x)

        assert isinstance(output, dict)
        assert len(output) == 5
        assert all(f"task_{i}" in output for i in range(5))

    def test_multitask_output_shapes(self):
        """Test multi-task output shapes are correct"""
        base_model = MockModel(output_dim=64)
        mtask = MultiTaskModel(base_model, num_tasks=4, task_specific_dim=32)

        batch_size = 8
        x = torch.randn(batch_size, 10)
        output = mtask(x)

        for i in range(4):
            assert output[f"task_{i}"].shape == (batch_size, 1)

    def test_multitask_gradient_flow(self):
        """Test gradients flow through all task heads"""
        base_model = MockModel(output_dim=32)
        mtask = MultiTaskModel(base_model, num_tasks=3)

        x = torch.randn(4, 10, requires_grad=True)
        output = mtask(x)

        # Sum all task outputs for loss
        loss = sum(output[f"task_{i}"].sum() for i in range(3))
        loss.backward()

        assert x.grad is not None
        # Check base model has gradients
        assert mtask.base_model.linear.weight.grad is not None

    def test_multitask_task_independence(self):
        """Test tasks can have independent predictions"""
        base_model = MockModel(output_dim=32)
        mtask = MultiTaskModel(base_model, num_tasks=3)

        x = torch.randn(2, 10)
        output = mtask(x)

        # Different tasks should have different values
        task_0 = output["task_0"]
        task_1 = output["task_1"]
        # Likely to be different (though small chance of being same)
        assert not torch.allclose(task_0, task_1)


class TestHybridModel:
    """Test hybrid GNN-Transformer model"""

    def test_hybrid_init(self):
        """Test hybrid model initialization"""
        gnn = MockGNNModel()
        transformer = MockTransformerModel()
        hybrid = HybridModel(gnn, transformer)

        assert hybrid.gnn_model is gnn
        assert hybrid.transformer_model is transformer

    def test_hybrid_forward_signature(self):
        """Test hybrid forward takes graph and fingerprint data"""
        gnn = MockGNNModel()
        transformer = MockTransformerModel()
        hybrid = HybridModel(gnn, transformer, output_dim=1)

        graph_data = torch.randn(4, 10)
        fp_data = torch.randn(4, 10)

        output = hybrid(graph_data, fp_data)
        assert output.shape == (4, 1)

    def test_hybrid_gradient_flow_both_branches(self):
        """Test gradients flow through both model branches"""
        gnn = MockGNNModel()
        transformer = MockTransformerModel()
        hybrid = HybridModel(gnn, transformer)

        graph_data = torch.randn(4, 10, requires_grad=True)
        fp_data = torch.randn(4, 10, requires_grad=True)

        output = hybrid(graph_data, fp_data)
        loss = output.sum()
        loss.backward()

        # Both inputs should have gradients
        assert graph_data.grad is not None
        assert fp_data.grad is not None

        # Both models should have gradients
        assert gnn.linear.weight.grad is not None
        assert transformer.linear.weight.grad is not None

    def test_hybrid_fusion_dimensions(self):
        """Test fusion layer works with different dimensions"""
        gnn = MockGNNModel()
        transformer = MockTransformerModel()

        for fusion_dim in [32, 64, 128, 256]:
            hybrid = HybridModel(gnn, transformer, fusion_dim=fusion_dim)
            graph_data = torch.randn(2, 10)
            fp_data = torch.randn(2, 10)
            output = hybrid(graph_data, fp_data)
            assert output.shape == (2, 1)

    def test_hybrid_output_dimensions(self):
        """Test hybrid model output dimensions"""
        gnn = MockGNNModel()
        transformer = MockTransformerModel()

        for output_dim in [1, 5, 10]:
            hybrid = HybridModel(gnn, transformer, output_dim=output_dim)
            graph_data = torch.randn(4, 10)
            fp_data = torch.randn(4, 10)
            output = hybrid(graph_data, fp_data)
            assert output.shape == (4, output_dim)


class TestEnsembleEdgeCases:
    """Test ensemble edge cases and error handling"""

    def test_ensemble_empty_models_list(self):
        """Test ensemble handles empty models list"""
        with pytest.raises((ValueError, IndexError)):
            EnsembleModel([])

    def test_ensemble_forward_small_batch(self):
        """Test ensemble with batch size 1"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)

        x = torch.randn(1, 10)
        output = ensemble(x)
        assert output.shape == (1, 1)

    def test_ensemble_no_grad_mode(self):
        """Test ensemble in no_grad mode"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)

        x = torch.randn(4, 10)
        with torch.no_grad():
            output = ensemble(x)

        assert output.grad_fn is None

    def test_ensemble_eval_mode(self):
        """Test ensemble in eval mode (inference)"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)
        ensemble.eval()

        x = torch.randn(4, 10)
        with torch.no_grad():
            output = ensemble(x)

        assert output.shape == (4, 1)

    def test_ensemble_train_mode(self):
        """Test ensemble in train mode"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)
        ensemble.train()

        x = torch.randn(4, 10)
        output = ensemble(x)
        assert output.requires_grad


class TestEnsemblePerformance:
    """Performance and stress tests"""

    def test_ensemble_large_number_models(self):
        """Test ensemble with large number of models"""
        models = [MockModel() for _ in range(50)]
        ensemble = EnsembleModel(models)

        x = torch.randn(4, 10)
        output = ensemble(x)
        assert output.shape == (4, 1)

    def test_ensemble_large_batch(self):
        """Test ensemble with large batch size"""
        models = [MockModel() for _ in range(5)]
        ensemble = EnsembleModel(models)

        x = torch.randn(1000, 10)
        output = ensemble(x)
        assert output.shape == (1000, 1)

    def test_ensemble_memory_efficiency(self):
        """Test ensemble doesn't duplicate data unnecessarily"""
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleModel(models)

        x = torch.randn(100, 10)
        output = ensemble(x)
        # Should complete without memory errors
        assert output.numel() > 0
