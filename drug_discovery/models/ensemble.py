"""Ensemble Models combining multiple approaches."""

from __future__ import annotations

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    """Ensemble of multiple models with learned weighting."""

    def __init__(self, models: list[nn.Module], learnable_weights: bool = True):
        """Initialize ensemble model.

        Args:
            models: List of models to ensemble.
            learnable_weights: Whether to learn ensemble weights.
        """
        super().__init__()

        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        else:
            self.register_buffer("weights", torch.ones(self.num_models) / self.num_models)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Forward pass through all models.

        Returns:
            Weighted ensemble prediction.
        """
        predictions = []

        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch_size, output_dim]

        # Normalize weights
        weights = torch.softmax(self.weights, dim=0)

        # Weighted average
        ensemble_pred = (predictions * weights.view(-1, 1, 1)).sum(dim=0)

        return ensemble_pred

    def get_individual_predictions(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Get predictions from individual models.

        Returns:
            Dictionary mapping model indices to their predictions.
        """
        predictions = {}

        for i, model in enumerate(self.models):
            pred = model(*args, **kwargs)
            predictions[f"model_{i}"] = pred

        return predictions


class MultiTaskModel(nn.Module):
    """Multi-task learning model for predicting multiple properties."""

    def __init__(self, base_model: nn.Module, num_tasks: int, shared_dim: int = 128, task_specific_dim: int = 64):
        """Initialize multi-task model.

        Args:
            base_model: Base feature extractor.
            num_tasks: Number of prediction tasks.
            shared_dim: Shared representation dimension.
            task_specific_dim: Task-specific dimension.
        """
        super().__init__()

        self.base_model = base_model
        self.num_tasks = num_tasks

        # Task-specific heads
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(shared_dim, task_specific_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(task_specific_dim, 1),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Forward pass through multi-task model.

        Returns:
            Dictionary of predictions for each task.
        """
        # Get shared representation
        shared_features = self.base_model(*args, **kwargs)

        # Task-specific predictions
        predictions = {}
        for i, head in enumerate(self.task_heads):
            predictions[f"task_{i}"] = head(shared_features)

        return predictions


class HybridModel(nn.Module):
    """Hybrid model combining GNN and Transformer."""

    def __init__(self, gnn_model: nn.Module, transformer_model: nn.Module, fusion_dim: int = 128, output_dim: int = 1):
        """Initialize hybrid model.

        Args:
            gnn_model: Graph neural network.
            transformer_model: Transformer model.
            fusion_dim: Fusion layer dimension.
            output_dim: Output dimension.
        """
        super().__init__()

        self.gnn_model = gnn_model
        self.transformer_model = transformer_model

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(2, fusion_dim),  # Combine 2 model outputs
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, output_dim),
        )

    def forward(self, graph_data: object, fingerprint_data: object) -> torch.Tensor:
        """Forward pass through hybrid model.

        Args:
            graph_data: Graph data for GNN.
            fingerprint_data: Fingerprint data for Transformer.

        Returns:
            Fused prediction tensor.
        """
        gnn_pred = self.gnn_model(graph_data)
        transformer_pred = self.transformer_model(fingerprint_data)

        # Concatenate predictions
        combined = torch.cat([gnn_pred, transformer_pred], dim=-1)

        # Fusion
        output = self.fusion(combined)

        return output
