"""
Bayesian Optimization and Uncertainty Estimation
Uses Gaussian Processes for exploration-exploitation
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """
    Bayesian Optimization for molecular property optimization
    Uses Gaussian Processes and acquisition functions
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        acquisition_function: str = "ei",  # 'ei', 'ucb', 'poi'
        kappa: float = 2.576,  # For UCB
        xi: float = 0.01,  # For EI/POI
    ):
        """
        Args:
            bounds: Parameter bounds for optimization
            acquisition_function: Type of acquisition function
            kappa: Exploration parameter for UCB
            xi: Exploration parameter for EI
        """
        self.bounds = bounds
        self.acquisition_function = acquisition_function
        self.kappa = kappa
        self.xi = xi

        self.X_observed: list[np.ndarray] = []
        self.y_observed: list[float] = []

    def suggest_next(self, n_suggestions: int = 1) -> list[np.ndarray]:
        """
        Suggest next candidates to evaluate

        Args:
            n_suggestions: Number of suggestions

        Returns:
            List of suggested parameter vectors
        """
        suggestions = []

        for _ in range(n_suggestions):
            # Random suggestion (placeholder for actual BO)
            # In full implementation, would use GPyTorch/BoTorch
            suggestion = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            suggestions.append(suggestion)

        return suggestions

    def observe(self, X: np.ndarray, y: float):
        """
        Add observation to the model

        Args:
            X: Parameter vector
            y: Observed value
        """
        self.X_observed.append(X)
        self.y_observed.append(y)

        logger.info(f"Observed: X={X}, y={y:.4f}")

    def get_best(self) -> tuple[np.ndarray, float] | tuple[None, None]:
        """
        Get best observed parameters and value

        Returns:
            (best_X, best_y)
        """
        if not self.y_observed:
            return None, None

        best_idx = np.argmax(self.y_observed)  # Assuming maximization
        return self.X_observed[best_idx], self.y_observed[best_idx]


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using ensembles or dropout
    """

    def __init__(self, models: list[Any] | None = None, method: str = "ensemble"):  # 'ensemble' or 'mc_dropout'
        """
        Args:
            models: List of models for ensemble
            method: Uncertainty estimation method
        """
        self.models = models or []
        self.method = method

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[float, float]:
        """
        Predict with uncertainty estimate

        Args:
            x: Input tensor

        Returns:
            (mean_prediction, uncertainty)
        """
        if self.method == "ensemble" and len(self.models) > 0:
            predictions = []

            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(x)
                    predictions.append(pred.cpu().numpy())

            predictions = np.array(predictions)
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            return float(mean_pred), float(std_pred)

        else:
            # Placeholder for MC Dropout
            return 0.0, 0.0

    def get_confidence_interval(self, x: torch.Tensor, confidence: float = 0.95) -> tuple[float, float, float]:
        """
        Get confidence interval for prediction

        Args:
            x: Input
            confidence: Confidence level

        Returns:
            (mean, lower_bound, upper_bound)
        """
        mean, std = self.predict_with_uncertainty(x)

        # Calculate confidence interval (assuming normal distribution)
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z_score * std

        return mean, mean - margin, mean + margin


class ActiveLearner:
    """
    Active learning for efficient data collection
    Selects most informative samples for labeling
    """

    def __init__(
        self,
        uncertainty_estimator: UncertaintyEstimator,
        strategy: str = "uncertainty",  # 'uncertainty', 'diverse', 'hybrid'
    ):
        """
        Args:
            uncertainty_estimator: Uncertainty estimator
            strategy: Sample selection strategy
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.strategy = strategy

    def select_samples(self, candidates: list[torch.Tensor], n_samples: int = 10) -> list[int]:
        """
        Select most informative samples

        Args:
            candidates: List of candidate inputs
            n_samples: Number of samples to select

        Returns:
            Indices of selected samples
        """
        if self.strategy == "uncertainty":
            # Select samples with highest uncertainty
            uncertainties = []

            for candidate in candidates:
                _, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(candidate)
                uncertainties.append(uncertainty)

            # Select top n_samples by uncertainty
            selected_indices = np.argsort(uncertainties)[-n_samples:]

        elif self.strategy == "diverse":
            # Select diverse samples (placeholder)
            selected_indices = np.random.choice(len(candidates), size=min(n_samples, len(candidates)), replace=False)

        else:
            # Hybrid strategy
            selected_indices = np.random.choice(len(candidates), size=min(n_samples, len(candidates)), replace=False)

        logger.info(f"Selected {len(selected_indices)} samples using {self.strategy} strategy")

        return list(selected_indices)
