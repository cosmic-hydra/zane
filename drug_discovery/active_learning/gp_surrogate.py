"""
Gaussian Process Surrogate Model for Molecular Property Prediction.

Implements a GP surrogate model that can score millions of low-fidelity
generated molecules efficiently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    logger.warning("PyTorch not available. Using sklearn GP.")

try:
    import gpytorch
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean, LinearMean
    from gpytorch.models import ExactGP

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    gpytorch = None
    ExactGP = None
    logger.warning("GPyTorch not available. Using sklearn GP.")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    from sklearn.gaussian_process.kernels import ConstantKernel as C

    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False
    GaussianProcessRegressor = None


@dataclass
class SurrogateConfig:
    """Configuration for Gaussian Process surrogate.

    Attributes:
        input_dim: Feature dimension.
        hidden_dim: Hidden layer dimension (for neural GP).
        kernel: Kernel type ('rbf', 'matern', 'poly').
        noise_std: Observation noise standard deviation.
        n_inducing: Number of inducing points (for sparse GP).
        learning_rate: Optimizer learning rate.
    """

    input_dim: int = 128
    hidden_dim: int = 64
    kernel: str = "rbf"
    noise_std: float = 0.01
    n_inducing: int = 100
    learning_rate: float = 0.001
    use_gpu: bool = True


class SimpleGPModel:
    """
    Simple Gaussian Process model using sklearn or manual implementation.

    Suitable for scoring millions of molecular candidates efficiently.
    """

    def __init__(
        self,
        input_dim: int = 128,
        kernel_type: str = "rbf",
        noise_std: float = 0.01,
    ):
        """
        Initialize GP model.

        Args:
            input_dim: Input feature dimension.
            kernel_type: Kernel type ('rbf', 'matern', 'linear').
            noise_std: Noise level.
        """
        self.input_dim = input_dim
        self.kernel_type = kernel_type
        self.noise_std = noise_std

        self.X_train = None
        self.y_train = None
        self._fitted = False

        if SKLEARN_GP_AVAILABLE:
            kernel = self._get_sklearn_kernel()
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=noise_std**2,
                n_restarts_optimizer=5,
                normalize_y=True,
            )
        else:
            self.model = None

        logger.info(f"SimpleGPModel initialized: dim={input_dim}, kernel={kernel_type}")

    def _get_sklearn_kernel(self):
        """Get sklearn kernel."""
        if self.kernel_type == "rbf":
            return C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * self.input_dim)
        elif self.kernel_type == "matern":
            return C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * self.input_dim, nu=2.5)
        else:
            return C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * self.input_dim)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GP model to data.

        Args:
            X: Training features (n_samples, input_dim).
            y: Training targets (n_samples,).
        """
        self.X_train = X
        self.y_train = y

        if SKLEARN_GP_AVAILABLE and self.model is not None:
            self.model.fit(X, y)
        else:
            # Manual GP fitting (simplified)
            self._fit_manual(X, y)

        self._fitted = True
        logger.info(f"GP fitted on {len(X)} samples")

    def _fit_manual(self, X: np.ndarray, y: np.ndarray) -> None:
        """Manual GP fitting using cholesky decomposition."""
        n = len(X)

        # Compute kernel matrix
        K = self._rbf_kernel(X, X)

        # Add noise
        K = K + self.noise_std**2 * np.eye(n)

        # Cholesky decomposition
        try:
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        except np.linalg.LinAlgError:
            # Fallback: add small jitter
            K = K + 1e-6 * np.eye(n)
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))

        self.X_train = X

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
        """RBF kernel computation."""
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)

        K = X1_sq + X2_sq.T - 2 * np.dot(X1, X2.T)
        K = np.exp(-0.5 * K / (length_scale**2))

        return K

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with GP.

        Args:
            X: Query points (n_samples, input_dim).
            return_std: Whether to return standard deviation.

        Returns:
            Tuple of (mean, std) predictions.
        """
        if not self._fitted:
            # Return prior
            return np.zeros(len(X)), np.ones(len(X))

        if SKLEARN_GP_AVAILABLE and self.model is not None:
            y_mean, y_std = self.model.predict(X, return_std=True)
            return y_mean, y_std
        else:
            return self._predict_manual(X)

    def _predict_manual(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Manual GP prediction."""
        # Kernel with training data
        K_star = self._rbf_kernel(X, self.X_train)

        # Predictive mean
        y_mean = np.dot(K_star, self.alpha)

        # Predictive variance
        K_ss = self._rbf_kernel(X, X)
        v = np.linalg.solve(self.L, K_star.T)
        y_var = np.diag(K_ss) - np.sum(v**2, axis=0)
        y_var = np.maximum(y_var, 1e-10)
        y_std = np.sqrt(y_var)

        return y_mean, y_std


class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model for molecular properties.

    Provides:
    - Fast prediction for scoring millions of candidates
    - Uncertainty quantification
    - Online learning with new data

    Example::

        surrogate = GaussianProcessSurrogate(input_dim=128)
        surrogate.fit(molecular_features, binding_energies)
        means, stds = surrogate.predict(candidate_features)
    """

    def __init__(self, config: SurrogateConfig | None = None):
        """
        Initialize GP surrogate.

        Args:
            config: Model configuration.
        """
        self.config = config or SurrogateConfig()

        if TORCH_AVAILABLE and GPYTORCH_AVAILABLE:
            self._setup_torch_gp()
        else:
            self._setup_sklearn_gp()

        self.X_buffer = []
        self.y_buffer = []
        self._fitted = False

        logger.info("GaussianProcessSurrogate initialized")

    def _setup_torch_gp(self) -> None:
        """Setup PyTorch/GPyTorch GP."""
        device = torch.device("cuda" if (torch.cuda.is_available() and self.config.use_gpu) else "cpu")
        self.device = device

        # Simple GP with RBF kernel
        self.gp = SimpleGPModel(
            input_dim=self.config.input_dim,
            kernel_type=self.config.kernel,
            noise_std=self.config.noise_std,
        )

    def _setup_sklearn_gp(self) -> None:
        """Setup sklearn GP fallback."""
        self.gp = SimpleGPModel(
            input_dim=self.config.input_dim,
            kernel_type=self.config.kernel,
            noise_std=self.config.noise_std,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int | None = None,
    ) -> None:
        """
        Fit surrogate model.

        Args:
            X: Training features.
            y: Training targets.
            batch_size: Batch size for mini-batch training (optional).
        """
        if batch_size is not None:
            # Mini-batch training
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i : i + batch_size]
                self._partial_fit(X[batch_idx], y[batch_idx])
        else:
            self._partial_fit(X, y)

        self._fitted = True

    def _partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally fit model."""
        self.X_buffer.append(X)
        self.y_buffer.append(y)

        X_all = np.vstack(self.X_buffer)
        y_all = np.concatenate(self.y_buffer)

        self.gp.fit(X_all, y_all)

    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 10000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and uncertainty.

        Args:
            X: Query features.
            batch_size: Batch size for prediction.

        Returns:
            Tuple of (means, stds).
        """
        if len(X) <= batch_size:
            return self.gp.predict(X)

        # Batch prediction
        means = []
        stds = []

        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            m, s = self.gp.predict(batch)
            means.append(m)
            stds.append(s)

        return np.concatenate(means), np.concatenate(stds)

    def score_batch(
        self,
        X: np.ndarray,
        exploitation_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Score candidates combining mean and uncertainty.

        Args:
            X: Candidate features.
            exploitation_weight: Weight for exploitation (0=exploration, 1=exploitation).

        Returns:
            Combined scores.
        """
        means, stds = self.predict(X)

        # Score: exploitation - exploration_weight * uncertainty_bonus
        # Higher uncertainty = potential for improvement
        scores = means - (1 - exploitation_weight) * stds

        return scores

    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> None:
        """
        Update model with new observations.

        Args:
            X_new: New feature points.
            y_new: New target values.
        """
        self._partial_fit(X_new, y_new)
        logger.debug(f"Surrogate updated with {len(X_new)} new points")

    def get_best_candidates(
        self,
        X_candidates: np.ndarray,
        n_select: int = 100,
        strategy: str = "ei",
        target_value: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select best candidates using acquisition function.

        Args:
            X_candidates: Candidate features.
            n_select: Number to select.
            strategy: Selection strategy ('ei', 'mean', 'ei_robust').
            target_value: Target for EI (default: max observed).

        Returns:
            Selected indices and their scores.
        """
        if strategy == "mean":
            scores = self.predict(X_candidates, return_std=False)[0]
        else:
            # EI-style scoring
            means, stds = self.predict(X_candidates)
            if target_value is None:
                target_value = max(np.concatenate(self.y_buffer)) if self.y_buffer else 0

            # Expected Improvement
            z = (means - target_value) / (stds + 1e-10)
            ei = stds * (z * self._norm_cdf(z) + self._norm_pdf(z))
            scores = ei

        # Select top candidates
        top_indices = np.argsort(scores)[-n_select:][::-1]

        return top_indices, scores[top_indices]

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def save(self, path: str) -> None:
        """Save model state."""
        np.savez(
            path,
            X_buffer=np.vstack(self.X_buffer) if self.X_buffer else np.array([]),
            y_buffer=np.concatenate(self.y_buffer) if self.y_buffer else np.array([]),
            config_input_dim=self.config.input_dim,
            config_kernel=self.config.kernel,
            config_noise_std=self.config.noise_std,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model state."""
        data = np.load(path)
        X = data["X_buffer"]
        y = data["y_buffer"]

        if len(X) > 0:
            self.fit(X, y)

        logger.info(f"Model loaded from {path}")
