"""
Acquisition Functions for Bayesian Optimization.

Implements Expected Improvement (EI) and related acquisition functions
to guide molecular selection.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.

    Acquisition functions guide the selection of candidates by balancing
    exploitation (high predicted values) and exploration (high uncertainty).
    """

    def __init__(self, surrogate, minimize: bool = False):
        """
        Initialize acquisition function.

        Args:
            surrogate: GP surrogate model.
            minimize: Whether we're minimizing (vs maximizing).
        """
        self.surrogate = surrogate
        self.minimize = minimize

    @abstractmethod
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate acquisition function.

        Args:
            X: Query points (n_samples, n_features).

        Returns:
            Acquisition values.
        """
        pass

    def sample(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Sample from acquisition function.

        Args:
            X: Query points.
            n_samples: Number of samples.

        Returns:
            Sampled acquisition values.
        """
        acq = self.evaluate(X)
        # Add noise for exploration
        noise = np.random.randn(*acq.shape) * 0.01
        return acq + noise


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.

    EI is one of the most popular acquisition functions, balancing
    exploitation and exploration by computing the expected improvement
    over the current best observation.

    For minimization problems:
        EI(x) = E[max(f_best - f(x), 0)]

    For maximization problems:
        EI(x) = E[max(f(x) - f_best, 0)]

    References:
        - Jones et al., "Efficient Global Optimization of Expensive Black-Box Functions"
    """

    def __init__(
        self,
        surrogate,
        target_value: float | None = None,
        minimize: bool = False,
        xi: float = 0.01,
    ):
        """
        Initialize EI.

        Args:
            surrogate: GP surrogate model.
            target_value: Target value (default: best observed).
            minimize: Whether minimizing.
            xi: Exploration parameter (higher = more exploration).
        """
        super().__init__(surrogate, minimize)
        self.target_value = target_value
        self.xi = xi

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate Expected Improvement.

        Args:
            X: Query points (n_samples, n_features).

        Returns:
            EI values.
        """
        # Get predictions
        if hasattr(self.surrogate, 'predict'):
            means, stds = self.surrogate.predict(X)
        else:
            means = self.surrogate(X)
            stds = np.ones_like(means) * 0.1

        # Handle single predictions
        if means.ndim == 1:
            means = means.reshape(-1, 1)
            stds = stds.reshape(-1, 1)

        # Target value
        if self.target_value is None:
            # Use best observed
            if hasattr(self.surrogate, 'y_buffer') and self.surrogate.y_buffer:
                y_obs = np.concatenate(self.surrogate.y_buffer)
                if self.minimize:
                    target = y_obs.min()
                else:
                    target = y_obs.max()
            else:
                target = means.mean()
        else:
            target = self.target_value

        # EI computation
        if self.minimize:
            diff = target - means
        else:
            diff = means - target

        # Standard normal PDF and CDF
        z = diff / (stds + 1e-10)
        pdf = self._norm_pdf(z)
        cdf = self._norm_cdf(z)

        # EI formula
        ei = diff * cdf + stds * pdf

        # Apply xi (exploration bonus)
        ei = ei - self.xi * np.abs(diff)

        # Ensure non-negative
        ei = np.maximum(ei, 0)

        # Handle NaN/Inf
        ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

        return ei.flatten()

    def evaluate_analytical(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate EI analytically (vectorized).

        Args:
            X: Query points.

        Returns:
            EI values.
        """
        means, stds = self.surrogate.predict(X)

        if self.minimize:
            best = means.min() if self.target_value is None else self.target_value
            diff = best - means
        else:
            best = means.max() if self.target_value is None else self.target_value
            diff = means - best

        z = diff / (stds + 1e-10)

        # EI = (best - f) * Phi(z) + sigma * phi(z)
        ei = diff * self._norm_cdf(z) + stds * self._norm_pdf(z)

        return np.nan_to_num(ei)

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function.

    UCB balances exploitation and exploration through a linear
    combination of predicted mean and uncertainty:

        UCB(x) = mu(x) + kappa * sigma(x)

    Higher kappa = more exploration.

    References:
        - Srinivas et al., "Gaussian Process Optimization in the Bandit Setting"
    """

    def __init__(
        self,
        surrogate,
        kappa: float = 2.0,
        minimize: bool = False,
    ):
        """
        Initialize UCB.

        Args:
            surrogate: GP surrogate model.
            kappa: Exploration parameter.
            minimize: Whether minimizing.
        """
        super().__init__(surrogate, minimize)
        self.kappa = kappa

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate UCB.

        Args:
            X: Query points.

        Returns:
            UCB values.
        """
        means, stds = self.surrogate.predict(X)

        if self.minimize:
            # Lower bound
            ucb = means - self.kappa * stds
        else:
            # Upper bound
            ucb = means + self.kappa * stds

        return ucb


class ThompsonSampling(AcquisitionFunction):
    """
    Thompson Sampling acquisition function.

    Samples from the posterior distribution and selects
    the point with highest sampled value.

    References:
        - Thompson, "On the Likelihood that One Unknown Probability Exceeds Another"
    """

    def __init__(
        self,
        surrogate,
        minimize: bool = False,
        n_samples: int = 1,
    ):
        """
        Initialize Thompson Sampling.

        Args:
            surrogate: GP surrogate model.
            minimize: Whether minimizing.
            n_samples: Number of posterior samples.
        """
        super().__init__(surrogate, minimize)
        self.n_samples = n_samples

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate Thompson Sampling.

        Args:
            X: Query points.

        Returns:
            Sampled acquisition values.
        """
        means, stds = self.surrogate.predict(X)

        # Sample from posterior
        samples = means + stds * np.random.randn(*means.shape)

        if self.minimize:
            return -samples
        return samples

    def select(self, X: np.ndarray) -> int:
        """
        Select single best point via Thompson Sampling.

        Args:
            X: Query points.

        Returns:
            Index of selected point.
        """
        samples = self.evaluate(X)
        return np.argmax(samples)


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function.

    Computes the probability that a point improves over the best observed:

        PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))

    References:
        - Kushner, "A New Method of Locating the Maximum of an Arbitrary Multivariate Curve"
    """

    def __init__(
        self,
        surrogate,
        target_value: float | None = None,
        minimize: bool = False,
        epsilon: float = 0.0,
    ):
        """
        Initialize PI.

        Args:
            surrogate: GP surrogate model.
            target_value: Target value.
            minimize: Whether minimizing.
            epsilon: Minimum improvement threshold.
        """
        super().__init__(surrogate, minimize)
        self.target_value = target_value
        self.epsilon = epsilon

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate Probability of Improvement.

        Args:
            X: Query points.

        Returns:
            PI values.
        """
        means, stds = self.surrogate.predict(X)

        if self.target_value is None:
            if hasattr(self.surrogate, 'y_buffer') and self.surrogate.y_buffer:
                y_obs = np.concatenate(self.surrogate.y_buffer)
                target = y_obs.min() if self.minimize else y_obs.max()
            else:
                target = means.mean()
        else:
            target = self.target_value

        if self.minimize:
            z = (target - self.epsilon - means) / (stds + 1e-10)
        else:
            z = (means - target + self.epsilon) / (stds + 1e-10)

        pi = self._norm_cdf(z)

        return pi

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))


class KnowledgeGradient(AcquisitionFunction):
    """
    Knowledge Gradient (KG) acquisition function.

    Computes the expected value of information gained by sampling a point.

    References:
        - Frazier et al., "The Knowledge Gradient Policy for Correlated Normal Rewards"
    """

    def __init__(
        self,
        surrogate,
        minimize: bool = False,
        n_discrete: int = 50,
    ):
        """
        Initialize Knowledge Gradient.

        Args:
            surrogate: GP surrogate model.
            minimize: Whether minimizing.
            n_discrete: Number of discretization points.
        """
        super().__init__(surrogate, minimize)
        self.n_discrete = n_discrete

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate Knowledge Gradient (approximate).

        Args:
            X: Query points.

        Returns:
            KG values.
        """
        means, stds = self.surrogate.predict(X)

        # Discretize the outcome space
        if hasattr(self.surrogate, 'y_buffer') and self.surrogate.y_buffer:
            y_obs = np.concatenate(self.surrogate.y_buffer)
            y_min, y_max = y_obs.min(), y_obs.max()
        else:
            y_min, y_max = means.min(), means.max()

        y_range = y_max - y_min + 1e-6
        y_grid = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, self.n_discrete)

        # Compute expected improvement over discretized outcomes
        kg = np.zeros(len(X))

        for i, (mu, sigma) in enumerate(zip(means, stds)):
            if sigma < 1e-10:
                continue

            # P(y | x)
            probs = self._norm_pdf((y_grid - mu) / sigma)
            probs = probs / (probs.sum() + 1e-10)

            # Best achievable with this sample
            if self.minimize:
                best_future = np.minimum(y_grid, mu)
            else:
                best_future = np.maximum(y_grid, mu)

            # Expected value
            expected_best = np.dot(probs, best_future)

            # Current best
            current_best = mu

            # Knowledge gradient
            kg[i] = expected_best - current_best

        return kg

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
