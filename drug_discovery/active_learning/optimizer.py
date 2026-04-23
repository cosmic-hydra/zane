"""
Bayesian Optimization and Resource Allocation.

Implements multi-fidelity Bayesian optimization and intelligent resource
allocation for quantum computing and molecular dynamics simulations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class OptimizationResult:
    """Result of Bayesian optimization.

    Attributes:
        best_value: Best observed value.
        best_point: Best point achieving value.
        history: History of observations.
        n_evaluations: Number of function evaluations.
        convergence_info: Information about convergence.
    """

    best_value: float = 0.0
    best_point: np.ndarray | None = None
    history_values: list[float] = field(default_factory=list)
    history_points: list[np.ndarray] = field(default_factory=list)
    n_evaluations: int = 0
    convergence_info: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_value": self.best_value,
            "best_point": self.best_point.tolist() if self.best_point is not None else None,
            "n_evaluations": self.n_evaluations,
            "convergence_info": self.convergence_info,
            "success": self.success,
        }


@dataclass
class ResourceBudget:
    """Resource budget for simulations.

    Attributes:
        qml_budget: Budget for QML calculations.
        md_budget: Budget for MD simulations.
        total_budget: Total compute budget.
        cost_per_qml: Cost per QML evaluation.
        cost_per_md: Cost per MD evaluation.
    """

    qml_budget: float = 0.0
    md_budget: float = 0.0
    total_budget: float = 0.0
    cost_per_qml: float = 10.0
    cost_per_md: float = 100.0

    def can_afford_qml(self) -> bool:
        return self.qml_budget >= self.cost_per_qml

    def can_afford_md(self) -> bool:
        return self.md_budget >= self.cost_per_md

    def spend_qml(self, amount: float = 1.0) -> bool:
        cost = self.cost_per_qml * amount
        if self.qml_budget >= cost:
            self.qml_budget -= cost
            self.total_budget -= cost
            return True
        return False

    def spend_md(self, amount: float = 1.0) -> bool:
        cost = self.cost_per_md * amount
        if self.md_budget >= cost:
            self.md_budget -= cost
            self.total_budget -= cost
            return True
        return False


class BayesianOptimizer:
    """
    Bayesian Optimizer for molecular property optimization.

    Implements efficient global optimization using GP surrogates
    and Expected Improvement acquisition.

    Example::

        optimizer = BayesianOptimizer(bounds=np.array([[0, 1], [0, 1]]))
        result = optimizer.optimize(
            f=objective_function,
            n_iterations=100,
        )
    """

    def __init__(
        self,
        bounds: np.ndarray | None = None,
        surrogate: Any = None,
        acquisition: str = "ei",
        minimize: bool = False,
        batch_size: int = 1,
        random_state: int | None = None,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            bounds: Optimization bounds (n_dims, 2).
            surrogate: Pre-configured GP surrogate.
            acquisition: Acquisition function ('ei', 'ucb', 'ts', 'pi').
            minimize: Whether minimizing objective.
            batch_size: Number of candidates per iteration.
            random_state: Random seed.
        """
        self.bounds = bounds
        self.acquisition_name = acquisition
        self.minimize = minimize
        self.batch_size = batch_size
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Setup surrogate
        if surrogate is not None:
            self.surrogate = surrogate
        else:
            from drug_discovery.active_learning.gp_surrogate import GaussianProcessSurrogate, SurrogateConfig
            input_dim = bounds.shape[0] if bounds is not None else 1
            config = SurrogateConfig(input_dim=input_dim)
            self.surrogate = GaussianProcessSurrogate(config=config)

        # Setup acquisition
        self.acquisition = self._create_acquisition(acquisition)

        # History
        self.X_observed = []
        self.y_observed = []

        logger.info(f"BayesianOptimizer initialized: acquisition={acquisition}, minimize={minimize}")

    def _create_acquisition(self, name: str):
        """Create acquisition function."""
        from drug_discovery.active_learning.acquisition import (
            ExpectedImprovement,
            UpperConfidenceBound,
            ThompsonSampling,
            ProbabilityOfImprovement,
        )

        if name == "ei":
            return ExpectedImprovement(self.surrogate, minimize=self.minimize)
        elif name == "ucb":
            return UpperConfidenceBound(self.surrogate, minimize=self.minimize)
        elif name == "ts":
            return ThompsonSampling(self.surrogate, minimize=self.minimize)
        elif name == "pi":
            return ProbabilityOfImprovement(self.surrogate, minimize=self.minimize)
        else:
            return ExpectedImprovement(self.surrogate, minimize=self.minimize)

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        n_iterations: int = 100,
        initial_samples: int = 5,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Optimize objective function.

        Args:
            f: Objective function to optimize.
            n_iterations: Number of optimization iterations.
            initial_samples: Number of initial random samples.
            verbose: Print progress.

        Returns:
            OptimizationResult.
        """
        try:
            # Initial random sampling
            for _ in range(initial_samples):
                x = self._sample_random()
                y = f(x)
                self._observe(x, y)

            # Optimization loop
            for iteration in range(n_iterations):
                # Select next candidate
                candidates = self._select_candidates()

                # Evaluate
                for x in candidates:
                    y = f(x)
                    self._observe(x, y)

                if verbose and iteration % 10 == 0:
                    best_idx = np.argmin(self.y_observed) if self.minimize else np.argmax(self.y_observed)
                    best_y = self.y_observed[best_idx]
                    logger.info(f"Iter {iteration}: best = {best_y:.4f}")

            # Return result
            best_idx = np.argmin(self.y_observed) if self.minimize else np.argmax(self.y_observed)

            return OptimizationResult(
                best_value=self.y_observed[best_idx],
                best_point=self.X_observed[best_idx],
                history_values=list(self.y_observed),
                history_points=[x.copy() for x in self.X_observed],
                n_evaluations=len(self.y_observed),
                convergence_info={"converged": True},
                success=True,
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(success=False, error=str(e))

    def _sample_random(self) -> np.ndarray:
        """Sample random point within bounds."""
        if self.bounds is None:
            return np.random.randn(10)  # Dummy dimension

        x = np.random.rand(len(self.bounds))
        x = x * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        return x

    def _select_candidates(self) -> list[np.ndarray]:
        """Select next candidates using acquisition."""
        candidates = []

        for _ in range(self.batch_size):
            # Generate random candidates
            X_cand = np.random.rand(1000, len(self.bounds))
            X_cand = X_cand * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

            # Evaluate acquisition
            if self.acquisition_name == "ts":
                # Thompson sampling
                acq_values = self.acquisition.evaluate(X_cand)
                idx = np.argmax(acq_values)
            else:
                # EI, UCB, etc.
                acq_values = self.acquisition.evaluate(X_cand)
                idx = np.argmax(acq_values)

            candidates.append(X_cand[idx])

            # Update acquisition with best so far
            self.acquisition.target_value = min(self.y_observed) if self.minimize else max(self.y_observed)

        return candidates

    def _observe(self, x: np.ndarray, y: float) -> None:
        """Record observation."""
        self.X_observed.append(x.copy())
        self.y_observed.append(y)

        # Update surrogate
        self.surrogate.update(x.reshape(1, -1), np.array([y]))


class MultiFidelityOptimizer:
    """
    Multi-Fidelity Bayesian Optimization.

    Optimizes using multiple levels of fidelity:
    - Low fidelity: Quick ML predictions (millions of candidates)
    - Medium fidelity: QML/VQE calculations (thousands of candidates)
    - High fidelity: MD/FEP simulations (hundreds of candidates)

    Example::

        optimizer = MultiFidelityOptimizer()
        optimizer.add_fidelity("ml", cost=0.001, accuracy=0.7)
        optimizer.add_fidelity("qml", cost=0.1, accuracy=0.9)
        optimizer.add_fidelity("md", cost=1.0, accuracy=1.0)
    """

    def __init__(
        self,
        budget: ResourceBudget | None = None,
        minimize: bool = False,
    ):
        """
        Initialize multi-fidelity optimizer.

        Args:
            budget: Available compute budget.
            minimize: Whether minimizing.
        """
        self.budget = budget or ResourceBudget(
            qml_budget=1000,
            md_budget=100,
            total_budget=1100,
        )
        self.minimize = minimize

        # Fidelity levels
        self.fidelities: dict[str, dict] = {}

        # Surrogate for each fidelity
        self.surrogates: dict[str, Any] = {}

        logger.info("MultiFidelityOptimizer initialized")

    def add_fidelity(
        self,
        name: str,
        predictor: Callable,
        cost: float,
        accuracy: float,
    ) -> None:
        """
        Add fidelity level.

        Args:
            name: Fidelity name.
            predictor: Prediction function.
            cost: Computational cost (normalized).
            accuracy: Expected accuracy (0-1).
        """
        self.fidelities[name] = {
            "predictor": predictor,
            "cost": cost,
            "accuracy": accuracy,
        }

        # Create surrogate for this fidelity
        from drug_discovery.active_learning.gp_surrogate import GaussianProcessSurrogate
        self.surrogates[name] = GaussianProcessSurrogate()

        logger.info(f"Added fidelity: {name}, cost={cost}, accuracy={accuracy}")

    def optimize(
        self,
        candidates: np.ndarray,
        ground_truth_fn: Callable[[np.ndarray], float],
        n_iterations: int = 50,
        selection_ratio: float = 0.01,
    ) -> dict[str, Any]:
        """
        Optimize with multi-fidelity evaluation.

        Args:
            candidates: Initial candidate pool (N, dim).
            ground_truth_fn: High-fidelity evaluation function.
            n_iterations: Number of BO iterations.
            selection_ratio: Fraction to select at each level.

        Returns:
            Optimization results with selected candidates.
        """
        # Phase 1: Screen with lowest fidelity
        if "ml" in self.fidelities:
            logger.info("Phase 1: ML screening")
            X = candidates
            scores = self._evaluate_fidelity("ml", X)

            # Select top candidates
            n_select = max(1, int(len(X) * selection_ratio))
            top_idx = np.argsort(scores)[-n_select:]
            ml_selected = X[top_idx]
        else:
            ml_selected = candidates[:int(len(candidates) * selection_ratio)]

        # Phase 2: QML evaluation (if budget allows)
        qml_selected = ml_selected
        if "qml" in self.fidelities and self.budget.can_afford_qml():
            logger.info("Phase 2: QML evaluation")
            scores = self._evaluate_fidelity("qml", ml_selected)

            n_select = max(1, int(len(ml_selected) * selection_ratio))
            top_idx = np.argsort(scores)[-n_select:]
            qml_selected = ml_selected[top_idx]

            self.budget.spend_qml(len(qml_selected))

        # Phase 3: MD/FEP (highest fidelity)
        if self.budget.can_afford_md():
            logger.info("Phase 3: MD simulation")
            final_scores = []

            for x in qml_selected:
                score = ground_truth_fn(x)
                final_scores.append(score)

            self.budget.spend_md(len(final_scores))

            # Best candidate
            best_idx = np.argmin(final_scores) if self.minimize else np.argmax(final_scores)
            best = qml_selected[best_idx]
            best_score = final_scores[best_idx]
        else:
            # Fallback to QML
            best_idx = np.argmin(scores) if self.minimize else np.argmax(scores)
            best = qml_selected[best_idx]
            best_score = scores[best_idx]

        return {
            "best_candidate": best,
            "best_score": best_score,
            "n_ml_evaluated": len(candidates),
            "n_qml_evaluated": len(qml_selected),
            "budget_remaining": self.budget.total_budget,
        }

    def _evaluate_fidelity(self, name: str, X: np.ndarray) -> np.ndarray:
        """Evaluate candidates at given fidelity."""
        if name not in self.fidelities:
            return np.zeros(len(X))

        predictor = self.fidelities[name]["predictor"]

        if callable(predictor):
            return predictor(X)

        return np.zeros(len(X))


class ResourceAllocator:
    """
    Intelligent resource allocator for compute budget.

    Decides which molecules to send to expensive simulations
    based on uncertainty and expected value of information.
    """

    def __init__(
        self,
        total_budget: float = 10000.0,
        qml_cost: float = 10.0,
        md_cost: float = 100.0,
    ):
        """
        Initialize resource allocator.

        Args:
            total_budget: Total compute budget.
            qml_cost: Cost per QML evaluation.
            md_cost: Cost per MD evaluation.
        """
        self.total_budget = total_budget
        self.qml_cost = qml_cost
        self.md_cost = md_cost

        self.qml_used = 0
        self.md_used = 0
        self.history = []

        logger.info(f"ResourceAllocator: budget={total_budget}")

    def allocate(
        self,
        candidates: np.ndarray,
        surrogate: Any,
        acquisition: Any,
        target_top_fraction: float = 0.001,
    ) -> dict[str, np.ndarray]:
        """
        Allocate resources to candidates.

        Args:
            candidates: All candidate molecules.
            surrogate: GP surrogate model.
            acquisition: Acquisition function.
            target_top_fraction: Fraction to send for expensive simulation.

        Returns:
            Dictionary with allocated candidates per level.
        """
        # Predict with surrogate
        means, stds = surrogate.predict(candidates)

        # Compute acquisition values
        acq_values = acquisition.evaluate(candidates)

        # Compute cost per candidate
        n_top_qml = max(1, int(len(candidates) * target_top_fraction * 10))
        n_top_md = max(1, int(len(candidates) * target_top_fraction))

        # Select for QML
        qml_idx = np.argsort(acq_values)[-n_top_qml:]
        qml_candidates = candidates[qml_idx]
        qml_cost = len(qml_candidates) * self.qml_cost

        # Check budget for QML
        if self.qml_used + qml_cost > self.total_budget * 0.3:
            # Reduce QML allocation
            available = (self.total_budget * 0.3 - self.qml_used) / self.qml_cost
            n_qml = max(0, int(available))
            qml_candidates = qml_candidates[:n_qml]
            qml_cost = len(qml_candidates) * self.qml_cost

        # Select for MD
        md_idx = np.argsort(acq_values)[-n_top_md:]
        md_candidates = candidates[md_idx]
        md_cost = len(md_candidates) * self.md_cost

        # Check budget for MD
        if self.md_used + md_cost > self.total_budget * 0.7:
            available = (self.total_budget * 0.7 - self.md_used) / self.md_cost
            n_md = max(0, int(available))
            md_candidates = md_candidates[:n_md]
            md_cost = len(md_candidates) * self.md_cost

        # Update budgets
        self.qml_used += qml_cost
        self.md_used += md_cost

        # Log
        self.history.append({
            "n_candidates": len(candidates),
            "n_qml": len(qml_candidates),
            "n_md": len(md_candidates),
            "qml_cost": qml_cost,
            "md_cost": md_cost,
        })

        return {
            "ml_candidates": candidates,  # All screened
            "qml_candidates": qml_candidates,
            "md_candidates": md_candidates,
        }

    def get_utilization(self) -> dict[str, float]:
        """Get resource utilization statistics."""
        return {
            "qml_utilization": self.qml_used / (self.total_budget * 0.3) if self.total_budget > 0 else 0,
            "md_utilization": self.md_used / (self.total_budget * 0.7) if self.total_budget > 0 else 0,
            "total_utilization": (self.qml_used + self.md_used) / self.total_budget,
            "budget_remaining": self.total_budget - self.qml_used - self.md_used,
        }
