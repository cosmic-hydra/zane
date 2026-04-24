"""
Active Learning Optimizer for Molecular Design.

Coordinates the selection of molecules for high-fidelity simulation using
Bayesian optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from drug_discovery.active_learning.acquisition import ExpectedImprovement
from drug_discovery.active_learning.gp_surrogate import GaussianProcessSurrogate

logger = logging.getLogger(__name__)


@dataclass
class ResourceBudget:
    """Computational budget for active learning."""

    max_simulations: int = 100
    max_qml_runs: int = 50
    max_md_steps: int = 1000000
    max_wall_time_hours: float = 24.0


@dataclass
class OptimizationResult:
    """Results from active learning optimization."""

    best_smiles: str
    best_value: float
    all_smiles: list[str]
    all_values: list[float]
    history: dict[str, list[float]]


class BayesianOptimizer:
    """
    Bayesian Optimizer for molecule selection.

    Uses a GP surrogate and acquisition functions to select molecules.
    """

    def __init__(
        self,
        surrogate: GaussianProcessSurrogate | None = None,
        acquisition: str = "ei",
        batch_size: int = 10,
        bounds: np.ndarray | None = None,
    ):
        """
        Initialize optimizer.

        Args:
            surrogate: GP surrogate model.
            acquisition: Acquisition function type.
            batch_size: Number of molecules to select in each round.
            bounds: Parameter bounds for optimization.
        """
        self.surrogate = surrogate or GaussianProcessSurrogate()
        self.acquisition_type = acquisition
        self.batch_size = batch_size
        self.bounds = bounds or np.array([[0, 1]] * 128)

        if acquisition == "ei":
            self.acquisition = ExpectedImprovement()
        else:
            self.acquisition = ExpectedImprovement()

        logger.info(f"BayesianOptimizer initialized with {acquisition} acquisition")

    def suggest(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Suggest next batch of candidates.

        Args:
            X_candidates: Candidate features.

        Returns:
            Indices of suggested candidates.
        """
        indices, _ = self.surrogate.get_best_candidates(
            X_candidates,
            n_select=self.batch_size,
            strategy=self.acquisition_type,
        )
        return indices

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Update surrogate with new observations.

        Args:
            X: Observed features.
            y: Observed target values.
        """
        self.surrogate.update(X, y)


class MultiFidelityOptimizer:
    """
    Multi-fidelity optimizer for drug discovery.

    Orchestrates scoring across multiple levels of fidelity:
    1. Fast ML surrogate (low cost, millions of molecules)
    2. Medium-fidelity QML (medium cost, thousands of molecules)
    3. High-fidelity MD/FEP (high cost, tens of molecules)
    """

    def __init__(
        self,
        fidelities: list[str] | None = None,
        budget: ResourceBudget | None = None,
    ):
        """
        Initialize multi-fidelity optimizer.

        Args:
            fidelities: List of fidelity levels to use.
            budget: Computational budget.
        """
        self.fidelities = fidelities or ["ml", "qml", "md"]
        self.budget = budget or ResourceBudget()
        self.surrogates = {f: GaussianProcessSurrogate() for f in self.fidelities}

        logger.info(f"MultiFidelityOptimizer initialized with fidelities: {self.fidelities}")

    def run_optimization_cycle(
        self,
        candidates: np.ndarray,
        selection_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Run one cycle of multi-fidelity optimization.

        Args:
            candidates: Initial pool of candidates.
            selection_ratio: Fraction to keep at each stage.

        Returns:
            Top candidates selected for high-fidelity evaluation.
        """
        current_pool = candidates

        if "ml" in self.fidelities:
            logger.info("Phase 1: ML screening")
            x_screen = current_pool
            scores = self._evaluate_fidelity("ml", x_screen)

            # Select top candidates
            n_select = max(1, int(len(x_screen) * selection_ratio))
            top_idx = np.argsort(scores)[-n_select:]
            current_pool = x_screen[top_idx]

        if "qml" in self.fidelities:
            logger.info("Phase 2: QML refinement")
            x_refine = current_pool
            scores = self._evaluate_fidelity("qml", x_refine)

            n_select = max(1, int(len(x_refine) * selection_ratio))
            top_idx = np.argsort(scores)[-n_select:]
            current_pool = x_refine[top_idx]

        return current_pool

    def _evaluate_fidelity(self, fidelity: str, X: np.ndarray) -> np.ndarray:
        """Evaluate candidates at a specific fidelity level."""
        # This would call the actual simulation modules in a real implementation
        # For now, we return surrogate scores
        return self.surrogates[fidelity].score_batch(X)


class ResourceAllocator:
    """
    Allocates compute resources based on value-of-information.
    """

    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self.consumed = {
            "simulations": 0,
            "qml_runs": 0,
            "md_steps": 0,
            "wall_time": 0.0,
        }

    def can_allocate(self, task_type: str, amount: float = 1.0) -> bool:
        """Check if resource can be allocated."""
        if task_type == "simulation":
            return self.consumed["simulations"] + amount <= self.budget.max_simulations
        elif task_type == "qml":
            return self.consumed["qml_runs"] + amount <= self.budget.max_qml_runs
        elif task_type == "md":
            return self.consumed["md_steps"] + amount <= self.budget.max_md_steps
        return True

    def consume(self, task_type: str, amount: float = 1.0) -> None:
        """Update consumed resources."""
        if task_type == "simulation":
            self.consumed["simulations"] += int(amount)
        elif task_type == "qml":
            self.consumed["qml_runs"] += int(amount)
        elif task_type == "md":
            self.consumed["md_steps"] += int(amount)
