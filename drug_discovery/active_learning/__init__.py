"""
Active Learning Brain & Bayesian Optimization Module.

Builds the overarching intelligence that routes compute resources for drug discovery.
Uses Gaussian Process surrogate models and acquisition functions to efficiently
select molecules for expensive high-fidelity simulations.

Features:
- Gaussian Process surrogate model
- Expected Improvement (EI) acquisition
- Multi-fidelity Bayesian optimization
- Resource allocation for QML and MD modules

Tech Stack: BoTorch, GPyTorch
"""

from __future__ import annotations

from drug_discovery.active_learning.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    ThompsonSampling,
    UpperConfidenceBound,
)
from drug_discovery.active_learning.gp_surrogate import (
    GaussianProcessSurrogate,
    SurrogateConfig,
)
from drug_discovery.active_learning.optimizer import (
    BayesianOptimizer,
    MultiFidelityOptimizer,
    OptimizationResult,
    ResourceAllocator,
    ResourceBudget,
)

__all__ = [
    "GaussianProcessSurrogate",
    "SurrogateConfig",
    "AcquisitionFunction",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ThompsonSampling",
    "BayesianOptimizer",
    "MultiFidelityOptimizer",
    "ResourceAllocator",
    "OptimizationResult",
    "ResourceBudget",
]
