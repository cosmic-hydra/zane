"""
Research-grade autonomous discovery stack.

This package hosts learnable docking, differentiable binding, adaptive compute
policies, and other advanced modules that can be imported selectively without
affecting existing lightweight workflows.
"""

from .autonomous_stack import (
    AdaptiveComputeAllocator,
    BindingPipelineOutput,
    CausalPropertyModel,
    DifferentiableBindingPipeline,
    FailureAwareTrainer,
    HybridSymbolicNeuralEngine,
    LearnableDockingEngine,
    MemoryAugmentedSearch,
    MetaLearnerMAML,
    MultiFidelityRegressor,
    NeuralConstraintProjector,
    NeuralDockingModel,
    QuantumCorrectionNetwork,
    ReactionConditionedBackend,
    StructuralUncertaintyHead,
    TrajectoryStabilityModel,
    WorkflowBenchmarkHarness,
)

__all__ = [
    "AdaptiveComputeAllocator",
    "BindingPipelineOutput",
    "CausalPropertyModel",
    "DifferentiableBindingPipeline",
    "FailureAwareTrainer",
    "HybridSymbolicNeuralEngine",
    "LearnableDockingEngine",
    "MemoryAugmentedSearch",
    "MetaLearnerMAML",
    "MultiFidelityRegressor",
    "NeuralConstraintProjector",
    "NeuralDockingModel",
    "QuantumCorrectionNetwork",
    "ReactionConditionedBackend",
    "StructuralUncertaintyHead",
    "TrajectoryStabilityModel",
    "WorkflowBenchmarkHarness",
]
