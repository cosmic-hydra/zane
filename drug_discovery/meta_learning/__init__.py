"""Meta-Learning Module — Recursive Self-Improvement and AutoML."""

from .self_improvement import (
    CodeMutator,
    HypothesisGenerator,
    SelfImprovementOrchestrator,
)

__all__ = ["CodeMutator", "HypothesisGenerator", "SelfImprovementOrchestrator"]
