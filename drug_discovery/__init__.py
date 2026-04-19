"""
Drug Discovery AI Platform
A state-of-the-art autonomous AI system for drug discovery
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"
__author__ = "AI Drug Discovery Team"

__all__ = [
    "DrugDiscoveryPipeline",
    "MolecularGNN",
    "MolecularTransformer",
    "DrugModeler",
    "LlamaSupportAssistant",
    "BoltzGenRunner",
    "GenerationManager",
    "ReinventBackend",
    "GT4SDBackend",
    "MolformerBackend",
    "MolecularDesignBackend",
    "BenchmarkRunner",
    "MosesBenchmarkBackend",
    "GuacamolBenchmarkBackend",
    "EliteStackPipeline",
    "ProgramStrategyEngine",
    "PhysicsAwareGenerator",
]


def __getattr__(name: str) -> Any:
    """Lazy attribute loader to keep CLI/dashboard startup lightweight."""
    if name == "DrugDiscoveryPipeline":
        from .pipeline import DrugDiscoveryPipeline

        return DrugDiscoveryPipeline
    if name in {"DrugModeler", "MolecularGNN", "MolecularTransformer"}:
        from .models import DrugModeler, MolecularGNN, MolecularTransformer

        return {
            "DrugModeler": DrugModeler,
            "MolecularGNN": MolecularGNN,
            "MolecularTransformer": MolecularTransformer,
        }[name]
    if name == "LlamaSupportAssistant":
        from .ai_support import LlamaSupportAssistant

        return LlamaSupportAssistant
    if name == "BoltzGenRunner":
        from .boltzgen_adapter import BoltzGenRunner

        return BoltzGenRunner
    if name in {"GenerationManager", "ReinventBackend", "GT4SDBackend", "MolformerBackend", "MolecularDesignBackend"}:
        from .generation.backends import (
            GT4SDBackend,
            GenerationManager,
            MolecularDesignBackend,
            MolformerBackend,
            ReinventBackend,
        )

        return {
            "GenerationManager": GenerationManager,
            "ReinventBackend": ReinventBackend,
            "GT4SDBackend": GT4SDBackend,
            "MolformerBackend": MolformerBackend,
            "MolecularDesignBackend": MolecularDesignBackend,
        }[name]
    if name == "PhysicsAwareGenerator":
        from .generation.physics_aware import PhysicsAwareGenerator

        return PhysicsAwareGenerator
    if name in {"BenchmarkRunner", "MosesBenchmarkBackend", "GuacamolBenchmarkBackend"}:
        from .benchmarking.backends import BenchmarkRunner, GuacamolBenchmarkBackend, MosesBenchmarkBackend

        return {
            "BenchmarkRunner": BenchmarkRunner,
            "MosesBenchmarkBackend": MosesBenchmarkBackend,
            "GuacamolBenchmarkBackend": GuacamolBenchmarkBackend,
        }[name]
    if name == "EliteStackPipeline":
        from .elite_stack import EliteStackPipeline

        return EliteStackPipeline
    if name == "ProgramStrategyEngine":
        from .strategy import ProgramStrategyEngine

        return ProgramStrategyEngine
    raise AttributeError(f"module 'drug_discovery' has no attribute {name!r}")
