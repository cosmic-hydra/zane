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
    raise AttributeError(f"module 'drug_discovery' has no attribute {name!r}")
