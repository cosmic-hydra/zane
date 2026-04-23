"""
Target-Aware 3D Molecular Diffusion Module.

Implements a generative AI model that designs drugs de novo based on physics
derived from previous modules, using structure-based equivariant diffusion models.

Features:
- Structure-based equivariant diffusion model
- Pocket-aware molecular generation
- Binding affinity optimization
- Synthetic accessibility scoring

Tech Stack: Diffusers, PyTorch3D, RDKit
"""

from __future__ import annotations

from drug_discovery.diffusion.diffusion_model import (
    EquivariantDiffusionModel,
    DiffusionConfig,
    DiffusionResult,
)
from drug_discovery.diffusion.pocket_generator import (
    PocketAwareGenerator,
    PocketContext,
    GeneratedMolecule,
)

__all__ = [
    "EquivariantDiffusionModel",
    "DiffusionConfig",
    "DiffusionResult",
    "PocketAwareGenerator",
    "PocketContext",
    "GeneratedMolecule",
]
