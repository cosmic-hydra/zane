"""
4D Geometric Deep Learning & Free Energy Perturbation Module.

Bridges static protein predictions with dynamic, real-world physics using
SE(3)-equivariant transformers and OpenMM molecular dynamics simulations.

Features:
- SE(3)-Equivariant Transformer for conformational dynamics
- Physics Validation Engine with FEP binding free energy calculations

Tech Stack: e3nn (Equivariant Neural Networks), PyTorch, OpenMM
"""

from __future__ import annotations

from drug_discovery.geometric_dl.se3_transformer import (
    SE3Transformer,
    SE3EquivariantBlock,
    EquivariantAttention,
    SE3Config,
)
from drug_discovery.geometric_dl.fep_engine import (
    BindingFreeEnergyCalculator,
    FEPConfig,
    FEPResult,
    OpenMMDriver,
)
from drug_discovery.geometric_dl.pocket_predictor import (
    TransientPocketPredictor,
    PocketPrediction,
)

__all__ = [
    "SE3Transformer",
    "SE3EquivariantBlock",
    "EquivariantAttention",
    "BindingFreeEnergyCalculator",
    "FEPConfig",
    "FEPResult",
    "OpenMMDriver",
    "TransientPocketPredictor",
    "PocketPrediction",
]
