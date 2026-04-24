"""
Quantum Machine Learning (QML) Engine for Drug Discovery.

A noise-resilient hybrid quantum-classical module for chemistry calculations.

Features:
- Active Space Approximation for molecular orbital selection
- Variational Quantum Eigensolver (VQE) with Hardware-Efficient Ansatz
- Error Mitigation via Zero-Noise Extrapolation (ZNE)

Tech Stack: PennyLane, PyTorch, Qiskit Nature, OpenFermion
"""

from __future__ import annotations

from drug_discovery.qml.error_mitigation import (
    ErrorMitigationConfig,
    ZeroNoiseExtrapolation,
    ZNEResult,
)
from drug_discovery.qml.quantum_chemistry import (
    ActiveSpaceApproximator,
    ActiveSpaceResult,
    MolecularOrbitals,
)
from drug_discovery.qml.quantum_driver import (
    AWSBraketDriver,
    LocalSimulator,
    QuantumDriver,
    QuantumSimulator,
)
from drug_discovery.qml.vqe import (
    HardwareEfficientAnsatz,
    VQECircuit,
    VQEResult,
)

__all__ = [
    # Active Space
    "ActiveSpaceApproximator",
    "ActiveSpaceResult",
    "MolecularOrbitals",
    # VQE
    "VQECircuit",
    "VQEResult",
    "HardwareEfficientAnsatz",
    # Error Mitigation
    "ZeroNoiseExtrapolation",
    "ZNEResult",
    "ErrorMitigationConfig",
    # Drivers
    "QuantumDriver",
    "QuantumSimulator",
    "AWSBraketDriver",
    "LocalSimulator",
]
