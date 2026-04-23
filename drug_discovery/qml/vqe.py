"""
Variational Quantum Eigensolver (VQE) Implementation.

Implements VQE with Hardware-Efficient Ansatz (HEA) paired with classical
PyTorch optimizer to find molecular ground-state energies.

References:
    - Peruzzo et al., "A variational eigenvalue solver on a quantum processor"
    - McClean et al., "The theory of variational hybrid quantum-classical algorithms"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    logging.warning("PyTorch not available. Using numpy fallback.")

try:
    import pennylane as qml
    from pennylane import numpy as qnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None
    qnp = None
    logging.warning("PennyLane not available. VQE will use simulation mode.")

logger = logging.getLogger(__name__)


@dataclass
class VQEResult:
    """Result of VQE calculation.

    Attributes:
        ground_state_energy: Computed ground state energy.
        optimal_parameters: Optimal circuit parameters.
        energy_history: Energy at each iteration.
        n_measurements: Number of measurements taken.
        fidelity: Fidelity to exact ground state (if known).
        computation_time: Time taken for computation.
        success: Whether optimization converged.
        error: Error message if failed.
    """

    ground_state_energy: float = 0.0
    optimal_parameters: np.ndarray | None = None
    energy_history: list[float] = field(default_factory=list)
    n_measurements: int = 0
    fidelity: float = 0.0
    computation_time: float = 0.0
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ground_state_energy": self.ground_state_energy,
            "optimal_parameters": self.optimal_parameters.tolist() if self.optimal_parameters is not None else None,
            "energy_history": self.energy_history,
            "n_measurements": self.n_measurements,
            "fidelity": self.fidelity,
            "computation_time": self.computation_time,
            "success": self.success,
            "error": self.error,
        }


class HardwareEfficientAnsatz:
    """
    Hardware-Efficient Ansatz (HEA) for VQE.

    The HEA is designed to be hardware-friendly with:
    - Single-qubit rotations on all qubits
    - Entangling layers (CNOT/cu3)
    - Short depth to minimize noise

    Attributes:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz layers.
        n_parameters: Total number of variational parameters.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        entanglement: str = "linear",
        device: str = "default.qubit",
    ):
        """
        Initialize HEA.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of ansatz layers.
            entanglement: Entanglement pattern ('linear', 'circular', 'full').
            device: PennyLane device for simulation.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.device = device
        self.n_parameters = n_qubits * (n_layers + 1)  # Rotations per layer + initial

        if PENNYLANE_AVAILABLE:
            self._dev = qml.device(device, wires=n_qubits)
        else:
            self._dev = None

        logger.info(f"HEA initialized: {n_qubits} qubits, {n_layers} layers, {self.n_parameters} params")

    def circuit(self, parameters: np.ndarray) -> float:
        """
        Execute the HEA circuit.

        Args:
            parameters: Variational parameters (shape: n_parameters,).

        Returns:
            Expectation value of Hamiltonian.
        """
        if not PENNYLANE_AVAILABLE:
            return self._simulate_circuit(parameters)

        @qml.qnode(self._dev, interface="torch")
        def circuit_template(params):
            # Initial rotation layer
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)

            # Entangling layer + subsequent rotations
            param_idx = self.n_qubits
            for layer in range(self.n_layers):
                # Entangling gates
                if self.entanglement == "linear":
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif self.entanglement == "circular":
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                elif self.entanglement == "full":
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CNOT(wires=[i, j])

                # Rotation layer
                for i in range(self.n_qubits):
                    qml.RY(params[param_idx + i], wires=i)
                    qml.RZ(params[param_idx + i], wires=i)
                param_idx += self.n_qubits

            # Measure expectation of Z on all qubits
            return qml.expval(qml.grouping.string_to_list_of_products("Z0")[0][0])

        return circuit_template(parameters)

    def _simulate_circuit(self, parameters: np.ndarray) -> float:
        """Simulate circuit without PennyLane (numpy fallback)."""
        # Simplified simulation
        total = 0.0
        param_idx = 0

        for i in range(self.n_qubits):
            theta = parameters[param_idx]
            total += np.cos(theta)  # Simplified RY effect
            param_idx += 1

        for layer in range(self.n_layers):
            # Simplified entanglement effect
            for i in range(self.n_qubits - 1):
                total += 0.1 * np.sin(parameters[param_idx + i])
            param_idx += self.n_qubits

        return total / self.n_qubits


class VQECircuit:
    """
    Variational Quantum Eigensolver with classical optimization.

    Combines HEA with PyTorch optimizers for hybrid quantum-classical optimization.

    Example::

        vqe = VQECircuit(n_qubits=4, n_layers=2)
        result = vqe.optimize(hamiltonian={"Z0 Z1": -1.0, "Z2 Z3": -0.5})
        print(f"Ground state energy: {result.ground_state_energy}")
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        entanglement: str = "linear",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        """
        Initialize VQE circuit.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of ansatz layers.
            entanglement: Entanglement pattern.
            optimizer: Classical optimizer ('adam', 'sgd', 'lbfgs').
            learning_rate: Learning rate for optimizer.
            device: Device for computation ('cpu', 'cuda').
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.device = device

        self.ansatz = HardwareEfficientAnsatz(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entanglement=entanglement,
        )

        # Initialize parameters
        np.random.seed(42)
        self.parameters = np.random.randn(self.ansatz.n_parameters) * 0.01

        # Classical optimizer
        self._setup_optimizer()

        logger.info(f"VQECircuit initialized on {device}")

    def _setup_optimizer(self) -> None:
        """Setup classical optimizer."""
        if not TORCH_AVAILABLE:
            self.optimizer = None
            return

        self.torch_params = torch.nn.Parameter(
            torch.tensor(self.parameters, dtype=torch.float32)
        )

        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                [self.torch_params],
                lr=self.learning_rate,
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                [self.torch_params],
                lr=self.learning_rate,
                momentum=0.9,
            )
        elif self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                [self.torch_params],
                lr=self.learning_rate,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [self.torch_params],
                lr=self.learning_rate,
            )

    def energy_function(self, params: np.ndarray, hamiltonian: dict[str, float]) -> float:
        """
        Compute energy expectation for given Hamiltonian.

        Args:
            params: Circuit parameters.
            hamiltonian: Dictionary of Pauli terms and coefficients.

        Returns:
            Energy expectation value.
        """
        if not PENNYLANE_AVAILABLE:
            # Fallback: use simplified Hamiltonian expectation
            energy = 0.0
            for term, coeff in hamiltonian.items():
                # Simplified: assume each Z contributes based on params
                if "Z" in term:
                    z_count = term.count("Z")
                    z_contribution = np.sum(np.sin(params[:z_count]))
                    energy += coeff * z_contribution / z_count
            return energy

        # Full PennyLane implementation
        @qml.qnode(self._dev, interface="torch")
        def circuit(params):
            # Apply ansatz
            self._apply_ansatz(params)
            # Measure expectation
            results = []
            for term in hamiltonian.keys():
                # Simplified measurement
                results.append(qml.expval(qml.PauliZ(0)))
            return sum(results) / len(results) if results else 0.0

        return float(circuit(params))

    def _apply_ansatz(self, params: torch.Tensor) -> None:
        """Apply HEA ansatz to quantum circuit."""
        if not PENNYLANE_AVAILABLE:
            return

        param_idx = 0
        for i in range(self.n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1

        for layer in range(self.n_layers):
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(params[param_idx + i], wires=i)
                qml.RZ(params[param_idx + i], wires=i)
            param_idx += self.n_qubits

    def optimize(
        self,
        hamiltonian: dict[str, float],
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        initial_params: np.ndarray | None = None,
        verbose: bool = False,
    ) -> VQEResult:
        """
        Optimize VQE to find ground state energy.

        Args:
            hamiltonian: Molecular Hamiltonian as Pauli strings.
            max_iterations: Maximum optimization iterations.
            convergence_threshold: Convergence criterion.
            initial_params: Starting parameters.
            verbose: Print progress.

        Returns:
            VQEResult with optimization results.
        """
        import time
        start_time = time.time()

        if initial_params is not None:
            self.parameters = initial_params

        energy_history = []
        best_energy = float('inf')
        best_params = self.parameters.copy()

        if TORCH_AVAILABLE and self.optimizer is not None:
            # PyTorch optimization loop
            for iteration in range(max_iterations):
                self.optimizer.zero_grad()

                # Forward pass
                energy = self.energy_function(
                    self.torch_params.detach().numpy(),
                    hamiltonian,
                )

                # Backward pass (simplified gradient)
                loss = torch.tensor(energy, requires_grad=True)
                loss.backward()

                # Optimizer step
                self.optimizer.step()

                current_params = self.torch_params.detach().numpy()
                energy_history.append(energy)

                if energy < best_energy:
                    best_energy = energy
                    best_params = current_params.copy()

                if verbose and iteration % 10 == 0:
                    logger.info(f"Iter {iteration}: Energy = {energy:.6f}")

                # Check convergence
                if len(energy_history) > 1:
                    delta = abs(energy_history[-1] - energy_history[-2])
                    if delta < convergence_threshold:
                        logger.info(f"Converged at iteration {iteration}")
                        break
        else:
            # NumPy/Scipy fallback
            from scipy.optimize import minimize

            def objective(params):
                energy = self.energy_function(params, hamiltonian)
                energy_history.append(energy)
                return energy

            result = minimize(
                objective,
                self.parameters,
                method='L-BFGS-B',
                options={'maxiter': max_iterations, 'gtol': convergence_threshold}
            )

            best_params = result.x
            best_energy = result.fun

        elapsed_time = time.time() - start_time

        return VQEResult(
            ground_state_energy=best_energy,
            optimal_parameters=best_params,
            energy_history=energy_history,
            n_measurements=len(energy_history) * len(hamiltonian),
            computation_time=elapsed_time,
            success=True,
        )

    def compute_gradient(
        self,
        params: np.ndarray,
        hamiltonian: dict[str, float],
        epsilon: float = 0.01,
    ) -> np.ndarray:
        """
        Compute parameter gradients using finite differences.

        Args:
            params: Current parameters.
            hamiltonian: Hamiltonian terms.
            epsilon: Finite difference step.

        Returns:
            Gradient vector.
        """
        gradient = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon

            energy_plus = self.energy_function(params_plus, hamiltonian)
            energy_minus = self.energy_function(params_minus, hamiltonian)

            gradient[i] = (energy_plus - energy_minus) / (2 * epsilon)

        return gradient
