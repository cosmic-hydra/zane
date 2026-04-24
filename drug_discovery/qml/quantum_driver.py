"""
Quantum Driver - Abstract Interface for Quantum Simulators and Hardware.

Provides a unified interface for quantum computing backends that can be
easily swapped for enterprise cloud counterparts (e.g., AWS Braket, IBM Quantum).

This abstraction allows the QML engine to run on:
- Local simulators (PennyLane default)
- AWS Braket (cloud hardware)
- IBM Quantum (cloud hardware)
- Custom simulators
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantumResult:
    """Result from a quantum computation.

    Attributes:
        expectation_values: Measurement expectation values.
        state_vector: Final state vector (if available).
        probabilities: Measurement probabilities.
        n_qubits: Number of qubits used.
        n_shots: Number of measurement shots.
        execution_time: Time taken for execution.
        backend: Backend name used.
    """

    expectation_values: dict[str, float] = field(default_factory=dict)
    state_vector: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    n_qubits: int = 0
    n_shots: int = 1000
    execution_time: float = 0.0
    backend: str = "unknown"
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "expectation_values": self.expectation_values,
            "n_qubits": self.n_qubits,
            "n_shots": self.n_shots,
            "execution_time": self.execution_time,
            "backend": self.backend,
            "success": self.success,
            "error": self.error,
        }


class QuantumSimulator(ABC):
    """
    Abstract base class for quantum simulators.

    Implement this interface to add support for new quantum backends.
    """

    @abstractmethod
    def initialize_circuit(self, n_qubits: int) -> Any:
        """Initialize a quantum circuit with n qubits."""
        pass

    @abstractmethod
    def add_gate(
        self,
        gate_name: str,
        qubits: list[int],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Add a gate to the circuit."""
        pass

    @abstractmethod
    def measure(self, observable: str) -> QuantumResult:
        """Execute circuit and measure observable."""
        pass

    @abstractmethod
    def get_state_vector(self) -> np.ndarray:
        """Get the final state vector."""
        pass

    def run_circuit(self, circuit: Any) -> QuantumResult:
        """Execute a complete circuit."""
        return self.measure("Z" * self.n_qubits)


class LocalSimulator(QuantumSimulator):
    """
    Local quantum simulator using NumPy/SciPy.

    Supports:
    - State vector simulation (exact)
    - Shot-based simulation (statistical)
    - Common gates (H, X, Y, Z, CNOT, RX, RY, RZ, etc.)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        precision: str = "float64",
        use_gpu: bool = False,
    ):
        """
        Initialize local simulator.

        Args:
            n_qubits: Maximum number of qubits.
            precision: Numerical precision ('float32', 'float64').
            use_gpu: Whether to attempt GPU acceleration.
        """
        self.n_qubits = n_qubits
        self.precision = precision
        self.use_gpu = use_gpu
        self.circuit = []
        self._state = None

        logger.info(f"LocalSimulator initialized: {n_qubits} qubits, {precision}")

    def initialize_circuit(self, n_qubits: int) -> None:
        """Initialize state vector to |0>^n."""
        self.n_qubits = n_qubits
        self.circuit = []

        # Initial state |0...0>
        self._state = np.zeros(2**n_qubits, dtype=complex)
        self._state[0] = 1.0 + 0j

    def add_gate(
        self,
        gate_name: str,
        qubits: list[int],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Add a gate to the circuit."""
        self.circuit.append(
            {
                "gate": gate_name,
                "qubits": qubits,
                "parameters": parameters or {},
            }
        )

    def _apply_single_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply single-qubit gate to state."""
        n = 2**self.n_qubits
        new_state = np.zeros(n, dtype=complex)

        for i in range(n):
            # Check qubit state
            if (i >> qubit) & 1:
                # |1> component
                new_state[i] = gate[1, 1] * self._state[i]
                # Mix with |0> component
                if i - (1 << qubit) >= 0:
                    new_state[i] += gate[1, 0] * self._state[i - (1 << qubit)]
            else:
                # |0> component
                new_state[i] = gate[0, 0] * self._state[i]
                # Mix with |1> component
                if i + (1 << qubit) < n:
                    new_state[i] += gate[0, 1] * self._state[i + (1 << qubit)]

        self._state = new_state

    def _apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate."""
        n = 2**self.n_qubits
        new_state = np.zeros(n, dtype=complex)

        for i in range(n):
            # Check control qubit
            if (i >> control) & 1:
                # Control is 1: flip target
                new_i = i ^ (1 << target)
                new_state[new_i] = self._state[i]
            else:
                new_state[i] = self._state[i]

        self._state = new_state

    def execute_circuit(self) -> np.ndarray:
        """Execute all gates in the circuit."""
        # Gate matrices
        gates = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
            "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        }

        for gate_def in self.circuit:
            gate_name = gate_def["gate"]
            qubits = gate_def["qubits"]
            params = gate_def["parameters"]

            if gate_name in gates:
                self._apply_single_gate(gates[gate_name], qubits[0])
            elif gate_name == "CNOT":
                self._apply_cnot(qubits[0], qubits[1])
            elif gate_name in ["RX", "RY", "RZ"]:
                theta = params.get("theta", 0)
                if gate_name == "RX":
                    mat = np.array(
                        [[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]],
                        dtype=complex,
                    )
                elif gate_name == "RY":
                    mat = np.array(
                        [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex
                    )
                else:  # RZ
                    mat = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
                self._apply_single_gate(mat, qubits[0])

        return self._state

    def measure(self, observable: str, n_shots: int = 1000) -> QuantumResult:
        """Measure observable."""
        import time

        start = time.time()

        # Execute circuit
        state = self.execute_circuit()

        # Compute expectation values
        expectation_values = {}

        if observable.startswith("Z"):
            # Z-basis measurement
            probs = np.abs(state) ** 2

            # Z^i expectation
            for i, char in enumerate(reversed(observable)):
                if char == "Z":
                    exp_z = 0.0
                    for j, p in enumerate(probs):
                        bit = (j >> i) & 1
                        exp_z += (-1 if bit else 1) * p
                    expectation_values[f"Z{i}"] = exp_z

            # Z^i Z^j expectation
            z_positions = [i for i, c in enumerate(reversed(observable)) if c == "Z"]
            for i, pos_i in enumerate(z_positions):
                for pos_j in z_positions[i + 1 :]:
                    exp_zz = 0.0
                    for j, p in enumerate(probs):
                        bit_i = (j >> pos_i) & 1
                        bit_j = (j >> pos_j) & 1
                        exp_zz += (-1 if bit_i ^ bit_j else 1) * p
                    expectation_values[f"Z{pos_i}Z{pos_j}"] = exp_zz

        # Compute probabilities
        probabilities = np.abs(state) ** 2

        return QuantumResult(
            expectation_values=expectation_values,
            state_vector=state,
            probabilities=probabilities,
            n_qubits=self.n_qubits,
            n_shots=n_shots,
            execution_time=time.time() - start,
            backend="local_simulator",
            success=True,
        )

    def get_state_vector(self) -> np.ndarray:
        """Get final state vector."""
        return self._state

    def reset(self) -> None:
        """Reset circuit."""
        self.circuit = []
        self._state = None


class AWSBraketDriver(QuantumSimulator):
    """
    AWS Braket driver for cloud quantum computing.

    Supports:
    - Local simulators via Braket SDK
    - Amazon Braket managed simulators (SV1, TN1)
    - Rigetti, IonQ, Oxford Quantum Circuits hardware

    Note: Requires boto3 and amazon-braket-sdk to be installed.
    """

    def __init__(
        self,
        device_arn: str | None = None,
        s3_bucket: str | None = None,
        region: str = "us-east-1",
        local: bool = True,
    ):
        """
        Initialize AWS Braket driver.

        Args:
            device_arn: AWS device ARN (for cloud access).
            s3_bucket: S3 bucket for results.
            region: AWS region.
            local: Use local simulator (vs cloud).
        """
        self.device_arn = device_arn
        self.s3_bucket = s3_bucket
        self.region = region
        self.local = local

        try:
            from braket.aws import AwsDevice
            from braket.devices import LocalSimulator as BraketLocal

            self._available = True
            self._AwsDevice = AwsDevice
            self._LocalSimulator = BraketLocal
            logger.info(f"AWSBraketDriver initialized: local={local}, region={region}")
        except ImportError:
            self._available = False
            logger.warning("AWS Braket not available. Install with: pip install amazon-braket-sdk")

    def initialize_circuit(self, n_qubits: int) -> Any:
        """Initialize Braket circuit."""
        if not self._available:
            raise RuntimeError("AWS Braket not installed")

        from braket.circuits import Circuit

        self._circuit = Circuit()
        self.n_qubits = n_qubits
        return self._circuit

    def add_gate(
        self,
        gate_name: str,
        qubits: list[int],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Add gate to Braket circuit."""
        if not self._available:
            raise RuntimeError("AWS Braket not installed")

        from braket.circuits import AngledGate, Gate

        gate_map = {
            "H": Gate.H,
            "X": Gate.X,
            "Y": Gate.Y,
            "Z": Gate.Z,
            "CNOT": Gate.CNot,
            "SWAP": Gate.Swap,
        }

        if gate_name in gate_map:
            self._circuit.add_gate(gate_map[gate_name](*qubits))
        elif gate_name in ["RX", "RY", "RZ"]:
            theta = parameters.get("theta", 0) if parameters else 0
            self._circuit.add_angled_gate(AngledGate(gate_name, theta, *qubits))

    def measure(self, observable: str, n_shots: int = 1000) -> QuantumResult:
        """Execute on Braket device."""
        if not self._available:
            return QuantumResult(success=False, error="AWS Braket not installed")

        import time

        start = time.time()

        try:
            if self.local:
                device = self._LocalSimulator()
            else:
                device = self._AwsDevice(self.device_arn)

            task = device.run(self._circuit, shots=n_shots)
            result = task.result()

            # Extract expectation values
            counts = result.measurement_counts

            expectation_values = {}
            probs = {state: count / n_shots for state, count in counts.items()}

            # Compute Z expectations
            for i in range(self.n_qubits):
                exp_z = 0.0
                for state, prob in probs.items():
                    bit = int(state[i]) if i < len(state) else 0
                    exp_z += (-1 if bit else 1) * prob
                expectation_values[f"Z{i}"] = exp_z

            return QuantumResult(
                expectation_values=expectation_values,
                probabilities=np.array(list(probs.values())),
                n_qubits=self.n_qubits,
                n_shots=n_shots,
                execution_time=time.time() - start,
                backend="aws_braket" + ("_local" if self.local else ""),
                success=True,
            )

        except Exception as e:
            return QuantumResult(success=False, error=str(e), execution_time=time.time() - start)

    def get_state_vector(self) -> np.ndarray:
        """Get state vector (statevector simulator only)."""
        if not self._available:
            return np.array([])

        try:
            device = self._LocalSimulator("default")
            task = device.run(self._circuit, shots=0)
            result = task.result()
            return result.state_vector()
        except Exception:
            return np.array([])


class QuantumDriver:
    """
    Unified quantum driver that manages multiple backends.

    Provides a simple interface for the QML engine while supporting
    multiple quantum computing backends.

    Example::

        driver = QuantumDriver(backend="local", n_qubits=4)
        result = driver.execute_circuit(gates=[...])
        print(f"Energy: {result.expectation_values}")
    """

    def __init__(
        self,
        backend: str = "local",
        n_qubits: int = 4,
        **kwargs,
    ):
        """
        Initialize quantum driver.

        Args:
            backend: Backend name ('local', 'aws_braket', 'ibmq').
            n_qubits: Number of qubits.
            **kwargs: Backend-specific options.
        """
        self.backend_name = backend
        self.n_qubits = n_qubits
        self.kwargs = kwargs

        if backend == "local":
            self._simulator = LocalSimulator(n_qubits=n_qubits, **kwargs)
        elif backend == "aws_braket":
            self._simulator = AWSBraketDriver(**kwargs)
        else:
            self._simulator = LocalSimulator(n_qubits=n_qubits)

        logger.info(f"QuantumDriver initialized: backend={backend}, qubits={n_qubits}")

    def execute_circuit(
        self,
        gates: list[dict[str, Any]],
        n_shots: int = 1000,
    ) -> QuantumResult:
        """
        Execute a circuit defined as gate list.

        Args:
            gates: List of gate dictionaries with 'name', 'qubits', 'params'.
            n_shots: Number of measurement shots.

        Returns:
            QuantumResult with measurement outcomes.
        """
        self._simulator.initialize_circuit(self.n_qubits)

        for gate in gates:
            self._simulator.add_gate(
                gate["name"],
                gate["qubits"],
                gate.get("params", {}),
            )

        return self._simulator.measure("Z" * self.n_qubits, n_shots=n_shots)

    def execute_ansatz(
        self,
        ansatz_fn: callable,
        parameters: np.ndarray,
        n_shots: int = 1000,
    ) -> QuantumResult:
        """
        Execute a parameterized ansatz circuit.

        Args:
            ansatz_fn: Function that builds circuit with given parameters.
            parameters: Circuit parameters.
            n_shots: Number of shots.

        Returns:
            QuantumResult.
        """
        circuit = ansatz_fn(parameters)
        return self.execute_circuit(circuit, n_shots=n_shots)

    def get_state_vector(self) -> np.ndarray:
        """Get current state vector."""
        return self._simulator.get_state_vector()

    @property
    def available_backends(self) -> list[str]:
        """List available quantum backends."""
        backends = ["local"]

        try:

            backends.append("aws_braket")
        except ImportError:
            pass

        return backends
