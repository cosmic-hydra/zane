"""
Error Mitigation via Zero-Noise Extrapolation (ZNE).

Implements ZNE to simulate ideal quantum calculation conditions and correct
for hardware noise, enabling mathematically precise binding affinity approximations.

References:
    - Temme et al., "Error Mitigation for Short-Depth Quantum Circuits"
    - Li & Benjamin, "Efficient Variational Quantum Simulator Incorporating Commuting"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logging.warning("PyTorch not available. Using numpy-only ZNE.")

logger = logging.getLogger(__name__)


@dataclass
class ErrorMitigationConfig:
    """Configuration for error mitigation.

    Attributes:
        noise_factors: Noise scaling factors for extrapolation.
        extrapolation_method: Method for fitting ('linear', 'polynomial', 'exponential').
        degree: Polynomial degree for extrapolation.
        n_samples: Number of samples per noise factor.
        confidence_level: Confidence level for error bars.
    """

    noise_factors: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    extrapolation_method: str = "polynomial"
    degree: int = 2
    n_samples: int = 100
    confidence_level: float = 0.95


@dataclass
class ZNEResult:
    """Result of Zero-Noise Extrapolation.

    Attributes:
        mitigated_energy: Extrapolated zero-noise energy estimate.
        uncertainty: Uncertainty in the estimate.
        noise_energies: Measured energies at each noise factor.
        noise_factors: Noise factors used.
        extrapolation_params: Fitted extrapolation parameters.
        confidence_interval: (lower, upper) confidence bounds.
        success: Whether extrapolation succeeded.
        error: Error message if failed.
    """

    mitigated_energy: float = 0.0
    uncertainty: float = 0.0
    noise_energies: list[float] = field(default_factory=list)
    noise_factors: list[float] = field(default_factory=list)
    extrapolation_params: np.ndarray | None = None
    confidence_interval: tuple[float, float] | None = None
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mitigated_energy": self.mitigated_energy,
            "uncertainty": self.uncertainty,
            "noise_energies": self.noise_energies,
            "noise_factors": self.noise_factors,
            "extrapolation_params": self.extrapolation_params.tolist() if self.extrapolation_params is not None else None,
            "confidence_interval": self.confidence_interval,
            "success": self.success,
            "error": self.error,
        }


class ZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation (ZNE) error mitigation.

    ZNE works by:
    1. Running quantum circuits at different noise levels (noise factors)
    2. Measuring expectation values at each noise level
    3. Extrapolating back to zero noise (ideal) condition

    Supports multiple extrapolation methods:
    - Linear: E(λ) = a + b*λ
    - Polynomial: E(λ) = Σ a_i * λ^i
    - Exponential: E(λ) = a + b*exp(-c*λ)

    Example::

        zne = ZeroNoiseExtrapolation(config=ErrorMitigationConfig(
            noise_factors=[1.0, 1.5, 2.0, 3.0],
            extrapolation_method='polynomial',
        ))

        result = zne.mitigate_energy(
            noisy_energy_fn=lambda f: compute_energy(scale_factor=f),
        )
        print(f"Mitigated energy: {result.mitigated_energy:.6f}")
    """

    def __init__(
        self,
        config: ErrorMitigationConfig | None = None,
        noise_model: str = " depolarizing",
    ):
        """
        Initialize ZNE.

        Args:
            config: Error mitigation configuration.
            noise_model: Model for noise scaling.
        """
        self.config = config or ErrorMitigationConfig()
        self.noise_model = noise_model

        logger.info(f"ZNE initialized with method={self.config.extrapolation_method}")

    def mitigate_energy(
        self,
        noisy_energy_fn: Callable[[float], float],
        noise_factors: list[float] | None = None,
    ) -> ZNEResult:
        """
        Mitigate noisy energy using ZNE.

        Args:
            noisy_energy_fn: Function that takes noise_factor and returns noisy energy.
            noise_factors: List of noise factors to evaluate.

        Returns:
            ZNEResult with mitigated energy estimate.
        """
        if noise_factors is None:
            noise_factors = self.config.noise_factors

        try:
            # Collect energies at different noise levels
            noise_energies = []
            for factor in noise_factors:
                energies = []
                for _ in range(self.config.n_samples):
                    energy = noisy_energy_fn(factor)
                    energies.append(energy)
                # Average over samples
                avg_energy = np.mean(energies)
                noise_energies.append(avg_energy)

            # Extrapolate to zero noise
            mitigated_energy, params = self._extrapolate(
                noise_factors, noise_energies
            )

            # Compute uncertainty
            uncertainty = self._compute_uncertainty(
                noise_factors, noise_energies, mitigated_energy
            )

            # Compute confidence interval
            confidence_interval = (
                mitigated_energy - uncertainty,
                mitigated_energy + uncertainty,
            )

            return ZNEResult(
                mitigated_energy=mitigated_energy,
                uncertainty=uncertainty,
                noise_energies=noise_energies,
                noise_factors=noise_factors,
                extrapolation_params=params,
                confidence_interval=confidence_interval,
                success=True,
            )

        except Exception as e:
            logger.error(f"ZNE mitigation failed: {e}")
            return ZNEResult(success=False, error=str(e))

    def _extrapolate(
        self,
        noise_factors: list[float],
        energies: list[float],
    ) -> tuple[float, np.ndarray]:
        """
        Extrapolate to zero noise.

        Args:
            noise_factors: List of noise factors.
            energies: Corresponding energies.

        Returns:
            Tuple of (extrapolated_energy, fit_params).
        """
        x = np.array(noise_factors)
        y = np.array(energies)

        if self.config.extrapolation_method == "linear":
            # Linear fit: E(λ) = a + b*λ
            coeffs = np.polyfit(x, y, 1)
            params = coeffs
            # Extrapolate to λ=0
            mitigated_energy = np.polyval(coeffs, 0.0)

        elif self.config.extrapolation_method == "polynomial":
            # Polynomial fit: E(λ) = Σ a_i * λ^i
            degree = min(self.config.degree, len(x) - 1)
            coeffs = np.polyfit(x, y, degree)
            params = coeffs
            mitigated_energy = np.polyval(coeffs, 0.0)

        elif self.config.extrapolation_method == "exponential":
            # Exponential fit: E(λ) = a + b*exp(-c*λ)
            mitigated_energy, params = self._exponential_fit(x, y)

        elif self.config.extrapolation_method == " Richardson":
            # Richardson extrapolation (special case for power law)
            mitigated_energy = self._richardson_extrapolation(x, y)
            params = np.array([mitigated_energy])

        else:
            # Default: linear extrapolation
            coeffs = np.polyfit(x, y, 1)
            params = coeffs
            mitigated_energy = np.polyval(coeffs, 0.0)

        return float(mitigated_energy), params

    def _exponential_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Fit exponential decay model."""
        try:
            # Initial guess
            a0 = y[-1]  # asymptotic value
            b0 = y[0] - y[-1]  # amplitude
            c0 = 0.5  # decay rate

            # Simple grid search for parameters
            best_error = float('inf')
            best_params = (a0, b0, c0)

            for a in np.linspace(min(y), max(y), 10):
                for b in np.linspace(-abs(y[0] - y[-1]), abs(y[0] - y[-1]), 10):
                    for c in np.linspace(0.1, 2.0, 10):
                        y_pred = a + b * np.exp(-c * x)
                        error = np.sum((y - y_pred) ** 2)
                        if error < best_error:
                            best_error = error
                            best_params = (a, b, c)

            # Extrapolate to λ=0
            a, b, c = best_params
            mitigated_energy = a + b * np.exp(-c * 0)  # = a + b

            return float(mitigated_energy), np.array(best_params)

        except Exception:
            # Fallback to linear
            coeffs = np.polyfit(x, y, 1)
            return float(np.polyval(coeffs, 0.0)), coeffs

    def _richardson_extrapolation(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Richardson extrapolation for power-law noise.

        Assumes noise scales as λ^k where k is unknown.
        """
        if len(x) < 3:
            # Fallback to linear
            coeffs = np.polyfit(x, y, 1)
            return float(np.polyval(coeffs, 0.0))

        # Use three points to estimate
        x1, x2, x3 = x[0], x[len(x)//2], x[-1]
        y1, y2, y3 = y[0], y[len(y)//2], y[-1]

        # Assume noise ~ λ^k
        # y(λ) = y(0) + a*λ^k
        # Use ratios to estimate k
        if y2 != y1 and y3 != y2:
            k = np.log((y3 - y1) / (y2 - y1)) / np.log(x3 / x2)

            # Extrapolate to λ=0
            # y(0) = y1 - a*λ1^k where a = (y2 - y1) / λ2^k
            k = max(0.5, min(k, 3.0))  # Bound k
            a = (y2 - y1) / (x2 ** k)
            mitigated_energy = y1 - a * (x1 ** k)

            return float(mitigated_energy)

        # Fallback
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, 0.0))

    def _compute_uncertainty(
        self,
        noise_factors: list[float],
        energies: list[float],
        mitigated_energy: float,
    ) -> float:
        """
        Compute uncertainty in mitigated estimate.

        Uses residual standard error from extrapolation fit.
        """
        x = np.array(noise_factors)
        y = np.array(energies)

        if len(x) < 3:
            return 0.1 * abs(mitigated_energy) if mitigated_energy != 0 else 0.1

        # Fit again to get residuals
        if self.config.extrapolation_method == "linear":
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
        else:
            degree = min(self.config.degree, len(x) - 1)
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)

        # Residual standard error
        residuals = y - y_pred
        n = len(x)
        p = len(coeffs)
        rse = np.sqrt(np.sum(residuals ** 2) / (n - p))

        # Uncertainty in extrapolation to λ=0
        # Use error propagation
        x0 = 0.0
        x_mean = np.mean(x)

        # Variance of prediction at x0
        ss_x = np.sum((x - x_mean) ** 2)
        if ss_x == 0:
            return rse

        se_y0 = rse * np.sqrt(1 + 1/n + (x0 - x_mean)**2 / ss_x)

        # Scale by confidence level
        from scipy import stats
        t_val = stats.t.ppf((1 + self.config.confidence_level) / 2, n - p)

        return float(se_y0 * t_val)

    def mitigate_circuit(
        self,
        circuit_fn: Callable,
        noise_scaling_fn: Callable[[float], Any],
    ) -> ZNEResult:
        """
        Mitigate a quantum circuit with noise scaling.

        Args:
            circuit_fn: Function executing the circuit without arguments.
            noise_scaling_fn: Function that takes scale factor and returns circuit result.

        Returns:
            ZNEResult with mitigated expectation value.
        """
        def noisy_energy(scale: float) -> float:
            result = noise_scaling_fn(scale)
            if hasattr(result, 'expectation_value'):
                return result.expectation_value
            return float(result)

        return self.mitigate_energy(noisy_energy)


class NoiseScaling:
    """
    Methods for scaling noise in quantum circuits.

    Different methods for increasing noise to enable extrapolation.
    """

    @staticmethod
    def identity_scaling(circuit: Any, scale_factor: float) -> Any:
        """
        Identity (no-op) noise scaling.

        Used for circuits already at noise level 1.
        """
        return circuit

    @staticmethod
    def gate_degradation(circuit: Any, scale_factor: float) -> Any:
        """
        Scale noise by adding identity (no-op) gates.

        Increases circuit depth proportionally.
        """
        n_id_gates = int((scale_factor - 1) * 10)  # Proportional to scale
        for _ in range(n_id_gates):
            circuit.add_gate("I")  # Identity gate
        return circuit

    @staticmethod
    def amplitude_damping(circuit: Any, scale_factor: float, base_error: float = 0.01) -> Any:
        """
        Apply amplitude damping noise scaled by factor.

        Args:
            circuit: Quantum circuit.
            scale_factor: Noise scaling factor.
            base_error: Base amplitude damping error rate.

        Returns:
            Circuit with scaled noise.
        """
        error_rate = base_error * scale_factor
        # Apply to all qubits
        for qubit in circuit.qubits:
            circuit.add_error("amplitude_damping", qubit, error_rate)
        return circuit

    @staticmethod
    def depolarizing(circuit: Any, scale_factor: float, base_error: float = 0.01) -> Any:
        """
        Apply depolarizing noise scaled by factor.

        Args:
            circuit: Quantum circuit.
            scale_factor: Noise scaling factor.
            base_error: Base depolarizing error rate.

        Returns:
            Circuit with scaled noise.
        """
        error_rate = base_error * scale_factor
        for qubit in circuit.qubits:
            circuit.add_error("depolarizing", qubit, error_rate)
        return circuit
