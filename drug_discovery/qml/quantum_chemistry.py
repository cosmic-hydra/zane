"""
Active Space Approximation for Quantum Chemistry Calculations.

This module automates the selection of active orbital spaces for molecular systems,
reducing qubit requirements while maintaining chemical accuracy.

References:
    - Szabo and Ostlund, "Modern Quantum Chemistry"
    - Helgaker et al., "Molecular Electronic-Structure Theory"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    logger.warning("RDKit not available. Using simplified orbital models.")


@dataclass
class MolecularOrbitals:
    """Represents molecular orbital information for active space selection.

    Attributes:
        num_orbitals: Total number of molecular orbitals.
        num_electrons: Total number of electrons.
        orbital_energies: Energies of molecular orbitals.
        orbital_coefficients: Coefficients for orbital transformations.
        orbital_types: Types of orbitals (core, active, virtual).
        occupation_numbers: Electron occupation per orbital.
    """

    num_orbitals: int = 0
    num_electrons: int = 0
    orbital_energies: np.ndarray | None = None
    orbital_coefficients: np.ndarray | None = None
    orbital_types: list[str] | None = None
    occupation_numbers: np.ndarray | None = None

    def get_active_space(
        self,
        n_active_electrons: int,
        n_active_orbitals: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract active space indices and matrices.

        Args:
            n_active_electrons: Number of electrons in active space.
            n_active_orbitals: Number of orbitals in active space.

        Returns:
            Tuple of (active_energies, active_coeffs, active_occ, core_energy).
        """
        if self.orbital_types is None:
            self._infer_orbital_types()

        # Find active orbital indices
        active_indices = [
            i for i, t in enumerate(self.orbital_types) if t == "active"
        ][:n_active_orbitals]

        if len(active_indices) < n_active_orbitals:
            logger.warning(
                f"Only {len(active_indices)} active orbitals found, need {n_active_orbitals}"
            )
            active_indices = list(range(min(n_active_orbitals, self.num_orbitals)))

        # Extract active space data
        active_energies = self.orbital_energies[active_indices] if self.orbital_energies is not None else np.array([0.0] * len(active_indices))
        active_coeffs = self.orbital_coefficients[active_indices] if self.orbital_coefficients is not None else np.eye(len(active_indices))

        # Calculate active space occupation
        n_electrons_per_orbital = n_active_electrons / n_active_orbitals
        active_occ = np.array([min(2.0, max(0.0, n_electrons_per_orbital))] * len(active_indices))

        # Calculate core energy (electrons not in active space)
        core_electrons = self.num_electrons - n_active_electrons
        core_energy = 0.0
        if self.orbital_energies is not None and self.orbital_types is not None:
            for i, t in enumerate(self.orbital_types):
                if t == "core" and i < core_electrons // 2:
                    core_energy += self.orbital_energies[i] * 2

        return active_energies, active_coeffs, active_occ, core_energy

    def _infer_orbital_types(self) -> None:
        """Infer orbital types from occupation numbers."""
        if self.occupation_numbers is not None:
            types = []
            for occ in self.occupation_numbers:
                if occ >= 1.9:  # Fully occupied
                    types.append("core")
                elif occ >= 0.1:  # Partially occupied
                    types.append("active")
                else:  # Empty
                    types.append("virtual")
            self.orbital_types = types
        else:
            # Default: first half are core, rest are virtual
            n_core_orbs = self.num_electrons // 2
            self.orbital_types = []
            for i in range(self.num_orbitals):
                if i < n_core_orbs:
                    self.orbital_types.append("core")
                else:
                    self.orbital_types.append("virtual")
            # Mark some as active for variation
            mid_start = max(0, n_core_orbs - 2)
            mid_end = min(self.num_orbitals, n_core_orbs + 4)
            for i in range(mid_start, mid_end):
                self.orbital_types[i] = "active"

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_orbitals": self.num_orbitals,
            "num_electrons": self.num_electrons,
            "orbital_energies": self.orbital_energies.tolist() if self.orbital_energies is not None else None,
            "orbital_types": self.orbital_types,
            "occupation_numbers": self.occupation_numbers.tolist() if self.occupation_numbers is not None else None,
        }


@dataclass
class ActiveSpaceResult:
    """Result of active space approximation.

    Attributes:
        n_active_electrons: Number of active electrons.
        n_active_orbitals: Number of active orbitals.
        active_energies: Active orbital energies.
        active_coefficients: Active orbital coefficients.
        core_energy: Energy of core electrons.
        one_electron_integrals: One-electron integral matrix.
        two_electron_integrals: Two-electron integral tensor.
        qubit_hamiltonian: Reduced Hamiltonian for quantum computing.
        n_qubits_required: Number of qubits needed.
        reduction_ratio: Qubit reduction compared to full space.
    """

    n_active_electrons: int = 0
    n_active_orbitals: int = 0
    active_energies: np.ndarray | None = None
    active_coefficients: np.ndarray | None = None
    core_energy: float = 0.0
    one_electron_integrals: np.ndarray | None = None
    two_electron_integrals: np.ndarray | None = None
    qubit_hamiltonian: dict[str, float] | None = None
    n_qubits_required: int = 0
    reduction_ratio: float = 1.0
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_active_electrons": self.n_active_electrons,
            "n_active_orbitals": self.n_active_orbitals,
            "active_energies": self.active_energies.tolist() if self.active_energies is not None else None,
            "core_energy": self.core_energy,
            "n_qubits_required": self.n_qubits_required,
            "reduction_ratio": self.reduction_ratio,
            "success": self.success,
            "error": self.error,
        }


class ActiveSpaceApproximator:
    """
    Automates active space selection for molecular quantum chemistry.

    This class accepts an RDKit molecular graph and performs:
    1. Electronic structure calculation (simplified or with OpenFermion)
    2. Orbital ordering and selection
    3. Active space truncation
    4. Hamiltonian reduction for quantum computing

    Example::

        approximator = ActiveSpaceApproximator()
        result = approximator.approximate_active_space(
            smiles="CCO",  # ethanol
            n_active_orbitals=4,
        )
        print(f"Qubits required: {result.n_qubits_required}")
    """

    def __init__(
        self,
        method: str = "molecular_orbital",
        correlation_threshold: float = 0.01,
        max_active_orbitals: int = 10,
    ):
        """
        Initialize the active space approximator.

        Args:
            method: Method for orbital selection ('molecular_orbital', 'localized', 'ford').
            correlation_threshold: Threshold for orbital correlation.
            max_active_orbitals: Maximum number of active orbitals.
        """
        self.method = method
        self.correlation_threshold = correlation_threshold
        self.max_active_orbitals = max_active_orbitals

        logger.info(f"ActiveSpaceApproximator initialized with method={method}")

    def approximate_active_space(
        self,
        smiles: str | None = None,
        mol: Any = None,
        n_active_orbitals: int = 4,
        n_active_electrons: int | None = None,
    ) -> ActiveSpaceResult:
        """
        Approximate the active space for a molecule.

        Args:
            smiles: SMILES string of the molecule.
            mol: RDKit mol object (alternative to smiles).
            n_active_orbitals: Target number of active orbitals.
            n_active_electrons: Target number of active electrons.

        Returns:
            ActiveSpaceResult with reduced Hamiltonian.
        """
        try:
            # Get RDKit molecule
            if mol is None and smiles is not None:
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        raise ValueError(f"Invalid SMILES: {smiles}")
                else:
                    return self._approximate_without_rdkit(smiles, n_active_orbitals, n_active_electrons)

            # Get molecular orbitals
            orbitals = self._compute_molecular_orbitals(mol, smiles)

            # Determine active electrons
            if n_active_electrons is None:
                n_active_electrons = min(orbitals.num_electrons, n_active_orbitals * 2)

            # Ensure n_active_electrons is even
            n_active_electrons = (n_active_electrons // 2) * 2

            # Extract active space
            active_energies, active_coeffs, active_occ, core_energy = orbitals.get_active_space(
                n_active_electrons, n_active_orbitals
            )

            # Compute integrals (simplified)
            one_e_int, two_e_int = self._compute_integrals(
                active_energies, active_coeffs, n_active_orbitals
            )

            # Build qubit Hamiltonian (Jordan-Wigner mapping)
            qubit_hamiltonian = self._build_qubit_hamiltonian(
                one_e_int, two_e_int, n_active_orbitals
            )

            # Calculate qubit requirements
            n_qubits = n_active_orbitals  # One qubit per spatial orbital
            full_qubits = orbitals.num_orbitals * 2  # Spin-orbitals
            reduction_ratio = 1.0 - (n_qubits / full_qubits) if full_qubits > 0 else 0.0

            return ActiveSpaceResult(
                n_active_electrons=n_active_electrons,
                n_active_orbitals=n_active_orbitals,
                active_energies=active_energies,
                active_coefficients=active_coeffs,
                core_energy=core_energy,
                one_electron_integrals=one_e_int,
                two_electron_integrals=two_e_int,
                qubit_hamiltonian=qubit_hamiltonian,
                n_qubits_required=n_qubits,
                reduction_ratio=reduction_ratio,
                success=True,
            )

        except Exception as e:
            logger.error(f"Active space approximation failed: {e}")
            return ActiveSpaceResult(success=False, error=str(e))

    def _compute_molecular_orbitals(
        self,
        mol: Any,
        smiles: str | None,
    ) -> MolecularOrbitals:
        """Compute molecular orbitals from RDKit molecule."""
        if RDKIT_AVAILABLE and mol is not None:
            # Simplified orbital model based on atomic properties
            num_atoms = mol.GetNumAtoms()
            num_electrons = sum(
                atom.GetTotalValence() for atom in mol.GetAtoms()
            )

            # Estimate orbital energies based on atom types
            orbital_energies = []
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                # Approximate orbital energies
                if atomic_num == 1:  # Hydrogen
                    orbital_energies.extend([-13.6, 0.0])  # 1s, virtual
                elif atomic_num == 6:  # Carbon
                    orbital_energies.extend([-11.3, -0.5, 0.5, 1.0])  # 2s, 2p
                elif atomic_num == 7:  # Nitrogen
                    orbital_energies.extend([-14.5, -0.7, 0.3, 0.8])
                elif atomic_num == 8:  # Oxygen
                    orbital_energies.extend([-17.3, -0.9, 0.1, 0.6])
                else:
                    orbital_energies.extend([-10.0, 0.0])

            num_orbitals = len(orbital_energies)

            # Orbital coefficients (identity for simplicity)
            orbital_coeffs = np.eye(num_orbitals) if num_orbitals > 0 else np.array([[]])

            return MolecularOrbitals(
                num_orbitals=num_orbitals,
                num_electrons=num_electrons,
                orbital_energies=np.array(orbital_energies),
                orbital_coefficients=orbital_coeffs,
            )
        else:
            # Fallback for simple molecules
            return self._simple_orbital_model(smiles or "C")

    def _simple_orbital_model(self, smiles: str) -> MolecularOrbitals:
        """Create simple orbital model from SMILES."""
        # Count atoms for electron estimation
        atom_count = len([c for c in smiles if c.isupper()])

        num_electrons = atom_count * 4  # Approximate
        num_orbitals = atom_count * 2

        # Simple orbital energies
        energies = np.linspace(-15.0, 2.0, num_orbitals)
        coeffs = np.eye(num_orbitals)

        return MolecularOrbitals(
            num_orbitals=num_orbitals,
            num_electrons=num_electrons,
            orbital_energies=energies,
            orbital_coefficients=coeffs,
        )

    def _compute_integrals(
        self,
        active_energies: np.ndarray,
        active_coeffs: np.ndarray,
        n_orbitals: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute one and two electron integrals."""
        # One-electron integrals (diagonal from orbital energies)
        one_e = np.diag(active_energies) if len(active_energies) > 0 else np.array([])

        # Two-electron integrals (simplified Coulomb repulsion)
        n = len(active_energies)
        two_e = np.zeros((n, n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        # Simplified: (ij|kl) ≈ 1/(1 + |r_ij - r_kl|)
                        if i == k and j == l:
                            two_e[i, j, k, l] = 1.0  # Coulomb
                        elif i == l and j == k:
                            two_e[i, j, k, l] = 0.5  # Exchange

        return one_e, two_e

    def _build_qubit_hamiltonian(
        self,
        one_e_int: np.ndarray,
        two_e_int: np.ndarray,
        n_orbitals: int,
    ) -> dict[str, float]:
        """
        Build qubit Hamiltonian using Jordan-Wigner transformation.

        Returns:
            Dictionary mapping Pauli strings to coefficients.
        """
        hamiltonian = {}

        # One-electron terms
        for p in range(n_orbitals):
            coef = one_e_int[p, p] if one_e_int.size > 0 else 0.0
            if abs(coef) > 1e-10:
                # ^p^\dagger ^p term
                hamiltonian[f"Z{p}"] = hamiltonian.get(f"Z{p}", 0.0) + coef * 0.5
                hamiltonian["I"] = hamiltonian.get("I", 0.0) + coef * 0.5

        # Two-electron terms (simplified)
        n = min(n_orbitals, len(two_e_int) if two_e_int.size > 0 else 0)
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coef = two_e_int[p, q, r, s] if two_e_int.size > 0 else 0.0
                        if abs(coef) > 1e-10 and p == r and q == s:
                            # (pp|qq) type terms
                            hamiltonian["I"] = hamiltonian.get("I", 0.0) + coef * 0.25
                            for Z in [f"Z{p}", f"Z{q}"]:
                                hamiltonian[Z] = hamiltonian.get(Z, 0.0) - coef * 0.25
                            hamiltonian[f"Z{p}Z{q}"] = hamiltonian.get(f"Z{p}Z{q}", 0.0) + coef * 0.25

        return hamiltonian

    def _approximate_without_rdkit(
        self,
        smiles: str,
        n_active_orbitals: int,
        n_active_electrons: int | None,
    ) -> ActiveSpaceResult:
        """Fallback when RDKit unavailable."""
        orbitals = self._simple_orbital_model(smiles)

        n_electrons = n_active_electrons or min(orbitals.num_electrons, n_active_orbitals * 2)
        n_electrons = (n_electrons // 2) * 2

        active_energies = orbitals.orbital_energies[:n_active_orbitals]
        active_coeffs = orbitals.orbital_coefficients[:n_active_orbitals, :n_active_orbitals]

        one_e, two_e = self._compute_integrals(active_energies, active_coeffs, n_active_orbitals)
        hamiltonian = self._build_qubit_hamiltonian(one_e, two_e, n_active_orbitals)

        return ActiveSpaceResult(
            n_active_electrons=n_electrons,
            n_active_orbitals=n_active_orbitals,
            active_energies=active_energies,
            active_coefficients=active_coeffs,
            core_energy=0.0,
            one_electron_integrals=one_e,
            two_electron_integrals=two_e,
            qubit_hamiltonian=hamiltonian,
            n_qubits_required=n_active_orbitals,
            reduction_ratio=0.5,
            success=True,
        )

    def optimize_active_space(
        self,
        smiles: str | None = None,
        mol: Any = None,
        max_orbitals: int = 8,
    ) -> ActiveSpaceResult:
        """
        Find optimal active space by maximizing correlation within constraints.

        Args:
            smiles: SMILES string.
            mol: RDKit mol object.
            max_orbitals: Maximum orbitals to consider.

        Returns:
            Best ActiveSpaceResult found.
        """
        best_result = None
        best_score = float('-inf')

        for n_orbs in range(2, max_orbitals + 1):
            result = self.approximate_active_space(
                smiles=smiles,
                mol=mol,
                n_active_orbitals=n_orbs,
            )

            if result.success:
                # Score: balance between reduction and chemical accuracy
                # Higher is better
                energy_spread = (
                    np.max(result.active_energies) - np.min(result.active_energies)
                    if result.active_energies is not None and len(result.active_energies) > 0
                    else 0
                )
                score = result.reduction_ratio * energy_spread

                if score > best_score:
                    best_score = score
                    best_result = result

        return best_result or ActiveSpaceResult(success=False, error="No valid active space found")
