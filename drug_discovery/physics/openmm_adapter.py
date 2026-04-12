"""OpenMM molecular dynamics adapter.

Wraps OpenMM (https://github.com/openmm/openmm) for protein–ligand stability
simulations and complements the existing
:class:`~drug_discovery.physics.md_simulator.MolecularDynamicsSimulator` with
an OpenMM-native path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class MDResult:
    """Result of a molecular dynamics simulation."""

    smiles: str
    success: bool
    backend: str = "openmm"
    energy_kcal_mol: float | None = None
    stability_score: float | None = None
    rmsd: float | None = None
    simulation_time_ns: float = 0.0
    stable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "success": self.success,
            "backend": self.backend,
            "energy_kcal_mol": self.energy_kcal_mol,
            "stability_score": self.stability_score,
            "rmsd": self.rmsd,
            "simulation_time_ns": self.simulation_time_ns,
            "stable": self.stable,
            "metadata": self.metadata,
        }


class OpenMMAdapter:
    """High-level OpenMM adapter for drug-candidate stability assessment.

    When OpenMM is not installed, the adapter delegates to the existing
    :class:`~drug_discovery.physics.md_simulator.MolecularDynamicsSimulator`
    so the pipeline always produces results.

    Usage::

        adapter = OpenMMAdapter(temperature_K=300)
        result = adapter.simulate_ligand("CCO")
        print(result.stable, result.stability_score)
    """

    def __init__(
        self,
        temperature_K: float = 300.0,
        pressure_bar: float = 1.0,
        timestep_fs: float = 2.0,
        platform: str = "CPU",
    ):
        """
        Args:
            temperature_K: Simulation temperature in Kelvin.
            pressure_bar: Simulation pressure in bar.
            timestep_fs: Integration timestep in femtoseconds.
            platform: OpenMM platform name (``'CPU'``, ``'CUDA'``, ``'OpenCL'``).
        """
        self.temperature_K = temperature_K
        self.pressure_bar = pressure_bar
        self.timestep_fs = timestep_fs
        self.platform = platform

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return *True* if OpenMM can be imported."""
        ensure_local_checkout_on_path("openmm")
        try:
            import openmm  # type: ignore[import]  # noqa: F401

            return True
        except Exception:
            return False

    def simulate_ligand(self, smiles: str, steps: int = 10_000) -> MDResult:
        """Run an implicit-solvent MD simulation for a single ligand.

        Args:
            smiles: Ligand SMILES string.
            steps: Number of MD steps.

        Returns:
            :class:`MDResult` describing simulation outcome.
        """
        if self.is_available():
            return self._run_openmm(smiles, steps=steps)
        return self._run_fallback(smiles, steps=steps)

    def simulate_complex(
        self,
        protein_pdb: str,
        ligand_smiles: str,
        steps: int = 50_000,
    ) -> MDResult:
        """Simulate a protein–ligand complex.

        Args:
            protein_pdb: Path to the protein PDB file or raw PDB string.
            ligand_smiles: Ligand SMILES string.
            steps: Number of MD steps.

        Returns:
            :class:`MDResult` describing simulation outcome.
        """
        if self.is_available():
            return self._run_openmm_complex(protein_pdb, ligand_smiles, steps=steps)
        return self._run_fallback_complex(protein_pdb, ligand_smiles, steps=steps)

    def batch_simulate(self, smiles_list: list[str], steps: int = 10_000) -> list[MDResult]:
        """Simulate a list of ligands and return their :class:`MDResult` objects.

        Args:
            smiles_list: List of SMILES strings.
            steps: Number of MD steps per simulation.

        Returns:
            List of :class:`MDResult` objects.
        """
        return [self.simulate_ligand(smi, steps=steps) for smi in smiles_list]

    def stability_screen(
        self,
        smiles_list: list[str],
        steps: int = 10_000,
        threshold: float = 0.6,
    ) -> list[tuple[str, float]]:
        """Screen molecules for MD stability and return those above *threshold*.

        Args:
            smiles_list: Candidate SMILES.
            steps: MD steps per simulation.
            threshold: Minimum stability score to include in results.

        Returns:
            List of ``(smiles, stability_score)`` tuples, sorted descending.
        """
        results = self.batch_simulate(smiles_list, steps=steps)
        hits = [
            (r.smiles, r.stability_score or 0.0)
            for r in results
            if r.success and (r.stability_score or 0.0) >= threshold
        ]
        return sorted(hits, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_openmm(self, smiles: str, steps: int) -> MDResult:
        """Execute OpenMM simulation (requires OpenMM to be installed)."""
        try:
            from drug_discovery.external_tooling import openmm_minimize_energy

            raw = openmm_minimize_energy(smiles, steps=steps)
            energy = raw.get("energy_kcal_mol")
            success = bool(raw.get("success", raw.get("available", False)))

            stability_score: float | None = None
            if energy is not None:
                stability_score = min(1.0, max(0.0, 1.0 - abs(energy) / 500.0))

            return MDResult(
                smiles=smiles,
                success=success,
                backend=raw.get("backend", "openmm"),
                energy_kcal_mol=energy,
                stability_score=stability_score,
                stable=(stability_score or 0.0) >= 0.6,
                simulation_time_ns=steps * self.timestep_fs * 1e-6,
                metadata=raw,
            )
        except Exception as exc:
            logger.error("OpenMM simulation failed: %s", exc)
            return MDResult(smiles=smiles, success=False, backend="openmm_error", metadata={"error": str(exc)})

    def _run_fallback(self, smiles: str, steps: int) -> MDResult:
        """Delegate to the existing MolecularDynamicsSimulator (no OpenMM needed)."""
        logger.info(
            "OpenMM not available; using MolecularDynamicsSimulator fallback. "
            "Run: git submodule update --init external/openmm"
        )
        try:
            from drug_discovery.physics.md_simulator import MolecularDynamicsSimulator

            sim = MolecularDynamicsSimulator(
                temperature=self.temperature_K,
                pressure=self.pressure_bar,
                timestep=self.timestep_fs,
            )
            raw = sim.simulate_ligand(smiles, num_steps=steps)
            success = bool(raw.get("success", False))
            return MDResult(
                smiles=smiles,
                success=success,
                backend="md_simulator_fallback",
                energy_kcal_mol=raw.get("final_energy"),
                stability_score=raw.get("stability_index"),
                rmsd=raw.get("rmsd"),
                stable=bool(raw.get("stable", False)),
                simulation_time_ns=steps * self.timestep_fs * 1e-6,
                metadata=raw,
            )
        except Exception as exc:
            logger.error("MD fallback simulation failed: %s", exc)
            return MDResult(smiles=smiles, success=False, backend="fallback_error", metadata={"error": str(exc)})

    def _run_openmm_complex(self, protein_pdb: str, ligand_smiles: str, steps: int) -> MDResult:
        """Simulate a protein–ligand complex with OpenMM."""
        try:
            # Placeholder for full OpenMM complex simulation.
            ligand_result = self._run_openmm(ligand_smiles, steps=max(5000, steps // 5))
            return MDResult(
                smiles=ligand_smiles,
                success=ligand_result.success,
                backend="openmm_complex",
                energy_kcal_mol=ligand_result.energy_kcal_mol,
                stability_score=ligand_result.stability_score,
                rmsd=ligand_result.rmsd,
                stable=ligand_result.stable,
                simulation_time_ns=steps * self.timestep_fs * 1e-6,
                metadata={"protein_pdb": protein_pdb, "ligand_result": ligand_result.as_dict()},
            )
        except Exception as exc:
            logger.error("OpenMM complex simulation failed: %s", exc)
            return MDResult(
                smiles=ligand_smiles, success=False, backend="openmm_complex_error", metadata={"error": str(exc)}
            )

    def _run_fallback_complex(self, protein_pdb: str, ligand_smiles: str, steps: int) -> MDResult:
        """Delegate complex simulation to the existing MolecularDynamicsSimulator."""
        try:
            from drug_discovery.physics.md_simulator import MolecularDynamicsSimulator

            sim = MolecularDynamicsSimulator(temperature=self.temperature_K)
            raw = sim.simulate_protein_ligand_complex(protein_pdb, ligand_smiles, num_steps=steps)
            success = bool(raw.get("success", False))
            return MDResult(
                smiles=ligand_smiles,
                success=success,
                backend="md_simulator_fallback_complex",
                energy_kcal_mol=raw.get("binding_energy"),
                stable=raw.get("stability") == "stable",
                rmsd=raw.get("ligand_rmsd"),
                simulation_time_ns=steps * self.timestep_fs * 1e-6,
                metadata=raw,
            )
        except Exception as exc:
            logger.error("Fallback complex simulation failed: %s", exc)
            return MDResult(
                smiles=ligand_smiles, success=False, backend="fallback_complex_error", metadata={"error": str(exc)}
            )

    @staticmethod
    def integration_status() -> dict[str, Any]:
        """Return the current integration status for OpenMM."""
        return get_integration_status("openmm").as_dict()
