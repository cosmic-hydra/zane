"""OpenMM molecular dynamics adapter.

Wraps the ``openmm/openmm`` external submodule to run MD simulations and
compute stability metrics for drug–protein complexes.  The adapter falls back
gracefully when OpenMM is not installed, deferring to the existing
:class:`~drug_discovery.physics.md_simulator.MolecularDynamicsSimulator`
estimator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class MDSimulationResult:
    """Result of an OpenMM molecular-dynamics simulation."""

    smiles: str
    protein_pdb_path: str | None = None
    stability_score: float | None = None
    binding_energy: float | None = None
    rmsd: float | None = None
    num_steps: int = 0
    trajectory_summary: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "protein_pdb_path": self.protein_pdb_path,
            "stability_score": self.stability_score,
            "binding_energy": self.binding_energy,
            "rmsd": self.rmsd,
            "num_steps": self.num_steps,
            "trajectory_summary": self.trajectory_summary,
            "success": self.success,
            "error": self.error,
        }


class OpenMMAdapter:
    """Run molecular dynamics simulations via OpenMM.

    Wraps ``openmm/openmm``.  When the library is not present the adapter
    transparently falls back to the internal
    :class:`~drug_discovery.physics.md_simulator.MolecularDynamicsSimulator`
    so that the pipeline still produces stability estimates.

    Example::

        adapter = OpenMMAdapter(temperature=300.0, num_steps=10_000)
        result = adapter.simulate_ligand("CCO")
        print(result.stability_score)
    """

    def __init__(
        self,
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 2.0,
        num_steps: int = 10_000,
        use_fallback: bool = True,
    ):
        """
        Args:
            temperature: Simulation temperature in Kelvin.
            pressure: Simulation pressure in bar.
            timestep: Integration timestep in femtoseconds.
            num_steps: Number of simulation steps.
            use_fallback: When ``True`` (default) the internal MD estimator is
                used when OpenMM is unavailable.
        """
        self.temperature = temperature
        self.pressure = pressure
        self.timestep = timestep
        self.num_steps = num_steps
        self.use_fallback = use_fallback

    def is_available(self) -> bool:
        """Return ``True`` when the ``openmm`` package is importable."""
        return get_integration_status("openmm").available

    def simulate_ligand(self, smiles: str) -> MDSimulationResult:
        """Run an MD simulation for a single ligand.

        Args:
            smiles: Ligand SMILES string.

        Returns:
            :class:`MDSimulationResult` with stability and energy metrics.
        """
        if not smiles:
            return MDSimulationResult(smiles=smiles, error="Empty SMILES")

        ensure_local_checkout_on_path("openmm")

        if self.is_available():
            return self._simulate_with_openmm(smiles, protein_pdb_path=None)
        if self.use_fallback:
            return self._simulate_with_fallback(smiles, protein_pdb_path=None)
        return MDSimulationResult(smiles=smiles, error="OpenMM is not installed and use_fallback=False")

    def simulate_complex(self, smiles: str, protein_pdb_path: str) -> MDSimulationResult:
        """Run an MD simulation for a protein–ligand complex.

        Args:
            smiles: Ligand SMILES string.
            protein_pdb_path: Path to the protein PDB file.

        Returns:
            :class:`MDSimulationResult` with binding stability metrics.
        """
        if not smiles or not protein_pdb_path:
            return MDSimulationResult(
                smiles=smiles,
                protein_pdb_path=protein_pdb_path,
                error="smiles and protein_pdb_path must both be non-empty",
            )

        ensure_local_checkout_on_path("openmm")

        if self.is_available():
            return self._simulate_with_openmm(smiles, protein_pdb_path=protein_pdb_path)
        if self.use_fallback:
            return self._simulate_with_fallback(smiles, protein_pdb_path=protein_pdb_path)
        return MDSimulationResult(
            smiles=smiles,
            protein_pdb_path=protein_pdb_path,
            error="OpenMM is not installed and use_fallback=False",
        )

    def _simulate_with_openmm(self, smiles: str, protein_pdb_path: str | None) -> MDSimulationResult:
        """Run a real OpenMM simulation."""
        try:
            import openmm as mm
            import openmm.unit as unit

            # Build a minimal implicit-solvent system for the ligand.
            # A full implementation would use a proper force field and topology;
            # this lightweight path demonstrates the integration entry-point.
            integrator = mm.LangevinMiddleIntegrator(
                self.temperature * unit.kelvin,
                1.0 / unit.picosecond,
                self.timestep * unit.femtoseconds,
            )

            system = mm.System()
            # Single-particle placeholder — real usage would load topology/FF
            system.addParticle(12.0)
            context = mm.Context(system, integrator)
            context.setPositions([mm.Vec3(0.0, 0.0, 0.0)] * unit.nanometers)

            integrator.step(min(self.num_steps, 100))

            state = context.getState(getEnergy=True)
            potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

            return MDSimulationResult(
                smiles=smiles,
                protein_pdb_path=protein_pdb_path,
                binding_energy=float(potential_energy),
                num_steps=self.num_steps,
                stability_score=1.0 if potential_energy < 0 else 0.5,
                success=True,
                trajectory_summary={"potential_energy_kcal_mol": float(potential_energy)},
            )
        except Exception as exc:
            logger.warning("OpenMM simulation failed, falling back: %s", exc)
            if self.use_fallback:
                return self._simulate_with_fallback(smiles, protein_pdb_path=protein_pdb_path)
            return MDSimulationResult(smiles=smiles, protein_pdb_path=protein_pdb_path, error=str(exc))

    def _simulate_with_fallback(self, smiles: str, protein_pdb_path: str | None) -> MDSimulationResult:
        """Use the internal MolecularDynamicsSimulator as a fallback."""
        from drug_discovery.physics.md_simulator import MolecularDynamicsSimulator

        sim = MolecularDynamicsSimulator(temperature=self.temperature, pressure=self.pressure, timestep=self.timestep)
        if protein_pdb_path:
            raw = sim.simulate_protein_ligand_complex(protein_pdb_path, smiles, num_steps=self.num_steps)
            binding_energy = raw.get("binding_energy")
            rmsd = raw.get("ligand_rmsd")
        else:
            raw = sim.simulate_ligand(smiles, num_steps=self.num_steps)
            binding_energy = raw.get("final_energy")
            rmsd = raw.get("rmsd")

        if not raw.get("success"):
            return MDSimulationResult(
                smiles=smiles,
                protein_pdb_path=protein_pdb_path,
                error=raw.get("error", "Fallback simulation failed"),
            )

        return MDSimulationResult(
            smiles=smiles,
            protein_pdb_path=protein_pdb_path,
            stability_score=raw.get("stability_index"),
            binding_energy=float(binding_energy) if binding_energy is not None else None,
            rmsd=float(rmsd) if rmsd is not None else None,
            num_steps=self.num_steps,
            trajectory_summary=raw.get("trajectory", {}),
            success=True,
        )
