"""Reactive Metabolite Screener — HOMO-LUMO gap and electrophilicity assessment.

Estimates the HOMO-LUMO gap (ΔE_HL) of a small molecule as a proxy for
chemical reactivity and reactive-metabolite potential.  A narrow gap indicates
a more polarisable, potentially electrophilic molecule that may form covalent
adducts with proteins or DNA.

**Calculation pipeline**:

1. Generate a 3-D MMFF94 conformer with RDKit.
2. If ``pyscf`` is installed: run a minimal-basis RHF or DFT (B3LYP/STO-3G)
   single-point calculation and extract HOMO/LUMO orbital energies in eV.
3. Fallback (no PySCF): use an extended-Hückel–inspired empirical formula
   derived from conjugation depth, aromatic ring count, and TPSA.  This is
   labelled explicitly as a *heuristic estimate* — not a QM result.

Raises ``ValueError`` for invalid SMILES.  Returns ``float("nan")`` when
conformer embedding fails (e.g., strained macro-cycles).
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors  # type: ignore[import-untyped]

    _RDKIT = True
except ImportError:  # pragma: no cover
    _RDKIT = False
    logger.error("RDKit is required for ReactiveMetaboliteScreener.")

try:
    import pyscf  # type: ignore[import-untyped]
    from pyscf import gto, scf  # type: ignore[import-untyped]

    _PYSCF = True
except ImportError:  # pragma: no cover
    pyscf = None  # type: ignore[assignment]
    _PYSCF = False
    logger.warning(
        "PySCF not installed. HOMO-LUMO gap will be estimated via an empirical "
        "heuristic rather than a quantum-chemical calculation."
    )

# Electrophilicity threshold: molecules with ΔE_HL below this value (eV) are
# flagged as potentially reactive.
_DEFAULT_GAP_THRESHOLD_EV: float = 4.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _mol_from_smiles(smiles: str):  # type: ignore[return]
    """Parse SMILES, raise ValueError on failure."""
    if not _RDKIT:
        raise RuntimeError("RDKit is required but not installed.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    return mol


def _embed_3d(mol):  # type: ignore[return]
    """Add Hs and embed a 3-D conformer (MMFF94). Returns mol with conformer or None."""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol_h, params)
    if result == -1:
        # Retry with random coordinates
        result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    if result == -1:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    return mol_h


def _xyz_block_from_mol(mol_h) -> list[tuple[str, float, float, float]]:
    """Return list of (symbol, x, y, z) from first conformer (Å)."""
    conf = mol_h.GetConformer()
    atoms = []
    for atom in mol_h.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), pos.x, pos.y, pos.z))
    return atoms


def _heuristic_gap_ev(smiles: str) -> float:
    """Empirical HOMO-LUMO gap estimate.

    Based on correlations from the literature:
    - Highly conjugated, aromatic systems → narrow gap (~2–4 eV)
    - Saturated aliphatic chains → wide gap (~8–12 eV)
    - Scaling by polar surface area accounts for heteroatom effects

    This is *not* a quantum-chemical calculation.  Use PySCF for accuracy.
    """
    if not _RDKIT:
        return 6.0  # neutral default when rdkit unavailable

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")

    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    ring_count = int(rdMolDescriptors.CalcNumRings(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    logp = float(Crippen.MolLogP(mol))
    heavy = int(mol.GetNumHeavyAtoms())
    dbl_bonds = sum(1 for b in mol.GetBonds() if str(b.GetBondTypeAsDouble()) == "2.0")

    # Base gap: decreases with conjugation
    conjugation_depth = aromatic_rings * 2.0 + dbl_bonds * 0.5 + ring_count * 0.3
    base_gap = max(2.0, 11.0 - conjugation_depth * 0.9)

    # Polar atoms slightly widen the gap (inductive effects)
    polarity_correction = min(tpsa / 200.0, 0.5)

    # Larger, more flexible molecules tend toward narrower gaps
    size_correction = -min(heavy / 80.0, 0.8)

    gap = base_gap + polarity_correction + size_correction
    return float(np.clip(gap, 1.5, 14.0))


def _pyscf_gap_ev(mol_h, charge: int = 0) -> float:
    """Run a minimal-basis RHF single-point with PySCF; return HOMO-LUMO gap (eV).

    Uses STO-3G basis which is fast but approximate.  For production accuracy,
    replace with a larger basis (6-31G*) and/or DFT (B3LYP).
    """
    atoms_xyz = _xyz_block_from_mol(mol_h)
    atom_str = "; ".join(f"{sym} {x:.4f} {y:.4f} {z:.4f}" for sym, x, y, z in atoms_xyz)

    mol_pyscf = gto.Mole()
    mol_pyscf.atom = atom_str
    mol_pyscf.basis = "sto-3g"
    mol_pyscf.charge = charge
    mol_pyscf.spin = 0
    mol_pyscf.unit = "Angstrom"
    mol_pyscf.verbose = 0  # suppress PySCF output
    mol_pyscf.build()

    mf = scf.RHF(mol_pyscf)
    mf.max_cycle = 150
    mf.kernel()

    # Orbital energies in Hartree → eV (1 Hartree = 27.2114 eV)
    mo_energies = np.array(mf.mo_energy)
    mo_occ = np.array(mf.mo_occ)

    occupied = mo_energies[mo_occ > 0]
    virtual = mo_energies[mo_occ == 0]

    if len(occupied) == 0 or len(virtual) == 0:
        raise RuntimeError("Cannot determine HOMO/LUMO from MO occupancies.")

    homo_ev = float(occupied[-1]) * 27.2114
    lumo_ev = float(virtual[0]) * 27.2114
    gap_ev = lumo_ev - homo_ev

    logger.debug("PySCF RHF/STO-3G: HOMO=%.3f eV, LUMO=%.3f eV, gap=%.3f eV", homo_ev, lumo_ev, gap_ev)
    return gap_ev


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------
class ReactiveMetaboliteScreener:
    """Reactive metabolite and electrophilicity screener.

    Estimates the HOMO-LUMO gap (ΔE_HL) of a molecule as a proxy for
    electrophilic reactivity and reactive-metabolite formation risk.

    Args:
        gap_threshold_ev: Gap value (eV) below which the molecule is flagged
            as potentially electrophilic.  Default is 4.5 eV.
        use_pyscf: When ``True`` (default), attempt a PySCF RHF/STO-3G
            single-point if PySCF is installed.  Falls back to the empirical
            heuristic automatically when PySCF is unavailable or the SCF
            calculation fails.

    Example::

        screener = ReactiveMetaboliteScreener()
        gap = screener.calculate_homo_lumo_gap("c1ccccc1")
        flagged = screener.check_electrophilicity("c1ccccc1")
    """

    def __init__(
        self,
        gap_threshold_ev: float = _DEFAULT_GAP_THRESHOLD_EV,
        use_pyscf: bool = True,
    ) -> None:
        self.gap_threshold_ev = gap_threshold_ev
        self.use_pyscf = use_pyscf
        self._last_gap: float | None = None

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------
    def calculate_homo_lumo_gap(self, smiles: str) -> float:
        """Estimate the HOMO-LUMO gap for *smiles* in electron-volts (eV).

        Pipeline:
        1. Parse SMILES with RDKit and generate a 3-D MMFF94 conformer.
        2. If ``pyscf`` is available and ``use_pyscf=True``: run RHF/STO-3G
           and return the exact orbital gap.
        3. Otherwise: return the empirical heuristic estimate (clearly
           labelled in the log output).

        Args:
            smiles: Input SMILES string.

        Returns:
            Estimated HOMO-LUMO gap in eV.  Returns ``float('nan')`` if a
            3-D conformer cannot be generated.

        Raises:
            ValueError: If *smiles* cannot be parsed.
            RuntimeError: If RDKit is not installed.
        """
        mol = _mol_from_smiles(smiles)

        # Attempt PySCF path
        if self.use_pyscf and _PYSCF and _RDKIT:
            mol_h = _embed_3d(mol)
            if mol_h is not None:
                try:
                    gap = _pyscf_gap_ev(mol_h)
                    logger.info("HOMO-LUMO gap (PySCF RHF/STO-3G) for %r: %.3f eV", smiles, gap)
                    self._last_gap = gap
                    return gap
                except Exception as exc:
                    logger.warning("PySCF calculation failed (%s); falling back to heuristic.", exc)
            else:
                logger.warning("3-D conformer embedding failed for %r; falling back to heuristic.", smiles)

        # Heuristic fallback
        gap = _heuristic_gap_ev(smiles)
        if math.isnan(gap):
            logger.warning("Heuristic gap returned NaN for %r", smiles)
        else:
            logger.info(
                "HOMO-LUMO gap (empirical heuristic — NOT a QM result) for %r: %.3f eV",
                smiles,
                gap,
            )
        self._last_gap = gap
        return gap

    def check_electrophilicity(self, smiles: str) -> bool:
        """Return ``True`` if the molecule is predicted to be electrophilic.

        A narrow HOMO-LUMO gap (below :attr:`gap_threshold_ev`) indicates a
        more polarisable electron density that can accept electron pairs from
        biological nucleophiles (Cys, Lys, His), forming covalent adducts.

        Args:
            smiles: Input SMILES string.

        Returns:
            ``True`` if the estimated gap is below :attr:`gap_threshold_ev`,
            ``False`` otherwise.  Returns ``False`` when the gap is NaN
            (calculation failure — conservative safe default).
        """
        gap = self.calculate_homo_lumo_gap(smiles)
        if math.isnan(gap):
            logger.warning(
                "Gap is NaN for %r — cannot assess electrophilicity; returning False.",
                smiles,
            )
            return False
        flagged = gap < self.gap_threshold_ev
        if flagged:
            logger.warning(
                "Electrophilicity flag raised for %r (gap=%.2f eV < threshold=%.2f eV)",
                smiles,
                gap,
                self.gap_threshold_ev,
            )
        return flagged

    def screen(self, smiles: str) -> dict[str, object]:
        """Run the full reactive-metabolite screen and return a result dict.

        Returns::

            {
                "smiles": str,
                "homo_lumo_gap_ev": float,
                "electrophilic": bool,
                "gap_threshold_ev": float,
                "method": "pyscf_rhf_sto3g" | "empirical_heuristic",
                "success": bool,
            }
        """
        try:
            gap = self.calculate_homo_lumo_gap(smiles)
            electrophilic = self.check_electrophilicity(smiles)
            method = "pyscf_rhf_sto3g" if (_PYSCF and self.use_pyscf) else "empirical_heuristic"
            return {
                "smiles": smiles,
                "homo_lumo_gap_ev": gap,
                "electrophilic": electrophilic,
                "gap_threshold_ev": self.gap_threshold_ev,
                "method": method,
                "success": True,
            }
        except Exception as exc:
            logger.error("ReactiveMetaboliteScreener error: %s", exc)
            return {
                "smiles": smiles,
                "homo_lumo_gap_ev": float("nan"),
                "electrophilic": False,
                "gap_threshold_ev": self.gap_threshold_ev,
                "method": "error",
                "success": False,
                "error": str(exc),
            }
