"""Lightweight helpers when RDKit is unavailable.

These utilities provide deterministic, low-fidelity heuristics so that
high-level workflows and tests can run in environments where RDKit wheels
are not installed (for example, slim CI containers). When RDKit is present,
the real implementations should be used instead.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly when RDKit is installed
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Crippen, Descriptors, rd_mol_descriptors  # type: ignore

    HAS_RDKIT = True
except Exception:  # pragma: no cover - default path in constrained environments
    Chem = None
    Crippen = None
    Descriptors = None
    rd_mol_descriptors = None
    HAS_RDKIT = False


def simple_inchikey(smiles: str) -> str:
    """Deterministic, collision-resistant hash as a stand-in for InChIKey."""
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest().upper()
    # Keep the formatting vaguely similar to a real InChIKey (14-10-1 blocks)
    return f"{digest[:14]}-{digest[14:24]}-{digest[24:25]}"


@dataclass
class HeuristicProperties:
    molecular_weight: float
    logp: float
    h_donors: float
    h_acceptors: float
    rotatable_bonds: float
    tpsa: float
    aromatic_rings: float


def _count_token(smiles: str, token: str) -> int:
    count = 0
    i = 0
    while i < len(smiles):
        if smiles.startswith(token, i):
            count += 1
            i += len(token)
        else:
            i += 1
    return count


def heuristic_props(smiles: str) -> HeuristicProperties:
    """Approximate basic molecular properties without RDKit."""
    s = smiles or ""
    length = max(len(s), 1)
    # Rough atom counts based on token frequency
    c = _count_token(s, "C")
    n = _count_token(s, "N")
    o = _count_token(s, "O")
    s_count = _count_token(s, "S")
    f = _count_token(s, "F")
    cl = _count_token(s, "Cl")
    br = _count_token(s, "Br")
    i = _count_token(s, "I")

    # Crude molecular weight estimate
    weight = 12.0 * c + 14.0 * n + 16.0 * o + 32.0 * s_count + 19.0 * f + 35.5 * cl + 79.9 * br + 126.9 * i
    if weight <= 0:
        weight = 5.0 * length + 50.0

    hetero = n + o + s_count + f + cl + br + i
    logp = max(-1.0, min(6.0, (c - hetero * 0.5) * 0.25))
    donors = float(max(0, n + o - 1))
    acceptors = float(max(0, n * 2 + o))
    rotatable = float(max(0, length // 6))
    tpsa = float(min(200.0, hetero * 12.0 + rotatable * 2.0))
    aromatic_rings = float(max(0, s.lower().count("c") // 6))

    return HeuristicProperties(
        molecular_weight=float(weight),
        logp=float(logp),
        h_donors=donors,
        h_acceptors=acceptors,
        rotatable_bonds=rotatable,
        tpsa=tpsa,
        aromatic_rings=aromatic_rings,
    )


def is_smiles_plausible(smiles: str) -> bool:
    """Quick, dependency-free plausibility check."""
    if not smiles or not isinstance(smiles, str):
        return False
    stripped = smiles.strip()
    if not stripped or " " in stripped:
        return False
    return any(ch.isalpha() for ch in stripped)


def rdkit_or_none():
    """Return RDKit modules when available."""
    return Chem, Descriptors, Crippen, rd_mol_descriptors


def get_props_with_rdkit(smiles: str) -> HeuristicProperties | None:
    """Use RDKit when available, otherwise heuristics."""
    if HAS_RDKIT and Chem and Descriptors and Crippen:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return HeuristicProperties(
                molecular_weight=float(Descriptors.MolWt(mol)),
                logp=float(Crippen.MolLogP(mol)),
                h_donors=float(Descriptors.NumHDonors(mol)),
                h_acceptors=float(Descriptors.NumHAcceptors(mol)),
                rotatable_bonds=float(Descriptors.NumRotatableBonds(mol)),
                tpsa=float(Descriptors.TPSA(mol)),
                aromatic_rings=float(Descriptors.NumAromaticRings(mol)),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("RDKit property computation failed: %s", exc)
    return heuristic_props(smiles)
