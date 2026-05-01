from __future__ import annotations

from dataclasses import dataclass, field
import re

@dataclass
class SwissADMEResult:
    smiles: str
    admet_profile: dict[str, float | int | bool]
    violations: int = 0
    developable: bool = True
    rule_checks: dict[str, bool] = field(default_factory=dict)
    source: str = "swissadme_proxy"
    notes: list[str] = field(default_factory=list)


_ATOM_PATTERN = re.compile(r"Br|Cl|[A-Z][a-z]?|[cnosp]")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _estimate_from_smiles(smiles: str) -> dict[str, float | int | bool]:
    atom_tokens = _ATOM_PATTERN.findall(smiles)
    atom_count = len(atom_tokens)
    carbon_like = sum(1 for atom in atom_tokens if atom in {"C", "c"})
    hetero_atoms = sum(1 for atom in atom_tokens if atom not in {"C", "c", "H"})
    hetero_polar = sum(1 for atom in atom_tokens if atom in {"N", "O", "S", "P", "n", "o", "s", "p"})
    ring_closures = sum(ch.isdigit() for ch in smiles)
    double_bonds = smiles.count("=")
    triple_bonds = smiles.count("#")
    halogens = sum(smiles.count(atom) for atom in ("F", "Cl", "Br", "I"))

    molecular_weight = float(max(0.0, 12.0 * carbon_like + 14.0 * hetero_atoms + 19.0 * halogens + 1.0 * smiles.count("H")))
    logp = round(0.54 * carbon_like - 1.35 * hetero_polar - 0.08 * ring_closures + 0.12 * double_bonds + 0.16 * triple_bonds, 2)
    hbd = int(min(10, hetero_polar))
    hba = int(min(15, hetero_polar + halogens))
    tpsa = round(11.0 * hetero_polar + 2.0 * ring_closures + 1.5 * double_bonds, 1)
    rotatable_bonds = int(max(0, atom_count - ring_closures - double_bonds - triple_bonds - 2) // 2)

    lipinski_violations = int(molecular_weight > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    veber_ok = rotatable_bonds <= 10 and tpsa <= 140
    lipinski_ok = lipinski_violations <= 1
    bbb_permeant = molecular_weight <= 450 and tpsa <= 90 and logp <= 4.5
    gi_absorption = max(0.0, min(1.0, 0.92 - 0.35 * _safe_div(tpsa, 140.0) - 0.06 * max(0.0, logp - 3.0)))
    cyp_inhib = max(0.0, min(1.0, 0.15 + 0.12 * max(0.0, logp) + 0.05 * max(0, halogens - 1)))

    return {
        "atom_count": atom_count,
        "molecular_weight": round(molecular_weight, 2),
        "logP": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotatable_bonds": rotatable_bonds,
        "ring_closures": ring_closures,
        "gi_absorption": round(gi_absorption, 3),
        "bbb_permeant": bbb_permeant,
        "cyp_inhib": round(cyp_inhib, 3),
        "lipinski_ok": lipinski_ok,
        "veber_ok": veber_ok,
    }


class SwissADMEProxy:
    """SwissADME-style screening with a dependency-light fallback.

    The proxy prefers RDKit descriptors when available, but it always
    returns a deterministic, rule-based profile so the repo can be used in
    slim environments.
    """

    def __init__(self, use_rdkit: bool = True):
        self.use_rdkit = use_rdkit

    def _rdkit_profile(self, smiles: str) -> dict[str, float | int | bool] | None:
        if not self.use_rdkit:
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
        except Exception:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        molecular_weight = float(Descriptors.MolWt(mol))
        logp = round(float(Crippen.MolLogP(mol)), 2)
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        tpsa = round(float(rdMolDescriptors.CalcTPSA(mol)), 1)
        rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))
        ring_closures = int(rdMolDescriptors.CalcNumRings(mol))
        lipinski_violations = int(molecular_weight > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)

        return {
            "atom_count": int(mol.GetNumAtoms()),
            "molecular_weight": round(molecular_weight, 2),
            "logP": logp,
            "hbd": hbd,
            "hba": hba,
            "tpsa": tpsa,
            "rotatable_bonds": rotatable_bonds,
            "ring_closures": ring_closures,
            "gi_absorption": round(max(0.0, min(1.0, 0.92 - 0.35 * _safe_div(tpsa, 140.0) - 0.06 * max(0.0, logp - 3.0))), 3),
            "bbb_permeant": molecular_weight <= 450 and tpsa <= 90 and logp <= 4.5,
            "cyp_inhib": round(max(0.0, min(1.0, 0.15 + 0.12 * max(0.0, logp) + 0.05 * max(0, sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in {"Cl", "Br", "I", "F"}) - 1)))),
            "lipinski_ok": lipinski_violations <= 1,
            "veber_ok": rotatable_bonds <= 10 and tpsa <= 140,
        }

    def predict(self, smiles: str) -> SwissADMEResult:
        clean_smiles = (smiles or "").strip()
        if not clean_smiles:
            raise ValueError("smiles cannot be empty")

        profile = self._rdkit_profile(clean_smiles) or _estimate_from_smiles(clean_smiles)
        lipinski_ok = bool(profile.get("lipinski_ok", True))
        veber_ok = bool(profile.get("veber_ok", True))
        gi_absorption = float(profile.get("gi_absorption", 0.0))
        bbb_permeant = bool(profile.get("bbb_permeant", False))
        cyp_inhib = float(profile.get("cyp_inhib", 0.0))

        rule_checks = {
            "lipinski": lipinski_ok,
            "veber": veber_ok,
            "high_gi_absorption": gi_absorption >= 0.7,
            "bbb_permeation": bbb_permeant,
            "low_cyp_inhibition": cyp_inhib <= 0.5,
        }
        violations = sum(1 for passed in rule_checks.values() if not passed)
        developable = violations <= 1

        notes = []
        if not lipinski_ok:
            notes.append("Lipinski rule-of-five exceeded")
        if not veber_ok:
            notes.append("Veber criteria not met")
        if not bbb_permeant:
            notes.append("Low BBB permeability expected")

        return SwissADMEResult(
            smiles=clean_smiles,
            admet_profile=profile,
            violations=violations,
            developable=developable,
            rule_checks=rule_checks,
            notes=notes,
        )