"""RDKit convenience helpers with safe fallbacks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drug_discovery.utils.rdkit_fallback import get_props_with_rdkit, rdkit_or_none

Chem, Descriptors, Crippen, _ = rdkit_or_none()
if Chem is not None:
    from rdkit.Chem import QED  # type: ignore
else:  # pragma: no cover - fallback mode
    QED = None


def smiles_to_mols(smiles_list: list[str]) -> list[object]:
    """Convert SMILES strings to molecule objects, dropping invalid entries."""
    if Chem is None:
        return [s for s in smiles_list if isinstance(s, str) and s.strip()]

    mols = []
    for smiles in smiles_list:
        if not isinstance(smiles, str):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
    return mols


def compute_descriptors(mols: list[object]) -> pd.DataFrame:
    """Compute a compact descriptor set used by the screening modules."""
    rows: list[dict[str, float]] = []
    for mol in mols:
        if Chem is not None and Descriptors is not None and Crippen is not None and QED is not None and hasattr(mol, "GetNumAtoms"):
            rows.append(
                {
                    "mol_wt": float(Descriptors.MolWt(mol)),
                    "logp": float(Crippen.MolLogP(mol)),
                    "h_donors": float(Descriptors.NumHDonors(mol)),
                    "h_acceptors": float(Descriptors.NumHAcceptors(mol)),
                    "tpsa": float(Descriptors.TPSA(mol)),
                    "qed": float(QED.qed(mol)),
                }
            )
        else:
            props = get_props_with_rdkit(str(mol))
            rows.append(
                {
                    "mol_wt": float(props.molecular_weight),
                    "logp": float(props.logp),
                    "h_donors": float(props.h_donors),
                    "h_acceptors": float(props.h_acceptors),
                    "tpsa": float(props.tpsa),
                    "qed": 0.5,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["mol_wt", "logp", "h_donors", "h_acceptors", "tpsa", "qed"])
    return pd.DataFrame(rows)


def smiles_to_sdf(smiles_list: list[str], path: str) -> str:
    """Write SMILES to an SDF file when RDKit is available; fallback to plain text."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if Chem is None:
        with output.open("w", encoding="utf-8") as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")
        return str(output)

    writer = Chem.SDWriter(str(output))
    try:
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                writer.write(mol)
    finally:
        writer.close()
    return str(output)
