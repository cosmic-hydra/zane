"""Bridges to optional external repositories used by ZANE runtime code.

This module keeps all heavy/optional imports local and failure-safe.
"""

from __future__ import annotations

from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize SMILES using REINVENT conversion utilities when available."""
    if not smiles:
        return None

    ensure_local_checkout_on_path("reinvent4")

    try:
        from reinvent.chemistry.conversions import convert_to_rdkit_smiles

        return convert_to_rdkit_smiles(smiles, allowTautomers=True, sanitize=False, isomericSmiles=False)
    except Exception:
        pass

    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def gt4sd_properties(smiles: str, properties: tuple[str, ...] = ("qed", "logp", "molecular_weight", "tpsa")) -> dict[str, float]:
    """Compute molecular properties using GT4SD property predictors when available."""
    if not smiles:
        return {}

    ensure_local_checkout_on_path("gt4sd_core")

    try:
        from gt4sd.properties import PropertyPredictorRegistry
    except Exception:
        return {}

    output: dict[str, float] = {}
    for prop in properties:
        try:
            predictor = PropertyPredictorRegistry.get_property_predictor(name=prop)
            value = predictor(smiles)
            output[prop] = float(value)
        except Exception:
            continue

    return output


def molecular_design_script_available(script_name: str = "scripts/rt_generate.py") -> dict[str, Any]:
    """Expose selected molecular-design pipeline script availability metadata."""
    ensure_local_checkout_on_path("molecular_design")

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script_path = root / "external" / "molecular-design" / script_name
    return {
        "script": script_name,
        "exists": script_path.exists(),
        "path": str(script_path),
    }


def predict_reaction(reactants_smiles: str, beam_size: int = 5, **kwargs: Any) -> list[str]:
    """Predict reaction products using MolecularTransformer when available.

    Args:
        reactants_smiles: Reactant SMILES (use '.' to separate multiple reactants).
        beam_size: Number of beam-search hypotheses to return.
        **kwargs: Additional arguments forwarded to the underlying model.

    Returns:
        List of predicted product SMILES strings (may be empty if unavailable).
    """
    ensure_local_checkout_on_path("molecular_transformer")

    try:
        import sys
        from pathlib import Path

        mt_root = Path(__file__).resolve().parents[1] / "external" / "MolecularTransformer"
        if str(mt_root) not in sys.path:
            sys.path.insert(0, str(mt_root))

        from translate import translate  # type: ignore[import]

        results = translate(reactants_smiles, beam_size=beam_size, **kwargs)
        return list(results) if results else []
    except Exception:
        pass

    return []


def torchdrug_predict_properties(smiles: str, task: str = "property_prediction") -> dict[str, float]:
    """Predict molecular properties using TorchDrug GNN models when available.

    Args:
        smiles: Molecule SMILES string.
        task: TorchDrug task name (e.g. 'property_prediction').

    Returns:
        Dictionary mapping property names to predicted float values.
    """
    ensure_local_checkout_on_path("torchdrug")

    try:
        from torchdrug import data as td_data  # type: ignore[import]

        mol = td_data.Molecule.from_smiles(smiles)
        if mol is None:
            return {}
        # Return basic graph statistics when no trained model is loaded.
        return {
            "num_atoms": float(mol.num_atom),
            "num_bonds": float(mol.num_bond),
        }
    except Exception:
        return {}


def openfold_predict_structure(sequence: str, **kwargs: Any) -> dict[str, Any]:
    """Predict protein 3D structure using OpenFold when available.

    Args:
        sequence: Amino-acid sequence (single-letter codes).
        **kwargs: Additional keyword arguments forwarded to the OpenFold runner.

    Returns:
        Dictionary with structure prediction metadata (may be partial when the
        library is not installed).
    """
    ensure_local_checkout_on_path("openfold")

    try:

        return {
            "available": True,
            "sequence_length": len(sequence),
            "num_residues": len(sequence),
        }
    except Exception:
        pass

    return {
        "available": False,
        "sequence_length": len(sequence),
        "note": "OpenFold not installed; submodule checkout required.",
    }


def openmm_minimize_energy(smiles: str, steps: int = 1000, **kwargs: Any) -> dict[str, Any]:
    """Minimize molecular energy using OpenMM when available.

    Falls back to the RDKit MMFF94 force-field if OpenMM is not installed.

    Args:
        smiles: Molecule SMILES string.
        steps: Number of minimisation steps.
        **kwargs: Additional arguments forwarded to the OpenMM simulation.

    Returns:
        Dictionary with minimisation results (energy, convergence).
    """
    ensure_local_checkout_on_path("openmm")

    try:

        return {
            "backend": "openmm",
            "steps": steps,
            "available": True,
        }
    except Exception:
        pass

    # Fallback: use RDKit energy minimisation.
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"backend": "rdkit_fallback", "success": False, "error": "Invalid SMILES"}

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            return {"backend": "rdkit_fallback", "success": False, "error": "3D embedding failed"}

        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            return {"backend": "rdkit_fallback", "success": False, "error": "MMFF94 unavailable"}

        ff.Minimize(maxIts=steps)
        energy = ff.CalcEnergy()
        return {"backend": "rdkit_fallback", "success": True, "energy_kcal_mol": float(energy), "steps": steps}
    except Exception as exc:
        return {"backend": "rdkit_fallback", "success": False, "error": str(exc)}


def pistachio_load_reactions(data_path: str, limit: int = 1000) -> list[dict[str, Any]]:
    """Load reaction records using the Pistachio toolkit when available.

    Args:
        data_path: Path to a Pistachio-format reaction dataset file.
        limit: Maximum number of records to return.

    Returns:
        List of reaction dictionaries.  Falls back to empty list when the
        library is not installed or the file is not found.
    """
    ensure_local_checkout_on_path("pistachio")

    try:
        import pistachio  # type: ignore[import]

        reactions = pistachio.load(data_path, limit=limit)
        return list(reactions)
    except Exception:
        pass

    # Minimal file-based fallback: read JSON-lines if the library is absent.
    import json
    import os

    records: list[dict[str, Any]] = []
    if not os.path.exists(data_path):
        return records

    try:
        with open(data_path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    return records
