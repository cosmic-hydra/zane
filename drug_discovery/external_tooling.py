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
