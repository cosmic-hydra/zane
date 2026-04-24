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


def gt4sd_properties(
    smiles: str, properties: tuple[str, ...] = ("qed", "logp", "molecular_weight", "tpsa")
) -> dict[str, float]:
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


def predict_reaction_outcome(reactants_smiles: str, beam_size: int = 5) -> dict[str, Any]:
    """Predict reaction products using MolecularTransformer when available.

    Args:
        reactants_smiles: Reactant SMILES (dot-separated for multiple reactants).
        beam_size: Beam width for the transformer search.

    Returns:
        Dictionary with ``predictions`` (list of SMILES strings), ``available`` flag,
        and optional ``error`` string.
    """
    ensure_local_checkout_on_path("molecular_transformer")

    try:
        import molecular_transformer as mt

        predictions = mt.predict(reactants_smiles, beam_size=beam_size)
        return {"available": True, "predictions": list(predictions), "error": None}
    except Exception as exc:
        return {"available": False, "predictions": [], "error": str(exc)}


def diffdock_predict_binding(ligand_smiles: str, protein_pdb_path: str, num_poses: int = 10) -> dict[str, Any]:
    """Predict ligand binding poses using DiffDock when available.

    Args:
        ligand_smiles: Ligand SMILES string.
        protein_pdb_path: Path to the target protein PDB file.
        num_poses: Number of docking poses to generate.

    Returns:
        Dictionary with ``poses`` list, ``available`` flag, and optional ``error``.
    """
    ensure_local_checkout_on_path("diffdock")

    try:
        import diffdock

        poses = diffdock.dock(ligand_smiles, protein_pdb_path, num_poses=num_poses)
        return {"available": True, "poses": list(poses), "error": None}
    except Exception as exc:
        return {"available": False, "poses": [], "error": str(exc)}


def torchdrug_score_properties(smiles: str, tasks: tuple[str, ...] = ("tox21",)) -> dict[str, Any]:
    """Score molecular properties using TorchDrug GNN models when available.

    Args:
        smiles: Molecule SMILES string.
        tasks: Tuple of TorchDrug task names to run (e.g. ``"tox21"``, ``"bace"``).

    Returns:
        Dictionary mapping task names to predicted scores, plus ``available`` and
        optional ``error`` keys.
    """
    ensure_local_checkout_on_path("torchdrug")

    try:
        import torchdrug  # noqa: F401
        from torchdrug import data as td_data
        from torchdrug import models
        from torchdrug import tasks as td_tasks

        mol = td_data.Molecule.from_smiles(smiles)
        scores: dict[str, Any] = {}
        for task_name in tasks:
            try:
                task = getattr(td_tasks, task_name, None)
                if task is None:
                    scores[task_name] = None
                    continue
                model = models.GIN(input_dim=td_data.feature.atom_default.feature_dim, hidden_dims=[256, 256])
                result = task(model, mol)
                scores[task_name] = float(result) if result is not None else None
            except Exception as task_exc:
                scores[task_name] = f"error: {task_exc}"
        return {"available": True, "scores": scores, "error": None}
    except Exception as exc:
        return {"available": False, "scores": {}, "error": str(exc)}


def openfold_predict_structure(sequence: str) -> dict[str, Any]:
    """Predict protein 3D structure using OpenFold when available.

    Args:
        sequence: Amino-acid sequence string (single-letter codes).

    Returns:
        Dictionary with ``pdb_string`` (predicted structure as PDB text),
        ``available`` flag, and optional ``error``.
    """
    ensure_local_checkout_on_path("openfold")

    try:
        from openfold.np import protein as of_protein
        from openfold.utils.script_utils import run_model

        pdb_string = run_model(sequence)
        if not isinstance(pdb_string, str):
            pdb_string = of_protein.to_pdb(pdb_string)
        return {"available": True, "pdb_string": pdb_string, "error": None}
    except Exception as exc:
        return {"available": False, "pdb_string": None, "error": str(exc)}


def pistachio_load_reactions(dataset_path: str, limit: int = 1000) -> dict[str, Any]:
    """Load reaction data from a Pistachio dataset file when available.

    Args:
        dataset_path: Path to the Pistachio reaction dataset file.
        limit: Maximum number of reactions to return.

    Returns:
        Dictionary with ``reactions`` list (each entry is a dict with ``reactants``
        and ``products`` keys), ``count``, ``available`` flag, and optional ``error``.
    """
    ensure_local_checkout_on_path("pistachio")

    try:
        import pistachio

        reactions_iter = pistachio.load(dataset_path)
        reactions = []
        for i, rxn in enumerate(reactions_iter):
            if i >= limit:
                break
            reactions.append(
                {
                    "reactants": getattr(rxn, "reactants", str(rxn)),
                    "products": getattr(rxn, "products", None),
                }
            )
        return {"available": True, "reactions": reactions, "count": len(reactions), "error": None}
    except Exception as exc:
        return {"available": False, "reactions": [], "count": 0, "error": str(exc)}
