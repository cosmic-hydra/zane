"""Environmental stability tests: pH stability and blood plasma binding estimators.

These are lightweight heuristic/estimator modules useful for gating compounds
before heavy experimental assays. Intended to be used as part of QA pipelines.
"""
from __future__ import annotations

from typing import Dict, Any
import math


def estimate_ph_stability(smiles: str, ph: float) -> Dict[str, Any]:
    """Estimate percent remaining after 24h at given pH using heuristics.

    This is a heuristic fallback when experimental data is not available.
    Returns a dict with 'percent_remaining' and 'confidence'.
    """
    # Heuristic: acids degrade in high pH, bases degrade in low pH.
    score = 0.8
    if any(x in smiles for x in ['C(=O)O', 'CO2H', 'O=C(O)']):  # carboxylic acid moieties
        # acids less stable at high pH
        score -= max(0.0, (ph - 7.0) * 0.08)
    if any(x in smiles for x in ['N', 'NH', 'N(']):
        # basic amines less stable at low pH
        score -= max(0.0, (7.0 - ph) * 0.06)
    # clamp
    score = max(0.05, min(0.99, score))
    return {"percent_remaining": round(score * 100.0, 1), "confidence": 0.45}


def estimate_plasma_binding(smiles: str) -> Dict[str, Any]:
    """Estimate fraction bound to plasma proteins (heuristic).

    Returns {'fraction_bound': 0-1, 'confidence': 0-1}
    """
    # Heuristic: high logP and aromatic rings increase plasma binding
    aromatic = smiles.count('c') + smiles.count('C') // 4
    logp_est = 1.0 + aromatic * 0.4 - smiles.count('N') * 0.3
    # map to 0-1
    frac = 1.0 - 1.0 / (1.0 + math.exp((logp_est - 2.5)))
    frac = max(0.0, min(0.99, frac))
    confidence = 0.4
    return {"fraction_bound": round(frac, 3), "confidence": confidence}


def run_environmental_tests(smiles: str, ph_values: tuple[float, ...] = (1.2, 4.5, 7.4, 9.0)) -> Dict[str, Any]:
    results = {"smiles": smiles, "ph_profiles": {}, "plasma_binding": None}
    for ph in ph_values:
        results["ph_profiles"][str(ph)] = estimate_ph_stability(smiles, ph)
    results["plasma_binding"] = estimate_plasma_binding(smiles)
    return results
