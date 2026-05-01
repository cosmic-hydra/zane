"""Utilities for ABFE / relative free-energy residual analysis.

Provides residual computations and outlier identification to track simulation
convergence and identify problematic systems.
"""
from __future__ import annotations

from typing import Sequence, List, Dict, Any
import math


def compute_residuals(predicted: Sequence[float], observed: Sequence[float]) -> List[float]:
    if len(predicted) != len(observed):
        raise ValueError("predicted and observed length mismatch")
    return [p - o for p, o in zip(predicted, observed)]


def rmse(residuals: Sequence[float]) -> float:
    if not residuals:
        return 0.0
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


def z_scores(residuals: Sequence[float]) -> List[float]:
    n = len(residuals)
    if n == 0:
        return []
    mean = sum(residuals) / n
    var = sum((r - mean) ** 2 for r in residuals) / n
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0.0:
        return [0.0 for _ in residuals]
    return [(r - mean) / sd for r in residuals]


def identify_outliers(residuals: Sequence[float], threshold_z: float = 2.5) -> List[int]:
    zs = z_scores(residuals)
    return [i for i, z in enumerate(zs) if abs(z) >= threshold_z]


def summarize_abfe(predicted: Sequence[float], observed: Sequence[float], top_n: int = 5) -> Dict[str, Any]:
    res = compute_residuals(predicted, observed)
    summary = {
        "n": len(res),
        "rmse": rmse(res),
        "mean_residual": (sum(res) / len(res)) if res else 0.0,
        "max_abs_residual": max((abs(x) for x in res), default=0.0),
        "outlier_indices": identify_outliers(res),
        "top_n_worst": sorted(range(len(res)), key=lambda i: abs(res[i]), reverse=True)[:top_n],
    }
    return summary
