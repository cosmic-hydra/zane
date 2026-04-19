"""
Torch extension-backed force-field + FEP routines.

The compiled extension is built lazily on first use. If compilation fails
or a GPU is unavailable, a vectorized PyTorch fallback is used so that
callers retain a working code path.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent
_EXT_NAME = f"zane_forcefield_{hashlib.sha1(str(_SRC_DIR).encode()).hexdigest()[:8]}"


def _has_cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _load_ext():
    sources = [str(_SRC_DIR / "forcefield.cpp")]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3"]
    with_cuda = _has_cuda()
    try:
        ext = load(
            name=_EXT_NAME,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            with_cuda=with_cuda,
            verbose=False,
        )
        logger.info("Loaded native force-field backend (cuda=%s)", with_cuda)
        return ext
    except Exception as exc:  # pragma: no cover - build system dependent
        logger.warning("Falling back to Torch implementation: %s", exc)
        return None


def _ensure_batch(coords: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 2:
        return coords.unsqueeze(0)
    if coords.dim() != 3:
        raise ValueError(f"coords must be (N,3) or (B,N,3); got {coords.shape}")
    return coords


def _torch_energy(coords: torch.Tensor, sigma: float = 3.4, epsilon: float = 0.2) -> torch.Tensor:
    coords = _ensure_batch(coords)
    dist = torch.cdist(coords, coords) + 1e-9
    mask = torch.eye(dist.shape[-1], device=coords.device, dtype=dist.dtype).unsqueeze(0)
    dist = dist + mask * 1e6
    inv = torch.pow((sigma / dist), 6)
    energy_mat = 4.0 * epsilon * (inv * inv - inv)
    energy = energy_mat.triu(1).sum(dim=(-1, -2))
    return energy


def _torch_forces(coords: torch.Tensor, sigma: float = 3.4, epsilon: float = 0.2) -> torch.Tensor:
    coords = _ensure_batch(coords)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
    dist2 = (diff * diff).sum(-1) + 1e-9
    n_atoms = diff.shape[1]
    mask = torch.eye(n_atoms, device=coords.device, dtype=dist2.dtype).unsqueeze(0)
    dist2 = dist2 + mask * 1e6

    inv2 = (sigma * sigma) / dist2
    inv6 = inv2 * inv2 * inv2
    coeff = 24.0 * epsilon * (2 * inv6 * inv6 - inv6) / dist2
    coeff = coeff * (1.0 - mask)
    forces = (coeff.unsqueeze(-1) * diff).sum(dim=2)
    return forces


def _torch_fep(
    ligand: torch.Tensor,
    protein: torch.Tensor,
    lambda_schedule: Optional[torch.Tensor] = None,
    sigma: float = 3.4,
    epsilon: float = 0.2,
) -> torch.Tensor:
    ligand = _ensure_batch(ligand)
    protein = _ensure_batch(protein)
    device = ligand.device
    lambdas = lambda_schedule if lambda_schedule is not None else torch.linspace(0.0, 1.0, steps=5, device=device)
    lambdas = lambdas.to(device=device, dtype=ligand.dtype)
    if lambdas.ndim == 0:
        lambdas = lambdas.unsqueeze(0)

    base_cross = torch.cdist(ligand, protein) + 1e-9
    inv = torch.pow((sigma / base_cross), 6)
    interaction = 4.0 * epsilon * (inv * inv - inv)  # (B, n_lig, n_prot)

    delta_f = torch.zeros((ligand.shape[0],), device=device, dtype=ligand.dtype)
    for i in range(len(lambdas) - 1):
        lam_left, lam_right = lambdas[i], lambdas[i + 1]
        lam_mid = 0.5 * (lam_left + lam_right)
        scaled = interaction * lam_mid
        slice_energy = scaled.sum(dim=(-1, -2))
        delta_f += (lam_right - lam_left) * slice_energy
    return delta_f


def compute_energy(
    coords: torch.Tensor,
    atom_types: Optional[torch.Tensor] = None,
    sigma: float = 3.4,
    epsilon: float = 0.2,
    reduce: bool = True,
) -> float | torch.Tensor:
    """
    Compute Lennard-Jones energy for batched coordinates.

    Args:
        coords: Coordinate tensor of shape ``(N, 3)`` or ``(B, N, 3)``.
        atom_types: Optional atom-type tensor (reserved for future mixing rules).
        sigma: Lennard-Jones sigma.
        epsilon: Lennard-Jones epsilon.
        reduce: If True, return mean energy as ``float``; otherwise return per-batch tensor.
    """
    backend = _load_ext()
    if backend is not None:
        energies = backend.compute_energy(coords, sigma, epsilon)
    else:
        energies = _torch_energy(coords, sigma=sigma, epsilon=epsilon)

    if reduce:
        return float(torch.mean(energies).item())
    return energies


def compute_forces(coords: torch.Tensor, sigma: float = 3.4, epsilon: float = 0.2) -> torch.Tensor:
    """
    Compute forces for each atom using the native backend, falling back to torch.
    """
    backend = _load_ext()
    if backend is not None:
        try:
            return backend.compute_forces(coords, sigma, epsilon)
        except Exception as exc:  # pragma: no cover - runtime backend error
            logger.warning("Native force computation failed, using torch fallback: %s", exc)
    return _torch_forces(coords, sigma=sigma, epsilon=epsilon)


def run_fep(
    ligand_coords: torch.Tensor,
    protein_coords: torch.Tensor,
    lambda_schedule: Optional[torch.Tensor] = None,
    sigma: float = 3.4,
    epsilon: float = 0.2,
    reduce: bool = True,
) -> float | torch.Tensor:
    """
    Estimate free energy perturbation for a ligand–protein pair.
    """
    backend = _load_ext()
    if backend is not None:
        try:
            delta_f = backend.run_fep(ligand_coords, protein_coords, lambda_schedule, sigma, epsilon)
        except Exception as exc:  # pragma: no cover
            logger.warning("Native FEP failed, falling back to torch: %s", exc)
            delta_f = _torch_fep(ligand_coords, protein_coords, lambda_schedule, sigma=sigma, epsilon=epsilon)
    else:
        delta_f = _torch_fep(ligand_coords, protein_coords, lambda_schedule, sigma=sigma, epsilon=epsilon)
    if reduce:
        return float(torch.mean(delta_f).item())
    return delta_f


def get_backend_status() -> dict:
    """Expose backend load status for health checks."""
    ext = _load_ext()
    return {
        "loaded": ext is not None,
        "cuda": _has_cuda(),
        "ext_name": _EXT_NAME,
        "source_dir": str(_SRC_DIR),
    }
