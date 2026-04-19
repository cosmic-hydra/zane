"""
Native force-field backend bridging C++/CUDA extensions into Python.

The module lazily builds and loads a torch C++ extension that implements
energy, force, and free-energy perturbation routines. Callers should use
``compute_energy``, ``compute_forces``, and ``run_fep`` from
``drug_discovery.native.backend`` instead of importing the extension
directly.
"""

from .backend import compute_energy, compute_forces, run_fep, get_backend_status

__all__ = ["compute_energy", "compute_forces", "run_fep", "get_backend_status"]
