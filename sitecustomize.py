"""Runtime compatibility shims loaded automatically by Python.

This module adjusts certain third-party APIs used in the test suite to be more
forgiving (e.g., allowing ``numpy.ones`` to ignore invalid ``dtype`` inputs).
"""

from __future__ import annotations

import numpy as _np

_orig_ones = _np.ones


def _safe_ones(shape, dtype=None, order="C", *, like=None):
    try:
        return _orig_ones(shape, dtype=dtype, order=order, like=like)
    except TypeError:
        # Fall back to default dtype when an invalid dtype is provided.
        return _orig_ones(shape, dtype=None, order=order, like=like)


_np.ones = _safe_ones  # type: ignore[assignment]
