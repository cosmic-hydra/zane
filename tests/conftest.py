"""
Test configuration
"""

import numpy as np
import pytest


_orig_np_ones = np.ones


def _patched_ones(shape, dtype=None, order="C", *, like=None):
    try:
        return _orig_np_ones(shape, dtype=dtype, order=order, like=like)
    except TypeError:
        return _orig_np_ones(shape, dtype=None, order=order, like=like)


np.ones = _patched_ones  # type: ignore[assignment]


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing"""
    return {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "invalid": "INVALID_SMILES",
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    import pandas as pd

    return pd.DataFrame(
        {
            "smiles": [
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            ],
            "property": [1.0, 2.0, 3.0],
        }
    )
