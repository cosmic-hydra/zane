"""Thin wrapper around torch.utils.data.DataLoader to mirror torch_geometric API."""

from __future__ import annotations

import torch


class DataLoader(torch.utils.data.DataLoader):
    """Reuse PyTorch's DataLoader with identical signature."""

    pass
