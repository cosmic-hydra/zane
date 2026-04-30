"""Minimal Data container compatible with code expectations."""

from __future__ import annotations

from typing import Any

import torch


class Data:
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, batch=None, **kwargs: Any):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.batch = batch if batch is not None else self._infer_batch(x)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _infer_batch(x):
        if x is None:
            return torch.tensor([], dtype=torch.long)
        n = x.shape[0]
        return torch.zeros(n, dtype=torch.long)

    def to(self, device):
        for attr in ["x", "edge_index", "edge_attr", "y", "batch"]:
            val = getattr(self, attr, None)
            if hasattr(val, "to"):
                setattr(self, attr, val.to(device))
        return self
