"""Lightweight torch_geometric.nn stand-ins."""

from __future__ import annotations

import torch
import torch.nn as nn


def _scatter_reduce(x: torch.Tensor, batch: torch.Tensor, reduce: str) -> torch.Tensor:
    if batch.numel() == 0:
        return torch.zeros((0, x.shape[-1]), device=x.device, dtype=x.dtype)
    unique = torch.unique(batch)
    out = []
    for idx in unique:
        mask = batch == idx
        if not torch.any(mask):
            continue
        if reduce == "mean":
            out.append(x[mask].mean(dim=0))
        elif reduce == "max":
            out.append(x[mask].max(dim=0).values)
    return torch.stack(out, dim=0) if out else torch.zeros((0, x.shape[-1]), device=x.device, dtype=x.dtype)


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return _scatter_reduce(x, batch, reduce="mean")


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return _scatter_reduce(x, batch, reduce="max")


class MessagePassing(nn.Module):
    def __init__(self, aggr: str = "add"):
        super().__init__()
        self.aggr = aggr

    def propagate(self, edge_index: torch.Tensor, **kwargs):
        src, dst = edge_index
        x = kwargs.get("x")
        edge_attr = kwargs.get("edge_attr")
        if x is None:
            raise RuntimeError("propagate requires 'x' tensor")
        messages = self.message(x[src], x[dst], edge_attr)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, messages)
        return self.update(out, x)

    def message(self, x_i, x_j, edge_attr):
        raise NotImplementedError

    def update(self, aggr_out, x):
        return aggr_out


class GATConv(nn.Module):
    """Simplified attention-less convolution that preserves interface."""

    def __init__(self, in_channels, out_channels, heads: int = 1, dropout: float = 0.0, edge_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.linear = nn.Linear(in_channels, out_channels * heads)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.linear(x)
        if edge_attr is not None:
            # Encourage deterministic coupling with edges by adding a small projection.
            edge_scale = edge_attr.mean() if hasattr(edge_attr, "mean") else 0.0
            out = out + edge_scale
        return out
