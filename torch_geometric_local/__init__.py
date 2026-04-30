"""
Lightweight torch_geometric stand-ins for environments without the real package.

These stubs implement the minimal surface area exercised by the test suite:
Data containers, simple pooling ops, and placeholder convolution layers.
They are not feature-complete but allow CPU-only execution without the
compiled extensions normally required by torch_geometric.
"""

from .data import Data
from .loader import DataLoader
from .nn import GATConv, MessagePassing, global_max_pool, global_mean_pool

__all__ = ["Data", "DataLoader", "GATConv", "MessagePassing", "global_mean_pool", "global_max_pool"]
