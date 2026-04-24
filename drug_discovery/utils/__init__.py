"""
Utility Functions for Drug Discovery
"""

import random

import numpy as np

try:
    import torch as torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)  # type: ignore[union-attr]
        if torch.cuda.is_available():  # type: ignore[union-attr]
            torch.cuda.manual_seed(seed)  # type: ignore[union-attr]
            torch.cuda.manual_seed_all(seed)  # type: ignore[union-attr]
            torch.backends.cudnn.deterministic = True  # type: ignore[union-attr]
            torch.backends.cudnn.benchmark = False  # type: ignore[union-attr]


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get the best available device

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if _TORCH_AVAILABLE and prefer_cuda and torch.cuda.is_available():  # type: ignore[union-attr]
        return "cuda"
    return "cpu"


def count_parameters(model: "torch.nn.Module") -> int:
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop

        Args:
            val_loss: Validation loss

        Returns:
            Whether to stop training
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
