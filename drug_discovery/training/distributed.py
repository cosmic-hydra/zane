"""
Distributed Training Infrastructure
Supports multi-GPU and multi-node training
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Distributed training for large-scale drug discovery models
    """

    def __init__(
        self,
        model: torch.nn.Module,
        rank: int = 0,
        world_size: int = 1,
        backend: str = 'nccl'
    ):
        """
        Args:
            model: Model to train
            rank: Process rank
            world_size: Total number of processes
            backend: Distributed backend ('nccl', 'gloo')
        """
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

        self.is_distributed = world_size > 1

        if self.is_distributed:
            self._setup_distributed()

    def _setup_distributed(self):
        """Initialize distributed training"""
        try:
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )

            # Move model to GPU
            device = torch.device(f'cuda:{self.rank}')
            self.model = self.model.to(device)

            # Wrap model with DDP
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )

            logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")

        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            self.is_distributed = False

    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed_environment():
    """
    Setup environment for distributed training
    """
    # Get rank and world size from environment
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank
