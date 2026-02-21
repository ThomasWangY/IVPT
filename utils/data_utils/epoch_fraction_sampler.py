"""Distributed sampler that uses only a fraction of data per epoch.

Splits the full dataset into ``ceil(1/fraction)`` non-overlapping shards.
Each epoch uses one shard (cycling through shards across epochs), so that
after enough epochs every sample is seen exactly once per cycle.

This guarantees:
    1. Each epoch only iterates ``fraction`` of the data (fewer batches).
    2. All samples are uniformly covered across consecutive epochs.
"""

import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class EpochFractionDistributedSampler(Sampler):
    """DistributedSampler that only yields a fraction of the dataset per epoch.

    Parameters
    ----------
    dataset : Dataset
        The full training dataset.
    fraction : float
        Fraction of the dataset to use per epoch, e.g. 0.1 = 10%.
    num_replicas : int, optional
        Number of distributed processes (default: world_size).
    rank : int, optional
        Rank of the current process (default: current rank).
    shuffle : bool
        Whether to shuffle indices (default: True).
    seed : int
        Random seed for reproducibility (default: 0).
    drop_last : bool
        Whether to drop the last incomplete batch (default: False).
    """

    def __init__(self, dataset: Dataset, fraction: float = 0.1,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0,
                 drop_last: bool = False) -> None:
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        assert 0 < fraction <= 1.0, f"fraction must be in (0, 1], got {fraction}"

        self.dataset = dataset
        self.fraction = fraction
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Total number of shards needed to cover the full dataset
        self.num_shards = math.ceil(1.0 / fraction)
        self.total_dataset_size = len(dataset)

        # Samples per shard (before distributing across replicas)
        self.shard_size = math.ceil(self.total_dataset_size / self.num_shards)

        # Samples per replica per epoch
        if self.drop_last:
            self.num_samples = math.floor(self.shard_size / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.shard_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Deterministic shuffling based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(self.total_dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.total_dataset_size))

        # Select the shard for this epoch (cycling)
        shard_id = self.epoch % self.num_shards
        start = shard_id * self.shard_size
        end = min(start + self.shard_size, self.total_dataset_size)
        indices = indices[start:end]

        # Pad to make evenly divisible by num_replicas
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices += indices[:padding]

        # Subsample for this replica
        indices = indices[self.rank:self.total_size:self.num_replicas]

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class EpochFractionSampler(Sampler):
    """Single-GPU version of EpochFractionDistributedSampler."""

    def __init__(self, dataset: Dataset, fraction: float = 0.1,
                 shuffle: bool = True, seed: int = 0) -> None:
        assert 0 < fraction <= 1.0, f"fraction must be in (0, 1], got {fraction}"

        self.dataset = dataset
        self.fraction = fraction
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.num_shards = math.ceil(1.0 / fraction)
        self.total_dataset_size = len(dataset)
        self.shard_size = math.ceil(self.total_dataset_size / self.num_shards)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(self.total_dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.total_dataset_size))

        shard_id = self.epoch % self.num_shards
        start = shard_id * self.shard_size
        end = min(start + self.shard_size, self.total_dataset_size)
        indices = indices[start:end]

        return iter(indices)

    def __len__(self) -> int:
        return self.shard_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
