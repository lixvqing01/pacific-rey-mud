from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class BlockBalancedSamplingConfig:
    alpha: float = 1.0
    min_count: int = 1
    num_samples: int | None = None
    replacement: bool = True
    seed: int = 42


class WeightedSubsetRandomSampler(Sampler[int]):
    """
    Weighted sampler over a fixed subset of dataset indices.

    Torch's built-in WeightedRandomSampler yields indices in [0, len(weights)),
    which doesn't directly map to global dataset indices when sampling a subset.
    This sampler returns the original dataset indices.
    """

    def __init__(
        self,
        *,
        indices: Sequence[int],
        weights: Sequence[float],
        num_samples: int | None = None,
        replacement: bool = True,
        seed: int = 42,
    ) -> None:
        if len(indices) != len(weights):
            raise ValueError("indices and weights must have the same length.")
        if len(indices) == 0:
            raise ValueError("indices cannot be empty.")
        self._indices = np.asarray(indices, dtype=int)
        w = np.asarray(weights, dtype=float)
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite.")
        if (w < 0).any():
            raise ValueError("weights must be non-negative.")
        if w.sum() <= 0:
            raise ValueError("weights sum must be > 0.")

        self._weights = torch.as_tensor(w, dtype=torch.double)
        self._num_samples = int(num_samples if num_samples is not None else len(self._indices))
        self._replacement = bool(replacement)
        self._seed = int(seed)

    def __iter__(self) -> Iterator[int]:


        sampled = torch.multinomial(
            self._weights,
            num_samples=self._num_samples,
            replacement=self._replacement,
        )
        for j in sampled.tolist():
            yield int(self._indices[j])

    def __len__(self) -> int:
        return int(self._num_samples)


def make_block_balanced_sampler(
    *,
    train_indices: Sequence[int],
    block_ids: Sequence[int] | np.ndarray,
    config: BlockBalancedSamplingConfig,
) -> WeightedSubsetRandomSampler:
    """
    Build a sampler that approximately equalizes per-block contribution.

    Weights are proportional to (1 / max(count(block), min_count)) ** alpha so that
    blocks with many samples do not dominate optimization.
    """
    train_indices_arr = np.asarray(train_indices, dtype=int)
    block_ids_arr = np.asarray(block_ids, dtype=np.int64)
    if block_ids_arr.shape[0] <= int(train_indices_arr.max(initial=-1)):
        raise ValueError("block_ids length is smaller than max(train_indices).")

    train_blocks = block_ids_arr[train_indices_arr]
    unique_blocks, counts = np.unique(train_blocks, return_counts=True)
    count_map = dict(zip(unique_blocks.tolist(), counts.tolist()))

    alpha = float(config.alpha)
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    min_count = int(config.min_count)
    if min_count < 1:
        min_count = 1

    weights = []
    for b in train_blocks.tolist():
        c = max(int(count_map.get(int(b), 1)), min_count)
        w = (1.0 / float(c)) ** alpha if alpha > 0 else 1.0
        weights.append(w)

    return WeightedSubsetRandomSampler(
        indices=train_indices_arr.tolist(),
        weights=weights,
        num_samples=config.num_samples if config.num_samples is not None else len(train_indices_arr),
        replacement=config.replacement,
        seed=config.seed,
    )
