from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class WalkForwardSplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def walk_forward_splits(
    rows: Sequence[T],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> tuple[WalkForwardSplit, ...]:
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    splits: list[WalkForwardSplit] = []
    start = 0
    total = len(rows)
    while start + train_size + test_size <= total:
        train = tuple(range(start, start + train_size))
        test = tuple(range(start + train_size, start + train_size + test_size))
        splits.append(WalkForwardSplit(train_indices=train, test_indices=test))
        start += step
    return tuple(splits)
