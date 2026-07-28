from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence


class LabeledInterval(Protocol):
    as_of: datetime
    label_end: datetime


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_indices: tuple[int, ...]
    embargoed_indices: tuple[int, ...]


def purged_walk_forward_splits(
    rows: Sequence[LabeledInterval],
    *,
    train_size: int,
    test_size: int,
    embargo_count: int = 0,
    step_size: int | None = None,
) -> tuple[PurgedWalkForwardSplit, ...]:
    if min(train_size, test_size) <= 0 or embargo_count < 0:
        raise ValueError("invalid split sizes")
    ordered_indices = tuple(
        sorted(range(len(rows)), key=lambda index: (rows[index].as_of, rows[index].label_end))
    )
    step = step_size or test_size
    splits: list[PurgedWalkForwardSplit] = []
    cursor = 0
    while cursor + train_size + test_size <= len(ordered_indices):
        raw_train = ordered_indices[cursor : cursor + train_size]
        test = ordered_indices[cursor + train_size : cursor + train_size + test_size]
        test_start = min(rows[index].as_of for index in test)
        test_end = max(rows[index].label_end for index in test)
        purged = tuple(
            index
            for index in raw_train
            if rows[index].label_end >= test_start
        )
        train = tuple(index for index in raw_train if index not in set(purged))
        embargo_candidates = ordered_indices[
            cursor + train_size + test_size :
            cursor + train_size + test_size + embargo_count
        ]
        embargoed = tuple(
            index for index in embargo_candidates if rows[index].as_of <= test_end
        )
        splits.append(
            PurgedWalkForwardSplit(
                train_indices=train,
                test_indices=tuple(test),
                purged_indices=purged,
                embargoed_indices=embargoed,
            )
        )
        cursor += step
    return tuple(splits)
