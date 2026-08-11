"""Purged, embargoed cross-validation over trades whose horizons overlap.

Why this exists next to ``app.evaluation.purged_walk_forward``
-------------------------------------------------------------
That module splits a *bar series* for model training. What is needed here is splitting a set
of *trades*, each with its own open and close time and its own horizon. The leak is the same
in both cases and so is the fix, but the unit is different: a trade that opens inside a test
fold and closes inside a training fold shares its outcome with the training data, so the
training fold has to lose it (purge) plus a buffer for serial dependence (embargo).

Without purging, a strategy with a 3,600-second horizon evaluated on 60-second-spaced
signals shares almost its entire label with its neighbours, and the resulting "out-of-sample"
score is in-sample. This project has already measured the size of that error in another
form: stride below the horizon inflated a sample count 56-fold.

Combinatorial purged CV
-----------------------
``combinatorial_splits`` implements the CPCV idea: choose ``test_groups`` of the ``n_groups``
blocks as the test set, in every combination, so each trade is tested under many different
training sets rather than exactly one. The number of paths grows as C(n, k), which is why the
default is a modest 6-choose-2 = 15 rather than something that would make an audit run take
minutes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

__all__ = [
    "PurgedSplit",
    "combinatorial_splits",
    "purged_kfold_splits",
]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class PurgedSplit:
    """One fold. Indices are positions in the ORIGINAL trade sequence."""

    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_indices: tuple[int, ...]
    embargoed_indices: tuple[int, ...]
    test_window: tuple[datetime, datetime]

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "purged": len(self.purged_indices),
            "embargoed": len(self.embargoed_indices),
            "test_window": [
                self.test_window[0].isoformat(),
                self.test_window[1].isoformat(),
            ],
        }


def _intervals(trades: Sequence[Any]) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    for trade in trades:
        opened = _aware(getattr(trade, "opened_at"))
        closed = _aware(getattr(trade, "closed_at"))
        intervals.append((opened, max(closed, opened)))
    return intervals


def _split_indices(
    intervals: Sequence[tuple[datetime, datetime]],
    test_positions: Sequence[int],
    *,
    embargo: timedelta,
    fold: int,
) -> PurgedSplit:
    test_set = set(test_positions)
    test_start = min(intervals[index][0] for index in test_positions)
    test_end = max(intervals[index][1] for index in test_positions)
    embargo_end = test_end + embargo

    train: list[int] = []
    purged: list[int] = []
    embargoed: list[int] = []
    for index, (opened, closed) in enumerate(intervals):
        if index in test_set:
            continue
        # PURGE: any overlap at all with the test window means shared information.
        if opened <= test_end and closed >= test_start:
            purged.append(index)
            continue
        # EMBARGO: the window immediately after the test set, where serial dependence
        # still links a training trade to a test outcome.
        if test_end < opened <= embargo_end:
            embargoed.append(index)
            continue
        train.append(index)

    return PurgedSplit(
        fold=fold,
        train_indices=tuple(train),
        test_indices=tuple(sorted(test_set)),
        purged_indices=tuple(purged),
        embargoed_indices=tuple(embargoed),
        test_window=(test_start, test_end),
    )


def purged_kfold_splits(
    trades: Sequence[Any],
    *,
    folds: int = 5,
    embargo_seconds: float | None = None,
) -> tuple[PurgedSplit, ...]:
    """Contiguous-in-time folds, with purge and embargo applied to each.

    ``embargo_seconds`` defaults to the MEDIAN trade horizon in the set. That is the
    principled choice rather than a fixed number: the dependence a trade induces lasts about
    as long as the trade does, and the same constant applied to a 150-second scalp and a
    64,800-second overnight carry would be wrong for at least one of them.
    """
    if len(trades) < max(2, int(folds)):
        return ()
    intervals = _intervals(trades)
    order = sorted(range(len(trades)), key=lambda index: intervals[index][0])
    embargo = timedelta(seconds=_embargo_seconds(intervals, embargo_seconds))

    fold_count = max(2, int(folds))
    size = len(order) / fold_count
    splits: list[PurgedSplit] = []
    for fold in range(fold_count):
        start = int(round(fold * size))
        end = int(round((fold + 1) * size))
        positions = order[start:end]
        if not positions:
            continue
        splits.append(_split_indices(intervals, positions, embargo=embargo, fold=fold))
    return tuple(splits)


def combinatorial_splits(
    trades: Sequence[Any],
    *,
    n_groups: int = 6,
    test_groups: int = 2,
    embargo_seconds: float | None = None,
    max_paths: int = 15,
) -> tuple[PurgedSplit, ...]:
    """CPCV: every combination of ``test_groups`` blocks out of ``n_groups``.

    ``max_paths`` caps the combinations actually returned and the cap is REPORTED by
    returning fewer splits than C(n, k) — a silent truncation would make a partial sweep
    look like a complete one.
    """
    groups = max(2, int(n_groups))
    held = max(1, min(groups - 1, int(test_groups)))
    if len(trades) < groups:
        return ()
    intervals = _intervals(trades)
    order = sorted(range(len(trades)), key=lambda index: intervals[index][0])
    embargo = timedelta(seconds=_embargo_seconds(intervals, embargo_seconds))

    size = len(order) / groups
    blocks = [
        order[int(round(index * size)) : int(round((index + 1) * size))]
        for index in range(groups)
    ]
    blocks = [block for block in blocks if block]

    splits: list[PurgedSplit] = []
    for fold, combination in enumerate(itertools.combinations(range(len(blocks)), held)):
        if len(splits) >= max(1, int(max_paths)):
            break
        positions = [index for block in combination for index in blocks[block]]
        if not positions:
            continue
        splits.append(_split_indices(intervals, positions, embargo=embargo, fold=fold))
    return tuple(splits)


def _embargo_seconds(
    intervals: Sequence[tuple[datetime, datetime]], override: float | None
) -> float:
    if override is not None and override >= 0:
        return float(override)
    horizons = sorted((end - start).total_seconds() for start, end in intervals)
    if not horizons:
        return 0.0
    middle = len(horizons) // 2
    median = (
        horizons[middle]
        if len(horizons) % 2
        else (horizons[middle - 1] + horizons[middle]) / 2.0
    )
    return max(0.0, median)
