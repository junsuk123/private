from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.evaluation.purged_walk_forward import purged_walk_forward_splits


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Row:
    as_of: datetime
    label_end: datetime


def test_overlapping_training_label_is_purged() -> None:
    rows = [
        Row(NOW + timedelta(days=index), NOW + timedelta(days=index + 1))
        for index in range(8)
    ]
    # Last training row has a label extending into the first test observation.
    rows[3] = Row(NOW + timedelta(days=3), NOW + timedelta(days=5))
    split = purged_walk_forward_splits(
        rows, train_size=4, test_size=2, embargo_count=1
    )[0]
    assert split.test_indices == (4, 5)
    assert split.purged_indices == (3,)
    assert split.train_indices == (0, 1, 2)
    assert split.embargoed_indices == (6,)


def test_future_rows_do_not_enter_training_set() -> None:
    rows = [
        Row(NOW + timedelta(minutes=index), NOW + timedelta(minutes=index + 1))
        for index in range(12)
    ]
    for split in purged_walk_forward_splits(rows, train_size=5, test_size=2):
        assert max(rows[index].as_of for index in split.train_indices) < min(
            rows[index].as_of for index in split.test_indices
        )
