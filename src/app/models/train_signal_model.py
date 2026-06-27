from __future__ import annotations

from dataclasses import dataclass

from app.models.dataset_builder import DatasetRow


@dataclass(frozen=True)
class TrainingPlan:
    row_count: int
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    no_lookahead_verified: bool


def build_training_plan(rows: tuple[DatasetRow, ...]) -> TrainingPlan:
    feature_names = tuple(sorted({name for row in rows for name in row.features}))
    label_names = tuple(sorted({name for row in rows for name in row.labels}))
    no_lookahead = all(row.metadata.get("no_lookahead") is True for row in rows)
    return TrainingPlan(
        row_count=len(rows),
        feature_names=feature_names,
        label_names=label_names,
        no_lookahead_verified=no_lookahead,
    )
