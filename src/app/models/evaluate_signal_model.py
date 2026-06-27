from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from app.models.dataset_builder import DatasetRow


@dataclass(frozen=True)
class SignalEvaluationSummary:
    row_count: int
    precision_at_k: float
    average_forward_return: float
    downside_hit_rate: float
    risk_rejection_rate: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2)


def evaluate_ranked_signal_rows(
    rows: tuple[DatasetRow, ...],
    *,
    score_feature: str = "signal_score",
    forward_return_label: str = "future_return_5d",
    k: int = 10,
) -> SignalEvaluationSummary:
    if not rows:
        return SignalEvaluationSummary(0, 0.0, 0.0, 0.0, 0.0)
    ranked = sorted(rows, key=lambda row: float(row.features.get(score_feature, 0.0)), reverse=True)
    top = ranked[: max(1, min(k, len(ranked)))]
    returns = [
        float(row.labels[forward_return_label])
        for row in top
        if row.labels.get(forward_return_label) is not None
    ]
    positives = sum(1 for value in returns if value > 0)
    downside = sum(1 for value in returns if value < 0)
    rejected = sum(1 for row in rows if row.metadata.get("risk_rejected") is True)
    return SignalEvaluationSummary(
        row_count=len(rows),
        precision_at_k=round(positives / max(1, len(returns)), 6),
        average_forward_return=round(sum(returns) / max(1, len(returns)), 6),
        downside_hit_rate=round(downside / max(1, len(returns)), 6),
        risk_rejection_rate=round(rejected / max(1, len(rows)), 6),
    )


def write_evaluation_summary(summary: SignalEvaluationSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.to_json(), encoding="utf-8")
