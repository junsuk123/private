"""Walk-forward ablation over the context stack, with leakage checks that can fail.

The question
------------
Each layer added by this refactor — session phase, day of week, seasonality, the global
tape, the domestic tape, the sector, the ontology, the GNN — costs latency and adds a way
to be wrong. This harness answers whether each one pays for itself, by rebuilding the
decision with that layer's features **removed** and comparing measured outcomes:

    BASE, TIME, DAY, SEASONALITY, GLOBAL, DOMESTIC, SECTOR, ONTOLOGY, GNN, FULL

Every arm is scored on the same rows and the same walk-forward splits, so the difference
between two arms is the layer and nothing else.

Leakage
-------
:func:`assert_no_future_leakage` is a real check, not a comment. It verifies that no
training row's label window ends after the first test row begins, which is the failure a
plain chronological split does not catch: a 30-minute label on a bar 10 minutes before
the boundary resolves *inside* the test window. It raises, and the ablation refuses to
report a score computed on a leaking split.

Metrics
-------
Both families the goals ask for. ML: precision, recall, F1, Brier, calibration error.
Trade: return, win rate, profit factor, expectancy, Sharpe, Sortino, max drawdown,
turnover, slippage. Sliced by day, session phase, regime and sector, because an edge that
exists only in one slice is a property of that slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

from app.evaluation.purged_walk_forward import (
    PurgedWalkForwardSplit,
    purged_walk_forward_splits,
)

__all__ = [
    "ABLATION_ARMS",
    "AblationArm",
    "AblationResult",
    "AblationRow",
    "LeakageError",
    "MlMetrics",
    "TradeMetrics",
    "assert_no_future_leakage",
    "ml_metrics",
    "run_ablation",
    "trade_metrics",
]

#: Feature groups each arm removes. ``FULL`` removes nothing; ``BASE`` keeps only the
#: symbol's own tape. Every other arm is FULL minus exactly one layer, which is what
#: makes the comparison attributable to that layer.
_ARM_REMOVES: dict[str, tuple[str, ...]] = {
    "BASE": (
        "time",
        "day",
        "seasonality",
        "global",
        "domestic",
        "sector",
        "ontology",
        "gnn",
    ),
    "TIME": ("time",),
    "DAY": ("day",),
    "SEASONALITY": ("seasonality",),
    "GLOBAL": ("global",),
    "DOMESTIC": ("domestic",),
    "SECTOR": ("sector",),
    "ONTOLOGY": ("ontology",),
    "GNN": ("gnn",),
    "FULL": (),
}

ABLATION_ARMS: tuple[str, ...] = tuple(_ARM_REMOVES)

#: Feature-name prefixes belonging to each layer. A feature whose prefix matches a removed
#: group is dropped from that arm's inputs.
_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "time": ("session_", "minutes_", "is_opening", "is_closing", "phase_"),
    "day": ("day_of_week", "holiday_adjacent", "month_end", "quarter_end", "expiry_"),
    "seasonality": ("seasonality_", "z_"),
    "global": ("global_",),
    "domestic": ("domestic_",),
    "sector": ("sector_",),
    "ontology": ("ontology_",),
    "gnn": ("gnn_", "model_"),
}

#: Slices every metric is additionally reported over.
SLICES: tuple[str, ...] = ("day", "session", "regime", "sector")


class LeakageError(AssertionError):
    """A split whose training labels resolve inside the test window."""


@dataclass(frozen=True)
class AblationRow:
    """One observation: features, what happened, and how it is sliced."""

    as_of: datetime
    #: When this row's label finished resolving. The leakage check needs it.
    label_end: datetime
    features: Mapping[str, float]
    #: 1 for a profitable outcome, 0 otherwise.
    label: int
    #: Realised net return in bps, signed.
    net_return_bps: float
    #: Position turnover attributable to this row, as a fraction of equity.
    turnover: float = 0.0
    #: Realised slippage in bps, unsigned.
    slippage_bps: float = 0.0
    day: str = ""
    session: str = ""
    regime: str = ""
    sector: str = ""

    def slice_value(self, name: str) -> str:
        return {
            "day": self.day,
            "session": self.session,
            "regime": self.regime,
            "sector": self.sector,
        }.get(name, "")


@dataclass(frozen=True)
class MlMetrics:
    precision: float
    recall: float
    f1: float
    brier: float
    calibration_error: float
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "brier": self.brier,
            "calibration": self.calibration_error,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class TradeMetrics:
    total_return_bps: float
    win_rate: float
    profit_factor: float
    expectancy_bps: float
    sharpe: float
    sortino: float
    max_drawdown_bps: float
    turnover: float
    slippage_bps: float
    trade_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "return": self.total_return_bps,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy_bps,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "MDD": self.max_drawdown_bps,
            "turnover": self.turnover,
            "slippage": self.slippage_bps,
            "trade_count": self.trade_count,
        }


@dataclass(frozen=True)
class AblationArm:
    name: str
    removed_groups: tuple[str, ...]
    ml: MlMetrics
    trade: TradeMetrics
    slices: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    feature_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "removed_groups": list(self.removed_groups),
            "feature_count": self.feature_count,
            "ml": self.ml.as_dict(),
            "trade": self.trade.as_dict(),
            "slices": {
                name: {key: dict(values) for key, values in buckets.items()}
                for name, buckets in self.slices.items()
            },
        }


@dataclass(frozen=True)
class AblationResult:
    arms: tuple[AblationArm, ...]
    split_count: int
    row_count: int
    embargo_count: int

    def by_name(self) -> dict[str, AblationArm]:
        return {arm.name: arm for arm in self.arms}

    def lift_over(self, baseline: str = "BASE") -> dict[str, float]:
        """Each arm's F1 minus the baseline's. Positive means the layer earned its place."""
        arms = self.by_name()
        reference = arms.get(baseline)
        if reference is None:
            return {}
        return {
            name: round(arm.ml.f1 - reference.ml.f1, 6)
            for name, arm in arms.items()
            if name != baseline
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_count": self.split_count,
            "row_count": self.row_count,
            "embargo_count": self.embargo_count,
            "arms": [arm.as_dict() for arm in self.arms],
            "lift_over_base": self.lift_over(),
        }


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def assert_no_future_leakage(
    rows: Sequence[AblationRow], splits: Iterable[PurgedWalkForwardSplit]
) -> None:
    """Raise when any training label resolves at or after the test window opens.

    This is the check a chronological split silently fails: a row observed before the
    boundary whose 30-minute label finishes after it has seen the test period's prices.
    Purging is supposed to remove exactly those rows, and this asserts that it did.
    """
    for index, split in enumerate(splits):
        if not split.test_indices or not split.train_indices:
            continue
        test_start = min(rows[position].as_of for position in split.test_indices)
        for position in split.train_indices:
            if rows[position].label_end >= test_start:
                raise LeakageError(
                    f"split {index}: training row at {rows[position].as_of.isoformat()} "
                    f"resolves at {rows[position].label_end.isoformat()}, "
                    f"at or after the test window opens at {test_start.isoformat()}"
                )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ml_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float = 0.5
) -> MlMetrics:
    """Precision/recall/F1 at ``threshold``, plus Brier and a 10-bin calibration error."""
    if not labels:
        return MlMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    predicted = [1 if value >= threshold else 0 for value in probabilities]
    true_positive = sum(1 for a, b in zip(labels, predicted) if a == 1 and b == 1)
    false_positive = sum(1 for a, b in zip(labels, predicted) if a == 0 and b == 1)
    false_negative = sum(1 for a, b in zip(labels, predicted) if a == 1 and b == 0)
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    brier = sum(
        (probability - label) ** 2 for label, probability in zip(labels, probabilities)
    ) / len(labels)

    # Expected calibration error over ten equal-width probability bins. Bins with no
    # observations contribute nothing rather than a zero, which would flatter a model
    # that never predicts in that range.
    bins: dict[int, list[tuple[int, float]]] = {}
    for label, probability in zip(labels, probabilities):
        bucket = min(9, max(0, int(probability * 10)))
        bins.setdefault(bucket, []).append((label, probability))
    calibration = 0.0
    for entries in bins.values():
        observed = sum(label for label, _ in entries) / len(entries)
        expected = sum(probability for _, probability in entries) / len(entries)
        calibration += abs(observed - expected) * len(entries) / len(labels)

    return MlMetrics(
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        brier=round(brier, 6),
        calibration_error=round(calibration, 6),
        sample_count=len(labels),
    )


def trade_metrics(rows: Sequence[AblationRow], taken: Sequence[bool]) -> TradeMetrics:
    """Realised performance over the rows the arm actually traded."""
    returns = [
        row.net_return_bps for row, took in zip(rows, taken) if took
    ]
    if not returns:
        return TradeMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    total = sum(returns)
    mean = total / len(returns)
    variance = (
        sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        if len(returns) > 1
        else 0.0
    )
    stdev = math.sqrt(variance)
    downside = [value for value in returns if value < 0]
    downside_stdev = (
        math.sqrt(sum(value**2 for value in downside) / len(downside))
        if downside
        else 0.0
    )
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    gross_loss = abs(sum(losses))
    return TradeMetrics(
        total_return_bps=round(total, 6),
        win_rate=round(len(wins) / len(returns), 6),
        # A run with no losing trade has an undefined profit factor. Reported as inf
        # rather than as a large finite number, so it cannot be averaged into a summary
        # and read as a real measurement.
        profit_factor=(
            round(sum(wins) / gross_loss, 6) if gross_loss > 0 else float("inf")
        ),
        expectancy_bps=round(mean, 6),
        sharpe=round(mean / stdev, 6) if stdev > 0 else 0.0,
        sortino=round(mean / downside_stdev, 6) if downside_stdev > 0 else 0.0,
        max_drawdown_bps=round(drawdown, 6),
        turnover=round(
            sum(row.turnover for row, took in zip(rows, taken) if took), 6
        ),
        slippage_bps=round(
            sum(row.slippage_bps for row, took in zip(rows, taken) if took)
            / len(returns),
            6,
        ),
        trade_count=len(returns),
    )


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
def filter_features(
    features: Mapping[str, float], removed_groups: Sequence[str]
) -> dict[str, float]:
    """Drop every feature belonging to a removed layer."""
    prefixes = tuple(
        prefix
        for group in removed_groups
        for prefix in _GROUP_PREFIXES.get(group, ())
    )
    if not prefixes:
        return dict(features)
    return {
        name: value
        for name, value in features.items()
        if not name.startswith(prefixes)
    }


def run_ablation(
    rows: Sequence[AblationRow],
    predictor: Callable[[Sequence[AblationRow], Sequence[AblationRow], Sequence[str]], Sequence[float]],
    *,
    train_size: int,
    test_size: int,
    embargo_count: int = 0,
    threshold: float = 0.5,
    arms: Sequence[str] = ABLATION_ARMS,
) -> AblationResult:
    """Score every arm on the same purged walk-forward splits.

    ``predictor`` is handed ``(train_rows, test_rows, removed_groups)`` and returns one
    probability per test row. It is a parameter rather than a fixed model so the same
    harness can score the rule stack, the GNN, or any candidate replacement without the
    comparison changing shape.
    """
    ordered = sorted(rows, key=lambda row: (row.as_of, row.label_end))
    splits = purged_walk_forward_splits(
        ordered,
        train_size=train_size,
        test_size=test_size,
        embargo_count=embargo_count,
    )
    if not splits:
        raise ValueError("not enough rows for the requested walk-forward split")
    assert_no_future_leakage(ordered, splits)

    results: list[AblationArm] = []
    for arm in arms:
        removed = _ARM_REMOVES.get(arm)
        if removed is None:
            raise ValueError(f"unknown ablation arm {arm!r}")
        labels: list[int] = []
        probabilities: list[float] = []
        evaluated: list[AblationRow] = []
        for split in splits:
            train_rows = [
                _with_features(ordered[index], removed) for index in split.train_indices
            ]
            test_rows = [
                _with_features(ordered[index], removed) for index in split.test_indices
            ]
            predictions = list(predictor(train_rows, test_rows, removed))
            if len(predictions) != len(test_rows):
                raise ValueError(
                    f"arm {arm}: predictor returned {len(predictions)} values for "
                    f"{len(test_rows)} test rows"
                )
            for row, probability in zip(test_rows, predictions):
                labels.append(int(row.label))
                probabilities.append(float(probability))
                evaluated.append(row)

        taken = [probability >= threshold for probability in probabilities]
        results.append(
            AblationArm(
                name=arm,
                removed_groups=removed,
                ml=ml_metrics(labels, probabilities, threshold=threshold),
                trade=trade_metrics(evaluated, taken),
                slices=_sliced_metrics(evaluated, labels, probabilities, taken, threshold),
                feature_count=(
                    len(filter_features(ordered[0].features, removed)) if ordered else 0
                ),
            )
        )
    return AblationResult(
        arms=tuple(results),
        split_count=len(splits),
        row_count=len(ordered),
        embargo_count=embargo_count,
    )


def _with_features(row: AblationRow, removed: Sequence[str]) -> AblationRow:
    from dataclasses import replace

    return replace(row, features=filter_features(row.features, removed))


def _sliced_metrics(
    rows: Sequence[AblationRow],
    labels: Sequence[int],
    probabilities: Sequence[float],
    taken: Sequence[bool],
    threshold: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    sliced: dict[str, dict[str, dict[str, Any]]] = {}
    for name in SLICES:
        buckets: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            key = row.slice_value(name)
            if not key:
                continue
            buckets.setdefault(key, []).append(index)
        if not buckets:
            continue
        sliced[name] = {
            key: {
                "ml": ml_metrics(
                    [labels[index] for index in indices],
                    [probabilities[index] for index in indices],
                    threshold=threshold,
                ).as_dict(),
                "trade": trade_metrics(
                    [rows[index] for index in indices],
                    [taken[index] for index in indices],
                ).as_dict(),
            }
            for key, indices in buckets.items()
        }
    return sliced
