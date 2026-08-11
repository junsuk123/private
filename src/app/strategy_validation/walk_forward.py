"""Anchored and rolling walk-forward over trades, with out-of-sample stability.

The number this produces that a plain backtest cannot
----------------------------------------------------
``out_of_sample_stability``: the fraction of walk-forward windows whose out-of-sample net EV
is positive. A strategy with one enormous winning month and five losing ones can show a
positive overall mean; its stability is 1/6, and that is the number that says the mean is an
artefact.

Purging applies here too. Windows are separated by an embargo equal to the median trade
horizon, so a trade that opens in-sample and closes out-of-sample does not appear in both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from app.strategy_validation.metrics import (
    StrategyMetrics,
    TradeObservation,
    compute_metrics,
)

__all__ = [
    "WalkForwardResult",
    "WalkForwardWindow",
    "walk_forward",
]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: StrategyMetrics
    test_metrics: StrategyMetrics
    embargoed_count: int

    @property
    def in_sample_net_bps(self) -> float | None:
        return self.train_metrics.net_ev_bps

    @property
    def out_of_sample_net_bps(self) -> float | None:
        return self.test_metrics.net_ev_bps

    @property
    def degradation_bps(self) -> float | None:
        """In-sample minus out-of-sample. Large positive = the fit did not carry."""
        if self.in_sample_net_bps is None or self.out_of_sample_net_bps is None:
            return None
        return self.in_sample_net_bps - self.out_of_sample_net_bps

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_window": [self.train_start.isoformat(), self.train_end.isoformat()],
            "test_window": [self.test_start.isoformat(), self.test_end.isoformat()],
            "in_sample_net_bps": _round(self.in_sample_net_bps),
            "out_of_sample_net_bps": _round(self.out_of_sample_net_bps),
            "degradation_bps": _round(self.degradation_bps),
            "train_trades": self.train_metrics.trigger_count,
            "test_trades": self.test_metrics.trigger_count,
            "embargoed_count": self.embargoed_count,
        }


@dataclass(frozen=True)
class WalkForwardResult:
    strategy_id: str
    windows: tuple[WalkForwardWindow, ...]
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def out_of_sample_stability(self) -> float | None:
        scored = [
            window
            for window in self.windows
            if window.out_of_sample_net_bps is not None
        ]
        if not scored:
            return None
        return sum(1 for window in scored if window.out_of_sample_net_bps > 0) / len(scored)

    @property
    def mean_out_of_sample_net_bps(self) -> float | None:
        values = [
            window.out_of_sample_net_bps
            for window in self.windows
            if window.out_of_sample_net_bps is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def mean_degradation_bps(self) -> float | None:
        values = [
            window.degradation_bps
            for window in self.windows
            if window.degradation_bps is not None
        ]
        return sum(values) / len(values) if values else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "window_count": len(self.windows),
            "out_of_sample_stability": _round(self.out_of_sample_stability, 4),
            "mean_out_of_sample_net_bps": _round(self.mean_out_of_sample_net_bps),
            "mean_degradation_bps": _round(self.mean_degradation_bps),
            "windows": [window.as_dict() for window in self.windows],
            "reason_codes": list(self.reason_codes),
        }


def walk_forward(
    strategy_id: str,
    trades: Sequence[TradeObservation],
    *,
    windows: int = 4,
    anchored: bool = True,
    embargo_seconds: float | None = None,
    minimum_test_trades: int = 5,
) -> WalkForwardResult:
    """Split ``trades`` chronologically into ``windows`` train/test pairs.

    ``anchored=True`` grows the training window from the start (the standard anchored
    walk-forward); ``False`` rolls a fixed-length one. Anchored is the default because it
    matches how the live system actually learns — it never forgets old outcomes, it
    down-weights them.
    """
    reasons: list[str] = []
    ordered = sorted(trades, key=lambda item: _aware(item.opened_at))
    if len(ordered) < max(2, int(windows)) * max(1, int(minimum_test_trades)):
        reasons.append("WALK_FORWARD_INSUFFICIENT_TRADES")
        return WalkForwardResult(
            strategy_id=strategy_id, windows=(), reason_codes=tuple(reasons)
        )

    embargo = timedelta(seconds=_embargo_seconds(ordered, embargo_seconds))
    count = max(2, int(windows))
    # ``count + 1`` blocks: the first is training-only, then each subsequent block is a test
    # fold with everything before it (anchored) or the previous block (rolling) as training.
    size = len(ordered) / (count + 1)
    built: list[WalkForwardWindow] = []
    for index in range(count):
        train_end_position = int(round((index + 1) * size))
        test_end_position = int(round((index + 2) * size))
        test_rows = ordered[train_end_position:test_end_position]
        if len(test_rows) < max(1, int(minimum_test_trades)):
            continue
        train_start_position = 0 if anchored else int(round(index * size))
        train_rows = ordered[train_start_position:train_end_position]
        if not train_rows:
            continue

        test_start = _aware(test_rows[0].opened_at)
        # Purge + embargo: a training trade that is still open when the test window starts,
        # or that opens inside the embargo before it, shares information with the test set.
        boundary = test_start - embargo
        kept = [row for row in train_rows if _aware(row.closed_at) <= boundary]
        embargoed = len(train_rows) - len(kept)
        if not kept:
            reasons.append(f"WALK_FORWARD_WINDOW_{index}_FULLY_PURGED")
            continue

        built.append(
            WalkForwardWindow(
                index=index,
                train_start=_aware(kept[0].opened_at),
                train_end=_aware(kept[-1].closed_at),
                test_start=test_start,
                test_end=_aware(test_rows[-1].closed_at),
                train_metrics=compute_metrics(strategy_id, kept, minimum_samples=10),
                test_metrics=compute_metrics(
                    strategy_id, test_rows, minimum_samples=minimum_test_trades
                ),
                embargoed_count=embargoed,
            )
        )
    if not built:
        reasons.append("WALK_FORWARD_NO_USABLE_WINDOWS")
    return WalkForwardResult(
        strategy_id=strategy_id, windows=tuple(built), reason_codes=tuple(dict.fromkeys(reasons))
    )


def _embargo_seconds(
    trades: Sequence[TradeObservation], override: float | None
) -> float:
    if override is not None and override >= 0:
        return float(override)
    horizons = sorted(trade.holding_seconds for trade in trades)
    if not horizons:
        return 0.0
    middle = len(horizons) // 2
    return (
        horizons[middle]
        if len(horizons) % 2
        else (horizons[middle - 1] + horizons[middle]) / 2.0
    )


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None
