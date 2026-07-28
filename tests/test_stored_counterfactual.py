from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.evaluation.stored_counterfactual import (
    EvaluationConfig,
    build_labels,
    causal_percentile,
)
from app.trading.contracts import Bar


def _bars(count: int) -> tuple[Bar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        Bar(
            symbol="TEST",
            venue="NASD",
            interval="1m",
            start_time=start + timedelta(minutes=index),
            end_time=start + timedelta(minutes=index + 1),
            open=100 + index * 0.01,
            high=100.1 + index * 0.01,
            low=99.9 + index * 0.01,
            close=100 + index * 0.01,
            volume=100 + index,
        )
        for index in range(count)
    )


def test_causal_percentile_does_not_consume_future_values() -> None:
    assert causal_percentile(3, [1, 2]) == 1.0
    assert causal_percentile(3, [1, 2, 100]) == 2 / 3


def test_stored_labels_have_strict_future_label_window() -> None:
    labels = build_labels(
        {"TEST": _bars(15)},
        EvaluationConfig(
            history_bars=5,
            horizon_bars=3,
            stride_bars=2,
            minimum_symbol_bars=10,
        ),
    )
    assert labels
    assert all(label.label_end > label.as_of for label in labels)
    assert {label.strategy_id for label in labels} == {
        "intraday_momentum",
        "breakout_volume",
        "vwap_mean_reversion",
        "liquidity_shock_reversal",
        "event_momentum",
        "cross_sectional_relative_strength",
        "gap_context",
    }
