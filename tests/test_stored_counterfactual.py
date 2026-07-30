from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from app.evaluation.stored_counterfactual import (
    EvaluationConfig,
    _execution_market_for_symbol,
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


def test_execution_market_is_inferred_from_symbol() -> None:
    config = EvaluationConfig()

    assert _execution_market_for_symbol("005930", config) == (
        "KRX",
        "KR",
        "domestic_stock",
    )
    assert _execution_market_for_symbol("AAPL", config) == (
        "NASD",
        "US",
        "overseas_stock",
    )


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
        "rvgi_box_breakout",
    }
    not_triggered = [label for label in labels if not label.triggered]
    assert not_triggered
    assert all(not label.filled for label in not_triggered)
    assert all(label.net_return_bps == 0.0 for label in not_triggered)
    assert all(label.cost_bps == 0.0 for label in not_triggered)
    assert all(
        label.exit_reason == "STRATEGY_NOT_TRIGGERED"
        for label in not_triggered
    )


def test_realtime_strategy_context_v2_contains_rvgi_box_descriptors() -> None:
    labels = build_labels(
        {"TEST": _bars(40)},
        EvaluationConfig(
            history_bars=25,
            horizon_bars=3,
            stride_bars=2,
            minimum_symbol_bars=30,
            feature_schema_name="realtime_strategy_context_v2",
        ),
    )
    assert labels
    assert all(len(label.features) == 27 for label in labels)


def test_realtime_strategy_graph_v4_contains_market_identity() -> None:
    bars = tuple(replace(bar, symbol="005930", venue="KRX") for bar in _bars(40))
    labels = build_labels(
        {"005930": bars},
        EvaluationConfig(
            history_bars=25,
            horizon_bars=3,
            stride_bars=2,
            minimum_symbol_bars=30,
            feature_schema_name="realtime_strategy_graph_v4_market",
        ),
    )

    assert labels
    assert all(len(label.features) == 28 for label in labels)
    assert all(label.features[-1] == 1.0 for label in labels)
