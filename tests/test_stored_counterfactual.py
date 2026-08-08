from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from app.evaluation.stored_counterfactual import (
    EvaluationConfig,
    _barrier_bps,
    _causal_horizon_sigma_bps,
    _execution_market_for_symbol,
    build_labels,
    causal_percentile,
)
from app.strategy.catalog import STRATEGY_IDS
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
    assert {label.strategy_id for label in labels} == set(STRATEGY_IDS)
    not_triggered = [label for label in labels if not label.triggered]
    assert not_triggered
    assert all(not label.filled for label in not_triggered)
    assert all(label.net_return_bps == 0.0 for label in not_triggered)
    assert all(label.cost_bps == 0.0 for label in not_triggered)
    assert all(
        label.exit_reason == "STRATEGY_NOT_TRIGGERED"
        for label in not_triggered
    )


def test_flat_tied_bars_are_not_momentum_breakout_or_reversal_signals() -> None:
    flat = tuple(
        replace(
            bar,
            open=100.0,
            high=100.1,
            low=99.9,
            close=100.0,
            volume=100.0,
        )
        for bar in _bars(15)
    )
    labels = build_labels(
        {"TEST": flat},
        EvaluationConfig(
            history_bars=5,
            horizon_bars=3,
            stride_bars=1,
            minimum_symbol_bars=10,
        ),
    )

    guarded = {
        "intraday_momentum",
        "breakout_volume",
        "liquidity_shock_reversal",
    }
    assert labels
    assert not any(label.triggered for label in labels if label.strategy_id in guarded)


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


def _scale_bars(closes: tuple[float, ...]) -> tuple[Bar, ...]:
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    return tuple(
        Bar(
            symbol="TEST",
            venue="NASD",
            interval="1m",
            start_time=start + timedelta(minutes=index),
            end_time=start + timedelta(minutes=index + 1),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100.0,
            source_event_ids=(),
        )
        for index, close in enumerate(closes)
    )


def test_horizon_sigma_survives_a_thin_tape_where_most_bars_do_not_move() -> None:
    """A median-based scale returns 0 when over half the bars print no change, and
    the caller then silently falls back to the fixed barrier pair. Measured while
    building this: every US name kept a 160.0bps target. The mean does not collapse."""
    closes = tuple(100.0 if index % 3 else 100.0 * (1 + 0.001) for index in range(31))

    sigma = _causal_horizon_sigma_bps(
        _scale_bars(closes), 30, 0, horizon_bars=60
    )

    assert sigma is not None
    assert sigma > 0.0


def test_horizon_sigma_is_none_only_when_nothing_moved_at_all() -> None:
    flat = _causal_horizon_sigma_bps(
        _scale_bars(tuple(100.0 for _ in range(31))), 30, 0, horizon_bars=60
    )
    assert flat is None


def test_horizon_sigma_reads_only_bars_up_to_the_decision_index() -> None:
    """A barrier sized with future bars is a look-ahead leak into every label."""
    quiet_then_violent = tuple(
        [100.0 + 0.01 * index for index in range(31)]
        + [500.0 for _ in range(30)]
    )
    bars = _scale_bars(quiet_then_violent)

    causal = _causal_horizon_sigma_bps(bars, 30, 0, horizon_bars=60)
    if_it_peeked = _causal_horizon_sigma_bps(bars, 60, 0, horizon_bars=60)

    assert causal is not None and if_it_peeked is not None
    assert causal < if_it_peeked


def test_barriers_scale_with_volatility_rather_than_one_fixed_pair() -> None:
    """One fixed pair produced opposite pathologies per market: measured on stored
    bars, stop 60 / target 160bps resolved 7.3% target / 70.2% timeout on US names
    and 21.3% / 69.3% stop on KR ones."""
    calm_target, calm_stop, _ = _barrier_bps(
        sigma_bps=40.0,
        cost_bps=28.0,
        safety_margin_bps=1.0,
        configured_target_bps=160.0,
        configured_stop_bps=60.0,
    )
    wild_target, wild_stop, _ = _barrier_bps(
        sigma_bps=400.0,
        cost_bps=28.0,
        safety_margin_bps=1.0,
        configured_target_bps=160.0,
        configured_stop_bps=60.0,
    )

    assert wild_target > calm_target
    assert wild_stop > calm_stop
    # The calm case must still clear the round trip, never target below cost.
    assert calm_target > 28.0


def test_cost_floor_domination_is_reported_not_hidden() -> None:
    """US: ~88bps needed to clear a 63bps round trip against a median 60-minute
    favourable excursion of 15.7bps. No barrier pair is both payable and reachable,
    and the caller has to be able to record that rather than label a coin flip."""
    target, _stop, dominated = _barrier_bps(
        sigma_bps=25.0,
        cost_bps=63.0,
        safety_margin_bps=1.0,
        configured_target_bps=160.0,
        configured_stop_bps=60.0,
    )
    assert dominated is True
    assert target > 25.0

    _t, _s, not_dominated = _barrier_bps(
        sigma_bps=400.0,
        cost_bps=28.0,
        safety_margin_bps=1.0,
        configured_target_bps=160.0,
        configured_stop_bps=60.0,
    )
    assert not_dominated is False


def test_stop_never_collapses_into_the_tick_grid() -> None:
    _target, stop, _ = _barrier_bps(
        sigma_bps=0.5,
        cost_bps=28.0,
        safety_margin_bps=1.0,
        configured_target_bps=160.0,
        configured_stop_bps=60.0,
    )
    assert stop >= 15.0


def test_flat_bar_does_not_score_as_top_percentile() -> None:
    """A value tied with its whole history is the least informative observation
    there is, so it must land mid-distribution, not at the top.

    Under the old ``sum(item <= value)/n`` a flat bar scored 1.0. Measured on
    stored minute bars 2026-08-07: 52.3% of US bars print exactly zero return, and
    72.8% of all ``intraday_momentum`` triggers were those flat bars — the
    "buy strength" family was mostly buying non-events in illiquid symbols.
    """
    assert causal_percentile(0.0, [0.0] * 30) == 0.5
    assert causal_percentile(0.0, [-0.001] * 15 + [0.0] * 15) == 0.75
    # A genuine move still ranks at the top, which is the property that made the
    # broken definition look correct in testing.
    assert causal_percentile(0.01, [0.0] * 30) == 1.0
    assert causal_percentile(-0.01, [0.0] * 30) == 0.0


def test_percentile_ties_use_midrank_not_upper_bound() -> None:
    history = [1.0, 2.0, 2.0, 2.0, 3.0]
    # below=1, equal=3  ->  (1 + 1.5) / 5
    assert causal_percentile(2.0, history) == 0.5
    assert causal_percentile(1.0, history) == 0.1
    assert causal_percentile(3.0, history) == 0.9


def test_empty_history_is_neutral_not_confident() -> None:
    assert causal_percentile(0.5, []) == 0.5
