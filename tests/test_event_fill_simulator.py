from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from app.backtesting.event_simulator import EventDrivenFillSimulator
from app.trading.contracts import Bar, TradePlan


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _plan() -> TradePlan:
    return TradePlan(
        strategy_id="momentum",
        strategy_instance_id="instance-1",
        symbol="005930",
        side="BUY",
        thesis="unit",
        entry_trigger={},
        entry_price_policy={"limit": 100},
        proposed_quantity=10,
        initial_stop={"price": 98},
        profit_policy={"price": 103},
        trailing_policy={},
        max_holding_seconds=120,
        invalidation_conditions=(),
        max_entry_slippage_bps=5,
        expires_at=NOW + timedelta(seconds=60),
        feature_snapshot_id="f",
        utility_evidence_id="u",
    )


def _bar(index: int, low: float, high: float, close: float) -> Bar:
    start = NOW + timedelta(minutes=index)
    return Bar(
        symbol="005930",
        venue="KRX",
        interval="1m",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        open=min(max(100, low), high),
        high=high,
        low=low,
        close=close,
        volume=1000,
    )


def test_cost_adjusted_counterfactual_and_no_trade_label() -> None:
    outcomes = EventDrivenFillSimulator().counterfactual_matrix(
        (_plan(),), (_bar(0, 99, 101, 100), _bar(1, 100, 104, 103)), as_of=NOW
    )
    trade, no_trade = outcomes
    assert trade.filled
    assert trade.exit_reason == "PROFIT_TARGET"
    assert trade.net_return_bps < trade.gross_return_bps
    assert trade.cost_bps > 0
    assert no_trade.strategy_id == "NO_TRADE"
    assert no_trade.net_return_bps == 0


def test_simulator_uses_shared_all_in_cost_and_stored_spread(monkeypatch) -> None:
    seen: list[tuple[str, float | None, float]] = []

    def shared_cost(symbol, *, spread_bps=None, fallback_bps=0.0):
        seen.append((symbol, spread_bps, fallback_bps))
        return 73.7

    monkeypatch.setattr(
        "app.backtesting.event_simulator.all_in_round_trip_bps",
        shared_cost,
    )
    outcome = EventDrivenFillSimulator().simulate(
        _plan(),
        (_bar(0, 99, 101, 100), _bar(1, 100, 104, 103)),
        as_of=NOW,
        spread_bps=7.5,
    )

    assert outcome.cost_bps == pytest.approx(73.7)
    assert seen
    assert all(symbol == "005930" and spread == 7.5 for symbol, spread, _ in seen)


def test_same_bar_stop_and_target_uses_conservative_stop() -> None:
    outcome = EventDrivenFillSimulator().simulate(
        _plan(), (_bar(0, 97, 104, 102),), as_of=NOW
    )
    assert outcome.exit_reason == "INITIAL_STOP"
    assert outcome.exit_price == 98


def test_future_event_does_not_change_earlier_entry_fill() -> None:
    simulator = EventDrivenFillSimulator()
    early = (_bar(0, 101, 102, 101.5),)
    first = simulator.simulate(_plan(), early, as_of=NOW)
    later = early + (_bar(2, 90, 110, 100),)
    second = simulator.simulate(_plan(), later, as_of=NOW)
    assert not first.filled
    assert not second.filled


def test_incomplete_future_is_censored_not_fake_time_exit() -> None:
    outcome = EventDrivenFillSimulator().simulate(
        _plan(), (_bar(0, 99, 101, 100),), as_of=NOW
    )

    assert outcome.filled
    assert outcome.exit_reason == "FUTURE_WINDOW_CENSORED"


def test_simulator_replays_trailing_stop_from_prior_watermark() -> None:
    plan = replace(
        _plan(),
        profit_policy={"price": 103.4},
        trailing_policy={"bps": 100},
        max_holding_seconds=300,
    )
    outcome = EventDrivenFillSimulator().simulate(
        plan,
        (
            _bar(0, 99, 103, 102),
            _bar(1, 101, 102.5, 101.5),
        ),
        as_of=NOW,
    )

    assert outcome.exit_reason == "TRAILING_STOP"
    assert outcome.exit_price == 103 * 0.99
    assert outcome.gross_return_bps > 0


def test_short_target_and_return_are_direction_aware() -> None:
    short = replace(
        _plan(),
        side="SELL",
        position_direction="SHORT",
        execution_product="CREDIT_BORROW",
        initial_stop={"price": 102},
        profit_policy={"price": 97},
    )
    outcome = EventDrivenFillSimulator().simulate(
        short, (_bar(0, 99, 101, 100), _bar(1, 96, 101, 97)), as_of=NOW
    )

    assert outcome.exit_reason == "PROFIT_TARGET"
    assert outcome.exit_price == 97
    assert outcome.gross_return_bps > 0
