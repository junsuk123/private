from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
