from __future__ import annotations

from datetime import datetime, timezone

from app.strategy_validation.bear_cash_equity import (
    BearValidationConfig,
    BearValidationTrade,
    _metrics,
)


def _trade(day: int, net: float, *, gross: float = 100.0) -> BearValidationTrade:
    moment = datetime(2026, 8, day, 1, tzinfo=timezone.utc)
    return BearValidationTrade(
        symbol="AAPL",
        market="US",
        signal_at=moment,
        exit_at=moment,
        gross_bps=gross,
        net_bps=net,
        cost_bps=gross - net,
        stressed_net_bps=gross - 1.25 * (gross - net),
        exit_reason="TARGET",
        holding_seconds=60.0,
    )


def test_validation_metrics_measure_days_lcb_and_cost_stress():
    rows = [_trade(day, 50.0 + day) for day in range(1, 6) for _ in range(6)]

    metrics = _metrics(rows, BearValidationConfig())

    assert metrics["samples"] == 30
    assert metrics["distinct_days"] == 5
    assert metrics["positive_day_fraction"] == 1.0
    assert metrics["lower_confidence_bound_bps"] > 0.0
    assert metrics["cost_stressed_mean_net_bps"] > 0.0
