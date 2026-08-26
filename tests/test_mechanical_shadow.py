from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.realtime_types import KIS_REALTIME_SOURCE, OrderbookLevel, RealtimeOrderbookSnapshot
from app.features.feature_provenance import FeatureProvenance
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import LiveFeatureFrame
from app.trading.mechanical_shadow import MechanicalShadowCollector


class _MarketStore:
    def __init__(self, book):
        self.book = book

    def latest_orderbook(self, symbol):
        return self.book if self.book.symbol == symbol else None


class _PlanStore:
    def __init__(self):
        self.plans = []

    def record_plan(self, plan):
        self.plans.append(plan)
        return True


class _EvaluationService:
    def __init__(self):
        self.plans = []

    def adopt(self, plans):
        accepted = tuple(plans)
        self.plans.extend(accepted)
        return len(accepted), ()


def _frame(
    now: datetime,
    *,
    symbol: str = "SOFI",
    return_30s: float = 0.0004,
    realized_volatility_10s: float = 0.001,
):
    values = {name: 0.0 for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names}
    values.update(
        {
            "return_30s": return_30s,
            "orderbook_imbalance": -0.40,
            "spread_change_5s": -0.2,
            "realized_volatility_10s": realized_volatility_10s,
            "tick_count_5s": 5.0,
            "second_data_ready": 1.0,
            "liquidity_score": 0.8,
            "spread_bps": 5.0,
        }
    )
    return LiveFeatureFrame(
        symbol=symbol,
        decision_time=now,
        schema=LIVE_SHORT_HORIZON_SCHEMA,
        values=tuple(values[name] for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names),
        provenance=FeatureProvenance(
            symbol=symbol,
            decision_time=now,
            tick_record_ids=("tick-1",),
            orderbook_record_id="book-1",
            source=KIS_REALTIME_SOURCE,
            max_input_age_ms=100.0,
        ),
        mark_price=10.0,
    )


def _book(now: datetime, symbol: str = "SOFI"):
    return RealtimeOrderbookSnapshot(
        symbol=symbol,
        exchange_timestamp=now - timedelta(milliseconds=50),
        received_at=now - timedelta(milliseconds=25),
        source=KIS_REALTIME_SOURCE,
        levels=(OrderbookLevel(9.99, 1_000, 10.01, 900),),
    )


def test_absorption_signal_is_journaled_and_adopted_without_order_path():
    now = datetime(2026, 8, 3, 14, 30, tzinfo=timezone.utc)
    plan_store = _PlanStore()
    evaluation = _EvaluationService()
    collector = MechanicalShadowCollector(
        market_store=_MarketStore(_book(now)),
        shadow_store=plan_store,
        evaluation_service=evaluation,
        cooldown_seconds=600,
    )

    result = collector.collect(
        (_frame(now),), observed_at=now, regime="TREND_UP"
    )

    assert result.evaluated == result.triggered == result.recorded == result.adopted == 1
    plan = plan_store.plans[0]
    assert plan.entry_reference_price == 10.01
    # The 600s absorption horizon is a FLOOR, not a constant: the barriers are sized
    # against this book's measured round-trip cost, and a target raised to clear that
    # cost needs proportionally more time to be reachable. Capped by the submode's own
    # ``max_absorption_horizon_seconds``.
    assert 600 <= plan.max_holding_seconds <= 3600
    assert plan.deployment_state == "SHADOW"
    assert plan.regime == "TREND_UP"
    # The invariant that matters, asserted at the cost actually measured rather than
    # at the KRX reference constant. Asserting only the reference is how the table
    # kept reading 1.53 while paying 0.83 on the tape it was being scored on.
    geometry = plan.diagnostics["exit_geometry"]
    assert geometry["cost_relative"] is True
    assert geometry["resolved_net_reward_risk_ratio"] >= 1.45
    assert plan.diagnostics["order_submission_capable"] is False
    assert "GNN_INDEPENDENT_MECHANICAL_SHADOW" in plan.signal_reason_codes
    assert evaluation.plans == plan_store.plans

    duplicate = collector.collect((_frame(now + timedelta(seconds=30)),), observed_at=now + timedelta(seconds=30))
    assert duplicate.triggered == 1
    assert duplicate.recorded == duplicate.adopted == 0


def test_absorption_without_market_regime_is_quarantined_as_unknown():
    now = datetime(2026, 8, 3, 14, 30, tzinfo=timezone.utc)
    plan_store = _PlanStore()
    collector = MechanicalShadowCollector(
        market_store=_MarketStore(_book(now)),
        shadow_store=plan_store,
        evaluation_service=_EvaluationService(),
    )

    result = collector.collect((_frame(now),), observed_at=now)

    assert result.recorded == 1
    assert plan_store.plans[0].regime == "UNKNOWN"


def test_absorption_is_not_collected_outside_us_regular_session():
    after_close = datetime(2026, 8, 3, 21, 58, tzinfo=timezone.utc)
    plan_store = _PlanStore()
    collector = MechanicalShadowCollector(
        market_store=_MarketStore(_book(after_close)),
        shadow_store=plan_store,
        evaluation_service=_EvaluationService(),
    )

    result = collector.collect((_frame(after_close),), observed_at=after_close)

    assert result == type(result)()
    assert not plan_store.plans


def test_raw_absorption_is_collected_when_sparse_rest_ticks_cannot_price_edge():
    now = datetime(2026, 8, 3, 14, 30, tzinfo=timezone.utc)
    plan_store = _PlanStore()
    collector = MechanicalShadowCollector(
        market_store=_MarketStore(_book(now)),
        shadow_store=plan_store,
        evaluation_service=_EvaluationService(),
    )

    result = collector.collect(
        (_frame(now, realized_volatility_10s=0.0),),
        observed_at=now,
    )

    assert result.recorded == result.adopted == 1
    assert plan_store.plans[0].diagnostics["algorithm_live_triggered"] is False
    assert 600 <= plan_store.plans[0].max_holding_seconds <= 3600
    # The floor that rejected it is the market's round-trip cost, so the reason
    # code names the cost rather than an arbitrary threshold.
    assert "EDGE_BELOW_COST_FLOOR" in plan_store.plans[0].signal_reason_codes


def test_collector_rejects_weak_recovery_krx_and_future_book():
    now = datetime(2026, 8, 3, 14, 30, tzinfo=timezone.utc)
    plan_store = _PlanStore()
    evaluation = _EvaluationService()
    future_book = _book(now + timedelta(seconds=1))
    collector = MechanicalShadowCollector(
        market_store=_MarketStore(future_book),
        shadow_store=plan_store,
        evaluation_service=evaluation,
    )

    result = collector.collect(
        (
            _frame(now, return_30s=0.0001),
            _frame(now, symbol="005930"),
            _frame(now),
        ),
        observed_at=now,
    )

    assert result.evaluated == 3
    assert result.triggered == 1
    assert result.recorded == result.adopted == 0
    assert not plan_store.plans
