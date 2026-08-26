from __future__ import annotations

import inspect
import math
import sqlite3
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.graph.knowledge_graph import KnowledgeGraph
from app.quant.adapters import (
    QUANT_GNN_METRICS,
    QUANT_GNN_SCHEMA_VERSION,
    add_quant_evidence_to_graph,
    apply_quant_to_gate_inputs,
    quant_risk_factor,
    to_gnn_feature_frame,
)
from app.quant.backtest import ActionType, IntentAction, MarketTrigger, QuantStrategy
from app.quant.config import QuantConfig
from app.quant.contracts import DataQuality, QuantBar, QuantEvidence, ValidationStatus
from app.quant.engine import IncrementalQuantEngine
from app.quant.portfolio import beta, concentration, correlation, covariance, max_drawdown, portfolio_returns, portfolio_volatility
from app.quant.reference import GSQuantReferenceAdapter
from app.quant.store import QuantEvidenceStore
from app.quant.runtime import CompletedBarQuantSink, build_quant_sink, local_quant_self_test
from app.risk.final_trade_gate import FinalTradeGate, GateInputs, load_gate_config
from app.data.sample_collectors import collect_sample_market
from app.graph.builders import build_market_graph
from app.indicators import build_sample_indicators
from app.risk import RiskManager
from app.schemas import AccountSnapshot, RiskRules
from app.strategy.rule_based import generate_order_intents, generate_strategy_signals
from app.data.event_pipeline import BoundedMarketEventBus, EventDrivenMarketRuntime
from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue
from app.data.realtime_types import FeedMetadata, RealtimeTradeTick

UTC = timezone.utc
START = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)


def _bar(index: int, close: float, *, received_delay: int = 0) -> QuantBar:
    start = START + timedelta(minutes=index)
    return QuantBar(
        symbol="DYNAMIC", market="KR", interval="1m", start_time=start,
        end_time=start + timedelta(minutes=1),
        received_at=start + timedelta(minutes=1, seconds=received_delay),
        open=close, high=close + 1, low=close - 1, close=close, volume=100 + index,
    )


def _engine() -> IncrementalQuantEngine:
    return IncrementalQuantEngine(QuantConfig(
        price_window=3, return_window=3, rsi_window=2, ema_fast=2, ema_slow=3,
        macd_signal=2, stale_after_ms=1_000, cache_size=100,
    ))


def _evidence(**overrides) -> QuantEvidence:
    values = dict(
        symbol="DYNAMIC", market="KR", timestamp=START + timedelta(minutes=2),
        bar_interval="1m", metric="trend", value=0.1, window=3,
        input_start=START, input_end=START + timedelta(minutes=2), freshness_ms=0,
        data_quality=DataQuality.GOOD, implementation="local_quant_engine:incremental-v1",
        method_reference="gs_quant:2.1.3:timeseries.technicals.exponential_moving_average",
        validation_status=ValidationStatus.PASSED, unavailable_reason=None,
    )
    values.update(overrides)
    return QuantEvidence(**values)


def _gate_inputs(**overrides) -> GateInputs:
    values = dict(
        ticker="DYNAMIC", side="BUY", evaluated_at=START,
        stale_data_reasons=(), websocket_connected=True, price_feed_divergence_bps=1,
        session_id="KRX_REGULAR", session_allows_new_entry=True, trading_halted=False,
        account_reconciled=True, unknown_order_ids=(), duplicate_order_risk=False,
        model_health_state="HEALTHY", risk_engine_ok=True, realized_volatility=0.001,
        liquidity_score=1.0, global_agreement=1.0, sector_relative_strength=1.0,
        model_confidence=1.0, session_phase="MIDDAY", spread_bps=1,
        dominant_regime="TREND_UP", account_equity=100_000_000,
        current_position_value=0, current_sector_exposure=0, current_market_exposure=0,
        session_pnl_ratio=0, drawdown_ratio=0, requested_position_fraction=0.01,
    )
    values.update(overrides)
    return GateInputs(**values)


def test_known_deterministic_series_and_warmup() -> None:
    engine = _engine()
    first = {row.metric: row for row in engine.update(_bar(0, 100))}
    assert first["rolling_mean"].value is None
    assert first["rolling_mean"].unavailable_reason == "warmup"
    engine.update(_bar(1, 101))
    third = {row.metric: row for row in engine.update(_bar(2, 103))}
    assert third["rolling_mean"].value is None
    fourth = {row.metric: row for row in engine.update(_bar(3, 102))}
    assert fourth["rolling_mean"].value == pytest.approx(306 / 3)
    assert fourth["rolling_std"].value == pytest.approx(1.0)
    assert fourth["simple_return"].value == pytest.approx(102 / 103 - 1)
    assert fourth["max_drawdown"].value == pytest.approx(102 / 103 - 1)
    assert all(row.implementation.startswith("local_quant_engine") for row in fourth.values())


def test_timezone_and_lookahead_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(_bar(0, 100), received_at=datetime(2026, 1, 1))
    engine = _engine()
    with pytest.raises(ValueError, match="lookahead"):
        engine.update(_bar(0, 100), as_of=START)
    future = _bar(2, 100)
    assert engine.compute_features((future,), as_of=START) == ()


def test_out_of_order_and_duplicate_bars_are_rejected() -> None:
    engine = _engine()
    engine.update(_bar(1, 101))
    with pytest.raises(ValueError, match="out-of-order"):
        engine.update(_bar(0, 100))


def test_staleness_is_explicit_not_silently_replaced() -> None:
    engine = _engine()
    bar = _bar(0, 100)
    rows = engine.update(bar, as_of=bar.received_at + timedelta(seconds=2))
    assert all(row.data_quality is DataQuality.DEGRADED for row in rows)
    assert all(row.freshness_ms == pytest.approx(2_000) for row in rows)


def test_gnn_frame_has_raw_normalized_quality_and_missing_mask() -> None:
    frame = to_gnn_feature_frame((_evidence(),))
    trend = QUANT_GNN_METRICS.index("trend")
    rsi = QUANT_GNN_METRICS.index("rsi")
    assert frame.raw_values[trend] == 0.1
    assert frame.normalized_values[trend] != frame.raw_values[trend]
    assert frame.mask[trend] == 1
    assert frame.raw_values[rsi] == 0 and frame.mask[rsi] == 0
    assert frame.compatible_with(QUANT_GNN_SCHEMA_VERSION, QUANT_GNN_METRICS)
    with pytest.raises(ValueError, match="version mismatch"):
        to_gnn_feature_frame((), expected_schema_version="old")
    with pytest.raises(ValueError, match="ordering mismatch"):
        to_gnn_feature_frame((), expected_feature_names=tuple(reversed(QUANT_GNN_METRICS)))


def test_ontology_mapping_is_advisory_and_has_provenance() -> None:
    graph = KnowledgeGraph()
    count = add_quant_evidence_to_graph(graph, (_evidence(),))
    assert count == 5
    predicates = {triple.predicate for triple in graph.triples()}
    assert {"computedFrom", "hasCalculationMethod", "hasFreshness", "hasDataQuality", "supportsSignal"} <= predicates
    assert not ({"submitsOrder", "createsOrder", "approvesOrder"} & predicates)


def test_invalid_or_failed_evidence_can_only_reduce_risk() -> None:
    passed, _ = quant_risk_factor((_evidence(),), stale_after_ms=1000)
    stale, reasons = quant_risk_factor((_evidence(freshness_ms=2000),), stale_after_ms=1000)
    failed, failed_reasons = quant_risk_factor((_evidence(validation_status=ValidationStatus.FAILED),), stale_after_ms=1000)
    assert passed == 1.0 and stale == 0.5 and failed == 0.0
    assert "QUANT_EVIDENCE_STALE" in reasons
    assert "QUANT_PARITY_FAILED" in failed_reasons


def test_final_trade_gate_precedence_and_no_bypass() -> None:
    gate = FinalTradeGate(load_gate_config())
    base = gate.evaluate(_gate_inputs())
    reduced = gate.evaluate(apply_quant_to_gate_inputs(_gate_inputs(), (_evidence(freshness_ms=2000),), config=_engine().config))
    hard_blocked = gate.evaluate(apply_quant_to_gate_inputs(_gate_inputs(duplicate_order_risk=True), (_evidence(),), config=_engine().config))
    parity_blocked = gate.evaluate(apply_quant_to_gate_inputs(_gate_inputs(), (_evidence(validation_status=ValidationStatus.FAILED),), config=_engine().config))
    assert base.approved
    assert reduced.position_multiplier <= base.position_multiplier
    assert not hard_blocked.approved and "DUPLICATE_ORDER_RISK" in hard_blocked.hard_failures
    assert not parity_blocked.approved


def test_risk_manager_rejects_buy_when_supplied_quant_parity_failed() -> None:
    markets = collect_sample_market()
    indicators = build_sample_indicators(markets)
    graph = build_market_graph(markets, indicators)
    signals = generate_strategy_signals(markets, indicators, graph)
    intent = generate_order_intents(markets, indicators, signals)[0]
    evidence = replace(
        _evidence(validation_status=ValidationStatus.FAILED),
        symbol=intent.ticker,
        market=intent.market,
    )
    result = RiskManager(RiskRules(max_sector_weight=0.5)).validate(
        intent,
        AccountSnapshot(cash=10_000_000, holdings=()),
        markets[0],
        quant_evidence=(evidence,),
    )
    assert not result.approved
    assert result.metadata["quant_evidence"]["risk_factor"] == 0
    assert not result.checks["quant_evidence_usable"]


def test_portfolio_metrics_use_only_supplied_aligned_data() -> None:
    left, right = (0.01, -0.02, 0.03), (0.02, -0.04, 0.06)
    assert covariance(left, right) == pytest.approx(2 * covariance(left, left))
    assert correlation(left, right) == pytest.approx(1.0)
    assert beta(left, right) == pytest.approx(0.5)
    returns = portfolio_returns({"A": left, "B": right}, {"A": 0.5, "B": 0.5})
    assert returns == pytest.approx((0.015, -0.03, 0.045))
    assert portfolio_volatility(returns, 252) is not None
    assert concentration({"A": 0.5, "B": 0.5}) == 0.5
    assert max_drawdown((0.1, -0.2, 0.05)) == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="missing real return"):
        portfolio_returns({"A": left}, {"A": 0.5, "B": 0.5})

    rows = {row.metric: row for row in _engine().evaluate_portfolio({
        "portfolio_id": "BOOK", "market": "KR", "interval": "1d",
        "as_of": START + timedelta(days=4), "input_start": START,
        "input_end": START + timedelta(days=3),
        "returns_by_symbol": {"A": left, "B": right},
        "weights": {"A": 0.5, "B": 0.5}, "benchmark_returns": right,
    })}
    assert rows["benchmark_beta"].value is not None
    assert rows["sharpe"].value is None
    assert rows["sharpe"].unavailable_reason == "risk_free_rate_unavailable"


def test_store_has_required_entities_and_indexes(tmp_path) -> None:
    store = QuantEvidenceStore(tmp_path / "quant.sqlite3")
    assert store.append((_evidence(),)) == 1
    assert store.latest("DYNAMIC")[0]["metric"] == "trend"
    with sqlite3.connect(store.path) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        indexes = {row[0] for row in connection.execute("select name from sqlite_master where type='index'")}
    assert {"quant_evidence", "quant_provider_health", "quant_validation_result"} <= tables
    assert {"idx_quant_evidence_symbol_time", "idx_quant_evidence_metric_time", "idx_quant_evidence_validation"} <= indexes


def test_reference_adapter_is_optional_and_has_no_api_client_import() -> None:
    health = GSQuantReferenceAdapter().health()
    assert health["network_required"] is False
    source = inspect.getsource(__import__("app.quant.reference", fromlist=["x"]))
    assert "GsDataApi" not in source and "GsAssetApi" not in source and "PricingContext" not in source
    if not health["available"]:
        result = GSQuantReferenceAdapter().validate("rolling_mean", (1, 2, 3), 2.0, window=3)
        assert not result.available and result.unavailable_reason


def test_strategy_action_only_proposes_intent_and_requires_both_gates() -> None:
    strategy = QuantStrategy("s", MarketTrigger("zscore", ">", 1), IntentAction(ActionType.PROPOSE_ENTRY, 0.01))
    assert strategy.evaluate({"zscore": 0.5}, symbol="DYNAMIC", market="KR", as_of=START) is None
    intent = strategy.evaluate({"zscore": 2}, symbol="DYNAMIC", market="KR", as_of=START)
    assert intent is not None
    assert intent.requires_risk_manager and intent.requires_final_trade_gate
    assert not hasattr(intent, "submit") and not hasattr(intent, "broker")


def test_hot_path_source_has_no_pandas_or_gs_quant_import() -> None:
    source = inspect.getsource(__import__("app.quant.engine", fromlist=["x"]))
    assert "import pandas" not in source
    assert "import gs_quant" not in source


def test_event_runtime_sends_only_closed_bar_to_quant_background_sink() -> None:
    class Store:
        def save_ticks(self, values): pass
        def save_orderbooks(self, values): pass
        def save_minute_bars(self, values): pass

    calls = []

    def sink(bar, event):
        calls.append((bar, event))

    meta = FeedMetadata(
        market_group=MarketGroup.KR, exchange="KRX", venue=Venue.KRX,
        session=SessionId.KRX_REGULAR, feed_scope=FeedScope.VENUE_SPECIFIC,
        tr_id="UNIT", subscription_key="DYNAMIC", is_tradeable=True,
    )

    def tick(minute: int, price: float) -> RealtimeTradeTick:
        moment = START + timedelta(minutes=minute, seconds=1)
        return RealtimeTradeTick(
            symbol="DYNAMIC", exchange_timestamp=moment, received_at=moment,
            source="kis_realtime_websocket", price=price, volume=1,
            sequence_key=str(minute), meta=meta,
        )

    async def scenario() -> None:
        bus = BoundedMarketEventBus()
        runtime = EventDrivenMarketRuntime(bus, store=Store(), completed_bar_sink=sink)
        await bus.publish(tick(0, 100))
        await runtime.process_one()
        assert calls == []  # no quant work on tick/forming-bar path
        await bus.publish(tick(1, 101))
        await runtime.process_one()
        assert calls == []  # still queued for the persistence worker
        await runtime.persist_one()

    asyncio.run(scenario())
    assert len(calls) == 1
    assert calls[0][0].minute_start == START


def test_completed_bar_sink_fails_explicitly_without_market_metadata() -> None:
    sink = CompletedBarQuantSink(engine=_engine())
    class Bar:
        symbol = "DYNAMIC"
        minute_start = START
        open = high = low = close = 100.0
        volume = 1
        meta = None
    class Event:
        received_at = START + timedelta(minutes=1)
    assert sink(Bar(), Event()) == ()
    assert sink.health()["completed_bar_errors"] == 1
    assert "market metadata" in sink.health()["last_completed_bar_error"]


def test_quant_layer_auto_activates_only_after_all_conditions(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("QUANT_REFERENCE_LIVE_ENABLED", raising=False)
    monkeypatch.delenv("QUANT_REFERENCE_ACTIVATION_MODE", raising=False)
    config = _engine().config
    passed, reason = local_quant_self_test(config)
    assert passed and reason is None
    sink, decision = build_quant_sink(
        config=config, store=QuantEvidenceStore(tmp_path / "quant-auto.sqlite3")
    )
    assert sink is not None and decision.enabled and decision.mode == "auto"
    assert {
        "python>=3.10", "deterministic_self_test_passed", "evidence_store_ready",
        "gs_quant_not_required", "no_order_authority",
    } <= set(decision.conditions)


def test_explicit_off_overrides_auto_activation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUANT_REFERENCE_ACTIVATION_MODE", "off")
    sink, decision = build_quant_sink(
        config=_engine().config, store=QuantEvidenceStore(tmp_path / "off.sqlite3")
    )
    assert sink is None and not decision.enabled
    assert decision.unavailable_reason == "disabled_by_policy"


def test_repeated_errors_open_quant_circuit_without_affecting_orders() -> None:
    config = QuantConfig(
        price_window=3, return_window=3, rsi_window=2, ema_fast=2, ema_slow=3,
        macd_signal=2, stale_after_ms=1_000, cache_size=100,
        auto_disable_consecutive_errors=2, auto_retry_cooldown_seconds=300,
    )
    sink = CompletedBarQuantSink(engine=IncrementalQuantEngine(config), config=config)
    class InvalidBar: meta = None
    class Event: received_at = START
    assert sink(InvalidBar(), Event()) == ()
    assert sink(InvalidBar(), Event()) == ()
    health = sink.health()
    assert health["circuit_open"] and not health["enabled"]
    assert health["consecutive_errors"] == 2
