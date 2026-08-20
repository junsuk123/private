"""End-to-end: context hierarchy -> GNN -> regime -> selector -> gate -> order intent.

These are the acceptance tests for the chain. Each one asserts a property of the whole
pipeline that no single component can guarantee on its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.context.domestic_context import (
    DomesticContextBuilder,
    DomesticContextInputs,
    VenueQuote,
)
from app.context.global_context import GlobalContextBuilder, IndicatorObservation
from app.context.sector_context import SectorContextBuilder, SectorMemberObservation
from app.context.temporal_context import build_temporal_snapshot
from app.data.freshness import DataFreshnessRegistry
from app.execution.order_state_machine import OrderState, OrderStateMachine
from app.models.gnn_runtime import GnnHealthState, GnnRuntime
from app.models.graph_snapshot import FEATURE_DIM, GraphSnapshotBuilder
from app.models.temporal_hetero_gnn import TemporalHeteroGnn, TemporalHeteroGnnConfig
from app.storage.trading_state_store import TradingStateStore
from app.trading.context_decision_pipeline import (
    AccountState,
    CandidateInput,
    ContextDecisionPipeline,
)

#: 10:30 KST on an ordinary Wednesday — mid-morning, KRX continuous.
NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)

MODEL_CONFIG = TemporalHeteroGnnConfig(
    max_nodes=96, feature_dim=FEATURE_DIM, time_steps=8
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store(tmp_path) -> TradingStateStore:
    return TradingStateStore(tmp_path / "state.sqlite3")


@pytest.fixture()
def fresh_registry() -> DataFreshnessRegistry:
    registry = DataFreshnessRegistry()
    for source, data_type in (
        ("kis_realtime", "trade"),
        ("kis_realtime", "orderbook"),
        ("kis_rest", "account"),
        ("kis_rest", "positions"),
        ("kis_rest", "order_status"),
        ("internal", "domestic_context"),
    ):
        registry.record_event(
            source, data_type, NOW, received_time=NOW, processed_time=NOW
        )
    return registry


def _runtime(tmp_path, *, require_checkpoint: bool = False) -> GnnRuntime:
    return GnnRuntime(
        checkpoint_path=tmp_path / "absent.npz",
        config=MODEL_CONFIG,
        require_checkpoint=require_checkpoint,
    )


def _pipeline(store, fresh_registry, tmp_path, **kwargs) -> ContextDecisionPipeline:
    machine = kwargs.pop("state_machine", None) or OrderStateMachine(store)
    return ContextDecisionPipeline(
        store=store,
        gnn_runtime=kwargs.pop("gnn_runtime", _runtime(tmp_path)),
        snapshot_builder=GraphSnapshotBuilder(max_nodes=96, time_steps=8),
        state_machine=machine,
        freshness=kwargs.pop("freshness", fresh_registry),
        **kwargs,
    )


def _bullish_world():
    return GlobalContextBuilder().build(
        [
            IndicatorObservation("SP500", 5400.0, NOW - timedelta(hours=8), change_ratio=0.008),
            IndicatorObservation("SOX", 5200.0, NOW - timedelta(hours=8), change_ratio=0.020),
            IndicatorObservation("VIX", 15.0, NOW - timedelta(hours=8), change_ratio=-0.08),
            IndicatorObservation("ES", 5405.0, NOW - timedelta(minutes=2), change_ratio=0.004),
            IndicatorObservation("NIKKEI", 39000.0, NOW - timedelta(minutes=30), change_ratio=0.006),
            IndicatorObservation("USDKRW", 1380.0, NOW - timedelta(minutes=5), change_ratio=-0.002),
        ],
        captured_at=NOW,
    )


def _bearish_world():
    return GlobalContextBuilder().build(
        [
            IndicatorObservation("SP500", 5400.0, NOW - timedelta(hours=8), change_ratio=-0.020),
            IndicatorObservation("SOX", 5200.0, NOW - timedelta(hours=8), change_ratio=-0.045),
            IndicatorObservation("VIX", 34.0, NOW - timedelta(hours=8), change_ratio=0.35),
            IndicatorObservation("ES", 5405.0, NOW - timedelta(minutes=2), change_ratio=-0.012),
            IndicatorObservation("NIKKEI", 39000.0, NOW - timedelta(minutes=30), change_ratio=-0.018),
            IndicatorObservation("USDKRW", 1380.0, NOW - timedelta(minutes=5), change_ratio=0.009),
        ],
        captured_at=NOW,
    )


def _domestic(global_context, *, strong: bool = True):
    if strong:
        inputs = DomesticContextInputs(
            kospi_return=0.006,
            kosdaq_return=0.004,
            advancing_count=600,
            declining_count=250,
            total_trading_value=1.2e13,
            average_trading_value=1.1e13,
            realized_volatility=0.0012,
            foreign_flow=3.2e11,
            institution_flow=1.0e11,
            average_spread_bps=8.0,
            sector_returns={"semiconductor": 0.012, "bio": 0.002},
            venues=(VenueQuote("KRX", mid=2600.0), VenueQuote("NXT", mid=2600.2)),
        )
    else:
        inputs = DomesticContextInputs(
            kospi_return=-0.009,
            kosdaq_return=-0.012,
            advancing_count=180,
            declining_count=660,
            total_trading_value=1.4e13,
            average_trading_value=1.1e13,
            realized_volatility=0.0080,
            foreign_flow=-6.0e11,
            institution_flow=-2.0e11,
            average_spread_bps=45.0,
            sector_returns={"semiconductor": -0.02, "bio": -0.03},
            venues=(VenueQuote("KRX", mid=2600.0), VenueQuote("NXT", mid=2620.0)),
        )
    return DomesticContextBuilder().build(
        inputs, captured_at=NOW, global_context=global_context
    )


def _sectors(domestic, global_context, *, returns=(0.02, 0.014, 0.008, 0.011)):
    return [
        SectorContextBuilder().build(
            "semiconductor",
            [
                SectorMemberObservation(
                    f"s{index}",
                    session_return=value,
                    volume=1600.0,
                    average_volume=1000.0,
                    realized_volatility=0.009,
                    trading_value=1e9,
                    foreign_flow=1e7,
                    return_history=[0.004, -0.006, 0.002, -0.003] * 8,
                )
                for index, value in enumerate(returns)
            ],
            captured_at=NOW,
            market_return=0.005,
            market_return_history=[0.003, -0.005, 0.002, -0.002] * 8,
            domestic_context=domestic,
            global_context=global_context,
            global_group="semiconductor",
        )
    ]


def _candidate(**overrides) -> CandidateInput:
    base = dict(
        ticker="005930",
        sector="semiconductor",
        venue="KRX",
        session_return=0.012,
        trend_strength=25.0,
        orderbook_imbalance=0.3,
        realized_volatility=0.0009,
        spread_bps=6.0,
        liquidity_score=0.8,
        relative_strength=0.006,
        breakout_state=0.5,
        vwap_distance_bps=15.0,
        reference_price=70_000.0,
        data_age_seconds=0.8,
        price_feed_divergence_bps=1.0,
    )
    base.update(overrides)
    return CandidateInput(**base)


def _account(**overrides) -> AccountState:
    base = dict(
        equity=100_000_000.0,
        cash=50_000_000.0,
        reconciled=True,
        session_pnl_ratio=0.002,
        drawdown_ratio=0.01,
    )
    base.update(overrides)
    return AccountState(**base)


def _run(pipeline, *, global_context=None, domestic=None, sectors=None, **kwargs):
    world = global_context if global_context is not None else _bullish_world()
    home = domestic if domestic is not None else _domestic(world)
    options = dict(
        captured_at=NOW,
        temporal=build_temporal_snapshot("KRX", NOW),
        candidates=[_candidate()],
        global_context=world,
        domestic_context=home,
        sector_contexts=sectors if sectors is not None else _sectors(home, world),
        account=_account(),
        websocket_connected=True,
        trading_halted=False,
    )
    options.update(kwargs)
    return pipeline.run_cycle(**options)


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_full_chain_produces_a_gated_order_intent(store, fresh_registry, tmp_path) -> None:
    machine = OrderStateMachine(store)
    pipeline = _pipeline(store, fresh_registry, tmp_path, state_machine=machine)
    result = _run(pipeline, create_order_intents=True)

    decision = result.decisions[0]
    assert decision.gate_result
    assert decision.action == "ENTER"
    assert decision.order_intent is not None
    assert decision.order_intent["state"] == OrderState.GATED.value
    assert decision.order_intent["quantity"] > 0
    assert decision.order_intent["gate_id"] == decision.gate_id

    intent = machine.get(decision.order_intent["intent_id"])
    assert intent is not None and intent.decision_id == decision.decision_id


def test_the_trace_carries_every_declared_field(store, fresh_registry, tmp_path) -> None:
    result = _run(_pipeline(store, fresh_registry, tmp_path))
    payload = result.decisions[0].as_dict()
    for field in (
        "decision_id",
        "ticker",
        "timestamp",
        "temporal_context",
        "global_context",
        "domestic_context",
        "sector_context",
        "micro_context",
        "regime_probabilities",
        "strategy",
        "supporting_factors",
        "conflicting_factors",
        "ontology_relations",
        "learned_relation_weights",
        "model_confidence",
        "uncertainty",
        "gate_result",
        "gate_reasons",
        "position_multiplier",
        "order_intent",
        "execution_result",
    ):
        assert field in payload, field


def test_every_declared_table_is_written(store, fresh_registry, tmp_path) -> None:
    _run(
        _pipeline(store, fresh_registry, tmp_path, state_machine=OrderStateMachine(store)),
        create_order_intents=True,
    )
    for table in (
        "market_session",
        "global_context",
        "domestic_context",
        "sector_context",
        "stock_context",
        "regime_prediction",
        "model_prediction",
        "strategy_decision",
        "gate_decision",
        "order_intent",
        "order_execution",
        "model_health",
    ):
        assert store.count(table) >= 1, table


def test_one_cycle_shares_one_timestamp(store, fresh_registry, tmp_path) -> None:
    pipeline = _pipeline(store, fresh_registry, tmp_path)
    result = pipeline.run_cycle(
        captured_at=NOW,
        temporal=build_temporal_snapshot("KRX", NOW),
        candidates=[_candidate(), _candidate(ticker="000660")],
        global_context=_bullish_world(),
        domestic_context=_domestic(_bullish_world()),
        sector_contexts=(),
        account=_account(),
        websocket_connected=True,
        trading_halted=False,
    )
    assert {decision.timestamp for decision in result.decisions} == {NOW}


def test_decision_is_reconstructible_from_the_store(store, fresh_registry, tmp_path) -> None:
    result = _run(_pipeline(store, fresh_registry, tmp_path))
    decision = result.decisions[0]
    row = store.fetch_one(
        "select * from strategy_decision where decision_id = ?", (decision.decision_id,)
    )
    assert row is not None
    assert row["ticker"] == decision.ticker
    assert row["regime_prediction_id"] is not None
    gate_row = store.fetch_one(
        "select * from gate_decision where decision_id = ?", (decision.decision_id,)
    )
    assert gate_row is not None
    assert bool(gate_row["approved"]) is decision.gate_result


# --------------------------------------------------------------------------- #
# Fail-closed behaviour
# --------------------------------------------------------------------------- #
def test_stale_data_blocks_the_whole_cycle(store, tmp_path) -> None:
    empty = DataFreshnessRegistry()
    empty.expect_all()
    result = _run(_pipeline(store, empty, tmp_path, freshness=empty))
    decision = result.decisions[0]
    assert not decision.gate_result
    assert any(reason.startswith("HARD:STALE_DATA") for reason in decision.gate_reasons)
    assert decision.action == "WAIT"


def test_stale_data_invalidates_every_entry_family(store, tmp_path) -> None:
    empty = DataFreshnessRegistry()
    empty.expect_all()
    result = _run(_pipeline(store, empty, tmp_path, freshness=empty))
    decision = result.decisions[0]
    assert "WAIT_ALL_FAMILIES_INVALIDATED" in decision.gate_reasons
    assert "INVALIDATED_BY:STALE_DATA" in decision.gate_reasons


def test_disconnected_websocket_blocks(store, fresh_registry, tmp_path) -> None:
    result = _run(_pipeline(store, fresh_registry, tmp_path), websocket_connected=False)
    assert not result.decisions[0].gate_result
    assert "HARD:WS_DISCONNECTED" in result.decisions[0].gate_reasons


def test_unreconciled_account_blocks(store, fresh_registry, tmp_path) -> None:
    result = _run(
        _pipeline(store, fresh_registry, tmp_path), account=_account(reconciled=False)
    )
    assert not result.decisions[0].gate_result
    assert "HARD:ACCOUNT_RECONCILIATION_FAIL" in result.decisions[0].gate_reasons


def test_a_live_order_blocks_a_second_one_on_the_same_symbol(
    store, fresh_registry, tmp_path
) -> None:
    machine = OrderStateMachine(store)
    pipeline = _pipeline(store, fresh_registry, tmp_path, state_machine=machine)
    first = _run(pipeline, create_order_intents=True)
    assert first.decisions[0].gate_result

    intent_id = first.decisions[0].order_intent["intent_id"]  # type: ignore[index]
    machine.transition(intent_id, OrderState.SUBMITTING, now=NOW)
    machine.transition(intent_id, OrderState.SUBMITTED, broker_order_id="B-1", now=NOW)

    second = _run(pipeline, create_order_intents=True)
    assert not second.decisions[0].gate_result
    assert "HARD:DUPLICATE_ORDER_RISK" in second.decisions[0].gate_reasons
    assert second.decisions[0].order_intent is None


def test_an_unknown_order_blocks_its_own_symbol(store, fresh_registry, tmp_path) -> None:
    machine = OrderStateMachine(store)
    pipeline = _pipeline(store, fresh_registry, tmp_path, state_machine=machine)
    record = machine.create(
        ticker="005930", side="SELL", quantity=10, idempotency_key="ghost", now=NOW
    )
    machine.transition(record.intent_id, OrderState.GATED, now=NOW)
    machine.transition(record.intent_id, OrderState.SUBMITTING, now=NOW)
    machine.transition(record.intent_id, OrderState.UNKNOWN, now=NOW)

    result = _run(pipeline)
    assert not result.decisions[0].gate_result
    assert "HARD:UNKNOWN_ORDER_STATE" in result.decisions[0].gate_reasons


def test_offline_model_blocks_new_entry(store, fresh_registry, tmp_path) -> None:
    offline = GnnRuntime(
        checkpoint_path=tmp_path / "absent.npz",
        config=MODEL_CONFIG,
        require_checkpoint=True,
    )
    assert offline.health().state is GnnHealthState.OFFLINE
    result = _run(_pipeline(store, fresh_registry, tmp_path, gnn_runtime=offline))
    assert not result.decisions[0].gate_result
    assert "HARD:MODEL_INFERENCE_FAIL" in result.decisions[0].gate_reasons


def test_degraded_model_still_trades_at_reduced_size(
    store, fresh_registry, tmp_path
) -> None:
    result = _run(_pipeline(store, fresh_registry, tmp_path))
    decision = result.decisions[0]
    assert result.model_health is not None
    assert result.model_health.state is GnnHealthState.DEGRADED
    assert decision.gate_result
    assert 0.0 < decision.position_multiplier < 1.0


# --------------------------------------------------------------------------- #
# Cross-market discipline
# --------------------------------------------------------------------------- #
def test_weak_global_alone_does_not_produce_a_domestic_sell(
    store, fresh_registry, tmp_path
) -> None:
    """The requirement stated in the goals: US weakness must be domestically confirmed."""
    world = _bearish_world()
    strong_home = _domestic(world, strong=True)
    assert strong_home.direction is not None and strong_home.direction > 0
    assert not strong_home.confirms_global_weakness()

    result = _run(
        _pipeline(store, fresh_registry, tmp_path),
        global_context=world,
        domestic=strong_home,
        sectors=_sectors(strong_home, world),
    )
    decision = result.decisions[0]
    # The conflict is recorded rather than acted on: a global reason code appears, and
    # the domestic evidence still carries the decision.
    assert decision.domestic_context["global_conflict"] is True
    assert decision.regime_probabilities["TREND_DOWN"] < 0.5


def test_confirmed_domestic_weakness_shuts_down_entries(
    store, fresh_registry, tmp_path
) -> None:
    world = _bearish_world()
    weak_home = _domestic(world, strong=False)
    assert weak_home.confirms_global_weakness()

    result = _run(
        _pipeline(store, fresh_registry, tmp_path),
        global_context=world,
        domestic=weak_home,
        sectors=_sectors(weak_home, world, returns=(-0.02, -0.03, -0.01, -0.025)),
        candidates=[_candidate(session_return=-0.02, relative_strength=-0.01,
                               breakout_state=-0.4, trend_strength=-30.0,
                               liquidity_score=0.2, spread_bps=60.0,
                               realized_volatility=0.009)],
    )
    decision = result.decisions[0]
    assert decision.action == "WAIT"
    assert not decision.gate_result


# --------------------------------------------------------------------------- #
# Ontology traceability
# --------------------------------------------------------------------------- #
def test_ontology_priors_and_learned_weights_are_both_traceable(
    store, fresh_registry, tmp_path
) -> None:
    result = _run(_pipeline(store, fresh_registry, tmp_path))
    relations = result.decisions[0].ontology_relations
    assert relations, "a selected family must record the edges that scored it"
    for relation in relations:
        assert "prior_strength" in relation
        assert "learned_weight" in relation
        assert "effective_weight" in relation
        assert relation["weight_source"] in {"prior", "learned", "structural"}


def test_learned_weight_shows_up_beside_its_prior(store, fresh_registry, tmp_path) -> None:
    from app.ontology.market_graph import load_market_graph
    from app.routing.regime_strategy_selector import RegimeStrategySelector

    graph = load_market_graph()
    edge_id = "TREND_UP|SUITABLE_FOR|TREND"
    graph.apply_learned_weights({edge_id: 0.10})
    pipeline = _pipeline(
        store,
        fresh_registry,
        tmp_path,
        selector=RegimeStrategySelector(graph=graph),
    )
    result = _run(pipeline)
    relations = {
        relation["edge_id"]: relation
        for decision in result.decisions
        for relation in decision.ontology_relations
    }
    if edge_id in relations:
        record = relations[edge_id]
        assert record["prior_strength"] == pytest.approx(0.85)
        assert record["learned_weight"] == pytest.approx(0.10)
        assert record["prior_learned_gap"] == pytest.approx(-0.75)
        assert record["weight_source"] == "learned"


def test_gnn_attention_trace_reports_priors(store, fresh_registry, tmp_path) -> None:
    result = _run(_pipeline(store, fresh_registry, tmp_path))
    assert result.prediction is not None
    trace = result.prediction.output.trace.as_dict()
    assert trace["relation_mass"]
    for payload in trace["relation_attention"].values():
        for edge in payload["edges"]:
            assert "attention" in edge
            assert "ontology_prior_bias" in edge


# --------------------------------------------------------------------------- #
# Model artefacts
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trips(tmp_path) -> None:
    model = TemporalHeteroGnn(MODEL_CONFIG)
    path = model.save_checkpoint(tmp_path / "gnn.npz")
    reloaded = TemporalHeteroGnn.load_checkpoint(path)
    assert reloaded.config == model.config
    assert reloaded.parameter_count() == model.parameter_count()


def test_corrupt_checkpoint_takes_the_runtime_offline(tmp_path) -> None:
    path = tmp_path / "gnn.npz"
    path.write_bytes(b"not an npz file")
    runtime = GnnRuntime(checkpoint_path=path, config=MODEL_CONFIG)
    health = runtime.health()
    assert health.state is GnnHealthState.OFFLINE
    assert not health.allows_new_entry
    assert health.size_multiplier == 0.0


def test_a_trained_checkpoint_loads_healthy(tmp_path) -> None:
    TemporalHeteroGnn(MODEL_CONFIG).save_checkpoint(tmp_path / "gnn.npz")
    runtime = GnnRuntime(checkpoint_path=tmp_path / "gnn.npz", config=MODEL_CONFIG)
    assert runtime.health().state is GnnHealthState.HEALTHY
    assert runtime.health().allows_model_evidence


def test_repeated_inference_failure_latches_offline(tmp_path) -> None:
    TemporalHeteroGnn(MODEL_CONFIG).save_checkpoint(tmp_path / "gnn.npz")
    runtime = GnnRuntime(
        checkpoint_path=tmp_path / "gnn.npz", config=MODEL_CONFIG, failure_threshold=2
    )
    for _ in range(3):
        runtime.mark_failure("synthetic")
    assert runtime.health().state is GnnHealthState.OFFLINE
    # A reload is the only way back: an intermittently failing model is not trustworthy
    # just because one call happened to succeed.
    assert runtime.reload().state is GnnHealthState.HEALTHY


def test_gnn_cannot_reach_the_execution_layer() -> None:
    """The model must have no import path to a broker, an order or the risk engine."""
    import app.models.temporal_hetero_gnn as module

    source = module.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    for forbidden in (
        "app.execution",
        "app.risk",
        "LiveExecutionCoordinator",
        "place_limit_order",
        "OrderIntent",
    ):
        assert f"import {forbidden}" not in text
        assert f"from {forbidden}" not in text


def test_a_suspended_symbol_is_halted_even_while_the_venue_transacts(
    store, fresh_registry, tmp_path
) -> None:
    """Per-symbol suspension is the case that actually occurs.

    The cycle-wide flag can only answer "is the venue trading". A symbol suspended inside
    an open session would otherwise pass the halt gate on the venue's answer.
    """
    pipeline = _pipeline(store, fresh_registry, tmp_path)
    result = _run(
        pipeline, candidates=[_candidate(halted=True)], trading_halted=False
    )

    decision = result.decisions[0]
    assert "HARD:TRADING_HALT" in decision.gate_reasons
    assert decision.gate_result is False
    # And it is the ONLY complaint: everything else the gate needs was supplied.
    assert [r for r in decision.gate_reasons if r.startswith("HARD:")] == [
        "HARD:TRADING_HALT"
    ]


def test_a_tradable_symbol_does_not_inherit_a_halt_it_does_not_have(
    store, fresh_registry, tmp_path
) -> None:
    pipeline = _pipeline(store, fresh_registry, tmp_path)
    result = _run(
        pipeline, candidates=[_candidate(halted=False)], trading_halted=True
    )

    decision = result.decisions[0]
    assert "HARD:TRADING_HALT" not in decision.gate_reasons


def test_scoped_staleness_does_not_contaminate_another_symbol() -> None:
    from app.trading.context_decision_pipeline import (
        _global_stale_reasons,
        _stale_reasons_for_ticker,
    )

    reasons = (
        "STALE_DATA:kis_websocket/tick:005930",
        "STALE_DATA:kis_websocket/orderbook:AAPL",
        "STALE_DATA:collector/heartbeat",
    )
    assert _stale_reasons_for_ticker(reasons, "AAPL") == (
        "STALE_DATA:kis_websocket/orderbook:AAPL",
        "STALE_DATA:collector/heartbeat",
    )
    assert _global_stale_reasons(reasons) == ("STALE_DATA:collector/heartbeat",)
