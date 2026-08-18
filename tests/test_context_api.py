"""HTTP and WebSocket contract for the context/decision surface.

The routes are exercised against a real :class:`ContextRuntime` over a temporary store,
so the assertions cover the wiring as well as the payload shapes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.context.domestic_context import (
    DomesticContextBuilder,
    DomesticContextInputs,
    VenueQuote,
)
from app.context.global_context import GlobalContextBuilder, IndicatorObservation
from app.context.temporal_context import build_temporal_snapshot
from app.data.freshness import DataFreshnessRegistry
from app.execution.order_state_machine import OrderState, OrderStateMachine
from app.models.gnn_runtime import GnnRuntime
from app.models.graph_snapshot import FEATURE_DIM, GraphSnapshotBuilder
from app.models.temporal_hetero_gnn import TemporalHeteroGnnConfig
from app.storage.trading_state_store import TradingStateStore
from app.trading.context_decision_pipeline import (
    AccountState,
    CandidateInput,
    ContextDecisionPipeline,
)
from app.trading.context_runtime import ContextRuntime
from app.web_context_routes import REALTIME_CHANNELS, create_context_router

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)
MODEL_CONFIG = TemporalHeteroGnnConfig(max_nodes=96, feature_dim=FEATURE_DIM, time_steps=8)


@pytest.fixture()
def runtime(tmp_path) -> ContextRuntime:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    registry = DataFreshnessRegistry()
    for source, data_type in (
        ("kis_realtime", "trade"),
        ("kis_realtime", "orderbook"),
        ("kis_rest", "account"),
        ("kis_rest", "positions"),
        ("kis_rest", "order_status"),
        ("internal", "domestic_context"),
    ):
        registry.record_event(source, data_type, NOW, received_time=NOW, processed_time=NOW)
    machine = OrderStateMachine(store)
    gnn = GnnRuntime(
        checkpoint_path=tmp_path / "absent.npz",
        config=MODEL_CONFIG,
        require_checkpoint=False,
    )
    pipeline = ContextDecisionPipeline(
        store=store,
        gnn_runtime=gnn,
        snapshot_builder=GraphSnapshotBuilder(max_nodes=96, time_steps=8),
        state_machine=machine,
        freshness=registry,
    )
    service = ContextRuntime(
        store=store,
        freshness=registry,
        gnn_runtime=gnn,
        pipeline=pipeline,
        state_machine=machine,
        require_checkpoint=False,
    )
    _seed(service, pipeline)
    return service


def _seed(service: ContextRuntime, pipeline: ContextDecisionPipeline) -> None:
    world = GlobalContextBuilder().build(
        [
            IndicatorObservation("SP500", 5400.0, NOW - timedelta(hours=8), change_ratio=0.008),
            IndicatorObservation("SOX", 5200.0, NOW - timedelta(hours=8), change_ratio=0.02),
            IndicatorObservation("VIX", 15.0, NOW - timedelta(hours=8), change_ratio=-0.08),
            IndicatorObservation("ES", 5405.0, NOW - timedelta(minutes=2), change_ratio=0.004),
            IndicatorObservation("NIKKEI", 39000.0, NOW - timedelta(minutes=30), change_ratio=0.006),
            IndicatorObservation("USDKRW", 1380.0, NOW - timedelta(minutes=5), change_ratio=-0.002),
        ],
        captured_at=NOW,
    )
    domestic = DomesticContextBuilder().build(
        DomesticContextInputs(
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
            sector_returns={"semiconductor": 0.012},
            venues=(VenueQuote("KRX", mid=2600.0), VenueQuote("NXT", mid=2600.2)),
        ),
        captured_at=NOW,
        global_context=world,
    )
    from app.context.sector_context import SectorContextBuilder, SectorMemberObservation

    sectors = [
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
                for index, value in enumerate((0.02, 0.014, 0.008, 0.011))
            ],
            captured_at=NOW,
            market_return=0.005,
            market_return_history=[0.003, -0.005, 0.002, -0.002] * 8,
            domestic_context=domestic,
            global_context=world,
            global_group="semiconductor",
        )
    ]
    result = pipeline.run_cycle(
        captured_at=NOW,
        temporal=build_temporal_snapshot("KRX", NOW),
        candidates=[
            CandidateInput(
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
        ],
        global_context=world,
        domestic_context=domestic,
        sector_contexts=sectors,
        account=AccountState(
            equity=1e8, cash=5e7, reconciled=True, session_pnl_ratio=0.002, drawdown_ratio=0.01
        ),
        websocket_connected=True,
        trading_halted=False,
        create_order_intents=True,
    )
    # Publish it as the runtime's latest cycle without re-reading the live stores.
    with service._lock:  # noqa: SLF001 - test seam; the alternative is a live feed.
        service._latest = result  # noqa: SLF001
        service._last_refresh_at = NOW  # noqa: SLF001
        service._refresh_count = 1  # noqa: SLF001


@pytest.fixture()
def client(runtime: ContextRuntime) -> TestClient:
    app = FastAPI()
    app.include_router(create_context_router(runtime_provider=lambda: runtime))
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Declared endpoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        "/api/session/current",
        "/api/context/global",
        "/api/context/domestic",
        "/api/context/sector/semiconductor",
        "/api/context/stock/005930",
        "/api/regime/current",
        "/api/candidates",
        "/api/decision/005930",
        "/api/gate/005930",
        "/api/model/health",
        "/api/data/health",
    ],
)
def test_every_declared_endpoint_answers(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_session_endpoint_reports_both_groups(client: TestClient) -> None:
    payload = client.get("/api/session/current").json()
    assert set(payload["groups"]) == {"KR", "US"}
    assert payload["groups"]["KR"]["session_phase"]
    assert payload["groups"]["KR"]["display_timezone"] == "Asia/Seoul"


def test_global_and_domestic_contexts_carry_their_outputs(client: TestClient) -> None:
    global_payload = client.get("/api/context/global").json()
    assert global_payload["available"]
    assert set(global_payload["context"]) >= {
        "direction",
        "momentum",
        "risk_sentiment",
        "volatility",
        "rates_pressure",
        "fx_pressure",
        "global_alignment",
        "confidence",
    }
    domestic_payload = client.get("/api/context/domestic").json()
    assert set(domestic_payload["context"]) >= {
        "direction",
        "breadth",
        "liquidity",
        "volatility",
        "flow",
        "leadership",
        "venue_divergence",
        "confidence",
    }


def test_regime_is_multi_label_and_does_not_sum_to_one(client: TestClient) -> None:
    payload = client.get("/api/regime/current").json()
    probabilities = payload["probabilities"]
    assert len(probabilities) == 13
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert payload["dominant"] in probabilities
    assert "entropy" in payload


def test_candidate_columns_match_the_dashboard_contract(client: TestClient) -> None:
    payload = client.get("/api/candidates").json()
    assert payload["available"]
    row = payload["candidates"][0]
    assert set(row) >= {
        "ticker",
        "sector",
        "relative_strength",
        "order_flow",
        "global_alignment",
        "strategy",
        "model_confidence",
        "gate",
    }


def test_decision_endpoint_returns_the_whole_trace(client: TestClient) -> None:
    payload = client.get("/api/decision/005930").json()
    assert payload["available"]
    assert payload["decision_id"]
    assert payload["regime_probabilities"]
    assert "ontology_relations" in payload
    assert "learned_relation_weights" in payload


def test_unknown_ticker_says_so_rather_than_returning_nothing(client: TestClient) -> None:
    payload = client.get("/api/decision/999999").json()
    assert payload["available"] is False
    assert payload["reason"] == "TICKER_NOT_IN_LAST_CYCLE"


def test_unknown_sector_is_distinguished_from_no_cycle(client: TestClient) -> None:
    payload = client.get("/api/context/sector/shipbuilding").json()
    assert payload["available"] is False
    assert payload["reason"] == "SECTOR_NOT_IN_LAST_CYCLE"


# --------------------------------------------------------------------------- #
# Health and readiness
# --------------------------------------------------------------------------- #
def test_model_health_reports_permissions_not_just_a_label(client: TestClient) -> None:
    payload = client.get("/api/model/health").json()
    assert payload["state"] in {"HEALTHY", "DEGRADED", "OFFLINE"}
    assert "allows_new_entry" in payload
    assert "allows_model_evidence" in payload
    assert "size_multiplier" in payload


def test_data_health_lists_every_stream(client: TestClient) -> None:
    payload = client.get("/api/data/health").json()
    assert payload["worst_state"] in {"HEALTHY", "DEGRADED", "STALE"}
    assert payload["streams"]
    for stream in payload["streams"]:
        assert set(stream) >= {"source", "data_type", "state", "critical"}


def test_dashboard_readiness_cannot_contradict_module_health(
    runtime: ContextRuntime, client: TestClient
) -> None:
    """The contradiction this refactor exists to remove: 100% ready, module OFFLINE."""
    payload = client.get("/api/context/dashboard").json()
    readiness = payload["readiness"]
    if payload["GNN_HEALTH"] == "OFFLINE" or payload["DATA_AGE"] == "STALE":
        assert readiness["new_entry_permitted"] is False
        assert payload["FINAL_GATE"] == "BLOCK"
    assert readiness["model"] == payload["GNN_HEALTH"]
    assert readiness["data"] == payload["DATA_AGE"]


def test_dashboard_top_strip_has_every_declared_field(client: TestClient) -> None:
    payload = client.get("/api/context/dashboard").json()
    for field in (
        "KST",
        "KRX_SESSION",
        "NXT_SESSION",
        "US_SESSION",
        "GLOBAL_REGIME",
        "KR_REGIME",
        "VOLATILITY",
        "BREADTH",
        "LIQUIDITY",
        "DATA_AGE",
        "GNN_HEALTH",
        "FINAL_GATE",
    ):
        assert field in payload, field


def test_orders_endpoint_exposes_the_gated_intent(client: TestClient) -> None:
    payload = client.get("/api/orders/open").json()
    assert payload["available"]
    assert payload["summary"]["open_count"] >= 1
    assert payload["open"][0]["state"] == OrderState.GATED.value


# --------------------------------------------------------------------------- #
# Failure behaviour
# --------------------------------------------------------------------------- #
def test_missing_runtime_degrades_rather_than_500s() -> None:
    app = FastAPI()
    app.include_router(create_context_router(runtime_provider=lambda: None))
    client = TestClient(app)
    response = client.get("/api/candidates")
    assert response.status_code == 503
    assert response.json()["reason"] == "RUNTIME_NOT_STARTED"


def test_a_broken_runtime_reports_the_error_in_the_body() -> None:
    def _explode():
        raise RuntimeError("store unavailable")

    app = FastAPI()
    app.include_router(create_context_router(runtime_provider=_explode))
    payload = TestClient(app).get("/api/model/health").json()
    assert payload["available"] is False
    assert "store unavailable" in payload["error"]


def test_no_route_can_place_an_order(runtime: ContextRuntime) -> None:
    app = FastAPI()
    app.include_router(create_context_router(runtime_provider=lambda: runtime))
    methods = {
        method
        for route in app.routes
        for method in getattr(route, "methods", set()) or set()
    }
    assert methods <= {"GET", "HEAD"}, "the context surface must be read-only"


# --------------------------------------------------------------------------- #
# WebSocket channels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("channel", REALTIME_CHANNELS)
def test_every_channel_pushes_a_frame(client: TestClient, channel: str) -> None:
    with client.websocket_connect(f"/ws/{channel}") as socket:
        frame = socket.receive_json()
    assert frame["channel"] == channel
    assert frame["as_of"]
    assert "payload" in frame


def test_unknown_channel_is_refused(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/nonsense") as socket:
            socket.receive_json()


def test_context_channel_carries_the_dashboard_strip(client: TestClient) -> None:
    with client.websocket_connect("/ws/context") as socket:
        frame = socket.receive_json()
    assert set(frame["payload"]) == {
        "session",
        "global",
        "domestic",
        "regime",
        "dashboard",
    }
