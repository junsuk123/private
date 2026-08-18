"""The diagnostics surface must describe the path the code actually takes.

A screen that names an authority which cannot act is worse than one that names none: an
operator will look for a veto that no longer exists, and will not look for the one that
does. These tests pin the description to the implementation.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.execution.execution_guard import FORBIDDEN_INVESTMENT_CHECKS
from app.web_context_routes import create_context_router


class _Runtime:
    """Minimal stand-in exposing the two views under test."""

    def __init__(self, session: dict | None = None) -> None:
        from app.trading.context_runtime import ContextRuntime

        self._real = ContextRuntime.__new__(ContextRuntime)
        self._real._session_snapshot_provider = (lambda: session) if session else None

    def authority_path_view(self, **kwargs):
        from app.trading.context_runtime import ContextRuntime

        return ContextRuntime.authority_path_view(self._real, **kwargs)

    def latency_view(self, **kwargs):
        from app.trading.context_runtime import ContextRuntime

        return ContextRuntime.latency_view(self._real, **kwargs)


def _client(session: dict | None = None) -> TestClient:
    app = FastAPI()
    runtime = _Runtime(session)
    app.include_router(create_context_router(runtime_provider=lambda: runtime))
    return TestClient(app)


# --------------------------------------------------------------------------- #
def test_the_authority_path_lists_every_stage_in_order() -> None:
    payload = _client().get("/api/execution/authority-path").json()
    stages = [item["stage"] for item in payload["stages"]]
    assert stages == [
        "pre_selection_context",
        "pre_selection_cost_size_risk",
        "election",
        "fast_loop",
        "execution_guard",
        "broker",
    ]


def test_cost_size_and_risk_are_shown_as_pre_selection() -> None:
    payload = _client().get("/api/execution/authority-path").json()
    stage = next(
        item for item in payload["stages"] if item["stage"] == "pre_selection_cost_size_risk"
    )
    assert stage["authority"] == "TradePlanBuilder"
    joined = " ".join(stage["decides"])
    assert "ProfitabilityGate" in joined
    assert "PositionSizer" in joined
    assert "RiskManager" in joined
    assert "post-election" in stage["moved_from"]


def test_the_fast_loop_declares_what_it_may_not_do() -> None:
    payload = _client().get("/api/execution/authority-path").json()
    stage = next(item for item in payload["stages"] if item["stage"] == "fast_loop")
    assert set(stage["forbidden"]) == {
        "ontology rebuild",
        "GNN inference",
        "portfolio risk",
        "position resizing",
        "profitability re-evaluation",
    }


def test_the_guard_stage_publishes_the_forbidden_check_list() -> None:
    payload = _client().get("/api/execution/authority-path").json()
    stage = next(item for item in payload["stages"] if item["stage"] == "execution_guard")
    assert set(stage["forbidden"]) == set(FORBIDDEN_INVESTMENT_CHECKS)


def test_no_stage_after_election_claims_an_investment_decision() -> None:
    payload = _client().get("/api/execution/authority-path").json()
    post_election = [
        item
        for item in payload["stages"]
        if item["stage"] in {"fast_loop", "execution_guard", "broker"}
    ]
    for stage in post_election:
        joined = " ".join(stage.get("decides", [])).lower()
        for banned in ("profitab", "position size", "portfolio risk", "expected return"):
            assert banned not in joined, (stage["stage"], banned)


def test_the_removed_vetoes_are_listed() -> None:
    payload = _client().get("/api/execution/authority-path").json()
    removed = payload["removed_post_selection_vetoes"]
    assert "post-selection ProfitabilityGate" in removed
    assert "post-selection PositionSizer" in removed
    assert "post-selection portfolio RiskManager" in removed


def test_a_live_session_surfaces_its_plan_and_state() -> None:
    session = {
        "execution_authority": "TRADE_PLAN",
        "phase": "ARMED",
        "selected_symbol": "000660",
        "selected_strategy": "intraday_momentum",
        "selected_direction": "LONG",
        "last_reason": "ELECTED",
        "trade_plan": {"plan_id": "plan-1", "quantity": 7},
    }
    payload = _client(session).get("/api/execution/authority-path").json()
    assert payload["authority"] == "TRADE_PLAN"
    assert payload["trade_plan"]["plan_id"] == "plan-1"
    assert payload["strategy_state"]["selected_strategy"] == "intraday_momentum"


def test_no_session_reports_the_legacy_path_rather_than_pretending() -> None:
    payload = _client().get("/api/execution/authority-path").json()
    assert payload["authority"] == "LEGACY_GATED_PATH"
    assert payload["trade_plan"] is None


def test_the_latency_endpoint_answers() -> None:
    payload = _client().get("/api/execution/latency").json()
    assert payload["available"] is True
    assert "summary" in payload and "recent" in payload


def test_the_diagnostics_surface_is_read_only() -> None:
    app = FastAPI()
    app.include_router(create_context_router(runtime_provider=lambda: _Runtime()))
    methods = {
        method
        for route in app.routes
        for method in getattr(route, "methods", set()) or set()
    }
    assert methods <= {"GET", "HEAD"}


def test_the_account_dashboard_no_longer_names_a_post_election_veto() -> None:
    """The page text used to say final approval rests with RiskManager /
    ProfitabilityGate. It does not, and the page must not say so."""
    import app.web_account_routes as module

    text = open(module.__file__, encoding="utf-8").read()  # type: ignore[arg-type]
    assert "최종 승인은 RiskManager·ProfitabilityGate가 가집니다" not in text
    assert "RiskManager·ProfitabilityGate가 최종 권한을 가지며" not in text
    assert "선출 이전" in text
