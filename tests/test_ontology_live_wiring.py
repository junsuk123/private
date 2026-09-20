from types import SimpleNamespace

from app.schemas.domain import RiskRules
from test_ontology_policy_integration import NOW, _request


def test_actual_web_factory_requires_one_shared_policy_without_starting_broker(monkeypatch):
    from app import web
    from app.trading import strategy_session

    store = SimpleNamespace(recent_minute_bars=lambda *a, **k: (), latest_orderbook=lambda *a: None)
    monkeypatch.setattr(web, "RealtimeMarketDataStore", lambda: store)
    monkeypatch.setattr(web, "_live_account_snapshot_for_analysis", lambda: _request().account)
    monkeypatch.setattr(web, "_live_risk_rules_for_account", lambda account: RiskRules())
    monkeypatch.setattr(web, "KisDevelopersApiClient", lambda **kwargs: object())
    monkeypatch.setattr(web, "_ensure_us_fast_poll_started", lambda: None)
    monkeypatch.setattr(web, "_build_macro_micro_observer", lambda engine: None)
    monkeypatch.setattr(web, "get_context_runtime", lambda: None)
    monkeypatch.setattr(web, "_live_shadow_service", None)
    monkeypatch.setattr(web, "_ontology_policy_runtime", None)
    monkeypatch.setattr(web, "SharedLiveDecisionEngine", lambda store, **kwargs: SimpleNamespace(store=store, **kwargs))
    monkeypatch.setattr(strategy_session, "StrategySessionManager", lambda **kwargs: SimpleNamespace(trade_plan_for=lambda symbol: None, **kwargs))
    monkeypatch.setattr(web, "LiveExecutionCoordinator", lambda broker, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(web, "RealtimeTradingEngine", lambda **kwargs: SimpleNamespace(**kwargs))

    engine = web._build_realtime_trading_engine()
    session = engine.strategy_session_manager
    runtime = engine.ontology_policy_resolver.__self__
    assert engine.decision_engine.ontology_policy_resolver.__self__ is runtime
    assert session.ontology_policy_resolver.__self__ is runtime
    assert session.plan_builder.ontology_policy_resolver.__self__ is runtime
    assert engine.ontology_policy_snapshot_provider.__self__ is runtime
    assert engine.decision_engine.risk_manager.ontology_policy_required
    assert session.plan_builder.risk_manager.ontology_policy_required
    assert session.graph_training_context_provider.__self__ is web._graph_policy_contexts
    assert engine.coordinator.cash_equity_only
    # An otherwise affordable profitable request must still fail when this
    # actual factory's store/context cannot substantiate the operating policy.
    result = session.plan_builder.build(_request(), now=NOW)
    assert result.plan is None
    assert result.no_trade.stage == "ontology_policy"


def test_live_forward_training_inputs_are_captured_without_authorized_or_enabled_gnn(monkeypatch):
    from app import web
    from app.models.strategy_utility.policy_context import GraphPolicyContextCache
    from test_shadow_intelligence import _stub_frame, _graph_context_values

    cache = GraphPolicyContextCache(clock=lambda: NOW)
    frame = _stub_frame(_graph_context_values())
    frame.decision_time = NOW
    monkeypatch.setattr(web, "_graph_policy_contexts", cache)
    monkeypatch.setattr(web, "_live_shadow_service", None)
    monkeypatch.setattr(web, "_live_shadow_state", {})
    monkeypatch.setenv("STRATEGY_SESSION_REQUIRE_LIVE_GNN", "false")
    monkeypatch.setattr(web.RefactorFeatureFlags, "from_env", lambda: SimpleNamespace(
        ontology_router=False, gnn_shadow=False, npu_inference=False))

    web._refresh_live_candidate_shadow({frame.symbol: frame}, NOW)
    captured = cache.latest(frame.symbol, NOW)
    assert captured is not None
    assert captured["source"] == "live_strategy_graph_context"
    assert captured["as_of"] == NOW.isoformat()
    assert web._live_shadow_service is None
    assert web._live_shadow_state["enabled"] is False
