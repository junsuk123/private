from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.context.global_context import GlobalContextBuilder, IndicatorObservation
from app.context.domestic_context import DomesticContext
from app.context.market_context import ContextIdentity, MacroContext, MarketContext, TemporalContext
from app.data.market_capabilities import MarketSessionService
from app.features.macro_feature_frame import MacroFeatureFrame
from app.ontology.indicator_graph import evaluate_market_indicators
from app.models.graph_snapshot import FEATURE_SLOTS, GraphSnapshotBuilder
from app.storage.trading_state_store import TradingStateStore
from app.strategy.market_policy import build_market_operating_plans
from app.strategy.proposal_engine import StrategyProposalEngine
from app.strategy.registry import default_strategy_registry, reset_default_strategy_registry
from app.technical.strategy_algorithms import ALGORITHM_IDS, ElectionContext, MACRO_FAMILY_BY_STRATEGY, macro_strategy_permitted
from app.trading.context_decision_pipeline import CandidateInput, ContextDecisionPipeline
from app.trading.context_runtime import ContextRuntime


NOW = datetime(2026, 8, 5, 1, tzinfo=timezone.utc)


def test_every_catalogued_strategy_has_macro_permission_vocabulary():
    assert set(ALGORITHM_IDS) == set(MACRO_FAMILY_BY_STRATEGY)
    assert macro_strategy_permitted("opening_range_breakout", ("breakout",), ())
    assert macro_strategy_permitted("market_intraday_momentum", ("momentum",), ())
    assert not macro_strategy_permitted("opening_range_breakout", ("breakout",), ("momentum",))


def test_algorithm_and_spec_overrides_are_isolated_by_market(monkeypatch):
    monkeypatch.setenv("ALGO_KR_BREAKOUT_VOLUME_HORIZON_SECONDS", "4200")
    monkeypatch.setenv("ALGO_US_BREAKOUT_VOLUME_HORIZON_SECONDS", "1800")
    reset_default_strategy_registry()
    try:
        engine = StrategyProposalEngine()
        kr = engine._algorithm_registry("KRX")
        us = engine._algorithm_registry("NASDAQ")
        assert kr["breakout_volume"].horizon_seconds == 4200
        assert us["breakout_volume"].horizon_seconds == 1800
        assert engine._algorithm_registry("KR") is kr
        assert default_strategy_registry("KRX").require("breakout_volume").horizon_seconds == 4200
        assert default_strategy_registry("US").require("breakout_volume").horizon_seconds == 1800
    finally:
        reset_default_strategy_registry()


def test_proposal_receives_own_market_regime_and_opening_clock():
    context = MarketContext(
        identity=ContextIdentity("ctx", NOW, "AAPL", "US"),
        macro=MacroContext(market_regime="TREND_DOWN"),
        temporal=TemporalContext(minutes_from_open=47),
    )
    payload = StrategyProposalEngine._election_payload(
        spec=default_strategy_registry("US").require("opening_range_breakout"),
        context=context, election_inputs={}, election_context_type=ElectionContext,
    )
    assert payload["market_trend"] == "TREND_DOWN"
    assert payload["minutes_since_session_open"] == 47


def test_market_operating_plans_preserve_simultaneous_sessions_and_cash_states():
    service = MarketSessionService()
    # Korean regular and US daytime are simultaneously observable.
    plans = build_market_operating_plans({"KR": "TREND_UP", "US": "TREND_UP"}, now_utc=NOW, service=service)
    assert "KRX_REGULAR" in plans["KR"].active_sessions
    assert "US_DAYTIME" in plans["US"].active_sessions
    assert plans["KR"].preferred_strategy_ids != plans["US"].preferred_strategy_ids
    assert plans["KR"].maximum_position_weight <= 0.15
    # Session discovery does not grant an unapproved extended-hours order.
    assert "US_DAYTIME" not in plans["US"].live_entry_sessions
    cash = build_market_operating_plans({"KR": "RISK_OFF", "US": "UNSEEN_STATE"}, now_utc=NOW, service=service)
    assert all(plan.mode == "CASH" and plan.maximum_position_weight == 0 for plan in cash.values())
    assert all(not plan.preferred_strategy_ids for plan in cash.values())


def test_indicator_graph_keeps_us_risk_reference_but_excludes_kr_local_data_for_us():
    observations = (
        IndicatorObservation("VIX", 20, NOW, source="fred"),
        IndicatorObservation("KOSPI", 2800, NOW, source="kis"),
        IndicatorObservation("USDKRW", 1300, NOW, source="ecos"),
    )
    kr = evaluate_market_indicators(observations, "KR", captured_at=NOW)
    us = evaluate_market_indicators(observations, "US", captured_at=NOW)
    assert all(item.usable for item in kr)
    assert kr[0].relation == "INFLUENCES"
    assert us[0].usable and us[0].relation == "CONFIRMS"
    assert not us[1].usable and not us[2].usable


@pytest.mark.parametrize("changes, reason", [
    ({"observed_at": NOW + timedelta(seconds=1)}, "INDICATOR_FROM_FUTURE"),
    ({"observed_at": NOW - timedelta(days=5)}, "INDICATOR_STALE"),
    ({"source": "unverified"}, "INDICATOR_SOURCE_UNVERIFIED"),
    ({"source": ""}, "INDICATOR_SOURCE_UNVERIFIED"),
    ({"value": float("nan")}, "INDICATOR_VALUE_INVALID"),
])
def test_unusable_evidence_never_reaches_global_context(changes, reason):
    observation = replace(IndicatorObservation("VIX", 20, NOW, source="fred"), **changes)
    context = GlobalContextBuilder().build((observation,), captured_at=NOW, market="US")
    assert reason in context.reason_codes
    assert context.indicator_relations[0]["usable"] is False
    assert "risk" not in context.groups


def test_runtime_keeps_local_breadth_sectors_and_clock_separate(tmp_path, monkeypatch):
    store = TradingStateStore(tmp_path / "context.sqlite3")
    kr_pipeline = ContextDecisionPipeline(store=store, persist=False)
    us_pipeline = ContextDecisionPipeline(store=store, persist=False)
    runtime = ContextRuntime(store=store, gnn_runtime=object(), pipeline=kr_pipeline)
    runtime._market_pipelines["US"] = us_pipeline
    monkeypatch.setattr(runtime, "_macro_observations", lambda moment: ())
    monkeypatch.setattr(runtime, "_investor_flows", lambda candidates, **kwargs: {})
    monkeypatch.setattr(runtime, "_venue_quotes", lambda moment, candidates: ())
    seen = []

    def frame(moment, candidates):
        seen.append(tuple(item.market_group for item in candidates))
        returns = {item.ticker: item.session_return for item in candidates}
        return MacroFeatureFrame(
            timestamp=moment, index_trend=sum(returns.values()) / len(returns),
            market_breadth=None, market_volatility=0.002, total_trading_value=None,
            per_symbol_return=returns, sector_snapshots={}, sector_of={}, symbol_count=len(returns),
        )

    monkeypatch.setattr(runtime, "_macro_frame", frame)
    candidates = (
        CandidateInput("005930", market_group="KR", sector="semiconductor", session_return=0.01),
        CandidateInput("NVDA", market_group="US", sector="semiconductor", session_return=-0.02),
        *(CandidateInput(f"{100000 + index}", market_group="KR", sector="semiconductor", session_return=0.01) for index in range(19)),
        *(CandidateInput(f"US{index}", market_group="US", sector="semiconductor", session_return=-0.02) for index in range(19)),
    )
    result = runtime.refresh(now=NOW, candidates=candidates)
    assert result is not None, runtime.status().last_error
    assert seen == [("KR",) * 20, ("US",) * 20]
    assert set(runtime.latest_by_market()) == {"KR", "US"}
    by_symbol = {item.ticker: item for item in result.decisions}
    assert by_symbol["005930"].domestic_context["breadth"] == 1.0
    assert by_symbol["NVDA"].domestic_context["breadth"] == -1.0
    assert by_symbol["NVDA"].domestic_context["market"] == "US"
    assert by_symbol["005930"].sector_context["market_group"] == "KR"
    assert by_symbol["NVDA"].sector_context["market_group"] == "US"
    assert by_symbol["005930"].temporal_context != by_symbol["NVDA"].temporal_context
    assert set(result.as_dict()["markets"]) == {"KR", "US"}


def test_us_local_evidence_populates_us_graph_node_without_global_overwrite():
    local = DomesticContext(captured_at=NOW, context_id="us-local", direction=-0.7, breadth=-0.8, market="US")
    global_context = GlobalContextBuilder().build(
        (IndicatorObservation("SP500", 5000, NOW, source="fred", change_ratio=0.02),),
        captured_at=NOW, market="US",
    )
    snapshot = GraphSnapshotBuilder().build(
        captured_at=NOW, domestic_context=local, global_context=global_context
    )
    direction_slot = FEATURE_SLOTS["Market"].index("direction")
    us = snapshot.index_of("US_MARKET")
    kr = snapshot.index_of("KR_MARKET")
    assert snapshot.features[-1, us, direction_slot] == pytest.approx(-0.7)
    assert snapshot.features[-1, kr, direction_slot] == 0.0
    assert any(
        snapshot.features[-1, index, 0] != 0
        for index, node_type in enumerate(snapshot.node_types) if node_type == "MacroFactor"
    )


def test_macro_store_uses_only_observed_history_without_fabricating_timestamps(monkeypatch):
    import app.storage

    def record(value, timestamp):
        return SimpleNamespace(name="us_vix_close", value=value, observed_at=timestamp,
                               source=SimpleNamespace(source_name="fred"))

    records = [record(20, NOW - timedelta(days=2)), record(21, NOW - timedelta(days=1)),
               record(99, NOW + timedelta(days=1)), record(100, None)]
    monkeypatch.setattr(app.storage, "LocalResearchStore", lambda: SimpleNamespace(
        load_analysis_inputs=lambda **kwargs: SimpleNamespace(macro_metrics=records)
    ))
    runtime = ContextRuntime.__new__(ContextRuntime)
    observations = tuple(runtime._macro_observations(NOW))
    assert len(observations) == 1
    assert observations[0].value == 21
    assert observations[0].observed_at == NOW - timedelta(days=1)
    assert observations[0].change_ratio == pytest.approx(0.05)


def test_daily_flow_keeps_completed_business_date_as_event_time(monkeypatch):
    import app.data.investor_flow_store as module

    history = (
        module.InvestorFlowDay("005930", "20260720", 100, -100, 50, 50),
        module.InvestorFlowDay("005930", "20260804", 100, -20, 12, 8),
        module.InvestorFlowDay("005930", "20260805", 100, -1000, 500, 500),
    )
    monkeypatch.setattr(module, "InvestorFlowStore", lambda: SimpleNamespace(history=lambda symbol: history))
    events = []
    runtime = ContextRuntime.__new__(ContextRuntime)
    runtime._freshness = SimpleNamespace(record_event=lambda *args, **kwargs: events.append(args))
    flows = runtime._investor_flows((CandidateInput("005930"),), moment=NOW)
    assert flows == {"foreign": 12, "institution": 8, "retail": -20}
    assert events[0][2] == datetime(2026, 8, 4, 6, 30, tzinfo=timezone.utc)
