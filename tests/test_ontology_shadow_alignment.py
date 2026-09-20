from dataclasses import replace
from datetime import timedelta

import pytest

from app.risk.ontology_thresholds import POLICY_FAMILY_VERSION
from app.trading.directional import StrategyDeploymentState
from app.trading.directional_shadow import QuoteObservation, ShadowFillSimulator, ShadowPlanStore, plan_signal_admissible
from app.trading.shadow_evaluation_service import ShadowEvaluationService
from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager, _ElectionProposal
from app.trading.strategy_performance_store import StrategyPerformanceStore
from test_ontology_thresholds import NOW, _policy


def _proposal(**changes):
    values = dict(
        symbol="005930", strategy_id="intraday_momentum", source="TEST", entry_price=100,
        target_return_rate=.06, stop_loss_rate=.04, trailing_stop_rate=.02, max_holding_seconds=900,
        score=.9, confidence=.9, expected_net_return_bps=590, expected_cost_bps=10,
        gnn_actionable=True, gnn_action="ACTIVATE_STRATEGY", gnn_reason_codes=[], ontology_reason_codes=[],
        macro_regime="TREND_UP", micro_regime="MOMENTUM", explanation_paths=[], intent=None,
        candidate_count=1, micro_result=None, evidence_row={}, last_reason="TEST",
        gnn_required_for_edge=False, algorithm_triggered=True, deployment_state=StrategyDeploymentState.SHADOW,
    )
    values.update(changes)
    return _ElectionProposal(**values)


def _manager(tmp_path, resolver):
    return StrategySessionManager(config=StrategySessionConfig(state_path=str(tmp_path / "session.json")),
                                  ontology_policy_resolver=resolver, selector_v2_runner=False)


def test_shadow_and_live_proposals_share_current_policy_without_promoting_shadow(tmp_path):
    policy = _policy(forecast_gross_bps=600)
    manager = _manager(tmp_path, lambda **kwargs: policy)
    shadow = _proposal()
    live = _proposal(deployment_state=StrategyDeploymentState.LIVE_FULL)
    assert manager._freeze_proposal_exit_contract(shadow, NOW, None)
    assert manager._freeze_proposal_exit_contract(live, NOW, None)
    assert shadow.exit_contract == live.exit_contract
    assert shadow.target_return_rate == policy.target_return_rate
    assert shadow.stop_loss_rate == policy.soft_stop_rate
    assert shadow.max_holding_seconds == policy.maximum_holding_seconds
    assert shadow.submits_orders is False
    assert live.submits_orders is True
    assert shadow.exit_contract["risk_policy_family"] == POLICY_FAMILY_VERSION
    assert shadow.exit_contract["policy_id"] == policy.policy_id


def test_original_forecast_and_horizon_survive_repeated_freezing(tmp_path):
    calls = []

    def resolver(**kwargs):
        calls.append(kwargs)
        return _policy(forecast_gross_bps=kwargs["forecast_gross_bps"], requested_horizon_seconds=kwargs["requested_horizon_seconds"])

    manager = _manager(tmp_path, resolver)
    proposal = _proposal()
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    assert [item["requested_horizon_seconds"] for item in calls] == [900, 900]
    assert [item["forecast_gross_bps"] for item in calls] == [600, 600]
    assert proposal.exit_contract["forecast_gross_bps"] != proposal.target_return_rate * 10000


@pytest.mark.parametrize("change", [
    {"expires_at": NOW - timedelta(seconds=1)}, {"market": "US"},
    {"valid_for_entry": False, "reason_codes": ("POLICY_METRIC_MISSING:spread_rate",)},
])
def test_invalid_market_evidence_cannot_create_promotable_shadow_geometry(tmp_path, change):
    manager = _manager(tmp_path, lambda **kwargs: replace(_policy(), **change))
    proposal = _proposal()
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None) is False
    assert proposal.exit_contract == {}


def test_shadow_journal_and_completed_outcome_retain_policy_family(tmp_path, monkeypatch):
    policy = _policy(forecast_gross_bps=600)
    manager = _manager(tmp_path, lambda **kwargs: policy)
    journal = ShadowPlanStore(tmp_path / "shadow.sqlite3")
    monkeypatch.setattr("app.trading.directional_shadow.default_shadow_store", lambda: journal)
    proposal = _proposal()
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    manager._journal_shadow_proposals([proposal], NOW)
    plans = manager.drain_shadow_plans()
    assert len(plans) == 1
    plan = plans[0]
    assert plan.diagnostics["exit_contract"]["policy_id"] == policy.policy_id
    assert plan.predicted_gross_edge_bps == 600
    assert plan_signal_admissible(plan)
    simulator = ShadowFillSimulator(entry_slippage_bps=0, exit_slippage_bps=0)
    simulator.submit(plan)
    assert not simulator.observe(QuoteObservation(NOW + timedelta(seconds=1), 99.99, 100))
    outcome = simulator.observe(QuoteObservation(NOW + timedelta(seconds=2), 102, 102.01))[0]
    assert outcome.scored
    assert outcome.risk_policy_family == POLICY_FAMILY_VERSION
    assert outcome.risk_policy_id == policy.policy_id
    store = StrategyPerformanceStore(tmp_path / "performance.sqlite3", clock=lambda: NOW + timedelta(seconds=3))
    service = ShadowEvaluationService(shadow_store=journal, performance_store=store)
    service._persist(outcome)
    row = store.recent_outcomes("intraday_momentum", as_of=NOW + timedelta(seconds=3))[0]
    assert row.risk_policy_family == POLICY_FAMILY_VERSION
    assert row.evaluation_source == "shadow"


def test_weak_forecast_remains_research_without_entry_or_promotion_permission(tmp_path, monkeypatch):
    policy = _policy(forecast_gross_bps=2)
    assert policy.reason_codes == ("POLICY_NET_REWARD_INSUFFICIENT",)
    manager = _manager(tmp_path, lambda **kwargs: policy)
    journal = ShadowPlanStore(tmp_path / "shadow.sqlite3")
    monkeypatch.setattr("app.trading.directional_shadow.default_shadow_store", lambda: journal)
    proposal = _proposal(deployment_state=StrategyDeploymentState.LIVE_FULL, expected_net_return_bps=-8)
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    assert not proposal.submits_orders
    manager._journal_shadow_proposals([proposal], NOW)
    plan = manager.drain_shadow_plans()[0]
    assert not plan_signal_admissible(plan)
    assert plan.target_rate == policy.target_return_rate
    simulator = ShadowFillSimulator(entry_slippage_bps=0, exit_slippage_bps=0)
    simulator.submit(plan)
    simulator.observe(QuoteObservation(NOW + timedelta(seconds=1), 99.99, 100))
    outcome = simulator.observe(QuoteObservation(NOW + timedelta(seconds=2), 102, 102.01))[0]
    assert outcome.scored and not outcome.signal_admissible
