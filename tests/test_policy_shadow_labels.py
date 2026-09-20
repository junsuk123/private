from copy import deepcopy
from datetime import datetime, timedelta
import json
import sqlite3

import pytest

from app.features.strategy_graph_context import STRATEGY_GRAPH_CONTEXT_DIM, STRATEGY_GRAPH_CONTEXT_SCHEMA, context_index
from app.models.strategy_utility.policy_context import GraphPolicyContextCache
from app.models.strategy_utility.policy_labels import LABEL_EXECUTION_POLICY, load_policy_shadow_labels
from app.strategy.catalog import STRATEGY_IDS
from app.trading.directional import DirectionalStrategyKey
from app.trading.directional_shadow import QuoteObservation, ShadowFillSimulator, ShadowPlanStore, ShadowTradePlan
from test_ontology_shadow_alignment import _manager, _proposal
from test_ontology_thresholds import NOW, _policy


def _context(*, change=0.):
    features = [0.] * STRATEGY_GRAPH_CONTEXT_DIM
    features[context_index("is_krx")] = 1.
    features[0] = change
    cache = GraphPolicyContextCache(clock=lambda: NOW - timedelta(milliseconds=100))
    assert cache.record(dict(
        symbol="005930", market="KR", as_of=NOW - timedelta(seconds=1),
        valid_until=NOW + timedelta(seconds=4), features=features,
        feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA, data_fresh=True, tradable=True,
        snapshot_id="observed-live-frame-42", feature_snapshot_id="observed-features-42",
    ))
    return cache.latest("005930", NOW)


def _record(tmp_path, *, strategy="intraday_momentum", context=None, plan_id="plan-1"):
    context = context or _context()
    policy = _policy(forecast_gross_bps=600)
    manager = _manager(tmp_path, lambda **kwargs: policy)
    proposal = _proposal(strategy_id=strategy)
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    plan = ShadowTradePlan(
        plan_id=plan_id, key=DirectionalStrategyKey(strategy), symbol="005930", signal_at=NOW,
        entry_reference_price=100, target_rate=proposal.target_return_rate,
        stop_rate=proposal.stop_loss_rate, max_holding_seconds=proposal.max_holding_seconds,
        expected_trading_cost_bps=10, predicted_gross_edge_bps=600,
        feature_snapshot_id=context["feature_snapshot_id"],
        diagnostics={"exit_contract": proposal.exit_contract, "graph_training_context": context},
    )
    simulator = ShadowFillSimulator(entry_slippage_bps=0, exit_slippage_bps=0)
    simulator.submit(plan)
    assert not simulator.observe(QuoteObservation(NOW + timedelta(seconds=1), 99.99, 100))
    outcome = simulator.observe(QuoteObservation(NOW + timedelta(seconds=2), 102, 102.01))[0]
    store = ShadowPlanStore(tmp_path / "shadow.sqlite3")
    assert store.record_plan(plan) and store.record_outcome(outcome)
    return store, plan, outcome


def _mutate_json(store, table, path, value):
    column = "plan_json" if table == "shadow_plans" else "outcome_json"
    with sqlite3.connect(store.path) as conn:
        raw = json.loads(conn.execute(f"SELECT {column} FROM {table}").fetchone()[0])
        target = raw
        for field in path[:-1]:
            target = target[field]
        target[path[-1]] = value
        conn.execute(f"UPDATE {table} SET {column}=?", (json.dumps(raw),))


def test_readonly_forward_join_and_absent_arms_are_censored(tmp_path):
    store, plan, outcome = _record(tmp_path)
    before = store.path.read_bytes()
    labels = load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3))
    assert store.path.read_bytes() == before
    assert len(labels) == len(STRATEGY_IDS)
    observed = [row for row in labels if row.outcome_observed]
    assert len(observed) == 1
    label = observed[0]
    assert label.net_return_bps == pytest.approx(outcome.net_return_bps)
    assert label.label_execution_policy == LABEL_EXECUTION_POLICY
    assert label.policy_id == outcome.risk_policy_id
    assert label.feature_snapshot_id == plan.feature_snapshot_id
    assert label.feature_snapshot_at < label.as_of < label.label_end
    assert len(label.features) == STRATEGY_GRAPH_CONTEXT_DIM
    for row in labels:
        if row is not label:
            assert row.exit_reason == "FUTURE_WINDOW_CENSORED"
            assert not row.triggered and not row.filled and row.fill_ratio == 0
            assert row.policy_id == "", "An unobserved arm has no invented policy outcome"


@pytest.mark.parametrize(("table", "path", "value"), [
    ("shadow_outcomes", ("plan_id",), "different-plan"),
    ("shadow_outcomes", ("strategy_id",), "breakout_volume"),
    ("shadow_outcomes", ("symbol",), "AAPL"),
    ("shadow_outcomes", ("market",), "US"),
    ("shadow_outcomes", ("direction",), "SHORT"),
    ("shadow_outcomes", ("execution_product",), "CREDIT_BORROW"),
    ("shadow_outcomes", ("signal_at",), (NOW + timedelta(seconds=1)).isoformat()),
    ("shadow_outcomes", ("resolved_at",), (NOW + timedelta(days=1)).isoformat()),
    ("shadow_outcomes", ("risk_policy_id",), "foreign-policy"),
    ("shadow_outcomes", ("risk_policy_family",), "legacy"),
    ("shadow_outcomes", ("signal_admissible",), False),
    ("shadow_outcomes", ("net_return_bps",), float("nan")),
    ("shadow_outcomes", ("net_return_bps",), 999),
    ("shadow_outcomes", ("fill_ratio",), 0),
    ("shadow_plans", ("feature_snapshot_id",), "unknown-snapshot"),
    ("shadow_plans", ("diagnostics", "graph_training_context"), None),
    ("shadow_plans", ("diagnostics", "graph_training_context", "schema"), "old-schema"),
    ("shadow_plans", ("diagnostics", "graph_training_context", "features"), [0.] * 72),
    ("shadow_plans", ("diagnostics", "graph_training_context", "source"), "historical-reconstruction"),
    ("shadow_plans", ("diagnostics", "graph_training_context", "source_provenance"), []),
    ("shadow_plans", ("diagnostics", "graph_training_context", "recorded_at"), (NOW + timedelta(seconds=1)).isoformat()),
    ("shadow_plans", ("diagnostics", "graph_training_context", "as_of"), (NOW + timedelta(seconds=1)).isoformat()),
    ("shadow_plans", ("diagnostics", "exit_contract", "ontology_entry_permitted"), False),
    ("shadow_plans", ("diagnostics", "exit_contract", "forecast_gross_bps"), 700),
    ("shadow_plans", ("diagnostics", "exit_contract", "ontology_risk_policy", "expires_at"), (NOW - timedelta(seconds=1)).isoformat()),
    ("shadow_plans", ("target_rate",), .5),
])
def test_mismatched_tampered_future_or_missing_evidence_never_yields_labels(tmp_path, table, path, value):
    store, _, _ = _record(tmp_path)
    _mutate_json(store, table, path, value)
    assert load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3)) == ()


def test_mutating_vector_without_recomputing_snapshot_is_rejected(tmp_path):
    store, _, _ = _record(tmp_path)
    features = _context()["features"]
    features[0] += .25
    _mutate_json(store, "shadow_plans", ("diagnostics", "graph_training_context", "features"), features)
    assert load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3)) == ()


def test_future_resolution_and_relational_identity_mismatch_are_rejected(tmp_path):
    store, _, _ = _record(tmp_path)
    assert load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=1)) == ()
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE shadow_outcomes SET symbol='AAPL'")
    assert load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3)) == ()


def test_conflicting_feature_snapshots_in_one_decision_are_not_combined(tmp_path):
    store, _, _ = _record(tmp_path)
    _record(tmp_path, strategy="breakout_volume", context=_context(change=.25), plan_id="plan-2")
    assert load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3)) == ()


def test_matching_snapshots_preserve_observed_arms_and_limit_bounds_join(tmp_path):
    store, _, _ = _record(tmp_path)
    _record(tmp_path, strategy="breakout_volume", plan_id="plan-2")
    labels = load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3))
    assert sum(row.outcome_observed for row in labels) == 2
    bounded = load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3), limit=1)
    assert len(bounded) == len(STRATEGY_IDS)
    assert sum(row.outcome_observed for row in bounded) == 1


def test_missing_or_legacy_database_is_never_created_or_migrated(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    assert load_policy_shadow_labels(missing, now=NOW) == ()
    assert not missing.exists()
    legacy = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy) as conn:
        conn.execute("CREATE TABLE old_history (value INTEGER)")
    before = legacy.read_bytes()
    assert load_policy_shadow_labels(legacy, now=NOW) == ()
    assert legacy.read_bytes() == before


def test_session_captures_context_at_actual_journal_time_once_per_symbol(tmp_path, monkeypatch):
    journal_now = NOW + timedelta(milliseconds=200)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return journal_now

    monkeypatch.setattr("app.trading.strategy_session.datetime", Clock)
    store = ShadowPlanStore(tmp_path / "session-shadow.sqlite3")
    monkeypatch.setattr("app.trading.directional_shadow.default_shadow_store", lambda: store)
    manager = _manager(tmp_path, lambda **kwargs: _policy(forecast_gross_bps=600))
    context = _context()
    calls = []

    def provider(symbol, as_of):
        calls.append((symbol, as_of))
        return context

    manager.graph_training_context_provider = provider
    proposals = [_proposal(), _proposal(strategy_id="breakout_volume")]
    for proposal in proposals:
        assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    manager._journal_shadow_proposals(proposals, NOW)
    plans = manager.drain_shadow_plans()
    assert len(plans) == 2 and calls == [("005930", journal_now)]
    assert all(plan.signal_at == journal_now for plan in plans)
    frozen = deepcopy(plans[0].diagnostics["graph_training_context"])
    context["features"][0] = 99
    assert plans[0].diagnostics["graph_training_context"] == frozen
    assert plans[0].feature_snapshot_id == frozen["feature_snapshot_id"]


def test_session_cannot_backdate_expired_policy_for_training_capture(tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW + timedelta(minutes=1)

    monkeypatch.setattr("app.trading.strategy_session.datetime", Clock)
    store = ShadowPlanStore(tmp_path / "session-shadow.sqlite3")
    monkeypatch.setattr("app.trading.directional_shadow.default_shadow_store", lambda: store)
    manager = _manager(tmp_path, lambda **kwargs: _policy(forecast_gross_bps=600))
    manager.graph_training_context_provider = lambda symbol, as_of: _context()
    proposal = _proposal()
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    manager._journal_shadow_proposals([proposal], NOW)
    assert not manager.drain_shadow_plans()


def test_real_cache_session_simulator_journal_chain_produces_policy_labels(tmp_path, monkeypatch):
    from app.models.strategy_utility.training import _target_mask

    captured_at = NOW + timedelta(milliseconds=100)
    journal_at = NOW + timedelta(milliseconds=200)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return journal_at

    monkeypatch.setattr("app.trading.strategy_session.datetime", Clock)
    features = [0.] * STRATEGY_GRAPH_CONTEXT_DIM
    features[context_index("is_krx")] = 1.
    cache = GraphPolicyContextCache(clock=lambda: captured_at)
    assert cache.record(dict(
        symbol="005930", market="KR", as_of=NOW,
        valid_until=NOW + timedelta(seconds=5), features=features,
        feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA, data_fresh=True, tradable=True,
        snapshot_id="actual-live-frame", feature_snapshot_id="actual-live-features",
    ))
    # The cycle began before capture; using the old cycle time would lose this
    # real observation or incorrectly backdate its availability.
    assert cache.latest("005930", as_of=NOW) is None
    store = ShadowPlanStore(tmp_path / "forward-chain.sqlite3")
    monkeypatch.setattr("app.trading.directional_shadow.default_shadow_store", lambda: store)
    manager = _manager(tmp_path, lambda **kwargs: _policy(forecast_gross_bps=600))
    manager.graph_training_context_provider = cache.latest
    proposal = _proposal()
    assert manager._freeze_proposal_exit_contract(proposal, NOW, None)
    manager._journal_shadow_proposals([proposal], NOW)
    plans = manager.drain_shadow_plans()
    assert len(plans) == 1
    plan = plans[0]
    assert plan.signal_at == journal_at
    assert store.plan(plan.plan_id)["feature_snapshot_id"] == plan.feature_snapshot_id
    simulator = ShadowFillSimulator(entry_slippage_bps=0, exit_slippage_bps=0)
    simulator.submit(plan)
    assert not simulator.observe(QuoteObservation(NOW + timedelta(seconds=1), 99.99, 100))
    outcomes = simulator.observe(QuoteObservation(NOW + timedelta(seconds=2), 102, 102.01))
    assert len(outcomes) == 1 and outcomes[0].scored
    assert store.record_outcome(outcomes[0])
    labels = load_policy_shadow_labels(store.path, now=NOW + timedelta(seconds=3))
    assert len(labels) == len(STRATEGY_IDS)
    observed = [row for row in labels if row.outcome_observed]
    assert len(observed) == 1
    assert observed[0].strategy_id == proposal.strategy_id
    assert observed[0].net_return_bps == pytest.approx(outcomes[0].net_return_bps)
    assert observed[0].feature_snapshot_id == plan.feature_snapshot_id
    assert observed[0].policy_id == proposal.exit_contract["policy_id"]
    assert observed[0].label_execution_policy == LABEL_EXECUTION_POLICY
    assert any(_target_mask(observed[0]))
    missing = [row for row in labels if not row.outcome_observed]
    assert len(missing) == len(STRATEGY_IDS) - 1
    assert all(not any(_target_mask(row)) for row in missing)
