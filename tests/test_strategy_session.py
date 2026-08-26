from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.schemas.domain import AccountSnapshot, Holding
from app.trading.directional import PositionDirection
from app.trading.strategy_session import (
    StrategySessionConfig,
    StrategySessionManager,
    _ElectionProposal,
    _cost_market_contract,
)
from app.technical.strategy_algorithms import round_trip_cost_bps


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _legacy_live_authority_for_session_state_machine_tests(monkeypatch):
    """These tests exercise ownership after an already-authorised election.

    Production now starts every unproven arm in SHADOW. Explicit test authority
    keeps the state-machine fixtures focused on their original subject.
    """
    for strategy_id in (
        "intraday_momentum",
        "breakout_volume",
        "vwap_mean_reversion",
        "liquidity_shock_reversal",
        "event_momentum",
        "cross_sectional_relative_strength",
        "gap_context",
        "rvgi_box_breakout",
        "residual_relative_strength",
        "adaptive_anchored_vwap_reversion",
        "ofi_microprice_exhaustion_reversal",
        "opening_range_breakout",
        "bar_confirmed_vwap_recovery",
    ):
        monkeypatch.setenv(f"ALGO_{strategy_id.upper()}_LIVE_AUTHORIZED", "1")
    yield
    monkeypatch.undo()
    from app.strategy.registry import reset_default_strategy_registry

    reset_default_strategy_registry()


def _config(tmp_path, **overrides):
    values = {
        "state_path": str(tmp_path / "session.json"),
        "cooldown_seconds": 5,
        "entry_timeout_seconds": 30,
        "invalidation_confirm_cycles": 2,
        "require_live_gnn": False,
    }
    values.update(overrides)
    return StrategySessionConfig(**values)


def _bundle(*, symbol="005930", strategy="intraday_momentum", exit_symbol=None,
            exit_price=71_120.0):
    # +160bps: the signal forecasts a move to the strategy's own target. It has to
    # be a real cost-clearing forecast, because the cost gate now divides the
    # SIGNAL's predicted move by the all-in cost. The previous +50bps fixture only
    # elected because the gate used to substitute the exit barrier for the forecast,
    # which made the ratio unfalsifiable.
    intent = SimpleNamespace(
        side="BUY",
        symbol=symbol,
        selected_strategy=strategy,
        expected_entry_price=70_000.0,
        expected_exit_price=exit_price,
        score=82.0,
        confidence=0.88,
        macro_regime="RISK_ON",
        micro_regime="MOMENTUM",
        reason_codes=("POSITIVE_NET_EDGE",),
        explanation_paths=({"from": "macro", "to": strategy},),
    )
    return SimpleNamespace(
        ranked_trade_intents=(intent,),
        buy_candidates=(symbol,),
        sell_reduce_candidates=((exit_symbol,) if exit_symbol else ()),
        blocked_candidates=(),
        micro_results=(),
    )


def _continuation_bundle(at, *, symbol="005930", exit_signal="RISK_REDUCE"):
    micro = SimpleNamespace(
        symbol=symbol,
        exit_signal=exit_signal,
        reason_codes=("MOMENTUM_LOSS", "VWAP_BREAKDOWN"),
    )
    return SimpleNamespace(
        timestamp=at,
        macro_result=None,
        micro_results=(micro,),
        ranked_trade_intents=(),
        sell_reduce_candidates=(symbol,),
        buy_candidates=(),
        blocked_candidates=(),
    )


def _account(*holdings):
    return AccountSnapshot(cash=200_000.0, holdings=tuple(holdings), total_equity_krw=200_000.0)


def test_session_uses_the_injected_live_plan_builder(tmp_path):
    live_builder = object()
    manager = StrategySessionManager(
        config=_config(tmp_path),
        plan_builder=live_builder,
    )

    assert manager._plan_builder is live_builder  # noqa: SLF001


def test_plan_cost_contract_matches_algorithm_market_cost_contract():
    assert _cost_market_contract("005930") == ("KRX", "domestic_stock")
    assert _cost_market_contract("F") == ("NASD", "overseas_stock")


def test_session_locks_one_candidate_until_position_is_flat(tmp_path):
    evidence = {
        "005930": {
            "decisions": [
                {
                    "path": "cpu_gnn",
                    "action": "ACTIVATE",
                    "strategy_id": "intraday_momentum",
                    "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                }
            ]
        }
    }
    manager = StrategySessionManager(
        config=_config(tmp_path),
        selection_evidence_provider=lambda symbols: evidence,
    )

    selected = manager.evaluate(_account(), ("005930", "000660"), _bundle(), NOW)
    assert selected["phase"] == "ARMED"
    assert selected["selected_symbol"] == "005930"
    assert selected["selection_source"] == "ONTOLOGY_GNN_AGREEMENT"
    assert manager.allowed_buy_candidates(("005930", "000660"), _account()) == ("005930",)

    manager.mark_entry_submitted("005930", NOW)
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=2,
        average_price=70_000.0,
        last_price=70_100.0,
        opened_at=NOW,
    )
    owned = manager.evaluate(_account(holding), ("000660",), _bundle(), NOW + timedelta(seconds=1))
    assert owned["phase"] == "OWNED"
    assert manager.allowed_buy_candidates(("000660",), _account(holding)) == ()

    target_holding = Holding(
        **{**holding.__dict__, "last_price": float(owned["target_price"]) + 1.0}
    )
    exiting = manager.evaluate(
        _account(target_holding),
        ("000660",),
        _bundle(),
        NOW + timedelta(seconds=2),
    )
    assert exiting["phase"] == "EXITING"
    assert manager.exit_reason_for(target_holding) == "STRATEGY_PROFIT_TARGET"

    # Only a broker-confirmed exit FILL closes the position. A balance response that
    # stopped listing the lot is a partial reply until it is corroborated.
    manager.mark_exit_filled(
        "005930", float(owned["target_price"]) + 1.0, NOW + timedelta(seconds=3)
    )
    flat = manager.evaluate(_account(), ("000660",), _bundle(symbol="000660"), NOW + timedelta(seconds=3))
    assert flat["phase"] == "COOLDOWN"
    assert manager.allowed_buy_candidates(("000660",), _account()) == ()

    reselection = manager.evaluate(
        _account(),
        ("000660",),
        _bundle(symbol="000660"),
        NOW + timedelta(seconds=9),
    )
    assert reselection["phase"] == "ARMED"
    assert reselection["selected_symbol"] == "000660"


def test_owned_strategy_exits_after_distinct_confirmed_edge_decay(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, invalidation_confirm_cycles=2),
    )
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=2,
        average_price=70_000.0,
        last_price=69_860.0,
        opened_at=NOW,
    )
    # Adopt the broker position first, then continuously consume the held-symbol
    # micro analysis. One deterioration observation is not enough to liquidate.
    manager.evaluate(_account(holding), (), _bundle(), NOW)
    first = manager.evaluate(
        _account(holding),
        (),
        _continuation_bundle(NOW + timedelta(seconds=5)),
        NOW + timedelta(seconds=5),
    )
    assert first["phase"] == "OWNED"
    assert first["invalidation_cycles"] == 1

    second = manager.evaluate(
        _account(holding),
        (),
        _continuation_bundle(NOW + timedelta(seconds=10)),
        NOW + timedelta(seconds=10),
    )
    assert second["phase"] == "EXITING"
    assert second["exit_reason"] == "STRATEGY_EDGE_DECAY_LOSS_LIMIT"
    assert "MICRO_RISK_REDUCE" in second["invalidation_reason_codes"]


def test_throttled_bundle_is_counted_only_once_for_invalidation(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, invalidation_confirm_cycles=2),
    )
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=1,
        average_price=70_000.0,
        last_price=69_900.0,
        opened_at=NOW,
    )
    manager.evaluate(_account(holding), (), _bundle(), NOW)
    stale_bundle = _continuation_bundle(NOW + timedelta(seconds=5))
    manager.evaluate(_account(holding), (), stale_bundle, NOW + timedelta(seconds=5))
    repeated = manager.evaluate(
        _account(holding), (), stale_bundle, NOW + timedelta(seconds=6)
    )
    assert repeated["phase"] == "OWNED"
    assert repeated["invalidation_cycles"] == 1


def test_cost_blind_time_exit_gets_one_bounded_extension_when_thesis_is_intact(
    tmp_path,
):
    manager = StrategySessionManager(config=_config(tmp_path))
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=1,
        average_price=70_000.0,
        last_price=70_000.0,
        opened_at=NOW,
    )
    manager.evaluate(_account(holding), (), _bundle(), NOW)
    state = manager._state  # noqa: SLF001
    state.max_holding_seconds = 60
    state.expected_cost_bps = 28.0
    state.invalidation_cycles = 0
    state.invalidation_reason_codes = []

    manager._evaluate_exit(holding, None, NOW + timedelta(seconds=61))  # noqa: SLF001

    assert state.phase == "OWNED"
    assert state.holding_extension_used is True
    assert state.holding_extension_seconds == 60
    assert state.max_holding_seconds == 120

    manager._evaluate_exit(holding, None, NOW + timedelta(seconds=121))  # noqa: SLF001
    assert state.phase == "EXITING"
    assert state.exit_reason == "STRATEGY_MAX_HOLDING_TIME"


def test_confirmed_edge_decay_protects_net_profit_before_static_target(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, invalidation_confirm_cycles=2),
    )
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=2,
        average_price=70_000.0,
        # +40bps gross clears the 28bps fallback cost but remains below the normal
        # strategy target, so this specifically exercises predictive profit protection.
        last_price=70_280.0,
        opened_at=NOW,
    )
    adopted = manager.evaluate(_account(holding), (), _bundle(), NOW)
    assert float(adopted["target_price"]) > holding.last_price
    manager.evaluate(
        _account(holding),
        (),
        _continuation_bundle(NOW + timedelta(seconds=5)),
        NOW + timedelta(seconds=5),
    )
    exiting = manager.evaluate(
        _account(holding),
        (),
        _continuation_bundle(NOW + timedelta(seconds=10)),
        NOW + timedelta(seconds=10),
    )
    assert exiting["phase"] == "EXITING"
    assert exiting["exit_reason"] == "STRATEGY_EDGE_DECAY_PROFIT_PROTECT"


def test_actionable_gnn_election_takes_execution_ownership(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path),
        selection_evidence_provider=lambda symbols: {
            "005930": {
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "ACTIVATE_STRATEGY",
                        "strategy_id": "breakout_volume",
                        "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                    }
                ]
            }
        },
    )
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "breakout_volume"
    assert state["selection_source"] == "GNN_STRATEGY_ELECTION"


def test_gnn_without_realtime_trust_cannot_take_execution_ownership(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, require_live_gnn=True),
        selection_evidence_provider=lambda symbols: {
            "005930": {
                "as_of": NOW.isoformat(),
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "ACTIVATE_STRATEGY",
                        "strategy_id": "gap_context",
                        "reason_codes": [
                            "MAX_NET_UTILITY",
                            "GNN_REALTIME_TRUST_NOT_READY",
                        ],
                    }
                ],
            }
        },
    )
    no_entry_bundle = SimpleNamespace(
        ranked_trade_intents=(),
        buy_candidates=(),
        sell_reduce_candidates=(),
        blocked_candidates=("005930",),
        micro_results=(),
    )

    state = manager.evaluate(_account(), ("005930",), no_entry_bundle, NOW)

    assert state["phase"] == "SCANNING"
    assert state["selected_strategy"] is None
    assert state["last_reason"] == "GNN_NOT_LIVE_AUTHORIZED"


def test_production_config_requires_live_gnn_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("STRATEGY_SESSION_REQUIRE_LIVE_GNN", raising=False)

    config = StrategySessionConfig(state_path=str(tmp_path / "session.json"))

    assert config.require_live_gnn is True


def test_production_config_disables_duplicate_bandit_authority_by_default(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("STRATEGY_SESSION_BANDIT_ENABLED", raising=False)

    config = StrategySessionConfig(state_path=str(tmp_path / "session.json"))

    assert config.algorithm_primary_election is True
    assert config.bandit_enabled is False


def test_algorithm_catalogue_is_evaluated_when_gnn_vector_is_empty(tmp_path, monkeypatch):
    """A checkpoint/schema failure must not turn implemented strategies into 0/0."""
    # This test isolates deterministic authority from the independent exit-economics
    # threshold, which has dedicated coverage elsewhere.
    monkeypatch.setenv("STRATEGY_SESSION_MIN_NET_REWARD_RISK", "0.0")

    class _Decision:
        def as_dict(self):
            return {
                "strategy_id": "intraday_momentum",
                "triggered": True,
                "score": 0.9,
                "confidence": 0.8,
                "expected_edge_bps": 90.0,
                "horizon_seconds": 180,
                "reason_codes": ["MECHANICAL_TEST_TRIGGER"],
                "diagnostics": {},
            }

    class _Algorithm:
        def p(self, name):
            return 1.0

        def entry(self, features, context):
            return _Decision()

        def exit_rule(self, entry_price, features, context):
            return SimpleNamespace(
                target_price=entry_price * 1.009,
                stop_price=entry_price * 0.995,
                trailing_bps=20.0,
                max_holding_seconds=180,
                target_basis="test_forecast",
                stop_basis="test_stop",
            )

    manager = StrategySessionManager(
        config=_config(
            tmp_path,
            bandit_enabled=False,
            require_live_gnn=True,
            algorithm_primary_election=True,
        ),
        selection_evidence_provider=lambda symbols: {
            "005930": {
                "as_of": NOW.isoformat(),
                "mark_price": 70_000.0,
                "technical_features": {"symbol": "005930", "price": 70_000.0},
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "NO_TRADE",
                        "strategy_id": None,
                        "reason_codes": ["GNN_FEATURE_SCHEMA_MISMATCH"],
                    }
                ],
                "validation_candidates": [],
            }
        },
    )
    manager._algorithm_registry = {"intraday_momentum": _Algorithm()}
    # The production path must not consult the historical selector after the
    # owned algorithm has fired. A call here is a duplicate decision authority.
    manager.bandit.select = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("bandit must not re-judge a deterministic trigger")
    )
    bundle = SimpleNamespace(
        ranked_trade_intents=(),
        buy_candidates=("005930",),
        sell_reduce_candidates=(),
        blocked_candidates=(),
        micro_results=(),
        macro_result=None,
    )

    state = manager.evaluate(_account(), ("005930",), bundle, NOW)

    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "intraday_momentum"
    assert state["selection_source"] == "ONTOLOGY_ALGORITHM_ELECTION"
    assert state["selection_authority"] == "DETERMINISTIC_ALGORITHM"
    assert state["bandit_evaluations"] == []
    assert manager.entry_order_size_fraction("005930") > 0.0
    evaluated = {
        item["strategy_id"]: item for item in state["algorithm_evaluations"]
    }
    assert "intraday_momentum" in evaluated
    assert evaluated["intraday_momentum"]["triggered"] is True


def test_untrusted_gnn_duplicate_cannot_replace_deterministic_algorithm_edge() -> None:
    common = dict(
        symbol="INTC",
        strategy_id="bar_confirmed_vwap_recovery",
        entry_price=91.38,
        target_return_rate=0.0225,
        stop_loss_rate=0.007,
        trailing_stop_rate=0.003,
        max_holding_seconds=5400,
        score=0.8,
        confidence=0.8,
        expected_cost_bps=46.09,
        gnn_action="NO_TRADE",
        gnn_reason_codes=["GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED"],
        ontology_reason_codes=[],
        macro_regime="TREND_UP",
        micro_regime="RECOVERY",
        explanation_paths=[],
        candidate_count=1,
        micro_result=None,
        evidence_row={},
        last_reason="TEST",
        direction=PositionDirection.LONG,
    )
    deterministic = _ElectionProposal(
        **common,
        source="ONTOLOGY_ALGORITHM_ELECTION",
        expected_net_return_bps=19.44,
        gnn_actionable=False,
        intent=None,
        gnn_required_for_edge=False,
    )
    validation_duplicate = _ElectionProposal(
        **{**common, "confidence": 0.99},
        source="ALGORITHM_MECHANICAL_ELECTION",
        expected_net_return_bps=19.44,
        gnn_actionable=False,
        intent=object(),
        gnn_required_for_edge=True,
    )

    selected = StrategySessionManager._deduplicate_joint_proposals(
        [deterministic, validation_duplicate]
    )

    assert selected == [deterministic]
    assert selected[0].predicted_net_edge_bps(15.0, 60.0) == 19.44


def test_insufficient_cost_coverage_cannot_arm_a_live_session(tmp_path) -> None:
    """A ranked proposal inside the cost error band must remain non-executable."""
    manager = StrategySessionManager(config=_config(tmp_path))
    proposal = _ElectionProposal(
        symbol="DYN",
        strategy_id="gap_context",
        source="ONTOLOGY_ALGORITHM_ELECTION",
        entry_price=26.16,
        target_return_rate=0.0063,
        stop_loss_rate=0.007,
        trailing_stop_rate=0.003,
        max_holding_seconds=900,
        score=0.8,
        confidence=0.8,
        # 13 + 50 = 63 gross bps; 63 / 50 = 1.26, below live 1.3.
        expected_net_return_bps=13.0,
        expected_cost_bps=50.0,
        gnn_actionable=True,
        gnn_action="ACTIVATE_STRATEGY",
        gnn_reason_codes=[],
        ontology_reason_codes=[],
        macro_regime="TREND_UP",
        micro_regime="GAP",
        explanation_paths=[],
        intent=None,
        candidate_count=1,
        micro_result=None,
        evidence_row={},
        last_reason="TEST",
        direction=PositionDirection.LONG,
        gnn_required_for_edge=False,
        algorithm_triggered=True,
    )

    armed = manager._arm(proposal, NOW, account=_account())  # noqa: SLF001

    state = manager.snapshot()
    assert armed is False
    assert state["phase"] == "SCANNING"
    assert state["selected_symbol"] is None
    assert state["cost_coverage_band"] == "INSUFFICIENT"
    assert state["last_reason"].startswith("ENTRY_COST_COVERAGE_REJECTED:INSUFFICIENT")


def test_cold_algorithm_exploration_is_capped_to_minimum_size(tmp_path):
    manager = StrategySessionManager(
        config=_config(
            tmp_path,
            bandit_exploration_size_fraction=0.10,
        ),
        selection_evidence_provider=lambda symbols: {},
    )
    manager._state.selected_symbol = "005930"
    manager._state.selected_strategy = "intraday_momentum"
    manager._state.selected_deployment_state = "LIVE_FULL"
    manager._state.selector_v2_order_size_fraction = 1.0
    manager._state.bandit_is_exploration = True

    assert manager.entry_order_size_fraction("005930") == 0.10


def test_rvgi_box_ontology_election_freezes_context_when_rollout_authorized(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ALGO_RVGI_BOX_BREAKOUT_LIVE_AUTHORIZED", "1")
    context = {
        "ontology_eligible": True,
        "rvgi": 0.1,
        "rvgi_signal": 0.05,
        "rvgi_diff": 0.05,
        "rvgi_bullish_cross": True,
        "box_high": 100.0,
        "box_low": 98.0,
        "box_mid": 99.0,
        "box_width_pct": 2 / 99,
        "box_position": 1.0,
        "box_context_timestamp": NOW.isoformat(),
        "box_previous_close": 99.8,
        "volume_confirmed": True,
    }
    manager = StrategySessionManager(
        config=_config(tmp_path),
        selection_evidence_provider=lambda symbols: {
            "005930": {
                "as_of": NOW.isoformat(),
                "rvgi_box_context": context,
                "decisions": [
                    {
                        "path": "ontology",
                        "action": "ACTIVATE_STRATEGY",
                        "strategy_id": "rvgi_box_breakout",
                        "reason_codes": ["RVGI_BOX_ALLOWED"],
                    }
                ],
            }
        },
    )
    state = manager.evaluate(
        _account(),
        ("005930",),
        _bundle(strategy="breakout"),
        NOW,
    )
    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "rvgi_box_breakout"
    assert state["election_context"]["box_high"] == 100.0
    assert state["election_context"]["box_context_timestamp"] == NOW.isoformat()


def test_unavailable_gnn_keeps_reason_and_uses_ontology_guard(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path),
        selection_evidence_provider=lambda symbols: {
            "005930": {
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "NO_TRADE",
                        "strategy_id": None,
                        "reason_codes": ["GNN_NOT_LIVE_AUTHORIZED"],
                    }
                ]
            }
        },
    )
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "intraday_momentum"
    assert state["selection_source"] == "ONTOLOGY_WITH_GNN_GUARD"
    assert state["gnn_reason_codes"] == ["GNN_NOT_LIVE_AUTHORIZED"]


def test_owned_strategy_ignores_later_ontology_block_and_uses_its_stop(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=1,
        average_price=70_000.0,
        last_price=70_010.0,
        opened_at=NOW,
    )
    adverse_ontology = SimpleNamespace(
        ranked_trade_intents=(),
        buy_candidates=(),
        sell_reduce_candidates=("005930",),
        blocked_candidates=("005930",),
        micro_results=(),
    )
    owned = manager.evaluate(
        _account(holding),
        (),
        adverse_ontology,
        NOW + timedelta(seconds=1),
    )
    assert owned["phase"] == "OWNED"
    assert manager.exit_reason_for(holding) is None

    stopped = Holding(
        **{
            **holding.__dict__,
            "last_price": 70_000.0 * (1.0 - owned["stop_loss_rate"]) - 1.0,
        }
    )
    state = manager.evaluate(
        _account(stopped),
        (),
        adverse_ontology,
        NOW + timedelta(seconds=2),
    )
    assert state["phase"] == "EXITING"
    assert manager.exit_reason_for(stopped) == "STRATEGY_STOP_LOSS"


def test_broker_confirmed_exit_fill_suppresses_duplicate_sell(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=1,
        average_price=70_000.0,
        last_price=69_000.0,
        opened_at=NOW,
    )
    manager.evaluate(_account(holding), (), _bundle(), NOW + timedelta(seconds=1))
    assert manager.exit_reason_for(holding) == "STRATEGY_STOP_LOSS"

    filled_at = NOW + timedelta(seconds=2)
    manager.mark_exit_filled("005930", 69_050.0, filled_at)

    state = manager.snapshot()
    assert state["exit_price"] == 69_050.0
    assert state["exit_filled_at"] == filled_at.isoformat()
    assert state["last_reason"] == "EXIT_FILLED_AWAITING_ACCOUNT_FLAT"
    assert manager.exit_reason_for(holding) is None


def test_fast_round_trip_closes_even_when_holdings_never_observed(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    # Simulate the broker filling both legs before account reconciliation saw
    # the holding. This was the DYN lifecycle that remained EXITING for hours.
    assert manager.request_halt("005930", "HARD", ("TEST_HALT",)) is True
    manager.mark_exit_filled("005930", 70_000.0, NOW + timedelta(seconds=2))
    assert manager.snapshot()["position_seen"] is False

    state = manager.evaluate(
        _account(), (), _bundle(), NOW + timedelta(seconds=3)
    )

    assert state["phase"] == "COOLDOWN"
    assert state["last_reason"] == "POSITION_FLAT_RESELECTION_COOLDOWN"


def test_missing_us_cost_uses_overseas_fee_policy_not_kr_fallback(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, fallback_round_trip_cost_bps=28.0),
        selection_evidence_provider=lambda symbols: {
            "AAPL": {
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "ACTIVATE_STRATEGY",
                        "strategy_id": "intraday_momentum",
                        "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                    }
                ]
            }
        },
    )

    us_regular = NOW.replace(hour=15)
    state = manager.evaluate(
        _account(), ("AAPL",), _bundle(symbol="AAPL"), us_regular
    )

    assert state["phase"] == "ARMED"
    assert state["expected_cost_bps"] == round_trip_cost_bps("AAPL")
    assert state["expected_cost_bps"] > 28.0


def test_generic_ontology_admissibility_does_not_elect_first_strategy(tmp_path):
    no_entry_bundle = SimpleNamespace(
        ranked_trade_intents=(),
        buy_candidates=(),
        sell_reduce_candidates=(),
        blocked_candidates=("005930",),
        micro_results=(),
    )
    manager = StrategySessionManager(
        config=_config(tmp_path),
        selection_evidence_provider=lambda symbols: {
            "005930": {
                "as_of": NOW.isoformat(),
                "decisions": [
                    {
                        "path": "ontology",
                        "action": "ADMISSIBLE",
                        "strategy_id": "intraday_momentum",
                        "reason_codes": [],
                    },
                    {
                        "path": "cpu_gnn",
                        "action": "NO_TRADE",
                        "strategy_id": None,
                        "reason_codes": ["NON_POSITIVE_NET_EDGE"],
                    },
                ],
            }
        },
    )
    state = manager.evaluate(
        _account(),
        ("005930",),
        no_entry_bundle,
        NOW,
    )
    assert state["phase"] == "SCANNING"
    assert state["selected_strategy"] is None
    assert state["last_reason"] == "NO_FRESH_STRATEGY_ELECTION"
    assert manager.allowed_buy_candidates((), _account()) == ()


def test_armed_strategy_releases_owner_when_entry_window_expires(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, armed_timeout_seconds=30),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    expired = manager.evaluate(
        _account(),
        ("005930",),
        _bundle(),
        NOW + timedelta(seconds=31),
    )
    assert expired["phase"] == "COOLDOWN"
    assert expired["last_reason"] == "STRATEGY_ENTRY_WINDOW_EXPIRED"
    assert manager.allowed_buy_candidates(("005930",), _account()) == ()


def _held(**overrides):
    values = {
        "ticker": "005930",
        "market": "KR",
        "company_name": "Samsung",
        "sector": "Technology",
        "quantity": 1,
        "average_price": 70_000.0,
        "last_price": 70_050.0,
        "opened_at": NOW,
    }
    values.update(overrides)
    return Holding(**values)


class _RecordingStore:
    """Counts what the session claims is a realized round trip.

    Also answers the two READ calls the deployment ladder makes, as an empty store
    would: no outcomes, and a posterior that cannot speak. Without them the ladder's
    safety boundary swallows an AttributeError and resolves every arm to SHADOW, so
    the session under test could never arm anything.
    """

    def __init__(self):
        self.records = []

    def record(self, **payload):
        self.records.append(payload)
        return True

    def recent_outcomes(self, *args, **kwargs):
        return ()

    def posterior(self, *args, **kwargs):
        return SimpleNamespace(conservative_edge_bps=float("-inf"), loss_streak=0)


def test_one_absent_holdings_snapshot_does_not_close_an_owned_position(tmp_path):
    """A partial KIS balance response is not an exit.

    This is the 064260 lifecycle: the lot was held continuously, but a balance
    response that omitted it ended the session, wrote a phantom outcome, and
    re-armed the same lot on the unknown-thesis fallback geometry.
    """
    store = _RecordingStore()
    manager = StrategySessionManager(
        config=_config(tmp_path),
        selection_evidence_provider=lambda symbols: {},
        performance_store=store,
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    holding = _held()
    owned = manager.evaluate(_account(holding), (), _bundle(), NOW + timedelta(seconds=1))
    assert owned["phase"] == "OWNED"

    vanished = manager.evaluate(_account(), (), _bundle(), NOW + timedelta(seconds=2))
    assert vanished["phase"] == "OWNED"
    assert vanished["last_reason"] == "POSITION_MISSING_FROM_SNAPSHOT_UNCONFIRMED"
    assert store.records == []

    restored = manager.evaluate(
        _account(holding), (), _bundle(), NOW + timedelta(seconds=3)
    )
    assert restored["phase"] == "OWNED"
    assert restored["selected_strategy"] == "intraday_momentum"
    assert restored["missing_holding_observations"] == 0
    assert store.records == []


def test_sustained_disappearance_still_closes_the_position(tmp_path):
    """The guard delays the close; it must never prevent one."""
    store = _RecordingStore()
    manager = StrategySessionManager(
        config=_config(
            tmp_path, flat_confirm_observations=3, flat_confirm_seconds=90.0
        ),
        selection_evidence_provider=lambda symbols: {},
        performance_store=store,
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    manager.evaluate(_account(_held()), (), _bundle(), NOW + timedelta(seconds=1))

    for offset in (2, 3, 4):
        state = manager.evaluate(
            _account(), (), _bundle(), NOW + timedelta(seconds=offset)
        )
        assert state["phase"] == "OWNED", offset

    closed = manager.evaluate(_account(), (), _bundle(), NOW + timedelta(seconds=200))
    assert closed["phase"] == "COOLDOWN"
    assert closed["last_reason"] == "POSITION_FLAT_RESELECTION_COOLDOWN"
    assert len(store.records) == 1
    # A close nobody asked for is recorded as such, not with a blank reason that
    # is indistinguishable from a phantom row afterwards.
    assert store.records[0]["exit_reason"] == "POSITION_CLOSED_EXTERNALLY"


def test_confirmed_exit_fill_closes_on_the_first_absent_snapshot(tmp_path):
    """The order endpoint is authoritative and must not wait for the balance."""
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    manager.evaluate(_account(_held()), (), _bundle(), NOW + timedelta(seconds=1))
    manager.mark_exit_filled("005930", 70_100.0, NOW + timedelta(seconds=2))

    closed = manager.evaluate(_account(), (), _bundle(), NOW + timedelta(seconds=3))
    assert closed["phase"] == "COOLDOWN"


def test_readopted_lot_keeps_its_armed_thesis_and_entry_clock(tmp_path):
    """A reset while the lot is still held must not downgrade it to `hold`.

    ``max_holding_seconds`` is the field that mattered live: re-adoption restarted
    the clock at the unknown-thesis 1200s, so 010140 was time-stopped at exactly
    its entry price 20 minutes into a thesis with a far longer horizon.
    """
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    owned = manager.evaluate(_account(_held()), (), _bundle(), NOW + timedelta(seconds=1))
    assert owned["selected_strategy"] == "intraday_momentum"
    armed_holding_seconds = owned["max_holding_seconds"]
    armed_stop_rate = owned["stop_loss_rate"]
    armed_opened_at = owned["position_opened_at"]

    # A reset that happens while the broker still holds the lot: cooldown expiry,
    # a supervisor halt, a rejected exit.
    manager._reset_to_scanning("TEST_RESET_WHILE_HELD")  # noqa: SLF001

    readopted = manager.evaluate(
        _account(_held()), (), _bundle(), NOW + timedelta(seconds=30)
    )
    assert readopted["phase"] == "OWNED"
    assert readopted["selected_strategy"] == "intraday_momentum"
    assert readopted["max_holding_seconds"] == armed_holding_seconds
    assert readopted["stop_loss_rate"] == armed_stop_rate
    assert readopted["position_opened_at"] == armed_opened_at
    assert readopted["last_reason"] == "EXISTING_POSITION_READOPTED_WITH_ARMED_THESIS"


def test_readopted_lot_at_a_different_average_price_does_not_inherit_the_memo(tmp_path):
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    manager.evaluate(_account(_held()), (), _bundle(), NOW + timedelta(seconds=1))
    manager._reset_to_scanning("TEST_RESET_WHILE_HELD")  # noqa: SLF001

    different = manager.evaluate(
        _account(_held(average_price=71_500.0)),
        (),
        _bundle(),
        NOW + timedelta(seconds=30),
    )
    assert different["last_reason"] == "EXISTING_POSITION_ADOPTED"
    assert different["position_opened_at"] == NOW.isoformat()


def test_micro_hold_verdict_never_becomes_the_adopted_strategy(tmp_path):
    """`hold` is an action, not a thesis, and resolves to no geometry row."""
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
    )
    bundle = SimpleNamespace(
        timestamp=NOW,
        macro_result=None,
        micro_results=(
            SimpleNamespace(symbol="005930", selected_strategy=SimpleNamespace(value="hold")),
        ),
        ranked_trade_intents=(),
        sell_reduce_candidates=(),
        buy_candidates=(),
        blocked_candidates=(),
    )
    adopted = manager.evaluate(_account(_held()), (), bundle, NOW)
    assert adopted["phase"] == "OWNED"
    assert adopted["selected_strategy"] == "risk_managed_existing_position"


def test_krx_cost_uses_the_fee_policy_not_the_configured_reference():
    """A domestic proposal was priced at the 28bps constant, never at the policy.

    The KRX policy round trip is 33.8bps (fees + 20bps transfer tax + slippage +
    safety margin). Short-circuiting KR to the configured reference understated every
    domestic cost by ~6bps before the spread was even considered.
    """
    from app.trading.strategy_session import _market_round_trip_cost_bps

    domestic = _market_round_trip_cost_bps("005930", 28.0)
    assert domestic > 28.0, "KR must resolve through the fee policy, not the constant"
    # The configured reference stays a floor, so a config gap cannot make a trade
    # look cheaper than the operator's own minimum.
    assert _market_round_trip_cost_bps("005930", 999.0) == 999.0


def test_all_in_cost_charges_the_spread_the_round_trip_actually_crosses():
    """`spread_rate` is 0 in the fee policy, so the spread has to be added here.

    This is the 2026-08-21 defect: 064260's predicted 41bps gross edge was divided by
    28bps and banded THIN (live-eligible). Against fees plus its own 18.4bps spread
    the ratio is 0.79 — NOT_COVERED, i.e. never trade.
    """
    from app.cost.cost_coverage import CostCoverageBand, evaluate_cost_coverage
    from app.trading.strategy_session import (
        _all_in_round_trip_cost_bps,
        _market_round_trip_cost_bps,
    )

    micro = SimpleNamespace(diagnostics={"spread_bps": 18.4})
    policy = _market_round_trip_cost_bps("064260", 28.0)
    all_in = _all_in_round_trip_cost_bps(
        "064260", fallback_bps=28.0, micro_result=micro
    )
    assert all_in == policy + 18.4

    gross_edge_bps = 41.1
    assert evaluate_cost_coverage(gross_edge_bps, 28.0).live_eligible is True
    verdict = evaluate_cost_coverage(gross_edge_bps, all_in)
    assert verdict.band is CostCoverageBand.NOT_COVERED
    assert verdict.live_eligible is False


def test_all_in_cost_floors_a_model_estimate_rather_than_trusting_it():
    """A cost estimate is only ever dangerous when it is too small."""
    from app.trading.strategy_session import _all_in_round_trip_cost_bps

    micro = SimpleNamespace(diagnostics={"spread_bps": 20.0})
    optimistic = _all_in_round_trip_cost_bps(
        "005930", fallback_bps=28.0, model_estimate_bps=5.0, micro_result=micro
    )
    assert optimistic > 40.0
    pessimistic = _all_in_round_trip_cost_bps(
        "005930", fallback_bps=28.0, model_estimate_bps=500.0, micro_result=micro
    )
    assert pessimistic == 500.0


def test_edge_below_its_own_spread_cannot_arm_a_live_session(tmp_path):
    """End to end: the election must refuse what the coverage arithmetic refuses."""
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
    )
    thin = SimpleNamespace(
        side="BUY",
        symbol="064260",
        selected_strategy="cross_sectional_relative_strength",
        expected_entry_price=5_440.0,
        # ~41bps of gross edge: clears a 28bps fee-only bar, not a 52bps all-in one.
        expected_exit_price=5_462.0,
        score=90.0,
        confidence=0.8,
        macro_regime="RISK_ON",
        micro_regime="MOMENTUM",
        reason_codes=("POSITIVE_NET_EDGE",),
        explanation_paths=(),
    )
    bundle = SimpleNamespace(
        ranked_trade_intents=(thin,),
        buy_candidates=("064260",),
        sell_reduce_candidates=(),
        blocked_candidates=(),
        micro_results=(SimpleNamespace(symbol="064260", diagnostics={"spread_bps": 18.4}),),
        timestamp=NOW,
    )
    state = manager.evaluate(_account(), ("064260",), bundle, NOW)
    assert state["phase"] != "ARMED"


def _perf_store(tmp_path, **kwargs):
    from app.trading.strategy_performance_store import (
        PosteriorConfig,
        StrategyPerformanceStore,
    )

    return StrategyPerformanceStore(
        tmp_path / "perf.sqlite3",
        posterior_config=PosteriorConfig(),
        cache_ttl_seconds=0.0,
        clock=lambda: NOW,
        **kwargs,
    )


def _seed(store, *, strategy="intraday_momentum", n, net_bps):
    for index in range(n):
        store.record(
            strategy_id=strategy,
            symbol="005930",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=net_bps,
            direction="LONG",
            execution_product="CASH",
            recorded_at=NOW - timedelta(minutes=index),
        )


def test_live_flag_cannot_outrank_a_measured_negative_edge(tmp_path):
    """The 2026-08-21 governance bypass.

    ``evaluate_long_promotion`` returned SHADOW for every LONG strategy in the
    catalogue — conservative edges of -85 to -281bps — while all of them armed at
    LIVE_FULL, because the ``live_authorized`` config flag short-circuited before the
    controller ran. The flag may permit live authority; it may not assert it over
    evidence that says the arm loses money.
    """
    from app.trading.directional import PositionDirection, StrategyDeploymentState

    store = _perf_store(tmp_path)
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
        performance_store=store,
    )
    authorized, _ = manager._deployment_authorized("intraday_momentum")  # noqa: SLF001
    assert authorized, "fixture assumes the flag grants this arm live authority"

    _seed(store, n=30, net_bps=-43.0)
    state = manager._directional_deployment_state(  # noqa: SLF001
        "intraday_momentum", PositionDirection.LONG, "KR"
    )
    assert state is StrategyDeploymentState.SHADOW
    assert state.submits_orders is False


def test_thin_evidence_leaves_the_operator_flag_alone(tmp_path):
    """"No evidence yet" is not "bad evidence".

    A cold arm keeps what the flag grants it: the flag is how a new thesis gets its
    first fills, and demoting on three observations would make the ladder
    unclimbable.
    """
    from app.trading.directional import PositionDirection, StrategyDeploymentState

    store = _perf_store(tmp_path)
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
        performance_store=store,
    )
    _seed(store, n=5, net_bps=-43.0)
    assert (
        manager._directional_deployment_state(  # noqa: SLF001
            "intraday_momentum", PositionDirection.LONG, "KR"
        )
        is StrategyDeploymentState.LIVE_FULL
    )


def test_measured_positive_edge_keeps_full_authority(tmp_path):
    """Demotion only. The rule must never hold back an arm that is working."""
    from app.trading.directional import PositionDirection, StrategyDeploymentState

    store = _perf_store(tmp_path)
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
        performance_store=store,
    )
    _seed(store, n=40, net_bps=65.0)
    assert (
        manager._directional_deployment_state(  # noqa: SLF001
            "intraday_momentum", PositionDirection.LONG, "KR"
        )
        is StrategyDeploymentState.LIVE_FULL
    )


def test_a_demoted_arm_cannot_submit_an_entry_order(tmp_path):
    """End to end: the demotion has to reach the order path, not just the label."""
    store = _perf_store(tmp_path)
    _seed(store, n=30, net_bps=-43.0)
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False),
        selection_evidence_provider=lambda symbols: {},
        performance_store=store,
    )
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "SCANNING"
    assert state["selected_symbol"] is None
    # Named for the ladder, not for the short book: a LONG demotion reported as
    # SHORT_STRATEGY_SHADOW_ONLY sends an operator hunting a borrow problem.
    assert state["last_reason"].startswith("STRATEGY_DEPLOYMENT_SHADOW_ONLY")
    assert manager.allowed_buy_candidates(("005930",), _account()) == ()


def test_confirmed_exit_fill_with_a_stale_balance_row_releases_the_session(tmp_path):
    """The DYN 2026-08-20 deadlock: EXITING was an absorbing state.

    The broker reported the exit FILLED, so ``exit_reason_for`` refused to emit a
    second SELL, while the balance endpoint kept returning the sold lot, so
    ``_reconcile_position`` took the "holding present" branch on every cycle and the
    flat transition was never reached. The session held phase EXITING for 8h27m --
    electing nothing on EITHER market, because ``allowed_entry_candidates`` returns
    nothing while any holding exists -- until the balance row happened to clear.
    """
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False, exit_reconcile_timeout_seconds=60.0),
        selection_evidence_provider=lambda symbols: {},
    )
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=2,
        average_price=70_000.0,
        last_price=70_100.0,
        opened_at=NOW,
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    manager.evaluate(_account(holding), (), _bundle(), NOW + timedelta(seconds=1))
    manager.mark_exit_submitted("005930", NOW + timedelta(seconds=2))
    manager.mark_exit_filled("005930", 70_100.0, NOW + timedelta(seconds=3))

    # Inside the reconciliation window the stale row is still respected: a lagging
    # balance response must not be able to reopen a closed thesis immediately.
    held = manager.evaluate(_account(holding), (), _bundle(), NOW + timedelta(seconds=30))
    assert held["phase"] == "EXITING"

    released = manager.evaluate(
        _account(holding), (), _bundle(), NOW + timedelta(seconds=180)
    )

    assert released["phase"] == "COOLDOWN"
    assert released["last_reason"] == "EXIT_UNRECONCILED_BALANCE_ROW_STALE"
    # The fill claim is retired with the phase, so a lot that turns out to still be
    # held can be exited again instead of sitting unmanaged with no stop.
    assert released["exit_filled_at"] is None
    assert manager.exit_reason_for(holding) is None  # not EXITING any more


def test_halt_before_any_fill_does_not_strand_the_session_in_exiting(tmp_path):
    """A HARD halt from ENTERING reached EXITING with nothing to exit.

    ``position_seen`` is False and no exit ever filled, so the flat-reconciliation
    branch excluded the state by its own guard; ``entry_timeout_seconds`` no longer
    applied because the phase was no longer ENTERING; and no holding existed for the
    exit path to act on. Nothing in the machine could leave it.
    """
    manager = StrategySessionManager(
        config=_config(tmp_path, record_outcomes=False, exit_reconcile_timeout_seconds=60.0),
        selection_evidence_provider=lambda symbols: {},
    )
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    assert manager.request_halt("005930", "HARD", ("DATA_STALE:16.0s>15s",)) is True

    waiting = manager.evaluate(_account(), (), _bundle(), NOW + timedelta(seconds=1))
    assert waiting["phase"] == "EXITING"
    assert waiting["last_reason"] == "EXIT_AWAITING_EXPOSURE_CONFIRMATION"

    released = manager.evaluate(_account(), (), _bundle(), NOW + timedelta(seconds=180))

    assert released["phase"] == "SCANNING"
    assert released["selected_symbol"] is None
