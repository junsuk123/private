from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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


def _bundle(*, symbol="005930", strategy="intraday_momentum", exit_symbol=None):
    intent = SimpleNamespace(
        side="BUY",
        symbol=symbol,
        selected_strategy=strategy,
        expected_entry_price=70_000.0,
        expected_exit_price=70_350.0,
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


def test_algorithm_catalogue_is_evaluated_when_gnn_vector_is_empty(tmp_path):
    """A checkpoint/schema failure must not turn implemented strategies into 0/0."""

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
