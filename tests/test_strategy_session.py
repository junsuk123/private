from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.schemas.domain import AccountSnapshot, Holding
from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager


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
