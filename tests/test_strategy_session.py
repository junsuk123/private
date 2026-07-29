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
                    "reason_codes": [],
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
                        "action": "ACTIVATE",
                        "strategy_id": "breakout_volume",
                        "reason_codes": [],
                    }
                ]
            }
        },
    )
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "breakout_volume"
    assert state["selection_source"] == "GNN_STRATEGY_ELECTION"


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
                        "reason_codes": ["UTILITY_MODEL_NOT_LIVE_AUTHORIZED"],
                    }
                ]
            }
        },
    )
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "intraday_momentum"
    assert state["selection_source"] == "ONTOLOGY_WITH_GNN_GUARD"
    assert state["gnn_reason_codes"] == ["UTILITY_MODEL_NOT_LIVE_AUTHORIZED"]


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


def test_fresh_ontology_election_arms_strategy_before_entry_trigger(tmp_path):
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
    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "intraday_momentum"
    assert state["selection_source"] == "ONTOLOGY_STRATEGY_ELECTION"
    assert state["last_reason"] == "STRATEGY_ELECTED_WAITING_FOR_ENTRY_TRIGGER"
    assert manager.allowed_buy_candidates((), _account()) == ("005930",)


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
