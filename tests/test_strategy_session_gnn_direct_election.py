"""GNN-direct election: the model's pick is armed without a second opinion.

The posture exists because an operator holds that a model trained to select the
best strategy should not then be re-judged by the layers below it. These tests
pin what that actually changes, including the part that is a loss of protection
-- a posture whose downside is untested is a posture nobody can evaluate later.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.graph.macro_reasoner import build_sector_rank_table
from app.schemas.domain import AccountSnapshot
from app.trading.conservative_bandit import BanditConfig, ConservativeStrategyBandit
from app.trading.strategy_performance_store import PosteriorConfig, StrategyPerformanceStore
from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager


# Monday 10:00 KST -- inside the KRX regular session. A weekend timestamp makes
# every test here fail on ``NEW_ENTRY_OUTSIDE_REGULAR_SESSION`` before election
# is ever reached, which looks exactly like the posture not working.
NOW = datetime(2026, 8, 10, 1, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _authorize_strategy_under_election(monkeypatch):
    """This module tests election posture after deployment authorization."""
    monkeypatch.setenv("ALGO_INTRADAY_MOMENTUM_LIVE_AUTHORIZED", "1")
    yield
    monkeypatch.undo()
    from app.strategy.registry import reset_default_strategy_registry

    reset_default_strategy_registry()


def _store(tmp_path) -> StrategyPerformanceStore:
    return StrategyPerformanceStore(
        tmp_path / "perf.sqlite3",
        posterior_config=PosteriorConfig(),
        cache_ttl_seconds=0.0,
    )


def _manager(tmp_path, *, store=None, **config_overrides):
    store = store if store is not None else _store(tmp_path)
    values = {
        "state_path": str(tmp_path / "session.json"),
        "cooldown_seconds": 5,
        "entry_timeout_seconds": 30,
        "require_live_gnn": False,
        # Control cases in this module explicitly compare GNN-direct with the
        # legacy bandit, not with the production deterministic selector.
        "algorithm_primary_election": False,
        "bandit_enabled": True,
    }
    values.update(config_overrides)
    return StrategySessionManager(
        config=StrategySessionConfig(**values),
        selection_evidence_provider=(lambda symbols: {}),
        performance_store=store,
        bandit=ConservativeStrategyBandit(store=store, config=BanditConfig()),
    )


def _macro(**overrides):
    base = dict(
        market_regime=SimpleNamespace(value="HIGH_VOL_TRENDING"),
        reason_codes=("MACRO_HIGH_VOL_TRENDING",),
        explanation_paths=(),
        allowed_micro_strategies=("relative_strength", "momentum"),
        blocked_micro_strategies=(),
        change_point_probability=0.0,
        regime_stability=0.9,
        volatility_percentile=0.8,
        spread_percentile=0.4,
        foreign_flow_zscore=0.2,
        diagnostics={"market_breadth": 0.4},
        sector_rank_table=build_sector_rank_table(sector_of={}, residual_returns={}),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _bundle(*, symbol="005930", strategy="intraday_momentum", exit_price=70_350.0):
    intent = SimpleNamespace(
        side="BUY",
        symbol=symbol,
        selected_strategy=strategy,
        expected_entry_price=70_000.0,
        expected_exit_price=exit_price,
        score=82.0,
        confidence=0.88,
        macro_regime="HIGH_VOL_TRENDING",
        micro_regime="MOMENTUM",
        reason_codes=("POSITIVE_NET_EDGE",),
        explanation_paths=({"from": "macro", "to": strategy},),
        rank=1,
    )
    return SimpleNamespace(
        ranked_trade_intents=(intent,),
        buy_candidates=(symbol,),
        sell_reduce_candidates=(),
        blocked_candidates=(),
        micro_results=(),
        macro_result=_macro(),
    )


def _account():
    return AccountSnapshot(cash=200_000.0, holdings=(), total_equity_krw=200_000.0)


def _losing_store(tmp_path) -> StrategyPerformanceStore:
    """History that makes the pessimistic bound refuse to trade.

    Deliberately BELOW ``LongPromotionConfig.minimum_shadow_samples`` (20). Past
    that count the store is allowed to speak about the arm, and a non-positive
    conservative edge demotes it out of live authority entirely — at which point no
    election posture can arm it and this test would be measuring the deployment
    ladder instead of the posture. Under the count, the bandit still refuses on
    pessimism while the arm keeps whatever the operator's flag grants it, which is
    exactly the cold-arm case this posture exists to override.
    """
    store = _store(tmp_path)
    for index in range(15):
        store.record(
            strategy_id="intraday_momentum",
            symbol="005930",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=-43.0,
            recorded_at=NOW - timedelta(minutes=index),
        )
    return store


def test_direct_election_arms_what_the_bandit_would_have_refused(tmp_path):
    """The defining difference, stated as a loss of protection.

    Same losing history that produces NO_TRADE under the bandit. Under this
    posture it arms instead. If this test ever passes for the wrong reason --
    say the history stopped being negative -- the sibling test below fails too,
    so the pair cannot both drift into vacuous agreement.
    """
    manager = _manager(
        tmp_path,
        store=_losing_store(tmp_path),
        gnn_direct_election=True,
    )
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)

    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "intraday_momentum"
    assert "GNN_DIRECT_ELECTION" in state["bandit_reason_codes"]
    # Never labelled exploration: this is a conviction pick, and calling it a
    # probe would let a loss read as a deliberate minimum-size experiment.
    assert state["bandit_is_exploration"] is False


def test_the_same_history_refuses_to_trade_under_the_bandit(tmp_path):
    """Control for the test above: the protection being given up is real."""
    manager = _manager(tmp_path, store=_losing_store(tmp_path))
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)

    assert state["phase"] == "SCANNING"
    assert state["selected_strategy"] is None
    assert state["bandit_selected_arm"] == "no_trade"


def test_posture_is_off_unless_explicitly_configured(tmp_path):
    """A live-order posture must never arrive by default."""
    assert StrategySessionConfig(state_path=str(tmp_path / "s.json")).gnn_direct_election is False


def test_arming_without_any_model_estimate_says_so(tmp_path):
    """The sharpest edge of this posture, pinned rather than discovered later.

    "Honour the model's pick" presupposes the model made one. With no evidence
    row there is no forward edge to honour, yet a lone proposal still wins by
    default -- the posture has no NO_TRADE to fall back to. That is arming a
    candidate the model never spoke to, so the reason codes must say exactly
    that; otherwise the session state is indistinguishable from a conviction
    pick backed by a real estimate.
    """
    manager = _manager(
        tmp_path,
        store=_losing_store(tmp_path),
        gnn_direct_election=True,
    )
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)

    assert state["phase"] == "ARMED"
    assert "GNN_DIRECT_ELECTION" in state["bandit_reason_codes"]
    assert "GNN_ESTIMATE_UNAVAILABLE_RANKED_LAST" in state["bandit_reason_codes"]
    # No fabricated zero: an absent edge stays absent rather than being recorded
    # as a measured 0.0bps, which would read as "flat" instead of "unknown".
    assert state["bandit_conservative_edge_bps"] is None
