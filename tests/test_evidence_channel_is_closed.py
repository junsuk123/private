"""The validation loop must be able to reach a verdict other than "no".

The defect this guards
----------------------
Repeated validation only means something if a failing verdict can be revisited.
Measured against the stored evidence base on 2026-08-11 it could not be, for two
compounding reasons:

* **The evidence channel was wired backwards.** ``_journal_shadow_proposals``
  skipped every ``submits_orders`` proposal, so a SHADOW arm accumulated a
  posterior on every cycle while a LIVE_FULL arm produced a sample only by winning
  an election — which it could only do on a posterior it had no way to build. All
  1,650 journaled shadow plans were US; the funded KR side had never produced one.

* **A single loss was terminal.** Cold-start exploration required
  ``loss_streak == 0``, and the posterior window only advanced when a new outcome
  arrived. After one losing outcome an arm was neither explorable (streak) nor
  exploitable (negative posterior), and nothing could arrive to change either.

Both are absorbing states, and an absorbing state is not a strict gate — it is a
gate that has stopped being a measurement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.conservative_bandit import (
    BANDIT_ARM_COLD_START_EXPLORATION,
    ArmCandidate,
    BanditConfig,
    BanditContext,
    ConservativeStrategyBandit,
)
from app.trading.strategy_performance_store import (
    PosteriorConfig,
    StrategyPerformanceStore,
)

NOW = datetime(2026, 8, 11, 1, 0, tzinfo=timezone.utc)


def _store(tmp_path, **posterior_overrides) -> StrategyPerformanceStore:
    return StrategyPerformanceStore(
        tmp_path / "perf.sqlite3",
        posterior_config=PosteriorConfig(**posterior_overrides),
        cache_ttl_seconds=0.0,
    )


def _loss(store: StrategyPerformanceStore, *, recorded_at: datetime, bps: float = -120.0) -> None:
    store.record(
        strategy_id="intraday_momentum",
        symbol="005930",
        market="KR",
        regime="HIGH_VOL_TRENDING",
        realized_net_bps=bps,
        recorded_at=recorded_at,
        evaluation_source="shadow",
    )


def _candidate() -> ArmCandidate:
    return ArmCandidate(
        arm="intraday_momentum",
        symbol="005930",
        predicted_net_edge_bps=30.0,
        confidence=0.7,
    )


def _context() -> BanditContext:
    return BanditContext(
        market="KR",
        macro_regime="HIGH_VOL_TRENDING",
        volatility_percentile=0.6,
        market_breadth=0.5,
        spread_percentile=0.4,
        liquidity_score=0.8,
    )


def test_one_loss_does_not_permanently_disqualify_a_cold_arm(tmp_path) -> None:
    """A single loss inside the first three trades is a coin flip, not an answer."""
    store = _store(tmp_path)
    _loss(store, recorded_at=NOW - timedelta(hours=2))
    bandit = ConservativeStrategyBandit(store=store, config=BanditConfig())

    selection = bandit.select((_candidate(),), _context(), now=NOW)

    assert not selection.is_no_trade, selection.reason_codes
    assert selection.is_exploration
    assert BANDIT_ARM_COLD_START_EXPLORATION in selection.evaluations[0].reason_codes


def test_two_consecutive_losses_still_stop_a_cold_arm(tmp_path) -> None:
    """The relaxation is one loss wide, not a removal of the check."""
    store = _store(tmp_path)
    _loss(store, recorded_at=NOW - timedelta(hours=3))
    _loss(store, recorded_at=NOW - timedelta(hours=2))
    bandit = ConservativeStrategyBandit(store=store, config=BanditConfig())

    selection = bandit.select((_candidate(),), _context(), now=NOW)

    assert selection.is_no_trade, selection.reason_codes


def test_evidence_ages_out_so_a_condemned_arm_goes_cold_again(tmp_path) -> None:
    """Without ageing the last verdict stands forever, because it is what stops
    the arm producing the next one."""
    store = _store(tmp_path, max_age_days=21.0)
    for offset in range(4):
        _loss(store, recorded_at=NOW - timedelta(days=40 + offset))

    stale = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING"
    )
    assert stale.sample_count == 0, "40-day-old fills must not still be voting"

    fresh_store = _store(tmp_path, max_age_days=0.0)
    unaged = fresh_store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING"
    )
    assert unaged.sample_count == 4, "ageing must be switchable off, not baked in"


def test_recent_evidence_is_not_aged_out(tmp_path) -> None:
    """Ageing must not become a way to forget a verdict that is still current."""
    store = _store(tmp_path, max_age_days=21.0)
    for offset in range(4):
        _loss(store, recorded_at=NOW - timedelta(days=offset))

    posterior = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING"
    )
    assert posterior.sample_count == 4
    assert posterior.conservative_edge_bps < 0.0


# --- Counterfactual journaling ---------------------------------------------- #


def _proposal(strategy_id: str, symbol: str = "005930"):
    from app.trading.strategy_session import _ElectionProposal

    return _ElectionProposal(
        symbol=symbol,
        strategy_id=strategy_id,
        source="TEST",
        entry_price=70_000.0,
        target_return_rate=0.016,
        stop_loss_rate=0.006,
        trailing_stop_rate=0.003,
        max_holding_seconds=3600,
        score=0.7,
        confidence=0.6,
        expected_net_return_bps=25.0,
        expected_cost_bps=28.0,
        gnn_actionable=True,
        gnn_action="ACTIVATE_STRATEGY",
        gnn_reason_codes=[],
        ontology_reason_codes=["TEST_SIGNAL"],
        macro_regime="HIGH_VOL_TRENDING",
        micro_regime="",
        explanation_paths=[],
        intent=None,
        candidate_count=2,
        micro_result=None,
        evidence_row=None,
        last_reason="",
    )


class _RecordingStore:
    def __init__(self) -> None:
        self.plans: list = []

    def record_plan(self, plan) -> bool:
        self.plans.append(plan)
        return True

    def has_recent_plan(self, *_args, **_kwargs) -> bool:
        return False


def _manager_with_store(tmp_path, store, monkeypatch):
    from app.trading import directional_shadow
    from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager

    monkeypatch.setattr(directional_shadow, "default_shadow_store", lambda: store)
    return StrategySessionManager(
        config=StrategySessionConfig(state_path=str(tmp_path / "session.json")),
        selection_evidence_provider=(lambda symbols: {}),
        performance_store=_store(tmp_path),
    )


def test_losing_order_authorised_arms_are_journaled_as_counterfactuals(
    tmp_path, monkeypatch
) -> None:
    """The arms that lost the election are exactly the ones with no other way to
    ever produce evidence."""
    store = _RecordingStore()
    manager = _manager_with_store(tmp_path, store, monkeypatch)
    winner = _proposal("intraday_momentum")
    loser = _proposal("breakout_volume", symbol="000660")

    manager._journal_shadow_proposals(
        [winner, loser], NOW, counterfactual=True, exclude=winner
    )

    journaled = {plan.key.strategy_id for plan in store.plans}
    assert journaled == {"breakout_volume"}, "the winner must not be scored twice"
    assert "COUNTERFACTUAL_UNSELECTED_ARM" in store.plans[0].signal_reason_codes


def test_counterfactual_pass_does_not_drop_the_shadow_pass_plan_ids(
    tmp_path, monkeypatch
) -> None:
    """Both passes run in the same cycle; the second must extend, not overwrite."""
    store = _RecordingStore()
    manager = _manager_with_store(tmp_path, store, monkeypatch)
    from app.trading.directional import StrategyDeploymentState

    shadow_arm = _proposal("residual_relative_strength", symbol="035420")
    shadow_arm.deployment_state = StrategyDeploymentState.SHADOW
    live_arm = _proposal("breakout_volume", symbol="000660")

    manager._journal_shadow_proposals([shadow_arm, live_arm], NOW)
    after_shadow = list(manager._state.shadow_plan_ids)
    manager._journal_shadow_proposals(
        [live_arm], NOW, counterfactual=True, exclude=None
    )

    assert len(after_shadow) == 1
    assert manager._state.shadow_plan_ids[: len(after_shadow)] == after_shadow
    assert len(manager._state.shadow_plan_ids) == 2
    assert len(manager._pending_shadow_plans) == 2


# --- Symbol conditioning ------------------------------------------------------ #
#
# What gets adopted is a (symbol, strategy) PAIR. Election has produced pairs since
# ``_evidence_proposals`` was written, but the realized half of a pair's score was
# looked up by strategy alone, so every candidate for one strategy received the same
# pooled mean and the ranking could only separate pairs through the forward estimate.
#
# The pooled mean was hiding a real effect. Across the 19 symbols with >=8 stored
# liquidity_shock_reversal fills, per-symbol means span -205..-54 bps; subtracting the
# spread that sampling noise alone produces (29.8 of the observed 39.1 sd) leaves a
# true symbol effect of 25.2 bps sd.


def _outcome(store, symbol: str, bps: float, *, recorded_at) -> None:
    store.record(
        strategy_id="intraday_momentum",
        symbol=symbol,
        market="KR",
        regime="HIGH_VOL_TRENDING",
        realized_net_bps=bps,
        recorded_at=recorded_at,
        evaluation_source="shadow",
    )


def test_the_same_strategy_scores_differently_on_different_symbols(tmp_path) -> None:
    store = _store(tmp_path, symbol_shrinkage_weight=10.0)
    for index in range(30):
        _outcome(store, "005930", 40.0, recorded_at=NOW - timedelta(hours=index + 1))
        _outcome(store, "000660", -40.0, recorded_at=NOW - timedelta(hours=index + 1))

    good = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING", symbol="005930"
    )
    bad = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING", symbol="000660"
    )

    assert good.posterior_mean_net_bps > bad.posterior_mean_net_bps
    assert good.symbol_weight == bad.symbol_weight > 0.0
    assert "STRATEGY_POSTERIOR_SYMBOL_CONDITIONED" in good.reason_codes


def test_an_unseen_symbol_falls_back_to_the_pooled_mean(tmp_path) -> None:
    """No symbol evidence must mean no adjustment — not a penalty.

    Penalising unseen names would make a strategy unreachable on every symbol it had
    not already traded, which is the absorbing state one level down.
    """
    store = _store(tmp_path)
    for index in range(20):
        _outcome(store, "005930", 30.0, recorded_at=NOW - timedelta(hours=index + 1))

    pooled = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    unseen = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING", symbol="035420"
    )

    assert unseen.symbol_sample_count == 0
    assert unseen.symbol_weight == 0.0
    assert unseen.posterior_mean_net_bps == pooled.posterior_mean_net_bps


def test_a_thin_symbol_barely_moves_the_score(tmp_path) -> None:
    """Shrinkage, not partitioning: three unlucky fills must not veto a name."""
    store = _store(tmp_path, symbol_shrinkage_weight=27.0)
    for index in range(40):
        _outcome(store, "005930", 50.0, recorded_at=NOW - timedelta(hours=index + 1))
    for index in range(3):
        _outcome(store, "000660", -400.0, recorded_at=NOW - timedelta(minutes=index + 1))

    pooled = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    thin = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING", symbol="000660"
    )

    assert thin.symbol_sample_count == 3
    assert thin.symbol_weight < 0.12
    # Moved, but nowhere near the -400 the three fills alone would imply.
    assert thin.posterior_mean_net_bps < pooled.posterior_mean_net_bps
    assert thin.posterior_mean_net_bps > pooled.posterior_mean_net_bps - 60.0


def test_symbol_conditioning_never_gates_the_exploration_machinery(tmp_path) -> None:
    """Counts and streaks stay strategy-level, so one bad name cannot suspend an arm."""
    store = _store(tmp_path)
    for index in range(20):
        _outcome(store, "005930", 30.0, recorded_at=NOW - timedelta(hours=index + 2))
    for index in range(5):
        _outcome(store, "000660", -50.0, recorded_at=NOW - timedelta(minutes=index + 1))

    conditioned = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING", symbol="005930"
    )
    pooled = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    assert conditioned.sample_count == pooled.sample_count
    assert conditioned.loss_streak == pooled.loss_streak


def test_shrinkage_weight_zero_restores_the_pooled_posterior(tmp_path) -> None:
    store = _store(tmp_path, symbol_shrinkage_weight=0.0)
    for index in range(20):
        _outcome(store, "005930", 30.0, recorded_at=NOW - timedelta(hours=index + 1))

    conditioned = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING", symbol="005930"
    )
    pooled = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    assert conditioned.posterior_mean_net_bps == pooled.posterior_mean_net_bps
    assert conditioned.symbol_weight == 0.0


def test_bandit_ranks_two_symbols_of_the_same_strategy_apart(tmp_path) -> None:
    """End to end: the pair, not the strategy, is what the bandit compares."""
    store = _store(tmp_path, symbol_shrinkage_weight=10.0)
    for index in range(30):
        _outcome(store, "005930", 60.0, recorded_at=NOW - timedelta(hours=index + 1))
        _outcome(store, "000660", -60.0, recorded_at=NOW - timedelta(hours=index + 1))

    bandit = ConservativeStrategyBandit(store=store, config=BanditConfig())
    candidates = (
        ArmCandidate(arm="intraday_momentum", symbol="000660", predicted_net_edge_bps=10.0),
        ArmCandidate(arm="intraday_momentum", symbol="005930", predicted_net_edge_bps=10.0),
    )
    selection = bandit.select(candidates, _context(), now=NOW)

    by_symbol = {item.symbol: item.conservative_edge_bps for item in selection.evaluations}
    assert by_symbol["005930"] > by_symbol["000660"]
    assert selection.selected_symbol == "005930"
