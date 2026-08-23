"""Election-path behaviour introduced with the conservative bandit.

Covers the three things the old first-admissible rule could not express:
NO_TRADE as a real outcome, a real within-sector rank, and a realized-outcome
feedback loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.graph.macro_reasoner import build_sector_rank_table
from app.schemas.domain import AccountSnapshot, Holding
from app.strategy.exit_geometry import exit_geometry
from app.trading.conservative_bandit import BanditConfig, ConservativeStrategyBandit
from app.trading.strategy_performance_store import PosteriorConfig, StrategyPerformanceStore
from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)


def _store(tmp_path) -> StrategyPerformanceStore:
    # The store's clock is pinned to ``NOW`` so the 21-day evidence window is measured
    # from the same instant the fixtures are dated from. While that window ran off the
    # real wall clock these tests expired silently on NOW + 21 days: the store read
    # back empty and the election saw no history to elect on.
    return StrategyPerformanceStore(
        tmp_path / "perf.sqlite3",
        posterior_config=PosteriorConfig(),
        cache_ttl_seconds=0.0,
        clock=lambda: NOW,
    )


def _manager(tmp_path, *, store=None, evidence=None, **config_overrides):
    store = store if store is not None else _store(tmp_path)
    values = {
        "state_path": str(tmp_path / "session.json"),
        "cooldown_seconds": 5,
        "entry_timeout_seconds": 30,
        "require_live_gnn": False,
        # This module exercises the legacy bandit deliberately. Production now
        # uses deterministic algorithm authority with the bandit disabled.
        "algorithm_primary_election": False,
        "bandit_enabled": True,
    }
    values.update(config_overrides)
    return StrategySessionManager(
        config=StrategySessionConfig(**values),
        selection_evidence_provider=(lambda symbols: evidence or {}),
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


# +160bps, not the old +50: the cost gate divides the SIGNAL's forecast by the all-in
# cost now, so a fixture has to state an edge that genuinely clears it.
def _bundle(*, symbol="005930", strategy="intraday_momentum", macro=None, exit_price=71_120.0):
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
        macro_result=macro if macro is not None else _macro(),
    )


def _account(*holdings):
    return AccountSnapshot(cash=200_000.0, holdings=tuple(holdings), total_equity_krw=200_000.0)


def test_measured_negative_expectancy_produces_no_trade(tmp_path):
    """The headline requirement of the whole change."""
    store = _store(tmp_path)
    for index in range(25):
        store.record(
            strategy_id="intraday_momentum",
            symbol="005930",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=-43.0,
            recorded_at=NOW - timedelta(minutes=index),
        )
    manager = _manager(tmp_path, store=store)
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "SCANNING"
    assert state["selected_strategy"] is None
    assert state["last_reason"].startswith("BANDIT_NO_TRADE")
    assert state["bandit_selected_arm"] == "no_trade"
    # The refusal is auditable, not a silent nothing-happened.
    assert state["bandit_evaluations"]
    assert state["bandit_evaluations"][0]["conservative_edge_bps"] < 0.0


def test_cold_arm_is_armed_as_a_flagged_exploration(tmp_path):
    manager = _manager(tmp_path)
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "intraday_momentum"
    assert state["cost_coverage_band"] in {"THIN", "SUFFICIENT", "INSUFFICIENT", "NOT_COVERED"}
    # With no realized history the lower bound is negative, so the ONLY reason this
    # armed is cold-start exploration — and that must be visible, otherwise a
    # negative edge on an ARMED position is indistinguishable from a bug.
    assert state["bandit_is_exploration"] is True
    assert state["bandit_conservative_edge_bps"] is not None
    assert state["bandit_conservative_edge_bps"] < 0.0
    assert "BANDIT_EXPLORATION_ARM_SELECTED" in state["bandit_reason_codes"]


def test_joint_election_compares_every_symbol_strategy_pair(tmp_path):
    """A strategy discarded by each symbol's old first pass may win globally."""
    store = _store(tmp_path)
    for index in range(30):
        store.record(
            strategy_id="intraday_momentum",
            symbol="005930",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=-35.0,
            recorded_at=NOW - timedelta(minutes=index),
        )
        store.record(
            strategy_id="breakout_volume",
            symbol="000660",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=75.0,
            recorded_at=NOW - timedelta(minutes=index),
        )

    def row(symbol, breakout_edge):
        return {
            "as_of": NOW.isoformat(),
            # The old per-symbol router selected momentum and discarded breakout.
            "decisions": [
                {
                    "path": "cpu_gnn",
                    "action": "ACTIVATE_STRATEGY",
                    "strategy_id": "intraday_momentum",
                    "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                    "expected_net_return_bps": 40.0,
                    "expected_cost_bps": 28.0,
                }
            ],
            "validation_candidates": [
                {
                    "path": "cpu_gnn_validation",
                    "action": "VALIDATE_ONLY",
                    "strategy_id": "intraday_momentum",
                    "utility": 1.0,
                    "probability_success": 0.7,
                    "expected_net_return_bps": 40.0,
                    "expected_cost_bps": 28.0,
                    "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                },
                {
                    "path": "cpu_gnn_validation",
                    "action": "VALIDATE_ONLY",
                    "strategy_id": "breakout_volume",
                    "utility": 0.8,
                    "probability_success": 0.65,
                    "expected_net_return_bps": breakout_edge,
                    "expected_cost_bps": 28.0,
                    "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                },
            ],
        }

    evidence = {
        "005930": row("005930", 10.0),
        "000660": row("000660", 55.0),
    }
    macro = _macro(allowed_micro_strategies=("momentum", "breakout"))
    manager = _manager(tmp_path, store=store, evidence=evidence)

    state = manager.evaluate(
        _account(),
        ("005930", "000660"),
        _bundle(macro=macro),
        NOW,
    )

    evaluated_pairs = {
        (item["symbol"], item["arm"])
        for item in state["bandit_evaluations"]
    }
    assert evaluated_pairs == {
        ("005930", "intraday_momentum"),
        ("005930", "breakout_volume"),
        ("000660", "intraday_momentum"),
        ("000660", "breakout_volume"),
    }
    assert state["selected_symbol"] == "000660"
    assert state["selected_strategy"] == "breakout_volume"
    assert state["selection_source"] == "GNN_JOINT_SYMBOL_STRATEGY_ELECTION"


def test_untrusted_full_vector_is_retained_as_non_executable_shadow_arm(tmp_path):
    evidence = {
        "005930": {
            "as_of": NOW.isoformat(),
            "mark_price": 70_000.0,
            "decisions": [],
            "validation_candidates": [
                {
                    "path": "cpu_gnn_validation",
                    "action": "VALIDATE_ONLY",
                    "strategy_id": "cross_sectional_relative_strength",
                    "utility": 0.7,
                    "probability_success": 0.61,
                    "expected_net_return_bps": 25.0,
                    "expected_cost_bps": 18.0,
                    "reason_codes": [
                        "GNN_REALTIME_TRUST_NOT_READY",
                        "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED",
                    ],
                }
            ],
        }
    }
    manager = _manager(tmp_path, evidence=evidence)
    manager._journal_shadow_proposals = lambda proposals, now, **_kwargs: None
    bundle = SimpleNamespace(
        ranked_trade_intents=(),
        buy_candidates=(),
        sell_reduce_candidates=(),
        blocked_candidates=(),
        micro_results=(),
        macro_result=_macro(allowed_micro_strategies=("relative_strength",)),
    )

    state = manager.evaluate(_account(), ("005930",), bundle, NOW)

    assert state["phase"] == "SCANNING"
    assert state["selected_strategy"] is None
    assert any(
        row["arm"].startswith("cross_sectional_relative_strength")
        for row in state["bandit_evaluations"]
    )
    assert "cross_sectional_relative_strength" in state["bandit_shadow_arms"]


def test_untrusted_vector_can_cold_probe_only_after_owned_algorithm_fires(tmp_path):
    evidence = {
        "005930": {
            "as_of": NOW.isoformat(),
            "mark_price": 70_000.0,
            "decisions": [],
            "technical_features": {
                "symbol": "005930",
                "price": 70_000.0,
                "second_data_ready": 1.0,
                "tick_count_5s": 9.0,
                "return_5s": 0.0008,
                "aggressor_imbalance_5s": 0.35,
                "realized_volatility_10s": 0.0025,
                "macd_histogram": 0.4,
                "ema_fast": 70_010.0,
                "ema_slow": 70_000.0,
            },
            "validation_candidates": [
                {
                    "path": "cpu_gnn_validation",
                    "action": "VALIDATE_ONLY",
                    "strategy_id": "intraday_momentum",
                    "expected_net_return_bps": 25.0,
                    "expected_cost_bps": 28.0,
                    "reason_codes": [
                        "GNN_REALTIME_TRUST_NOT_READY",
                        "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED",
                    ],
                }
            ],
        }
    }
    manager = _manager(
        tmp_path,
        evidence=evidence,
        algorithm_primary_election=True,
        bandit_enabled=False,
    )
    manager._journal_shadow_proposals = lambda proposals, now, **_kwargs: None
    bundle = SimpleNamespace(
        ranked_trade_intents=(), buy_candidates=(), sell_reduce_candidates=(),
        blocked_candidates=(), micro_results=(), macro_result=_macro(),
    )

    state = manager.evaluate(_account(), ("005930",), bundle, NOW)

    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == "intraday_momentum"
    # The validation vector may confirm that the owned algorithm fired, but it
    # cannot replace that deterministic proposal or charge a model-absence
    # penalty to its edge.
    assert state["selection_source"] == "ONTOLOGY_ALGORITHM_ELECTION"
    assert state["bandit_is_exploration"] is False
    assert state["bandit_evaluations"] == []


def test_string_macro_regime_is_preserved_for_posterior_context(tmp_path):
    manager = _manager(tmp_path)

    state = manager.evaluate(
        _account(),
        ("005930",),
        _bundle(macro=_macro(market_regime="HIGH_VOL_TRENDING")),
        NOW,
    )

    assert state["macro_regime"] == "HIGH_VOL_TRENDING"


def test_validation_vector_cannot_create_outcome_without_algorithm_trigger(tmp_path):
    evidence = {
        "005930": {
            "as_of": NOW.isoformat(),
            "mark_price": 70_000.0,
            "decisions": [],
            "technical_features": {
                "symbol": "005930",
                "price": 70_000.0,
                "second_data_ready": 0.0,
                "tick_count_5s": 0.0,
            },
            "validation_candidates": [
                {
                    "path": "cpu_gnn_validation",
                    "action": "VALIDATE_ONLY",
                    "strategy_id": "intraday_momentum",
                    "expected_net_return_bps": 100.0,
                    "expected_cost_bps": 28.0,
                    "reason_codes": ["GNN_REALTIME_TRUST_NOT_READY"],
                }
            ],
        }
    }
    manager = _manager(tmp_path, evidence=evidence)
    manager._journal_shadow_proposals = lambda proposals, now, **_kwargs: None
    bundle = SimpleNamespace(
        ranked_trade_intents=(), buy_candidates=(), sell_reduce_candidates=(),
        blocked_candidates=(), micro_results=(), macro_result=_macro(),
    )

    state = manager.evaluate(_account(), ("005930",), bundle, NOW)

    assert state["bandit_evaluations"] == []
    assert state["algorithm_evaluations"][0]["triggered"] is False
    assert "TICK_WINDOW_NOT_READY" in state["algorithm_evaluations"][0]["reason_codes"]


def test_bar_confirmed_vwap_recovery_is_a_regular_gnn_bandit_arm(tmp_path):
    strategy_id = "bar_confirmed_vwap_recovery"
    evidence = {
        "005930": {
            "as_of": NOW.isoformat(),
            "mark_price": 98.50,
            "decisions": [],
            "technical_features": {
                "symbol": "005930",
                "price": 98.50,
                "vwap": 100.0,
                "vwap_distance_bps": -150.0,
                "realized_volatility": 0.0015,
                # Prove the strategy is independent of the sparse tick blockers.
                "second_data_ready": 0.0,
                "tick_count_5s": 0.0,
                "ema_fast": 98.40,
                "macd_histogram": 0.08,
                "rsi": 35.0,
                "momentum_persistence": 0.60,
                "liquidity_score": 0.80,
                "spread_bps": 10.0,
            },
            "validation_candidates": [
                {
                    "path": "cpu_gnn_validation",
                    "action": "VALIDATE_ONLY",
                    "strategy_id": strategy_id,
                    "utility": 0.8,
                    "probability_success": 0.65,
                    "expected_net_return_bps": 20.0,
                    "expected_cost_bps": 18.0,
                    "reason_codes": ["GNN_REALTIME_TRUST_NOT_READY"],
                }
            ],
        }
    }
    manager = _manager(tmp_path, evidence=evidence)
    manager._journal_shadow_proposals = lambda proposals, now, **_kwargs: None
    bundle = SimpleNamespace(
        ranked_trade_intents=(), buy_candidates=(), sell_reduce_candidates=(),
        blocked_candidates=(), micro_results=(),
        macro_result=_macro(
            allowed_micro_strategies=("vwap_reversion", "mean_reversion")
        ),
    )

    state = manager.evaluate(_account(), ("005930",), bundle, NOW)

    assert state["phase"] == "ARMED"
    assert state["selected_strategy"] == strategy_id
    assert state["selected_deployment_state"] == "LIVE_FULL"
    assert any(
        row["arm"].startswith(strategy_id) for row in state["bandit_evaluations"]
    )
    assert strategy_id not in state["bandit_shadow_arms"]


def test_demonstrated_edge_is_armed_as_exploitation_not_exploration(tmp_path):
    store = _store(tmp_path)
    for index in range(40):
        store.record(
            strategy_id="intraday_momentum",
            symbol="005930",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=70.0 + (index % 4),
            recorded_at=NOW - timedelta(minutes=index),
        )
    manager = _manager(tmp_path, store=store)
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["bandit_is_exploration"] is False
    assert state["bandit_conservative_edge_bps"] > 0.0


def test_election_context_carries_a_real_within_sector_rank(tmp_path):
    macro = _macro(
        sector_rank_table=build_sector_rank_table(
            sector_of={"005930": "semi", "000660": "semi"},
            residual_returns={"005930": 0.004, "000660": 0.001},
            long_residual_returns={"005930": 0.006, "000660": 0.002},
            market_betas={"005930": 0.9},
        )
    )
    manager = _manager(tmp_path)
    state = manager.evaluate(_account(), ("005930",), _bundle(macro=macro), NOW)
    context = state["election_context"]
    assert context["sector_rank"] == 1
    assert context["sector_candidate_count"] == 2
    assert context["sector"] == "semi"
    assert context["residual_return_short_bps"] == 40.0
    assert context["residual_return_long_bps"] == 60.0
    assert context["market_beta"] == 0.9


def test_election_context_omits_the_rank_when_the_sector_is_unknown(tmp_path):
    """An unanswerable rank must be absent, not faked from a global ordering."""
    manager = _manager(tmp_path)
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    context = state["election_context"]
    assert "sector_rank" not in context
    assert "sector_candidate_count" not in context


def test_election_context_carries_measured_live_session_diagnostics(tmp_path):
    micro_result = SimpleNamespace(
        symbol="005930",
        diagnostics={
            "opening_range_high": 71_000.0,
            "opening_range_low": 69_500.0,
            "opening_range_minutes": 30,
            "first_half_hour_return_bps": 85.0,
            "first_half_hour_volatility_percentile": .8,
        },
    )
    context = _manager(tmp_path)._election_context(  # noqa: SLF001
        "opening_range_breakout", NOW,
        micro_result=micro_result, symbol="005930", macro=_macro(),
    )
    assert context["opening_range_high"] == 71_000.0
    assert context["opening_range_low"] == 69_500.0
    assert context["first_half_hour_return_bps"] == 85.0
    assert context["first_half_hour_volatility_percentile"] == .8


def test_change_point_stand_down_prevents_election(tmp_path):
    manager = _manager(tmp_path)
    macro = _macro(change_point_probability=0.85)
    state = manager.evaluate(_account(), ("005930",), _bundle(macro=macro), NOW)
    assert state["phase"] == "SCANNING"
    assert state["change_point_probability"] == 0.85
    assert "BANDIT_CHANGE_POINT_STAND_DOWN" in state["bandit_reason_codes"]


def test_closed_position_records_a_realized_outcome(tmp_path):
    store = _store(tmp_path)
    manager = _manager(tmp_path, store=store)
    armed = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert armed["phase"] == "ARMED"
    manager.mark_entry_submitted("005930", NOW)
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=1,
        average_price=70_000.0,
        last_price=70_100.0,
        opened_at=NOW,
    )
    manager.evaluate(_account(holding), (), _bundle(), NOW + timedelta(seconds=1))
    # Derived from the live geometry rather than hardcoded: a fixed +1.43% used to
    # clear a 100bps target and silently stopped clearing it when the target moved.
    target_bps = exit_geometry("intraday_momentum").take_profit_bps
    exit_price = 70_000.0 * (1.0 + (target_bps + 25.0) / 10_000.0)
    target = Holding(**{**holding.__dict__, "last_price": exit_price})
    exiting = manager.evaluate(_account(target), (), _bundle(), NOW + timedelta(seconds=2))
    assert exiting["phase"] == "EXITING"
    # The broker confirming the exit FILL is what closes the position. A balance
    # response that merely stopped listing the lot is not evidence of anything.
    manager.mark_exit_filled("005930", exit_price, NOW + timedelta(seconds=3))
    flat = manager.evaluate(_account(), (), _bundle(), NOW + timedelta(seconds=3))
    assert flat["phase"] == "COOLDOWN"

    outcomes = store.recent_outcomes("intraday_momentum", market="KR")
    assert len(outcomes) == 1
    outcome = outcomes[0]
    # Gross must clear the target it was triggered by, and the round trip on top.
    assert outcome.realized_gross_bps > target_bps
    # Charged at the venue's own fee policy, not at the configured KR reference
    # constant. With no measured spread in this bundle the all-in cost IS the policy
    # round trip; hardcoding 28.0 here is what let the live gate divide an edge by
    # fees alone and call it sufficient.
    from app.trading.strategy_session import _market_round_trip_cost_bps

    policy_cost = _market_round_trip_cost_bps("005930", 28.0)
    assert policy_cost >= 28.0
    assert outcome.realized_net_bps == outcome.realized_gross_bps - policy_cost
    assert outcome.regime == "HIGH_VOL_TRENDING"
    assert outcome.exit_reason == "STRATEGY_PROFIT_TARGET"


def test_outcome_is_recorded_once_only(tmp_path):
    store = _store(tmp_path)
    manager = _manager(tmp_path, store=store)
    manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    manager.mark_entry_submitted("005930", NOW)
    holding = Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Technology",
        quantity=1,
        average_price=70_000.0,
        last_price=71_000.0,
        opened_at=NOW,
    )
    manager.evaluate(_account(holding), (), _bundle(), NOW + timedelta(seconds=1))
    # The close has to be CONFIRMED before anything is recorded — a single absent
    # balance response is a partial broker reply, not an exit. Once it is confirmed,
    # every further flat observation must still leave exactly one outcome.
    for offset in (200, 201, 202, 203):
        manager.evaluate(_account(), (), _bundle(), NOW + timedelta(seconds=offset))
    assert len(store.recent_outcomes("intraday_momentum", market="KR")) == 1


def test_bandit_can_be_disabled_to_restore_first_admissible_election(tmp_path):
    # 15 samples, not 25: past LongPromotionConfig.minimum_shadow_samples (20) a
    # non-positive conservative edge demotes the arm out of live authority, and this
    # test would then be measuring the deployment ladder rather than the election
    # posture. The losing history is only here to prove the bandit is not what armed
    # the proposal.
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
    manager = _manager(tmp_path, store=store, bandit_enabled=False)
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["bandit_selected_arm"] is None


def test_new_entry_is_refused_outside_the_regular_session(tmp_path):
    """The measured production failure: 11,614 cycles, 0 buys, misleading reason.

    Outside the regular session every candidate failed individually on thin-book
    liquidity, so the surviving reason blamed the GNN (NO_POSITIVE_NET_GNN_EDGE)
    while the actual constraint was that no scanned market was open for trading.
    """
    manager = _manager(tmp_path)
    # 23:45 UTC — KRX pre-market (08:45 KST), 미국은 19:45 EDT.
    #
    # 이 시각의 미국 위상은 "after" 가 아니라 "closed" 다. KIS 공식 애프터마켓 주문 시간은
    # 06:00~07:00 KST (Summer Time 05:00~07:00) = ET 16:00-17:00/18:00 이므로 19:45 ET 는
    # 완전 마감이다. 이전 구현이 애프터마켓을 20:00 ET 까지로 잡고 있었고, 그 구간의
    # 주문은 브로커에서 거부됐다.
    # 근거: docs/kis_market_session_capability_matrix.md §5.1
    after_hours = datetime(2026, 7, 30, 23, 45, tzinfo=timezone.utc)
    state = manager.evaluate(
        _account(), ("F", "BAC"), _bundle(symbol="F"), after_hours
    )
    assert state["phase"] == "SCANNING"
    assert state["last_reason"].startswith("NEW_ENTRY_OUTSIDE_REGULAR_SESSION")
    assert state["session_phases"]["US"] == "closed"
    # The reason names the phase, so no clock correlation is needed.
    assert "US=closed" in state["last_reason"]


def test_new_entry_is_refused_during_a_real_us_after_hours_session(tmp_path):
    """실제 애프터마켓(17:00 EDT) 에서도 신규 진입은 기본적으로 거부된다."""
    manager = _manager(tmp_path)
    # 21:00 UTC = 17:00 EDT — 공식 애프터마켓 창 안 (16:00-18:00 EDT).
    after_hours = datetime(2026, 7, 30, 21, 0, tzinfo=timezone.utc)
    state = manager.evaluate(
        _account(), ("F", "BAC"), _bundle(symbol="F"), after_hours
    )
    assert state["last_reason"].startswith("NEW_ENTRY_OUTSIDE_REGULAR_SESSION")
    assert state["session_phases"]["US"] == "after"


def test_regular_session_candidates_still_elect(tmp_path):
    manager = _manager(tmp_path)
    # 01:00 UTC — 10:00 KST, KRX regular session, KR candidate.
    state = manager.evaluate(_account(), ("005930",), _bundle(), NOW)
    assert state["phase"] == "ARMED"
    assert state["session_phases"]["KRX"] == "regular"


def test_extended_hours_entry_can_be_re_enabled(tmp_path, monkeypatch):
    """전역 플래그는 backward-compatible alias 로 세션 게이트를 완화한다.

    시각을 실제 애프터마켓 창(17:00 EDT) 으로 잡는다. 이전에는 19:45 EDT 를 썼는데,
    공식 애프터마켓은 ET 18:00 에 끝나므로 그 시각은 완전 마감이고 "장외 진입 허용"이
    적용될 여지가 없다 — 닫힌 시장을 여는 플래그는 존재하지 않아야 한다.

    세션 게이트가 열려도 실주문은 세션별 ``live_order_authorized`` 가 추가로 필요하다
    (라우터가 강제). 이 테스트는 세션 게이트만 검증한다.
    """
    monkeypatch.setenv("TRADING_ALLOW_EXTENDED_HOURS_ENTRY", "true")
    manager = _manager(tmp_path)
    after_hours = datetime(2026, 7, 30, 21, 0, tzinfo=timezone.utc)  # 17:00 EDT
    state = manager.evaluate(_account(), ("F",), _bundle(symbol="F"), after_hours)
    assert not state["last_reason"].startswith("NEW_ENTRY_OUTSIDE_REGULAR_SESSION")


def test_extended_hours_flag_cannot_open_a_fully_closed_market(tmp_path, monkeypatch):
    """장외 진입 플래그로 완전 마감된 시장을 열 수는 없다."""
    monkeypatch.setenv("TRADING_ALLOW_EXTENDED_HOURS_ENTRY", "true")
    manager = _manager(tmp_path)
    closed = datetime(2026, 7, 30, 23, 45, tzinfo=timezone.utc)  # 19:45 EDT
    state = manager.evaluate(_account(), ("F",), _bundle(symbol="F"), closed)
    assert state["last_reason"].startswith("NEW_ENTRY_OUTSIDE_REGULAR_SESSION")
    assert state["session_phases"]["US"] == "closed"


def test_session_gate_is_scoped_to_the_scanned_market(tmp_path):
    """A universe of US names during the KR session is not "market open"."""
    manager = _manager(tmp_path)
    # 01:00 UTC: KRX regular, but US is fully closed and the universe is US-only.
    state = manager.evaluate(_account(), ("F",), _bundle(symbol="F"), NOW)
    assert state["last_reason"].startswith("NEW_ENTRY_OUTSIDE_REGULAR_SESSION")
    assert state["session_phases"] == {"US": "closed"}


def test_bandit_diagnostics_do_not_survive_into_a_cycle_that_did_not_run_them(tmp_path):
    """bandit 진단값은 계산된 사이클 안에서만 유효해야 한다.

    실제로 오진을 유발한 결함: ``proposals`` 가 비면 ``_bandit_choice`` 가 호출되지
    않는데 ``bandit_*`` 필드는 초기화되지 않았다. 그래서 국내장 시간에 계산된
    "rvgi_box_breakout / 069500 / conservative_edge -109bps" 가 미국 정규장 진단으로
    그대로 노출됐고, "미국 정규장인데 왜 마감된 국내 종목 arm 을 평가하나"라는 존재하지
    않는 결함을 추적하게 됐다.
    """
    # 측정된 음의 기대값 → bandit 이 no_trade 를 고르고 SCANNING 을 유지한다.
    # (ARMED 가 되면 그 진단값은 무장 결정의 감사 기록이므로 지우면 안 된다.)
    store = _store(tmp_path)
    krx_regular = datetime(2026, 8, 5, 1, 0, tzinfo=timezone.utc)  # 10:00 KST
    for index in range(25):
        store.record(
            strategy_id="intraday_momentum",
            symbol="005930",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=-43.0,
            recorded_at=krx_regular - timedelta(minutes=index),
        )
    manager = _manager(tmp_path, store=store)

    # 1) bandit 이 실제로 도는 사이클.
    first = manager.evaluate(_account(), ("005930",), _bundle(), krx_regular)
    assert first["phase"] == "SCANNING"
    assert first["bandit_evaluated_at"] == "2026-08-05T01:00:00+00:00"
    assert first["bandit_evaluations"]
    assert first["bandit_selected_arm"] == "no_trade"

    # 2) 후보가 없어 제안이 만들어지지 않는 사이클 → 진단값이 비어 있어야 한다.
    #    이 초기화가 없으면 위의 KR 평가가 이후 모든 사이클(다른 시장 세션 포함)의
    #    현재 진단으로 계속 보고된다.
    second = manager.evaluate(
        _account(), (), _bundle(), krx_regular + timedelta(seconds=1)
    )
    assert second["bandit_evaluated_at"] is None
    assert second["bandit_evaluations"] == []
    assert second["bandit_selected_arm"] is None
    assert second["bandit_reason_codes"] == []
    assert second["directional_comparison"] == {}


def test_closed_market_candidates_are_dropped_before_proposals(tmp_path):
    """자기 시장이 마감된 후보는 제안 대상에서 제외된다.

    이전에는 세션 게이트가 "후보 그룹 중 하나라도 열려 있으면" 통과시키고, 두 제안
    경로가 후보 **전체** 를 순회했다. KR+US 혼합 유니버스에서 미국 정규장 시간이면
    마감된 국내 종목까지 제안 대상이 됐다.
    """
    manager = _manager(tmp_path)
    # 14:00 UTC = 10:00 ET (미국 정규장), 23:00 KST (국내 마감).
    us_regular = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)
    state = manager.evaluate(
        _account(), ("005930", "AAPL"), _bundle(symbol="005930"), us_regular
    )
    # 국내 종목에 대한 의도였으므로 제안이 만들어지지 않고, 국내 종목이 선택되지 않는다.
    assert state["selected_symbol"] != "005930"
    assert state["phase"] == "SCANNING"
    # 세션 위상 보고에는 두 시장이 모두 보인다 (관측은 되어야 한다).
    assert set(state["session_phases"]) == {"KRX", "US"}
    assert state["session_phases"]["US"] == "regular"
    assert state["session_phases"]["KRX"] == "closed"
