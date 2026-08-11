"""Counterfactual shadow positions, evidence separation, and selector regret."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation.counterfactual_engine import CounterfactualEngine
from app.evaluation.outcome_resolver import (
    EvidenceWeights,
    OutcomeResolver,
    PROMOTABLE_SOURCES,
)
from app.evaluation.selector_evaluator import SelectorEvaluator, StrategyVerdict
from app.evaluation.selector_regret import (
    NO_TRADE_OUTCOME_BPS,
    compute_context_regret,
    summarize_regret,
)
from app.evaluation.shadow_position import (
    EVIDENCE_LIVE,
    EVIDENCE_SHADOW,
    ShadowExitReason,
    ShadowPosition,
)

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


def _position(**overrides) -> ShadowPosition:
    values = {
        "position_id": "p1",
        "context_id": "ctx-1",
        "strategy_id": "intraday_momentum",
        "symbol": "005930",
        "market": "KR",
        "direction": "LONG",
        "entry_price": 100.0,
        "target_price": 101.6,
        "stop_price": 99.4,
        "trailing_bps": 30.0,
        "max_holding_seconds": 600,
        "cost_bps": 28.0,
        "opened_at": AT,
    }
    values.update(overrides)
    return ShadowPosition(**values)


# --------------------------------------------------------------------------- #
# Shadow position mechanics                                                    #
# --------------------------------------------------------------------------- #
def test_target_resolves_with_direction_signed_return() -> None:
    position = _position()
    assert position.update(100.4, AT + timedelta(seconds=10)) is None
    outcome = position.update(101.7, AT + timedelta(seconds=20))
    assert outcome is not None
    assert outcome.exit_reason == ShadowExitReason.TARGET
    assert outcome.gross_return_bps == pytest.approx(160.0, abs=0.1)
    assert outcome.net_return_bps == pytest.approx(132.0, abs=0.1)
    assert outcome.evidence_source == EVIDENCE_SHADOW
    assert outcome.fill_assumed is True


def test_stop_wins_when_one_quote_breaches_both_barriers() -> None:
    """Conservative reading: a gap through both most likely traded the near one first."""
    position = _position(target_price=100.5, stop_price=99.5)
    outcome = position.update(101.0, AT + timedelta(seconds=5))
    assert outcome is not None
    # The quote is above target AND (trivially) not below stop, so target is correct here.
    assert outcome.exit_reason == ShadowExitReason.TARGET

    gapped = _position(target_price=101.0, stop_price=100.5)
    breached = gapped.update(100.0, AT + timedelta(seconds=5))
    assert breached is not None
    assert breached.exit_reason == ShadowExitReason.STOP


def test_short_direction_sign_is_applied_once() -> None:
    position = _position(
        direction="SHORT", entry_price=100.0, target_price=98.4, stop_price=100.6
    )
    outcome = position.update(98.3, AT + timedelta(seconds=10))
    assert outcome is not None
    assert outcome.gross_return_bps > 0, "a short covering lower made money"


def test_trailing_stop_arms_only_after_a_favourable_move() -> None:
    position = _position(trailing_bps=50.0, target_price=200.0, stop_price=1.0)
    # Straight down from entry: the trailing stop must not fire from the entry price.
    assert position.update(99.8, AT + timedelta(seconds=5)) is None
    # Up, then back through the trail.
    assert position.update(101.0, AT + timedelta(seconds=10)) is None
    outcome = position.update(100.3, AT + timedelta(seconds=15))
    assert outcome is not None
    assert outcome.exit_reason == ShadowExitReason.TRAILING


def test_quote_before_the_signal_cannot_resolve_a_position() -> None:
    position = _position()
    assert position.update(99.0, AT - timedelta(seconds=30)) is None
    assert not position.resolved


def test_never_observed_position_reports_no_outcome_rather_than_break_even() -> None:
    position = _position()
    assert position.expire(AT + timedelta(seconds=900)) is None
    marker = position.mark_unobserved(AT + timedelta(seconds=900))
    assert marker.quotes_observed == 0
    assert marker.exit_reason == ShadowExitReason.UNRESOLVED


# --------------------------------------------------------------------------- #
# The engine                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Proposal:
    strategy_id: str
    direction: str = "LONG"
    eligible: bool = True
    entry_ready: bool = True
    reference_entry_price: float | None = 100.0
    target_price: float | None = 101.6
    stop_price: float | None = 99.4
    expected_horizon_seconds: int = 600
    proposal_id: str = "prop"


@dataclass(frozen=True)
class _Candidate:
    strategy_id: str
    expected_cost_bps: float = 28.0
    final_utility_bps: float = 10.0
    expected_net_return_bps: float = 40.0


@dataclass(frozen=True)
class _Selection:
    selected_strategy: str | None
    decision: str
    ranked_candidates: tuple


@dataclass(frozen=True)
class _Macro:
    market_regime: str = "TREND_UP"


@dataclass(frozen=True)
class _Context:
    context_id: str = "ctx-1"
    symbol_id: str = "005930"
    market: str = "KR"
    captured_at: datetime = AT
    macro: _Macro = _Macro()


def _engine_with_group(selected: str | None = "intraday_momentum"):
    calls: list = []
    engine = CounterfactualEngine(outcome_sink=calls.append)
    strategies = ("intraday_momentum", "vwap_mean_reversion", "breakout_volume")
    engine.open_from_selection(
        context=_Context(),
        selection=_Selection(
            selected_strategy=selected,
            decision="SELECT" if selected else "NO_TRADE",
            ranked_candidates=tuple(_Candidate(name) for name in strategies),
        ),
        proposals=tuple(
            _Proposal(
                strategy_id=name,
                target_price=101.6 if name != "breakout_volume" else 105.0,
                stop_price=99.4,
            )
            for name in strategies
        ),
        trailing_bps_by_strategy={name: 30.0 for name in strategies},
    )
    return engine, calls


def test_engine_opens_a_position_for_every_entry_ready_alternative() -> None:
    engine, _ = _engine_with_group()
    group = engine.group("ctx-1")
    assert group is not None
    assert set(group.positions) == {
        "intraday_momentum",
        "vwap_mean_reversion",
        "breakout_volume",
    }


def test_selected_strategy_outcome_is_not_emitted_to_the_sink() -> None:
    """Its evidence is the real fill; emitting the simulation would double count it."""
    engine, emitted = _engine_with_group(selected="intraday_momentum")
    engine.observe_quote("005930", 101.7, AT + timedelta(seconds=20))
    emitted_ids = {item.strategy_id for item in emitted}
    assert "intraday_momentum" not in emitted_ids
    assert "vwap_mean_reversion" in emitted_ids


def test_engine_has_no_broker_path() -> None:
    engine, _ = _engine_with_group()
    assert not hasattr(engine, "submit")
    assert not hasattr(engine, "coordinator")
    assert not hasattr(engine, "broker")


def test_proposal_without_entry_reference_is_rejected() -> None:
    engine = CounterfactualEngine()
    group = engine.open_from_selection(
        context=_Context(context_id="ctx-2"),
        selection=_Selection(
            selected_strategy=None, decision="NO_TRADE", ranked_candidates=()
        ),
        proposals=(_Proposal(strategy_id="intraday_momentum", reference_entry_price=None),),
    )
    assert group is None
    assert engine.stats().positions_rejected == 1


# --------------------------------------------------------------------------- #
# Evidence separation                                                          #
# --------------------------------------------------------------------------- #
def test_shadow_evidence_cannot_count_as_live() -> None:
    resolver = OutcomeResolver(weights=EvidenceWeights())
    shadow = resolver.resolve(
        {
            "strategy_id": "intraday_momentum",
            "net_return_bps": 50.0,
            "evidence_source": EVIDENCE_SHADOW,
            "quotes_observed": 3,
        }
    )
    live = resolver.resolve(
        {
            "strategy_id": "intraday_momentum",
            "net_return_bps": -20.0,
            "evidence_source": EVIDENCE_LIVE,
        }
    )
    assert shadow is not None and live is not None
    assert not shadow.is_live
    assert live.is_live
    assert shadow.weight < live.weight
    assert EVIDENCE_SHADOW not in PROMOTABLE_SOURCES


def test_aggregate_reports_live_only_separately() -> None:
    resolver = OutcomeResolver()
    rows = [
        {"strategy_id": "s", "net_return_bps": 100.0, "evidence_source": EVIDENCE_SHADOW,
         "quotes_observed": 2},
        {"strategy_id": "s", "net_return_bps": -50.0, "evidence_source": EVIDENCE_LIVE},
    ]
    aggregate = resolver.aggregate_by_strategy(rows)["s"]
    assert aggregate.live_only_net_bps == pytest.approx(-50.0)
    assert aggregate.weighted_net_bps != aggregate.live_only_net_bps
    assert aggregate.mix.promotable


def test_unobserved_shadow_row_is_dropped() -> None:
    resolver = OutcomeResolver()
    assert (
        resolver.resolve(
            {
                "strategy_id": "s",
                "net_return_bps": 0.0,
                "evidence_source": EVIDENCE_SHADOW,
                "quotes_observed": 0,
            }
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Regret                                                                       #
# --------------------------------------------------------------------------- #
def test_regret_is_zero_when_the_selector_picked_the_best() -> None:
    engine, _ = _engine_with_group(selected="breakout_volume")
    engine.observe_quote("005930", 105.5, AT + timedelta(seconds=30))
    group = engine.group("ctx-1")
    engine.record_live_outcome(
        context_id="ctx-1",
        strategy_id="breakout_volume",
        net_return_bps=472.0,
        evidence_source=EVIDENCE_LIVE,
    )
    regret = compute_context_regret(group)
    assert regret is not None
    assert regret.regret_bps == pytest.approx(0.0, abs=1e-6)
    assert regret.top1_hit
    assert regret.selected_from_live


def test_no_trade_competes_and_declining_can_be_correct() -> None:
    engine, _ = _engine_with_group(selected=None)
    # Everything stops out.
    engine.observe_quote("005930", 99.0, AT + timedelta(seconds=20))
    group = engine.group("ctx-1")
    regret = compute_context_regret(group)
    assert regret is not None
    assert regret.selected_strategy is None
    assert regret.selected_outcome_bps == NO_TRADE_OUTCOME_BPS
    assert regret.best_strategy is None, "nothing beat doing nothing"
    assert regret.regret_bps == 0.0
    assert regret.top1_hit


def test_summary_reports_no_trade_precision_and_missed_opportunity() -> None:
    engine, _ = _engine_with_group(selected=None)
    engine.observe_quote("005930", 101.7, AT + timedelta(seconds=20))
    good = engine.group("ctx-1")

    engine2, _ = _engine_with_group(selected=None)
    engine2.observe_quote("005930", 99.0, AT + timedelta(seconds=20))
    bad = engine2.group("ctx-1")

    regrets = [item for item in (compute_context_regret(good), compute_context_regret(bad)) if item]
    summary = summarize_regret(regrets, groups=(good, bad), minimum_contexts=2)
    assert summary.context_count == 2
    assert summary.no_trade_precision == pytest.approx(0.5)
    assert summary.missed_opportunity_bps is not None
    assert summary.oracle_best_strategy_net_ev_bps >= summary.selected_strategy_net_ev_bps


def test_evaluator_separates_strategy_from_selector_problems() -> None:
    """A losing thesis is a strategy problem; a good thesis picked badly is a selector one."""
    evaluator = SelectorEvaluator()

    groups = []
    for index in range(20):
        engine, _ = _engine_with_group(selected="vwap_mean_reversion")
        # breakout_volume's wide target never fills; the others stop out. So the SELECTED
        # strategy loses, and so does every alternative -> a strategy problem, not a
        # selector one.
        engine.observe_quote("005930", 99.0, AT + timedelta(seconds=20 + index))
        engine.expire_stale(AT + timedelta(seconds=1200 + index))
        group = engine.group("ctx-1")
        object.__setattr__(group, "context_id", f"ctx-{index}")
        groups.append(group)

    evaluation = evaluator.evaluate(groups)
    verdicts = {item.strategy_id: item.verdict for item in evaluation.diagnoses}
    assert verdicts, "expected per-strategy diagnoses"
    for verdict in verdicts.values():
        assert verdict is not StrategyVerdict.SELECTOR_PROBLEM
