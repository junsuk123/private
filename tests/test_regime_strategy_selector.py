"""Regime-driven strategy family selection, and the conditions that force WAIT."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.context.regime import RegimeEstimate, RegimeEstimator, RegimeEvidence
from app.models.temporal_hetero_gnn import REGIME_LABELS, STRATEGY_FAMILIES
from app.ontology.market_graph import load_market_graph
from app.routing.regime_strategy_selector import (
    FAMILY_STRATEGY_IDS,
    WAIT,
    RegimeStrategySelector,
    SelectorPolicy,
)

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)


def _regime(**probabilities) -> RegimeEstimate:
    values = {label: 0.0 for label in REGIME_LABELS}
    values.update(probabilities)
    return RegimeEstimate(
        probabilities=values,
        confidence=1.0,
        evaluated_at=NOW,
        source="test",
    )


@pytest.fixture()
def selector() -> RegimeStrategySelector:
    return RegimeStrategySelector()


# --------------------------------------------------------------------------- #
# Family mapping
# --------------------------------------------------------------------------- #
def test_every_ontology_family_maps_to_catalogue_ids() -> None:
    assert set(FAMILY_STRATEGY_IDS) == set(STRATEGY_FAMILIES)
    # DEFENSIVE is the stance of not opening risk; it has no entry strategy by design.
    assert FAMILY_STRATEGY_IDS["DEFENSIVE"] == ()
    for family in set(STRATEGY_FAMILIES) - {"DEFENSIVE"}:
        assert FAMILY_STRATEGY_IDS[family], family


def test_no_strategy_id_is_claimed_by_two_families() -> None:
    seen: dict[str, str] = {}
    for family, ids in FAMILY_STRATEGY_IDS.items():
        for strategy_id in ids:
            assert strategy_id not in seen, f"{strategy_id} in {family} and {seen[strategy_id]}"
            seen[strategy_id] = family


# --------------------------------------------------------------------------- #
# Regime -> family
# --------------------------------------------------------------------------- #
def test_trend_regime_prefers_trend_over_mean_reversion(selector) -> None:
    selection = selector.select(
        ticker="005930", regime=_regime(TREND_UP=0.9, RISK_ON=0.7), decided_at=NOW
    )
    scores = {item.family: item.score for item in selection.ranked}
    assert scores["TREND"] > scores["MEAN_REVERSION"]


def test_quiet_range_prefers_mean_reversion(selector) -> None:
    selection = selector.select(
        ticker="005930", regime=_regime(RANGE_LOW_VOL=0.9), decided_at=NOW
    )
    scores = {item.family: item.score for item in selection.ranked}
    assert scores["MEAN_REVERSION"] > scores["TREND"]
    assert scores["MEAN_REVERSION"] > scores["BREAKOUT"]


def test_liquidity_stress_prefers_defensive_and_suppresses_order_flow(selector) -> None:
    selection = selector.select(
        ticker="005930", regime=_regime(LIQUIDITY_STRESS=0.95), decided_at=NOW
    )
    scores = {item.family: item.score for item in selection.ranked}
    assert scores["DEFENSIVE"] > scores["ORDER_FLOW"]
    assert scores["ORDER_FLOW"] < 0.0
    assert selection.action == WAIT


def test_defensive_winning_is_expressed_as_wait(selector) -> None:
    selection = selector.select(
        ticker="005930", regime=_regime(RISK_OFF=0.95, BREAKDOWN=0.7), decided_at=NOW
    )
    assert selection.action == WAIT
    assert "WAIT_DEFENSIVE_REGIME" in selection.reasons
    assert selection.strategy_ids() == ()


# --------------------------------------------------------------------------- #
# Hard exclusions
# --------------------------------------------------------------------------- #
def test_stale_data_invalidates_every_entry_family(selector) -> None:
    selection = selector.select(
        ticker="005930",
        regime=_regime(TREND_UP=0.9),
        decided_at=NOW,
        active_risk_conditions={"STALE_DATA": 1.0},
    )
    excluded = {item.family for item in selection.ranked if item.excluded}
    assert excluded == set(STRATEGY_FAMILIES) - {"DEFENSIVE"}
    assert selection.action == WAIT
    assert "WAIT_ALL_FAMILIES_INVALIDATED" in selection.reasons
    assert "INVALIDATED_BY:STALE_DATA" in selection.reasons


def test_an_inactive_risk_condition_does_not_exclude(selector) -> None:
    selection = selector.select(
        ticker="005930",
        regime=_regime(TREND_UP=0.9),
        decided_at=NOW,
        active_risk_conditions={"STALE_DATA": 0.0},
    )
    assert not any(item.excluded for item in selection.ranked)


# --------------------------------------------------------------------------- #
# WAIT conditions
# --------------------------------------------------------------------------- #
def test_low_regime_confidence_waits(selector) -> None:
    unsure = RegimeEstimate(
        probabilities={label: 0.5 for label in REGIME_LABELS},
        confidence=0.05,
        evaluated_at=NOW,
        source="test",
    )
    selection = selector.select(ticker="005930", regime=unsure, decided_at=NOW)
    assert selection.action == WAIT
    assert "WAIT_LOW_REGIME_CONFIDENCE" in selection.reasons


def test_nothing_convincing_waits(selector) -> None:
    selection = selector.select(
        ticker="005930", regime=_regime(TRANSITION=0.05), decided_at=NOW
    )
    assert selection.action == WAIT
    assert "WAIT_BELOW_MINIMUM_SCORE" in selection.reasons


def test_an_opposed_photo_finish_waits() -> None:
    tight = RegimeStrategySelector(policy=SelectorPolicy(ambiguity_margin=5.0))
    selection = tight.select(
        ticker="005930", regime=_regime(TREND_UP=0.6, RANGE_LOW_VOL=0.6), decided_at=NOW
    )
    assert selection.action == WAIT
    assert any(reason.startswith("WAIT_AMBIGUOUS") for reason in selection.reasons)


# --------------------------------------------------------------------------- #
# Evidence sources
# --------------------------------------------------------------------------- #
def test_model_evidence_is_ignored_when_the_model_is_not_healthy(selector) -> None:
    regime = _regime(TREND_UP=0.9)
    suitability = {family: 0.99 for family in STRATEGY_FAMILIES}
    unhealthy = selector.select(
        ticker="005930",
        regime=regime,
        decided_at=NOW,
        model_suitability=suitability,
        model_healthy=False,
    )
    healthy = selector.select(
        ticker="005930",
        regime=regime,
        decided_at=NOW,
        model_suitability=suitability,
        model_healthy=True,
    )
    assert not unhealthy.model_used
    assert healthy.model_used
    assert all(item.model_term == 0.0 for item in unhealthy.ranked)
    assert any(item.model_term > 0.0 for item in healthy.ranked)


def test_micro_confirmation_can_contradict_the_regime(selector) -> None:
    regime = _regime(TREND_UP=0.9)
    agreeing = selector.select(
        ticker="005930",
        regime=regime,
        decided_at=NOW,
        micro=selector.micro_confirmation_from_context(trend_strength=40.0),
    )
    disagreeing = selector.select(
        ticker="005930",
        regime=regime,
        decided_at=NOW,
        micro=selector.micro_confirmation_from_context(trend_strength=-40.0),
    )
    agreeing_trend = next(item for item in agreeing.ranked if item.family == "TREND")
    disagreeing_trend = next(item for item in disagreeing.ranked if item.family == "TREND")
    assert disagreeing_trend.score < agreeing_trend.score


def test_micro_terms_are_family_specific_not_one_bullishness_score(selector) -> None:
    micro = selector.micro_confirmation_from_context(
        trend_strength=30.0,
        orderflow_imbalance=-0.6,
        breakout_state=0.4,
        relative_strength=0.01,
        liquidity_score=0.9,
    )
    assert micro.get("TREND") > 0.0
    assert micro.get("ORDER_FLOW") < 0.0
    assert micro.get("DEFENSIVE") < 0.0


def test_order_flow_confirmation_is_scaled_by_liquidity(selector) -> None:
    deep = selector.micro_confirmation_from_context(
        orderflow_imbalance=0.8, liquidity_score=0.9
    )
    thin = selector.micro_confirmation_from_context(
        orderflow_imbalance=0.8, liquidity_score=0.05
    )
    assert deep.get("ORDER_FLOW") > thin.get("ORDER_FLOW")


def test_session_phase_shifts_the_ranking(selector) -> None:
    regime = _regime(TREND_UP=0.55, BREAKOUT_UP=0.5)
    opening = selector.select(
        ticker="005930", regime=regime, decided_at=NOW, session_phase="OPENING"
    )
    midday = selector.select(
        ticker="005930", regime=regime, decided_at=NOW, session_phase="MIDDAY"
    )
    opening_gap = next(item for item in opening.ranked if item.family == "GAP")
    midday_gap = next(item for item in midday.ranked if item.family == "GAP")
    assert opening_gap.score > midday_gap.score


def test_learned_weights_change_the_ranking_and_stay_traceable() -> None:
    graph = load_market_graph()
    baseline = RegimeStrategySelector(graph=graph).select(
        ticker="005930", regime=_regime(TREND_UP=0.9), decided_at=NOW
    )
    graph.apply_learned_weights({"TREND_UP|SUITABLE_FOR|TREND": 0.05})
    learned = RegimeStrategySelector(graph=graph).select(
        ticker="005930", regime=_regime(TREND_UP=0.9), decided_at=NOW
    )
    baseline_trend = next(item for item in baseline.ranked if item.family == "TREND")
    learned_trend = next(item for item in learned.ranked if item.family == "TREND")
    assert learned_trend.score < baseline_trend.score
    record = next(
        item
        for item in learned_trend.ontology_relations
        if item["edge_id"] == "TREND_UP|SUITABLE_FOR|TREND"
    )
    assert record["prior_strength"] == pytest.approx(0.85)
    assert record["learned_weight"] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def test_selection_serialises_the_whole_ranking(selector) -> None:
    payload = selector.select(
        ticker="005930", regime=_regime(TREND_UP=0.9), decided_at=NOW
    ).as_dict()
    assert len(payload["ranked"]) == len(STRATEGY_FAMILIES)
    for item in payload["ranked"]:
        assert set(item) >= {
            "family",
            "score",
            "regime_term",
            "ontology_term",
            "model_term",
            "micro_term",
            "excluded",
            "strategy_ids",
        }


def test_selector_has_no_path_to_execution() -> None:
    import app.routing.regime_strategy_selector as module

    text = open(module.__file__, encoding="utf-8").read()  # type: ignore[arg-type]
    for forbidden in ("app.execution", "app.risk", "LiveExecutionCoordinator"):
        assert f"import {forbidden}" not in text
        assert f"from {forbidden}" not in text


# --------------------------------------------------------------------------- #
# Regime estimator
# --------------------------------------------------------------------------- #
def test_regime_probabilities_are_independent_not_a_distribution() -> None:
    estimate = RegimeEstimator().estimate(
        RegimeEvidence(
            direction=-0.7,
            breadth=-0.6,
            volatility=0.008,
            liquidity=0.15,
            flow=-0.8,
            global_risk_sentiment=-0.9,
            global_volatility=1.6,
            venue_divergence=0.8,
        ),
        evaluated_at=NOW,
    )
    total = sum(estimate.probabilities.values())
    assert total > 1.0, "multi-label probabilities must not be normalised"
    assert estimate.probability("RISK_OFF") > 0.7
    assert estimate.probability("LIQUIDITY_STRESS") > 0.7


def test_model_evidence_is_blended_and_both_sides_stay_visible() -> None:
    evidence = RegimeEvidence(direction=0.6, breadth=0.4, volatility=0.001, liquidity=0.9)
    estimator = RegimeEstimator()
    rule_only = estimator.estimate(evidence, evaluated_at=NOW)
    blended = estimator.estimate(
        evidence,
        evaluated_at=NOW,
        model_probabilities={"TREND_DOWN": 1.0},
        model_version="test:1",
    )
    assert rule_only.source == "rule"
    assert blended.source == "rule+model"
    assert blended.probability("TREND_DOWN") > rule_only.probability("TREND_DOWN")
    contribution = blended.contributions["TREND_DOWN"]
    assert contribution["rule"] == pytest.approx(rule_only.probability("TREND_DOWN"))
    assert contribution["model"] == pytest.approx(1.0)


def test_missing_evidence_lowers_confidence_and_is_named() -> None:
    estimate = RegimeEstimator().estimate(RegimeEvidence(), evaluated_at=NOW)
    assert estimate.confidence == 0.0
    assert "REGIME_NO_DIRECTION" in estimate.reasons
    assert "REGIME_NO_BREADTH" in estimate.reasons


def test_index_breadth_divergence_is_detectable() -> None:
    estimate = RegimeEstimator().estimate(
        RegimeEvidence(direction=0.5, breadth=-0.4, volatility=0.002, liquidity=0.8),
        evaluated_at=NOW,
    )
    assert estimate.probability("INDEX_UP_BREADTH_DOWN") > 0.5
    assert estimate.probability("INDEX_DOWN_BREADTH_UP") == 0.0
