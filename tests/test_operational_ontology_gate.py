from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.ontology.operational_gate import (
    ClosedWorldOntologyGate,
    OperationalFact,
    OperationalOntologySnapshot,
    StrategyGateRule,
)


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _fact(name: str, value, *, age_seconds: int = 0, ttl_seconds: int = 5):
    observed = NOW - timedelta(seconds=age_seconds)
    return OperationalFact(
        name=name,
        value=value,
        observed_at=observed,
        valid_from=observed,
        valid_until=observed + timedelta(seconds=ttl_seconds),
        source="unit",
        confidence=0.9,
    )


def _snapshot(facts):
    return OperationalOntologySnapshot(
        snapshot_id="snapshot-1",
        symbol="005930",
        as_of=NOW,
        valid_until=NOW + timedelta(seconds=2),
        facts={fact.name: fact for fact in facts},
    )


def test_closed_world_missing_fact_is_a_hard_block() -> None:
    decision = ClosedWorldOntologyGate().evaluate(
        _snapshot((_fact("data_fresh", True),)),
        (
            StrategyGateRule(
                "momentum",
                required_true=("data_fresh", "liquid"),
            ),
        ),
    )
    assert decision.allowed_strategy_ids == ()
    assert decision.blocked_strategy_reasons["momentum"] == (
        "MISSING_REQUIRED_FACT:liquid",
    )


def test_stale_fact_cannot_authorize_strategy() -> None:
    decision = ClosedWorldOntologyGate().evaluate(
        _snapshot((_fact("data_fresh", True, age_seconds=10, ttl_seconds=5),)),
        (StrategyGateRule("momentum", required_true=("data_fresh",)),),
    )
    assert "momentum" not in decision.allowed_strategy_ids
    assert any(
        "STALE" in reason
        for reason in decision.blocked_strategy_reasons["momentum"]
    )


def test_valid_snapshot_allows_and_explains_strategy() -> None:
    decision = ClosedWorldOntologyGate().evaluate(
        _snapshot(
            (
                _fact("data_fresh", True),
                _fact("liquid", True),
                _fact("liquidity_score", 0.8),
            )
        ),
        (
            StrategyGateRule(
                "momentum",
                required_true=("data_fresh", "liquid"),
                minimum_confidence=0.8,
                compatibility_weights={"liquidity_score": 1.0},
            ),
        ),
    )
    assert decision.allowed_strategy_ids == ("momentum",)
    assert decision.compatibility_scores["momentum"] == 0.8
    assert len(decision.explanation_paths["momentum"]) == 2
