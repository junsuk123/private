"""Every catalogue strategy must be declared in the routing relation map.

The GNN gate requires ``compatible:{strategy_id}``, and that fact came from
``compatibility.get(strategy_id, 0.0)``. Eight of the sixteen ids were simply
absent from the map, so the default made them permanently unselectable - among
them ``residual_relative_strength``, the best-performing KRX candidate in the
counterfactual report (+24.6bp net). A missing key was indistinguishable from a
computed zero, so nothing reported it.
"""

from __future__ import annotations

from app.routing.shadow_intelligence import (
    COMPATIBILITY_UNAVAILABLE_REASONS,
    _strategy_compatibility,
    compatibility_coverage,
)
from app.strategy.catalog import STRATEGY_IDS


def test_every_strategy_appears_in_the_relation_map() -> None:
    compatibility = _strategy_compatibility(tuple([0.0] * 28))

    assert set(compatibility) == set(STRATEGY_IDS)


def test_no_strategy_is_undeclared() -> None:
    coverage = compatibility_coverage()

    undeclared = [key for key, value in coverage.items() if value == "UNDECLARED"]
    # Adding a strategy to the catalogue without either a relation or a stated
    # reason it cannot have one is the defect this test exists to catch.
    assert undeclared == []


def test_unavailable_reasons_name_the_missing_fact() -> None:
    coverage = compatibility_coverage()

    for strategy_id, reason in COMPATIBILITY_UNAVAILABLE_REASONS.items():
        assert strategy_id in STRATEGY_IDS, f"stale entry for removed strategy {strategy_id}"
        # "CONTEXT_UNAVAILABLE:PEER_RESIDUALS" tells an engineer what to supply;
        # a bare zero tells them nothing.
        assert reason.startswith("CONTEXT_UNAVAILABLE:")
        assert coverage[strategy_id] == reason


def test_computed_relations_respond_to_their_facts() -> None:
    # Feature layout: 2 = spread, 4 = close location, 6 = signed return proxy.
    quiet = [0.0] * 28
    trending = list(quiet)
    trending[6] = 1.0  # strong upward aggressor proxy

    quiet_score = _strategy_compatibility(tuple(quiet))["intraday_momentum"]
    trending_score = _strategy_compatibility(tuple(trending))["intraday_momentum"]

    assert trending_score > quiet_score


def test_context_unavailable_strategies_stay_zero() -> None:
    # Whatever the snapshot says, a relation this contract cannot express must
    # not acquire a value by accident - a fabricated prior is worse than none.
    loud = [1.0] * 28

    compatibility = _strategy_compatibility(tuple(loud))

    for strategy_id in COMPATIBILITY_UNAVAILABLE_REASONS:
        assert compatibility[strategy_id] == 0.0
