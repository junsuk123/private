"""StrategySpec / registry invariants — including the append-only identity contract."""

from __future__ import annotations

from dataclasses import fields

import pytest

from app.context import declared_context_fields
from app.strategy.catalog import STRATEGY_IDS, STRATEGY_INDEX, SHORT_STRATEGY_IDS
from app.strategy.registry import (
    LIFECYCLE_RECOMMENDATIONS,
    StrategyRegistry,
    default_strategy_registry,
)
from app.strategy.spec import StrategyFamily, StrategyLifecycleState
from app.technical.signals import TechnicalFeatureSet
from app.technical.strategy_algorithms import ElectionContext


@pytest.fixture(scope="module")
def registry() -> StrategyRegistry:
    return default_strategy_registry()


def test_spec_order_matches_catalog_exactly(registry: StrategyRegistry) -> None:
    """Model output indices and persisted masks depend on this order being byte-stable."""
    assert tuple(spec.strategy_id for spec in registry.all_specs()) == STRATEGY_IDS
    for index, spec in enumerate(registry.all_specs()):
        assert STRATEGY_INDEX[spec.strategy_id] == index


def test_every_catalogued_strategy_has_a_spec(registry: StrategyRegistry) -> None:
    for strategy_id in STRATEGY_IDS:
        assert registry.get(strategy_id) is not None


def test_required_features_are_real_feature_fields(registry: StrategyRegistry) -> None:
    """A renamed feature must not leave a dead requirement behind."""
    known = {member.name for member in fields(TechnicalFeatureSet)}
    derived = {"microprice_edge_bps", "vwap_zscore", "residual_volatility_bps", "tick_data_ready"}
    for spec in registry.all_specs():
        for name in spec.required_features:
            assert name in known | derived, f"{spec.strategy_id} requires unknown feature {name}"


def test_required_election_inputs_are_real_election_context_fields(
    registry: StrategyRegistry,
) -> None:
    known = set(ElectionContext.__dataclass_fields__)
    for spec in registry.all_specs():
        for name in spec.required_election_inputs:
            assert name in known, f"{spec.strategy_id} requires unknown election input {name}"


def test_required_context_are_real_market_context_fields(
    registry: StrategyRegistry,
) -> None:
    known = set(declared_context_fields())
    for spec in registry.all_specs():
        for name in spec.required_context:
            assert name in known, f"{spec.strategy_id} requires unknown context field {name}"


def test_horizons_come_from_the_algorithm_config(registry: StrategyRegistry) -> None:
    from app.technical.strategy_algorithms import build_algorithm_registry

    algorithms = build_algorithm_registry()
    for spec in registry.all_specs():
        algorithm = algorithms[spec.strategy_id]
        assert spec.horizon_seconds == int(algorithm.horizon_seconds)


def test_short_strategies_are_pinned_to_research(registry: StrategyRegistry) -> None:
    """LONG_ONLY: this account cannot trade 대주/공매도, whatever the algorithm defaults say."""
    for strategy_id in SHORT_STRATEGY_IDS:
        spec = registry.require(strategy_id)
        assert spec.is_short
        assert spec.lifecycle_state is StrategyLifecycleState.RESEARCH
        assert not spec.lifecycle_state.is_live_candidate


def test_families_span_more_than_one_group(registry: StrategyRegistry) -> None:
    """Coverage across families is the point; one family would not be coverage."""
    families = registry.families()
    assert len(families) >= 5
    assert StrategyFamily.MEAN_REVERSION in families
    assert StrategyFamily.BREAKOUT in families


def test_lifecycle_recommendations_are_advisory_by_default(
    registry: StrategyRegistry,
) -> None:
    recommendations = registry.lifecycle_recommendations()
    assert "range_support_reversion" in recommendations
    row = recommendations["range_support_reversion"]
    assert row["applied"] is False
    # The operating state is derived from deployment authorization. Negative,
    # statistically inconclusive tradable-universe evidence must remain shadow
    # even when the advisory recommendation is not auto-applied.
    assert registry.require("range_support_reversion").lifecycle_state is (
        StrategyLifecycleState.SHADOW
    )


def test_recommendations_can_only_lower_when_applied() -> None:
    applied = StrategyRegistry(apply_recommendations=True)
    for strategy_id, (recommended, _) in LIFECYCLE_RECOMMENDATIONS.items():
        spec = applied.get(strategy_id)
        if spec is None:
            continue
        default_state = default_strategy_registry().require(strategy_id).lifecycle_state
        assert spec.lifecycle_state.rank <= default_state.rank


def test_lifecycle_is_live_candidate_excludes_degraded() -> None:
    assert StrategyLifecycleState.LIVE.is_live_candidate
    assert StrategyLifecycleState.LIVE_PROBE.is_live_candidate
    assert not StrategyLifecycleState.DEGRADED.is_live_candidate
    assert not StrategyLifecycleState.SHADOW.is_live_candidate
    assert not StrategyLifecycleState.RETIRED.is_live_candidate


def test_spec_declares_no_authorisation_api(registry: StrategyRegistry) -> None:
    """A spec must not be mistakable for permission to trade."""
    spec = registry.require("intraday_momentum")
    assert not hasattr(spec, "submits_orders")
    assert not hasattr(spec, "live_authorized")
