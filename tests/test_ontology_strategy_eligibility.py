"""Hard mask vs soft evidence: only hard relations may block."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.context import MarketContextBuilder, SymbolContextInputs
from app.ontology.strategy_eligibility import (
    ELIGIBILITY_REASONS,
    EligibilityConfig,
    StrategyEligibilityEngine,
)
from app.ontology.strategy_ontology import (
    HARD_RELATION_TYPES,
    SOFT_RELATION_TYPES,
    StrategyOntology,
    StrategyRelationType,
    default_strategy_ontology,
)
from app.technical.signals import TechnicalFeatureSet

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


def _context(**feature_overrides):
    values = {
        "symbol": "005930",
        "price": 70_000.0,
        "ema_fast": 70_100.0,
        "ema_slow": 69_900.0,
        "macd_histogram": 5.0,
        "vwap": 70_200.0,
        "vwap_distance_bps": -30.0,
        "spread_bps": 12.0,
        "orderbook_imbalance": 0.2,
        "liquidity_score": 0.6,
        "aggressor_imbalance_5s": 0.25,
        "realized_volatility": 0.0009,
        "realized_volatility_10s": 0.0006,
        "return_1s": 0.0001,
        "return_5s": 0.0004,
        "return_10s": 0.0006,
        "return_30s": 0.0009,
        "tick_count_5s": 8.0,
        "second_data_ready": 1.0,
        "donchian_low": 69_500.0,
        "donchian_high": 70_300.0,
        "donchian_low_distance": 0.007,
        "momentum_persistence": 0.65,
        "relative_volume": 1.8,
        "volume_spike_ratio": 2.0,
        "breakout_strength": -0.004,
        "rsi": 32.0,
        "bb_percent_b": 0.18,
        "atr_pct": 0.01,
        "spread_change_5s": -0.5,
        "orderbook_imbalance_change_5s": 0.05,
        "bid_depth": 1_000.0,
        "ask_depth": 800.0,
        "depth_ratio": 1.25,
        "short_return": 0.003,
        "breakout_distance_bps": -40.0,
        "box_position": 0.4,
    }
    values.update(feature_overrides)
    return MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(**values),
            context_id="ctx-1",
            tick_freshness_sec=0.5,
            orderbook_freshness_sec=0.8,
            history_bar_count=40,
            election_inputs={"sector_rank": 1, "sector_candidate_count": 5},
        ),
        captured_at=AT,
    )


ELECTION_INPUTS = {"sector_rank": 1, "sector_candidate_count": 5}


def test_relation_types_are_partitioned() -> None:
    assert HARD_RELATION_TYPES & SOFT_RELATION_TYPES == set()
    assert HARD_RELATION_TYPES | SOFT_RELATION_TYPES == set(StrategyRelationType)
    for relation in HARD_RELATION_TYPES:
        assert relation.is_hard
    for relation in SOFT_RELATION_TYPES:
        assert not relation.is_hard


def test_ontology_addresses_real_strategy_ids_only() -> None:
    from app.strategy.catalog import STRATEGY_INDEX

    ontology = default_strategy_ontology()
    payload = ontology.as_dict()
    for strategy_id in {*payload["soft_relations"], *payload["forbidden_under"]}:
        assert strategy_id in STRATEGY_INDEX, f"{strategy_id} is not a catalogued id"


def test_generic_methodology_names_are_not_addressable() -> None:
    """The alias vocabulary must not reach the eligibility layer."""
    ontology = default_strategy_ontology()
    for generic in ("momentum", "breakout", "mean_reversion", "vwap_reversion"):
        assert ontology.soft_relations(generic) == ()
        assert ontology.forbidden_states(generic) == ()


def test_soft_evidence_never_blocks() -> None:
    engine = StrategyEligibilityEngine()
    result = engine.evaluate(_context(), election_inputs=ELECTION_INPUTS)
    for item in result.eligibilities:
        if item.compatibility_score < 0:
            # A negative compatibility score is evidence, so it must not appear as a block.
            assert not any("worksWellUnder" in reason for reason in item.hard_block_reasons)
            assert not any("prefers" in reason for reason in item.hard_block_reasons)


def test_missing_required_feature_blocks() -> None:
    engine = StrategyEligibilityEngine()
    result = engine.evaluate(
        _context(aggressor_imbalance_5s=None), election_inputs=ELECTION_INPUTS
    ).by_id()
    blocked = result["intraday_momentum"]
    assert not blocked.eligible
    assert any(
        reason.startswith(ELIGIBILITY_REASONS.MISSING_FEATURE)
        for reason in blocked.hard_block_reasons
    )


def test_missing_election_input_blocks_but_keeps_soft_score() -> None:
    engine = StrategyEligibilityEngine()
    item = engine.evaluate(_context(), election_inputs={}).by_id()[
        "cross_sectional_relative_strength"
    ]
    assert not item.eligible
    assert any(
        reason.startswith(ELIGIBILITY_REASONS.MISSING_ELECTION_INPUT)
        for reason in item.hard_block_reasons
    )


def test_liquidity_floor_and_spread_ceiling_block() -> None:
    engine = StrategyEligibilityEngine()
    thin = engine.evaluate(
        _context(liquidity_score=0.1), election_inputs=ELECTION_INPUTS
    ).by_id()["bar_confirmed_vwap_recovery"]
    assert not thin.eligible
    assert any(
        reason.startswith(ELIGIBILITY_REASONS.LIQUIDITY_BELOW_FLOOR)
        for reason in thin.hard_block_reasons
    )

    wide = engine.evaluate(
        _context(spread_bps=90.0), election_inputs=ELECTION_INPUTS
    ).by_id()["bar_confirmed_vwap_recovery"]
    assert not wide.eligible
    assert any(
        reason.startswith(ELIGIBILITY_REASONS.SPREAD_ABOVE_CEILING)
        for reason in wide.hard_block_reasons
    )


def test_history_requirement_blocks() -> None:
    context = MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=_context().feature_snapshot and TechnicalFeatureSet(
                **{
                    key: value
                    for key, value in _context().feature_snapshot.items()
                    if key in TechnicalFeatureSet.__dataclass_fields__
                }
            ),
            context_id="ctx-thin-history",
            tick_freshness_sec=0.5,
            orderbook_freshness_sec=0.8,
            history_bar_count=3,
        ),
        captured_at=AT,
    )
    item = StrategyEligibilityEngine().evaluate(
        context, election_inputs=ELECTION_INPUTS
    ).by_id()["breakout_volume"]
    assert not item.eligible
    assert any(
        reason.startswith(ELIGIBILITY_REASONS.HISTORY_INSUFFICIENT)
        for reason in item.hard_block_reasons
    )


def test_short_direction_is_hard_blocked_under_long_only() -> None:
    engine = StrategyEligibilityEngine(long_only=True)
    result = engine.evaluate(_context(), election_inputs=ELECTION_INPUTS).by_id()
    for strategy_id in (
        "market_intraday_momentum_short",
        "opening_range_breakdown",
        "residual_relative_weakness",
    ):
        item = result[strategy_id]
        assert not item.eligible
        assert any(
            reason.startswith(ELIGIBILITY_REASONS.DIRECTION_NOT_PERMITTED)
            for reason in item.hard_block_reasons
        )


def test_no_new_entry_market_state_blocks_everything() -> None:
    class _Macro:
        market_regime = "HIGH_VOL_DISLOCATED"
        risk_regime = None
        change_point_probability = 0.8
        regime_stability = None
        volatility_percentile = None
        diagnostics: dict = {}

    features = TechnicalFeatureSet(
        **{
            key: value
            for key, value in _context().feature_snapshot.items()
            if key in TechnicalFeatureSet.__dataclass_fields__
        }
    )
    context = MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=features,
            context_id="ctx-dislocated",
            tick_freshness_sec=0.5,
            orderbook_freshness_sec=0.8,
            history_bar_count=40,
        ),
        captured_at=AT,
        macro=_Macro(),
    )
    result = StrategyEligibilityEngine().evaluate(
        context, election_inputs=ELECTION_INPUTS
    )
    assert result.eligible_ids == ()
    for item in result.eligibilities:
        assert any(
            reason.startswith(ELIGIBILITY_REASONS.NO_NEW_ENTRY_MARKET_STATE)
            for reason in item.hard_block_reasons
        )


def test_unresolved_regime_does_not_block() -> None:
    """An unanswerable permission check is not a withdrawal."""
    result = StrategyEligibilityEngine().evaluate(
        _context(), election_inputs=ELECTION_INPUTS
    )
    assert result.eligible_ids, "an absent regime label must not empty the mask"


def test_macro_family_block_is_respected() -> None:
    engine = StrategyEligibilityEngine()
    item = engine.evaluate(
        _context(),
        election_inputs=ELECTION_INPUTS,
        macro_blocked=("momentum",),
    ).by_id()["intraday_momentum"]
    assert not item.eligible
    assert any(
        reason.startswith(ELIGIBILITY_REASONS.MACRO_FAMILY_BLOCKED)
        for reason in item.hard_block_reasons
    )


def test_mask_is_zero_or_one() -> None:
    mask = StrategyEligibilityEngine().evaluate(
        _context(), election_inputs=ELECTION_INPUTS
    ).mask()
    assert set(mask.values()) <= {0.0, 1.0}


def test_completeness_floor_blocks_an_empty_context() -> None:
    empty = MarketContextBuilder().build(
        SymbolContextInputs(symbol="005930", features=TechnicalFeatureSet(symbol="005930")),
        captured_at=AT,
    )
    engine = StrategyEligibilityEngine(
        config=EligibilityConfig(minimum_feature_completeness=0.9)
    )
    result = engine.evaluate(empty)
    assert result.eligible_ids == ()
    assert any(
        reason.startswith(ELIGIBILITY_REASONS.COMPLETENESS_BELOW_FLOOR)
        for item in result.eligibilities
        for reason in item.hard_block_reasons
    )


def test_custom_ontology_is_injectable() -> None:
    engine = StrategyEligibilityEngine(
        ontology=StrategyOntology(
            soft_relations={},
            forbidden_under={"intraday_momentum": ("TREND_UP",)},
            no_new_entry_states=frozenset(),
        )
    )
    result = engine.evaluate(_context(), election_inputs=ELECTION_INPUTS).by_id()
    assert result["intraday_momentum"].compatibility_score == 0.0
