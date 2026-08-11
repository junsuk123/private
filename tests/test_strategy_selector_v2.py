"""StrategySelectorV2: utility decomposition, NO_TRADE, and what it cannot reach."""

from __future__ import annotations

import ast
import pathlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from app.context import MarketContextBuilder, SymbolContextInputs
from app.ontology.strategy_eligibility import StrategyEligibilityEngine
from app.routing.no_trade_policy import (
    NO_TRADE_REASONS,
    NoTradePolicy,
    NoTradePolicyConfig,
)
from app.routing.ontology_strategy_mask import MASK_DISABLED, OntologyStrategyMask
from app.routing.strategy_selector import (
    SELECTION_REASON_ENTRY_NOT_READY,
    SELECTION_REASON_LIFECYCLE_NOT_LIVE,
    SELECTION_VERSION,
    StrategySelectorV2,
    UtilityWeights,
)
from app.routing.strategy_utility import (
    CostEstimate,
    StrategyUtilityPrediction,
    TradingCostAdapter,
)
from app.strategy.registry import default_strategy_registry
from app.technical.signals import TechnicalFeatureSet

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)
ELECTION_INPUTS = {"sector_rank": 1, "sector_candidate_count": 5}


def _features(**overrides) -> TechnicalFeatureSet:
    values = {
        "symbol": "005930",
        "price": 70_000.0,
        "ema_fast": 70_400.0,
        "ema_slow": 69_900.0,
        "macd_histogram": 9.0,
        "vwap": 70_200.0,
        "vwap_distance_bps": -30.0,
        "spread_bps": 8.0,
        "orderbook_imbalance": 0.3,
        "liquidity_score": 0.7,
        "aggressor_imbalance_5s": 0.4,
        "realized_volatility": 0.006,
        "realized_volatility_10s": 0.004,
        "return_1s": 0.0003,
        "return_5s": 0.0012,
        "return_10s": 0.0016,
        "return_30s": 0.0020,
        "tick_count_5s": 12.0,
        "second_data_ready": 1.0,
        "donchian_low": 69_500.0,
        "donchian_high": 70_300.0,
        "donchian_low_distance": 0.007,
        "momentum_persistence": 0.75,
        "relative_volume": 2.2,
        "volume_spike_ratio": 2.4,
        "breakout_strength": -0.004,
        "rsi": 33.0,
        "bb_percent_b": 0.18,
        "atr_pct": 0.012,
        "spread_change_5s": -0.5,
        "orderbook_imbalance_change_5s": 0.08,
        "bid_depth": 1_400.0,
        "ask_depth": 800.0,
        "depth_ratio": 1.75,
        "short_return": 0.004,
        "breakout_distance_bps": -40.0,
        "box_position": 0.4,
    }
    values.update(overrides)
    return TechnicalFeatureSet(**values)


def _context(**overrides):
    features = overrides.pop("features", _features())
    return MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=features,
            context_id=overrides.pop("context_id", "ctx-1"),
            tick_freshness_sec=0.5,
            orderbook_freshness_sec=0.8,
            history_bar_count=40,
            election_inputs=ELECTION_INPUTS,
            **overrides,
        ),
        captured_at=AT,
    )


class _StubPredictor:
    """Fixed utility numbers, so the selector's arithmetic is what is under test."""

    def __init__(self, *, gross_bps: float, downside_bps: float = 40.0, uncertainty_bps: float = 10.0):
        self._gross = gross_bps
        self._downside = downside_bps
        self._uncertainty = uncertainty_bps

    def predict(self, context, proposals, costs, **_kwargs):
        return tuple(
            StrategyUtilityPrediction(
                strategy_id=proposal.strategy_id,
                context_id=context.context_id,
                symbol=context.symbol_id,
                probability_profit=0.6,
                expected_gross_return_bps=self._gross,
                expected_cost_bps=costs[proposal.strategy_id].expected_cost_bps,
                expected_downside_bps=self._downside,
                expected_holding_seconds=float(proposal.expected_horizon_seconds or 600),
                uncertainty_bps=self._uncertainty,
                model_version="stub-1",
                source="UTILITY_SOURCE_STUB",
            )
            for proposal in proposals
        )


class _NoBanditAdapter:
    def correct_all(self, *_args, **_kwargs):
        return {}


def _selector(**overrides) -> StrategySelectorV2:
    registry = default_strategy_registry()
    kwargs = {
        "registry": registry,
        "mask": OntologyStrategyMask(engine=StrategyEligibilityEngine(registry=registry)),
        "utility_predictor": _StubPredictor(gross_bps=400.0),
        "bandit_adapter": _NoBanditAdapter(),
        "weights": UtilityWeights(),
    }
    kwargs.update(overrides)
    return StrategySelectorV2(**kwargs)


# --------------------------------------------------------------------------- #
# Selection and NO_TRADE                                                       #
# --------------------------------------------------------------------------- #
def test_strong_edge_produces_a_selection() -> None:
    result = _selector().select(_context(), election_inputs=ELECTION_INPUTS)
    assert result.decision == "SELECT"
    assert result.selected_strategy is not None
    assert result.utility is not None and result.utility > 0
    assert result.selection_version == SELECTION_VERSION


def test_weak_edge_produces_no_trade_and_that_is_normal() -> None:
    result = _selector(utility_predictor=_StubPredictor(gross_bps=10.0)).select(
        _context(), election_inputs=ELECTION_INPUTS
    )
    assert result.decision == "NO_TRADE"
    assert result.selected_strategy is None
    assert result.utility is None
    assert NO_TRADE_REASONS.BELOW_MINIMUM_EDGE in result.reason_codes


def test_no_trade_utility_is_the_minimum_edge_bar() -> None:
    policy = NoTradePolicy(config=NoTradePolicyConfig())
    assert policy.no_trade_utility_bps(market="KR") == 10.0
    assert policy.no_trade_utility_bps(market="US") == 20.0
    # Per-market, per-horizon, and higher when the cost was not measured.
    assert policy.no_trade_utility_bps(market="KR", horizon_seconds=120.0) > 10.0
    assert policy.no_trade_utility_bps(market="KR", measured=False) > 10.0


def test_lower_bound_rule_can_veto_a_positive_mean() -> None:
    """net > bar but net - uncertainty <= bar must not trade."""
    result = _selector(
        utility_predictor=_StubPredictor(gross_bps=120.0, downside_bps=0.0, uncertainty_bps=90.0)
    ).select(_context(), election_inputs=ELECTION_INPUTS)
    assert result.decision == "NO_TRADE"
    assert NO_TRADE_REASONS.NEGATIVE_LOWER_BOUND in result.reason_codes


# --------------------------------------------------------------------------- #
# Utility decomposition                                                        #
# --------------------------------------------------------------------------- #
def test_every_term_is_recorded_and_sums_to_the_total() -> None:
    weights = UtilityWeights(
        lambda_downside=0.5, lambda_uncertainty=1.0, lambda_ontology_bps=8.0, lambda_bandit=1.0
    )
    result = _selector(weights=weights).select(_context(), election_inputs=ELECTION_INPUTS)
    for candidate in result.ranked_candidates:
        recomputed = (1.0 if candidate.eligible else 0.0) * (
            candidate.expected_gross_return_bps
            - candidate.expected_cost_bps
            - candidate.downside_penalty_bps
            - candidate.uncertainty_penalty_bps
            + candidate.ontology_adjustment_bps
            + candidate.bandit_adjustment_bps
        )
        assert candidate.final_utility_bps == pytest.approx(recomputed, abs=1e-9)
        assert candidate.expected_net_return_bps == pytest.approx(
            candidate.expected_gross_return_bps - candidate.expected_cost_bps, abs=1e-9
        )


def test_ranking_is_stable_and_sorted_by_utility() -> None:
    result = _selector().select(_context(), election_inputs=ELECTION_INPUTS)
    utilities = [item.final_utility_bps for item in result.ranked_candidates]
    assert utilities == sorted(utilities, reverse=True)
    again = _selector().select(_context(), election_inputs=ELECTION_INPUTS)
    assert [item.strategy_id for item in result.ranked_candidates] == [
        item.strategy_id for item in again.ranked_candidates
    ]


def test_weights_are_config_not_hardcoded() -> None:
    from app.routing.selector_v2_shadow import load_utility_weights

    weights = load_utility_weights("config/strategy_selector_v2.yaml")
    assert weights.lambda_downside > 0
    assert weights.lambda_ontology_bps > 0
    # And a mapping overrides them.
    override = UtilityWeights.from_mapping({"lambda_downside": 0.9})
    assert override.lambda_downside == 0.9


# --------------------------------------------------------------------------- #
# What may not be selected                                                     #
# --------------------------------------------------------------------------- #
def test_hard_blocked_strategy_never_appears_as_a_candidate() -> None:
    result = _selector().select(_context(), election_inputs={})
    ranked = {item.strategy_id for item in result.ranked_candidates}
    for strategy_id, reasons in result.blocked.items():
        assert reasons, f"{strategy_id} was blocked with no reason"
        assert strategy_id not in ranked


def test_entry_not_ready_is_ranked_but_not_selectable() -> None:
    result = _selector().select(_context(), election_inputs=ELECTION_INPUTS)
    not_ready = [item for item in result.ranked_candidates if not item.entry_ready]
    assert not_ready, "expected at least one eligible strategy whose trigger did not fire"
    for item in not_ready:
        assert not item.selectable
        assert SELECTION_REASON_ENTRY_NOT_READY in item.reason_codes
    assert result.selected_strategy not in {item.strategy_id for item in not_ready}


def test_non_live_lifecycle_is_ranked_but_not_selectable() -> None:
    result = _selector().select(_context(), election_inputs=ELECTION_INPUTS)
    for item in result.ranked_candidates:
        if SELECTION_REASON_LIFECYCLE_NOT_LIVE in item.reason_codes:
            assert not item.selectable
            assert result.selected_strategy != item.strategy_id


def test_coverage_gap_resolves_to_no_trade() -> None:
    """Strategies fired, none is authorised: NO_TRADE, not the nearest look-alike."""

    class _AllShadowRegistry:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def get(self, strategy_id):
            spec = self._inner.get(strategy_id)
            if spec is None:
                return None
            from app.strategy.spec import StrategyLifecycleState

            return replace(spec, lifecycle_state=StrategyLifecycleState.SHADOW)

        def require(self, strategy_id):
            spec = self.get(strategy_id)
            if spec is None:
                raise KeyError(strategy_id)
            return spec

        def all_specs(self):
            return tuple(self.get(spec.strategy_id) for spec in self._inner.all_specs())

    registry = _AllShadowRegistry(default_strategy_registry())
    result = _selector(registry=registry).select(_context(), election_inputs=ELECTION_INPUTS)
    assert result.decision == "NO_TRADE"
    assert NO_TRADE_REASONS.COVERAGE_GAP in result.reason_codes


def test_low_data_quality_resolves_to_no_trade() -> None:
    empty = MarketContextBuilder().build(
        SymbolContextInputs(symbol="005930", features=TechnicalFeatureSet(symbol="005930")),
        captured_at=AT,
    )
    result = _selector().select(empty)
    assert result.decision == "NO_TRADE"
    assert result.selected_strategy is None


def test_all_negative_utilities_resolve_to_no_trade() -> None:
    result = _selector(utility_predictor=_StubPredictor(gross_bps=-200.0)).select(
        _context(), election_inputs=ELECTION_INPUTS
    )
    assert result.decision == "NO_TRADE"
    assert all(item.final_utility_bps < 0 for item in result.ranked_candidates if item.eligible)


# --------------------------------------------------------------------------- #
# Cost separation                                                              #
# --------------------------------------------------------------------------- #
def test_cost_comes_from_the_cost_engine_not_the_model() -> None:
    adapter = TradingCostAdapter()
    estimate = adapter.estimate(
        strategy_id="intraday_momentum",
        symbol="005930",
        market="KR",
        reference_price=70_000.0,
        spread_bps=8.0,
    )
    assert estimate.measured
    # KRX round trip is ~28bps before spread; a mis-built orderbook once produced ~15,000.
    assert 20.0 < estimate.expected_cost_bps < 80.0

    us = adapter.estimate(
        strategy_id="intraday_momentum",
        symbol="AAPL",
        market="US",
        reference_price=200.0,
        spread_bps=8.0,
    )
    assert us.expected_cost_bps > estimate.expected_cost_bps


def test_unmeasured_cost_is_flagged_not_silently_defaulted() -> None:
    adapter = TradingCostAdapter(fallback_bps=28.0)
    estimate = adapter.estimate(
        strategy_id="intraday_momentum",
        symbol="005930",
        market="KR",
        reference_price=None,
    )
    assert not estimate.measured
    assert estimate.source == "fallback"


def test_net_return_is_an_identity_over_gross_and_cost() -> None:
    prediction = StrategyUtilityPrediction(
        strategy_id="intraday_momentum",
        context_id="ctx-1",
        symbol="005930",
        probability_profit=0.6,
        expected_gross_return_bps=100.0,
        expected_cost_bps=28.0,
        expected_downside_bps=40.0,
        expected_holding_seconds=600.0,
        uncertainty_bps=10.0,
        model_version="stub",
        source="stub",
    )
    assert prediction.expected_net_return_bps == pytest.approx(72.0)


# --------------------------------------------------------------------------- #
# Structural guarantees                                                        #
# --------------------------------------------------------------------------- #
_FORBIDDEN_IMPORT_PREFIXES = (
    "app.execution",
    "app.risk",
    "app.cost.profitability_gate",
    "app.trading.realtime_trading_engine",
    "app.trading.shared_decision_engine",
)

_SELECTION_MODULES = (
    "src/app/routing/strategy_selector.py",
    "src/app/routing/strategy_utility.py",
    "src/app/routing/no_trade_policy.py",
    "src/app/routing/ontology_strategy_mask.py",
    "src/app/routing/bandit_adapter.py",
    "src/app/routing/selector_v2_shadow.py",
    "src/app/strategy/proposal.py",
    "src/app/strategy/proposal_engine.py",
    "src/app/strategy/spec.py",
    "src/app/ontology/strategy_eligibility.py",
    "src/app/ontology/strategy_ontology.py",
    "src/app/evaluation/shadow_position.py",
    "src/app/evaluation/counterfactual_engine.py",
)


def _imported_modules(path: str) -> set[str]:
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", _SELECTION_MODULES)
def test_selection_layer_cannot_reach_execution(path: str) -> None:
    """A risk gate cannot be bypassed by a module that cannot import one."""
    for module in _imported_modules(path):
        for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
            assert not module.startswith(forbidden), f"{path} imports {module}"


def test_selection_result_carries_the_proposal_ids() -> None:
    result = _selector().select(_context(), election_inputs=ELECTION_INPUTS)
    payload = result.as_dict()
    assert payload["context_id"] == "ctx-1"
    assert payload["proposals"], "selection must record the proposals it ranked"
    for proposal in payload["proposals"]:
        assert proposal["proposal_id"]
        assert proposal["context_id"] == "ctx-1"


def test_mask_disabled_is_visible_rather_than_silent() -> None:
    """All-pass must be distinguishable from 'every strategy was genuinely eligible'."""
    registry = default_strategy_registry()
    engine = StrategyEligibilityEngine(registry=registry)
    context = _context()

    enabled = OntologyStrategyMask(engine=engine, enabled=True).evaluate(
        context, election_inputs={}
    )
    disabled = OntologyStrategyMask(engine=engine, enabled=False).evaluate(
        context, election_inputs={}
    )
    assert len(disabled.eligible_ids) > len(enabled.eligible_ids)
    for item in disabled.eligibilities:
        assert item.eligible
        # The reason the strategy WOULD have been blocked is retained, prefixed with the
        # marker, so a reader can tell an all-pass mask from a permissive market.
        assert item.hard_block_reasons[0] == MASK_DISABLED
    # And the soft compatibility score still comes from the real relations.
    assert {item.strategy_id: item.compatibility_score for item in disabled.eligibilities} == {
        item.strategy_id: item.compatibility_score for item in enabled.eligibilities
    }

    # Ranked candidates still get more entries, which is the operational consequence.
    permissive = _selector(
        mask=OntologyStrategyMask(engine=engine, enabled=False)
    ).select(context, election_inputs={})
    strict = _selector().select(context, election_inputs={})
    assert len(permissive.ranked_candidates) > len(strict.ranked_candidates)
    assert permissive.blocked == {}
