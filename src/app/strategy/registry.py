"""The single source of truth for strategy specs, derived from the live code.

Derivation, not duplication
---------------------------
Almost nothing here is a literal. Ids and their order come from
``app.strategy.catalog.STRATEGY_IDS``; direction from
``strategy_algorithms.strategy_direction``; horizon, ``min_liquidity_score`` and
``max_spread_bps`` from the resolved ``AlgorithmConfig`` (built-in defaults < YAML <
env), so a YAML edit moves the spec; the lifecycle floor from
``strategy_live_authorized`` / ``strategy_shadow_authorized``.

The two things that ARE declared here are ``required_features`` and
``required_election_inputs``. They were extracted from each algorithm's ``entry()``
body — the fields it actually dereferences — because a Python method cannot report its
own requirements without being run, and running every algorithm to compute an
eligibility mask is exactly the cost this layer exists to avoid.
``tests/test_strategy_spec_registry.py`` asserts every declared name is a real field of
``TechnicalFeatureSet`` / ``ElectionContext`` / ``MarketContext``, so a rename cannot
leave a dead requirement behind.

Lifecycle recommendations vs. lifecycle changes
----------------------------------------------
Where the stored evidence contradicts a strategy's current authorisation, the registry
records a RECOMMENDATION and leaves the operating state alone. Applying the
recommendations requires the explicit migration flag
``STRATEGY_LIFECYCLE_APPLY_RECOMMENDATIONS``. Guessing a strategy into or out of LIVE
from a code read is precisely what the task forbids, and the flag is what makes the
change an audited operator act.

Short strategies
----------------
``config/short_strategy_deployment.yaml`` has ``enabled: false`` and every short arm
individually disabled, because this account cannot trade 대주/공매도. The registry
reports short specs (so coverage analysis can say *what is missing* rather than
silently omitting it) and pins their lifecycle at ``RESEARCH`` regardless of the
``live_authorized: 1.0`` in their algorithm defaults. Nothing here can re-enable them:
the promotion controller and the borrow locate remain the gates.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Mapping

from app.strategy.catalog import STRATEGY_IDS, is_short_strategy
from app.strategy.spec import StrategyFamily, StrategyLifecycleState, StrategySpec

__all__ = [
    "LIFECYCLE_RECOMMENDATIONS",
    "StrategyRegistry",
    "default_strategy_registry",
    "reset_default_strategy_registry",
]


# --------------------------------------------------------------------------- #
# Families                                                                     #
# --------------------------------------------------------------------------- #
# Cross-checked against ``strategy_algorithms.MACRO_FAMILY_BY_STRATEGY``: where that
# map gives a strategy several coarse macro families (``gap_context`` is both momentum
# and breakout) the PRIMARY thesis decides the family here, because coverage has to
# partition the catalogue rather than double-count it.
_FAMILY: dict[str, StrategyFamily] = {
    "intraday_momentum": StrategyFamily.TREND_FOLLOWING,
    "breakout_volume": StrategyFamily.BREAKOUT,
    "vwap_mean_reversion": StrategyFamily.MEAN_REVERSION,
    "liquidity_shock_reversal": StrategyFamily.MICROSTRUCTURE_REVERSAL,
    "event_momentum": StrategyFamily.EVENT_DRIVEN,
    "cross_sectional_relative_strength": StrategyFamily.TREND_FOLLOWING,
    # Primary thesis is the overnight gap's continuation/fill, which is an event-like
    # dislocation rather than a range break.
    "gap_context": StrategyFamily.EVENT_DRIVEN,
    "rvgi_box_breakout": StrategyFamily.BREAKOUT,
    "residual_relative_strength": StrategyFamily.TREND_FOLLOWING,
    "adaptive_anchored_vwap_reversion": StrategyFamily.MEAN_REVERSION,
    "ofi_microprice_exhaustion_reversal": StrategyFamily.MICROSTRUCTURE_REVERSAL,
    "opening_range_breakout": StrategyFamily.BREAKOUT,
    "market_intraday_momentum": StrategyFamily.TREND_FOLLOWING,
    "market_intraday_momentum_short": StrategyFamily.TREND_FOLLOWING_SHORT,
    "opening_range_breakdown": StrategyFamily.BREAKDOWN_SHORT,
    "residual_relative_weakness": StrategyFamily.TREND_FOLLOWING_SHORT,
    "bar_confirmed_vwap_recovery": StrategyFamily.MEAN_REVERSION,
    "overnight_gap_carry": StrategyFamily.CROSS_SESSION,
    "range_support_reversion": StrategyFamily.MEAN_REVERSION,
    "bar_trend_continuation": StrategyFamily.TREND_FOLLOWING,
    "supertrend_dmi_continuation": StrategyFamily.TREND_FOLLOWING,
    "keltner_volatility_breakout": StrategyFamily.BREAKOUT,
    "choppiness_range_reversion": StrategyFamily.MEAN_REVERSION,
}


# --------------------------------------------------------------------------- #
# Requirements, extracted from the algorithms' entry() bodies                   #
# --------------------------------------------------------------------------- #
# ``symbol`` is deliberately excluded: it is an identifier, always present, and listing
# it would put a trivially-satisfied requirement on every strategy.
_REQUIRED_FEATURES: dict[str, tuple[str, ...]] = {
    "intraday_momentum": (
        "aggressor_imbalance_5s", "ema_fast", "ema_slow", "macd_histogram", "return_5s",
    ),
    "breakout_volume": (
        "aggressor_imbalance_5s", "breakout_strength", "donchian_high", "return_5s",
        "volume_spike_ratio",
    ),
    "vwap_mean_reversion": (
        "aggressor_imbalance_5s", "bb_percent_b", "orderbook_imbalance_change_5s",
        "return_5s", "rsi", "vwap", "vwap_distance_bps",
    ),
    "liquidity_shock_reversal": (
        "aggressor_imbalance_5s", "orderbook_imbalance", "return_10s", "return_30s",
        "spread_change_5s",
    ),
    "event_momentum": (
        "aggressor_imbalance_5s", "realized_volatility_10s", "return_10s",
        "volume_spike_ratio",
    ),
    "cross_sectional_relative_strength": ("aggressor_imbalance_5s", "short_return"),
    "gap_context": ("aggressor_imbalance_5s", "volume_spike_ratio"),
    "rvgi_box_breakout": (
        "aggressor_imbalance_5s", "price", "return_1s", "return_5s", "volume_spike_ratio",
    ),
    "residual_relative_strength": (
        "aggressor_imbalance_5s", "relative_volume", "spread_bps", "orderbook_imbalance",
    ),
    "adaptive_anchored_vwap_reversion": (
        "orderbook_imbalance_change_5s", "price", "return_1s", "spread_change_5s",
        "spread_bps", "orderbook_imbalance",
    ),
    "ofi_microprice_exhaustion_reversal": (
        "aggressor_imbalance_5s", "depth_ratio", "orderbook_imbalance",
        "orderbook_imbalance_change_5s", "return_10s", "spread_change_5s", "spread_bps",
    ),
    "opening_range_breakout": (
        "aggressor_imbalance_5s", "price", "spread_change_5s", "volume_spike_ratio",
    ),
    "market_intraday_momentum": ("aggressor_imbalance_5s",),
    "market_intraday_momentum_short": ("aggressor_imbalance_5s",),
    "opening_range_breakdown": (
        "aggressor_imbalance_5s", "price", "spread_change_5s", "volume_spike_ratio",
    ),
    "residual_relative_weakness": (
        "aggressor_imbalance_5s", "relative_volume", "spread_bps", "orderbook_imbalance",
    ),
    "bar_confirmed_vwap_recovery": (
        "ema_fast", "liquidity_score", "macd_histogram", "momentum_persistence", "price",
        "realized_volatility", "rsi", "spread_bps", "vwap", "vwap_distance_bps",
    ),
    "overnight_gap_carry": (
        "aggressor_imbalance_5s", "liquidity_score", "momentum_persistence",
        "vwap_distance_bps",
    ),
    "range_support_reversion": (
        "atr_pct", "donchian_high", "donchian_low", "donchian_low_distance",
        "liquidity_score", "price", "spread_bps",
    ),
    "bar_trend_continuation": (
        "atr_pct", "ema_fast", "ema_slow", "liquidity_score", "macd_histogram",
        "momentum_persistence", "price", "relative_volume", "spread_bps",
        "vwap_distance_bps",
    ),
    "supertrend_dmi_continuation": (
        "adx", "atr_pct", "dmi_spread", "liquidity_score",
        "momentum_persistence", "price", "relative_volume", "spread_bps",
        "supertrend", "supertrend_direction", "supertrend_distance_bps",
        "vwap_distance_bps",
    ),
    "keltner_volatility_breakout": (
        "adx", "atr_pct", "bb_bandwidth", "dmi_spread", "keltner_bandwidth",
        "keltner_upper", "liquidity_score", "price", "relative_volume",
        "spread_bps", "volatility_expansion", "vwap_distance_bps",
    ),
    "choppiness_range_reversion": (
        "adx", "atr_pct", "bb_percent_b", "choppiness", "liquidity_score",
        "macd_histogram", "price", "relative_volume", "rsi", "spread_bps",
        "vwap", "vwap_distance_bps",
    ),
}

# Slow context only the electing layer can resolve. ``change_point_probability`` is the
# shared regime guard every newer algorithm reads; it is listed where the algorithm
# actually compares against it.
_REQUIRED_ELECTION_INPUTS: dict[str, tuple[str, ...]] = {
    "event_momentum": ("event_fresh", "event_age_seconds", "event_ttl_seconds"),
    "cross_sectional_relative_strength": ("sector_rank", "sector_candidate_count"),
    "gap_context": (
        "gap_rate", "gap_submode", "session_open_price", "previous_close_price",
    ),
    "rvgi_box_breakout": (
        "rvgi", "rvgi_signal", "rvgi_diff", "rvgi_bullish_cross", "box_high", "box_low",
        "box_mid", "box_width_pct", "box_previous_close", "box_context_timestamp",
        "volume_confirmed",
    ),
    "residual_relative_strength": (
        "residual_return_short_bps", "residual_return_long_bps", "market_beta",
        "sector_beta", "sector_rank", "sector_candidate_count", "change_point_probability",
    ),
    "adaptive_anchored_vwap_reversion": (
        "anchored_vwap", "anchor_basis", "change_point_probability",
    ),
    "ofi_microprice_exhaustion_reversal": ("flow_toxicity", "change_point_probability"),
    "opening_range_breakout": (
        "opening_range_high", "opening_range_low", "relative_volume",
        "change_point_probability",
    ),
    "market_intraday_momentum": (
        "first_half_hour_return_bps", "first_half_hour_volatility_percentile",
        "in_last_continuous_half_hour", "minutes_to_continuous_close",
        "change_point_probability",
    ),
    "market_intraday_momentum_short": (
        "first_half_hour_return_bps", "first_half_hour_volatility_percentile",
        "in_last_continuous_half_hour", "minutes_to_continuous_close", "market_breadth",
        "change_point_probability", "borrow_available", "borrow_fee_bps_annualised",
        "short_sale_permitted", "borrow_available_quantity", "borrow_observed_at",
        "days_to_cover", "short_interest_ratio",
    ),
    "opening_range_breakdown": (
        "opening_range_high", "opening_range_low", "relative_volume", "market_breadth",
        "change_point_probability", "borrow_available", "borrow_fee_bps_annualised",
        "short_sale_permitted", "borrow_available_quantity", "borrow_observed_at",
        "days_to_cover", "short_interest_ratio",
    ),
    "residual_relative_weakness": (
        "residual_short_bps", "residual_long_bps", "market_beta", "sector_beta",
        "sector_rank", "sector_candidate_count", "change_point_probability",
        "borrow_available", "borrow_fee_bps_annualised", "short_sale_permitted",
        "borrow_available_quantity", "borrow_observed_at", "days_to_cover",
        "short_interest_ratio",
    ),
    "overnight_gap_carry": (
        "in_last_continuous_half_hour", "minutes_to_continuous_close",
        "change_point_probability",
    ),
}

# ``MarketContext`` fields a thesis is undefined without. Only structural requirements
# appear here — a strategy that merely *prefers* a state states that as a soft ontology
# relation, because a preference must never zero the mask.
_REQUIRED_CONTEXT: dict[str, tuple[str, ...]] = {
    "cross_sectional_relative_strength": ("relative_strength_rank",),
    "residual_relative_strength": ("relative_strength_rank",),
    "residual_relative_weakness": ("relative_strength_rank",),
    "opening_range_breakout": ("opening_range_position", "minutes_from_open"),
    "opening_range_breakdown": ("opening_range_position", "minutes_from_open"),
    "market_intraday_momentum": ("minutes_to_close",),
    "market_intraday_momentum_short": ("minutes_to_close",),
    "overnight_gap_carry": ("minutes_to_close",),
    "range_support_reversion": ("support_distance_bps",),
    "vwap_mean_reversion": ("vwap_distance_bps",),
    "bar_confirmed_vwap_recovery": ("vwap_distance_bps",),
    "adaptive_anchored_vwap_reversion": ("anchored_vwap_distance_bps",),
    "rvgi_box_breakout": ("breakout_distance_bps", "range_position"),
    "breakout_volume": ("breakout_distance_bps",),
    "event_momentum": ("event_recency_seconds",),
}

# Completed-bar history each thesis needs. Derived from the window each algorithm's
# indicators are computed over, not chosen: the box/Donchian lookback is 20 bars, the
# shared context window is 30, and the market-intraday-momentum thesis measures a
# first half-hour (30 one-minute bars) before it can compare anything.
_MINIMUM_HISTORY_BARS: dict[str, int] = {
    "breakout_volume": 20,
    "rvgi_box_breakout": 20,
    "range_support_reversion": 20,
    "bar_trend_continuation": 30,
    "supertrend_dmi_continuation": 30,
    "keltner_volatility_breakout": 30,
    "choppiness_range_reversion": 30,
    "vwap_mean_reversion": 30,
    "bar_confirmed_vwap_recovery": 30,
    "adaptive_anchored_vwap_reversion": 30,
    "market_intraday_momentum": 30,
    "market_intraday_momentum_short": 30,
    "opening_range_breakout": 30,
    "opening_range_breakdown": 30,
    "overnight_gap_carry": 30,
    "residual_relative_strength": 30,
    "residual_relative_weakness": 30,
}

# Session phases in which a NEW entry for this thesis is defined at all. Empty/absent
# means the thesis makes no session claim and the shared session gate decides.
#
# ``regular`` is the only phase these theses are defined in; the two that name it
# explicitly do so because their entry window is a specific part of the continuous
# session and firing them in a pre/after auction would price them against a different
# matching mechanism.
_ALLOWED_SESSIONS: dict[str, tuple[str, ...]] = {
    "opening_range_breakout": ("regular",),
    "opening_range_breakdown": ("regular",),
    "market_intraday_momentum": ("regular",),
    "market_intraday_momentum_short": ("regular",),
    "overnight_gap_carry": ("regular",),
}

# Market restrictions that exist in the code rather than in preference.
#
# ``liquidity_shock_reversal`` carries ``absorption_us_only: 1.0``, but that gates ONE
# of its two branches, so the strategy as a whole stays market-agnostic and this map
# does not restrict it. Nothing else in the catalogue restricts by market, and adding a
# restriction here would be a trading decision disguised as a spec.
_ALLOWED_MARKETS: dict[str, tuple[str, ...]] = {}


# --------------------------------------------------------------------------- #
# Lifecycle recommendations                                                    #
# --------------------------------------------------------------------------- #
#: ``strategy_id -> (recommended state, evidence)``. Recommendations only; applying
#: them needs ``STRATEGY_LIFECYCLE_APPLY_RECOMMENDATIONS``.
LIFECYCLE_RECOMMENDATIONS: dict[str, tuple[StrategyLifecycleState, str]] = {
    # SATISFIED by config, kept as the audit trail.
    #
    # Evidence is the measurement recorded in ``app.strategy.catalog`` at registration: the
    # screen that produced this thesis scored t=3.01 across ALL stored symbols, but restricted
    # to instruments this account can order it falls to t=1.56, and to t=0.65 on the
    # sub-period where the data lives. A later three-way comparison put every tested
    # condition within |t| <= 1.02 of buying at random.
    #
    # This entry originally recommended LIVE_PROBE against a LIVE authorisation.
    # ``config/strategy_algorithms.yaml`` has since set ``live_authorized: false``, which goes
    # further than the recommendation, so the derived lifecycle is already SHADOW and applying
    # this changes nothing. It stays here because deleting it would erase why the config says
    # what it says — and a recommendation can only ever LOWER a state, so it cannot become a
    # route back up.
    "range_support_reversion": (
        StrategyLifecycleState.SHADOW,
        "tradable-universe t=1.56 (sub-period t=0.65); significance carried by 2X "
        "inverse ETFs this account cannot order; registered on operator decision. "
        "config/strategy_algorithms.yaml now sets live_authorized=false, so the operating "
        "state already satisfies this",
    ),
    # Measured 2026-08-11 by ``scripts/report_strategy_selection_v2.py`` over the stored
    # performance data: 739 outcomes, mean net -119.6bps, median -94.7bps, one-sided 95%
    # lower bound -127.5bps, and not one positive walk-forward window (out-of-sample
    # stability 0.00). The gross edge is negative too, so no cost multiple rescues it —
    # this is a strategy problem, not a cost problem. (The count grows as the shadow
    # evaluator keeps scoring; re-run the script for the current figure.)
    #
    # The scope of the claim, stated because it bounds the recommendation: every one of
    # those rows is ``evaluation_source=shadow`` and every one is US. So this is a large
    # SIMULATED sample on one market, which is ample reason to stop trading the thesis and
    # no reason at all to close the file on it — hence SHADOW rather than RETIRED, and hence
    # the audit runner refuses to retire on shadow-only evidence.
    #
    # Unlike ``range_support_reversion`` this one is NOT yet satisfied by config: the
    # strategy still resolves to LIVE.
    "liquidity_shock_reversal": (
        StrategyLifecycleState.SHADOW,
        "739 shadow outcomes (US only), mean net -119.6bps, lower bound -127.5bps, "
        "out-of-sample stability 0.00; gross edge also negative",
    ),
}

_APPLY_RECOMMENDATIONS_ENV = "STRATEGY_LIFECYCLE_APPLY_RECOMMENDATIONS"

#: Version stamped onto every spec built from this module. Bump when the declared
#: requirements change, so a stored selection can be matched to the spec that produced it.
REGISTRY_VERSION = "spec-v1"


def _apply_recommendations() -> bool:
    return os.getenv(_APPLY_RECOMMENDATIONS_ENV, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class StrategyRegistry:
    """Specs for every catalogued strategy, in ``STRATEGY_IDS`` order."""

    def __init__(
        self,
        *,
        algorithm_config: Any | None = None,
        apply_recommendations: bool | None = None,
    ) -> None:
        self._apply_recommendations = (
            _apply_recommendations()
            if apply_recommendations is None
            else bool(apply_recommendations)
        )
        self._specs: dict[str, StrategySpec] = {}
        self._resolved: Mapping[str, Mapping[str, float]] = {}
        self._build(algorithm_config)

    # -- construction ------------------------------------------------------- #
    def _build(self, algorithm_config: Any | None) -> None:
        from app.technical.strategy_algorithms import (
            AlgorithmConfig,
            build_algorithm_registry,
            strategy_direction,
        )

        config = algorithm_config or AlgorithmConfig()
        self._resolved = config.as_dict()
        algorithms = build_algorithm_registry(config)

        for strategy_id in STRATEGY_IDS:
            algorithm = algorithms.get(strategy_id)
            values: Mapping[str, float] = self._resolved.get(strategy_id, {})
            horizon = (
                int(algorithm.horizon_seconds)
                if algorithm is not None
                else int(values.get("horizon_seconds", 300.0) or 300.0)
            )
            self._specs[strategy_id] = StrategySpec(
                strategy_id=strategy_id,
                family=_FAMILY.get(strategy_id, StrategyFamily.TREND_FOLLOWING),
                direction=strategy_direction(strategy_id),
                horizon_seconds=horizon,
                required_features=_REQUIRED_FEATURES.get(strategy_id, ()),
                required_context=_REQUIRED_CONTEXT.get(strategy_id, ()),
                required_election_inputs=_REQUIRED_ELECTION_INPUTS.get(strategy_id, ()),
                minimum_history_bars=_MINIMUM_HISTORY_BARS.get(strategy_id, 0),
                min_liquidity_score=_optional(values.get("min_liquidity_score")),
                max_spread_bps=_optional(values.get("max_spread_bps")),
                allowed_sessions=_ALLOWED_SESSIONS.get(strategy_id, ()),
                allowed_markets=_ALLOWED_MARKETS.get(strategy_id, ()),
                lifecycle_state=self._lifecycle_for(strategy_id),
                algorithm_version=REGISTRY_VERSION,
                validation_version=_validation_version(strategy_id),
                notes=self._notes_for(strategy_id),
            )

    def _lifecycle_for(self, strategy_id: str) -> StrategyLifecycleState:
        from app.technical.strategy_algorithms import (
            strategy_live_authorized,
            strategy_shadow_authorized,
        )

        if is_short_strategy(strategy_id):
            # Long-only account. The three short theses stay at RESEARCH here no matter
            # what their algorithm defaults say, and no code path in this package can
            # raise them — promotion belongs to ShortStrategyPromotionController, which
            # is itself disabled in config/short_strategy_deployment.yaml.
            return StrategyLifecycleState.RESEARCH

        try:
            live = strategy_live_authorized(strategy_id)
            shadow = strategy_shadow_authorized(strategy_id)
        except Exception:  # noqa: BLE001 - an unreadable authorisation fails closed.
            return StrategyLifecycleState.RESEARCH

        observed = (
            StrategyLifecycleState.LIVE
            if live
            else StrategyLifecycleState.SHADOW
            if shadow
            else StrategyLifecycleState.RESEARCH
        )
        recommendation = LIFECYCLE_RECOMMENDATIONS.get(strategy_id)
        if recommendation is None or not self._apply_recommendations:
            return observed
        # A recommendation may only LOWER the state. An upgrade must come from measured
        # forward outcomes through the promotion path, never from this table.
        recommended = recommendation[0]
        return recommended if recommended.rank < observed.rank else observed

    def _notes_for(self, strategy_id: str) -> tuple[str, ...]:
        notes: list[str] = []
        recommendation = LIFECYCLE_RECOMMENDATIONS.get(strategy_id)
        if recommendation is not None:
            state, evidence = recommendation
            applied = "applied" if self._apply_recommendations else "advisory-only"
            notes.append(f"LIFECYCLE_RECOMMENDATION={state} ({applied}): {evidence}")
        if is_short_strategy(strategy_id):
            notes.append(
                "SHORT_DISABLED: account cannot trade 대주/공매도; "
                "config/short_strategy_deployment.yaml enabled=false"
            )
        return tuple(notes)

    # -- reads -------------------------------------------------------------- #
    def get(self, strategy_id: str) -> StrategySpec | None:
        return self._specs.get(str(strategy_id or "").strip().lower())

    def require(self, strategy_id: str) -> StrategySpec:
        spec = self.get(strategy_id)
        if spec is None:
            raise KeyError(f"unknown strategy id: {strategy_id!r}")
        return spec

    def all_specs(self) -> tuple[StrategySpec, ...]:
        """Every spec, in ``STRATEGY_IDS`` order (index-stable)."""
        return tuple(self._specs[strategy_id] for strategy_id in STRATEGY_IDS)

    def long_specs(self) -> tuple[StrategySpec, ...]:
        return tuple(spec for spec in self.all_specs() if not spec.is_short)

    def specs_for_family(self, family: Any) -> tuple[StrategySpec, ...]:
        wanted = str(family)
        return tuple(spec for spec in self.all_specs() if str(spec.family) == wanted)

    def families(self) -> tuple[StrategyFamily, ...]:
        seen: dict[StrategyFamily, None] = {}
        for spec in self.all_specs():
            seen.setdefault(spec.family, None)
        return tuple(seen)

    def resolved_parameters(self, strategy_id: str) -> dict[str, float]:
        """The effective YAML/env-resolved knobs, for audit and parameter stress."""
        return dict(self._resolved.get(str(strategy_id or "").strip().lower(), {}))

    def lifecycle_recommendations(self) -> dict[str, dict[str, Any]]:
        return {
            strategy_id: {
                "recommended_state": str(state),
                "evidence": evidence,
                "applied": self._apply_recommendations,
                "current_state": str(self.require(strategy_id).lifecycle_state),
            }
            for strategy_id, (state, evidence) in LIFECYCLE_RECOMMENDATIONS.items()
            if strategy_id in self._specs
        }

    def as_table(self) -> list[dict[str, Any]]:
        return [spec.as_dict() for spec in self.all_specs()]


def _optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _validation_version(strategy_id: str) -> str:
    """What validation evidence exists, as a label rather than a number.

    Deliberately not a score. The honest state for most of this catalogue is "forward
    outcomes only", and inventing a version string that implies a completed validation
    run would be the fabricated-performance failure the task calls out.
    """
    if is_short_strategy(strategy_id):
        return "short-disabled-no-forward-evidence"
    if strategy_id in LIFECYCLE_RECOMMENDATIONS:
        return "measured-insufficient"
    return "forward-outcomes-only"


_default: StrategyRegistry | None = None
_default_lock = threading.Lock()


def default_strategy_registry() -> StrategyRegistry:
    global _default
    with _default_lock:
        if _default is None:
            _default = StrategyRegistry()
        return _default


def reset_default_strategy_registry() -> None:
    """Test hook; also used after an env/YAML change in a long-lived process."""
    global _default
    with _default_lock:
        _default = None
