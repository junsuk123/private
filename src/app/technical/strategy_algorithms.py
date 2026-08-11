"""Seven mechanically distinct trading algorithms.

Design contract
---------------
The ontology (and an authorised GNN) elect ONE symbol and ONE strategy, and
supply the slow context that election required — event freshness, sector rank,
gap reference. From that moment this module owns the trade: it decides *when*
to fire, *where* the stop and target sit, and *what* invalidates the thesis.

What deliberately does NOT live here:

* regime / liquidity / volatility / spread admissibility — resolved once at
  election time by the ontology layer, then watched continuously by
  ``app.trading.strategy_supervisor``, which can halt this algorithm;
* account safety and executability — cash, duplicate orders, tick rules,
  RiskManager. Those are the submission boundary, not market analysis.

Each algorithm therefore contains only its own thesis. That is the whole point:
before this module, four of the seven strategy ids resolved to the *same*
momentum provider, so electing "event_momentum" and electing "gap_context"
executed byte-identical code.

Entry triggers read the sub-second window (``return_1s/5s/10s``,
``aggressor_imbalance_5s``, ``spread_change_5s``,
``orderbook_imbalance_change_5s``); minute-bar columns supply direction and
structure only. Everything is pure, deterministic and NaN-safe: a missing input
means "cannot fire", never a fabricated value.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.technical import reason_codes as rc
from app.technical.signals import TechnicalFeatureSet
from app.strategy.catalog import STRATEGY_IDS, is_short_strategy
from app.trading.directional import ShortReasonCodes

DEFAULT_CONFIG_PATH = "config/strategy_algorithms.yaml"


# --------------------------------------------------------------------------- #
# Contracts                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ElectionContext:
    """Slow context resolved ONCE when the ontology elected this strategy.

    The algorithm never recomputes these; it consumes them. A field left at its
    default means the electing layer did not supply it, and any algorithm that
    requires it fails closed.
    """

    strategy_id: str
    elected_at: datetime | None = None
    reference_price: float | None = None
    expected_net_return_bps: float | None = None
    # event_momentum
    event_fresh: bool = False
    event_age_seconds: float | None = None
    event_ttl_seconds: float | None = None
    # cross_sectional_relative_strength
    sector_rank: int | None = None
    sector_candidate_count: int | None = None
    # gap_context
    gap_rate: float | None = None
    gap_submode: str | None = None
    session_open_price: float | None = None
    previous_close_price: float | None = None
    # rvgi_box_breakout (all frozen at election)
    rvgi: float | None = None
    rvgi_signal: float | None = None
    rvgi_diff: float | None = None
    rvgi_bullish_cross: bool | None = None
    box_high: float | None = None
    box_low: float | None = None
    box_mid: float | None = None
    box_width_pct: float | None = None
    box_position: float | None = None
    box_context_timestamp: str | None = None
    box_previous_close: float | None = None
    volume_confirmed: bool | None = None
    # residual_relative_strength. Residuals are cross-sectional, so only the
    # electing layer can compute them; an algorithm cannot invent them from one
    # symbol's ticks. Absent -> the strategy fails closed.
    residual_return_short_bps: float | None = None
    residual_return_long_bps: float | None = None
    market_beta: float | None = None
    sector_beta: float | None = None
    foreign_flow_zscore: float | None = None
    institution_flow_zscore: float | None = None
    # adaptive_anchored_vwap_reversion. When the electing layer can resolve a
    # meaningful anchor (session open, volatility-spike time, news time) it passes
    # it here; otherwise the algorithm uses the session VWAP and says so.
    anchored_vwap: float | None = None
    anchor_basis: str | None = None
    # Shared regime guard: no thesis in this module is valid across a structural
    # break, so every new algorithm reads it rather than each re-deriving one.
    change_point_probability: float | None = None
    # ofi_microprice_exhaustion_reversal adverse-selection guard.
    flow_toxicity: float | None = None
    spread_percentile: float | None = None
    # opening_range_breakout. The opening range is session structure, so only the
    # electing layer can resolve it; an algorithm cannot recover it from the
    # sub-second window. Absent -> the strategy fails closed.
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    opening_range_minutes: float | None = None
    relative_volume: float | None = None
    # market_intraday_momentum. The first half-hour return is measured from the
    # PREVIOUS close (it includes the overnight gap), and the entry window is session
    # structure — neither is recoverable from the sub-second tick window, so the
    # electing layer supplies them and the algorithm fails closed without them.
    first_half_hour_return_bps: float | None = None
    first_half_hour_volatility_percentile: float | None = None
    in_last_continuous_half_hour: bool | None = None
    minutes_to_continuous_close: float | None = None
    # --- SHORT-side context -------------------------------------------------- #
    # Borrow facts, frozen at signal time. Every one defaults to the FAIL-CLOSED
    # value (``None`` / False), so a short algorithm asked to decide without them
    # refuses rather than assuming stock can be located. That asymmetry is
    # deliberate: an absent long-side feature costs a missed trade, an absent
    # borrow fact costs a rejected or force-closed position.
    #
    # These are point-in-time observations, never re-read at evaluation. Using a
    # later borrow lookup to score an earlier signal is the exact leak that would
    # make shadow results unachievable live.
    borrow_available: bool | None = None
    borrow_available_quantity: int | None = None
    # ANNUALISED bps, matching what the broker quotes and what
    # ``app.trading.borrow`` stores. The name carries the unit because mixing it
    # with a per-trade figure is a ~10,000x error: an annualised 800bps compared
    # against a per-trade 40bps ceiling rejects every borrowable name (this
    # actually happened during development, and every short silently reported
    # SHORT_BORROW_COST_TOO_HIGH).
    borrow_fee_bps_annualised: float | None = None
    borrow_observed_at: str | None = None
    short_sale_permitted: bool | None = None
    return_deadline: str | None = None
    # Squeeze pressure. A crowded short is where the unbounded loss actually
    # happens, so this is an exclusion input rather than a score contribution.
    short_interest_ratio: float | None = None
    days_to_cover: float | None = None
    # Mirror of the long residual fields. Stored separately rather than reusing the
    # long ones with a flipped sign: a residual measured for the strength thesis is
    # not the same measurement as one taken for the weakness thesis (different
    # window, different universe filter), and reusing it would silently make the
    # short label a function of the long label.
    residual_short_bps: float | None = None
    residual_long_bps: float | None = None
    # opening_range_breakdown: how far BELOW the opening-range low price has gone.
    breakdown_excess_bps: float | None = None
    aggressor_imbalance: float | None = None
    market_alignment: float | None = None
    market_trend: str | None = None
    market_breadth: float | None = None
    liquidity_score: float | None = None
    spread_bps: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "elected_at": self.elected_at.isoformat() if self.elected_at else None,
            "reference_price": self.reference_price,
            "expected_net_return_bps": self.expected_net_return_bps,
            "event_fresh": self.event_fresh,
            "event_age_seconds": self.event_age_seconds,
            "event_ttl_seconds": self.event_ttl_seconds,
            "sector_rank": self.sector_rank,
            "sector_candidate_count": self.sector_candidate_count,
            "gap_rate": self.gap_rate,
            "gap_submode": self.gap_submode,
            "rvgi": self.rvgi,
            "rvgi_signal": self.rvgi_signal,
            "rvgi_diff": self.rvgi_diff,
            "rvgi_bullish_cross": self.rvgi_bullish_cross,
            "box_high": self.box_high,
            "box_low": self.box_low,
            "box_mid": self.box_mid,
            "box_width_pct": self.box_width_pct,
            "box_position": self.box_position,
            "box_context_timestamp": self.box_context_timestamp,
            "box_previous_close": self.box_previous_close,
            "volume_confirmed": self.volume_confirmed,
            "residual_return_short_bps": self.residual_return_short_bps,
            "residual_return_long_bps": self.residual_return_long_bps,
            "market_beta": self.market_beta,
            "sector_beta": self.sector_beta,
            "foreign_flow_zscore": self.foreign_flow_zscore,
            "institution_flow_zscore": self.institution_flow_zscore,
            "anchored_vwap": self.anchored_vwap,
            "anchor_basis": self.anchor_basis,
            "change_point_probability": self.change_point_probability,
            "flow_toxicity": self.flow_toxicity,
            "spread_percentile": self.spread_percentile,
            "opening_range_high": self.opening_range_high,
            "opening_range_low": self.opening_range_low,
            "opening_range_minutes": self.opening_range_minutes,
            "relative_volume": self.relative_volume,
            "first_half_hour_return_bps": self.first_half_hour_return_bps,
            "first_half_hour_volatility_percentile": self.first_half_hour_volatility_percentile,
            "in_last_continuous_half_hour": self.in_last_continuous_half_hour,
            "minutes_to_continuous_close": self.minutes_to_continuous_close,
            "borrow_available": self.borrow_available,
            "borrow_available_quantity": self.borrow_available_quantity,
            "borrow_fee_bps_annualised": self.borrow_fee_bps_annualised,
            "borrow_observed_at": self.borrow_observed_at,
            "short_sale_permitted": self.short_sale_permitted,
            "return_deadline": self.return_deadline,
            "short_interest_ratio": self.short_interest_ratio,
            "days_to_cover": self.days_to_cover,
            "residual_short_bps": self.residual_short_bps,
            "residual_long_bps": self.residual_long_bps,
            "breakdown_excess_bps": self.breakdown_excess_bps,
            "aggressor_imbalance": self.aggressor_imbalance,
            "market_alignment": self.market_alignment,
            "market_trend": self.market_trend,
            "market_breadth": self.market_breadth,
            "liquidity_score": self.liquidity_score,
            "spread_bps": self.spread_bps,
        }


@dataclass(frozen=True)
class AlgorithmDecision:
    """Mechanical entry verdict for one evaluation tick."""

    strategy_id: str
    triggered: bool
    score: float
    confidence: float
    expected_edge_bps: float
    horizon_seconds: int
    reason_codes: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "triggered": self.triggered,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "expected_edge_bps": round(self.expected_edge_bps, 3),
            "horizon_seconds": self.horizon_seconds,
            "reason_codes": list(self.reason_codes),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ExitRule:
    """Thesis-specific exit geometry, resolved at fill time."""

    strategy_id: str
    stop_price: float | None
    target_price: float | None
    trailing_bps: float | None
    max_holding_seconds: int
    stop_basis: str
    target_basis: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "trailing_bps": self.trailing_bps,
            "max_holding_seconds": self.max_holding_seconds,
            "stop_basis": self.stop_basis,
            "target_basis": self.target_basis,
        }


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def tick_expected_move_bps(
    realized_volatility_10s: float | None,
    horizon_seconds: int,
    *,
    window_seconds: int = 10,
    capture_fraction: float = 0.5,
) -> float:
    """Conservative favourable move over the horizon from the tick window.

    Sub-second analogue of the minute-bar estimator in ``signals.py``: scale the
    10-second realised volatility by sqrt(time) and keep a capture fraction
    below 1. Returns 0.0 when the proxy is missing — no fabricated floor.
    """
    if not realized_volatility_10s or realized_volatility_10s <= 0 or horizon_seconds <= 0:
        return 0.0
    scaled = realized_volatility_10s * math.sqrt(horizon_seconds / max(1, window_seconds))
    return max(0.0, capture_fraction * scaled * 10_000.0)


def _present(*values: float | None) -> bool:
    return all(value is not None for value in values)


# --------------------------------------------------------------------------- #
# Cost-derived entry floor                                                     #
# --------------------------------------------------------------------------- #
# The floor used to be a single constant, ``min_expected_edge_bps: 8``, applied
# to every algorithm in every market. Round-trip cost is ~28bp on KRX and
# ~51-70bp on US, so an 8bp floor let every algorithm fire on edges that the
# ProfitabilityGate then had to reject — the measured result being strategies
# with a POSITIVE gross edge and a deeply negative net one (gap_context: +12.1bp
# gross, -15.8bp net). The two layers were answering different questions with
# different arithmetic, and the algorithm layer's answer was structurally wrong.
#
# The floor is now derived from the same fee policy the gate charges:
#
#     required_edge = round_trip_cost * cost_multiple + min_net_buffer
#
# so a trigger means "this edge can survive its own costs" in whichever market
# the symbol trades. ``absolute_floor_bps`` remains as a lower bound for the case
# where the cost config is unreadable, which must not silently become "free".
_KRX_SYMBOL_LENGTH = 6


def _resolve_market(symbol: str) -> tuple[str, str]:
    """Map a symbol to (venue, instrument_type) for cost lookup.

    Same 6-digit rule the counterfactual evaluator and the account dashboard
    use; keeping one rule means the floor a trigger is held to and the cost the
    gate charges cannot disagree about which market a symbol is in.
    """
    normalized = str(symbol or "").upper().strip()
    if normalized.isdigit() and len(normalized) == _KRX_SYMBOL_LENGTH:
        return "KRX", "domestic_stock"
    return "NASD", "overseas_stock"


_cost_engine = None
_round_trip_cache: dict[tuple[str, str], float] = {}


def round_trip_cost_bps(symbol: str) -> float | None:
    """Round-trip cost in bps for this symbol's market, or None if unresolvable.

    Cached per market: the fee policy is a config read, and this runs inside the
    per-tick entry path of every algorithm.
    """
    global _cost_engine
    venue, instrument_type = _resolve_market(symbol)
    cached = _round_trip_cache.get((venue, instrument_type))
    if cached is not None:
        return cached
    try:
        if _cost_engine is None:
            from app.cost.trading_cost_engine import TradingCostEngine

            _cost_engine = TradingCostEngine()
        policy = _cost_engine.policy_for(venue=venue, instrument_type=instrument_type)
    except Exception:  # noqa: BLE001 - an unreadable cost config must not crash entry.
        return None
    total_rate = (
        policy.buy_fee_rate
        + policy.sell_fee_rate
        + policy.sell_tax_rate
        # Slippage is paid on both legs; spread and impact are charged once each
        # by the gate, so the floor charges them the same way.
        + 2.0 * policy.slippage_rate
        + policy.spread_rate
        + policy.market_impact_rate
        + policy.safety_margin_rate
    )
    cost_bps = max(0.0, total_rate * 10_000.0)
    _round_trip_cache[(venue, instrument_type)] = cost_bps
    return cost_bps


def reset_cost_floor_cache() -> None:
    """Drop the cached fee policies (tests, and config edits without a restart)."""
    global _cost_engine
    _cost_engine = None
    _round_trip_cache.clear()


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, dict[str, float]] = {
    "shared": {
        "capture_fraction": 0.5,
        "min_tick_count_5s": 3.0,
        # Absolute lower bound only. The operative floor is cost-derived (see
        # ``round_trip_cost_bps``); this value survives as the fallback for an
        # unreadable cost config, where "no cost data" must not read as "free".
        "min_expected_edge_bps": 8.0,
        # 1.0 = the edge must merely cover its own round trip. Above 1.0 demands
        # the edge exceed cost by that multiple, which is the honest posture when
        # both terms are estimates: equality is a loss once either is off.
        "cost_floor_multiple": 1.0,
        # Net bps that must remain after cost. This is what makes a trigger worth
        # taking rather than merely break-even.
        "min_net_buffer_bps": 10.0,
        # Escape hatch for replaying historical configs: 0 restores the old
        # constant floor. Not for making a strategy trade again — a strategy that
        # only fires below its cost floor has no edge to recover.
        "cost_aware_floor_enabled": 1.0,
    },
    "intraday_momentum": {
        "min_aggressor_imbalance": 0.15,
        "min_return_5s_bps": 2.0,
        "horizon_seconds": 180.0,
        "stop_volatility_multiple": 1.5,
        "trailing_bps": 12.0,
        "reverse_aggressor_exit": -0.25,
    },
    "breakout_volume": {
        "min_volume_spike_ratio": 1.5,
        "min_breakout_excess_bps": 2.0,
        "acceptance_return_5s_bps": 1.0,
        "min_aggressor_imbalance": 0.05,
        "horizon_seconds": 300.0,
        "stop_buffer_bps": 8.0,
        "trailing_bps": 15.0,
        "failure_tolerance_bps": 10.0,
    },
    "vwap_mean_reversion": {
        "entry_deviation_bps": 25.0,
        "max_entry_deviation_bps": 150.0,
        "max_entry_zscore": 4.0,
        "max_rsi": 38.0,
        "max_percent_b": 0.22,
        "min_recovery_return_5s_bps": 1.0,
        "min_aggressor_imbalance": 0.05,
        "min_book_improvement": 0.0,
        "horizon_seconds": 240.0,
        "max_horizon_seconds": 5400.0,
        "stop_volatility_multiple": 2.0,
        "target_capture_fraction": 0.7,
        "trailing_bps": 10.0,
    },
    "bar_confirmed_vwap_recovery": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        "min_displacement_bps": 75.0,
        "max_displacement_zscore": 15.0,
        "max_rsi": 45.0,
        "min_macd_histogram": 0.0,
        "min_momentum_persistence": 0.30,
        "min_liquidity_score": 0.40,
        "max_spread_bps": 25.0,
        "target_capture_fraction": 0.50,
        "stop_volatility_multiple": 2.5,
        "trailing_bps": 30.0,
        "horizon_seconds": 5400.0,
        # ``exit_rule`` reads this to size the holding clock from the structural
        # edge; without it every elected position raised KeyError. Equal to the
        # base horizon, so the resolved clock stays exactly what it was before the
        # knob existed — raising it is a deliberate tuning decision, not a repair.
        "max_horizon_seconds": 5400.0,
    },
    "liquidity_shock_reversal": {
        # The new ask-heavy absorption branch is validated on two US sessions
        # only.  Keep the whole arm shadow-only until forward outcomes provide
        # enough independent sessions for promotion.
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "live_authorized": 1.0,
        "shock_return_10s_bps": -40.0,
        "max_spread_change_5s": 0.0,
        "min_aggressor_imbalance": -0.35,
        "min_orderbook_imbalance": 0.0,
        "horizon_seconds": 120.0,
        "retrace_fraction": 0.4,
        "stop_buffer_bps": 6.0,
        "trailing_bps": 14.0,
        "absorption_us_only": 1.0,
        "max_absorption_orderbook_imbalance": -0.30,
        "min_absorption_return_30s_bps": 2.0,
        "max_absorption_spread_change_5s": 0.0,
        "absorption_capture_fraction": 0.35,
        "absorption_horizon_seconds": 600.0,
        "max_absorption_horizon_seconds": 3600.0,
    },
    "event_momentum": {
        "min_volume_spike_ratio": 2.0,
        "min_aggressor_imbalance": 0.10,
        "exhaustion_return_10s_bps": 120.0,
        "horizon_seconds": 420.0,
        "stop_volatility_multiple": 2.0,
        "trailing_bps": 20.0,
    },
    "cross_sectional_relative_strength": {
        "max_sector_rank": 3.0,
        "min_short_return_bps": 0.0,
        "min_aggressor_imbalance": 0.05,
        "horizon_seconds": 420.0,
        "stop_volatility_multiple": 2.0,
        "trailing_bps": 18.0,
    },
    "gap_context": {
        "min_gap_rate": 0.01,
        "min_volume_spike_ratio": 1.5,
        "min_aggressor_imbalance": 0.10,
        "horizon_seconds": 300.0,
        "fill_capture_fraction": 0.5,
        "stop_buffer_bps": 10.0,
        "trailing_bps": 16.0,
    },
    "rvgi_box_breakout": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        "rvgi_period": 10.0,
        "box_lookback": 20.0,
        "cross_confirm_bars": 3.0,
        "entry_buffer_bps": 3.0,
        "max_extension_bps": 35.0,
        "min_volume_spike_ratio": 1.5,
        "min_aggressor_imbalance": 0.05,
        "min_box_width_pct": 0.002,
        "max_box_width_pct": 0.04,
        "target_capture_fraction": 0.5,
        "max_target_return": 0.012,
        "stop_buffer_bps": 10.0,
        "stop_atr_multiple": 1.0,
        "failure_tolerance_bps": 12.0,
        "trailing_bps": 15.0,
        "horizon_seconds": 300.0,
    },
    # --- Added for the current high-volatility, flow-driven tape --------------
    # Deployment is enabled; common cost, trust, bandit and risk gates remain.
    "residual_relative_strength": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        "min_residual_short_bps": 5.0,
        "min_residual_long_bps": 0.0,
        "max_sector_rank": 3.0,
        "min_relative_volume": 1.2,
        "min_aggressor_imbalance": 0.05,
        "min_microprice_edge_bps": 0.0,
        "require_flow_confirmation": 1.0,
        "min_flow_zscore": 0.0,
        "max_change_point_probability": 0.5,
        "horizon_seconds": 420.0,
        "stop_volatility_multiple": 2.0,
        "trailing_bps": 20.0,
    },
    "adaptive_anchored_vwap_reversion": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        "min_entry_zscore": 2.0,
        "max_entry_zscore": 6.0,
        "min_displacement_bps": 15.0,
        "max_spread_change_5s": 0.0,
        "min_microprice_edge_bps": 0.0,
        "max_change_point_probability": 0.4,
        "target_capture_fraction": 0.6,
        "max_target_bps": 150.0,
        "stop_zscore_multiple": 1.5,
        "trailing_bps": 16.0,
        "horizon_seconds": 300.0,
    },
    "ofi_microprice_exhaustion_reversal": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        "shock_return_10s_bps": -35.0,
        "min_depth_ratio": 1.1,
        "min_orderbook_imbalance": 0.05,
        "min_ofi_slope": 0.0,
        "min_microprice_edge_bps": 0.2,
        "max_spread_change_5s": 0.0,
        "min_aggressor_imbalance": -0.25,
        "max_flow_toxicity": 0.7,
        "max_change_point_probability": 0.4,
        "retrace_fraction": 0.35,
        "stop_buffer_bps": 6.0,
        "trailing_bps": 14.0,
        "horizon_seconds": 150.0,
    },
    "market_intraday_momentum": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        # Unvalidated locally: only 2 of 360 stored symbol-days carry BOTH the first
        # and the last continuous half-hour, so this cannot yet be measured here. It
        # starts shadow-only and must earn promotion from realized outcomes, like
        # every other new algorithm in this module.
        "live_authorized": 0.0,
        # Minimum first half-hour return to call the day "up". Below this the signal
        # is noise, and the published slope is estimated on meaningful moves.
        "min_first_half_hour_return_bps": 15.0,
        # The effect concentrates on volatile days, and only a volatile day travels
        # far enough to clear a ~33bps KRX round trip. Percentile of the symbol's own
        # first-half-hour range history.
        "min_first_half_hour_volatility_percentile": 0.6,
        "min_aggressor_imbalance": 0.0,
        "max_change_point_probability": 0.5,
        "stop_buffer_bps": 8.0,
        "trailing_bps": 30.0,
        # 14:50 -> 15:15, flat before the 15:20 closing auction.
        "horizon_seconds": 1500.0,
    },
    "overnight_gap_carry": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        # Direction is unvalidated: the local sample measures the overnight MOVE
        # (median 69.1bps, above the 51.2bps round trip) but not its sign, and the
        # unconditional mean is +15.6bps, i.e. a loser before gating. Promotion has
        # to come from forward outcomes, not from this file.
        "live_authorized": 0.0,
        # Variance-equivalent trading minutes carried by an overnight gap.
        # Calibrated from the stored US sample: median |overnight| 69.1bps against
        # median |30-minute| 16.4bps is 4.21x, and 4.21^2 x 30 = 532.
        "overnight_variance_minutes": 532.0,
        # Only near the close: the carry is a decision about the closing print.
        "max_minutes_to_close": 20.0,
        # Closing above the session VWAP is the observable form of "buyers ended
        # the day in control". Zero is the neutral line, so this asks for a real
        # premium rather than a rounding error.
        "min_vwap_premium_bps": 10.0,
        "min_momentum_persistence": 0.55,
        "min_aggressor_imbalance": 0.10,
        "max_change_point_probability": 0.5,
        # An overnight gap jumps rather than fills, so a stop inside the typical
        # gap is not a stop; it is a worse fill on a move that already happened.
        "stop_volatility_multiple": 3.0,
        "trailing_bps": 45.0,
        # 15:45 New York to the next open plus settling room.
        "horizon_seconds": 64800.0,
    },
    "opening_range_breakout": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        # Live authorisation is withheld until this strategy has its own realized
        # history, exactly like every other newly added algorithm here.
        "live_authorized": 1.0,
        # "Stocks in play": relative volume is the load-bearing filter in the
        # published result, not a nicety. Practitioner studies place the useful
        # threshold at 1.5-2.0x average volume; 1.5 is the permissive end.
        "min_relative_volume": 1.5,
        # How far past the opening-range high counts as a real break rather than a
        # wick, expressed in bps of the range width.
        "min_breakout_excess_bps": 3.0,
        # Order flow must agree with the break; a breakout into selling is the
        # classic false break.
        "min_aggressor_imbalance": 0.05,
        # A break with a widening spread is being sold into.
        "max_spread_change_5s": 0.0,
        "max_change_point_probability": 0.4,
        # Stop sits below the opening-range LOW (the published rule uses the
        # opposite end of the range), with a small buffer for noise.
        "stop_buffer_bps": 8.0,
        "trailing_bps": 30.0,
        # A day-long thesis, not a scalp.
        "horizon_seconds": 7200.0,
    },
    # --- SHORT theses -------------------------------------------------------- #
    # Deployment flags are enabled. Production also carries an explicit audited
    # operator override in short_strategy_deployment.yaml; borrow and short-risk
    # checks remain independent hard gates.
    #
    # Thresholds are the mirror of each long counterpart's, with the direction of
    # every comparison reversed by the algorithm rather than by negating the number,
    # so a reader can diff the two tables. Borrow-specific knobs have no long analogue.
    "market_intraday_momentum_short": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        # Magnitude, applied to a NEGATIVE first half-hour return. Slightly wider
        # than the long side's 15bps because the borrow leg raises the cost floor.
        "min_first_half_hour_drop_bps": 20.0,
        "min_first_half_hour_volatility_percentile": 0.6,
        # Sell-side aggression required (negative imbalance).
        "max_aggressor_imbalance": 0.0,
        "max_change_point_probability": 0.5,
        # Shorting into a rising broad market is fighting the tape; the market leg
        # must at least not be strongly up.
        "max_market_breadth": 0.55,
        "max_borrow_fee_bps_annualised": 1500.0,
        "min_borrow_quantity": 1.0,
        "max_days_to_cover": 5.0,
        "max_short_interest_ratio": 0.15,
        "min_liquidity_score": 0.45,
        "max_spread_bps": 40.0,
        "stop_buffer_bps": 8.0,
        "trailing_bps": 30.0,
        "horizon_seconds": 1500.0,
    },
    "opening_range_breakdown": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        "min_relative_volume": 1.5,
        # Magnitude below the opening-range LOW, in bps of range width.
        "min_breakdown_excess_bps": 3.0,
        "max_aggressor_imbalance": -0.05,
        # A breakdown with a widening spread is being bid, not sold.
        "max_spread_change_5s": 0.0,
        "max_change_point_probability": 0.4,
        "max_market_breadth": 0.6,
        "max_borrow_fee_bps_annualised": 1500.0,
        "min_borrow_quantity": 1.0,
        "max_days_to_cover": 5.0,
        "max_short_interest_ratio": 0.15,
        "min_liquidity_score": 0.45,
        "max_spread_bps": 40.0,
        "stop_buffer_bps": 8.0,
        "trailing_bps": 30.0,
        "horizon_seconds": 3600.0,
    },
    "residual_relative_weakness": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 1.0,
        # Magnitudes applied to NEGATIVE residuals.
        "min_residual_short_weakness_bps": 5.0,
        "min_residual_long_weakness_bps": 0.0,
        # Rank counted from the WEAK end of the sector.
        "max_sector_weakness_rank": 3.0,
        "min_relative_volume": 1.2,
        "max_aggressor_imbalance": -0.05,
        "max_microprice_edge_bps": 0.0,
        "require_flow_confirmation": 1.0,
        # Distribution, not accumulation: flow must be NEGATIVE.
        "max_flow_zscore": 0.0,
        "max_change_point_probability": 0.5,
        "max_borrow_fee_bps_annualised": 1500.0,
        "min_borrow_quantity": 1.0,
        "max_days_to_cover": 5.0,
        "max_short_interest_ratio": 0.15,
        "min_liquidity_score": 0.45,
        "max_spread_bps": 40.0,
        "horizon_seconds": 2700.0,
        "stop_volatility_multiple": 2.0,
        "trailing_bps": 32.0,
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # local import: optional dependency at import time
    except ImportError:  # pragma: no cover - yaml ships with the runtime
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


class AlgorithmConfig:
    """Per-strategy thresholds: built-in default < YAML < environment.

    Environment key format ``ALGO_<STRATEGY>_<PARAM>`` in upper case, e.g.
    ``ALGO_LIQUIDITY_SHOCK_REVERSAL_SHOCK_RETURN_10S_BPS``. Effective values are
    exposed through :meth:`as_dict` so the resolved policy is auditable.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        overlay = _load_yaml(Path(config_path))
        resolved: dict[str, dict[str, float]] = {}
        for section, defaults in _DEFAULTS.items():
            values = dict(defaults)
            section_overlay = overlay.get(section)
            if isinstance(section_overlay, Mapping):
                for key, value in section_overlay.items():
                    if key in values:
                        try:
                            values[key] = float(value)
                        except (TypeError, ValueError):
                            continue
            for key in list(values):
                env_key = f"ALGO_{section}_{key}".upper()
                raw = os.getenv(env_key)
                if raw is None:
                    continue
                try:
                    values[key] = float(raw)
                except (TypeError, ValueError):
                    continue
            resolved[section] = values
        self._values = resolved

    def get(self, section: str, key: str) -> float:
        return float(self._values.get(section, {}).get(key, _DEFAULTS[section][key]))

    def shared(self, key: str) -> float:
        return self.get("shared", key)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {section: dict(values) for section, values in self._values.items()}


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #
class TradingAlgorithm:
    strategy_id = "base"
    thesis = "base"
    # Which way the exposure this algorithm opens points. LONG for everything that
    # existed before shorts, so no subclass needs to restate it.
    direction = "LONG"

    def __init__(self, config: AlgorithmConfig | None = None) -> None:
        self.config = config or AlgorithmConfig()

    # -- knobs ------------------------------------------------------------- #
    def p(self, key: str) -> float:
        return self.config.get(self.strategy_id, key)

    @property
    def horizon_seconds(self) -> int:
        return int(self.p("horizon_seconds"))

    # -- API --------------------------------------------------------------- #
    def entry(self, features: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        """Mechanical trigger. Must not consult regime/liquidity admissibility."""
        raise NotImplementedError

    def exit_rule(
        self,
        entry_price: float,
        features: TechnicalFeatureSet,
        context: ElectionContext,
    ) -> ExitRule:
        raise NotImplementedError

    def invalidation(
        self,
        features: TechnicalFeatureSet,
        context: ElectionContext,
        *,
        entry_price: float | None = None,
    ) -> tuple[str, ...]:
        """Thesis-specific invalidation reasons (empty == thesis intact)."""
        return ()

    # -- shared building blocks -------------------------------------------- #
    def _tick_ready(self, f: TechnicalFeatureSet) -> tuple[bool, tuple[str, ...]]:
        if not f.tick_data_ready:
            return False, ("TICK_WINDOW_NOT_READY",)
        if (f.tick_count_5s or 0.0) < self.config.shared("min_tick_count_5s"):
            return False, ("TICK_COUNT_TOO_LOW",)
        return True, ()

    def _volatility_edge(self, f: TechnicalFeatureSet) -> float:
        volatility = f.realized_volatility_10s
        window_seconds = 10
        if not volatility or volatility <= 0:
            # Sparse but genuine websocket prints can still resolve direction and
            # price continuation with two events. Expected move then comes from
            # completed one-minute history, never from a fabricated minimum.
            volatility = f.realized_volatility
            window_seconds = 60
        return tick_expected_move_bps(
            volatility,
            self.horizon_seconds,
            window_seconds=window_seconds,
            capture_fraction=self.config.shared("capture_fraction"),
        )

    @staticmethod
    def _horizon_for_structural_edge(
        f: TechnicalFeatureSet,
        edge_bps: float,
        base_horizon_seconds: int,
        maximum_horizon_seconds: int,
    ) -> int:
        """Estimate a feasible holding clock from completed-bar volatility."""

        base = max(1, int(base_horizon_seconds))
        maximum = max(base, int(maximum_horizon_seconds))
        minute_vol_bps = max(0.0, float(f.realized_volatility or 0.0) * 10_000.0)
        if minute_vol_bps <= 0 or edge_bps <= 0:
            return base
        estimated = 60.0 * (edge_bps / minute_vol_bps) ** 2
        return max(base, min(maximum, int(math.ceil(estimated))))

    def _cost_feasible_volatility_edge(
        self,
        f: TechnicalFeatureSet,
        *,
        base_horizon_seconds: int,
        maximum_horizon_seconds: int,
        capture_fraction: float,
    ) -> tuple[float, int]:
        """Find the shortest completed-bar volatility horizon clearing costs."""

        minimum, _ = self.entry_floor_bps(f.symbol)
        minute_volatility = max(0.0, float(f.realized_volatility or 0.0))
        captured_minute_bps = minute_volatility * 10_000.0 * max(0.0, capture_fraction)
        base = max(1, int(base_horizon_seconds))
        maximum = max(base, int(maximum_horizon_seconds))
        if captured_minute_bps <= 0:
            return 0.0, base
        required_seconds = 60.0 * (minimum / captured_minute_bps) ** 2
        horizon = max(base, min(maximum, int(math.ceil(required_seconds))))
        edge = tick_expected_move_bps(
            minute_volatility,
            horizon,
            window_seconds=60,
            capture_fraction=capture_fraction,
        )
        return edge, horizon

    def _reject(self, reasons: tuple[str, ...], **diagnostics: Any) -> AlgorithmDecision:
        return AlgorithmDecision(
            strategy_id=self.strategy_id,
            triggered=False,
            score=0.0,
            confidence=0.0,
            expected_edge_bps=0.0,
            horizon_seconds=self.horizon_seconds,
            reason_codes=tuple(dict.fromkeys(reasons)),
            diagnostics=diagnostics,
        )

    def entry_floor_bps(self, symbol: str) -> tuple[float, dict[str, Any]]:
        """Minimum expected edge this algorithm may fire on, and why.

        Cost-derived rather than constant, so the same trigger is held to a
        ~28bp bar on KRX and a ~51bp+ bar on US instead of 8bp on both.
        """
        absolute = self.config.shared("min_expected_edge_bps")
        if self.config.shared("cost_aware_floor_enabled") <= 0.0:
            return absolute, {"floor_basis": "absolute_only"}
        cost_bps = round_trip_cost_bps(symbol)
        if cost_bps is None:
            return absolute, {"floor_basis": "cost_config_unreadable"}
        multiple = max(0.0, self.config.shared("cost_floor_multiple"))
        buffer_bps = max(0.0, self.config.shared("min_net_buffer_bps"))
        required = cost_bps * multiple + buffer_bps
        venue, instrument_type = _resolve_market(symbol)
        return max(absolute, required), {
            "floor_basis": "round_trip_cost",
            "venue": venue,
            "instrument_type": instrument_type,
            "round_trip_cost_bps": round(cost_bps, 3),
            "cost_floor_multiple": multiple,
            "min_net_buffer_bps": buffer_bps,
        }

    def _fire(
        self,
        *,
        score: float,
        confidence: float,
        edge_bps: float,
        reasons: tuple[str, ...],
        horizon_seconds: int | None = None,
        symbol: str = "",
        **diagnostics: Any,
    ) -> AlgorithmDecision:
        minimum, floor_diagnostics = self.entry_floor_bps(symbol)
        if edge_bps < minimum:
            # Distinct reason code from the old constant-floor rejection: an edge
            # that cannot cover its market's costs is a different diagnosis from
            # one below an arbitrary threshold, and the dashboard has to be able
            # to tell an operator which of the two happened.
            below_cost = floor_diagnostics.get("floor_basis") == "round_trip_cost"
            return self._reject(
                (
                    *reasons,
                    rc.TECHNICAL_EDGE_NON_POSITIVE,
                    "EDGE_BELOW_COST_FLOOR" if below_cost else "EDGE_BELOW_ALGORITHM_FLOOR",
                ),
                expected_edge_bps=round(edge_bps, 3),
                minimum_edge_bps=round(minimum, 3),
                **floor_diagnostics,
                **diagnostics,
            )
        return AlgorithmDecision(
            strategy_id=self.strategy_id,
            triggered=True,
            score=_clamp(score),
            confidence=_clamp(confidence),
            expected_edge_bps=edge_bps,
            horizon_seconds=(
                max(1, int(horizon_seconds))
                if horizon_seconds is not None
                else self.horizon_seconds
            ),
            reason_codes=tuple(dict.fromkeys(reasons)),
            diagnostics=diagnostics,
        )

    def _volatility_stop(self, entry_price: float, f: TechnicalFeatureSet, multiple: float) -> float | None:
        volatility = f.realized_volatility_10s or f.realized_volatility
        if not volatility or volatility <= 0 or entry_price <= 0:
            return None
        return entry_price * (1.0 - multiple * volatility)


# --------------------------------------------------------------------------- #
# 1. Intraday momentum — tick order-flow continuation                          #
# --------------------------------------------------------------------------- #
class IntradayMomentumAlgorithm(TradingAlgorithm):
    strategy_id = "intraday_momentum"
    thesis = "sub-second buy-side order flow continues while the bar trend agrees"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        if not _present(f.return_5s, f.aggressor_imbalance_5s):
            return self._reject(("MOMENTUM_TICK_INPUTS_MISSING",))

        return_5s_bps = (f.return_5s or 0.0) * 10_000.0
        aggressor = f.aggressor_imbalance_5s or 0.0
        if return_5s_bps < self.p("min_return_5s_bps"):
            return self._reject(("MOMENTUM_TICK_RETURN_TOO_WEAK",), return_5s_bps=return_5s_bps)
        if aggressor < self.p("min_aggressor_imbalance"):
            return self._reject((rc.MOMENTUM_WEAKENED, "AGGRESSOR_FLOW_NOT_BUY_SIDE"), aggressor=aggressor)
        # Bar context is direction agreement only, not an admissibility gate.
        if f.macd_histogram is not None and f.macd_histogram < 0:
            return self._reject((rc.MOMENTUM_WEAKENED, "BAR_TREND_DISAGREES"))
        if f.ema_fast is not None and f.ema_slow is not None and f.ema_fast < f.ema_slow:
            return self._reject((rc.MOMENTUM_WEAKENED, "BAR_TREND_DISAGREES"))

        edge = self._volatility_edge(f)
        score = _clamp(0.5 * _clamp(aggressor) + 0.5 * _clamp(return_5s_bps / 20.0))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(0.4 + 0.6 * _clamp(aggressor)),
            edge_bps=edge,
            reasons=(rc.MOMENTUM_CONFIRMED, "TICK_ORDER_FLOW_CONTINUATION"),
            return_5s_bps=round(return_5s_bps, 3),
            aggressor_imbalance_5s=round(aggressor, 4),
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(entry_price, f, self.p("stop_volatility_multiple")),
            target_price=entry_price * (1.0 + self._volatility_edge(f) / 10_000.0),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="tick_volatility_multiple",
            target_basis="tick_volatility_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        if (f.aggressor_imbalance_5s or 0.0) <= self.p("reverse_aggressor_exit"):
            codes.append("AGGRESSOR_FLOW_REVERSED")
        if f.macd_histogram is not None and f.macd_histogram < 0 and (f.return_5s or 0.0) < 0:
            codes.append(rc.MOMENTUM_WEAKENED)
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 2. Breakout volume — structural level, acceptance confirmed on ticks         #
# --------------------------------------------------------------------------- #
class BreakoutVolumeAlgorithm(TradingAlgorithm):
    strategy_id = "breakout_volume"
    thesis = "price accepted above a causally known range high on abnormal volume"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        if f.breakout_strength is None or f.donchian_high is None:
            return self._reject(("BREAKOUT_LEVEL_UNAVAILABLE",))
        breakout_excess_bps = float(f.breakout_strength) * 10_000.0
        if breakout_excess_bps < self.p("min_breakout_excess_bps"):
            return self._reject(("PRICE_BELOW_BREAKOUT_LEVEL",), breakout_strength=f.breakout_strength)
        volume_ratio = f.volume_spike_ratio
        if volume_ratio is None or volume_ratio < self.p("min_volume_spike_ratio"):
            return self._reject((rc.VOLUME_CONFIRMATION_MISSING,), volume_spike_ratio=volume_ratio)
        # Acceptance: the break must not be fading back on the very next ticks.
        # One sparse print used to become a synthetic 0bp one-second return and
        # pass a >= 0 gate.  A breakout is accepted only when the fully populated
        # five-second window has actually advanced beyond the broken level.
        if f.return_5s is None:
            return self._reject(("BREAKOUT_ACCEPTANCE_WINDOW_MISSING",))
        return_5s_bps = float(f.return_5s) * 10_000.0
        aggressor = f.aggressor_imbalance_5s
        if return_5s_bps < self.p("acceptance_return_5s_bps"):
            return self._reject(
                (rc.FALSE_BREAKOUT_RISK_HIGH, "BREAKOUT_FADING_ON_TICKS"),
                return_5s_bps=round(return_5s_bps, 3),
            )
        if aggressor is None or aggressor < self.p("min_aggressor_imbalance"):
            return self._reject((rc.FALSE_BREAKOUT_RISK_HIGH, "BREAKOUT_NOT_FLOW_CONFIRMED"))

        edge = self._volatility_edge(f)
        score = _clamp(0.5 * _clamp(volume_ratio / 3.0) + 0.5 * _clamp(aggressor))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(0.35 + 0.4 * _clamp(volume_ratio / 3.0) + 0.25 * _clamp(aggressor)),
            edge_bps=edge,
            reasons=(rc.BREAKOUT_CONFIRMED, "BREAKOUT_ACCEPTED_ON_TICKS"),
            donchian_high=f.donchian_high,
            breakout_excess_bps=round(breakout_excess_bps, 3),
            return_5s_bps=round(return_5s_bps, 3),
            volume_spike_ratio=volume_ratio,
            aggressor_imbalance_5s=round(aggressor, 4),
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        # Structural stop: back inside the broken range invalidates the thesis.
        level = f.donchian_high or entry_price
        stop = level * (1.0 - self.p("stop_buffer_bps") / 10_000.0)
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=min(stop, entry_price) if entry_price > 0 else stop,
            target_price=entry_price * (1.0 + self._volatility_edge(f) / 10_000.0),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="broken_range_high_minus_buffer",
            target_basis="tick_volatility_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        level = f.donchian_high
        if level and f.price:
            tolerance = level * (1.0 - self.p("failure_tolerance_bps") / 10_000.0)
            if f.price < tolerance:
                codes.append("BREAKOUT_FAILED_BACK_INSIDE_RANGE")
        if f.volume_spike_ratio is not None and f.volume_spike_ratio < 1.0:
            codes.append("BREAKOUT_VOLUME_EXPANSION_LOST")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 3. VWAP mean reversion — structural target, stabilisation confirmed on ticks #
# --------------------------------------------------------------------------- #
class VwapMeanReversionAlgorithm(TradingAlgorithm):
    strategy_id = "vwap_mean_reversion"
    thesis = "displacement below VWAP reverts once selling stops worsening"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        deviation = f.vwap_distance_bps
        if deviation is None or f.vwap is None:
            return self._reject(("VWAP_UNAVAILABLE",))
        if deviation > -self.p("entry_deviation_bps"):
            return self._reject(("VWAP_DISPLACEMENT_TOO_SMALL",), vwap_distance_bps=deviation)
        # A very deep displacement is more often repricing than stationary mean
        # reversion.  Bound it in both raw bps and volatility-normalised units.
        # The adaptive VWAP strategy owns the wider, explicitly normalised case.
        if abs(deviation) > self.p("max_entry_deviation_bps"):
            return self._reject(
                ("VWAP_DISPLACEMENT_STRUCTURAL_REPRICING",),
                vwap_distance_bps=round(deviation, 3),
            )
        zscore = f.vwap_zscore
        if zscore is not None and abs(zscore) > self.p("max_entry_zscore"):
            return self._reject(
                ("VWAP_DISPLACEMENT_ZSCORE_TOO_EXTREME",),
                vwap_zscore=round(zscore, 3),
            )
        oversold = (f.rsi is not None and f.rsi <= self.p("max_rsi")) or (
            f.bb_percent_b is not None and f.bb_percent_b <= self.p("max_percent_b")
        )
        if not oversold:
            return self._reject(("DISPLACEMENT_NOT_OVERSOLD",), rsi=f.rsi, percent_b=f.bb_percent_b)
        # Stabilisation must be observed, not inferred from a missing one-second
        # window becoming zero. Require a positive five-second recovery, buy-side
        # aggressor flow, and a non-worsening book.
        if f.return_5s is None or f.aggressor_imbalance_5s is None:
            return self._reject(("VWAP_RECOVERY_WINDOW_MISSING",))
        recovery_bps = float(f.return_5s) * 10_000.0
        aggressor = float(f.aggressor_imbalance_5s)
        book_change = f.orderbook_imbalance_change_5s
        if recovery_bps < self.p("min_recovery_return_5s_bps"):
            return self._reject(
                ("STILL_FALLING_ON_TICKS",), recovery_return_5s_bps=round(recovery_bps, 3)
            )
        if aggressor < self.p("min_aggressor_imbalance"):
            return self._reject(
                ("VWAP_RECOVERY_NOT_BUY_FLOW_CONFIRMED",), aggressor_imbalance_5s=round(aggressor, 4)
            )
        if book_change is None or book_change < self.p("min_book_improvement"):
            return self._reject(
                ("SELL_PRESSURE_STILL_WORSENING",), orderbook_imbalance_change_5s=book_change
            )

        # Target is structural (VWAP), so the edge is the captured distance.
        edge = abs(deviation) * self.p("target_capture_fraction")
        horizon = self._horizon_for_structural_edge(
            f,
            edge,
            self.horizon_seconds,
            int(self.p("max_horizon_seconds")),
        )
        score = _clamp(abs(deviation) / (2.0 * self.p("entry_deviation_bps")))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(0.35 + 0.45 * score),
            edge_bps=edge,
            horizon_seconds=horizon,
            reasons=(rc.MEAN_REVERSION_CANDIDATE, "VWAP_DISPLACEMENT_STABILISED"),
            vwap=f.vwap,
            vwap_distance_bps=round(deviation, 3),
            vwap_zscore=round(zscore, 3) if zscore is not None else None,
            recovery_return_5s_bps=round(recovery_bps, 3),
            aggressor_imbalance_5s=round(aggressor, 4),
            estimated_holding_seconds=horizon,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        vwap = f.vwap
        capture = self.p("target_capture_fraction")
        target = (
            entry_price + capture * (vwap - entry_price)
            if vwap and vwap > entry_price
            else None
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(entry_price, f, self.p("stop_volatility_multiple")),
            target_price=target,
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="tick_volatility_multiple",
            target_basis="partial_vwap_reversion",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        # Never average down: a deeper displacement is an exit signal, not a top-up.
        if (f.return_5s or 0.0) < 0 and (f.orderbook_imbalance_change_5s or 0.0) < 0:
            codes.append("REVERSION_FAILED_NEW_LOW")
        if f.ema_fast is not None and f.ema_slow is not None and f.ema_fast < f.ema_slow:
            if (f.macd_histogram or 0.0) < 0:
                codes.append(rc.MEAN_REVERSION_BLOCKED_BY_DOWNTREND)
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 4. Liquidity shock reversal — spread normalisation after a mechanical drop   #
# --------------------------------------------------------------------------- #
# First-class completed-bar VWAP recovery strategy                            #
class BarConfirmedVwapRecoveryAlgorithm(TradingAlgorithm):
    """Recover a deep VWAP dislocation only after a completed-bar turn.

    This first-class strategy uses completed one-minute bars because sparse
    premarket names often have a current book but too few prints for a causal
    sub-second window. It remains SHADOW-only until its own forward outcomes
    support promotion.
    """

    strategy_id = "bar_confirmed_vwap_recovery"
    thesis = "a deep VWAP dislocation recovers after the completed one-minute trend turns"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        required = (
            f.price,
            f.vwap,
            f.vwap_distance_bps,
            f.ema_fast,
            f.macd_histogram,
            f.rsi,
            f.momentum_persistence,
            f.realized_volatility,
            f.liquidity_score,
            f.spread_bps,
        )
        if not _present(*required):
            return self._reject(("BAR_VWAP_RECOVERY_INPUTS_MISSING",))

        displacement = float(f.vwap_distance_bps)
        if displacement > -self.p("min_displacement_bps"):
            return self._reject(
                ("BAR_VWAP_DISPLACEMENT_TOO_SMALL",),
                vwap_distance_bps=round(displacement, 3),
            )
        zscore = f.vwap_zscore
        if zscore is None:
            return self._reject(("BAR_VWAP_VOLATILITY_SCALE_MISSING",))
        if abs(zscore) > self.p("max_displacement_zscore"):
            return self._reject(
                ("BAR_VWAP_DISLOCATION_TOO_EXTREME",),
                vwap_zscore=round(zscore, 3),
            )
        if float(f.rsi) > self.p("max_rsi"):
            return self._reject(("BAR_VWAP_NOT_OVERSOLD",), rsi=round(float(f.rsi), 3))

        # The completed-bar turn is the entry clock. No 1s/5s return, tick count,
        # aggressor flow, or order-book delta participates in this decision.
        if float(f.price) < float(f.ema_fast):
            return self._reject(
                ("BAR_VWAP_FAST_EMA_NOT_RECLAIMED",),
                price=round(float(f.price), 6),
                ema_fast=round(float(f.ema_fast), 6),
            )
        if float(f.macd_histogram) <= self.p("min_macd_histogram"):
            return self._reject(
                ("BAR_VWAP_MACD_NOT_TURNED",),
                macd_histogram=round(float(f.macd_histogram), 8),
            )
        if float(f.momentum_persistence) < self.p("min_momentum_persistence"):
            return self._reject(
                ("BAR_VWAP_RECOVERY_NOT_PERSISTENT",),
                momentum_persistence=round(float(f.momentum_persistence), 3),
            )
        if float(f.liquidity_score) < self.p("min_liquidity_score"):
            return self._reject(
                ("BAR_VWAP_LIQUIDITY_TOO_LOW",),
                liquidity_score=round(float(f.liquidity_score), 3),
            )
        if float(f.spread_bps) > self.p("max_spread_bps"):
            return self._reject(
                ("BAR_VWAP_SPREAD_TOO_WIDE",),
                spread_bps=round(float(f.spread_bps), 3),
            )

        edge = abs(displacement) * self.p("target_capture_fraction")
        score = _clamp(abs(zscore) / self.p("max_displacement_zscore"))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(
                0.30
                + 0.30 * score
                + 0.20 * float(f.momentum_persistence)
                + 0.20 * float(f.liquidity_score)
            ),
            edge_bps=edge,
            reasons=("BAR_CONFIRMED_VWAP_RECOVERY", "COMPLETED_MINUTE_TREND_TURNED"),
            vwap_distance_bps=round(displacement, 3),
            vwap_zscore=round(zscore, 3),
            target_capture_fraction=self.p("target_capture_fraction"),
            price=round(float(f.price), 6),
            ema_fast=round(float(f.ema_fast), 6),
            macd_histogram=round(float(f.macd_histogram), 8),
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        target = (
            entry_price
            + self.p("target_capture_fraction") * (float(f.vwap) - entry_price)
            if f.vwap is not None and f.vwap > entry_price
            else None
        )
        # Same capture fraction the entry sized its edge with; ``exit_rule`` had no
        # such local and raised NameError for every elected position.
        edge = abs(float(f.vwap_distance_bps or 0.0)) * self.p("target_capture_fraction")
        horizon = self._horizon_for_structural_edge(
            f,
            edge,
            self.horizon_seconds,
            int(self.p("max_horizon_seconds")),
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(
                entry_price, f, self.p("stop_volatility_multiple")
            ),
            target_price=target,
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=horizon,
            stop_basis="completed_bar_volatility_multiple",
            target_basis="partial_recovery_to_session_vwap",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        if (
            f.price is not None
            and f.ema_fast is not None
            and f.price < f.ema_fast
            and (f.macd_histogram or 0.0) < 0
        ):
            return ("BAR_VWAP_RECOVERY_FAILED",)
        return ()


# --------------------------------------------------------------------------- #
# Liquidity shock reversal - normalisation after a mechanical drop            #
# --------------------------------------------------------------------------- #
class LiquidityShockReversalAlgorithm(TradingAlgorithm):
    strategy_id = "liquidity_shock_reversal"
    thesis = "a mechanical sub-minute drop partially retraces once spread and depth normalise"

    def _absorption_mode(self, f: TechnicalFeatureSet) -> bool:
        """Ask-heavy book whose price has stopped following the displayed supply.

        This is deliberately not "negative imbalance means buy".  The price must
        already have recovered over 30 seconds and the spread must be contracting,
        which is the observable decoupling that distinguishes absorption from a
        still-toxic sell wave.
        """

        if self.p("absorption_us_only") >= 1.0:
            symbol = str(f.symbol or "").strip().upper()
            if symbol.isdigit() and len(symbol) == 6:
                return False
        if not _present(
            f.return_30s,
            f.spread_change_5s,
            f.orderbook_imbalance,
        ):
            return False
        return (
            (f.return_30s or 0.0) * 10_000.0
            >= self.p("min_absorption_return_30s_bps")
            and (f.orderbook_imbalance or 0.0)
            <= self.p("max_absorption_orderbook_imbalance")
            and (f.spread_change_5s or 0.0)
            <= self.p("max_absorption_spread_change_5s")
        )

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        # US books are REST-polled, so they cannot honestly satisfy the 5-second
        # streamed-tick readiness gate used by the original shock branch.  The
        # absorption thesis has its own complete 30-second/book/spread inputs;
        # evaluate that closed set first.  The original sub-second shock thesis
        # still fails closed unless the streamed tick window is ready.
        if self._absorption_mode(f):
            edge, horizon = self._cost_feasible_volatility_edge(
                f,
                base_horizon_seconds=int(self.p("absorption_horizon_seconds")),
                maximum_horizon_seconds=int(self.p("max_absorption_horizon_seconds")),
                capture_fraction=self.p("absorption_capture_fraction"),
            )
            imbalance = abs(float(f.orderbook_imbalance or 0.0))
            recovery_bps = float(f.return_30s or 0.0) * 10_000.0
            return self._fire(
                symbol=f.symbol,
                score=_clamp(0.55 * imbalance + 0.45 * _clamp(recovery_bps / 20.0)),
                confidence=_clamp(0.35 + 0.5 * imbalance),
                edge_bps=edge,
                horizon_seconds=horizon,
                reasons=("ASK_HEAVY_ABSORPTION_CONFIRMED", "THIRTY_SECOND_PRICE_RECOVERY"),
                return_30s_bps=round(recovery_bps, 3),
                spread_change_5s=f.spread_change_5s,
                orderbook_imbalance=f.orderbook_imbalance,
                validation_scope="US_LIVE_COST_FEASIBLE_HORIZON",
                cost_feasible_horizon_seconds=horizon,
            )
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        if not _present(f.return_10s, f.spread_change_5s, f.orderbook_imbalance):
            return self._reject(("SHOCK_TICK_INPUTS_MISSING",))
        shock_bps = (f.return_10s or 0.0) * 10_000.0
        if shock_bps > self.p("shock_return_10s_bps"):
            return self._reject(("NO_LIQUIDITY_SHOCK_DETECTED",), return_10s_bps=round(shock_bps, 3))
        # Spread must be contracting from its shock peak, not still widening.
        if (f.spread_change_5s or 0.0) > self.p("max_spread_change_5s"):
            return self._reject(("SPREAD_STILL_WIDENING",), spread_change_5s=f.spread_change_5s)
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("SELL_IMBALANCE_NOT_DECREASING",))
        if (f.orderbook_imbalance or -1.0) <= self.p("min_orderbook_imbalance"):
            return self._reject(("BID_DEPTH_NOT_RESTORED",))

        # Structural edge: a fraction of the shock retraces.
        edge = abs(shock_bps) * self.p("retrace_fraction")
        score = _clamp(abs(shock_bps) / (2.0 * abs(self.p("shock_return_10s_bps"))))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(0.3 + 0.5 * score),
            edge_bps=edge,
            reasons=("LIQUIDITY_SHOCK_STABILISED", "SPREAD_CONTRACTING"),
            return_10s_bps=round(shock_bps, 3),
            spread_change_5s=f.spread_change_5s,
            orderbook_imbalance=f.orderbook_imbalance,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        if self._absorption_mode(f):
            target_bps, horizon = self._cost_feasible_volatility_edge(
                f,
                base_horizon_seconds=int(self.p("absorption_horizon_seconds")),
                maximum_horizon_seconds=int(self.p("max_absorption_horizon_seconds")),
                capture_fraction=self.p("absorption_capture_fraction"),
            )
            return ExitRule(
                strategy_id=self.strategy_id,
                stop_price=self._volatility_stop(entry_price, f, 2.0),
                target_price=(
                    entry_price * (1.0 + target_bps / 10_000.0)
                    if target_bps > 0
                    else None
                ),
                trailing_bps=self.p("trailing_bps"),
                max_holding_seconds=horizon,
                stop_basis="absorption_tick_volatility_multiple",
                target_basis="absorption_cost_aware_expected_move",
            )
        # Stop just under the shock low implied by the observed drop.
        shock_bps = abs((f.return_10s or 0.0) * 10_000.0)
        shock_low = entry_price * (1.0 - shock_bps / 10_000.0) if shock_bps else None
        stop = (
            shock_low * (1.0 - self.p("stop_buffer_bps") / 10_000.0)
            if shock_low
            else self._volatility_stop(entry_price, f, 2.0)
        )
        target = entry_price * (1.0 + (shock_bps * self.p("retrace_fraction")) / 10_000.0) if shock_bps else None
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=stop,
            target_price=target,
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="shock_low_minus_buffer",
            target_basis="partial_shock_retracement",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        if (f.spread_change_5s or 0.0) > 0 and (f.return_5s or 0.0) < 0:
            codes.append("SHOCK_REVERSAL_FAILED_SPREAD_REEXPANDING")
        if (f.orderbook_imbalance or 0.0) < 0 and (f.return_5s or 0.0) < 0:
            codes.append("SHOCK_REVERSAL_FAILED_NEW_LOW")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 5. Event momentum — TTL-bounded repricing, exhaustion guarded                #
# --------------------------------------------------------------------------- #
class EventMomentumAlgorithm(TradingAlgorithm):
    strategy_id = "event_momentum"
    thesis = "fresh material information keeps repricing until its TTL expires"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        # Event evidence comes from the electing ontology; absence fails closed.
        if not context.event_fresh:
            return self._reject(("EVENT_EVIDENCE_ABSENT",))
        age, ttl = context.event_age_seconds, context.event_ttl_seconds
        if age is None or ttl is None or age > ttl:
            return self._reject(("EVENT_TTL_EXPIRED",), event_age_seconds=age, event_ttl_seconds=ttl)
        volume_ratio = f.volume_spike_ratio
        if volume_ratio is None or volume_ratio < self.p("min_volume_spike_ratio"):
            return self._reject(("EVENT_VOLUME_NOT_ABNORMAL",), volume_spike_ratio=volume_ratio)
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("EVENT_FLOW_NOT_CONFIRMING",))
        # Underreaction only: if the move already happened, there is nothing left.
        move_bps = (f.return_10s or 0.0) * 10_000.0
        if move_bps >= self.p("exhaustion_return_10s_bps"):
            return self._reject(("EVENT_MOVE_ALREADY_EXHAUSTED",), return_10s_bps=round(move_bps, 3))

        remaining = max(0.0, ttl - age)
        horizon = int(min(self.horizon_seconds, remaining)) or self.horizon_seconds
        edge = tick_expected_move_bps(
            f.realized_volatility_10s,
            horizon,
            capture_fraction=self.config.shared("capture_fraction"),
        )
        freshness = _clamp(1.0 - age / ttl) if ttl > 0 else 0.0
        score = _clamp(0.5 * freshness + 0.5 * _clamp(volume_ratio / 4.0))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(0.3 + 0.5 * freshness),
            edge_bps=edge,
            reasons=("EVENT_REPRICING_CONFIRMED", "EVENT_WITHIN_TTL"),
            event_age_seconds=age,
            event_ttl_seconds=ttl,
            event_freshness=round(freshness, 4),
            volume_spike_ratio=volume_ratio,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        remaining = (
            max(0.0, (context.event_ttl_seconds or 0.0) - (context.event_age_seconds or 0.0))
            if context.event_ttl_seconds
            else 0.0
        )
        horizon = int(min(self.horizon_seconds, remaining)) or self.horizon_seconds
        edge = tick_expected_move_bps(
            f.realized_volatility_10s,
            horizon,
            capture_fraction=self.config.shared("capture_fraction"),
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(entry_price, f, self.p("stop_volatility_multiple")),
            target_price=entry_price * (1.0 + edge / 10_000.0) if edge else None,
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=horizon,
            stop_basis="tick_volatility_multiple",
            target_basis="event_horizon_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        age, ttl = context.event_age_seconds, context.event_ttl_seconds
        if age is not None and ttl is not None and age > ttl:
            codes.append("EVENT_TTL_EXPIRED")
        if (f.aggressor_imbalance_5s or 0.0) < 0 and (f.return_5s or 0.0) < 0:
            codes.append("EVENT_REPRICING_REVERSED")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 6. Cross-sectional relative strength — rank supplied by the ontology         #
# --------------------------------------------------------------------------- #
class CrossSectionalRelativeStrengthAlgorithm(TradingAlgorithm):
    strategy_id = "cross_sectional_relative_strength"
    thesis = "the strongest name in a supportive sector keeps outperforming"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        rank = context.sector_rank
        universe = context.sector_candidate_count
        if rank is None or universe is None or universe <= 1:
            return self._reject(("CROSS_SECTIONAL_RANK_ABSENT",))
        if rank > int(self.p("max_sector_rank")):
            return self._reject(("SECTOR_RANK_TOO_LOW",), sector_rank=rank, universe=universe)
        short_return_bps = (f.short_return or 0.0) * 10_000.0
        if short_return_bps < self.p("min_short_return_bps"):
            return self._reject(("RELATIVE_STRENGTH_NOT_CONFIRMED",), short_return_bps=short_return_bps)
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("RELATIVE_STRENGTH_FLOW_NOT_CONFIRMED",))

        edge = self._volatility_edge(f)
        rank_score = _clamp(1.0 - (rank - 1) / max(1, universe - 1))
        return self._fire(
            symbol=f.symbol,
            score=_clamp(0.6 * rank_score + 0.4 * _clamp(short_return_bps / 30.0)),
            confidence=_clamp(0.3 + 0.5 * rank_score),
            edge_bps=edge,
            reasons=("CROSS_SECTIONAL_LEADER", "RELATIVE_STRENGTH_CONFIRMED"),
            sector_rank=rank,
            sector_candidate_count=universe,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(entry_price, f, self.p("stop_volatility_multiple")),
            target_price=entry_price * (1.0 + self._volatility_edge(f) / 10_000.0),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="tick_volatility_multiple",
            target_basis="tick_volatility_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        rank = context.sector_rank
        # Hysteresis: only a clear rank decay exits, not a one-place slip.
        if rank is not None and rank > int(self.p("max_sector_rank")) + 1:
            codes.append("SECTOR_RANK_DECAYED")
        if (f.short_return or 0.0) < 0 and (f.aggressor_imbalance_5s or 0.0) < 0:
            codes.append("RELATIVE_STRENGTH_LOST")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 7. Gap context — exactly one submode, never blended                          #
# --------------------------------------------------------------------------- #
class GapContextAlgorithm(TradingAlgorithm):
    strategy_id = "gap_context"
    thesis = "an opening gap either continues on confirmation or fills after exhaustion"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        gap = context.gap_rate
        submode = (context.gap_submode or "").strip().lower()
        if gap is None or not submode:
            return self._reject(("GAP_CONTEXT_ABSENT",))
        if submode not in {"continuation", "fade"}:
            return self._reject(("GAP_SUBMODE_INVALID",), gap_submode=submode)
        if abs(gap) < self.p("min_gap_rate"):
            return self._reject(("GAP_TOO_SMALL",), gap_rate=gap)

        if submode == "continuation":
            return self._continuation(f, context, gap)
        return self._fade(f, context, gap)

    def _continuation(self, f, context, gap: float) -> AlgorithmDecision:
        # Long-only: only an up-gap can continue upward.
        if gap <= 0:
            return self._reject(("GAP_CONTINUATION_REQUIRES_UP_GAP",), gap_rate=gap)
        volume_ratio = f.volume_spike_ratio
        if volume_ratio is None or volume_ratio < self.p("min_volume_spike_ratio"):
            return self._reject(("GAP_VOLUME_NOT_CONFIRMED",), volume_spike_ratio=volume_ratio)
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("GAP_FLOW_NOT_CONFIRMED",))
        open_price = context.session_open_price
        if open_price and f.price and f.price < open_price:
            return self._reject(("GAP_CONTINUATION_LOST_OPEN",), price=f.price, open_price=open_price)
        edge = self._volatility_edge(f)
        return self._fire(
            symbol=f.symbol,
            score=_clamp(0.5 * _clamp(abs(gap) / 0.05) + 0.5 * _clamp(volume_ratio / 3.0)),
            confidence=_clamp(0.3 + 0.4 * _clamp(volume_ratio / 3.0)),
            edge_bps=edge,
            reasons=("GAP_CONTINUATION_CONFIRMED",),
            gap_rate=gap,
            gap_submode="continuation",
        )

    def _fade(self, f, context, gap: float) -> AlgorithmDecision:
        # Long-only: only a down-gap can be bought for the fill.
        if gap >= 0:
            return self._reject(("GAP_FADE_REQUIRES_DOWN_GAP",), gap_rate=gap)
        if (f.return_1s or 0.0) < 0:
            return self._reject(("GAP_FADE_STILL_FALLING",))
        if (f.orderbook_imbalance_change_5s or 0.0) < 0:
            return self._reject(("GAP_FADE_PRESSURE_WORSENING",))
        previous_close = context.previous_close_price
        if not previous_close or not f.price or f.price >= previous_close:
            return self._reject(("GAP_ALREADY_FILLED",))
        # Structural edge: a fraction of the gap fills back toward prior close.
        fill_bps = (previous_close / f.price - 1.0) * 10_000.0
        edge = max(0.0, fill_bps) * self.p("fill_capture_fraction")
        return self._fire(
            symbol=f.symbol,
            score=_clamp(abs(gap) / 0.05),
            confidence=_clamp(0.3 + 0.4 * _clamp(abs(gap) / 0.05)),
            edge_bps=edge,
            reasons=("GAP_FADE_STABILISED",),
            gap_rate=gap,
            gap_submode="fade",
            previous_close_price=previous_close,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        submode = (context.gap_submode or "").strip().lower()
        if submode == "fade" and context.previous_close_price:
            capture = self.p("fill_capture_fraction")
            target = entry_price + capture * (context.previous_close_price - entry_price)
            target_basis = "partial_gap_fill_to_previous_close"
        else:
            target = entry_price * (1.0 + self._volatility_edge(f) / 10_000.0)
            target_basis = "tick_volatility_expected_move"
        reference = context.session_open_price if submode == "continuation" else None
        stop = (
            reference * (1.0 - self.p("stop_buffer_bps") / 10_000.0)
            if reference
            else self._volatility_stop(entry_price, f, 2.0)
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=stop,
            target_price=target,
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="session_open_minus_buffer" if reference else "tick_volatility_multiple",
            target_basis=target_basis,
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        submode = (context.gap_submode or "").strip().lower()
        open_price = context.session_open_price
        if submode == "continuation" and open_price and f.price and f.price < open_price:
            codes.append("GAP_CONTINUATION_INVALIDATED_BELOW_OPEN")
        if submode == "fade" and (f.return_5s or 0.0) < 0 and (f.aggressor_imbalance_5s or 0.0) < 0:
            codes.append("GAP_FADE_INVALIDATED_NEW_LOW")
        return tuple(codes)


class RvgiBoxBreakoutAlgorithm(TradingAlgorithm):
    strategy_id = "rvgi_box_breakout"
    thesis = "bullish RVGI confirms acceptance above a frozen causal price box"

    def _context_reasons(self, context: ElectionContext) -> tuple[str, ...]:
        required = (
            context.rvgi,
            context.rvgi_signal,
            context.box_high,
            context.box_low,
            context.box_mid,
            context.box_width_pct,
            context.box_previous_close,
        )
        if not _present(*required) or not context.box_context_timestamp:
            return ("RVGI_BOX_CONTEXT_MISSING",)
        if context.rvgi <= context.rvgi_signal or not context.rvgi_bullish_cross:
            return ("RVGI_BOX_RVGI_NOT_CONFIRMED",)
        if not context.volume_confirmed:
            return ("RVGI_BOX_VOLUME_NOT_CONFIRMED",)
        if not self.p("min_box_width_pct") <= context.box_width_pct <= self.p("max_box_width_pct"):
            return ("RVGI_BOX_GEOMETRY_INVALID",)
        return ()

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        context_reasons = self._context_reasons(context)
        if context_reasons:
            return self._reject(context_reasons)
        ready, _ = self._tick_ready(f)
        if not ready:
            return self._reject(("RVGI_BOX_TICK_WINDOW_NOT_READY",))
        assert context.box_high is not None
        assert context.box_low is not None
        assert context.box_previous_close is not None
        price = f.price
        if price is None or price <= 0:
            return self._reject(("RVGI_BOX_CONTEXT_MISSING",))
        trigger = context.box_high * (1.0 + self.p("entry_buffer_bps") / 10_000.0)
        distance_bps = (price / context.box_high - 1.0) * 10_000.0
        if price <= trigger:
            return self._reject(
                ("RVGI_BOX_NOT_ABOVE_FROZEN_HIGH",),
                frozen_box_high=context.box_high,
                breakout_distance_bps=distance_bps,
            )
        if context.box_previous_close > context.box_high:
            return self._reject(("RVGI_BOX_BREAKOUT_NOT_FRESH",))
        if distance_bps > self.p("max_extension_bps"):
            return self._reject(("RVGI_BOX_OVEREXTENDED",), breakout_distance_bps=distance_bps)
        if f.volume_spike_ratio is None or f.volume_spike_ratio < self.p("min_volume_spike_ratio"):
            return self._reject(("RVGI_BOX_VOLUME_NOT_CONFIRMED",))
        if (
            f.aggressor_imbalance_5s is None
            or f.aggressor_imbalance_5s < self.p("min_aggressor_imbalance")
        ):
            return self._reject(("RVGI_BOX_FLOW_NOT_CONFIRMED",))
        if (f.return_1s or 0.0) < 0 and (f.return_5s or 0.0) < 0:
            return self._reject(("RVGI_BOX_FALSE_BREAKOUT_RISK",))
        box_height_return = max(0.0, (context.box_high - context.box_low) / price)
        captured = min(
            self.p("max_target_return"),
            box_height_return * self.p("target_capture_fraction"),
        )
        edge_bps = captured * 10_000.0
        if edge_bps <= 0:
            return self._reject(("RVGI_BOX_EXPECTED_EDGE_TOO_LOW",))
        score = _clamp(
            0.35
            + 0.25 * min(1.0, (f.volume_spike_ratio or 0.0) / 3.0)
            + 0.2 * max(0.0, f.aggressor_imbalance_5s or 0.0)
            + 0.2 * min(1.0, max(0.0, context.rvgi_diff or 0.0) * 10.0)
        )
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=score,
            edge_bps=edge_bps,
            reasons=("RVGI_BOX_CONFIRMED_BREAKOUT_LONG",),
            rvgi=context.rvgi,
            rvgi_signal=context.rvgi_signal,
            frozen_box_high=context.box_high,
            frozen_box_low=context.box_low,
            breakout_distance_bps=distance_bps,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        box_high = context.box_high or entry_price
        box_low = context.box_low or box_high
        bps_buffer = box_high * self.p("stop_buffer_bps") / 10_000.0
        atr_buffer = entry_price * max(0.0, f.atr_pct or 0.0) * self.p("stop_atr_multiple")
        stop = box_high - max(bps_buffer, atr_buffer)
        height = max(0.0, box_high - box_low)
        target_return = min(
            self.p("max_target_return"),
            (height / entry_price if entry_price > 0 else 0.0)
            * self.p("target_capture_fraction"),
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=min(box_high, stop),
            target_price=entry_price * (1.0 + target_return),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="frozen_box_high_minus_max_bps_or_atr_buffer",
            target_basis="fraction_of_frozen_box_height",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        if context.box_high and f.price:
            failure = context.box_high * (
                1.0 - self.p("failure_tolerance_bps") / 10_000.0
            )
            if f.price < failure:
                codes.append("RVGI_BOX_FALSE_BREAKOUT")
        if f.rvgi_bearish_cross:
            codes.append("RVGI_BEARISH_CROSS")
        if (
            f.volume_spike_ratio is not None
            and f.volume_spike_ratio < 1.0
            and context.box_high
            and f.price
            and f.price < context.box_high
        ):
            codes.append("RVGI_BOX_VOLUME_COLLAPSE_INSIDE_BOX")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 9. Residual relative strength — market/sector-neutral idiosyncratic leadership #
# --------------------------------------------------------------------------- #
class ResidualRelativeStrengthAlgorithm(TradingAlgorithm):
    """Buys the name that is strong AFTER removing market and sector beta.

    ``cross_sectional_relative_strength`` ranks on raw return, which in a tape
    dominated by the index and a handful of semiconductor megacaps is mostly a
    ranking of market beta — i.e. it buys the index with extra steps. This
    algorithm consumes the residual

        ResidualReturn = Return - beta_market * MarketReturn - beta_sector * SectorReturn

    resolved by the electing layer, so what it buys is the stock-specific bid.
    Because the residual is cross-sectional it cannot be recomputed from one
    symbol's ticks: absent residuals fail closed rather than degrade to raw
    return, which would silently reproduce the defect.
    """

    strategy_id = "residual_relative_strength"
    thesis = "idiosyncratic strength net of market and sector beta persists while flow confirms it"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        short_residual = context.residual_return_short_bps
        long_residual = context.residual_return_long_bps
        if not _present(short_residual, long_residual):
            return self._reject(("RESIDUAL_STRENGTH_CONTEXT_ABSENT",))
        rank = context.sector_rank
        universe = context.sector_candidate_count
        if rank is None or universe is None or universe <= 1:
            return self._reject(("RESIDUAL_SECTOR_RANK_ABSENT",))
        change_point = context.change_point_probability
        if (
            change_point is not None
            and change_point > self.p("max_change_point_probability")
        ):
            return self._reject(
                ("RESIDUAL_STRENGTH_REGIME_UNSTABLE",), change_point_probability=change_point
            )
        if short_residual < self.p("min_residual_short_bps"):
            return self._reject(
                ("RESIDUAL_SHORT_HORIZON_NOT_POSITIVE",), residual_return_short_bps=short_residual
            )
        if long_residual < self.p("min_residual_long_bps"):
            return self._reject(
                ("RESIDUAL_LONG_HORIZON_NOT_POSITIVE",), residual_return_long_bps=long_residual
            )
        if rank > int(self.p("max_sector_rank")):
            return self._reject(("RESIDUAL_SECTOR_RANK_TOO_LOW",), sector_rank=rank, universe=universe)
        relative_volume = f.relative_volume
        if relative_volume is None or relative_volume < self.p("min_relative_volume"):
            return self._reject(
                (rc.VOLUME_CONFIRMATION_MISSING, "RESIDUAL_VOLUME_NOT_CONFIRMED"),
                relative_volume=relative_volume,
            )
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("RESIDUAL_FLOW_NOT_CONFIRMED",))
        microprice_edge = f.microprice_edge_bps
        if microprice_edge is None:
            return self._reject(("RESIDUAL_MICROPRICE_UNAVAILABLE",))
        if microprice_edge < self.p("min_microprice_edge_bps"):
            return self._reject(
                ("RESIDUAL_MICROPRICE_NOT_SUPPORTIVE",), microprice_edge_bps=microprice_edge
            )
        # Investor-flow corroboration. Required by default, because idiosyncratic
        # strength with no informed flow behind it is usually a squeeze.
        flow_scores = [
            value
            for value in (context.foreign_flow_zscore, context.institution_flow_zscore)
            if value is not None
        ]
        if self.p("require_flow_confirmation") >= 1.0:
            if not flow_scores:
                return self._reject(("RESIDUAL_INVESTOR_FLOW_ABSENT",))
            if max(flow_scores) < self.p("min_flow_zscore"):
                return self._reject(
                    ("RESIDUAL_INVESTOR_FLOW_NEGATIVE",), flow_zscores=flow_scores
                )

        edge = self._volatility_edge(f)
        rank_score = _clamp(1.0 - (rank - 1) / max(1, universe - 1))
        residual_score = _clamp(short_residual / 30.0)
        return self._fire(
            symbol=f.symbol,
            score=_clamp(0.45 * rank_score + 0.35 * residual_score + 0.2 * _clamp(microprice_edge)),
            confidence=_clamp(0.3 + 0.4 * rank_score + 0.2 * residual_score),
            edge_bps=edge,
            reasons=("RESIDUAL_STRENGTH_CONFIRMED", "MARKET_AND_SECTOR_NEUTRAL_LEADER"),
            residual_return_short_bps=round(short_residual, 3),
            residual_return_long_bps=round(long_residual, 3),
            sector_rank=rank,
            sector_candidate_count=universe,
            microprice_edge_bps=round(microprice_edge, 4),
            market_beta=context.market_beta,
            sector_beta=context.sector_beta,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(entry_price, f, self.p("stop_volatility_multiple")),
            target_price=entry_price * (1.0 + self._volatility_edge(f) / 10_000.0),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="tick_volatility_multiple",
            target_basis="tick_volatility_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        rank = context.sector_rank
        if rank is not None and rank > int(self.p("max_sector_rank")) + 1:
            codes.append("RESIDUAL_SECTOR_RANK_DECAYED")
        if (f.short_return or 0.0) < 0 and (f.aggressor_imbalance_5s or 0.0) < 0:
            codes.append("RESIDUAL_STRENGTH_LOST")
        edge = f.microprice_edge_bps
        if edge is not None and edge < 0 and (f.return_5s or 0.0) < 0:
            codes.append("RESIDUAL_MICROPRICE_TURNED_OFFERED")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 10. Adaptive anchored VWAP reversion — volatility-normalised displacement     #
# --------------------------------------------------------------------------- #
class AdaptiveAnchoredVwapReversionAlgorithm(TradingAlgorithm):
    """VWAP reversion on a z-score, not on a fixed basis-point band.

    ``vwap_mean_reversion`` fires at a fixed 25bps displacement. On a day the
    index moves 4-10%, 25bps is inside the noise, so the strategy buys constantly
    and reverts rarely. Normalising by intraday realised volatility

        VWAP_Z = (price - anchoredVWAP) / intradayResidualVolatility

    makes one threshold mean the same thing in a calm and a violent tape. Entry
    additionally requires evidence that liquidity is coming BACK (spread
    contracting, book tilting bid, drop stopped) rather than simply that the price
    is low — buying oversold without that is how a mean-reversion book gets run
    over in a repricing.
    """

    strategy_id = "adaptive_anchored_vwap_reversion"
    thesis = "a volatility-normalised displacement below anchored VWAP reverts once liquidity returns"

    def _anchor(self, f: TechnicalFeatureSet, context: ElectionContext) -> tuple[float | None, str]:
        if context.anchored_vwap and context.anchored_vwap > 0:
            return float(context.anchored_vwap), str(context.anchor_basis or "election_anchor")
        if f.vwap and f.vwap > 0:
            return float(f.vwap), "session_vwap"
        return None, "unavailable"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        anchor, anchor_basis = self._anchor(f, context)
        price = f.price
        if anchor is None or price is None or price <= 0:
            return self._reject(("ADAPTIVE_VWAP_ANCHOR_UNAVAILABLE",))
        volatility_bps = f.residual_volatility_bps
        if not volatility_bps:
            # Without a volatility scale the z-score is undefined. Falling back to
            # a fixed band here is exactly the defect this algorithm exists to fix.
            return self._reject(("ADAPTIVE_VWAP_VOLATILITY_UNAVAILABLE",))
        displacement_bps = (price / anchor - 1.0) * 10_000.0
        zscore = displacement_bps / volatility_bps
        change_point = context.change_point_probability
        if (
            change_point is not None
            and change_point > self.p("max_change_point_probability")
        ):
            return self._reject(
                ("ADAPTIVE_VWAP_REGIME_UNSTABLE",), change_point_probability=change_point
            )
        if zscore > -self.p("min_entry_zscore"):
            return self._reject(
                ("ADAPTIVE_VWAP_DISPLACEMENT_TOO_SMALL",),
                vwap_zscore=round(zscore, 4),
                displacement_bps=round(displacement_bps, 3),
            )
        if zscore < -self.p("max_entry_zscore"):
            # Beyond this the move is no longer a displacement around a mean; it is
            # a repricing, and there is no mean left to revert to.
            return self._reject(
                ("ADAPTIVE_VWAP_DISLOCATION_NOT_REVERSION",), vwap_zscore=round(zscore, 4)
            )
        if abs(displacement_bps) < self.p("min_displacement_bps"):
            return self._reject(
                ("ADAPTIVE_VWAP_DISPLACEMENT_BELOW_FLOOR",),
                displacement_bps=round(displacement_bps, 3),
            )
        if (f.return_1s or 0.0) < 0:
            return self._reject(("ADAPTIVE_VWAP_STILL_FALLING",))
        if (f.orderbook_imbalance_change_5s or 0.0) <= 0:
            return self._reject(("ADAPTIVE_VWAP_OFI_NOT_TURNING",))
        if (f.spread_change_5s or 0.0) > self.p("max_spread_change_5s"):
            return self._reject(("ADAPTIVE_VWAP_SPREAD_STILL_WIDENING",))
        microprice_edge = f.microprice_edge_bps
        if microprice_edge is None:
            return self._reject(("ADAPTIVE_VWAP_MICROPRICE_UNAVAILABLE",))
        if microprice_edge < self.p("min_microprice_edge_bps"):
            return self._reject(
                ("ADAPTIVE_VWAP_MICROPRICE_BELOW_MID",), microprice_edge_bps=microprice_edge
            )

        edge = min(
            self.p("max_target_bps"),
            abs(displacement_bps) * self.p("target_capture_fraction"),
        )
        score = _clamp(abs(zscore) / (2.0 * self.p("min_entry_zscore")))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(0.3 + 0.45 * score),
            edge_bps=edge,
            reasons=(
                rc.MEAN_REVERSION_CANDIDATE,
                "ADAPTIVE_VWAP_DISPLACEMENT_CONFIRMED",
                "LIQUIDITY_RECOVERY_CONFIRMED",
            ),
            anchored_vwap=anchor,
            anchor_basis=anchor_basis,
            vwap_zscore=round(zscore, 4),
            displacement_bps=round(displacement_bps, 3),
            residual_volatility_bps=round(volatility_bps, 3),
            microprice_edge_bps=round(microprice_edge, 4),
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        anchor, _basis = self._anchor(f, context)
        capture = self.p("target_capture_fraction")
        target = (
            min(
                entry_price + capture * (anchor - entry_price),
                entry_price * (1.0 + self.p("max_target_bps") / 10_000.0),
            )
            if anchor and anchor > entry_price
            else None
        )
        volatility_bps = f.residual_volatility_bps
        stop = (
            entry_price
            * (1.0 - self.p("stop_zscore_multiple") * volatility_bps / 10_000.0)
            if volatility_bps and entry_price > 0
            else None
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=stop,
            target_price=target,
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="volatility_zscore_multiple",
            target_basis="partial_reversion_to_anchored_vwap",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        zscore = f.vwap_zscore
        if (
            zscore is not None
            and zscore < -self.p("max_entry_zscore")
        ):
            codes.append("ADAPTIVE_VWAP_DISPLACEMENT_BECAME_DISLOCATION")
        if (f.return_5s or 0.0) < 0 and (f.orderbook_imbalance_change_5s or 0.0) < 0:
            codes.append("ADAPTIVE_VWAP_REVERSION_FAILED_NEW_LOW")
        if (f.spread_change_5s or 0.0) > 0 and (f.return_1s or 0.0) < 0:
            codes.append("ADAPTIVE_VWAP_LIQUIDITY_RECOVERY_REVERSED")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 11. OFI / microprice exhaustion reversal — order-flow confirmed, not price     #
# --------------------------------------------------------------------------- #
class OfiMicropriceExhaustionReversalAlgorithm(TradingAlgorithm):
    """Buys a sell-side exhaustion confirmed by the book, not by the tape.

    Distinct from ``liquidity_shock_reversal``, which keys on the price drop and
    a contracting spread. Here the drop is only the precondition; the trigger is
    that the *book* has turned: order-flow imbalance slope positive, bid depth
    restored relative to ask, and the depth-weighted microprice above the mid.

    Deliberately not a stock-selection signal. Published evidence on order-flow
    imbalance is consistent — statistically predictive, and routinely eaten by
    transaction costs when traded standalone — so it is used here for entry timing
    and adverse-selection avoidance inside a thesis that already exists, with the
    shortest holding window of any strategy in the catalogue.
    """

    strategy_id = "ofi_microprice_exhaustion_reversal"
    thesis = "sell-side exhaustion retraces once order flow, depth and microprice all turn"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        if not _present(f.return_10s, f.spread_change_5s, f.orderbook_imbalance):
            return self._reject(("OFI_EXHAUSTION_TICK_INPUTS_MISSING",))
        change_point = context.change_point_probability
        if (
            change_point is not None
            and change_point > self.p("max_change_point_probability")
        ):
            return self._reject(
                ("OFI_EXHAUSTION_REGIME_UNSTABLE",), change_point_probability=change_point
            )
        toxicity = context.flow_toxicity
        if toxicity is not None and toxicity > self.p("max_flow_toxicity"):
            # Flow toxicity is a risk statement, never a buy reason: a toxic tape
            # means the counterparty is better informed than this thesis is.
            return self._reject(("OFI_EXHAUSTION_FLOW_TOXIC",), flow_toxicity=toxicity)
        shock_bps = (f.return_10s or 0.0) * 10_000.0
        if shock_bps > self.p("shock_return_10s_bps"):
            return self._reject(("OFI_EXHAUSTION_NO_SELLOFF",), return_10s_bps=round(shock_bps, 3))
        ofi_slope = f.orderbook_imbalance_change_5s
        if ofi_slope is None or ofi_slope <= self.p("min_ofi_slope"):
            return self._reject(("OFI_SLOPE_NOT_POSITIVE",), ofi_slope=ofi_slope)
        if (f.orderbook_imbalance or -1.0) < self.p("min_orderbook_imbalance"):
            return self._reject(
                ("OFI_BID_SIDE_NOT_RESTORED",), orderbook_imbalance=f.orderbook_imbalance
            )
        depth_ratio = f.depth_ratio
        if depth_ratio is None:
            return self._reject(("OFI_DEPTH_UNAVAILABLE",))
        if depth_ratio < self.p("min_depth_ratio"):
            return self._reject(("OFI_ASK_NOT_DEPLETED",), depth_ratio=depth_ratio)
        if (f.spread_change_5s or 0.0) > self.p("max_spread_change_5s"):
            return self._reject(("OFI_SPREAD_STILL_WIDENING",), spread_change_5s=f.spread_change_5s)
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("OFI_SELL_AGGRESSION_NOT_EASING",))
        microprice_edge = f.microprice_edge_bps
        if microprice_edge is None:
            return self._reject(("OFI_MICROPRICE_UNAVAILABLE",))
        if microprice_edge < self.p("min_microprice_edge_bps"):
            return self._reject(
                ("OFI_MICROPRICE_NOT_ABOVE_MID",), microprice_edge_bps=microprice_edge
            )

        edge = abs(shock_bps) * self.p("retrace_fraction")
        score = _clamp(abs(shock_bps) / (2.0 * abs(self.p("shock_return_10s_bps"))))
        return self._fire(
            symbol=f.symbol,
            score=score,
            confidence=_clamp(0.25 + 0.4 * score + 0.2 * _clamp(ofi_slope)),
            edge_bps=edge,
            reasons=("OFI_EXHAUSTION_CONFIRMED", "MICROPRICE_ABOVE_MID", "DEPTH_RECOVERING"),
            return_10s_bps=round(shock_bps, 3),
            ofi_slope=round(ofi_slope, 5),
            depth_ratio=round(depth_ratio, 4),
            microprice_edge_bps=round(microprice_edge, 4),
            spread_change_5s=f.spread_change_5s,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        shock_bps = abs((f.return_10s or 0.0) * 10_000.0)
        shock_low = entry_price * (1.0 - shock_bps / 10_000.0) if shock_bps else None
        stop = (
            shock_low * (1.0 - self.p("stop_buffer_bps") / 10_000.0)
            if shock_low
            else self._volatility_stop(entry_price, f, 2.0)
        )
        target = (
            entry_price * (1.0 + (shock_bps * self.p("retrace_fraction")) / 10_000.0)
            if shock_bps
            else None
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=stop,
            target_price=target,
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="selloff_low_minus_buffer",
            target_basis="partial_retracement_of_selloff",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        if (f.orderbook_imbalance_change_5s or 0.0) < 0 and (f.return_5s or 0.0) < 0:
            codes.append("OFI_SLOPE_REVERSED")
        edge = f.microprice_edge_bps
        if edge is not None and edge < 0:
            codes.append("OFI_MICROPRICE_BACK_BELOW_MID")
        if (f.spread_change_5s or 0.0) > 0 and (f.return_5s or 0.0) < 0:
            codes.append("OFI_EXHAUSTION_FAILED_SPREAD_REEXPANDING")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 12. Opening-range breakout — gated on relative volume ("stocks in play")      #
# --------------------------------------------------------------------------- #
class OpeningRangeBreakoutAlgorithm(TradingAlgorithm):
    """Break of the session's opening range, taken only on stocks in play.

    The published day-trading result this implements is explicit that the
    breakout alone does not pay: the profitability comes from restricting it to
    the names with the highest opening relative volume. So relative volume is a
    hard precondition here, not a score contribution — firing without it would be
    implementing a different, unprofitable strategy under the same name.
    """

    strategy_id = "opening_range_breakout"
    thesis = "price clears the session opening range on unusually high relative volume"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)

        high = context.opening_range_high
        low = context.opening_range_low
        if high is None or low is None or high <= low or high <= 0:
            return self._reject(("ORB_OPENING_RANGE_ABSENT",))
        price = f.price
        if not price or price <= 0:
            return self._reject(("ORB_PRICE_ABSENT",))

        # Relative volume: prefer the electing layer's figure, fall back to the
        # tick-window volume spike. Absent entirely -> fail closed, because this
        # gate is the thesis.
        relative_volume = context.relative_volume
        if relative_volume is None:
            relative_volume = f.volume_spike_ratio
        if relative_volume is None:
            return self._reject(("ORB_RELATIVE_VOLUME_ABSENT",))
        if relative_volume < self.p("min_relative_volume"):
            return self._reject(
                ("ORB_NOT_IN_PLAY",),
                relative_volume=relative_volume,
                minimum=self.p("min_relative_volume"),
            )

        span = high - low
        excess_bps = (price - high) / span * 10_000.0
        if excess_bps < self.p("min_breakout_excess_bps"):
            return self._reject(
                ("ORB_RANGE_NOT_CLEARED",),
                excess_bps=round(excess_bps, 3),
                opening_range_high=high,
            )
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("ORB_FLOW_NOT_CONFIRMED",))
        if (f.spread_change_5s or 0.0) > self.p("max_spread_change_5s"):
            return self._reject(("ORB_SPREAD_WIDENING_INTO_BREAK",))
        change_point = context.change_point_probability
        if change_point is not None and change_point > self.p("max_change_point_probability"):
            return self._reject(
                ("ORB_STRUCTURAL_BREAK",), change_point_probability=change_point
            )

        edge = self._volatility_edge(f)
        # A wider opening range is a stronger in-play signal, but it also means a
        # wider stop; the score reflects flow agreement and RVOL, not range width.
        return self._fire(
            symbol=f.symbol,
            score=_clamp(
                0.5 * _clamp(relative_volume / 3.0)
                + 0.5 * _clamp((f.aggressor_imbalance_5s or 0.0) + 0.5)
            ),
            confidence=_clamp(0.3 + 0.4 * _clamp(relative_volume / 3.0)),
            edge_bps=edge,
            reasons=("ORB_RANGE_CLEARED_IN_PLAY",),
            relative_volume=relative_volume,
            excess_bps=round(excess_bps, 3),
            opening_range_high=high,
            opening_range_low=low,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        # The published rule stops at the opposite end of the opening range.
        low = context.opening_range_low
        stop = (
            low * (1.0 - self.p("stop_buffer_bps") / 10_000.0)
            if low
            else self._volatility_stop(entry_price, f, 2.0)
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=stop,
            target_price=entry_price * (1.0 + self._volatility_edge(f) / 10_000.0),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis=(
                "opening_range_low_minus_buffer" if low else "tick_volatility_multiple"
            ),
            target_basis="tick_volatility_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        high = context.opening_range_high
        # Falling back inside the range is the definition of a failed break.
        if high and f.price and f.price < high:
            codes.append("ORB_FELL_BACK_INTO_RANGE")
        if (f.aggressor_imbalance_5s or 0.0) < 0 and (f.return_5s or 0.0) < 0:
            codes.append("ORB_FLOW_REVERSED")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# 13. Market intraday momentum — one round trip per day                         #
# --------------------------------------------------------------------------- #
class MarketIntradayMomentumAlgorithm(TradingAlgorithm):
    """The first half-hour return continues into the last half-hour.

    Chosen for this account on cost grounds as much as edge grounds. 20bps of the
    27.8bps KRX round-trip cost is statutory securities tax, charged per ROUND TRIP,
    so a strategy's trip count is as decisive as its signal: twelve scalps a day pay
    the tax twelve times against a measured gross edge of ~0bps. This takes ONE trip
    per day.

    Published effect: first half-hour return (from the previous close, so including
    the overnight gap) predicts the last half-hour return; R² 1.6%, rising to 3.3%
    when first-half-hour volatility is high; present in 12 of 16 developed markets.

    Long-only, so only the positive leg is tradable. Flat before the 15:20 KRX
    closing auction — the auction is a different matching mechanism and this system
    does not model it.
    """

    strategy_id = "market_intraday_momentum"
    thesis = "a positive first half-hour continues into the last continuous half-hour"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)

        in_window = context.in_last_continuous_half_hour
        if in_window is None:
            return self._reject(("MIM_SESSION_CONTEXT_ABSENT",))
        if not in_window:
            return self._reject(("MIM_OUTSIDE_ENTRY_WINDOW",))

        # Never open a position with too little continuous trading left to exit into.
        remaining = context.minutes_to_continuous_close
        if remaining is not None and remaining < 5.0:
            return self._reject(
                ("MIM_TOO_CLOSE_TO_AUCTION",), minutes_to_continuous_close=remaining
            )

        r1 = context.first_half_hour_return_bps
        if r1 is None:
            return self._reject(("MIM_FIRST_HALF_HOUR_ABSENT",))
        if r1 < self.p("min_first_half_hour_return_bps"):
            return self._reject(("MIM_FIRST_HALF_HOUR_NOT_UP",), first_half_hour_return_bps=r1)

        volatility_percentile = context.first_half_hour_volatility_percentile
        if volatility_percentile is None:
            return self._reject(("MIM_VOLATILITY_CONTEXT_ABSENT",))
        if volatility_percentile < self.p("min_first_half_hour_volatility_percentile"):
            # Both the published effect and the cost arithmetic require a volatile
            # day; on a quiet day the last half-hour does not travel 33bps.
            return self._reject(
                ("MIM_DAY_NOT_VOLATILE_ENOUGH",),
                first_half_hour_volatility_percentile=volatility_percentile,
            )

        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("MIM_FLOW_NOT_CONFIRMED",))
        change_point = context.change_point_probability
        if change_point is not None and change_point > self.p("max_change_point_probability"):
            return self._reject(("MIM_STRUCTURAL_BREAK",), change_point_probability=change_point)

        edge = self._volatility_edge(f)
        return self._fire(
            symbol=f.symbol,
            score=_clamp(0.5 * _clamp(r1 / 100.0) + 0.5 * _clamp(volatility_percentile)),
            confidence=_clamp(0.3 + 0.4 * _clamp(volatility_percentile)),
            edge_bps=edge,
            reasons=("MIM_FIRST_HALF_HOUR_CONTINUATION",),
            first_half_hour_return_bps=r1,
            first_half_hour_volatility_percentile=volatility_percentile,
            minutes_to_continuous_close=remaining,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        # The intended exit is TIME: be flat before the closing auction. The horizon
        # is shortened to whatever continuous trading actually remains, so a late
        # entry does not inherit a full 25-minute leash it cannot use.
        remaining = context.minutes_to_continuous_close
        horizon = self.horizon_seconds
        if remaining is not None and remaining > 0:
            horizon = int(min(horizon, max(60.0, (remaining - 2.0) * 60.0)))
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(entry_price, f, 2.0),
            target_price=entry_price * (1.0 + self._volatility_edge(f) / 10_000.0),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=horizon,
            stop_basis="tick_volatility_multiple",
            target_basis="tick_volatility_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        remaining = context.minutes_to_continuous_close
        if remaining is not None and remaining <= 2.0:
            # Must not be carried into the auction.
            codes.append("MIM_CONTINUOUS_CLOSE_IMMINENT")
        if (f.aggressor_imbalance_5s or 0.0) < 0 and (f.return_5s or 0.0) < 0:
            codes.append("MIM_MOMENTUM_REVERSED")
        return tuple(codes)


# --------------------------------------------------------------------------- #
# SHORT theses                                                                 #
# --------------------------------------------------------------------------- #
class ShortTradingAlgorithm(TradingAlgorithm):
    """Base for short entries: the borrow and squeeze preconditions, once.

    Every short thesis in this module has to answer four questions no long thesis
    ever asks, and they are checked here so no individual algorithm can forget one:

    1. Is short selling permitted on this name at all (규제/공매도 금지)?
    2. Is stock actually locatable, in sufficient quantity, at an acceptable fee?
    3. Is the borrow observation FRESH, or are we about to act on a stale locate?
    4. Is the name crowded enough that covering could become the loss?

    All four fail CLOSED on absent data. The asymmetry against the long path is
    intentional: a missing long-side feature costs a missed trade, while a missing
    borrow fact costs a rejected order, a forced buy-in, or an unbounded loss.
    """

    direction = "SHORT"

    # Beyond this the locate is a memory, not a fact. Short thresholds are in
    # seconds because borrow availability moves intraday, and the whole point of
    # storing ``borrow_observed_at`` is to be able to refuse a stale one.
    borrow_snapshot_max_age_seconds = 120.0

    def _short_preconditions(
        self, f: TechnicalFeatureSet, context: ElectionContext
    ) -> tuple[str, ...]:
        """Reason codes blocking a short entry; empty means clear."""
        if context.short_sale_permitted is not True:
            return (ShortReasonCodes.SHORT_SALE_NOT_PERMITTED,)
        if context.borrow_available is not True:
            return (ShortReasonCodes.BORROW_UNAVAILABLE,)
        quantity = context.borrow_available_quantity
        if quantity is None or quantity < int(self.p("min_borrow_quantity")):
            return (ShortReasonCodes.BORROW_QUANTITY_INSUFFICIENT,)
        fee_bps = context.borrow_fee_bps_annualised
        if fee_bps is None:
            # An unknown fee is not a zero fee. Pricing a short at zero borrow cost
            # is how a negative-expectancy trade passes a cost gate.
            return (ShortReasonCodes.BORROW_COST_TOO_HIGH,)
        if fee_bps > self.p("max_borrow_fee_bps_annualised"):
            return (ShortReasonCodes.BORROW_COST_TOO_HIGH,)
        if not self._borrow_snapshot_fresh(context):
            return (ShortReasonCodes.BORROW_SNAPSHOT_STALE,)
        # Squeeze exclusion. Absent metrics do NOT block — unlike the borrow facts,
        # these are supplementary risk colour rather than execution preconditions,
        # and requiring them would make the strategies inert on every name without
        # published short interest. The borrow gates above already carry the
        # fail-closed weight.
        days = context.days_to_cover
        if days is not None and days > self.p("max_days_to_cover"):
            return ("SHORT_CROWDED_DAYS_TO_COVER",)
        ratio = context.short_interest_ratio
        if ratio is not None and ratio > self.p("max_short_interest_ratio"):
            return ("SHORT_CROWDED_SHORT_INTEREST",)
        # Execution quality. A short exits by BUYING, so a thin or wide book is
        # worse here than for a long: the covering trade is the one that must not
        # slip, and it is the one taken under pressure.
        liquidity = context.liquidity_score
        if liquidity is not None and liquidity < self.p("min_liquidity_score"):
            return ("SHORT_EXECUTION_LIQUIDITY_INSUFFICIENT",)
        spread = context.spread_bps
        if spread is not None and spread > self.p("max_spread_bps"):
            return ("SHORT_SPREAD_TOO_WIDE",)
        return ()

    def _borrow_snapshot_fresh(self, context: ElectionContext) -> bool:
        observed = context.borrow_observed_at
        if not observed:
            return False
        elected = context.elected_at
        if elected is None:
            # Without a signal timestamp the age is unknowable, so it cannot be
            # asserted fresh.
            return False
        try:
            moment = datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        reference = elected if elected.tzinfo else elected.replace(tzinfo=timezone.utc)
        age = (reference - moment).total_seconds()
        # A NEGATIVE age means the borrow observation is timestamped after the
        # signal — i.e. it is future information relative to the decision. That is a
        # look-ahead leak, not freshness, and it is refused rather than clamped.
        return 0.0 <= age <= self.borrow_snapshot_max_age_seconds

    def _short_volatility_stop(
        self, entry_price: float, f: TechnicalFeatureSet, multiple: float
    ) -> float | None:
        """Stop ABOVE entry. The long ``_volatility_stop`` subtracts; a short adds."""
        volatility = f.realized_volatility_10s or f.realized_volatility
        if not volatility or volatility <= 0 or entry_price <= 0:
            return None
        return entry_price * (1.0 + multiple * volatility)

    def _short_target(self, entry_price: float, f: TechnicalFeatureSet) -> float | None:
        """Target BELOW entry, floored at zero."""
        if entry_price <= 0:
            return None
        return max(0.0, entry_price * (1.0 - self._volatility_edge(f) / 10_000.0))


class MarketIntradayMomentumShortAlgorithm(ShortTradingAlgorithm):
    """The NEGATIVE leg of the intraday-momentum effect.

    Gao/Han/Li/Zhou (JFE 2018) is a two-sided finding: the first half-hour return
    predicts the last half-hour return, in both signs. Only the positive leg was
    expressible before, because the negative one requires 대주 — so
    ``market_intraday_momentum`` documents itself as "long-only, so only the
    positive leg is tradable". This is the other half.

    It is NOT that algorithm with a flipped sign. The borrow fee falls only on this
    side, the volatility precondition therefore binds harder, and the outcome series
    is evaluated separately — pooling the two would let a profitable long leg carry
    an unprofitable short one.
    """

    strategy_id = "market_intraday_momentum_short"
    thesis = "a negative first half-hour continues into the last continuous half-hour"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        blocked = self._short_preconditions(f, context)
        if blocked:
            return self._reject(blocked)

        in_window = context.in_last_continuous_half_hour
        if in_window is None:
            return self._reject(("MIMS_SESSION_CONTEXT_ABSENT",))
        if not in_window:
            return self._reject(("MIMS_OUTSIDE_ENTRY_WINDOW",))
        # Never open with too little continuous trading left to COVER into. Tighter
        # than the long side's 5 minutes: an unfilled buy-to-cover leaves a borrow
        # position carried overnight, which policy forbids outright.
        remaining = context.minutes_to_continuous_close
        if remaining is not None and remaining < 8.0:
            return self._reject(
                ("MIMS_TOO_CLOSE_TO_AUCTION",), minutes_to_continuous_close=remaining
            )

        r1 = context.first_half_hour_return_bps
        if r1 is None:
            return self._reject(("MIMS_FIRST_HALF_HOUR_ABSENT",))
        drop_bps = -r1
        if drop_bps < self.p("min_first_half_hour_drop_bps"):
            return self._reject(("MIMS_FIRST_HALF_HOUR_NOT_DOWN",), first_half_hour_return_bps=r1)

        volatility_percentile = context.first_half_hour_volatility_percentile
        if volatility_percentile is None:
            return self._reject(("MIMS_VOLATILITY_CONTEXT_ABSENT",))
        if volatility_percentile < self.p("min_first_half_hour_volatility_percentile"):
            return self._reject(
                ("MIMS_DAY_NOT_VOLATILE_ENOUGH",),
                first_half_hour_volatility_percentile=volatility_percentile,
            )
        # Sell-side aggression: +1.0 default so an ABSENT imbalance fails the test
        # (mirrors the long side's -1.0 default doing the same).
        if (f.aggressor_imbalance_5s if f.aggressor_imbalance_5s is not None else 1.0) > self.p(
            "max_aggressor_imbalance"
        ):
            return self._reject(("MIMS_FLOW_NOT_SELL_SIDE",))
        breadth = context.market_breadth
        if breadth is not None and breadth > self.p("max_market_breadth"):
            return self._reject(("MIMS_MARKET_BREADTH_TOO_STRONG",), market_breadth=breadth)
        change_point = context.change_point_probability
        if change_point is not None and change_point > self.p("max_change_point_probability"):
            return self._reject(("MIMS_STRUCTURAL_BREAK",), change_point_probability=change_point)

        edge = self._volatility_edge(f)
        return self._fire(
            symbol=f.symbol,
            score=_clamp(0.5 * _clamp(drop_bps / 100.0) + 0.5 * _clamp(volatility_percentile)),
            confidence=_clamp(0.3 + 0.4 * _clamp(volatility_percentile)),
            edge_bps=edge,
            reasons=("MIMS_FIRST_HALF_HOUR_CONTINUATION_DOWN",),
            first_half_hour_return_bps=r1,
            first_half_hour_volatility_percentile=volatility_percentile,
            minutes_to_continuous_close=remaining,
            borrow_fee_bps_annualised=context.borrow_fee_bps_annualised,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        remaining = context.minutes_to_continuous_close
        horizon = self.horizon_seconds
        if remaining is not None and remaining > 0:
            # 4 minutes of slack rather than the long side's 2: a buy-to-cover that
            # does not fill cannot be left open, so the covering window is wider.
            horizon = int(min(horizon, max(60.0, (remaining - 4.0) * 60.0)))
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._short_volatility_stop(entry_price, f, 2.0),
            target_price=self._short_target(entry_price, f),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=horizon,
            stop_basis="tick_volatility_multiple_above_entry",
            target_basis="tick_volatility_expected_move_below_entry",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        remaining = context.minutes_to_continuous_close
        if remaining is not None and remaining <= 4.0:
            codes.append("MIMS_CONTINUOUS_CLOSE_IMMINENT")
        if (f.aggressor_imbalance_5s or 0.0) > 0 and (f.return_5s or 0.0) > 0:
            codes.append("MIMS_MOMENTUM_REVERSED")
        # Borrow withdrawal invalidates the position regardless of the price thesis.
        if context.borrow_available is False:
            codes.append(ShortReasonCodes.BORROW_UNAVAILABLE)
        return tuple(codes)


class OpeningRangeBreakdownAlgorithm(ShortTradingAlgorithm):
    """Break BELOW the session's opening range, on stocks in play.

    Mirror of ``opening_range_breakout`` in structure, and the relative-volume
    restriction is load-bearing for the same reason: the unrestricted break does
    not pay, and confining it to the highest-RVOL names is what carries the result.
    """

    strategy_id = "opening_range_breakdown"
    thesis = "price loses the session opening range on high relative volume and sell-side aggression"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        blocked = self._short_preconditions(f, context)
        if blocked:
            return self._reject(blocked)

        high = context.opening_range_high
        low = context.opening_range_low
        if high is None or low is None or high <= low or low <= 0:
            return self._reject(("ORBD_OPENING_RANGE_ABSENT",))
        price = f.price
        if not price or price <= 0:
            return self._reject(("ORBD_PRICE_ABSENT",))

        relative_volume = context.relative_volume
        if relative_volume is None:
            relative_volume = f.volume_spike_ratio
        if relative_volume is None:
            return self._reject(("ORBD_RELATIVE_VOLUME_ABSENT",))
        if relative_volume < self.p("min_relative_volume"):
            return self._reject(
                ("ORBD_NOT_IN_PLAY",),
                relative_volume=relative_volume,
                minimum=self.p("min_relative_volume"),
            )

        span = high - low
        # Positive when price is BELOW the range low, in bps of range width — the
        # mirror of the breakout's ``(price - high) / span``.
        excess_bps = (low - price) / span * 10_000.0
        if excess_bps < self.p("min_breakdown_excess_bps"):
            return self._reject(
                ("ORBD_RANGE_NOT_LOST",),
                excess_bps=round(excess_bps, 3),
                opening_range_low=low,
            )
        if (f.aggressor_imbalance_5s if f.aggressor_imbalance_5s is not None else 1.0) > self.p(
            "max_aggressor_imbalance"
        ):
            return self._reject(("ORBD_FLOW_NOT_SELL_SIDE",))
        if (f.spread_change_5s or 0.0) > self.p("max_spread_change_5s"):
            return self._reject(("ORBD_SPREAD_WIDENING_INTO_BREAK",))
        breadth = context.market_breadth
        if breadth is not None and breadth > self.p("max_market_breadth"):
            return self._reject(("ORBD_MARKET_BREADTH_TOO_STRONG",), market_breadth=breadth)
        change_point = context.change_point_probability
        if change_point is not None and change_point > self.p("max_change_point_probability"):
            return self._reject(("ORBD_STRUCTURAL_BREAK",), change_point_probability=change_point)

        edge = self._volatility_edge(f)
        return self._fire(
            symbol=f.symbol,
            score=_clamp(
                0.5 * _clamp(relative_volume / 3.0)
                + 0.5 * _clamp(0.5 - (f.aggressor_imbalance_5s or 0.0))
            ),
            confidence=_clamp(0.3 + 0.4 * _clamp(relative_volume / 3.0)),
            edge_bps=edge,
            reasons=("ORBD_RANGE_LOST_IN_PLAY",),
            relative_volume=relative_volume,
            excess_bps=round(excess_bps, 3),
            opening_range_high=high,
            opening_range_low=low,
            borrow_fee_bps_annualised=context.borrow_fee_bps_annualised,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        # Mirror of the published rule: stop at the OPPOSITE end of the opening
        # range, which for a breakdown is the range HIGH.
        high = context.opening_range_high
        stop = (
            high * (1.0 + self.p("stop_buffer_bps") / 10_000.0)
            if high
            else self._short_volatility_stop(entry_price, f, 2.0)
        )
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=stop,
            target_price=self._short_target(entry_price, f),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis=(
                "opening_range_high_plus_buffer" if high else "tick_volatility_multiple_above_entry"
            ),
            target_basis="tick_volatility_expected_move_below_entry",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        low = context.opening_range_low
        # Recovering back INTO the range is the definition of a failed breakdown.
        if low and f.price and f.price > low:
            codes.append("ORBD_RECOVERED_INTO_RANGE")
        if (f.aggressor_imbalance_5s or 0.0) > 0 and (f.return_5s or 0.0) > 0:
            codes.append("ORBD_FLOW_REVERSED")
        if context.borrow_available is False:
            codes.append(ShortReasonCodes.BORROW_UNAVAILABLE)
        return tuple(codes)


class ResidualRelativeWeaknessAlgorithm(ShortTradingAlgorithm):
    """Shorts the name that is weak AFTER removing market and sector beta.

    The one short thesis here that does not need the index to fall. Shorting on raw
    return in a rising tape mostly shorts low beta — i.e. it shorts the market with
    extra steps, from the wrong side. Consuming the residual

        ResidualReturn = Return - beta_market * MarketReturn - beta_sector * SectorReturn

    isolates the stock-specific offer, which is what stays valid when the index is
    up. Residuals are cross-sectional, so an absent residual fails closed rather
    than degrading to raw return — degrading would silently reproduce the defect.

    Note the residual fields are ``residual_short_bps`` / ``residual_long_bps``,
    NOT the long strategy's ``residual_return_*_bps`` with a flipped sign. Reusing
    the long measurement would make this thesis's label a deterministic function of
    the other one's, and the two would then never disagree — which is the whole
    reason for running them as separate arms.
    """

    strategy_id = "residual_relative_weakness"
    thesis = "idiosyncratic weakness net of market and sector beta persists while distribution confirms it"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)
        blocked = self._short_preconditions(f, context)
        if blocked:
            return self._reject(blocked)

        short_residual = context.residual_short_bps
        long_residual = context.residual_long_bps
        if not _present(short_residual, long_residual):
            return self._reject(("RESIDUAL_WEAKNESS_CONTEXT_ABSENT",))
        rank = context.sector_rank
        universe = context.sector_candidate_count
        if rank is None or universe is None or universe <= 1:
            return self._reject(("RESIDUAL_WEAKNESS_SECTOR_RANK_ABSENT",))
        change_point = context.change_point_probability
        if change_point is not None and change_point > self.p("max_change_point_probability"):
            return self._reject(
                ("RESIDUAL_WEAKNESS_REGIME_UNSTABLE",), change_point_probability=change_point
            )
        # Magnitudes of NEGATIVE residuals, so both horizons must be weak. One
        # window alone cannot separate persistent idiosyncratic weakness from a
        # single-window dip.
        if -short_residual < self.p("min_residual_short_weakness_bps"):
            return self._reject(
                ("RESIDUAL_SHORT_HORIZON_NOT_NEGATIVE",), residual_short_bps=short_residual
            )
        if -long_residual < self.p("min_residual_long_weakness_bps"):
            return self._reject(
                ("RESIDUAL_LONG_HORIZON_NOT_NEGATIVE",), residual_long_bps=long_residual
            )
        # Rank is counted from the WEAK end here (rank 1 == weakest residual in the
        # sector), which is why the same comparison reads correctly.
        if rank > int(self.p("max_sector_weakness_rank")):
            return self._reject(
                ("RESIDUAL_WEAKNESS_SECTOR_RANK_TOO_HIGH",), sector_rank=rank, universe=universe
            )
        relative_volume = f.relative_volume
        if relative_volume is None or relative_volume < self.p("min_relative_volume"):
            return self._reject(
                (rc.VOLUME_CONFIRMATION_MISSING, "RESIDUAL_WEAKNESS_VOLUME_NOT_CONFIRMED"),
                relative_volume=relative_volume,
            )
        if (f.aggressor_imbalance_5s if f.aggressor_imbalance_5s is not None else 1.0) > self.p(
            "max_aggressor_imbalance"
        ):
            return self._reject(("RESIDUAL_WEAKNESS_FLOW_NOT_CONFIRMED",))
        microprice_edge = f.microprice_edge_bps
        if microprice_edge is None:
            return self._reject(("RESIDUAL_WEAKNESS_MICROPRICE_UNAVAILABLE",))
        if microprice_edge > self.p("max_microprice_edge_bps"):
            return self._reject(
                ("RESIDUAL_WEAKNESS_MICROPRICE_NOT_SUPPORTIVE",),
                microprice_edge_bps=microprice_edge,
            )
        # Distribution corroboration. Required by default: idiosyncratic weakness
        # with no informed selling behind it is usually just a dip, and shorting a
        # dip is where the squeeze comes from.
        flow_scores = [
            value
            for value in (context.foreign_flow_zscore, context.institution_flow_zscore)
            if value is not None
        ]
        if self.p("require_flow_confirmation") >= 1.0:
            if not flow_scores:
                return self._reject(("RESIDUAL_WEAKNESS_INVESTOR_FLOW_ABSENT",))
            if min(flow_scores) > self.p("max_flow_zscore"):
                return self._reject(
                    ("RESIDUAL_WEAKNESS_INVESTOR_FLOW_POSITIVE",), flow_zscores=flow_scores
                )

        edge = self._volatility_edge(f)
        rank_score = _clamp(1.0 - (rank - 1) / max(1, universe - 1))
        residual_score = _clamp(-short_residual / 30.0)
        return self._fire(
            symbol=f.symbol,
            score=_clamp(
                0.45 * rank_score + 0.35 * residual_score + 0.2 * _clamp(-microprice_edge)
            ),
            confidence=_clamp(0.3 + 0.4 * rank_score + 0.2 * residual_score),
            edge_bps=edge,
            reasons=("RESIDUAL_WEAKNESS_CONFIRMED", "MARKET_AND_SECTOR_NEUTRAL_LAGGARD"),
            residual_short_bps=round(short_residual, 3),
            residual_long_bps=round(long_residual, 3),
            sector_rank=rank,
            sector_candidate_count=universe,
            microprice_edge_bps=round(microprice_edge, 4),
            market_beta=context.market_beta,
            sector_beta=context.sector_beta,
            borrow_fee_bps_annualised=context.borrow_fee_bps_annualised,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._short_volatility_stop(
                entry_price, f, self.p("stop_volatility_multiple")
            ),
            target_price=self._short_target(entry_price, f),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=self.horizon_seconds,
            stop_basis="tick_volatility_multiple_above_entry",
            target_basis="tick_volatility_expected_move_below_entry",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        codes: list[str] = []
        rank = context.sector_rank
        if rank is not None and rank > int(self.p("max_sector_weakness_rank")) + 1:
            codes.append("RESIDUAL_WEAKNESS_SECTOR_RANK_DECAYED")
        if (f.short_return or 0.0) > 0 and (f.aggressor_imbalance_5s or 0.0) > 0:
            codes.append("RESIDUAL_WEAKNESS_LOST")
        edge = f.microprice_edge_bps
        if edge is not None and edge > 0 and (f.return_5s or 0.0) > 0:
            codes.append("RESIDUAL_WEAKNESS_MICROPRICE_TURNED_BID")
        if context.borrow_available is False:
            codes.append(ShortReasonCodes.BORROW_UNAVAILABLE)
        return tuple(codes)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Overnight gap carry - the only window whose move clears a US round trip      #
# --------------------------------------------------------------------------- #
class OvernightGapCarryAlgorithm(TradingAlgorithm):
    """Carry a closing drive through the overnight gap, on a volatile day only.

    Why this thesis exists at all is arithmetic, not preference. This account's US
    round trip is 51.2bps, so ``entry_floor_bps`` demands ~61bps of expected move.
    Measured on the stored US tape, the median absolute move is 6.7bps over three
    minutes and 16.4 over thirty — a perfect direction oracle at those horizons
    still loses on 98% of trades. The overnight gap is the first window that
    clears the bar: median 69.1bps, and 62% of nights move more than the round
    trip. One position, one round trip.

    Magnitude is what was measured; direction is not. The unconditional overnight
    return on that sample is +15.6bps (n=55) — comfortably BELOW cost — so the
    gates here are not decoration. Without them the thesis is a known loser, and
    with them it is unproven, which is why the strategy ships shadow-only.

    Three preconditions, each doing distinct work:

    ``last continuous half hour``
        The carry is a decision about the CLOSE, so it can only be taken near it.
        The window comes from ``session_structure.regular_session``, i.e. 15:30-16:00
        New York for a US name — not the KRX clock every session-boxed strategy
        used to read.
    ``expected gap clears the cost floor``
        Sized from the symbol's own completed-bar volatility over an overnight
        variance-equivalent horizon, then handed to ``_fire``, which rejects it
        against the venue's round-trip cost. A quiet name is not carried.
    ``the day closed with buyers in control``
        Price above session VWAP, positive completed-bar persistence and buy-side
        aggressor flow. This is the directional half, and it is the half the local
        sample cannot yet confirm.
    """

    strategy_id = "overnight_gap_carry"
    thesis = "a closing drive on a volatile day carries through the overnight gap"

    def _expected_gap_bps(self, f: TechnicalFeatureSet) -> float:
        """Expected overnight move from completed-bar volatility.

        Wall-clock hours are the wrong scale for a gap: an overnight is not 17
        hours of trading. The knob is a VARIANCE-equivalent number of trading
        minutes, calibrated from the stored sample — median |overnight| 69.1bps
        against median |30-minute| 16.4bps is a ratio of 4.21, so the gap carries
        about 4.21^2 x 30 = 532 minutes of continuous-session variance.
        """
        volatility = max(0.0, float(f.realized_volatility or 0.0))
        if volatility <= 0.0:
            return 0.0
        return tick_expected_move_bps(
            volatility,
            int(max(60.0, self.p("overnight_variance_minutes") * 60.0)),
            window_seconds=60,
            capture_fraction=self.config.shared("capture_fraction"),
        )

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
        ready, reasons = self._tick_ready(f)
        if not ready:
            return self._reject(reasons)

        in_window = context.in_last_continuous_half_hour
        if in_window is None:
            return self._reject(("OGC_SESSION_CONTEXT_ABSENT",))
        if not in_window:
            return self._reject(("OGC_OUTSIDE_ENTRY_WINDOW",))
        remaining = context.minutes_to_continuous_close
        if remaining is not None and remaining > self.p("max_minutes_to_close"):
            # Entering at the start of the half hour would pay a full session of
            # intraday noise before reaching the gap this thesis is about.
            return self._reject(
                ("OGC_TOO_EARLY_TO_CARRY",), minutes_to_continuous_close=remaining
            )

        expected_gap = self._expected_gap_bps(f)
        if expected_gap <= 0.0:
            return self._reject(("OGC_VOLATILITY_CONTEXT_ABSENT",))

        displacement = f.vwap_distance_bps
        if displacement is None:
            return self._reject(("OGC_VWAP_CONTEXT_ABSENT",))
        if displacement < self.p("min_vwap_premium_bps"):
            # Closing BELOW the session VWAP is the sellers' day; the long leg of
            # this thesis has nothing to carry.
            return self._reject(("OGC_CLOSE_NOT_ABOVE_VWAP",), vwap_distance_bps=displacement)

        persistence = f.momentum_persistence
        if persistence is None or persistence < self.p("min_momentum_persistence"):
            return self._reject(("OGC_CLOSING_DRIVE_NOT_CONFIRMED",))
        if (f.aggressor_imbalance_5s or -1.0) < self.p("min_aggressor_imbalance"):
            return self._reject(("OGC_FLOW_NOT_BUY_SIDE",))

        change_point = context.change_point_probability
        if change_point is not None and change_point > self.p("max_change_point_probability"):
            return self._reject(("OGC_STRUCTURAL_BREAK",), change_point_probability=change_point)

        return self._fire(
            symbol=f.symbol,
            score=_clamp(0.5 * _clamp(displacement / 100.0) + 0.5 * _clamp(persistence)),
            confidence=_clamp(0.25 + 0.35 * _clamp(persistence) + 0.20 * _clamp(float(f.liquidity_score or 0.0))),
            edge_bps=expected_gap,
            reasons=("OGC_CLOSING_DRIVE_CARRIED", "OGC_GAP_CLEARS_ROUND_TRIP"),
            expected_overnight_gap_bps=round(expected_gap, 3),
            vwap_distance_bps=round(float(displacement), 3),
            momentum_persistence=round(float(persistence), 4),
            minutes_to_continuous_close=remaining,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
        # The intended exit is the next session's opening liquidity, so the clock
        # is the whole carry. Stop and target are wide for the same reason the
        # thesis exists: an overnight gap routinely travels further in one print
        # than an intraday stop allows for, and a stop inside the typical gap
        # would be jumped rather than filled.
        return ExitRule(
            strategy_id=self.strategy_id,
            stop_price=self._volatility_stop(
                entry_price, f, self.p("stop_volatility_multiple")
            ),
            target_price=entry_price * (1.0 + self._expected_gap_bps(f) / 10_000.0),
            trailing_bps=self.p("trailing_bps"),
            max_holding_seconds=int(self.p("horizon_seconds")),
            stop_basis="overnight_gap_volatility_multiple",
            target_basis="overnight_gap_expected_move",
        )

    def invalidation(self, f, context, *, entry_price=None) -> tuple[str, ...]:
        # The thesis is about the close that was carried. Once the next session is
        # open the carry is over, and holding on is a different, unstated trade.
        remaining = context.minutes_to_continuous_close
        if remaining is not None and remaining > self.p("max_minutes_to_close"):
            return ("OGC_CARRY_COMPLETE",)
        return ()


ALL_ALGORITHM_TYPES: tuple[type[TradingAlgorithm], ...] = (
    IntradayMomentumAlgorithm,
    BreakoutVolumeAlgorithm,
    VwapMeanReversionAlgorithm,
    LiquidityShockReversalAlgorithm,
    EventMomentumAlgorithm,
    CrossSectionalRelativeStrengthAlgorithm,
    GapContextAlgorithm,
    RvgiBoxBreakoutAlgorithm,
    ResidualRelativeStrengthAlgorithm,
    AdaptiveAnchoredVwapReversionAlgorithm,
    OfiMicropriceExhaustionReversalAlgorithm,
    OpeningRangeBreakoutAlgorithm,
    MarketIntradayMomentumAlgorithm,
    MarketIntradayMomentumShortAlgorithm,
    OpeningRangeBreakdownAlgorithm,
    ResidualRelativeWeaknessAlgorithm,
    # Appended so every existing GNN/catalogue output index remains unchanged.
    BarConfirmedVwapRecoveryAlgorithm,
    OvernightGapCarryAlgorithm,
)

ALGORITHM_IDS: tuple[str, ...] = tuple(kind.strategy_id for kind in ALL_ALGORITHM_TYPES)
assert ALGORITHM_IDS == STRATEGY_IDS

# The macro ontology speaks a coarser vocabulary than these algorithm ids
# (``app.graph.macro_micro_common.SelectedStrategy``). Permission checks must
# translate before comparing, otherwise every fine id looks "not allowed".
MACRO_FAMILY_BY_STRATEGY: dict[str, tuple[str, ...]] = {
    "intraday_momentum": ("momentum",),
    "event_momentum": ("momentum",),
    "cross_sectional_relative_strength": ("relative_strength", "momentum"),
    "gap_context": ("momentum", "breakout"),
    "breakout_volume": ("breakout",),
    "vwap_mean_reversion": ("vwap_reversion", "mean_reversion"),
    "liquidity_shock_reversal": ("mean_reversion",),
    # Hybrid thesis: the location is mean reversion, but the entry clock is a
    # completed-bar momentum turn. It must remain admissible in TREND_UP regimes
    # where the macro layer allows momentum but not blind mean-reversion entries.
    "bar_confirmed_vwap_recovery": (
        "vwap_reversion",
        "mean_reversion",
        "momentum",
    ),
    "rvgi_box_breakout": ("breakout", "momentum_confirmation"),
    # The residual variant is a relative-strength thesis, NOT directional
    # momentum: it is precisely the family that stays valid in a falling index,
    # which is why TREND_DOWN allows relative_strength but blocks momentum.
    "residual_relative_strength": ("relative_strength",),
    "adaptive_anchored_vwap_reversion": ("vwap_reversion", "mean_reversion"),
    "ofi_microprice_exhaustion_reversal": ("mean_reversion",),
    # --- SHORT theses -------------------------------------------------------- #
    # Mapped to SHORT-side families so the macro allow/block lists can permit a
    # falling-tape short without simultaneously permitting a long momentum entry.
    # Reusing the long families would have made TREND_DOWN's "block momentum" also
    # block the short momentum leg — the one thing TREND_DOWN most wants to allow.
    "market_intraday_momentum_short": ("momentum_short", "short"),
    "opening_range_breakdown": ("breakdown", "short"),
    # Beta-neutral, so it belongs to the relative family rather than to directional
    # shorting: it is the short thesis that survives a RISING index.
    "residual_relative_weakness": ("relative_weakness", "short"),
    # A carry is a continuation thesis, so it belongs to the momentum family and
    # is correctly blocked when the macro layer blocks momentum: carrying a long
    # position through a close is precisely what a falling tape punishes.
    "overnight_gap_carry": ("momentum",),
}


def strategy_direction(strategy_id: str) -> str:
    """``"LONG"`` or ``"SHORT"`` for a catalogued strategy id."""
    return "SHORT" if is_short_strategy(str(strategy_id or "").strip().lower()) else "LONG"


# Strategies whose deployment is gated by their own ``live_authorized`` knob.
# Everything else is live by default; these must earn it with per-regime samples.
#
# DERIVED from the defaults, never hand-listed. It used to be a literal set, and it
# fell out of sync the moment strategies were added: ``opening_range_breakout`` and
# ``market_intraday_momentum`` both declared ``live_authorized: 0.0`` and were both
# missing from the set, so ``strategy_live_authorized`` returned True for them and
# two deliberately shadow-only strategies were treated as live-tradable. Declaring
# the knob IS the gate now, so the two can no longer disagree.
_DEPLOYMENT_GATED_STRATEGIES: frozenset[str] = frozenset(
    strategy_id
    for strategy_id, values in _DEFAULTS.items()
    if strategy_id != "shared" and "live_authorized" in values
)


def macro_strategy_permitted(
    strategy_id: str,
    allowed: tuple[str, ...],
    blocked: tuple[str, ...],
) -> bool | None:
    """Is ``strategy_id`` still permitted by the macro allow/block lists?

    Returns ``None`` when the question cannot be answered — no lists supplied,
    or a strategy id with no known macro family. An unanswerable permission
    check must not be treated as a withdrawal.
    """
    strategy = str(strategy_id or "").strip().lower()
    if not strategy:
        return None
    names = {strategy, *MACRO_FAMILY_BY_STRATEGY.get(strategy, ())}
    allow_set = {str(value).strip().lower() for value in allowed if str(value).strip()}
    block_set = {str(value).strip().lower() for value in blocked if str(value).strip()}
    if not allow_set and not block_set:
        return None
    if names & block_set:
        return False
    if not allow_set:
        return True
    return bool(names & allow_set)


def build_algorithm_registry(
    config: AlgorithmConfig | None = None,
) -> dict[str, TradingAlgorithm]:
    shared = config or AlgorithmConfig()
    return {kind.strategy_id: kind(shared) for kind in ALL_ALGORITHM_TYPES}


def get_algorithm(
    strategy_id: str,
    *,
    registry: Mapping[str, TradingAlgorithm] | None = None,
) -> TradingAlgorithm | None:
    source = registry if registry is not None else build_algorithm_registry()
    return source.get(str(strategy_id or "").strip().lower())


def strategy_live_authorized(strategy_id: str) -> bool:
    """Deployment flag only; model/ontology authorization remains separate."""
    algorithm = get_algorithm(strategy_id)
    if algorithm is None:
        return False
    if strategy_id not in _DEPLOYMENT_GATED_STRATEGIES:
        return True
    return algorithm.p("enabled") >= 1.0 and algorithm.p("live_authorized") >= 1.0


def strategy_shadow_authorized(strategy_id: str) -> bool:
    """May this strategy be evaluated and journaled without a live order?

    A deployment-gated strategy that is not yet live-authorized is still expected
    to run in shadow — that is how it accumulates the per-regime samples the
    conservative bandit needs before it can ever be selected live.
    """
    algorithm = get_algorithm(strategy_id)
    if algorithm is None:
        return False
    if strategy_id not in _DEPLOYMENT_GATED_STRATEGIES:
        return True
    return algorithm.p("enabled") >= 1.0 and algorithm.p("shadow_enabled") >= 1.0
