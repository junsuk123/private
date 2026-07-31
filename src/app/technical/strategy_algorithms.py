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
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from app.technical import reason_codes as rc
from app.technical.signals import TechnicalFeatureSet
from app.strategy.catalog import STRATEGY_IDS

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
# Configuration                                                                #
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, dict[str, float]] = {
    "shared": {
        "capture_fraction": 0.5,
        "min_tick_count_5s": 3.0,
        "min_expected_edge_bps": 8.0,
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
        "acceptance_return_1s_bps": 0.0,
        "min_aggressor_imbalance": 0.05,
        "horizon_seconds": 300.0,
        "stop_buffer_bps": 8.0,
        "trailing_bps": 15.0,
        "failure_tolerance_bps": 10.0,
    },
    "vwap_mean_reversion": {
        "entry_deviation_bps": 25.0,
        "max_rsi": 38.0,
        "max_percent_b": 0.22,
        "horizon_seconds": 240.0,
        "stop_volatility_multiple": 2.0,
        "target_capture_fraction": 0.7,
        "trailing_bps": 10.0,
    },
    "liquidity_shock_reversal": {
        "shock_return_10s_bps": -40.0,
        "max_spread_change_5s": 0.0,
        "min_aggressor_imbalance": -0.35,
        "min_orderbook_imbalance": 0.0,
        "horizon_seconds": 120.0,
        "retrace_fraction": 0.4,
        "stop_buffer_bps": 6.0,
        "trailing_bps": 14.0,
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
        "live_authorized": 0.0,
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
    # All three ship with live_authorized = 0: they run in shadow until each has
    # accumulated enough per-regime samples in the strategy performance store for
    # the conservative bandit to give them a positive lower bound.
    "residual_relative_strength": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        "live_authorized": 0.0,
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
        "live_authorized": 0.0,
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
        "live_authorized": 0.0,
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
    "opening_range_breakout": {
        "enabled": 1.0,
        "shadow_enabled": 1.0,
        "paper_enabled": 1.0,
        # Live authorisation is withheld until this strategy has its own realized
        # history, exactly like every other newly added algorithm here.
        "live_authorized": 0.0,
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
        return tick_expected_move_bps(
            f.realized_volatility_10s,
            self.horizon_seconds,
            capture_fraction=self.config.shared("capture_fraction"),
        )

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

    def _fire(
        self,
        *,
        score: float,
        confidence: float,
        edge_bps: float,
        reasons: tuple[str, ...],
        **diagnostics: Any,
    ) -> AlgorithmDecision:
        minimum = self.config.shared("min_expected_edge_bps")
        if edge_bps < minimum:
            return self._reject(
                (*reasons, rc.TECHNICAL_EDGE_NON_POSITIVE, "EDGE_BELOW_ALGORITHM_FLOOR"),
                expected_edge_bps=round(edge_bps, 3),
                minimum_edge_bps=minimum,
                **diagnostics,
            )
        return AlgorithmDecision(
            strategy_id=self.strategy_id,
            triggered=True,
            score=_clamp(score),
            confidence=_clamp(confidence),
            expected_edge_bps=edge_bps,
            horizon_seconds=self.horizon_seconds,
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
        if f.breakout_strength < 0:
            return self._reject(("PRICE_BELOW_BREAKOUT_LEVEL",), breakout_strength=f.breakout_strength)
        volume_ratio = f.volume_spike_ratio
        if volume_ratio is None or volume_ratio < self.p("min_volume_spike_ratio"):
            return self._reject((rc.VOLUME_CONFIRMATION_MISSING,), volume_spike_ratio=volume_ratio)
        # Acceptance: the break must not be fading back on the very next ticks.
        return_1s_bps = (f.return_1s or 0.0) * 10_000.0
        aggressor = f.aggressor_imbalance_5s
        if return_1s_bps < self.p("acceptance_return_1s_bps"):
            return self._reject((rc.FALSE_BREAKOUT_RISK_HIGH, "BREAKOUT_FADING_ON_TICKS"))
        if aggressor is None or aggressor < self.p("min_aggressor_imbalance"):
            return self._reject((rc.FALSE_BREAKOUT_RISK_HIGH, "BREAKOUT_NOT_FLOW_CONFIRMED"))

        edge = self._volatility_edge(f)
        score = _clamp(0.5 * _clamp(volume_ratio / 3.0) + 0.5 * _clamp(aggressor))
        return self._fire(
            score=score,
            confidence=_clamp(0.35 + 0.4 * _clamp(volume_ratio / 3.0) + 0.25 * _clamp(aggressor)),
            edge_bps=edge,
            reasons=(rc.BREAKOUT_CONFIRMED, "BREAKOUT_ACCEPTED_ON_TICKS"),
            donchian_high=f.donchian_high,
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
        oversold = (f.rsi is not None and f.rsi <= self.p("max_rsi")) or (
            f.bb_percent_b is not None and f.bb_percent_b <= self.p("max_percent_b")
        )
        if not oversold:
            return self._reject(("DISPLACEMENT_NOT_OVERSOLD",), rsi=f.rsi, percent_b=f.bb_percent_b)
        # Stabilisation: never buy while the drop is still accelerating.
        if (f.return_1s or 0.0) < 0:
            return self._reject(("STILL_FALLING_ON_TICKS",))
        if (f.orderbook_imbalance_change_5s or 0.0) < 0:
            return self._reject(("SELL_PRESSURE_STILL_WORSENING",))

        # Target is structural (VWAP), so the edge is the captured distance.
        edge = abs(deviation) * self.p("target_capture_fraction")
        score = _clamp(abs(deviation) / (2.0 * self.p("entry_deviation_bps")))
        return self._fire(
            score=score,
            confidence=_clamp(0.35 + 0.45 * score),
            edge_bps=edge,
            reasons=(rc.MEAN_REVERSION_CANDIDATE, "VWAP_DISPLACEMENT_STABILISED"),
            vwap=f.vwap,
            vwap_distance_bps=round(deviation, 3),
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
class LiquidityShockReversalAlgorithm(TradingAlgorithm):
    strategy_id = "liquidity_shock_reversal"
    thesis = "a mechanical sub-minute drop partially retraces once spread and depth normalise"

    def entry(self, f: TechnicalFeatureSet, context: ElectionContext) -> AlgorithmDecision:
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
            score=score,
            confidence=_clamp(0.3 + 0.5 * score),
            edge_bps=edge,
            reasons=("LIQUIDITY_SHOCK_STABILISED", "SPREAD_CONTRACTING"),
            return_10s_bps=round(shock_bps, 3),
            spread_change_5s=f.spread_change_5s,
            orderbook_imbalance=f.orderbook_imbalance,
        )

    def exit_rule(self, entry_price, f, context) -> ExitRule:
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
# Registry                                                                     #
# --------------------------------------------------------------------------- #
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
    "rvgi_box_breakout": ("breakout", "momentum_confirmation"),
    # The residual variant is a relative-strength thesis, NOT directional
    # momentum: it is precisely the family that stays valid in a falling index,
    # which is why TREND_DOWN allows relative_strength but blocks momentum.
    "residual_relative_strength": ("relative_strength",),
    "adaptive_anchored_vwap_reversion": ("vwap_reversion", "mean_reversion"),
    "ofi_microprice_exhaustion_reversal": ("mean_reversion",),
}


# Strategies whose deployment is gated by their own ``live_authorized`` knob.
# Everything else is live by default; these must earn it with per-regime samples.
_DEPLOYMENT_GATED_STRATEGIES: frozenset[str] = frozenset(
    {
        "rvgi_box_breakout",
        "residual_relative_strength",
        "adaptive_anchored_vwap_reversion",
        "ofi_microprice_exhaustion_reversal",
    }
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
