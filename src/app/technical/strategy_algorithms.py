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
)

ALGORITHM_IDS: tuple[str, ...] = tuple(kind.strategy_id for kind in ALL_ALGORITHM_TYPES)

# The macro ontology speaks a coarser vocabulary than these algorithm ids
# (``app.graph.macro_micro_common.SelectedStrategy``). Permission checks must
# translate before comparing, otherwise every fine id looks "not allowed".
MACRO_FAMILY_BY_STRATEGY: dict[str, tuple[str, ...]] = {
    "intraday_momentum": ("momentum",),
    "event_momentum": ("momentum",),
    "cross_sectional_relative_strength": ("momentum",),
    "gap_context": ("momentum", "breakout"),
    "breakout_volume": ("breakout",),
    "vwap_mean_reversion": ("vwap_reversion", "mean_reversion"),
    "liquidity_shock_reversal": ("mean_reversion",),
}


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
