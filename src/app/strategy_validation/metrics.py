"""Strategy metrics, with an explicit rule: no number is invented.

Every metric returns ``None`` when the data cannot support it. That is the whole design
constraint — a validation framework whose empty cells read as zeros produces confident
verdicts about strategies it has never measured, and this catalogue has several such
strategies right now (``market_intraday_momentum`` has produced no training row at all).

The confidence interval and the lower bound
------------------------------------------
``lower_confidence_bound_bps`` is a one-sided normal bound on the MEAN:

    mean - z * stdev / sqrt(n_effective)

``n_effective`` is not the row count. Trades that overlap in time are not independent
observations — the project has already measured what ignoring that does: a forward-return
analysis with ``stride < horizon`` inflated n by 56x. So overlapping trades are down-weighted
by the fraction of their horizon that does not overlap, and the effective count is reported
alongside the raw one so the difference is visible.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "StrategyMetrics",
    "TradeObservation",
    "compute_metrics",
    "effective_sample_count",
]

#: One-sided 95% normal quantile.
Z_95 = 1.645


@dataclass(frozen=True)
class TradeObservation:
    """One evaluated trade, real or simulated."""

    strategy_id: str
    symbol: str
    market: str
    regime: str
    opened_at: datetime
    closed_at: datetime
    gross_return_bps: float
    net_return_bps: float
    cost_bps: float
    max_adverse_excursion_bps: float | None = None
    max_favorable_excursion_bps: float | None = None
    evidence_source: str = "SHADOW"
    session_phase: str = "unknown"
    predicted_net_bps: float | None = None
    predicted_probability: float | None = None
    parameters: Mapping[str, float] = field(default_factory=dict)

    @property
    def holding_seconds(self) -> float:
        return max(0.0, (_aware(self.closed_at) - _aware(self.opened_at)).total_seconds())

    @property
    def is_win(self) -> bool:
        return self.net_return_bps > 0.0


@dataclass(frozen=True)
class StrategyMetrics:
    """The full metric set. ``None`` everywhere the data cannot answer."""

    strategy_id: str
    trigger_count: int
    effective_sample_count: float
    gross_ev_bps: float | None
    net_ev_bps: float | None
    hit_rate: float | None
    profit_factor: float | None
    max_drawdown_bps: float | None
    mean_adverse_excursion_bps: float | None
    mean_favorable_excursion_bps: float | None
    mean_holding_seconds: float | None
    cost_to_edge_ratio: float | None
    turnover_per_day: float | None
    confidence_interval_bps: tuple[float, float] | None
    lower_confidence_bound_bps: float | None
    prediction_calibration: Mapping[str, float | None] = field(default_factory=dict)
    market_breakdown: Mapping[str, float | None] = field(default_factory=dict)
    regime_breakdown: Mapping[str, float | None] = field(default_factory=dict)
    session_breakdown: Mapping[str, float | None] = field(default_factory=dict)
    evidence_mix: Mapping[str, int] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "trigger_count": self.trigger_count,
            "effective_sample_count": round(self.effective_sample_count, 3),
            "gross_ev_bps": _round(self.gross_ev_bps),
            "net_ev_bps": _round(self.net_ev_bps),
            "hit_rate": _round(self.hit_rate, 4),
            "profit_factor": _round(self.profit_factor, 3),
            "max_drawdown_bps": _round(self.max_drawdown_bps),
            "MAE_bps": _round(self.mean_adverse_excursion_bps),
            "MFE_bps": _round(self.mean_favorable_excursion_bps),
            "mean_holding_seconds": _round(self.mean_holding_seconds, 1),
            "cost_to_edge_ratio": _round(self.cost_to_edge_ratio, 3),
            "turnover_per_day": _round(self.turnover_per_day, 3),
            "confidence_interval_bps": (
                [round(self.confidence_interval_bps[0], 3), round(self.confidence_interval_bps[1], 3)]
                if self.confidence_interval_bps
                else None
            ),
            "lower_confidence_bound_bps": _round(self.lower_confidence_bound_bps),
            "prediction_calibration": {
                key: _round(value, 4) for key, value in dict(self.prediction_calibration).items()
            },
            "market_breakdown": {
                key: _round(value) for key, value in dict(self.market_breakdown).items()
            },
            "regime_breakdown": {
                key: _round(value) for key, value in dict(self.regime_breakdown).items()
            },
            "session_breakdown": {
                key: _round(value) for key, value in dict(self.session_breakdown).items()
            },
            "evidence_mix": dict(self.evidence_mix),
            "reason_codes": list(self.reason_codes),
        }


def effective_sample_count(trades: Sequence[TradeObservation]) -> float:
    """Sample count discounted for temporal overlap within a symbol.

    Two trades on the same symbol whose holding periods overlap are largely the same price
    path observed twice. Each trade contributes the fraction of its own horizon that is NOT
    covered by an earlier trade on that symbol, floored at a small positive value so a
    fully-overlapped trade still counts for something rather than vanishing.

    Trades on DIFFERENT symbols are treated as independent here. They are not perfectly
    independent (a market-wide move touches all of them) but discounting for that needs a
    cross-sectional correlation estimate the framework does not have, and pretending to one
    would be worse than a stated approximation.
    """
    if not trades:
        return 0.0
    by_symbol: dict[str, list[TradeObservation]] = {}
    for trade in trades:
        by_symbol.setdefault(trade.symbol, []).append(trade)
    total = 0.0
    for rows in by_symbol.values():
        ordered = sorted(rows, key=lambda item: _aware(item.opened_at))
        covered_until: datetime | None = None
        for trade in ordered:
            start = _aware(trade.opened_at)
            end = _aware(trade.closed_at)
            span = max(1.0, (end - start).total_seconds())
            if covered_until is None or start >= covered_until:
                fresh = span
            else:
                fresh = max(0.0, (end - covered_until).total_seconds())
            total += max(0.05, min(1.0, fresh / span))
            covered_until = end if covered_until is None else max(covered_until, end)
    return total


def compute_metrics(
    strategy_id: str,
    trades: Sequence[TradeObservation],
    *,
    minimum_samples: int = 20,
) -> StrategyMetrics:
    """Full metric set for one strategy over one set of trades."""
    reasons: list[str] = []
    if not trades:
        return StrategyMetrics(
            strategy_id=strategy_id,
            trigger_count=0,
            effective_sample_count=0.0,
            gross_ev_bps=None,
            net_ev_bps=None,
            hit_rate=None,
            profit_factor=None,
            max_drawdown_bps=None,
            mean_adverse_excursion_bps=None,
            mean_favorable_excursion_bps=None,
            mean_holding_seconds=None,
            cost_to_edge_ratio=None,
            turnover_per_day=None,
            confidence_interval_bps=None,
            lower_confidence_bound_bps=None,
            reason_codes=("METRICS_NO_TRADES",),
        )

    nets = [trade.net_return_bps for trade in trades]
    grosses = [trade.gross_return_bps for trade in trades]
    costs = [trade.cost_bps for trade in trades]
    wins = [value for value in nets if value > 0]
    losses = [-value for value in nets if value < 0]

    net_ev = sum(nets) / len(nets)
    gross_ev = sum(grosses) / len(grosses)
    effective = effective_sample_count(trades)
    if len(trades) < minimum_samples:
        reasons.append(f"METRICS_SAMPLE_BELOW_{minimum_samples}")
    if effective < len(trades) * 0.75:
        # Worth saying out loud: the raw count overstates the evidence here.
        reasons.append("METRICS_OVERLAPPING_TRADES_DISCOUNTED")

    stdev = statistics.stdev(nets) if len(nets) >= 2 else None
    interval: tuple[float, float] | None = None
    lower_bound: float | None = None
    if stdev is not None and effective > 0:
        margin = Z_95 * stdev / math.sqrt(effective)
        interval = (net_ev - margin, net_ev + margin)
        lower_bound = net_ev - margin
    else:
        reasons.append("METRICS_NO_INTERVAL_SINGLE_SAMPLE")

    span_days = _span_days(trades)
    adverse = [
        trade.max_adverse_excursion_bps
        for trade in trades
        if trade.max_adverse_excursion_bps is not None
    ]
    favourable = [
        trade.max_favorable_excursion_bps
        for trade in trades
        if trade.max_favorable_excursion_bps is not None
    ]
    evidence_mix: dict[str, int] = {}
    for trade in trades:
        evidence_mix[trade.evidence_source] = evidence_mix.get(trade.evidence_source, 0) + 1
    if not any(
        source.upper() in {"LIVE", "LIVE_PROBE"} for source in evidence_mix
    ):
        reasons.append("METRICS_NO_LIVE_EVIDENCE")

    return StrategyMetrics(
        strategy_id=strategy_id,
        trigger_count=len(trades),
        effective_sample_count=effective,
        gross_ev_bps=gross_ev,
        net_ev_bps=net_ev,
        hit_rate=len(wins) / len(nets),
        profit_factor=(sum(wins) / sum(losses)) if losses and sum(losses) > 0 else None,
        max_drawdown_bps=_max_drawdown_bps(nets),
        mean_adverse_excursion_bps=(sum(adverse) / len(adverse)) if adverse else None,
        mean_favorable_excursion_bps=(
            sum(favourable) / len(favourable) if favourable else None
        ),
        mean_holding_seconds=sum(trade.holding_seconds for trade in trades) / len(trades),
        cost_to_edge_ratio=(
            (sum(costs) / len(costs)) / abs(gross_ev) if gross_ev not in (0.0,) else None
        ),
        turnover_per_day=(len(trades) / span_days) if span_days and span_days > 0 else None,
        confidence_interval_bps=interval,
        lower_confidence_bound_bps=lower_bound,
        prediction_calibration=_calibration(trades),
        market_breakdown=_breakdown(trades, lambda trade: trade.market),
        regime_breakdown=_breakdown(trades, lambda trade: trade.regime),
        session_breakdown=_breakdown(trades, lambda trade: trade.session_phase),
        evidence_mix=evidence_mix,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _calibration(trades: Sequence[TradeObservation]) -> dict[str, float | None]:
    predicted = [
        (trade.predicted_net_bps, trade.net_return_bps)
        for trade in trades
        if trade.predicted_net_bps is not None
    ]
    probabilities = [
        (trade.predicted_probability, 1.0 if trade.is_win else 0.0)
        for trade in trades
        if trade.predicted_probability is not None
    ]
    return {
        "net_mae_bps": (
            sum(abs(pred - actual) for pred, actual in predicted) / len(predicted)
            if predicted
            else None
        ),
        "net_bias_bps": (
            sum(pred - actual for pred, actual in predicted) / len(predicted)
            if predicted
            else None
        ),
        "sign_accuracy": (
            sum(1 for pred, actual in predicted if (pred > 0) == (actual > 0)) / len(predicted)
            if predicted
            else None
        ),
        "brier_score": (
            sum((pred - actual) ** 2 for pred, actual in probabilities) / len(probabilities)
            if probabilities
            else None
        ),
    }


def _breakdown(
    trades: Sequence[TradeObservation], key: Any
) -> dict[str, float | None]:
    grouped: dict[str, list[float]] = {}
    for trade in trades:
        grouped.setdefault(str(key(trade) or "UNKNOWN"), []).append(trade.net_return_bps)
    return {
        # A single-trade bucket reports ``None`` rather than that trade's return: one
        # observation is not a mean, and printing it as one invites reading it as one.
        name: (sum(values) / len(values)) if len(values) >= 2 else None
        for name, values in sorted(grouped.items())
    }


def _span_days(trades: Sequence[TradeObservation]) -> float | None:
    if len(trades) < 2:
        return None
    starts = [_aware(trade.opened_at) for trade in trades]
    span = (max(starts) - min(starts)).total_seconds() / 86_400.0
    return span if span > 0 else None


def _max_drawdown_bps(values: Sequence[float]) -> float | None:
    if not values:
        return None
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None
