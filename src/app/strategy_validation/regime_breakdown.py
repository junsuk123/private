"""Where a strategy works, and whether "where" is even measurable.

Why a separate module from ``metrics.market_breakdown``
------------------------------------------------------
That breakdown is a mean per bucket. What a lifecycle decision needs is whether the buckets
DIFFER by more than sampling noise would produce — otherwise "works in TREND_UP, fails in
RANGE_BOUND" is a story told about two samples of eight.

So this module reports, per bucket, the mean AND its lower confidence bound, and then a
single ``discriminates`` verdict: do the best and worst buckets' confidence intervals fail to
overlap? Only then is a regime-conditional claim supportable, and only then should the
ontology's eligibility relations be tightened on the strength of it.

The project has already been burned by the alternative: a condition scored t=3.01 across all
symbols and t=0.65 on the sub-period where the data actually lives.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from app.strategy_validation.metrics import TradeObservation, Z_95, effective_sample_count

__all__ = [
    "BucketStats",
    "RegimeBreakdown",
    "regime_breakdown",
]


@dataclass(frozen=True)
class BucketStats:
    bucket: str
    sample_count: int
    effective_sample_count: float
    net_ev_bps: float | None
    lower_confidence_bound_bps: float | None
    upper_confidence_bound_bps: float | None
    hit_rate: float | None

    @property
    def interval(self) -> tuple[float, float] | None:
        if (
            self.lower_confidence_bound_bps is None
            or self.upper_confidence_bound_bps is None
        ):
            return None
        return (self.lower_confidence_bound_bps, self.upper_confidence_bound_bps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "sample_count": self.sample_count,
            "effective_sample_count": round(self.effective_sample_count, 3),
            "net_ev_bps": _round(self.net_ev_bps),
            "lower_confidence_bound_bps": _round(self.lower_confidence_bound_bps),
            "upper_confidence_bound_bps": _round(self.upper_confidence_bound_bps),
            "hit_rate": _round(self.hit_rate, 4),
        }


@dataclass(frozen=True)
class RegimeBreakdown:
    strategy_id: str
    dimension: str
    buckets: tuple[BucketStats, ...]
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def measurable(self) -> tuple[BucketStats, ...]:
        return tuple(item for item in self.buckets if item.interval is not None)

    @property
    def best(self) -> BucketStats | None:
        scored = [item for item in self.measurable]
        return max(scored, key=lambda item: item.net_ev_bps or 0.0) if scored else None

    @property
    def worst(self) -> BucketStats | None:
        scored = [item for item in self.measurable]
        return min(scored, key=lambda item: item.net_ev_bps or 0.0) if scored else None

    @property
    def discriminates(self) -> bool | None:
        """Do the best and worst buckets' confidence intervals fail to overlap?

        ``None`` when fewer than two buckets are measurable — that is "we cannot tell",
        which is a different answer from "no difference" and must not be reported as one.
        """
        best, worst = self.best, self.worst
        if best is None or worst is None or best.bucket == worst.bucket:
            return None
        best_interval, worst_interval = best.interval, worst.interval
        if best_interval is None or worst_interval is None:
            return None
        return best_interval[0] > worst_interval[1]

    @property
    def positive_buckets(self) -> tuple[str, ...]:
        """Buckets whose LOWER bound clears zero — the only defensible 'works here'."""
        return tuple(
            item.bucket
            for item in self.buckets
            if item.lower_confidence_bound_bps is not None
            and item.lower_confidence_bound_bps > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "dimension": self.dimension,
            "discriminates": self.discriminates,
            "positive_buckets": list(self.positive_buckets),
            "best_bucket": self.best.bucket if self.best else None,
            "worst_bucket": self.worst.bucket if self.worst else None,
            "buckets": [item.as_dict() for item in self.buckets],
            "reason_codes": list(self.reason_codes),
        }


def regime_breakdown(
    strategy_id: str,
    trades: Sequence[TradeObservation],
    *,
    dimension: str = "regime",
    key: Callable[[TradeObservation], str] | None = None,
    minimum_samples: int = 8,
) -> RegimeBreakdown:
    """Per-bucket net EV with confidence bounds, and whether the buckets differ."""
    resolve = key or _default_key(dimension)
    grouped: dict[str, list[TradeObservation]] = {}
    for trade in trades:
        grouped.setdefault(str(resolve(trade) or "UNKNOWN"), []).append(trade)

    reasons: list[str] = []
    buckets: list[BucketStats] = []
    for bucket, rows in sorted(grouped.items()):
        nets = [row.net_return_bps for row in rows]
        effective = effective_sample_count(rows)
        mean = sum(nets) / len(nets)
        stdev = statistics.stdev(nets) if len(nets) >= 2 else None
        lower = upper = None
        if stdev is not None and effective > 0 and len(rows) >= minimum_samples:
            margin = Z_95 * stdev / math.sqrt(effective)
            lower, upper = mean - margin, mean + margin
        elif len(rows) < minimum_samples:
            reasons.append(f"REGIME_BUCKET_BELOW_{minimum_samples}:{bucket}")
        buckets.append(
            BucketStats(
                bucket=bucket,
                sample_count=len(rows),
                effective_sample_count=effective,
                net_ev_bps=mean,
                lower_confidence_bound_bps=lower,
                upper_confidence_bound_bps=upper,
                hit_rate=sum(1 for value in nets if value > 0) / len(nets),
            )
        )
    if not buckets:
        reasons.append("REGIME_BREAKDOWN_NO_TRADES")
    return RegimeBreakdown(
        strategy_id=strategy_id,
        dimension=dimension,
        buckets=tuple(buckets),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _default_key(dimension: str) -> Callable[[TradeObservation], str]:
    name = str(dimension or "regime").strip().lower()
    if name == "market":
        return lambda trade: trade.market
    if name == "session":
        return lambda trade: trade.session_phase
    if name == "market_regime":
        return lambda trade: f"{trade.market}|{trade.regime}"
    return lambda trade: trade.regime


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None
