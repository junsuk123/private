"""Does a strategy exist for the state the market is actually in?

``strategy_space_coverage`` is the first root cause in the problem statement: the context
space is large and the catalogue may not span it. Nineteen strategies sound like broad
coverage until you notice five of them are VWAP/range reversions and none is defined in a
low-liquidity close.

This module buckets a context along six axes and reports, per bucket, how many strategies
are *eligible*, how many *fired*, and how many of those that fired have the validation
standing to be selected. The third number is the one that matters: a bucket where
strategies fire but none is validated is a bucket the system trades on hope.

The rule that follows
---------------------
``validated_positive_strategy_count == 0`` records a ``STRATEGY_COVERAGE_GAP`` and the
bucket resolves to NO_TRADE. The nearest existing strategy is NOT forced — that is the
``forced_selection`` failure, and filling a gap with an unvalidated look-alike is the
``prohibited`` item in the strategy-addition policy. Repeated gaps accumulate as research
candidates, which is the legitimate way the catalogue grows.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.context.market_context import MarketContext

__all__ = [
    "COVERAGE_GAP_REASON",
    "CONTEXT_DIMENSIONS",
    "ContextBucket",
    "CoverageObservation",
    "StrategyCoverageAnalyzer",
    "bucket_for_context",
]

COVERAGE_GAP_REASON = "STRATEGY_COVERAGE_GAP"

#: The bucketing axes. Values are ordered from adverse to favourable where that is
#: meaningful, so a report reads left to right.
CONTEXT_DIMENSIONS: Mapping[str, tuple[str, ...]] = {
    "trend": ("strong_down", "weak_down", "flat", "weak_up", "strong_up"),
    "volatility": ("low", "medium", "high", "extreme"),
    "liquidity": ("low", "medium", "high"),
    "session": ("open", "morning", "midday", "close", "overnight"),
    "event": ("none", "positive", "negative", "uncertain"),
    "microstructure": ("balanced", "buy_pressure", "sell_pressure", "liquidity_shock"),
}

#: Cut points, in the units of the context fields they read.
#:
#: ``trend`` is bucketed on ``trend_strength`` (EMA separation in bps of price), because
#: that is scale-free across a 70,000 KRW name and a 40 USD one. +-10bps is roughly one
#: typical KRX top-of-book spread, so "flat" means "inside the noise".
_TREND_BPS_CUTS: tuple[float, ...] = (-40.0, -10.0, 10.0, 40.0)
#: ``volatility`` on per-observation realised volatility as a fraction. 0.001 = 10bps per
#: observation; 0.005 = 50bps, which on a one-minute bar is an extreme tape.
_VOLATILITY_CUTS: tuple[float, ...] = (0.001, 0.0025, 0.005)
_LIQUIDITY_CUTS: tuple[float, ...] = (0.35, 0.65)
#: ``microstructure`` on order-flow imbalance in [-1, 1]. +-0.15 matches the lowest
#: ``min_aggressor_imbalance`` any algorithm in the catalogue fires on.
_FLOW_CUTS: tuple[float, ...] = (-0.15, 0.15)
#: A liquidity shock is a spread blowing out relative to the volatility being captured.
_SHOCK_PRICE_IMPACT = 3.0


@dataclass(frozen=True)
class ContextBucket:
    """One cell of the coverage matrix."""

    trend: str
    volatility: str
    liquidity: str
    session: str
    event: str
    microstructure: str

    def as_key(self) -> str:
        return "|".join(
            (
                self.trend,
                self.volatility,
                self.liquidity,
                self.session,
                self.event,
                self.microstructure,
            )
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "trend": self.trend,
            "volatility": self.volatility,
            "liquidity": self.liquidity,
            "session": self.session,
            "event": self.event,
            "microstructure": self.microstructure,
        }

    @classmethod
    def from_key(cls, key: str) -> "ContextBucket":
        parts = str(key or "").split("|")
        if len(parts) != 6:
            raise ValueError(f"malformed coverage bucket key: {key!r}")
        return cls(*parts)


def _bucket(value: float | None, cuts: Sequence[float], labels: Sequence[str]) -> str:
    """Label for ``value`` against ascending ``cuts``; last label when above all cuts."""
    if value is None or not math.isfinite(float(value)):
        # There is no "unknown" label in the declared dimension values, and inventing one
        # would make every incomplete context its own bucket. The middle label is used and
        # the context's own ``feature_completeness`` is what records the doubt.
        return labels[len(labels) // 2]
    number = float(value)
    for index, cut in enumerate(cuts):
        if number < cut:
            return labels[index]
    return labels[-1]


def _session_label(context: MarketContext) -> str:
    phase = str(context.temporal.session_phase or "").strip().lower()
    if phase in {"closed", "after", "pre"}:
        return "overnight"
    if context.temporal.is_opening_window:
        return "open"
    if context.temporal.is_closing_window:
        return "close"
    minutes = context.temporal.minutes_from_open
    if minutes is None:
        return "midday"
    return "morning" if minutes <= 120.0 else "midday"


def _event_label(context: MarketContext) -> str:
    positive = context.event.positive_event_score or 0.0
    negative = context.event.negative_event_score or 0.0
    uncertainty = context.event.event_uncertainty or 0.0
    if positive <= 0.0 and negative <= 0.0:
        return "none"
    if uncertainty >= max(positive, negative):
        return "uncertain"
    return "positive" if positive >= negative else "negative"


def _microstructure_label(context: MarketContext) -> str:
    impact = context.microstructure.short_term_price_impact
    if impact is not None and impact >= _SHOCK_PRICE_IMPACT:
        # Spread per unit of capturable volatility this high means the book is not
        # absorbing; that is a shock regardless of which side the flow is on.
        return "liquidity_shock"
    flow = context.microstructure.orderflow_imbalance
    if flow is None:
        return "balanced"
    if flow < _FLOW_CUTS[0]:
        return "sell_pressure"
    if flow > _FLOW_CUTS[1]:
        return "buy_pressure"
    return "balanced"


def bucket_for_context(context: MarketContext) -> ContextBucket:
    """Deterministic bucket for one context. No IO, no clock."""
    return ContextBucket(
        trend=_bucket(
            context.symbol.trend_strength, _TREND_BPS_CUTS, CONTEXT_DIMENSIONS["trend"]
        ),
        volatility=_bucket(
            context.symbol.realized_volatility,
            _VOLATILITY_CUTS,
            CONTEXT_DIMENSIONS["volatility"],
        ),
        liquidity=_bucket(
            context.microstructure.liquidity_score,
            _LIQUIDITY_CUTS,
            CONTEXT_DIMENSIONS["liquidity"],
        ),
        session=_session_label(context),
        event=_event_label(context),
        microstructure=_microstructure_label(context),
    )


@dataclass
class CoverageObservation:
    """Accumulated counts for one bucket. Mutable: it is a running tally."""

    bucket: ContextBucket
    observations: int = 0
    eligible_strategy_count: int = 0
    entry_ready_strategy_count: int = 0
    validated_positive_strategy_count: int = 0
    no_trade_count: int = 0
    best_net_bps_total: float = 0.0
    best_net_bps_samples: int = 0
    #: Strategy ids that were ever entry-ready here, for the research-candidate report.
    contributing_strategies: Counter = field(default_factory=Counter)
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None

    @property
    def mean_eligible(self) -> float:
        return self.eligible_strategy_count / self.observations if self.observations else 0.0

    @property
    def mean_entry_ready(self) -> float:
        return (
            self.entry_ready_strategy_count / self.observations if self.observations else 0.0
        )

    @property
    def mean_validated_positive(self) -> float:
        return (
            self.validated_positive_strategy_count / self.observations
            if self.observations
            else 0.0
        )

    @property
    def no_trade_rate(self) -> float:
        return self.no_trade_count / self.observations if self.observations else 0.0

    @property
    def best_strategy_expected_net_bps(self) -> float | None:
        """Mean of the best candidate's net edge. ``None`` when never measurable.

        ``None`` rather than 0.0: a bucket in which no candidate was ever priced has an
        UNKNOWN best edge, and reporting zero would read as "measured break-even".
        """
        if self.best_net_bps_samples <= 0:
            return None
        return self.best_net_bps_total / self.best_net_bps_samples

    @property
    def is_coverage_gap(self) -> bool:
        """No observation in this bucket ever produced a selectable, validated strategy."""
        return self.observations > 0 and self.validated_positive_strategy_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.bucket.as_dict(),
            "bucket": self.bucket.as_key(),
            "observations": self.observations,
            "eligible_strategy_count": self.eligible_strategy_count,
            "entry_ready_strategy_count": self.entry_ready_strategy_count,
            "validated_positive_strategy_count": self.validated_positive_strategy_count,
            "mean_eligible_per_observation": round(self.mean_eligible, 3),
            "mean_entry_ready_per_observation": round(self.mean_entry_ready, 3),
            "mean_validated_positive_per_observation": round(
                self.mean_validated_positive, 3
            ),
            "best_strategy_expected_net_bps": (
                round(value, 3)
                if (value := self.best_strategy_expected_net_bps) is not None
                else None
            ),
            "no_trade_rate": round(self.no_trade_rate, 4),
            "coverage_gap": self.is_coverage_gap,
            "contributing_strategies": dict(self.contributing_strategies.most_common(8)),
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }


class StrategyCoverageAnalyzer:
    """Accumulates the coverage matrix from selection results.

    Kept off the trading hot path by design: ``record`` is O(1) and touches only an
    in-memory dict, and persistence is an explicit ``flush`` the caller schedules. The
    expensive part of coverage analysis is the reporting, which reads the snapshot.
    """

    def __init__(
        self,
        *,
        state_path: str | Path | None = "data/store/strategy-coverage.json",
        max_buckets: int = 4096,
    ) -> None:
        self._state_path = Path(state_path) if state_path else None
        self._max_buckets = max(1, int(max_buckets))
        self._lock = threading.RLock()
        self._buckets: dict[str, CoverageObservation] = {}
        self._load()

    # -- recording ---------------------------------------------------------- #
    def record(
        self,
        context: MarketContext,
        *,
        eligible_count: int,
        entry_ready_count: int,
        validated_positive_count: int,
        no_trade: bool,
        best_net_bps: float | None = None,
        entry_ready_strategies: Iterable[str] = (),
        observed_at: datetime | None = None,
    ) -> CoverageObservation:
        bucket = bucket_for_context(context)
        key = bucket.as_key()
        moment = observed_at or context.captured_at
        with self._lock:
            observation = self._buckets.get(key)
            if observation is None:
                if len(self._buckets) >= self._max_buckets:
                    # Drop the least-observed bucket rather than refusing to record. The
                    # matrix is 5*4*3*5*4*4 = 4,800 cells and most are never visited, so
                    # the cap only ever bites on pathological input.
                    victim = min(self._buckets.values(), key=lambda item: item.observations)
                    self._buckets.pop(victim.bucket.as_key(), None)
                observation = CoverageObservation(bucket=bucket, first_seen_at=moment)
                self._buckets[key] = observation
            observation.observations += 1
            observation.eligible_strategy_count += max(0, int(eligible_count))
            observation.entry_ready_strategy_count += max(0, int(entry_ready_count))
            observation.validated_positive_strategy_count += max(
                0, int(validated_positive_count)
            )
            if no_trade:
                observation.no_trade_count += 1
            if best_net_bps is not None and math.isfinite(float(best_net_bps)):
                observation.best_net_bps_total += float(best_net_bps)
                observation.best_net_bps_samples += 1
            for strategy_id in entry_ready_strategies:
                name = str(strategy_id or "").strip().lower()
                if name:
                    observation.contributing_strategies[name] += 1
            observation.last_seen_at = moment
            return observation

    def record_selection(
        self, context: MarketContext, selection: Any
    ) -> CoverageObservation:
        """Convenience over ``record`` for a :class:`StrategySelectionResult`."""
        candidates = tuple(getattr(selection, "ranked_candidates", ()) or ())
        entry_ready = tuple(item for item in candidates if getattr(item, "entry_ready", False))
        selectable = tuple(item for item in candidates if getattr(item, "selectable", False))
        best_net = None
        if selectable:
            best_net = max(
                float(getattr(item, "expected_net_return_bps", 0.0)) for item in selectable
            )
        return self.record(
            context,
            eligible_count=sum(1 for item in candidates if getattr(item, "eligible", False)),
            entry_ready_count=len(entry_ready),
            validated_positive_count=len(selectable),
            no_trade=bool(getattr(selection, "is_no_trade", False)),
            best_net_bps=best_net,
            entry_ready_strategies=[
                str(getattr(item, "strategy_id", "")) for item in entry_ready
            ],
        )

    # -- reporting ---------------------------------------------------------- #
    def observations(self) -> tuple[CoverageObservation, ...]:
        with self._lock:
            return tuple(self._buckets.values())

    def gaps(self, *, minimum_observations: int = 5) -> tuple[CoverageObservation, ...]:
        """Buckets seen often enough to matter with no selectable strategy.

        ``minimum_observations`` exists because a single visit to a bucket is not evidence
        of a gap — it may simply be a bucket the market touched once.
        """
        return tuple(
            sorted(
                (
                    item
                    for item in self.observations()
                    if item.is_coverage_gap and item.observations >= max(1, int(minimum_observations))
                ),
                key=lambda item: -item.observations,
            )
        )

    def research_candidates(
        self, *, minimum_observations: int = 20, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Recurring gaps, ranked — the input to the strategy-addition process.

        A candidate is a *bucket*, not a strategy: the output of this method is "the market
        spends 8% of its time in a state nothing validated covers", and what to build for
        it is a research question, not something this module should answer.
        """
        return [
            {
                **item.bucket.as_dict(),
                "observations": item.observations,
                "no_trade_rate": round(item.no_trade_rate, 4),
                "eligible_per_observation": round(item.mean_eligible, 3),
                "entry_ready_per_observation": round(item.mean_entry_ready, 3),
                "nearest_firing_strategies": dict(
                    item.contributing_strategies.most_common(5)
                ),
                "reason": COVERAGE_GAP_REASON,
            }
            for item in self.gaps(minimum_observations=minimum_observations)[: max(0, int(limit))]
        ]

    def matrix(self) -> list[dict[str, Any]]:
        return [
            item.as_dict()
            for item in sorted(self.observations(), key=lambda row: -row.observations)
        ]

    def summary(self) -> dict[str, Any]:
        observations = self.observations()
        total = sum(item.observations for item in observations)
        gaps = tuple(item for item in observations if item.is_coverage_gap)
        gap_observations = sum(item.observations for item in gaps)
        return {
            "buckets_seen": len(observations),
            "buckets_possible": _possible_bucket_count(),
            "observations": total,
            "coverage_gap_buckets": len(gaps),
            "coverage_gap_observation_share": (
                round(gap_observations / total, 4) if total else 0.0
            ),
            "no_trade_rate": (
                round(sum(item.no_trade_count for item in observations) / total, 4)
                if total
                else 0.0
            ),
        }

    # -- persistence -------------------------------------------------------- #
    def flush(self) -> bool:
        """Write the tally. Called on a schedule, never from the decision path."""
        if self._state_path is None:
            return False
        with self._lock:
            payload = {
                "version": 1,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "buckets": [item.as_dict() for item in self._buckets.values()],
            }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self._state_path)
            return True
        except OSError:
            return False

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in payload.get("buckets") or ():
            if not isinstance(row, Mapping):
                continue
            try:
                bucket = ContextBucket.from_key(str(row.get("bucket") or ""))
            except ValueError:
                continue
            observations = int(row.get("observations") or 0)
            if observations <= 0:
                continue
            self._buckets[bucket.as_key()] = CoverageObservation(
                bucket=bucket,
                observations=observations,
                # Stored means are re-expanded into totals so a resumed tally keeps
                # weighting old observations correctly instead of restarting at zero.
                eligible_strategy_count=int(
                    round(float(row.get("mean_eligible_per_observation") or 0.0) * observations)
                ),
                entry_ready_strategy_count=int(
                    round(
                        float(row.get("mean_entry_ready_per_observation") or 0.0)
                        * observations
                    )
                ),
                validated_positive_strategy_count=int(
                    round(
                        float(row.get("mean_validated_positive_per_observation") or 0.0)
                        * observations
                    )
                ),
                no_trade_count=int(round(float(row.get("no_trade_rate") or 0.0) * observations)),
                contributing_strategies=Counter(
                    dict(row.get("contributing_strategies") or {})
                ),
                first_seen_at=_parse(row.get("first_seen_at")),
                last_seen_at=_parse(row.get("last_seen_at")),
            )


def _possible_bucket_count() -> int:
    total = 1
    for values in CONTEXT_DIMENSIONS.values():
        total *= len(values)
    return total


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
