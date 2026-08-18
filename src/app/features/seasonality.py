"""Rolling seasonality: what is unusual *for this kind of time*.

The problem this replaces
-------------------------
A fixed calendar rule ("Monday opens are weak", "avoid Friday afternoons") is a constant
fitted to a window that has already ended. It cannot say how strong the effect is now,
whether it still exists, or how much data stands behind it, and it goes on firing at full
strength after the regime that produced it is gone.

What replaces it
----------------
Each raw feature is normalised against a rolling baseline conditioned on the *kind of
time* it was observed in::

    z = (x - mu[d, s, r]) / (sigma[d, s, r] + eps)

where ``d`` is the day of week, ``s`` the session phase and ``r`` the regime label. A
Monday-open reading is compared with other Monday opens in the same regime, so "weak" is
measured rather than assumed, and the answer decays as the market changes.

Small samples
-------------
A (day, phase, regime) bucket is thin by construction — five weekdays times nine phases
times a dozen regimes. Thin buckets are shrunk toward the metric's global baseline::

    estimate = (n / (n + k)) * bucket + (k / (n + k)) * global      k = 30

so a bucket with two observations reports almost the global baseline and one with three
hundred reports almost its own. Nothing is suppressed for being thin: a shrunk estimate
carrying its sample count and confidence is more useful downstream than a missing one,
and ``confidence = n / (n + k)`` is exactly what the gate and the position sizer read.

Leakage
-------
Two properties, both enforced rather than documented:

* :meth:`SeasonalityEngine.observe` normalises against the baseline **as it stands
  before** the observation is folded in, so a z-score never sees its own value.
* An observation older than the bucket's ``last_observed_at`` is **rejected**, not
  merged. A baseline that has absorbed a future observation cannot be un-poisoned, and a
  walk-forward run that silently accepted one would report a score it did not earn. The
  rejection is counted per bucket and surfaced in :meth:`SeasonalityEngine.report`.

The rolling window is exponential rather than a stored ring of observations: the
sufficient statistics decay by ``(W-1)/W`` per observation, so the effective sample size
saturates at ``baseline_window`` and old regimes fade out without a re-read of history.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from app.config.temporal_config import SeasonalityConfig, default_temporal_config
from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
)

__all__ = [
    "GLOBAL_BUCKET",
    "SeasonalityBaseline",
    "SeasonalityEngine",
    "SeasonalityKey",
    "SeasonalityScore",
]

#: Sentinel used for the metric-wide baseline that thin buckets are shrunk toward. It is
#: stored as an ordinary row so the global estimate is reloadable and auditable rather
#: than recomputed from whatever happens to be in memory.
GLOBAL_BUCKET = "__GLOBAL__"


@dataclass(frozen=True)
class SeasonalityKey:
    """Identity of one seasonality bucket."""

    metric: str
    market_group: str
    day_of_week: str
    session_phase: str
    regime: str

    @property
    def global_key(self) -> "SeasonalityKey":
        return SeasonalityKey(
            metric=self.metric,
            market_group=self.market_group,
            day_of_week=GLOBAL_BUCKET,
            session_phase=GLOBAL_BUCKET,
            regime=GLOBAL_BUCKET,
        )

    @property
    def is_global(self) -> bool:
        return self.day_of_week == GLOBAL_BUCKET

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            self.metric,
            self.market_group,
            self.day_of_week,
            self.session_phase,
            self.regime,
        )


@dataclass(frozen=True)
class SeasonalityBaseline:
    """Decayed sufficient statistics for one bucket."""

    key: SeasonalityKey
    weight: float
    mean: float
    m2: float
    sample_count: int
    baseline_window: int
    baseline_updated_at: datetime | None
    last_observed_at: datetime | None
    out_of_order_rejected: int = 0

    @property
    def variance(self) -> float:
        if self.weight <= 0.0:
            return 0.0
        return max(0.0, self.m2 / self.weight)

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)

    def confidence(self, shrinkage_k: float) -> float:
        if self.weight <= 0.0:
            return 0.0
        return float(self.weight / (self.weight + max(0.0, shrinkage_k)))

    def is_stale(self, *, now: datetime, staleness_days: int) -> bool:
        if self.last_observed_at is None:
            return True
        return now - self.last_observed_at > timedelta(days=max(1, int(staleness_days)))

    @classmethod
    def empty(cls, key: SeasonalityKey, *, baseline_window: int) -> "SeasonalityBaseline":
        return cls(
            key=key,
            weight=0.0,
            mean=0.0,
            m2=0.0,
            sample_count=0,
            baseline_window=baseline_window,
            baseline_updated_at=None,
            last_observed_at=None,
        )

    def updated(self, value: float, *, observed_at: datetime) -> "SeasonalityBaseline":
        """Fold in one observation with exponential decay.

        West's weighted incremental algorithm, with the decay applied to the prior
        statistics so the effective sample size saturates at ``baseline_window``.
        """
        decay = 1.0 - 1.0 / float(max(2, self.baseline_window))
        prior_weight = self.weight * decay
        prior_m2 = self.m2 * decay
        weight = prior_weight + 1.0
        delta = value - self.mean
        mean = self.mean + delta / weight
        m2 = prior_m2 + delta * (value - mean)
        return SeasonalityBaseline(
            key=self.key,
            weight=weight,
            mean=mean,
            m2=max(0.0, m2),
            sample_count=self.sample_count + 1,
            baseline_window=self.baseline_window,
            baseline_updated_at=observed_at,
            last_observed_at=observed_at,
            out_of_order_rejected=self.out_of_order_rejected,
        )

    def with_rejection(self) -> "SeasonalityBaseline":
        return SeasonalityBaseline(
            key=self.key,
            weight=self.weight,
            mean=self.mean,
            m2=self.m2,
            sample_count=self.sample_count,
            baseline_window=self.baseline_window,
            baseline_updated_at=self.baseline_updated_at,
            last_observed_at=self.last_observed_at,
            out_of_order_rejected=self.out_of_order_rejected + 1,
        )


@dataclass(frozen=True)
class SeasonalityScore:
    """One normalised reading, with everything needed to judge how much it is worth."""

    key: SeasonalityKey
    value: float
    z_score: float
    mean: float
    stdev: float
    #: Effective (decayed) sample size of the conditioned bucket.
    sample_count: int
    effective_sample_size: float
    confidence: float
    shrinkage_weight: float
    baseline_window: int
    baseline_updated_at: datetime | None
    stale: bool
    #: True when the bucket had no prior observations and the score rests entirely on the
    #: global baseline (or, if that is empty too, on nothing — z is then 0.0).
    cold_start: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.key.metric,
            "market_group": self.key.market_group,
            "day_of_week": self.key.day_of_week,
            "session_phase": self.key.session_phase,
            "regime": self.key.regime,
            "value": self.value,
            "z_score": self.z_score,
            "mean": self.mean,
            "stdev": self.stdev,
            "sample_count": self.sample_count,
            "effective_sample_size": self.effective_sample_size,
            "confidence": self.confidence,
            "shrinkage_weight": self.shrinkage_weight,
            "baseline_window": self.baseline_window,
            "baseline_updated_at": iso_column(self.baseline_updated_at),
            "stale": self.stale,
            "cold_start": self.cold_start,
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


class SeasonalityEngine:
    """Rolling, regime-conditioned normalisation for calendar-linked features.

    Baselines are held in memory and written through to ``seasonality_baseline`` so a
    restart resumes with the statistics it had rather than a cold start. Reads never
    touch the database once a bucket is loaded, which is what keeps this usable on the
    realtime path.
    """

    def __init__(
        self,
        *,
        store: TradingStateStore | None = None,
        config: SeasonalityConfig | None = None,
        persist: bool = True,
    ) -> None:
        self._config = config or default_temporal_config().seasonality
        self._persist = persist
        self._store = store if store is not None else (
            default_trading_state_store() if persist else None
        )
        self._lock = threading.RLock()
        self._baselines: dict[tuple[str, str, str, str, str], SeasonalityBaseline] = {}
        self._loaded = False

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> SeasonalityConfig:
        return self._config

    def score(
        self,
        key: SeasonalityKey,
        value: float,
        *,
        now: datetime | None = None,
    ) -> SeasonalityScore:
        """Normalise ``value`` against the current baseline WITHOUT updating it."""
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            self._ensure_loaded()
            return self._score_locked(key, float(value), moment)

    def observe(
        self,
        key: SeasonalityKey,
        value: float,
        *,
        observed_at: datetime,
        update: bool = True,
    ) -> SeasonalityScore:
        """Score ``value`` against the pre-observation baseline, then fold it in.

        The returned score is computed before the update, so it can never contain
        information from the observation it describes. An observation that predates the
        bucket's ``last_observed_at`` is scored but NOT folded in, and the rejection is
        recorded.
        """
        moment = _aware(observed_at)
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"seasonality observation for {key.metric} is not finite")
        with self._lock:
            self._ensure_loaded()
            score = self._score_locked(key, numeric, moment)
            if not update:
                return score
            self._update_locked(key, numeric, moment)
            return score

    def observe_many(
        self,
        observations: Iterable[tuple[SeasonalityKey, float]],
        *,
        observed_at: datetime,
    ) -> dict[str, SeasonalityScore]:
        """Score and fold in several metrics sharing one timestamp."""
        return {
            key.metric: self.observe(key, value, observed_at=observed_at)
            for key, value in observations
        }

    def baseline(self, key: SeasonalityKey) -> SeasonalityBaseline:
        with self._lock:
            self._ensure_loaded()
            return self._baseline_locked(key)

    def report(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Operator view: coverage, thin buckets, staleness and rejected replays."""
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            self._ensure_loaded()
            baselines = list(self._baselines.values())
        conditioned = [item for item in baselines if not item.key.is_global]
        thin = [
            item
            for item in conditioned
            if item.sample_count < self._config.minimum_samples
        ]
        stale = [
            item
            for item in conditioned
            if item.is_stale(now=moment, staleness_days=self._config.staleness_days)
        ]
        return {
            "as_of": iso_column(moment),
            "bucket_count": len(conditioned),
            "global_bucket_count": len(baselines) - len(conditioned),
            "metrics": sorted({item.key.metric for item in baselines}),
            "thin_bucket_count": len(thin),
            "stale_bucket_count": len(stale),
            "out_of_order_rejected": sum(item.out_of_order_rejected for item in baselines),
            "total_observations": sum(item.sample_count for item in conditioned),
            "shrinkage_k": self._config.shrinkage_k,
            "baseline_window": self._config.baseline_window,
            "minimum_samples": self._config.minimum_samples,
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _score_locked(
        self, key: SeasonalityKey, value: float, now: datetime
    ) -> SeasonalityScore:
        bucket = self._baseline_locked(key)
        global_baseline = self._baseline_locked(key.global_key)
        shrink_k = self._config.shrinkage_k
        n = bucket.weight
        weight = n / (n + shrink_k) if (n + shrink_k) > 0 else 0.0

        mean = weight * bucket.mean + (1.0 - weight) * global_baseline.mean
        # The standard deviations are blended, not the variances: blending variances
        # would let a single wide global bucket dominate the scale of every thin bucket,
        # which shrinks z-scores toward zero exactly where the evidence is weakest.
        stdev = weight * bucket.stdev + (1.0 - weight) * global_baseline.stdev
        cold_start = bucket.weight <= 0.0

        if stdev <= 0.0 and global_baseline.weight <= 0.0:
            z = 0.0
        else:
            z = (value - mean) / (stdev + self._config.epsilon)
        return SeasonalityScore(
            key=key,
            value=value,
            z_score=z,
            mean=mean,
            stdev=stdev,
            sample_count=bucket.sample_count,
            effective_sample_size=n,
            confidence=bucket.confidence(shrink_k),
            shrinkage_weight=weight,
            baseline_window=bucket.baseline_window,
            baseline_updated_at=bucket.baseline_updated_at,
            stale=bucket.is_stale(now=now, staleness_days=self._config.staleness_days),
            cold_start=cold_start,
        )

    def _update_locked(self, key: SeasonalityKey, value: float, moment: datetime) -> None:
        dirty: list[SeasonalityBaseline] = []
        for target in (key, key.global_key):
            baseline = self._baseline_locked(target)
            if baseline.last_observed_at is not None and moment < baseline.last_observed_at:
                updated = baseline.with_rejection()
            else:
                updated = baseline.updated(value, observed_at=moment)
            self._baselines[target.as_tuple()] = updated
            dirty.append(updated)
        self._flush(dirty)

    def _baseline_locked(self, key: SeasonalityKey) -> SeasonalityBaseline:
        existing = self._baselines.get(key.as_tuple())
        if existing is not None:
            return existing
        created = SeasonalityBaseline.empty(
            key, baseline_window=self._config.baseline_window
        )
        self._baselines[key.as_tuple()] = created
        return created

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._store is None:
            return
        rows = self._store.fetch_all(
            "select metric, market_group, day_of_week, session_phase, regime,"
            " weight, mean, m2, sample_count, baseline_window,"
            " baseline_updated_at, last_observed_at, out_of_order_rejected"
            " from seasonality_baseline"
        )
        for row in rows:
            key = SeasonalityKey(
                metric=str(row["metric"]),
                market_group=str(row["market_group"]),
                day_of_week=str(row["day_of_week"]),
                session_phase=str(row["session_phase"]),
                regime=str(row["regime"]),
            )
            self._baselines[key.as_tuple()] = SeasonalityBaseline(
                key=key,
                weight=float(row["weight"]),
                mean=float(row["mean"]),
                m2=float(row["m2"]),
                sample_count=int(row["sample_count"]),
                baseline_window=int(row["baseline_window"]),
                baseline_updated_at=_parse(row["baseline_updated_at"]),
                last_observed_at=_parse(row["last_observed_at"]),
                out_of_order_rejected=int(row["out_of_order_rejected"] or 0),
            )

    def _flush(self, baselines: Iterable[SeasonalityBaseline]) -> None:
        if self._store is None or not self._persist:
            return
        rows = list(baselines)
        if not rows:
            return
        with self._store.transaction() as conn:
            for baseline in rows:
                conn.execute(
                    "insert into seasonality_baseline"
                    " (metric, market_group, day_of_week, session_phase, regime,"
                    "  weight, mean, m2, sample_count, confidence, baseline_window,"
                    "  baseline_updated_at, last_observed_at, out_of_order_rejected)"
                    " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    " on conflict(metric, market_group, day_of_week, session_phase, regime)"
                    " do update set weight = excluded.weight, mean = excluded.mean,"
                    "  m2 = excluded.m2, sample_count = excluded.sample_count,"
                    "  confidence = excluded.confidence,"
                    "  baseline_window = excluded.baseline_window,"
                    "  baseline_updated_at = excluded.baseline_updated_at,"
                    "  last_observed_at = excluded.last_observed_at,"
                    "  out_of_order_rejected = excluded.out_of_order_rejected",
                    (
                        *baseline.key.as_tuple(),
                        baseline.weight,
                        baseline.mean,
                        baseline.m2,
                        baseline.sample_count,
                        baseline.confidence(self._config.shrinkage_k),
                        baseline.baseline_window,
                        iso_column(baseline.baseline_updated_at),
                        iso_column(baseline.last_observed_at),
                        baseline.out_of_order_rejected,
                    ),
                )


def key_for(
    metric: str,
    *,
    market_group: str,
    day_of_week: str,
    session_phase: str,
    regime: str,
) -> SeasonalityKey:
    """Bucket key with the field normalisation every caller must share."""
    return SeasonalityKey(
        metric=str(metric).strip().lower(),
        market_group=str(market_group).strip().upper(),
        day_of_week=str(day_of_week).strip().upper(),
        session_phase=str(session_phase).strip().upper(),
        regime=str(regime).strip().upper(),
    )


def keys_from_context(
    metrics: Mapping[str, float],
    *,
    market_group: str,
    day_of_week: str,
    session_phase: str,
    regime: str,
) -> list[tuple[SeasonalityKey, float]]:
    """Pair every metric with its bucket key for one observation instant."""
    return [
        (
            key_for(
                name,
                market_group=market_group,
                day_of_week=day_of_week,
                session_phase=session_phase,
                regime=regime,
            ),
            float(value),
        )
        for name, value in metrics.items()
        if value is not None and math.isfinite(float(value))
    ]
