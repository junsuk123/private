"""The bandit as a BOUNDED correction, not as the selector.

Role change
-----------
``ConservativeStrategyBandit`` currently *is* the final selector: ``_bandit_choice``
computes a pessimistic lower bound per arm, decides NO_TRADE, and returns the winner. In
V2 that responsibility moves to ``StrategySelectorV2``, and the bandit answers one
narrower question — *has this strategy recently become better or worse than the utility
model thinks?* — as a term ``B_s`` in the utility.

The store, the shrinkage and the pessimism are reused as-is
(``StrategyPerformanceStore.posterior``, which already blends a symbol mean toward the
strategy mean and discounts the effective sample count by the change-point probability).
Only the *use* changes.

Why the correction is bounded
-----------------------------
An unbounded posterior term would let realized history override the model entirely, which
reproduces the failure this refactor is unwinding from the other side: a single arm with a
handful of bad fills would dominate every context signal. The bound is a config value
(``config/bandit_adapter.yaml``, default ±20bps) and the *clamping is reported*, so a
correction that wanted to be larger than the bound is visible rather than silently
truncated.

Context key
-----------
``(strategy_id, market, regime_cluster, volatility_bucket)`` — deliberately WITHOUT
symbol. Per-symbol posteriors are far too thin: only one strategy in the store has three
symbols with eight fills each. Symbol still *conditions* the estimate through the store's
own shrinkage, which is the measured compromise already in place.

Non-stationarity
----------------
Two mechanisms, both already meaningful in the store's data model:

* ``change_point_probability`` is passed through and discounts the effective sample
  count — a regime break does not make history wrong, it makes it less relevant;
* ``half_life_seconds`` applies exponential recency weighting when the caller supplies
  outcome timestamps, so a strategy that deteriorated last week does not keep its old
  posterior at full strength.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

__all__ = [
    "BANDIT_CORRECTION_CLAMPED",
    "BANDIT_NO_HISTORY",
    "BANDIT_POSTERIOR_UNAVAILABLE",
    "BanditAdapterConfig",
    "BanditContextKey",
    "BanditCorrection",
    "StrategyBanditAdapter",
]

BANDIT_CORRECTION_CLAMPED = "BANDIT_CORRECTION_CLAMPED"
BANDIT_NO_HISTORY = "BANDIT_NO_HISTORY"
BANDIT_POSTERIOR_UNAVAILABLE = "BANDIT_POSTERIOR_UNAVAILABLE"
BANDIT_SHRUNK_TO_PARENT = "BANDIT_SHRUNK_TO_PARENT"
BANDIT_RECENCY_WEIGHTED = "BANDIT_RECENCY_WEIGHTED"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class BanditAdapterConfig:
    """All knobs are config, none are hardcoded in the scoring path."""

    #: Symmetric cap on ``B_s``, in bps. ±20 is a starting value to be recalibrated from
    #: data, not a measured constant — see the module docstring.
    max_correction_bps: float = 20.0
    #: Samples below which the context posterior is shrunk toward its parent (the
    #: market-wide posterior for the same strategy).
    minimum_samples: int = 8
    #: Recency half-life. ``None`` disables the decay entirely.
    half_life_seconds: float | None = 7 * 24 * 3600.0
    #: Outcomes older than this are ignored regardless of decay, so a posterior cannot be
    #: carried by a market that no longer exists.
    max_age_seconds: float | None = 60 * 24 * 3600.0
    #: Volatility percentile cut points for the bucket label.
    volatility_buckets: tuple[float, ...] = (0.33, 0.66)

    @classmethod
    def from_env(cls) -> "BanditAdapterConfig":
        half_life = _env_float("BANDIT_ADAPTER_HALF_LIFE_SECONDS", 7 * 24 * 3600.0)
        max_age = _env_float("BANDIT_ADAPTER_MAX_AGE_SECONDS", 60 * 24 * 3600.0)
        return cls(
            max_correction_bps=max(
                0.0, _env_float("BANDIT_ADAPTER_MAX_CORRECTION_BPS", 20.0)
            ),
            minimum_samples=max(1, _env_int("BANDIT_ADAPTER_MINIMUM_SAMPLES", 8)),
            half_life_seconds=half_life if half_life > 0 else None,
            max_age_seconds=max_age if max_age > 0 else None,
        )

    def volatility_bucket(self, percentile: float | None) -> str:
        if percentile is None:
            return "UNKNOWN"
        low, high = self.volatility_buckets
        if percentile < low:
            return "LOW"
        if percentile < high:
            return "MID"
        return "HIGH"


@dataclass(frozen=True)
class BanditContextKey:
    """``(strategy, market, regime_cluster, volatility_bucket)``. No symbol — see docstring."""

    strategy_id: str
    market: str
    regime_cluster: str
    volatility_bucket: str
    direction: str = "LONG"

    @property
    def parent(self) -> "BanditContextKey":
        """Broader context the posterior is shrunk toward when samples are thin.

        Widens regime and volatility but never market or direction: KR costs ~28bps and
        US 51-70, and a short's history must never be borrowed from its long counterpart
        — that is exactly how an unvalidated arm would inherit an edge it never earned.
        """
        return BanditContextKey(
            strategy_id=self.strategy_id,
            market=self.market,
            regime_cluster="ALL",
            volatility_bucket="ALL",
            direction=self.direction,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "market": self.market,
            "regime_cluster": self.regime_cluster,
            "volatility_bucket": self.volatility_bucket,
            "direction": self.direction,
        }

    def __str__(self) -> str:
        return (
            f"{self.strategy_id}|{self.market}|{self.regime_cluster}"
            f"|{self.volatility_bucket}|{self.direction}"
        )


@dataclass(frozen=True)
class BanditCorrection:
    """``B_s`` plus everything needed to explain it."""

    strategy_id: str
    key: BanditContextKey
    correction_bps: float
    raw_correction_bps: float
    sample_count: int
    effective_sample_count: float
    posterior_mean_net_bps: float | None
    conservative_edge_bps: float | None
    loss_streak: int
    reason_codes: tuple[str, ...] = ()

    @property
    def clamped(self) -> bool:
        return abs(self.raw_correction_bps - self.correction_bps) > 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "context_key": str(self.key),
            "correction_bps": round(self.correction_bps, 3),
            "raw_correction_bps": round(self.raw_correction_bps, 3),
            "clamped": self.clamped,
            "sample_count": self.sample_count,
            "effective_sample_count": round(self.effective_sample_count, 3),
            "posterior_mean_net_bps": self.posterior_mean_net_bps,
            "conservative_edge_bps": self.conservative_edge_bps,
            "loss_streak": self.loss_streak,
            "reason_codes": list(self.reason_codes),
        }


class StrategyBanditAdapter:
    """Bounded online correction from realized outcomes.

    The correction is ``posterior_mean_net_bps - predicted_net_bps`` — how much the
    realized history disagrees with the model — shrunk toward zero when samples are thin
    and clamped to ``max_correction_bps``. It is a *correction to a forecast*, not a
    forecast: with no history it is exactly 0.0, which leaves the utility model's estimate
    untouched rather than penalising a cold arm twice (the uncertainty term already does
    that).
    """

    def __init__(
        self,
        *,
        store: Any | None = None,
        config: BanditAdapterConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or BanditAdapterConfig.from_env()

    @property
    def config(self) -> BanditAdapterConfig:
        return self._config

    def _resolve_store(self) -> Any | None:
        if self._store is None:
            try:
                from app.trading.strategy_performance_store import default_store

                self._store = default_store()
            except Exception:  # noqa: BLE001 - no store means no correction, not a crash.
                return None
        return self._store

    def context_key(
        self,
        strategy_id: str,
        *,
        market: str,
        regime: str | None,
        volatility_percentile: float | None,
        direction: str = "LONG",
    ) -> BanditContextKey:
        return BanditContextKey(
            strategy_id=str(strategy_id or "").strip().lower(),
            market=str(market or "UNKNOWN").strip().upper(),
            regime_cluster=str(regime or "UNKNOWN").strip().upper(),
            volatility_bucket=self._config.volatility_bucket(volatility_percentile),
            direction=str(direction or "LONG").strip().upper(),
        )

    def correct(
        self,
        *,
        strategy_id: str,
        market: str,
        regime: str | None,
        volatility_percentile: float | None,
        predicted_net_bps: float | None,
        change_point_probability: float = 0.0,
        direction: str = "LONG",
        symbol: str | None = None,
        now: datetime | None = None,
    ) -> BanditCorrection:
        key = self.context_key(
            strategy_id,
            market=market,
            regime=regime,
            volatility_percentile=volatility_percentile,
            direction=direction,
        )
        store = self._resolve_store()
        if store is None:
            return _zero_correction(key, (BANDIT_POSTERIOR_UNAVAILABLE,))

        posterior = self._posterior(
            store,
            key,
            change_point_probability=change_point_probability,
            symbol=symbol,
        )
        if posterior is None:
            return _zero_correction(key, (BANDIT_POSTERIOR_UNAVAILABLE,))

        sample_count = int(getattr(posterior, "sample_count", 0) or 0)
        reasons: list[str] = []
        if sample_count <= 0:
            return _zero_correction(key, (BANDIT_NO_HISTORY,))

        posterior_mean = float(getattr(posterior, "posterior_mean_net_bps", 0.0) or 0.0)
        conservative = _optional(getattr(posterior, "conservative_edge_bps", None))
        effective = float(
            getattr(posterior, "effective_sample_count", float(sample_count)) or 0.0
        )

        # Shrinkage toward "no correction" when the context is thin. Linear in the
        # effective sample count so an arm with two fills moves the score a little and one
        # with sixteen moves it fully — the same shape the store already uses to blend a
        # symbol mean toward a strategy mean.
        minimum = float(self._config.minimum_samples)
        confidence = min(1.0, effective / minimum) if minimum > 0 else 1.0
        if confidence < 1.0:
            reasons.append(BANDIT_SHRUNK_TO_PARENT)

        recency = self._recency_weight(store, key, now=now)
        if recency is not None and recency < 1.0:
            reasons.append(BANDIT_RECENCY_WEIGHTED)
            confidence *= recency

        baseline = predicted_net_bps if predicted_net_bps is not None else posterior_mean
        raw = (posterior_mean - float(baseline)) * confidence
        bound = self._config.max_correction_bps
        corrected = max(-bound, min(bound, raw))
        if abs(raw - corrected) > 1e-9:
            reasons.append(BANDIT_CORRECTION_CLAMPED)

        return BanditCorrection(
            strategy_id=key.strategy_id,
            key=key,
            correction_bps=corrected,
            raw_correction_bps=raw,
            sample_count=sample_count,
            effective_sample_count=effective,
            posterior_mean_net_bps=posterior_mean,
            conservative_edge_bps=conservative,
            loss_streak=int(getattr(posterior, "loss_streak", 0) or 0),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    def correct_all(
        self,
        predictions: Sequence[Any],
        *,
        market: str,
        regime: str | None,
        volatility_percentile: float | None,
        change_point_probability: float = 0.0,
        directions: Mapping[str, str] | None = None,
        symbol: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, BanditCorrection]:
        direction_by_id = dict(directions or {})
        corrections: dict[str, BanditCorrection] = {}
        for prediction in predictions:
            strategy_id = str(getattr(prediction, "strategy_id", "") or "")
            if not strategy_id:
                continue
            corrections[strategy_id] = self.correct(
                strategy_id=strategy_id,
                market=market,
                regime=regime,
                volatility_percentile=volatility_percentile,
                predicted_net_bps=_optional(
                    getattr(prediction, "expected_net_return_bps", None)
                ),
                change_point_probability=change_point_probability,
                direction=direction_by_id.get(strategy_id, "LONG"),
                symbol=symbol,
                now=now,
            )
        return corrections

    # -- internals ---------------------------------------------------------- #
    def _posterior(
        self,
        store: Any,
        key: BanditContextKey,
        *,
        change_point_probability: float,
        symbol: str | None,
    ) -> Any | None:
        """Context posterior, falling back to the parent context when empty.

        ``StrategyPerformanceStore.posterior`` already widens a regime miss to
        market-wide, so the explicit parent call only matters when the volatility bucket
        is what emptied the sample.
        """
        try:
            posterior = store.posterior(
                key.strategy_id,
                market=key.market,
                regime=None if key.regime_cluster in {"UNKNOWN", "ALL"} else key.regime_cluster,
                change_point_probability=change_point_probability,
                direction=key.direction,
                symbol=symbol,
            )
        except Exception:  # noqa: BLE001
            return None
        if int(getattr(posterior, "sample_count", 0) or 0) > 0:
            return posterior
        parent = key.parent
        try:
            return store.posterior(
                parent.strategy_id,
                market=parent.market,
                regime=None,
                change_point_probability=change_point_probability,
                direction=parent.direction,
                symbol=symbol,
            )
        except Exception:  # noqa: BLE001
            return posterior

    def _recency_weight(
        self, store: Any, key: BanditContextKey, *, now: datetime | None
    ) -> float | None:
        """Exponential recency weight in ``(0, 1]``, or ``None`` when unmeasurable.

        Computed from the outcome timestamps the store already keeps, so no new
        persistence is needed. When the store cannot answer, the weight is ``None`` and
        the correction is not decayed — guessing a decay would be worse than not decaying.
        """
        half_life = self._config.half_life_seconds
        if half_life is None or half_life <= 0:
            return None
        reader = getattr(store, "recent_outcomes", None)
        if not callable(reader):
            return None
        try:
            outcomes = reader(
                key.strategy_id,
                market=key.market,
                regime=None,
                direction=key.direction,
            )
        except Exception:  # noqa: BLE001
            return None
        if not outcomes:
            return None
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        weights: list[float] = []
        for outcome in outcomes:
            recorded = getattr(outcome, "recorded_at", None)
            if not isinstance(recorded, datetime):
                continue
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=timezone.utc)
            age = (moment - recorded).total_seconds()
            if age < 0:
                continue
            if self._config.max_age_seconds is not None and age > self._config.max_age_seconds:
                continue
            weights.append(0.5 ** (age / half_life))
        if not weights:
            return None
        # Mean weight across the surviving sample: a posterior built entirely from
        # month-old fills is worth roughly half of one built from fresh ones at a
        # week-long half life.
        return max(1e-6, min(1.0, sum(weights) / len(weights)))


def _zero_correction(
    key: BanditContextKey, reasons: tuple[str, ...]
) -> BanditCorrection:
    return BanditCorrection(
        strategy_id=key.strategy_id,
        key=key,
        correction_bps=0.0,
        raw_correction_bps=0.0,
        sample_count=0,
        effective_sample_count=0.0,
        posterior_mean_net_bps=None,
        conservative_edge_bps=None,
        loss_streak=0,
        reason_codes=reasons,
    )


def _optional(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
