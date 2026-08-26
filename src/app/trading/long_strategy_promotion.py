"""Automatic, evidence-gated deployment ladder for long strategy arms.

New long theses start in SHADOW.  This module is the only bridge from that
configuration state to a real order: it reads durable *after-cost* forward
outcomes and grants at most LIVE_PROBE until real broker outcomes independently
earn the higher rungs.  The decision is recomputed every cycle, so deteriorating
evidence automatically demotes the arm without an operator toggle.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

from app.trading.directional import StrategyDeploymentState
from app.trading.strategy_performance_store import (
    EVALUATION_SOURCE_LIVE,
    EVALUATION_SOURCE_LIVE_PROBE,
    StrategyOutcome,
    StrategyPerformanceStore,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_at_least(name: str, floor: int) -> int:
    try:
        return max(floor, int(os.getenv(name, str(floor))))
    except (TypeError, ValueError):
        return floor


def _env_float_at_least(name: str, floor: float) -> float:
    try:
        return max(floor, float(os.getenv(name, str(floor))))
    except (TypeError, ValueError):
        return floor


@dataclass(frozen=True)
class LongPromotionConfig:
    enabled: bool = True
    minimum_shadow_samples: int = 30
    minimum_shadow_positive_samples: int = 10
    minimum_shadow_days: int = 3
    minimum_shadow_conservative_edge_bps: float = 0.0
    minimum_positive_day_fraction: float = 0.60
    cost_stress_multiple: float = 1.25
    maximum_shadow_loss_streak: int = 2
    minimum_live_probe_samples: int = 30
    minimum_live_probe_positive_samples: int = 10
    minimum_live_probe_days: int = 10
    minimum_live_probe_lcb_bps: float = 0.0
    minimum_live_full_samples: int = 60
    minimum_live_full_positive_samples: int = 20
    minimum_live_full_days: int = 20
    minimum_live_full_lcb_bps: float = 5.0
    probe_size_fraction: float = 0.10
    limited_size_fraction: float = 0.35

    @classmethod
    def from_env(cls) -> "LongPromotionConfig":
        # Environment variables may make production stricter, never weaker than
        # the safety floor documented above.
        return cls(
            enabled=_env_bool("LONG_STRATEGY_AUTO_PROMOTION_ENABLED", True),
            minimum_shadow_samples=_env_int_at_least("LONG_PROMOTION_MIN_SHADOW_SAMPLES", 30),
            minimum_shadow_positive_samples=_env_int_at_least("LONG_PROMOTION_MIN_SHADOW_POSITIVE", 10),
            minimum_shadow_days=_env_int_at_least("LONG_PROMOTION_MIN_SHADOW_DAYS", 3),
            minimum_shadow_conservative_edge_bps=_env_float_at_least("LONG_PROMOTION_MIN_SHADOW_EDGE_BPS", 0.0),
            minimum_positive_day_fraction=max(
                0.60,
                min(1.0, float(os.getenv("LONG_PROMOTION_MIN_POSITIVE_DAY_FRACTION", "0.60"))),
            ),
            cost_stress_multiple=_env_float_at_least(
                "LONG_PROMOTION_COST_STRESS_MULTIPLE", 1.25
            ),
            maximum_shadow_loss_streak=max(0, min(2, int(os.getenv("LONG_PROMOTION_MAX_LOSS_STREAK", "2")))),
            minimum_live_probe_samples=_env_int_at_least("LONG_PROMOTION_MIN_LIVE_SAMPLES", 30),
            minimum_live_probe_positive_samples=_env_int_at_least("LONG_PROMOTION_MIN_LIVE_POSITIVE", 10),
            minimum_live_probe_days=_env_int_at_least("LONG_PROMOTION_MIN_LIVE_DAYS", 10),
            minimum_live_probe_lcb_bps=_env_float_at_least("LONG_PROMOTION_MIN_LIVE_LCB_BPS", 0.0),
            minimum_live_full_samples=_env_int_at_least("LONG_PROMOTION_MIN_FULL_SAMPLES", 60),
            minimum_live_full_positive_samples=_env_int_at_least("LONG_PROMOTION_MIN_FULL_POSITIVE", 20),
            minimum_live_full_days=_env_int_at_least("LONG_PROMOTION_MIN_FULL_DAYS", 20),
            minimum_live_full_lcb_bps=_env_float_at_least("LONG_PROMOTION_MIN_FULL_LCB_BPS", 5.0),
            probe_size_fraction=max(0.01, min(0.10, float(os.getenv("LONG_PROMOTION_PROBE_SIZE_FRACTION", "0.10")))),
            limited_size_fraction=max(0.01, min(0.35, float(os.getenv("LONG_PROMOTION_LIMITED_SIZE_FRACTION", "0.35")))),
        )


@dataclass(frozen=True)
class LongPromotionDecision:
    state: StrategyDeploymentState
    reason_codes: tuple[str, ...]
    sample_count: int
    positive_sample_count: int
    distinct_days: int
    conservative_edge_bps: float
    live_sample_count: int
    live_lower_confidence_bound_bps: float
    positive_day_fraction: float = 0.0
    cost_stressed_mean_net_bps: float = float("-inf")


def _days(outcomes: Iterable[StrategyOutcome]) -> int:
    return len({item.recorded_at.date() for item in outcomes})


def _lower_confidence_bound(outcomes: tuple[StrategyOutcome, ...]) -> float:
    values = [float(item.realized_net_bps) for item in outcomes]
    if not values:
        return float("-inf")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return float("-inf")
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean - 1.64 * math.sqrt(max(0.0, variance)) / math.sqrt(len(values))


def _positive_day_fraction(outcomes: tuple[StrategyOutcome, ...]) -> float:
    by_day: dict[object, list[float]] = {}
    for item in outcomes:
        by_day.setdefault(item.recorded_at.date(), []).append(float(item.realized_net_bps))
    if not by_day:
        return 0.0
    return sum(sum(values) / len(values) > 0.0 for values in by_day.values()) / len(by_day)


def _cost_stressed_mean(
    outcomes: tuple[StrategyOutcome, ...], multiple: float
) -> float:
    stressed: list[float] = []
    for item in outcomes:
        if item.realized_gross_bps is None:
            return float("-inf")
        cost = max(0.0, float(item.realized_gross_bps) - float(item.realized_net_bps))
        stressed.append(float(item.realized_gross_bps) - max(1.0, multiple) * cost)
    return sum(stressed) / len(stressed) if stressed else float("-inf")


def evaluate_long_promotion(
    strategy_id: str,
    market: str,
    store: StrategyPerformanceStore,
    *,
    regime: str | None = None,
    config: LongPromotionConfig | None = None,
) -> LongPromotionDecision:
    cfg = config or LongPromotionConfig.from_env()
    if not cfg.enabled:
        return LongPromotionDecision(
            StrategyDeploymentState.SHADOW, ("LONG_AUTO_PROMOTION_DISABLED",),
            0, 0, 0, float("-inf"), 0, float("-inf"),
        )

    outcomes = store.recent_outcomes(
        strategy_id,
        market=market,
        regime=regime,
        direction="LONG",
        execution_product="CASH",
        limit=500,
    )
    posterior = store.posterior(
        strategy_id,
        market=market,
        regime=regime,
        direction="LONG",
        execution_product="CASH",
        # Promotion evidence may never widen from the requested regime.  A
        # TREND_UP history cannot authorize a TREND_DOWN order.
        allow_regime_fallback=False,
    )
    positives = sum(item.realized_net_bps > 0.0 for item in outcomes)
    days = _days(outcomes)
    positive_day_fraction = _positive_day_fraction(outcomes)
    cost_stressed_mean = _cost_stressed_mean(outcomes, cfg.cost_stress_multiple)
    reasons: list[str] = []
    if len(outcomes) < cfg.minimum_shadow_samples:
        reasons.append("LONG_PROMOTION_SAMPLE_INSUFFICIENT")
    if positives < cfg.minimum_shadow_positive_samples:
        reasons.append("LONG_PROMOTION_POSITIVE_SAMPLE_INSUFFICIENT")
    if days < cfg.minimum_shadow_days:
        reasons.append("LONG_PROMOTION_STABILITY_DAYS_INSUFFICIENT")
    if posterior.conservative_edge_bps <= cfg.minimum_shadow_conservative_edge_bps:
        reasons.append("LONG_PROMOTION_CONSERVATIVE_EDGE_NON_POSITIVE")
    if positive_day_fraction < cfg.minimum_positive_day_fraction:
        reasons.append("LONG_PROMOTION_OUT_OF_SAMPLE_STABILITY_INSUFFICIENT")
    if cost_stressed_mean <= 0.0:
        reasons.append("LONG_PROMOTION_COST_STRESS_FAILED")
    if posterior.loss_streak > cfg.maximum_shadow_loss_streak:
        reasons.append("LONG_PROMOTION_LOSS_STREAK")

    live = tuple(
        item for item in outcomes
        if item.evaluation_source in {EVALUATION_SOURCE_LIVE_PROBE, EVALUATION_SOURCE_LIVE}
    )
    live_positive = sum(item.realized_net_bps > 0.0 for item in live)
    live_days = _days(live)
    live_lcb = _lower_confidence_bound(live)
    state = StrategyDeploymentState.SHADOW
    if not reasons:
        state = StrategyDeploymentState.LIVE_PROBE
        if (
            len(live) >= cfg.minimum_live_probe_samples
            and live_positive >= cfg.minimum_live_probe_positive_samples
            and live_days >= cfg.minimum_live_probe_days
            and live_lcb > cfg.minimum_live_probe_lcb_bps
        ):
            state = StrategyDeploymentState.LIVE_LIMITED
        if (
            len(live) >= cfg.minimum_live_full_samples
            and live_positive >= cfg.minimum_live_full_positive_samples
            and live_days >= cfg.minimum_live_full_days
            and live_lcb > cfg.minimum_live_full_lcb_bps
        ):
            state = StrategyDeploymentState.LIVE_FULL

    return LongPromotionDecision(
        state=state,
        reason_codes=tuple(reasons) or (f"LONG_AUTO_PROMOTED_{state}",),
        sample_count=len(outcomes),
        positive_sample_count=positives,
        distinct_days=days,
        conservative_edge_bps=float(posterior.conservative_edge_bps),
        live_sample_count=len(live),
        live_lower_confidence_bound_bps=live_lcb,
        positive_day_fraction=positive_day_fraction,
        cost_stressed_mean_net_bps=cost_stressed_mean,
    )


def deployment_size_cap(state: StrategyDeploymentState, config: LongPromotionConfig | None = None) -> float:
    cfg = config or LongPromotionConfig.from_env()
    if state is StrategyDeploymentState.LIVE_PROBE:
        return cfg.probe_size_fraction
    if state is StrategyDeploymentState.LIVE_LIMITED:
        return cfg.limited_size_fraction
    if state is StrategyDeploymentState.LIVE_FULL:
        return 1.0
    return 0.0
