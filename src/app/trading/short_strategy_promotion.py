"""Automatic promotion and demotion of directional (short) strategy arms.

What this module is
-------------------
The single authority on *how much real money* a given
``(strategy, direction, market, product)`` arm may touch. It reads forward
evidence and writes a :class:`~app.trading.directional.StrategyDeploymentState`.
It never places an order and never authorises one directly — the submission path
re-reads the committed state and applies its own final gates.

The two properties that matter
------------------------------
**Demotion is faster than promotion, by construction.** Promotion requires N
*consecutive* passing cycles; demotion acts on the first failing cycle, and
suspension acts immediately. This asymmetry is the whole safety argument: the cost
of a slow promotion is a missed opportunity, while the cost of a slow demotion is
a compounding loss on a position whose downside is unbounded.

**No single flag can shortcut the ladder.** ``SHADOW -> LIVE_FULL`` is not in the
transition whitelist, so it cannot happen through any combination of config,
environment variable or operator action — the most a manual override can do is
force one legal step, and it leaves an audit event. Model calibration
(``live_authorized`` on the checkpoint) is a *separate* precondition, not a
substitute: a well-calibrated model that predicts an unprofitable-after-borrow
strategy is a correct model of a bad trade.

Why hard gates precede the confidence score
-------------------------------------------
``confidence_score`` is a weighted blend, so it can be high while one component is
catastrophic — 0.85 overall is achievable with a borrow availability rate of 0.2 if
everything else is excellent. A blend is the right shape for *ranking* and the
wrong shape for *permission*. So every hard gate is evaluated independently and
any single failure blocks promotion regardless of the score.

Evidence weighting
------------------
Shadow and live outcomes are combined with live weighted far higher, because they
are not the same kind of evidence: a shadow fill is this repository's own
simulator agreeing with itself, and a live fill is the market disagreeing. Shadow
evidence is sufficient to reach ``LIVE_PROBE`` (there is no other way to start) and
is progressively discounted above it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from app.strategy.catalog import SHORT_STRATEGY_IDS
from app.trading.borrow import BorrowSnapshotStore, default_borrow_store
from app.trading.directional import (
    DirectionalStrategyKey,
    PositionDirection,
    ShortReasonCodes,
    StrategyDeploymentState,
    next_promotion_state,
    parse_state,
    previous_safer_state,
    transition_allowed,
)
from app.trading.directional_shadow import ShadowPlanStore, default_shadow_store
from app.trading.strategy_performance_store import (
    EVALUATION_SOURCE_LIVE,
    EVALUATION_SOURCE_LIVE_PROBE,
    EVALUATION_SOURCE_SHADOW,
    StrategyPerformanceStore,
    default_store as default_performance_store,
)

logger = logging.getLogger(__name__)

DEFAULT_DEPLOYMENT_STORE_PATH = "data/store/strategy-deployment.sqlite3"
DEFAULT_CONFIG_PATH = "config/short_strategy_deployment.yaml"


# --------------------------------------------------------------------------- #
# Thresholds                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromotionThresholds:
    """Hard gates for one rung of the ladder.

    ``None`` means "not gated at this rung", which is different from 0.0. A
    threshold of 0.0 on ``minimum_conservative_edge_bps`` would admit a break-even
    strategy; ``None`` means the rung does not ask that question at all.
    """

    minimum_executable_signals: int = 0
    minimum_filled_trades: int = 0
    minimum_distinct_trading_days: int = 0
    minimum_distinct_symbols: int = 0
    minimum_confidence_score: float = 0.0
    minimum_conservative_edge_bps: float | None = None
    minimum_lower_confidence_bound_net_bps: float | None = None
    minimum_profit_factor: float | None = None
    minimum_cost_coverage_ratio: float | None = None
    maximum_drawdown_bps: float | None = None
    maximum_expected_shortfall_95_bps: float | None = None
    maximum_loss_streak: int | None = None
    maximum_calibration_error: float | None = None
    maximum_mean_absolute_slippage_error_bps: float | None = None
    minimum_borrow_availability_rate: float | None = None
    minimum_data_freshness_pass_rate: float | None = None
    minimum_short_rescue_rate: float | None = None
    maximum_broker_rejection_rate: float | None = None
    required_holdout_windows_passed: int = 0
    required_consecutive_evaluation_cycles: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PromotionThresholds":
        if not isinstance(payload, Mapping):
            return cls()
        fields = set(cls.__dataclass_fields__)
        resolved: dict[str, Any] = {}
        for raw_key, value in payload.items():
            key = _THRESHOLD_ALIASES.get(str(raw_key), str(raw_key))
            if key in fields:
                resolved[key] = value
        return cls(**resolved)


# Built-in defaults. Deliberately at the conservative end of the published spec:
# an operator lowering them has to do so explicitly in YAML, and the change is
# visible in the resolved-config audit event.
DEFAULT_SHADOW_TO_LIVE_PROBE = PromotionThresholds(
    minimum_executable_signals=120,
    minimum_filled_trades=60,
    minimum_distinct_trading_days=20,
    minimum_distinct_symbols=10,
    minimum_confidence_score=0.72,
    minimum_conservative_edge_bps=8.0,
    minimum_lower_confidence_bound_net_bps=3.0,
    minimum_profit_factor=1.15,
    minimum_cost_coverage_ratio=1.7,
    maximum_drawdown_bps=250.0,
    maximum_expected_shortfall_95_bps=120.0,
    maximum_loss_streak=5,
    maximum_calibration_error=0.08,
    maximum_mean_absolute_slippage_error_bps=12.0,
    minimum_borrow_availability_rate=0.7,
    minimum_data_freshness_pass_rate=0.98,
    minimum_short_rescue_rate=0.03,
    required_holdout_windows_passed=3,
    required_consecutive_evaluation_cycles=5,
)

DEFAULT_LIVE_PROBE_TO_LIVE_LIMITED = PromotionThresholds(
    minimum_filled_trades=20,
    minimum_distinct_trading_days=10,
    minimum_confidence_score=0.78,
    minimum_conservative_edge_bps=3.0,
    minimum_profit_factor=1.1,
    maximum_drawdown_bps=150.0,
    maximum_mean_absolute_slippage_error_bps=15.0,
    maximum_broker_rejection_rate=0.03,
    required_consecutive_evaluation_cycles=5,
)

DEFAULT_LIVE_LIMITED_TO_LIVE_FULL = PromotionThresholds(
    minimum_filled_trades=60,
    minimum_distinct_trading_days=30,
    minimum_confidence_score=0.84,
    minimum_conservative_edge_bps=10.0,
    minimum_profit_factor=1.2,
    maximum_drawdown_bps=200.0,
    maximum_broker_rejection_rate=0.02,
    minimum_borrow_availability_rate=0.8,
    required_consecutive_evaluation_cycles=10,
)


@dataclass(frozen=True)
class DemotionThresholds:
    """Triggers for stepping an arm DOWN one rung.

    Every value is looser than the promotion threshold at the same rung, which
    creates a hysteresis band. Without it an arm sitting exactly on its promotion
    boundary would oscillate up and down every cycle, and each oscillation is a real
    change in live position limits.
    """

    live_full_confidence: float = 0.80
    live_full_conservative_edge_bps: float = 5.0
    live_full_profit_factor: float = 1.05
    live_full_slippage_error_bps: float = 15.0
    live_full_borrow_availability_rate: float = 0.70
    live_full_consecutive_failures: int = 2

    live_limited_confidence: float = 0.72
    live_limited_recent_net_mean_bps: float = 0.0
    live_limited_loss_streak: int = 4
    live_limited_drawdown_fraction_of_limit: float = 0.8

    live_probe_confidence: float = 0.65
    live_probe_profit_factor: float = 0.9
    live_probe_conservative_edge_bps: float = 0.0
    live_probe_broker_rejection_rate: float = 0.05

    recent_window: int = 20
    short_recent_window: int = 10

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DemotionThresholds":
        if not isinstance(payload, Mapping):
            return cls()
        fields = {f for f in cls.__dataclass_fields__}
        aliases = {"live_probe_to_shadow_confidence": "live_probe_confidence"}
        resolved = {}
        for raw_key, value in payload.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key in fields:
                resolved[key] = value
        return cls(**resolved)


@dataclass(frozen=True)
class ShortStrategyPromotionConfig:
    enabled: bool = True
    operator_live_full_override: bool = False
    evaluation_interval_seconds: int = 300
    default_initial_state: StrategyDeploymentState = StrategyDeploymentState.SHADOW
    strategies: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    shadow_to_live_probe: PromotionThresholds = DEFAULT_SHADOW_TO_LIVE_PROBE
    live_probe_to_live_limited: PromotionThresholds = DEFAULT_LIVE_PROBE_TO_LIVE_LIMITED
    live_limited_to_live_full: PromotionThresholds = DEFAULT_LIVE_LIMITED_TO_LIVE_FULL
    demotion: DemotionThresholds = field(default_factory=DemotionThresholds)
    immediate_suspend_change_point_probability: float = 0.7
    immediate_suspend_on_position_mismatch: bool = True
    immediate_suspend_on_missing_loan_date: bool = True
    # Live evidence outweighs simulated evidence by this factor when the two are
    # blended. 3.0 means one live trade counts as much as three shadow trades.
    live_evidence_weight: float = 3.0
    # Promotion-holdout window length. The most recent contiguous stretch of shadow
    # data is held out from every threshold-tuning decision.
    holdout_window_days: int = 5
    config_version: str = ""

    def thresholds_for(
        self, state: StrategyDeploymentState
    ) -> PromotionThresholds | None:
        return {
            StrategyDeploymentState.SHADOW: self.shadow_to_live_probe,
            StrategyDeploymentState.LIVE_PROBE: self.live_probe_to_live_limited,
            StrategyDeploymentState.LIVE_LIMITED: self.live_limited_to_live_full,
        }.get(state)

    def strategy_enabled(self, strategy_id: str) -> bool:
        entry = self.strategies.get(strategy_id)
        if not isinstance(entry, Mapping):
            # Not listed == not enabled. An unlisted strategy silently defaulting to
            # enabled is how a half-finished config turns on a short.
            return False
        return bool(entry.get("enabled", False))

    def initial_state_for(self, strategy_id: str) -> StrategyDeploymentState:
        entry = self.strategies.get(strategy_id)
        raw = (
            entry.get("initial_state")
            if isinstance(entry, Mapping)
            else None
        ) or str(self.default_initial_state)
        requested = parse_state(raw, self.default_initial_state)
        # A config file may NOT seed an arm into a live state. This is the load-
        # bearing line for "a short strategy is always SHADOW immediately after it
        # is added": the initial state is clamped, so editing YAML cannot skip the
        # ladder even by accident.
        if requested.submits_orders:
            logger.warning(
                "short_strategy_deployment: initial_state %s for %s is not permitted; "
                "clamped to SHADOW",
                requested,
                strategy_id,
            )
            return StrategyDeploymentState.SHADOW
        return requested

    @classmethod
    def load(
        cls, config_path: str | Path = DEFAULT_CONFIG_PATH
    ) -> "ShortStrategyPromotionConfig":
        raw = _load_yaml(Path(config_path))
        promotion = raw.get("promotion") if isinstance(raw.get("promotion"), Mapping) else {}
        strategies = raw.get("strategies") if isinstance(raw.get("strategies"), Mapping) else {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            operator_live_full_override=bool(
                raw.get("operator_live_full_override", False)
            ),
            evaluation_interval_seconds=max(
                30, int(raw.get("evaluation_interval_seconds", 300) or 300)
            ),
            default_initial_state=parse_state(
                raw.get("default_initial_state"), StrategyDeploymentState.SHADOW
            ),
            strategies={str(k): dict(v) for k, v in strategies.items() if isinstance(v, Mapping)},
            shadow_to_live_probe=_merge_thresholds(
                DEFAULT_SHADOW_TO_LIVE_PROBE, promotion.get("shadow_to_live_probe")
            ),
            live_probe_to_live_limited=_merge_thresholds(
                DEFAULT_LIVE_PROBE_TO_LIVE_LIMITED, promotion.get("live_probe_to_live_limited")
            ),
            live_limited_to_live_full=_merge_thresholds(
                DEFAULT_LIVE_LIMITED_TO_LIVE_FULL, promotion.get("live_limited_to_live_full")
            ),
            demotion=DemotionThresholds.from_mapping(raw.get("demotion")),
            immediate_suspend_change_point_probability=float(
                (raw.get("demotion") or {}).get("immediate_suspend_change_point_probability", 0.7)
            ),
            immediate_suspend_on_position_mismatch=bool(
                (raw.get("demotion") or {}).get("immediate_suspend_on_position_mismatch", True)
            ),
            immediate_suspend_on_missing_loan_date=bool(
                (raw.get("demotion") or {}).get("immediate_suspend_on_missing_loan_date", True)
            ),
            live_evidence_weight=max(1.0, float(raw.get("live_evidence_weight", 3.0))),
            holdout_window_days=max(1, int(raw.get("holdout_window_days", 5) or 5)),
            config_version=str(raw.get("config_version", "") or ""),
        )


def _merge_thresholds(
    base: PromotionThresholds, overlay: Mapping[str, Any] | None
) -> PromotionThresholds:
    """YAML overrides only the thresholds it actually names.

    Deliberately a key-by-key merge rather than a wholesale replacement. Building a
    fresh ``PromotionThresholds`` from a partial YAML block would fill every
    unmentioned field with the DATACLASS default — 0 / None, i.e. "no gate" — so a
    config that tuned one threshold would silently disable all the others. That
    failure mode is invisible and points the wrong way.
    """
    if not isinstance(overlay, Mapping):
        return base
    partial = PromotionThresholds.from_mapping(overlay)
    resolved = asdict(base)
    for key in _written_threshold_keys(overlay):
        resolved[key] = getattr(partial, key)
    return PromotionThresholds(**resolved)


# YAML aliases -> canonical field. The spec's per-rung blocks use rung-specific
# names ("minimum_live_probe_trades") for what is one metric here, so both spellings
# resolve to the same gate instead of silently doing nothing.
_THRESHOLD_ALIASES: Mapping[str, str] = {
    "minimum_live_trades": "minimum_filled_trades",
    "minimum_live_probe_trades": "minimum_filled_trades",
    "minimum_total_live_trades": "minimum_filled_trades",
    "minimum_trading_days": "minimum_distinct_trading_days",
    "minimum_live_days": "minimum_distinct_trading_days",
    "minimum_live_probe_days": "minimum_distinct_trading_days",
    "minimum_total_live_days": "minimum_distinct_trading_days",
    "minimum_live_realized_net_bps": "minimum_conservative_edge_bps",
    "minimum_live_profit_factor": "minimum_profit_factor",
    "maximum_live_drawdown_bps": "maximum_drawdown_bps",
    "maximum_live_slippage_error_bps": "maximum_mean_absolute_slippage_error_bps",
    "required_consecutive_cycles": "required_consecutive_evaluation_cycles",
}


def _written_threshold_keys(overlay: Mapping[str, Any]) -> set[str]:
    fields = set(PromotionThresholds.__dataclass_fields__)
    return {
        _THRESHOLD_ALIASES.get(str(raw_key), str(raw_key))
        for raw_key in overlay
        if _THRESHOLD_ALIASES.get(str(raw_key), str(raw_key)) in fields
    }


# --------------------------------------------------------------------------- #
# Validation snapshot                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectionalValidationSnapshot:
    """Every metric the promotion decision reads, for one arm at one moment.

    Immutable and fully serialised into the audit event, so a promotion months ago
    can be re-argued from the numbers that actually caused it rather than from
    numbers recomputed against today's data.
    """

    key: DirectionalStrategyKey
    evaluated_at: datetime
    state: StrategyDeploymentState
    # -- sample --
    executable_signal_count: int = 0
    filled_trade_count: int = 0
    distinct_trading_days: int = 0
    distinct_symbols: int = 0
    distinct_regimes: int = 0
    # -- edge --
    mean_net_return_bps: float = 0.0
    median_net_return_bps: float = 0.0
    win_rate: float = 0.0
    profit_factor: float | None = None
    conservative_edge_bps: float = 0.0
    lower_confidence_bound_net_bps: float = 0.0
    cost_coverage_ratio: float | None = None
    # -- risk --
    maximum_drawdown_bps: float = 0.0
    expected_shortfall_95_bps: float = 0.0
    loss_streak: int = 0
    # -- calibration / execution --
    prediction_calibration_error: float | None = None
    fill_probability_error: float | None = None
    mean_slippage_error_bps: float | None = None
    # -- borrow --
    borrow_availability_rate: float | None = None
    borrow_rejection_rate: float | None = None
    # -- data / regime --
    data_freshness_pass_rate: float | None = None
    strategy_regime_stability: float | None = None
    change_point_probability: float = 0.0
    # -- the short-specific question --
    # Fraction of evaluation snapshots where the best LONG had no positive edge but
    # the best SHORT did. Measures whether short support actually converts NO_TRADE
    # into a real opportunity, rather than just adding arms.
    short_rescue_rate: float | None = None
    # -- live-only --
    live_trade_count: int = 0
    live_mean_net_bps: float | None = None
    live_profit_factor: float | None = None
    live_maximum_drawdown_bps: float = 0.0
    broker_rejection_rate: float | None = None
    # -- holdout --
    holdout_windows_passed: int = 0
    holdout_windows_evaluated: int = 0
    # -- health flags feeding immediate suspension --
    model_calibrated: bool = False
    broker_state_restored: bool = True
    position_direction_mismatch: bool = False
    loan_date_missing: bool = False
    credit_contract_failure: bool = False
    borrow_quantity_exceeded: bool = False
    data_quality_hard_fail: bool = False
    regime_dislocated: bool = False
    daily_loss_limit_breached: bool = False
    stop_order_submission_failed: bool = False
    abnormal_market_data: bool = False
    duplicate_short_order: bool = False
    confidence_score: float = 0.0
    confidence_components: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key not in {"key", "evaluated_at", "state", "confidence_components"}
        }
        payload.update(self.key.as_dict())
        payload["evaluated_at"] = self.evaluated_at.isoformat()
        payload["state"] = str(self.state)
        payload["confidence_components"] = dict(self.confidence_components)
        return payload


# --------------------------------------------------------------------------- #
# Confidence score                                                             #
# --------------------------------------------------------------------------- #
CONFIDENCE_WEIGHTS: Mapping[str, float] = {
    "edge_quality": 0.25,
    "sample_reliability": 0.20,
    "calibration_quality": 0.15,
    "execution_quality": 0.15,
    "borrow_reliability": 0.10,
    "regime_coverage": 0.10,
    "stability_quality": 0.05,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value != value:  # NaN
        return low
    return max(low, min(high, value))


def _saturating(value: float, target: float) -> float:
    """0 at 0, 1 at ``target``, concave in between.

    Concave rather than linear so the marginal value of the 60th sample is lower
    than the 6th, which is how sample sufficiency actually behaves.
    """
    if target <= 0:
        return 1.0
    return _clamp(math.sqrt(max(0.0, value) / target))


def compute_confidence_components(
    snapshot: DirectionalValidationSnapshot, thresholds: PromotionThresholds
) -> dict[str, float]:
    """Each component in [0, 1], scored against the rung's own thresholds.

    Every unknown resolves to 0.0, never to a neutral 0.5. An unmeasured component
    is not a middling one: if calibration has never been measured, the arm has not
    demonstrated calibration, and a 0.5 would let absence of evidence carry half the
    weight of evidence.
    """
    edge_target = thresholds.minimum_conservative_edge_bps or 10.0
    pf_target = thresholds.minimum_profit_factor or 1.2
    edge_quality = _clamp(
        0.6 * _clamp(snapshot.conservative_edge_bps / max(1e-9, edge_target))
        + 0.4
        * (
            _clamp((snapshot.profit_factor - 1.0) / max(1e-9, pf_target - 1.0))
            if snapshot.profit_factor is not None
            else 0.0
        )
    )
    sample_reliability = _clamp(
        (
            _saturating(snapshot.filled_trade_count, max(1, thresholds.minimum_filled_trades))
            + _saturating(
                snapshot.distinct_trading_days, max(1, thresholds.minimum_distinct_trading_days)
            )
            + _saturating(
                snapshot.distinct_symbols, max(1, thresholds.minimum_distinct_symbols)
            )
        )
        / 3.0
    )
    calibration_ceiling = thresholds.maximum_calibration_error or 0.08
    calibration_quality = (
        _clamp(1.0 - snapshot.prediction_calibration_error / max(1e-9, calibration_ceiling))
        if snapshot.prediction_calibration_error is not None
        else 0.0
    )
    slippage_ceiling = thresholds.maximum_mean_absolute_slippage_error_bps or 12.0
    execution_quality = (
        _clamp(1.0 - abs(snapshot.mean_slippage_error_bps) / max(1e-9, slippage_ceiling))
        if snapshot.mean_slippage_error_bps is not None
        else 0.0
    )
    if snapshot.fill_probability_error is not None:
        execution_quality = _clamp(
            0.7 * execution_quality + 0.3 * _clamp(1.0 - abs(snapshot.fill_probability_error))
        )
    # A LONG arm has no borrow leg, so borrow reliability is vacuously perfect. For a
    # SHORT an unmeasured rate is 0.0 — untested borrow is not reliable borrow.
    if snapshot.key.direction is PositionDirection.LONG:
        borrow_reliability = 1.0
    elif snapshot.borrow_availability_rate is None:
        borrow_reliability = 0.0
    else:
        floor = thresholds.minimum_borrow_availability_rate or 0.7
        borrow_reliability = _clamp(snapshot.borrow_availability_rate / max(1e-9, floor))
    # Two regimes is the minimum that says anything about regime robustness; three
    # is treated as full coverage.
    regime_coverage = _saturating(snapshot.distinct_regimes, 3)
    stability_quality = _clamp(
        (snapshot.strategy_regime_stability if snapshot.strategy_regime_stability is not None else 0.0)
        * (1.0 - _clamp(snapshot.change_point_probability))
    )
    return {
        "edge_quality": edge_quality,
        "sample_reliability": sample_reliability,
        "calibration_quality": calibration_quality,
        "execution_quality": execution_quality,
        "borrow_reliability": borrow_reliability,
        "regime_coverage": regime_coverage,
        "stability_quality": stability_quality,
    }


def compute_confidence_score(components: Mapping[str, float]) -> float:
    return _clamp(
        sum(CONFIDENCE_WEIGHTS[name] * _clamp(float(components.get(name, 0.0))) for name in CONFIDENCE_WEIGHTS)
    )


# --------------------------------------------------------------------------- #
# Decision                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromotionDecision:
    key: DirectionalStrategyKey
    from_state: StrategyDeploymentState
    to_state: StrategyDeploymentState
    changed: bool
    reason_codes: tuple[str, ...]
    consecutive_passes: int
    confidence_score: float
    snapshot: DirectionalValidationSnapshot
    failed_gates: tuple[str, ...] = ()
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def promoted(self) -> bool:
        return self.changed and self.to_state.rank > self.from_state.rank

    @property
    def demoted(self) -> bool:
        return self.changed and self.to_state.rank < self.from_state.rank

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.key.as_dict(),
            "from_state": str(self.from_state),
            "to_state": str(self.to_state),
            "changed": self.changed,
            "promoted": self.promoted,
            "demoted": self.demoted,
            "reason_codes": list(self.reason_codes),
            "failed_gates": list(self.failed_gates),
            "consecutive_passes": self.consecutive_passes,
            "confidence_score": round(self.confidence_score, 4),
            "evaluated_at": self.evaluated_at.isoformat(),
            "snapshot": self.snapshot.as_dict(),
        }


# --------------------------------------------------------------------------- #
# Hard gates                                                                   #
# --------------------------------------------------------------------------- #
def evaluate_hard_gates(
    snapshot: DirectionalValidationSnapshot, thresholds: PromotionThresholds
) -> tuple[str, ...]:
    """Reason codes for every FAILING gate. Empty means all gates pass.

    Every gate is evaluated independently and all failures are returned, not just
    the first. An operator asking "what does this strategy still need" wants the
    whole list; returning one at a time turns a single diagnosis into twenty
    evaluation cycles.
    """
    failures: list[str] = []
    R = ShortReasonCodes

    if snapshot.executable_signal_count < thresholds.minimum_executable_signals:
        failures.append(R.PROMOTION_SAMPLE_INSUFFICIENT)
    if snapshot.filled_trade_count < thresholds.minimum_filled_trades:
        failures.append(R.PROMOTION_SAMPLE_INSUFFICIENT)
    if snapshot.distinct_trading_days < thresholds.minimum_distinct_trading_days:
        failures.append("SHORT_PROMOTION_TRADING_DAYS_INSUFFICIENT")
    if snapshot.distinct_symbols < thresholds.minimum_distinct_symbols:
        failures.append("SHORT_PROMOTION_SYMBOL_BREADTH_INSUFFICIENT")
    if snapshot.confidence_score < thresholds.minimum_confidence_score:
        failures.append(R.CONFIDENCE_BELOW_THRESHOLD)
    if (
        thresholds.minimum_conservative_edge_bps is not None
        and snapshot.conservative_edge_bps < thresholds.minimum_conservative_edge_bps
    ):
        failures.append(R.CONSERVATIVE_EDGE_NON_POSITIVE)
    if (
        thresholds.minimum_lower_confidence_bound_net_bps is not None
        and snapshot.lower_confidence_bound_net_bps
        < thresholds.minimum_lower_confidence_bound_net_bps
    ):
        failures.append("SHORT_LOWER_CONFIDENCE_BOUND_INSUFFICIENT")
    if thresholds.minimum_profit_factor is not None:
        # An unmeasurable profit factor (no losing trades yet) FAILS rather than
        # passes. A 3-trade all-winners sample is the classic small-sample fluke,
        # and treating it as infinite profit factor would promote on it.
        if snapshot.profit_factor is None or snapshot.profit_factor < thresholds.minimum_profit_factor:
            failures.append("SHORT_PROFIT_FACTOR_INSUFFICIENT")
    if thresholds.minimum_cost_coverage_ratio is not None:
        if (
            snapshot.cost_coverage_ratio is None
            or snapshot.cost_coverage_ratio < thresholds.minimum_cost_coverage_ratio
        ):
            failures.append(R.COST_COVERAGE_INSUFFICIENT)
    if (
        thresholds.maximum_drawdown_bps is not None
        and snapshot.maximum_drawdown_bps > thresholds.maximum_drawdown_bps
    ):
        failures.append(R.DRAWDOWN_EXCEEDED)
    if (
        thresholds.maximum_expected_shortfall_95_bps is not None
        and snapshot.expected_shortfall_95_bps > thresholds.maximum_expected_shortfall_95_bps
    ):
        failures.append("SHORT_EXPECTED_SHORTFALL_EXCEEDED")
    if (
        thresholds.maximum_loss_streak is not None
        and snapshot.loss_streak > thresholds.maximum_loss_streak
    ):
        failures.append(R.LOSS_STREAK_EXCEEDED)
    if thresholds.maximum_calibration_error is not None:
        if (
            snapshot.prediction_calibration_error is None
            or snapshot.prediction_calibration_error > thresholds.maximum_calibration_error
        ):
            failures.append(R.CALIBRATION_ERROR_HIGH)
    if thresholds.maximum_mean_absolute_slippage_error_bps is not None:
        if (
            snapshot.mean_slippage_error_bps is None
            or abs(snapshot.mean_slippage_error_bps)
            > thresholds.maximum_mean_absolute_slippage_error_bps
        ):
            failures.append(R.SLIPPAGE_ERROR_HIGH)
    if thresholds.minimum_borrow_availability_rate is not None and snapshot.key.is_short:
        if (
            snapshot.borrow_availability_rate is None
            or snapshot.borrow_availability_rate < thresholds.minimum_borrow_availability_rate
        ):
            failures.append(R.BORROW_AVAILABILITY_RATE_LOW)
    if thresholds.minimum_data_freshness_pass_rate is not None:
        if (
            snapshot.data_freshness_pass_rate is None
            or snapshot.data_freshness_pass_rate < thresholds.minimum_data_freshness_pass_rate
        ):
            failures.append(R.DATA_QUALITY_FAILED)
    if thresholds.minimum_short_rescue_rate is not None and snapshot.key.is_short:
        if (
            snapshot.short_rescue_rate is None
            or snapshot.short_rescue_rate < thresholds.minimum_short_rescue_rate
        ):
            failures.append(R.RESCUE_RATE_INSUFFICIENT)
    if thresholds.maximum_broker_rejection_rate is not None:
        if (
            snapshot.broker_rejection_rate is not None
            and snapshot.broker_rejection_rate > thresholds.maximum_broker_rejection_rate
        ):
            failures.append(R.BROKER_REJECTION_RATE_HIGH)
    if snapshot.holdout_windows_passed < thresholds.required_holdout_windows_passed:
        failures.append(R.HOLDOUT_NOT_PASSED)
    # Model calibration is a SEPARATE precondition from strategy validation, and it
    # is required for every live rung. A strategy cannot be promoted on the strength
    # of an uncalibrated model's predictions even if its realized outcomes look good,
    # because those outcomes were selected BY those predictions.
    if not snapshot.model_calibrated:
        failures.append(R.MODEL_NOT_CALIBRATED)
    return tuple(dict.fromkeys(failures))


def evaluate_immediate_suspension(
    snapshot: DirectionalValidationSnapshot, config: ShortStrategyPromotionConfig
) -> tuple[str, ...]:
    """Faults that suspend an arm THIS cycle, with no consecutive-failure grace.

    Every one of these is a statement that internal state and broker state disagree,
    or that the market is not in a condition this thesis was validated in. Neither
    is something to average over several cycles: if the position we think we hold is
    not the position that exists, one more cycle of trading makes it worse.
    """
    R = ShortReasonCodes
    failures: list[str] = []
    if config.immediate_suspend_on_position_mismatch and snapshot.position_direction_mismatch:
        failures.append(R.POSITION_DIRECTION_MISMATCH)
    if config.immediate_suspend_on_missing_loan_date and snapshot.loan_date_missing:
        failures.append(R.LOAN_DATE_MISSING)
    if snapshot.credit_contract_failure:
        failures.append(R.ORDER_CONTRACT_INCOMPLETE)
    if snapshot.borrow_quantity_exceeded:
        failures.append(R.BORROW_QUANTITY_INSUFFICIENT)
    if snapshot.data_quality_hard_fail:
        failures.append(R.DATA_QUALITY_FAILED)
    if snapshot.regime_dislocated:
        failures.append(R.HIGH_VOL_DISLOCATED)
    if snapshot.change_point_probability >= config.immediate_suspend_change_point_probability:
        failures.append(R.REGIME_UNSTABLE)
    if snapshot.daily_loss_limit_breached:
        failures.append(R.DAILY_LOSS_LIMIT)
    if snapshot.stop_order_submission_failed:
        failures.append(R.STOP_ORDER_CAPABILITY_MISSING)
    if not snapshot.broker_state_restored:
        failures.append(R.BROKER_STATE_UNRESTORED)
    if snapshot.abnormal_market_data:
        failures.append("SHORT_ABNORMAL_MARKET_DATA")
    if snapshot.duplicate_short_order:
        failures.append("SHORT_DUPLICATE_ORDER_DETECTED")
    return tuple(dict.fromkeys(failures))


def evaluate_demotion(
    snapshot: DirectionalValidationSnapshot,
    config: ShortStrategyPromotionConfig,
    *,
    consecutive_failures: int,
) -> tuple[str, ...]:
    """Reason codes demanding a step DOWN from the arm's current state."""
    cfg = config.demotion
    state = snapshot.state
    failures: list[str] = []
    if state is StrategyDeploymentState.LIVE_FULL:
        if (
            snapshot.confidence_score < cfg.live_full_confidence
            and consecutive_failures + 1 >= cfg.live_full_consecutive_failures
        ):
            failures.append(ShortReasonCodes.CONFIDENCE_BELOW_THRESHOLD)
        if snapshot.conservative_edge_bps < cfg.live_full_conservative_edge_bps:
            failures.append(ShortReasonCodes.CONSERVATIVE_EDGE_NON_POSITIVE)
        if (
            snapshot.live_profit_factor is not None
            and snapshot.live_profit_factor < cfg.live_full_profit_factor
        ):
            failures.append("SHORT_PROFIT_FACTOR_INSUFFICIENT")
        if (
            snapshot.mean_slippage_error_bps is not None
            and abs(snapshot.mean_slippage_error_bps) > cfg.live_full_slippage_error_bps
        ):
            failures.append(ShortReasonCodes.SLIPPAGE_ERROR_HIGH)
        if (
            snapshot.key.is_short
            and snapshot.borrow_availability_rate is not None
            and snapshot.borrow_availability_rate < cfg.live_full_borrow_availability_rate
        ):
            failures.append(ShortReasonCodes.BORROW_AVAILABILITY_RATE_LOW)
    elif state is StrategyDeploymentState.LIVE_LIMITED:
        if snapshot.confidence_score < cfg.live_limited_confidence:
            failures.append(ShortReasonCodes.CONFIDENCE_BELOW_THRESHOLD)
        if (
            snapshot.live_mean_net_bps is not None
            and snapshot.live_mean_net_bps <= cfg.live_limited_recent_net_mean_bps
        ):
            failures.append("SHORT_LIVE_NET_NON_POSITIVE")
        if snapshot.loss_streak >= cfg.live_limited_loss_streak:
            failures.append(ShortReasonCodes.LOSS_STREAK_EXCEEDED)
        limit = config.live_limited_to_live_full.maximum_drawdown_bps
        if (
            limit is not None
            and snapshot.maximum_drawdown_bps
            >= limit * cfg.live_limited_drawdown_fraction_of_limit
        ):
            failures.append(ShortReasonCodes.DRAWDOWN_EXCEEDED)
    elif state is StrategyDeploymentState.LIVE_PROBE:
        if snapshot.confidence_score < cfg.live_probe_confidence:
            failures.append(ShortReasonCodes.CONFIDENCE_BELOW_THRESHOLD)
        if (
            snapshot.live_profit_factor is not None
            and snapshot.live_profit_factor < cfg.live_probe_profit_factor
        ):
            failures.append("SHORT_PROFIT_FACTOR_INSUFFICIENT")
        if snapshot.conservative_edge_bps <= cfg.live_probe_conservative_edge_bps:
            failures.append(ShortReasonCodes.CONSERVATIVE_EDGE_NON_POSITIVE)
        if (
            snapshot.broker_rejection_rate is not None
            and snapshot.broker_rejection_rate > cfg.live_probe_broker_rejection_rate
        ):
            failures.append(ShortReasonCodes.BROKER_REJECTION_RATE_HIGH)
    return tuple(dict.fromkeys(failures))


# --------------------------------------------------------------------------- #
# Deployment state store                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeploymentRecord:
    key: DirectionalStrategyKey
    state: StrategyDeploymentState
    state_version: int
    confidence_score: float
    consecutive_passes: int
    consecutive_failures: int
    last_transition_at: datetime | None
    last_reason_codes: tuple[str, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.key.as_dict(),
            "state": str(self.state),
            "state_version": self.state_version,
            "confidence_score": round(self.confidence_score, 4),
            "consecutive_passes": self.consecutive_passes,
            "consecutive_failures": self.consecutive_failures,
            "last_transition_at": (
                self.last_transition_at.isoformat() if self.last_transition_at else None
            ),
            "last_reason_codes": list(self.last_reason_codes),
            "submits_orders": self.state.submits_orders,
            "metrics": dict(self.metrics),
        }


class DeploymentStateStore:
    """Authoritative, versioned deployment state with an append-only audit log.

    State changes and their audit event are written in ONE transaction. Without
    that, a crash between the two produces a live-authorised arm with no record of
    why — which is unauditable in exactly the situation where an audit matters.

    Readers see only committed state (``state_version`` increases monotonically), so
    a runtime cache can detect a change without holding a lock across the trading
    loop.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path or os.getenv("STRATEGY_DEPLOYMENT_STORE_PATH", DEFAULT_DEPLOYMENT_STORE_PATH)
        )
        self._lock = threading.RLock()
        self._available = True
        self._migrate()

    def get(self, key: DirectionalStrategyKey) -> DeploymentRecord | None:
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    select strategy_key, state, state_version, confidence_score,
                           consecutive_passes, consecutive_failures, last_transition_at,
                           last_reason_codes, metrics_json
                    from directional_strategy_deployment where strategy_key = ?
                    """,
                    (key.as_text(),),
                ).fetchone()
        except sqlite3.Error:
            return None
        return _record_from_row(row) if row else None

    def state_of(
        self,
        key: DirectionalStrategyKey,
        *,
        default: StrategyDeploymentState = StrategyDeploymentState.SHADOW,
    ) -> StrategyDeploymentState:
        """Committed state, or ``default`` when there is no record.

        The default is SHADOW and callers should not override it upward. An arm with
        no persisted state has never been evaluated, and "never evaluated" must read
        as "not authorised".
        """
        record = self.get(key)
        return record.state if record is not None else default

    def all_records(self) -> tuple[DeploymentRecord, ...]:
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    select strategy_key, state, state_version, confidence_score,
                           consecutive_passes, consecutive_failures, last_transition_at,
                           last_reason_codes, metrics_json
                    from directional_strategy_deployment order by strategy_key
                    """
                ).fetchall()
        except sqlite3.Error:
            return ()
        return tuple(_record_from_row(row) for row in rows)

    def ensure(
        self, key: DirectionalStrategyKey, state: StrategyDeploymentState
    ) -> DeploymentRecord:
        """Create the record if absent; never downgrade or upgrade an existing one."""
        existing = self.get(key)
        if existing is not None:
            return existing
        self._write(
            key,
            state=state,
            confidence_score=0.0,
            consecutive_passes=0,
            consecutive_failures=0,
            reason_codes=("DEPLOYMENT_STATE_INITIALISED",),
            metrics={},
            from_state=None,
            transitioned=True,
            audit_reason="INITIALISED",
        )
        record = self.get(key)
        assert record is not None  # just written
        return record

    def apply(
        self,
        decision: PromotionDecision,
        *,
        config_version: str = "",
        model_version: str = "",
    ) -> bool:
        """Commit a decision atomically, with its audit event.

        Refuses any transition outside the whitelist. That check lives here — at the
        persistence boundary — rather than only in the controller, so a future caller
        that builds a decision by hand still cannot write ``SHADOW -> LIVE_FULL``.
        """
        if not transition_allowed(decision.from_state, decision.to_state):
            logger.error(
                "refusing forbidden deployment transition %s: %s -> %s",
                decision.key.as_text(),
                decision.from_state,
                decision.to_state,
            )
            return False
        return self._write(
            decision.key,
            state=decision.to_state,
            confidence_score=decision.confidence_score,
            consecutive_passes=decision.consecutive_passes,
            consecutive_failures=(
                0 if decision.changed else _failure_count_from(decision)
            ),
            reason_codes=decision.reason_codes,
            metrics=decision.snapshot.as_dict(),
            from_state=decision.from_state,
            transitioned=decision.changed,
            audit_reason="TRANSITION" if decision.changed else "EVALUATION",
            config_version=config_version,
            model_version=model_version,
        )

    def force_state(
        self,
        key: DirectionalStrategyKey,
        state: StrategyDeploymentState,
        *,
        actor: str,
        reason: str,
    ) -> bool:
        """Operator override. Still bound by the transition whitelist.

        A manual override may move an arm ONE legal step; it cannot skip the ladder.
        It always writes an audit event tagged with the actor, because the spec
        requires that even a policy bypass is explicitly recorded — an undocumented
        manual promotion is indistinguishable from a bug six months later.
        """
        current = self.state_of(key)
        if not transition_allowed(current, state):
            logger.error(
                "manual override refused for %s: %s -> %s is not an allowed transition",
                key.as_text(),
                current,
                state,
            )
            return False
        return self._write(
            key,
            state=state,
            confidence_score=0.0,
            consecutive_passes=0,
            consecutive_failures=0,
            reason_codes=(f"MANUAL_OVERRIDE:{actor}", reason or "UNSPECIFIED"),
            metrics={"manual_override": True, "actor": actor, "reason": reason},
            from_state=current,
            transitioned=current is not state,
            audit_reason="MANUAL_OVERRIDE",
        )

    def audit_history(
        self, key: DirectionalStrategyKey | None = None, *, limit: int = 100
    ) -> tuple[dict[str, Any], ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if key is not None:
            clauses.append("strategy_key = ?")
            params.append(key.as_text())
        where = f"where {' and '.join(clauses)} " if clauses else ""
        params.append(max(1, int(limit)))
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    "select event_id, strategy_key, from_state, to_state, timestamp, "
                    "reason_codes, event_reason, config_version, model_version, metrics_json "
                    f"from promotion_audit {where}order by timestamp desc, rowid desc limit ?",
                    params,
                ).fetchall()
        except sqlite3.Error:
            return ()
        events: list[dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "event_id": str(row[0]),
                    "strategy_key": str(row[1]),
                    "from_state": row[2],
                    "to_state": str(row[3]),
                    "timestamp": str(row[4]),
                    "reason_codes": _json_or(row[5], []),
                    "event_reason": str(row[6] or ""),
                    "config_version": str(row[7] or ""),
                    "model_version": str(row[8] or ""),
                    "metrics": _json_or(row[9], {}),
                }
            )
        return tuple(events)

    # -- internals ---------------------------------------------------------- #
    def _write(
        self,
        key: DirectionalStrategyKey,
        *,
        state: StrategyDeploymentState,
        confidence_score: float,
        consecutive_passes: int,
        consecutive_failures: int,
        reason_codes: Sequence[str],
        metrics: Mapping[str, Any],
        from_state: StrategyDeploymentState | None,
        transitioned: bool,
        audit_reason: str,
        config_version: str = "",
        model_version: str = "",
    ) -> bool:
        if not self._available:
            return False
        moment = datetime.now(timezone.utc)
        try:
            with self._lock, closing(self._connect()) as conn:
                # ONE transaction for state + audit. A crash between them would
                # leave a live-authorised arm with no record of why.
                conn.execute("begin immediate")
                conn.execute(
                    """
                    insert into directional_strategy_deployment(
                        strategy_key, strategy_id, direction, market, execution_product,
                        state, state_version, confidence_score, consecutive_passes,
                        consecutive_failures, last_transition_at, last_reason_codes,
                        metrics_json, updated_at
                    ) values (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(strategy_key) do update set
                        state = excluded.state,
                        state_version = directional_strategy_deployment.state_version + 1,
                        confidence_score = excluded.confidence_score,
                        consecutive_passes = excluded.consecutive_passes,
                        consecutive_failures = excluded.consecutive_failures,
                        last_transition_at = case
                            when excluded.state != directional_strategy_deployment.state
                            then excluded.last_transition_at
                            else directional_strategy_deployment.last_transition_at end,
                        last_reason_codes = excluded.last_reason_codes,
                        metrics_json = excluded.metrics_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        key.as_text(),
                        key.strategy_id,
                        str(key.direction),
                        key.market,
                        str(key.execution_product),
                        str(state),
                        float(confidence_score),
                        int(consecutive_passes),
                        int(consecutive_failures),
                        moment.isoformat() if transitioned else None,
                        json.dumps(list(reason_codes), ensure_ascii=False),
                        json.dumps(dict(metrics), ensure_ascii=False, default=str),
                        moment.isoformat(),
                    ),
                )
                if transitioned:
                    conn.execute(
                        """
                        insert into promotion_audit(
                            event_id, strategy_key, from_state, to_state, timestamp,
                            reason_codes, event_reason, config_version, model_version,
                            metrics_json
                        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"promo-{uuid4().hex}",
                            key.as_text(),
                            str(from_state) if from_state is not None else None,
                            str(state),
                            moment.isoformat(),
                            json.dumps(list(reason_codes), ensure_ascii=False),
                            audit_reason,
                            config_version,
                            model_version,
                            json.dumps(dict(metrics), ensure_ascii=False, default=str),
                        ),
                    )
                conn.commit()
        except sqlite3.Error:
            logger.exception("failed to persist deployment state for %s", key.as_text())
            return False
        return True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.execute("pragma journal_mode = wal")
        return conn

    def _migrate(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, closing(self._connect()) as conn:
                conn.executescript(
                    """
                    create table if not exists directional_strategy_deployment (
                        strategy_key text primary key,
                        strategy_id text not null,
                        direction text not null,
                        market text not null,
                        execution_product text not null,
                        state text not null,
                        state_version integer not null default 1,
                        confidence_score real not null default 0.0,
                        consecutive_passes integer not null default 0,
                        consecutive_failures integer not null default 0,
                        last_transition_at text,
                        last_reason_codes text,
                        metrics_json text,
                        updated_at text
                    );
                    create table if not exists promotion_audit (
                        event_id text primary key,
                        strategy_key text not null,
                        from_state text,
                        to_state text not null,
                        timestamp text not null,
                        reason_codes text,
                        event_reason text,
                        config_version text,
                        model_version text,
                        metrics_json text
                    );
                    create index if not exists idx_promotion_audit_key
                        on promotion_audit(strategy_key, timestamp desc);
                    create index if not exists idx_promotion_audit_time
                        on promotion_audit(timestamp desc);
                    """
                )
        except (OSError, sqlite3.Error):
            # An unwritable store means no arm can be promoted, and every
            # ``state_of`` returns SHADOW. Shorting stops; longs are unaffected.
            self._available = False


def _record_from_row(row: Sequence[Any]) -> DeploymentRecord:
    return DeploymentRecord(
        key=DirectionalStrategyKey.parse(str(row[0])),
        state=parse_state(row[1]),
        state_version=int(row[2] or 1),
        confidence_score=float(row[3] or 0.0),
        consecutive_passes=int(row[4] or 0),
        consecutive_failures=int(row[5] or 0),
        last_transition_at=_parse_iso(row[6]),
        last_reason_codes=tuple(_json_or(row[7], [])),
        metrics=_json_or(row[8], {}),
    )


def _failure_count_from(decision: PromotionDecision) -> int:
    return 1 if decision.failed_gates else 0


# --------------------------------------------------------------------------- #
# Controller                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimeHealth:
    """Live operational facts the controller cannot derive from stored outcomes.

    Supplied by the engine each cycle. Defaults are the SAFE reading of "unknown"
    for each field: ``model_calibrated=False`` blocks promotion, while
    ``broker_state_restored=True`` avoids suspending an arm merely because nobody
    reported on reconciliation. The difference is which way the absence of
    information should push, judged per field rather than uniformly.
    """

    model_calibrated: bool = False
    broker_state_restored: bool = True
    position_direction_mismatch: bool = False
    loan_date_missing: bool = False
    credit_contract_failure: bool = False
    borrow_quantity_exceeded: bool = False
    data_quality_hard_fail: bool = False
    regime_dislocated: bool = False
    daily_loss_limit_breached: bool = False
    stop_order_submission_failed: bool = False
    abnormal_market_data: bool = False
    duplicate_short_order: bool = False
    change_point_probability: float = 0.0
    data_freshness_pass_rate: float | None = None
    regime_stability: float | None = None
    broker_rejection_rate: float | None = None
    short_rescue_rate: float | None = None
    fill_probability_error: float | None = None
    model_version: str = ""


class ShortStrategyPromotionController:
    """Evaluates every managed arm and moves it at most one rung per cycle.

    "At most one rung" is not a performance concession — it is the ladder. Each rung
    exists to gather evidence at a specific position size, and skipping one means
    trading a size that was never validated.
    """

    def __init__(
        self,
        *,
        config: ShortStrategyPromotionConfig | None = None,
        state_store: DeploymentStateStore | None = None,
        performance_store: StrategyPerformanceStore | None = None,
        shadow_store: ShadowPlanStore | None = None,
        borrow_store: BorrowSnapshotStore | None = None,
    ) -> None:
        self.config = config or ShortStrategyPromotionConfig.load()
        self.state_store = state_store or DeploymentStateStore()
        self.performance_store = performance_store or default_performance_store()
        self.shadow_store = shadow_store or default_shadow_store()
        self.borrow_store = borrow_store or default_borrow_store()
        self._last_evaluated_at: datetime | None = None

    # -- public API --------------------------------------------------------- #
    def managed_keys(self, markets: Sequence[str] = ("KR",)) -> tuple[DirectionalStrategyKey, ...]:
        """Every short arm the config enables, one per market."""
        keys: list[DirectionalStrategyKey] = []
        for strategy_id in SHORT_STRATEGY_IDS:
            if not self.config.strategy_enabled(strategy_id):
                continue
            for market in markets:
                keys.append(DirectionalStrategyKey.for_short(strategy_id, market))
        return tuple(keys)

    def authorized_state(self, key: DirectionalStrategyKey) -> StrategyDeploymentState:
        """The committed state, defaulting to SHADOW when unknown."""
        if self.config.operator_live_full_override and self.config.strategy_enabled(
            key.strategy_id
        ):
            return StrategyDeploymentState.LIVE_FULL
        return self.state_store.state_of(key)

    def may_submit_orders(self, key: DirectionalStrategyKey) -> tuple[bool, tuple[str, ...]]:
        """Is this arm authorised to place a real order right now?

        Advisory in the sense that the submission path applies its own gates on top —
        but authoritative in the sense that a False here is final. Both layers must
        agree before an order exists.
        """
        if self.config.operator_live_full_override and self.config.strategy_enabled(
            key.strategy_id
        ):
            return True, ("OPERATOR_LIVE_FULL_OVERRIDE",)
        record = self.state_store.get(key)
        if record is None:
            return False, (ShortReasonCodes.SHADOW_ONLY,)
        state = record.state
        if state is StrategyDeploymentState.SUSPENDED:
            return False, (ShortReasonCodes.DEPLOYMENT_SUSPENDED, *record.last_reason_codes)
        if state is StrategyDeploymentState.DISABLED:
            return False, (ShortReasonCodes.DEPLOYMENT_DISABLED,)
        if state is StrategyDeploymentState.SHADOW:
            return False, (ShortReasonCodes.SHADOW_ONLY,)
        return True, ()

    def evaluate_all(
        self,
        *,
        health: Mapping[str, RuntimeHealth] | RuntimeHealth | None = None,
        markets: Sequence[str] = ("KR",),
        now: datetime | None = None,
    ) -> tuple[PromotionDecision, ...]:
        if not self.config.enabled:
            return ()
        moment = _aware(now or datetime.now(timezone.utc))
        decisions: list[PromotionDecision] = []
        for key in self.managed_keys(markets):
            arm_health = (
                health.get(key.as_text(), RuntimeHealth())
                if isinstance(health, Mapping)
                else (health or RuntimeHealth())
            )
            decisions.append(self.evaluate(key, health=arm_health, now=moment))
        self._last_evaluated_at = moment
        return tuple(decisions)

    def evaluate(
        self,
        key: DirectionalStrategyKey,
        *,
        health: RuntimeHealth | None = None,
        now: datetime | None = None,
    ) -> PromotionDecision:
        """One arm, one cycle. Returns the decision, already committed."""
        moment = _aware(now or datetime.now(timezone.utc))
        record = self.state_store.ensure(key, self.config.initial_state_for(key.strategy_id))
        snapshot = self.build_snapshot(key, record.state, health=health, now=moment)
        decision = self.decide(snapshot, record, now=moment)
        self.state_store.apply(
            decision,
            config_version=self.config.config_version,
            model_version=(health.model_version if health else ""),
        )
        return decision

    # -- decision ----------------------------------------------------------- #
    def decide(
        self,
        snapshot: DirectionalValidationSnapshot,
        record: DeploymentRecord,
        *,
        now: datetime | None = None,
    ) -> PromotionDecision:
        """Pure decision function: snapshot + current record -> next state.

        Order is fixed and matters: suspend, then demote, then promote. Evaluating
        promotion first would let an arm meeting its promotion gates be promoted in
        the same cycle a suspension condition fired, because the gates and the faults
        measure different things and can both be true.
        """
        moment = _aware(now or datetime.now(timezone.utc))
        current = record.state
        suspensions = evaluate_immediate_suspension(snapshot, self.config)
        if suspensions:
            target = StrategyDeploymentState.SUSPENDED
            return PromotionDecision(
                key=snapshot.key,
                from_state=current,
                to_state=target if transition_allowed(current, target) else current,
                changed=current is not target and transition_allowed(current, target),
                reason_codes=(ShortReasonCodes.DEPLOYMENT_SUSPENDED, *suspensions),
                consecutive_passes=0,
                confidence_score=snapshot.confidence_score,
                snapshot=snapshot,
                failed_gates=suspensions,
                evaluated_at=moment,
            )

        demotions = evaluate_demotion(
            snapshot, self.config, consecutive_failures=record.consecutive_failures
        )
        if demotions:
            target = previous_safer_state(current)
            if target is not None:
                return PromotionDecision(
                    key=snapshot.key,
                    from_state=current,
                    to_state=target,
                    changed=True,
                    reason_codes=("SHORT_AUTOMATIC_DEMOTION", *demotions),
                    consecutive_passes=0,
                    confidence_score=snapshot.confidence_score,
                    snapshot=snapshot,
                    failed_gates=demotions,
                    evaluated_at=moment,
                )
            # Already at the bottom of the live ladder; record the failure without a
            # state change so the streak keeps accumulating.
            return PromotionDecision(
                key=snapshot.key,
                from_state=current,
                to_state=current,
                changed=False,
                reason_codes=("SHORT_DEMOTION_FLOOR_REACHED", *demotions),
                consecutive_passes=0,
                confidence_score=snapshot.confidence_score,
                snapshot=snapshot,
                failed_gates=demotions,
                evaluated_at=moment,
            )

        thresholds = self.config.thresholds_for(current)
        target = next_promotion_state(current)
        if thresholds is None or target is None:
            return PromotionDecision(
                key=snapshot.key,
                from_state=current,
                to_state=current,
                changed=False,
                reason_codes=("SHORT_DEPLOYMENT_STATE_TERMINAL",),
                consecutive_passes=record.consecutive_passes,
                confidence_score=snapshot.confidence_score,
                snapshot=snapshot,
                evaluated_at=moment,
            )

        failed = evaluate_hard_gates(snapshot, thresholds)
        if failed:
            # A failed cycle RESETS the streak rather than decrementing it. The gate
            # asks for N consecutive passes; "5 passes, 1 fail, 1 pass" is not 5
            # consecutive passes, and decrementing would let an arm that passes 60%
            # of cycles eventually accumulate its way to live.
            return PromotionDecision(
                key=snapshot.key,
                from_state=current,
                to_state=current,
                changed=False,
                reason_codes=failed,
                consecutive_passes=0,
                confidence_score=snapshot.confidence_score,
                snapshot=snapshot,
                failed_gates=failed,
                evaluated_at=moment,
            )

        passes = record.consecutive_passes + 1
        if passes < thresholds.required_consecutive_evaluation_cycles:
            return PromotionDecision(
                key=snapshot.key,
                from_state=current,
                to_state=current,
                changed=False,
                reason_codes=(
                    ShortReasonCodes.CONSECUTIVE_CYCLES_PENDING,
                    f"PASSES:{passes}/{thresholds.required_consecutive_evaluation_cycles}",
                ),
                consecutive_passes=passes,
                confidence_score=snapshot.confidence_score,
                snapshot=snapshot,
                evaluated_at=moment,
            )
        if not transition_allowed(current, target):
            # Defence in depth: the whitelist is also checked in the store. Reaching
            # here would mean ``next_promotion_state`` and ``ALLOWED_TRANSITIONS``
            # disagree, which is a bug worth surfacing rather than silently allowing.
            logger.error(
                "promotion target %s not permitted from %s for %s",
                target,
                current,
                snapshot.key.as_text(),
            )
            return PromotionDecision(
                key=snapshot.key,
                from_state=current,
                to_state=current,
                changed=False,
                reason_codes=("SHORT_PROMOTION_TRANSITION_FORBIDDEN",),
                consecutive_passes=0,
                confidence_score=snapshot.confidence_score,
                snapshot=snapshot,
                evaluated_at=moment,
            )
        return PromotionDecision(
            key=snapshot.key,
            from_state=current,
            to_state=target,
            changed=True,
            reason_codes=(
                "SHORT_AUTOMATIC_PROMOTION",
                f"CONSECUTIVE_PASSES:{passes}",
                f"CONFIDENCE:{snapshot.confidence_score:.3f}",
            ),
            # Reset on transition: the NEXT rung's consecutive-pass requirement is
            # its own, and carrying the count forward would promote twice on one
            # body of evidence.
            consecutive_passes=0,
            confidence_score=snapshot.confidence_score,
            snapshot=snapshot,
            evaluated_at=moment,
        )

    # -- snapshot construction ---------------------------------------------- #
    def build_snapshot(
        self,
        key: DirectionalStrategyKey,
        state: StrategyDeploymentState,
        *,
        health: RuntimeHealth | None = None,
        now: datetime | None = None,
    ) -> DirectionalValidationSnapshot:
        moment = _aware(now or datetime.now(timezone.utc))
        facts = health or RuntimeHealth()
        thresholds = self.config.thresholds_for(state) or DEFAULT_SHADOW_TO_LIVE_PROBE

        # Evidence sources depend on the rung. SHADOW may only be promoted on shadow
        # evidence (there is nothing else yet); every live rung requires LIVE
        # evidence, so an arm cannot climb past LIVE_PROBE on simulator output.
        if state is StrategyDeploymentState.SHADOW:
            sources = (EVALUATION_SOURCE_SHADOW,)
        else:
            sources = (EVALUATION_SOURCE_LIVE_PROBE, EVALUATION_SOURCE_LIVE)
        metrics = self.performance_store.directional_metrics(
            key, evaluation_sources=sources
        )
        live_metrics = (
            self.performance_store.directional_metrics(
                key, evaluation_sources=(EVALUATION_SOURCE_LIVE_PROBE, EVALUATION_SOURCE_LIVE)
            )
            if state is not StrategyDeploymentState.SHADOW
            else {}
        )
        posterior = self.performance_store.posterior_for_key(
            key, change_point_probability=facts.change_point_probability
        )
        holdout_passed, holdout_evaluated = self._holdout_result(key, moment, thresholds)
        coverage = self._cost_coverage_ratio(key, sources)
        calibration_error = self._calibration_error(key, sources)
        borrow_rate = metrics.get("borrow_availability_rate")
        if borrow_rate is None and key.is_short:
            # Fall back to the desk-wide rate over the same window. Still ``None`` if
            # nothing was ever asked, which fails the gate rather than passing it.
            borrow_rate = self.borrow_store.availability_rate(
                since=moment - timedelta(days=self.config.holdout_window_days)
            )

        partial = DirectionalValidationSnapshot(
            key=key,
            evaluated_at=moment,
            state=state,
            executable_signal_count=int(metrics.get("executable_signal_count") or 0),
            filled_trade_count=int(metrics.get("filled_trade_count") or 0),
            distinct_trading_days=int(metrics.get("distinct_trading_days") or 0),
            distinct_symbols=int(metrics.get("distinct_symbols") or 0),
            distinct_regimes=int(metrics.get("distinct_regimes") or 0),
            mean_net_return_bps=float(metrics.get("mean_net_return_bps") or 0.0),
            median_net_return_bps=float(metrics.get("median_net_return_bps") or 0.0),
            win_rate=float(metrics.get("win_rate") or 0.0),
            profit_factor=metrics.get("profit_factor"),
            conservative_edge_bps=posterior.conservative_edge_bps,
            lower_confidence_bound_net_bps=posterior.conservative_edge_bps,
            cost_coverage_ratio=coverage,
            maximum_drawdown_bps=float(metrics.get("maximum_drawdown_bps") or 0.0),
            expected_shortfall_95_bps=float(metrics.get("expected_shortfall_95_bps") or 0.0),
            loss_streak=int(metrics.get("loss_streak") or 0),
            prediction_calibration_error=calibration_error,
            fill_probability_error=facts.fill_probability_error,
            mean_slippage_error_bps=metrics.get("mean_slippage_error_bps"),
            borrow_availability_rate=borrow_rate,
            borrow_rejection_rate=(None if borrow_rate is None else 1.0 - float(borrow_rate)),
            data_freshness_pass_rate=facts.data_freshness_pass_rate,
            strategy_regime_stability=facts.regime_stability,
            change_point_probability=facts.change_point_probability,
            short_rescue_rate=facts.short_rescue_rate,
            live_trade_count=int(live_metrics.get("filled_trade_count") or 0),
            live_mean_net_bps=live_metrics.get("mean_net_return_bps"),
            live_profit_factor=live_metrics.get("profit_factor"),
            live_maximum_drawdown_bps=float(live_metrics.get("maximum_drawdown_bps") or 0.0),
            broker_rejection_rate=facts.broker_rejection_rate,
            holdout_windows_passed=holdout_passed,
            holdout_windows_evaluated=holdout_evaluated,
            model_calibrated=facts.model_calibrated,
            broker_state_restored=facts.broker_state_restored,
            position_direction_mismatch=facts.position_direction_mismatch,
            loan_date_missing=facts.loan_date_missing,
            credit_contract_failure=facts.credit_contract_failure,
            borrow_quantity_exceeded=facts.borrow_quantity_exceeded,
            data_quality_hard_fail=facts.data_quality_hard_fail,
            regime_dislocated=facts.regime_dislocated,
            daily_loss_limit_breached=facts.daily_loss_limit_breached,
            stop_order_submission_failed=facts.stop_order_submission_failed,
            abnormal_market_data=facts.abnormal_market_data,
            duplicate_short_order=facts.duplicate_short_order,
        )
        components = compute_confidence_components(partial, thresholds)
        score = compute_confidence_score(components)
        # Rebuilt rather than mutated: the snapshot is frozen, and the score is a
        # function OF the snapshot, so it has to be computed then folded back in.
        return DirectionalValidationSnapshot(
            **{
                **{
                    field_name: getattr(partial, field_name)
                    for field_name in DirectionalValidationSnapshot.__dataclass_fields__
                    if field_name not in {"confidence_score", "confidence_components"}
                },
                "confidence_score": score,
                "confidence_components": components,
            }
        )

    def _holdout_result(
        self,
        key: DirectionalStrategyKey,
        now: datetime,
        thresholds: PromotionThresholds,
    ) -> tuple[int, int]:
        """How many recent contiguous holdout windows the arm passed.

        The windows are the most recent ``required_holdout_windows_passed`` blocks of
        ``holdout_window_days``, walked backwards from now. A window PASSES if its
        own mean net return is positive and it contains at least one scored outcome.

        Two properties make this a real out-of-sample test rather than a restatement
        of the pooled mean:

        * windows are DISJOINT, so a single spectacular week cannot carry several of
          them;
        * a window with no data does not pass. Silence is not evidence.
        """
        required = max(0, thresholds.required_holdout_windows_passed)
        if required <= 0:
            return 0, 0
        window = timedelta(days=self.config.holdout_window_days)
        passed = 0
        evaluated = 0
        for index in range(required):
            until = now - window * index
            since = until - window
            outcomes = self.shadow_store.outcomes(
                key, since=since, until=until, scored_only=True, limit=2000
            )
            nets = [
                float(item.get("net_return_bps"))
                for item in outcomes
                if item.get("net_return_bps") is not None
            ]
            evaluated += 1
            if nets and (sum(nets) / len(nets)) > 0.0:
                passed += 1
        return passed, evaluated

    def _cost_coverage_ratio(
        self, key: DirectionalStrategyKey, sources: Sequence[str]
    ) -> float | None:
        """Realized gross return as a multiple of realized all-in cost.

        Computed from the shadow journal rather than from predictions, so it answers
        "did the edge actually cover the cost" instead of "did we expect it to".
        """
        outcomes = self.shadow_store.outcomes(key, scored_only=True, limit=2000)
        gross = 0.0
        cost = 0.0
        for item in outcomes:
            value = item.get("gross_return_bps")
            if value is None:
                continue
            gross += float(value)
            cost += float(item.get("trading_cost_bps") or 0.0) + float(
                item.get("borrow_cost_bps") or 0.0
            )
        if cost <= 0:
            return None
        return gross / cost

    def _calibration_error(
        self, key: DirectionalStrategyKey, sources: Sequence[str]
    ) -> float | None:
        """Mean absolute prediction error, normalised into a [0, 1]-ish scale.

        The stored ``slippage_error_bps`` is (realized net - predicted net). Divided
        by a 100bps reference so it lands on roughly the same scale as the Brier /
        ECE figures the threshold was written against, and clamped so one wild
        outcome cannot make the whole metric meaningless.
        """
        outcomes = self.shadow_store.outcomes(key, scored_only=True, limit=2000)
        errors = [
            abs(float(item["slippage_error_bps"]))
            for item in outcomes
            if item.get("slippage_error_bps") is not None
        ]
        if not errors:
            return None
        return _clamp((sum(errors) / len(errors)) / 100.0)

    def status(self, markets: Sequence[str] = ("KR",)) -> dict[str, Any]:
        """Dashboard payload: state, confidence, and what each arm still needs."""
        arms: list[dict[str, Any]] = []
        for key in self.managed_keys(markets):
            record = self.state_store.get(key)
            if record is None:
                arms.append(
                    {
                        **key.as_dict(),
                        "state": str(self.config.initial_state_for(key.strategy_id)),
                        "submits_orders": False,
                        "confidence_score": 0.0,
                        "note": "NOT_YET_EVALUATED",
                    }
                )
                continue
            thresholds = self.config.thresholds_for(record.state)
            payload = record.as_dict()
            payload["next_state"] = (
                str(next_promotion_state(record.state) or "")
            )
            payload["required_consecutive_cycles"] = (
                thresholds.required_consecutive_evaluation_cycles if thresholds else None
            )
            payload["remaining_conditions"] = list(record.last_reason_codes)
            arms.append(payload)
        return {
            "enabled": self.config.enabled,
            "evaluation_interval_seconds": self.config.evaluation_interval_seconds,
            "last_evaluated_at": (
                self._last_evaluated_at.isoformat() if self._last_evaluated_at else None
            ),
            "config_version": self.config.config_version,
            "arms": arms,
            "borrow_health": self.borrow_store.health(),
        }


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def _json_or(value: Any, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:  # noqa: BLE001 - a malformed config must not enable anything.
        logger.warning("failed to load %s; short strategies stay disabled", path)
        return {}


_DEFAULT_CONTROLLER: ShortStrategyPromotionController | None = None
_DEFAULT_CONTROLLER_LOCK = threading.Lock()


def default_promotion_controller() -> ShortStrategyPromotionController:
    global _DEFAULT_CONTROLLER
    if _DEFAULT_CONTROLLER is None:
        with _DEFAULT_CONTROLLER_LOCK:
            if _DEFAULT_CONTROLLER is None:
                _DEFAULT_CONTROLLER = ShortStrategyPromotionController()
    return _DEFAULT_CONTROLLER


def reset_default_promotion_controller() -> None:
    global _DEFAULT_CONTROLLER
    with _DEFAULT_CONTROLLER_LOCK:
        _DEFAULT_CONTROLLER = None
