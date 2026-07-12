"""Single, versioned view of the live trading policy.

The TP / SL / horizon / net-edge / sizing numbers that govern live trading are
resolved from environment variables in several independent places (the dynamic
exit policy, the profitability gate, the position sizer, the label builder). This
module reads those SAME environment variables once and freezes them into an
immutable ``TradingPolicySnapshot`` with a deterministic ``policy_version`` hash.

It is intentionally read-only and additive: it does not change how any subsystem
resolves its own values. Its job is to give the readiness checker, the startup
log and the audit trail one object to stamp and to detect contradictory settings
(e.g. a stop-loss that is disabled in every direction) before live arming.

The label side (``LIVE_LABEL_*``) is captured too so the readiness report can flag
when the model was trained against a TP/SL/horizon that no longer matches the live
exit policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field


def _f(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _confidence_threshold(default: float = 0.5) -> float:
    """Live minimum probability-of-success floor, from the safety config if present."""
    try:  # local import to avoid a config import cycle at module load
        from app.config.live_config import load_live_trading_safety_config

        return float(load_live_trading_safety_config().minimum_probability_success)
    except Exception:  # noqa: BLE001 - readiness runs even when the config is absent
        return float(default)


@dataclass(frozen=True)
class PolicyConflict:
    code: str
    severity: str  # "FAIL" | "WARNING"
    message: str


@dataclass(frozen=True)
class TradingPolicySnapshot:
    """Immutable resolved view of the live trading policy + a version hash."""

    # Exit / risk (net-of-cost rates unless noted).
    take_profit_net_rate: float
    quick_take_profit_net_rate: float
    stop_loss_net_rate: float
    hard_stop_rate: float
    emergency_stop_rate: float
    allow_loss_exit: bool
    profit_time_exit_seconds: float

    # Entry economics.
    minimum_net_edge_kr: float
    minimum_net_edge_us: float

    # Sizing.
    maximum_position_weight: float

    # Model gate.
    confidence_threshold: float

    # Label policy the live_short_horizon model is trained against (bps / seconds).
    label_take_profit_bps: float
    label_stop_loss_bps: float
    label_horizon_seconds: float

    source: str = "environment"
    policy_version: str = field(default="", compare=False)

    @classmethod
    def from_environment(cls) -> "TradingPolicySnapshot":
        snap = cls(
            take_profit_net_rate=_f("REALTIME_TAKE_PROFIT", 0.0025),
            quick_take_profit_net_rate=_f("REALTIME_QUICK_TAKE_PROFIT_NET", 0.008),
            stop_loss_net_rate=_f("REALTIME_STOP_LOSS_NET", 0.0),
            hard_stop_rate=_f("REALTIME_HARD_STOP_LOSS", 0.08),
            emergency_stop_rate=_f("REALTIME_EMERGENCY_STOP_LOSS", 0.05),
            allow_loss_exit=_b("REALTIME_ALLOW_LOSS_EXIT", False),
            profit_time_exit_seconds=_f("REALTIME_PROFIT_TIME_EXIT_SEC", 300.0),
            minimum_net_edge_kr=_f("REALTIME_MIN_BUY_NET_RETURN_KR", 0.008),
            minimum_net_edge_us=_f("REALTIME_MIN_BUY_NET_RETURN_US", 0.012),
            maximum_position_weight=_f("REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT", 0.10),
            confidence_threshold=_confidence_threshold(),
            label_take_profit_bps=_f("LIVE_LABEL_TAKE_PROFIT_BPS", 25.0),
            label_stop_loss_bps=_f("LIVE_LABEL_STOP_LOSS_BPS", 100.0),
            label_horizon_seconds=_f("LIVE_LABEL_HORIZON_SECONDS", 600.0),
        )
        return snap.with_version()

    def with_version(self) -> "TradingPolicySnapshot":
        payload = {k: v for k, v in asdict(self).items() if k != "policy_version"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        return TradingPolicySnapshot(**{**payload, "policy_version": f"tp_{digest}"})

    def conflicts(self) -> list[PolicyConflict]:
        """Detect contradictory / unsafe policy combinations."""
        out: list[PolicyConflict] = []

        # A stop that is disabled in every direction leaves positions to bleed to the
        # hard/emergency capital backstop with no routine stop-out.
        if not self.allow_loss_exit and self.stop_loss_net_rate <= 0.0:
            out.append(
                PolicyConflict(
                    "STOP_LOSS_DISABLED",
                    "FAIL",
                    "REALTIME_ALLOW_LOSS_EXIT=false and REALTIME_STOP_LOSS_NET=0.0 disable "
                    "every routine stop-loss; only the hard/emergency backstop remains.",
                )
            )

        # Stop ordering must be routine <= hard <= emergency.
        if self.stop_loss_net_rate > 0.0 and self.hard_stop_rate > 0.0 and self.hard_stop_rate <= self.stop_loss_net_rate:
            out.append(
                PolicyConflict(
                    "HARD_STOP_NOT_WIDER_THAN_STOP",
                    "FAIL",
                    "REALTIME_HARD_STOP_LOSS must be wider than REALTIME_STOP_LOSS_NET.",
                )
            )
        if self.emergency_stop_rate > 0.0 and self.hard_stop_rate > 0.0 and self.emergency_stop_rate < self.hard_stop_rate:
            out.append(
                PolicyConflict(
                    "EMERGENCY_STOP_NOT_WIDER_THAN_HARD",
                    "WARNING",
                    "REALTIME_EMERGENCY_STOP_LOSS is tighter than REALTIME_HARD_STOP_LOSS.",
                )
            )

        if self.take_profit_net_rate <= 0.0 and self.quick_take_profit_net_rate <= 0.0:
            out.append(
                PolicyConflict("TAKE_PROFIT_DISABLED", "FAIL", "No positive take-profit target is configured.")
            )

        # Small-account sizing above 100% is a deliberate deviation (a single minimum-lot
        # order can exceed the per-name budget) — flagged as a warning, not a hard failure.
        if self.maximum_position_weight > 1.0:
            out.append(
                PolicyConflict(
                    "POSITION_WEIGHT_ABOVE_ONE",
                    "WARNING",
                    f"maximum_position_weight={self.maximum_position_weight} exceeds 1.0 "
                    "(intentional for small-account single-lot affordability; RiskManager "
                    "still clamps per-name exposure).",
                )
            )

        return out

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["conflicts"] = [asdict(c) for c in self.conflicts()]
        return d


def load_trading_policy_snapshot() -> TradingPolicySnapshot:
    return TradingPolicySnapshot.from_environment()
