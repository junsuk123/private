"""FinalTradeGate — the last authority before an order intent becomes an order.

Fail-closed, and specifically about *unknowns*
-----------------------------------------------
Hard gates are not "bad conditions". They are conditions under which the system **does not
know** something an order depends on: how old the price is, whether the session is open,
whether the account balance is real, whether an identical order is already live, whether
the risk engine even ran. Every one of them defaults to blocking when its input is
missing, so a caller who forgets to supply account state gets a refusal rather than an
approval computed from nothing.

The hard set is fixed in code and has no thresholds. A tunable that could switch off
``ACCOUNT_RECONCILIATION_FAIL`` would make the gate advisory.

Soft gates size down rather than refuse
----------------------------------------
Wide spreads, thin liquidity, a conflicting global tape: real reasons to take less risk,
not reasons the system is blind. Each contributes a multiplier and they **compound**, so
two mild problems produce a smaller position than either alone. Once the compounded
multiplier falls below ``block_below`` the trade is refused — a position too small to
clear its own round-trip cost is not a smaller version of the trade, it is a losing one.

Model authority
---------------
::

    size = base * model_confidence * regime_factor * liquidity_factor * risk_factor

and then the exposure limits apply. The limits are a **ceiling the model cannot raise**:
``min(model_requested, policy_permitted)``, never a blend, never a scale. A model that
becomes confident cannot enlarge ``max_position_per_stock``; the most it can do is use
the room that already exists.

Exits are not gated the same way
--------------------------------
:meth:`FinalTradeGate.evaluate` is for **new exposure**. Reducing or closing a position
runs through :meth:`evaluate_exit`, which enforces only the gates that would make an exit
*unsafe to route* (unknown session, unknown order state, duplicate risk) and never blocks
on staleness, model health or exposure. Being unable to close a position because the feed
went stale is the failure this asymmetry exists to prevent.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "FinalTradeGate",
    "GateConfig",
    "GateDecision",
    "GateInputs",
    "HARD_GATES",
    "SOFT_GATES",
    "default_gate_config",
    "load_gate_config",
    "reset_gate_config_cache",
]

DEFAULT_CONFIG_PATH = Path("config/final_trade_gate.yaml")

#: Fixed. Not configurable, not maskable, not overridable by any model output.
HARD_GATES: tuple[str, ...] = (
    "STALE_DATA",
    "WS_DISCONNECTED",
    "PRICE_FEED_CONFLICT",
    "UNKNOWN_SESSION",
    "TRADING_HALT",
    "ACCOUNT_RECONCILIATION_FAIL",
    "UNKNOWN_ORDER_STATE",
    "DUPLICATE_ORDER_RISK",
    "MODEL_INFERENCE_FAIL",
    "RISK_ENGINE_FAIL",
)

SOFT_GATES: tuple[str, ...] = (
    "HIGH_VOLATILITY",
    "LOW_LIQUIDITY",
    "GLOBAL_CONFLICT",
    "SECTOR_CONFLICT",
    "LOW_MODEL_CONFIDENCE",
    "OPENING_EXTREME_VOL",
    "ABNORMAL_SPREAD",
)

#: Hard gates that also apply when reducing or closing a position. Everything else is
#: deliberately absent: an exit must remain possible when the data is stale, the model is
#: down or the account is at its exposure limit.
EXIT_HARD_GATES: frozenset[str] = frozenset(
    {"UNKNOWN_SESSION", "UNKNOWN_ORDER_STATE", "DUPLICATE_ORDER_RISK", "TRADING_HALT"}
)

#: Limit breaches. Reported separately from hard gates because their cause is different —
#: the system knows exactly what is going on and is declining on policy grounds.
LIMIT_CODES: tuple[str, ...] = (
    "MAX_POSITION_PER_STOCK",
    "MAX_SECTOR_EXPOSURE",
    "MAX_MARKET_EXPOSURE",
    "MAX_DAILY_LOSS",
    "MAX_DRAWDOWN",
    "POSITION_BELOW_MINIMUM",
)


class GateConfigError(RuntimeError):
    """The gate policy exists but does not parse."""


@dataclass(frozen=True)
class SoftGateRule:
    name: str
    threshold: float
    multiplier: float


@dataclass(frozen=True)
class ExposureLimits:
    max_position_per_stock: float = 0.10
    max_sector_exposure: float = 0.30
    max_market_exposure: float = 0.80
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10


@dataclass(frozen=True)
class SizingPolicy:
    base_position_fraction: float = 0.05
    block_below: float = 0.15
    model_confidence_floor: float = 0.20
    model_confidence_ceiling: float = 1.0


@dataclass(frozen=True)
class GateConfig:
    sizing: SizingPolicy = field(default_factory=SizingPolicy)
    regime_factors: Mapping[str, float] = field(default_factory=dict)
    soft_gates: Mapping[str, SoftGateRule] = field(default_factory=dict)
    limits: ExposureLimits = field(default_factory=ExposureLimits)
    model_health_factors: Mapping[str, float] = field(
        default_factory=lambda: {"HEALTHY": 1.0, "DEGRADED": 0.5}
    )
    source_path: str | None = None


@dataclass(frozen=True)
class GateInputs:
    """Everything the gate reads. Absent critical fields BLOCK; they never default open.

    The distinction that matters: ``None`` means "not supplied". For every hard gate that
    is the blocking value, because the gate's question is "do we know this", and the
    answer to an unsupplied field is no.
    """

    ticker: str
    side: str
    evaluated_at: datetime

    # -- hard-gate inputs ------------------------------------------------ #
    #: Reason codes from the freshness registry. Non-empty blocks.
    stale_data_reasons: Sequence[str] = ()
    #: ``None`` (unknown) or ``False`` both block.
    websocket_connected: bool | None = None
    #: Disagreement between independent price sources, in bps. ``None`` blocks only when
    #: ``require_price_cross_check`` is set.
    price_feed_divergence_bps: float | None = None
    require_price_cross_check: bool = False
    max_price_feed_divergence_bps: float = 50.0
    #: Resolved session identifier; ``None`` / UNKNOWN / CLOSED block a new entry.
    session_id: str | None = None
    session_allows_new_entry: bool | None = None
    trading_halted: bool | None = None
    account_reconciled: bool | None = None
    #: Orders whose state the system cannot currently determine.
    unknown_order_ids: Sequence[str] = ()
    #: True when an equivalent order is already live or an idempotency key is in flight.
    duplicate_order_risk: bool | None = None
    #: GNN / model runtime health. ``OFFLINE`` or ``None`` blocks a new entry.
    model_health_state: str | None = None
    #: Set by the caller when the risk engine itself failed to produce a verdict.
    risk_engine_ok: bool | None = None

    # -- soft-gate inputs -------------------------------------------------- #
    realized_volatility: float | None = None
    liquidity_score: float | None = None
    global_agreement: float | None = None
    sector_relative_strength: float | None = None
    model_confidence: float | None = None
    session_phase: str | None = None
    opening_volatility_multiple: float | None = None
    spread_bps: float | None = None

    # -- sizing inputs ------------------------------------------------------ #
    dominant_regime: str | None = None
    account_equity: float | None = None
    current_position_value: float = 0.0
    current_sector_exposure: float = 0.0
    current_market_exposure: float = 0.0
    session_pnl_ratio: float | None = None
    drawdown_ratio: float | None = None
    #: Fraction of equity the strategy asked for, before any gate applies.
    requested_position_fraction: float | None = None

    decision_id: str | None = None
    sector: str | None = None


@dataclass(frozen=True)
class GateDecision:
    """The verdict, with every term that produced it."""

    approved: bool
    ticker: str
    side: str
    evaluated_at: datetime
    hard_failures: tuple[str, ...] = ()
    soft_failures: tuple[str, ...] = ()
    limit_failures: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    position_multiplier: float = 0.0
    approved_position_fraction: float = 0.0
    factors: Mapping[str, float] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)
    decision_id: str | None = None
    gate_id: str = ""

    @property
    def blocked(self) -> bool:
        return not self.approved

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "decision_id": self.decision_id,
            "approved": self.approved,
            "ticker": self.ticker,
            "side": self.side,
            "evaluated_at": self.evaluated_at.isoformat(),
            "hard_failures": list(self.hard_failures),
            "soft_failures": list(self.soft_failures),
            "limit_failures": list(self.limit_failures),
            "reasons": list(self.reasons),
            "position_multiplier": self.position_multiplier,
            "approved_position_fraction": self.approved_position_fraction,
            "factors": dict(self.factors),
            "detail": dict(self.detail),
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_gate_config(path: str | Path = DEFAULT_CONFIG_PATH) -> GateConfig:
    target = Path(path)
    if not target.exists():
        return GateConfig(
            regime_factors={},
            soft_gates={
                name: SoftGateRule(name, threshold, multiplier)
                for name, threshold, multiplier in (
                    ("HIGH_VOLATILITY", 0.005, 0.60),
                    ("LOW_LIQUIDITY", 0.35, 0.50),
                    ("GLOBAL_CONFLICT", -0.25, 0.70),
                    ("SECTOR_CONFLICT", 0.0, 0.75),
                    ("LOW_MODEL_CONFIDENCE", 0.45, 0.60),
                    ("OPENING_EXTREME_VOL", 2.5, 0.50),
                    ("ABNORMAL_SPREAD", 25.0, 0.55),
                )
            },
        )
    try:
        import yaml

        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - a malformed gate policy is fatal.
        raise GateConfigError(f"cannot parse {target}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise GateConfigError(f"{target} must be a mapping")

    sizing_raw = raw.get("sizing") or {}
    sizing = SizingPolicy(
        base_position_fraction=float(sizing_raw.get("base_position_fraction", 0.05)),
        block_below=float(sizing_raw.get("block_below", 0.15)),
        model_confidence_floor=float(sizing_raw.get("model_confidence_floor", 0.20)),
        model_confidence_ceiling=float(sizing_raw.get("model_confidence_ceiling", 1.0)),
    )
    soft: dict[str, SoftGateRule] = {}
    for name, entry in (raw.get("soft_gates") or {}).items():
        if str(name) not in SOFT_GATES:
            raise GateConfigError(f"{target}: unknown soft gate {name!r}")
        soft[str(name)] = SoftGateRule(
            name=str(name),
            threshold=float(entry.get("threshold", 0.0)),
            multiplier=float(entry.get("multiplier", 1.0)),
        )
    limits_raw = raw.get("limits") or {}
    limits = ExposureLimits(
        max_position_per_stock=float(limits_raw.get("max_position_per_stock", 0.10)),
        max_sector_exposure=float(limits_raw.get("max_sector_exposure", 0.30)),
        max_market_exposure=float(limits_raw.get("max_market_exposure", 0.80)),
        max_daily_loss=float(limits_raw.get("max_daily_loss", 0.02)),
        max_drawdown=float(limits_raw.get("max_drawdown", 0.10)),
    )
    return GateConfig(
        sizing=sizing,
        regime_factors={
            str(name): float(value) for name, value in (raw.get("regime_factors") or {}).items()
        },
        soft_gates=soft,
        limits=limits,
        model_health_factors={
            str(name): float(value)
            for name, value in (raw.get("model_health") or {"HEALTHY": 1.0, "DEGRADED": 0.5}).items()
        },
        source_path=str(target),
    )


_config_cache: GateConfig | None = None
_config_lock = threading.Lock()


def default_gate_config() -> GateConfig:
    global _config_cache
    with _config_lock:
        if _config_cache is None:
            _config_cache = load_gate_config()
        return _config_cache


def reset_gate_config_cache() -> None:
    """Test hook. Never called from the trading path."""
    global _config_cache
    with _config_lock:
        _config_cache = None


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
class FinalTradeGate:
    """Evaluates the hard gates, the soft gates and the exposure limits, in that order."""

    def __init__(self, config: GateConfig | None = None) -> None:
        self._config = config or default_gate_config()

    @property
    def config(self) -> GateConfig:
        return self._config

    @property
    def limits(self) -> ExposureLimits:
        return self._config.limits

    # ------------------------------------------------------------------ #
    def evaluate(self, inputs: GateInputs) -> GateDecision:
        """Verdict for opening or increasing exposure.

        Never raises. An unexpected failure inside the gate becomes ``RISK_ENGINE_FAIL``
        and a refusal, because a gate that can crash is a gate that can be bypassed by
        crashing it.
        """
        moment = _aware(inputs.evaluated_at)
        try:
            return self._evaluate(inputs, moment)
        except Exception as exc:  # noqa: BLE001 - fail-closed is the entire point.
            return GateDecision(
                approved=False,
                ticker=inputs.ticker,
                side=inputs.side,
                evaluated_at=moment,
                hard_failures=("RISK_ENGINE_FAIL",),
                reasons=(f"RISK_ENGINE_FAIL:{type(exc).__name__}: {exc}",),
                decision_id=inputs.decision_id,
                gate_id=_gate_id(inputs.ticker, moment),
            )

    def evaluate_exit(self, inputs: GateInputs) -> GateDecision:
        """Verdict for reducing or closing a position.

        Only the gates that make an exit unroutable apply. Staleness, model health and
        exposure limits do not: a stale feed is the moment a position most needs closing,
        and being over an exposure limit is a reason to reduce rather than a reason to be
        unable to.
        """
        moment = _aware(inputs.evaluated_at)
        try:
            hard = tuple(
                code for code in self._hard_failures(inputs) if code in EXIT_HARD_GATES
            )
            return GateDecision(
                approved=not hard,
                ticker=inputs.ticker,
                side=inputs.side,
                evaluated_at=moment,
                hard_failures=hard,
                reasons=tuple(f"EXIT:{code}" for code in hard),
                position_multiplier=1.0 if not hard else 0.0,
                approved_position_fraction=0.0,
                detail={"mode": "exit", "evaluated_gates": sorted(EXIT_HARD_GATES)},
                decision_id=inputs.decision_id,
                gate_id=_gate_id(inputs.ticker, moment),
            )
        except Exception as exc:  # noqa: BLE001
            return GateDecision(
                approved=False,
                ticker=inputs.ticker,
                side=inputs.side,
                evaluated_at=moment,
                hard_failures=("RISK_ENGINE_FAIL",),
                reasons=(f"RISK_ENGINE_FAIL:{type(exc).__name__}: {exc}",),
                decision_id=inputs.decision_id,
                gate_id=_gate_id(inputs.ticker, moment),
            )

    # ------------------------------------------------------------------ #
    def _evaluate(self, inputs: GateInputs, moment: datetime) -> GateDecision:
        gate_id = _gate_id(inputs.ticker, moment)
        hard = self._hard_failures(inputs)
        if hard:
            return GateDecision(
                approved=False,
                ticker=inputs.ticker,
                side=inputs.side,
                evaluated_at=moment,
                hard_failures=hard,
                reasons=tuple(f"HARD:{code}" for code in hard),
                detail={"stale_data_reasons": list(inputs.stale_data_reasons)},
                decision_id=inputs.decision_id,
                gate_id=gate_id,
            )

        soft, soft_factor, soft_detail = self._soft_failures(inputs)
        factors = self._sizing_factors(inputs, soft_factor)
        multiplier = 1.0
        for value in factors.values():
            multiplier *= value
        multiplier = round(max(0.0, min(1.0, multiplier)), 6)

        limits, approved_fraction, limit_detail = self._apply_limits(inputs, multiplier)
        approved = not limits
        if multiplier < self._config.sizing.block_below:
            limits = (*limits, "POSITION_BELOW_MINIMUM")
            approved = False

        reasons = tuple(
            [*(f"SOFT:{code}" for code in soft), *(f"LIMIT:{code}" for code in limits)]
        )
        return GateDecision(
            approved=approved,
            ticker=inputs.ticker,
            side=inputs.side,
            evaluated_at=moment,
            soft_failures=soft,
            limit_failures=limits,
            reasons=reasons,
            position_multiplier=multiplier,
            approved_position_fraction=round(approved_fraction, 8) if approved else 0.0,
            factors=factors,
            detail={**soft_detail, **limit_detail},
            decision_id=inputs.decision_id,
            gate_id=gate_id,
        )

    # ------------------------------------------------------------------ #
    def _hard_failures(self, inputs: GateInputs) -> tuple[str, ...]:
        failures: list[str] = []

        if inputs.stale_data_reasons:
            failures.append("STALE_DATA")
        if inputs.websocket_connected is not True:
            failures.append("WS_DISCONNECTED")

        divergence = _finite(inputs.price_feed_divergence_bps)
        if divergence is None:
            if inputs.require_price_cross_check:
                failures.append("PRICE_FEED_CONFLICT")
        elif abs(divergence) > inputs.max_price_feed_divergence_bps:
            failures.append("PRICE_FEED_CONFLICT")

        session = str(inputs.session_id or "").strip().upper()
        if (
            not session
            or session in {"UNKNOWN", "KR_CLOSED", "US_CLOSED", "CLOSED"}
            or inputs.session_allows_new_entry is not True
        ):
            failures.append("UNKNOWN_SESSION")

        if inputs.trading_halted is not False:
            failures.append("TRADING_HALT")
        if inputs.account_reconciled is not True:
            failures.append("ACCOUNT_RECONCILIATION_FAIL")
        if inputs.unknown_order_ids:
            failures.append("UNKNOWN_ORDER_STATE")
        if inputs.duplicate_order_risk is not False:
            failures.append("DUPLICATE_ORDER_RISK")

        health = str(inputs.model_health_state or "").strip().upper()
        if health not in {"HEALTHY", "DEGRADED"}:
            failures.append("MODEL_INFERENCE_FAIL")
        if inputs.risk_engine_ok is not True:
            failures.append("RISK_ENGINE_FAIL")

        return tuple(dict.fromkeys(failures))

    def _soft_failures(
        self, inputs: GateInputs
    ) -> tuple[tuple[str, ...], float, dict[str, Any]]:
        rules = self._config.soft_gates
        triggered: list[str] = []
        factor = 1.0
        detail: dict[str, Any] = {}

        def trigger(name: str) -> None:
            nonlocal factor
            rule = rules.get(name)
            if rule is None:
                return
            triggered.append(name)
            factor *= rule.multiplier
            detail[f"soft_{name.lower()}_multiplier"] = rule.multiplier

        volatility = _finite(inputs.realized_volatility)
        if volatility is not None and "HIGH_VOLATILITY" in rules:
            if volatility > rules["HIGH_VOLATILITY"].threshold:
                trigger("HIGH_VOLATILITY")

        liquidity = _finite(inputs.liquidity_score)
        if liquidity is not None and "LOW_LIQUIDITY" in rules:
            if liquidity < rules["LOW_LIQUIDITY"].threshold:
                trigger("LOW_LIQUIDITY")

        agreement = _finite(inputs.global_agreement)
        if agreement is not None and "GLOBAL_CONFLICT" in rules:
            if agreement < rules["GLOBAL_CONFLICT"].threshold:
                trigger("GLOBAL_CONFLICT")

        # Sector conflict is side-dependent: a negative sector RS conflicts with a BUY and
        # supports a SELL. Evaluating it without the side is how a short gets penalised
        # for the very condition that justifies it.
        sector_rs = _finite(inputs.sector_relative_strength)
        if sector_rs is not None and "SECTOR_CONFLICT" in rules:
            threshold = rules["SECTOR_CONFLICT"].threshold
            side = str(inputs.side or "").strip().upper()
            conflicting = (
                sector_rs < threshold if side in {"BUY", "LONG"} else sector_rs > threshold
            )
            if conflicting:
                trigger("SECTOR_CONFLICT")

        confidence = _finite(inputs.model_confidence)
        if confidence is not None and "LOW_MODEL_CONFIDENCE" in rules:
            if confidence < rules["LOW_MODEL_CONFIDENCE"].threshold:
                trigger("LOW_MODEL_CONFIDENCE")

        phase = str(inputs.session_phase or "").strip().upper()
        opening_multiple = _finite(inputs.opening_volatility_multiple)
        if (
            phase in {"OPEN_TRANSITION", "OPENING"}
            and opening_multiple is not None
            and "OPENING_EXTREME_VOL" in rules
            and opening_multiple > rules["OPENING_EXTREME_VOL"].threshold
        ):
            trigger("OPENING_EXTREME_VOL")

        spread = _finite(inputs.spread_bps)
        if spread is not None and "ABNORMAL_SPREAD" in rules:
            if spread > rules["ABNORMAL_SPREAD"].threshold:
                trigger("ABNORMAL_SPREAD")

        return tuple(dict.fromkeys(triggered)), factor, detail

    def _sizing_factors(self, inputs: GateInputs, soft_factor: float) -> dict[str, float]:
        sizing = self._config.sizing
        confidence = _finite(inputs.model_confidence)
        if confidence is None:
            # No confidence supplied is not full confidence. The floor is the honest
            # reading: the trade may proceed, at the smallest size the policy allows.
            confidence = sizing.model_confidence_floor
        confidence = max(
            sizing.model_confidence_floor,
            min(sizing.model_confidence_ceiling, confidence),
        )
        regime_factor = float(
            self._config.regime_factors.get(str(inputs.dominant_regime or ""), 1.0)
        )
        liquidity = _finite(inputs.liquidity_score)
        liquidity_factor = 1.0 if liquidity is None else max(0.0, min(1.0, liquidity))
        health_factor = float(
            self._config.model_health_factors.get(
                str(inputs.model_health_state or "").strip().upper(), 1.0
            )
        )
        return {
            "model_confidence": round(confidence, 6),
            "regime_factor": round(regime_factor, 6),
            "liquidity_factor": round(liquidity_factor, 6),
            "risk_factor": round(soft_factor, 6),
            "model_health_factor": round(health_factor, 6),
        }

    def _apply_limits(
        self, inputs: GateInputs, multiplier: float
    ) -> tuple[tuple[str, ...], float, dict[str, Any]]:
        limits = self._config.limits
        failures: list[str] = []
        detail: dict[str, Any] = {}

        equity = _finite(inputs.account_equity)
        if equity is None or equity <= 0.0:
            # Without equity no exposure fraction can be computed, so no new exposure can
            # be authorised. This is a limit failure rather than a hard gate: the system
            # knows what is wrong, it simply cannot size.
            return ("MAX_POSITION_PER_STOCK",), 0.0, {"reason": "no_account_equity"}

        requested = _finite(inputs.requested_position_fraction)
        base = self._config.sizing.base_position_fraction
        desired = (requested if requested is not None else base) * multiplier
        detail["requested_position_fraction"] = requested
        detail["desired_position_fraction"] = round(desired, 8)

        # The limits are ceilings the model cannot raise: min(), never a blend.
        position_headroom = limits.max_position_per_stock - (
            max(0.0, inputs.current_position_value) / equity
        )
        sector_headroom = limits.max_sector_exposure - (
            max(0.0, inputs.current_sector_exposure) / equity
        )
        market_headroom = limits.max_market_exposure - (
            max(0.0, inputs.current_market_exposure) / equity
        )
        detail["position_headroom"] = round(position_headroom, 8)
        detail["sector_headroom"] = round(sector_headroom, 8)
        detail["market_headroom"] = round(market_headroom, 8)

        if position_headroom <= 0.0:
            failures.append("MAX_POSITION_PER_STOCK")
        if sector_headroom <= 0.0:
            failures.append("MAX_SECTOR_EXPOSURE")
        if market_headroom <= 0.0:
            failures.append("MAX_MARKET_EXPOSURE")

        pnl = _finite(inputs.session_pnl_ratio)
        if pnl is not None and pnl <= -abs(limits.max_daily_loss):
            failures.append("MAX_DAILY_LOSS")
        drawdown = _finite(inputs.drawdown_ratio)
        if drawdown is not None and abs(drawdown) >= abs(limits.max_drawdown):
            failures.append("MAX_DRAWDOWN")

        allowed = min(
            desired,
            max(0.0, position_headroom),
            max(0.0, sector_headroom),
            max(0.0, market_headroom),
        )
        detail["approved_position_fraction"] = round(allowed, 8)
        return tuple(dict.fromkeys(failures)), allowed, detail


def _gate_id(ticker: str, moment: datetime) -> str:
    from uuid import uuid4

    slug = "".join(ch for ch in str(ticker).upper() if ch.isalnum())[:12] or "NA"
    return f"gate-{slug}-{moment.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:6]}"
