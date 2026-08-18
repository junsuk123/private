"""Where cost, size and risk are decided — before the strategy is elected.

The move this module performs
-----------------------------
``ProfitabilityGate``, ``PositionSizer`` and ``RiskManager`` used to run *after* a strategy
had been elected, in ``SharedLiveDecisionEngine.evaluate_buy``. Each could veto or resize
what the election had already committed to, which meant the election's own numbers were
provisional and three more authorities got a say on the same question.

The calculations are not deleted — they are **the same calculations, run earlier**. They
now produce the :class:`~app.trading.trade_plan.TradePlan` instead of judging one:

1. :class:`~app.cost.ProfitabilityGate` computes the all-in cost and the net edge. A
   candidate whose net edge does not clear its cost never becomes a plan, so there is
   nothing downstream to veto.
2. :class:`~app.risk.position_sizing.PositionSizer` turns that net edge, the confidence
   and the liquidity into a position weight.
3. :class:`~app.risk.manager.RiskManager` validates the resulting intent against the real
   account — exposure, concentration, daily loss, instrument eligibility — and its
   approved ``FinalOrder.quantity`` becomes the plan's quantity.

The result is one number for the size and one verdict on the risk, both computed once,
both frozen into the plan, and both replayable from ``cost_snapshot`` / ``risk_snapshot``.

What comes out
--------------
Exactly one of:

* a :class:`TradePlan` that is executable as written, or
* :class:`NoTradeDecision` with the reason codes that stopped it.

``NO_TRADE`` is a first-class outcome and carries the same provenance a plan does, because
"why did nothing trade" has to be answerable from stored evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from app.cost import ProfitabilityGate, ProfitabilityInput, TradingCostEngine
from app.risk.manager import RiskManager
from app.risk.position_sizing import PositionSizer, SizingInputs
from app.schemas.domain import (
    AccountSnapshot,
    MarketSnapshot,
    OrderAction,
    OrderIntent,
    RiskRules,
)
from app.trading.trade_plan import (
    DEFAULT_PLAN_TTL_SECONDS,
    EntryRule,
    ExitRules,
    TradePlan,
    TradePlanError,
    TradePlanStatus,
    new_plan_id,
)

__all__ = [
    "NoTradeDecision",
    "PlanRequest",
    "TradePlanBuilder",
    "TradePlanOutcome",
]

#: Fraction either side of the reference price the entry band spans. The election priced
#: the edge against ``reference_price``; a fill 40bps away is a materially different trade
#: from the one that was approved, and this is the envelope that says so.
ENTRY_BAND_RATE = 0.004

#: Confidence used when the caller supplies none. The sizing floor, not a neutral 0.5:
#: an unstated confidence must size small rather than average.
DEFAULT_CONFIDENCE = 0.35


@dataclass(frozen=True)
class PlanRequest:
    """Everything the builder needs. Nothing is fetched inside — the builder is pure."""

    symbol: str
    strategy_id: str
    market: str
    account: AccountSnapshot
    market_snapshot: MarketSnapshot
    reference_price: float
    #: Rates off the entry price, from the strategy's own exit geometry.
    take_profit_rate: float
    stop_loss_rate: float
    trailing_rate: float | None
    max_holding_seconds: int
    #: Gross expected move, in bps, as the election measured it.
    gross_edge_bps: float
    direction: str = "LONG"
    confidence: float | None = None
    liquidity_score: float = 1.0
    spread_bps: float | None = None
    realized_volatility: float | None = None
    orderbook_snapshot: Any = None
    account_drawdown_rate: float = 0.0
    recent_same_strategy_loss: bool = False
    #: Cap from the deployment ladder / selector authority. Applied to the elected size
    #: HERE, so nothing downstream has to re-clip it.
    authority_size_fraction: float = 1.0
    entry_trigger: str = "STRATEGY_ENTRY"
    strategy_exit_trigger: str | None = None
    cancel_rule: str = "PLAN_EXPIRY_OR_STRATEGY_INVALIDATION"
    weekday_time_context: Mapping[str, Any] = field(default_factory=dict)
    election_context: Mapping[str, Any] = field(default_factory=dict)
    order_contract: Mapping[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    decision_id: str | None = None
    session_id: str | None = None
    plan_ttl_seconds: float = DEFAULT_PLAN_TTL_SECONDS
    venue: str = ""
    instrument_type: str = "EQUITY"
    #: Weight ceiling from the caller's policy, before edge-aware sizing narrows it.
    max_position_weight: float = 0.05


@dataclass(frozen=True)
class NoTradeDecision:
    """Why no plan was produced. As traceable as a plan.

    ``NO_TRADE`` competes with the strategies rather than being the absence of one, so it
    carries the same cost and risk snapshots a plan would — a rejection whose numbers are
    unavailable cannot be argued with later.
    """

    symbol: str
    strategy_id: str
    decided_at: datetime
    reason_codes: tuple[str, ...]
    stage: str
    cost_snapshot: Mapping[str, Any] = field(default_factory=dict)
    risk_snapshot: Mapping[str, Any] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": "NO_TRADE",
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "decided_at": _aware(self.decided_at).isoformat(),
            "reason_codes": list(self.reason_codes),
            "stage": self.stage,
            "cost_snapshot": dict(self.cost_snapshot),
            "risk_snapshot": dict(self.risk_snapshot),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class TradePlanOutcome:
    """Exactly one of ``plan`` or ``no_trade`` is set."""

    plan: TradePlan | None = None
    no_trade: NoTradeDecision | None = None

    @property
    def tradable(self) -> bool:
        return self.plan is not None

    def as_dict(self) -> dict[str, Any]:
        if self.plan is not None:
            return {"decision": "TRADE", "plan": self.plan.as_dict()}
        return self.no_trade.as_dict() if self.no_trade else {"decision": "NO_TRADE"}


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class TradePlanBuilder:
    """Runs cost, sizing and risk once, before election, and emits a plan or NO_TRADE."""

    def __init__(
        self,
        *,
        cost_engine: TradingCostEngine | None = None,
        profitability_gate: ProfitabilityGate | None = None,
        position_sizer: PositionSizer | None = None,
        risk_manager: RiskManager | None = None,
        risk_rules: RiskRules | None = None,
    ) -> None:
        self.cost_engine = cost_engine or TradingCostEngine()
        self.profitability_gate = profitability_gate or ProfitabilityGate(
            cost_engine=self.cost_engine
        )
        self.position_sizer = position_sizer or PositionSizer()
        self.risk_manager = risk_manager or RiskManager(risk_rules or RiskRules())

    # ------------------------------------------------------------------ #
    def build(self, request: PlanRequest, *, now: datetime) -> TradePlanOutcome:
        """Cost -> size -> risk -> plan. Deterministic for a given request."""
        moment = _aware(now)
        price = _finite(request.reference_price)
        if price <= 0.0:
            return TradePlanOutcome(
                no_trade=NoTradeDecision(
                    symbol=request.symbol,
                    strategy_id=request.strategy_id,
                    decided_at=moment,
                    reason_codes=("NO_REFERENCE_PRICE",),
                    stage="input",
                )
            )

        # -- 1. cost and net edge ------------------------------------------ #
        decision = self._profitability(request, price)
        cost_snapshot = decision.as_dict()
        if not decision.allowed:
            return TradePlanOutcome(
                no_trade=NoTradeDecision(
                    symbol=request.symbol,
                    strategy_id=request.strategy_id,
                    decided_at=moment,
                    reason_codes=tuple(
                        (*decision.rejection_reasons, "PRE_ELECTION_NET_EDGE_INSUFFICIENT")
                    ),
                    stage="profitability",
                    cost_snapshot=cost_snapshot,
                )
            )

        # -- 2. size ---------------------------------------------------------- #
        confidence = (
            _finite(request.confidence, DEFAULT_CONFIDENCE)
            if request.confidence is not None
            else DEFAULT_CONFIDENCE
        )
        sizing = self.position_sizer.size(
            SizingInputs(
                net_expected_return=decision.net_expected_return,
                target_net_return=decision.required_min_net_return,
                confidence_score=confidence,
                liquidity_score=max(0.0, min(1.0, _finite(request.liquidity_score, 1.0))),
                account_drawdown_rate=_finite(request.account_drawdown_rate),
                recent_same_strategy_loss=bool(request.recent_same_strategy_loss),
            )
        )
        authority = max(0.0, min(1.0, _finite(request.authority_size_fraction, 1.0)))
        if authority <= 0.0:
            return TradePlanOutcome(
                no_trade=NoTradeDecision(
                    symbol=request.symbol,
                    strategy_id=request.strategy_id,
                    decided_at=moment,
                    reason_codes=("AUTHORITY_NOT_ORDERABLE",),
                    stage="sizing",
                    cost_snapshot=cost_snapshot,
                    detail={"sizing": sizing.as_dict()},
                )
            )
        # The deployment/selector cap is applied HERE, once. Applying it downstream (as
        # the engine used to) meant the elected size and the submitted size were two
        # different numbers with no single place that knew both.
        weight = min(
            max(0.0, _finite(request.max_position_weight, 0.05)),
            max(0.0, sizing.position_weight),
        ) * authority
        if weight <= 0.0:
            return TradePlanOutcome(
                no_trade=NoTradeDecision(
                    symbol=request.symbol,
                    strategy_id=request.strategy_id,
                    decided_at=moment,
                    reason_codes=("POSITION_WEIGHT_ZERO",),
                    stage="sizing",
                    cost_snapshot=cost_snapshot,
                    detail={"sizing": sizing.as_dict()},
                )
            )

        # -- 3. risk ---------------------------------------------------------- #
        intent = self._intent(request, decision, weight, confidence, moment)
        risk = self.risk_manager.validate(
            intent, request.account, request.market_snapshot
        )
        risk_snapshot = {
            "approved": bool(risk.approved),
            "rejection_reasons": list(risk.rejection_reasons),
            "metadata": dict(risk.metadata or {}),
            "sizing": sizing.as_dict(),
            "position_weight": round(weight, 8),
            "authority_size_fraction": authority,
            "confidence": round(confidence, 6),
            # The methodology is frozen with the plan: a later reader must be able to see
            # WHICH sizing rule produced this quantity, not merely the number.
            "sizing_methodology": "fractional_kelly_edge_liquidity_drawdown",
            "risk_rules_version": getattr(self.risk_manager.rules, "version", "default"),
        }
        if not risk.approved or risk.final_order is None:
            return TradePlanOutcome(
                no_trade=NoTradeDecision(
                    symbol=request.symbol,
                    strategy_id=request.strategy_id,
                    decided_at=moment,
                    reason_codes=tuple(
                        (*risk.rejection_reasons, "PRE_ELECTION_RISK_REJECTED")
                    ),
                    stage="risk",
                    cost_snapshot=cost_snapshot,
                    risk_snapshot=risk_snapshot,
                )
            )

        quantity = int(getattr(risk.final_order, "quantity", 0) or 0)
        if quantity <= 0:
            return TradePlanOutcome(
                no_trade=NoTradeDecision(
                    symbol=request.symbol,
                    strategy_id=request.strategy_id,
                    decided_at=moment,
                    reason_codes=("RISK_APPROVED_ZERO_QUANTITY",),
                    stage="risk",
                    cost_snapshot=cost_snapshot,
                    risk_snapshot=risk_snapshot,
                )
            )

        # -- 4. the plan --------------------------------------------------------- #
        try:
            plan = TradePlan(
                plan_id=new_plan_id(request.symbol, moment),
                created_at=moment,
                expires_at=moment
                + timedelta(seconds=max(30.0, float(request.plan_ttl_seconds))),
                symbol=str(request.symbol).upper(),
                market=str(request.market),
                direction=str(request.direction).upper(),
                strategy_id=str(request.strategy_id),
                quantity=quantity,
                max_notional=quantity * price,
                entry_rule=EntryRule(
                    trigger=request.entry_trigger,
                    min_price=price * (1.0 - ENTRY_BAND_RATE),
                    max_price=price * (1.0 + ENTRY_BAND_RATE),
                    max_wait_seconds=float(request.plan_ttl_seconds),
                ),
                exit_rules=ExitRules(
                    take_profit_rate=request.take_profit_rate,
                    stop_loss_rate=request.stop_loss_rate,
                    trailing_rate=request.trailing_rate,
                    max_holding_seconds=int(request.max_holding_seconds),
                    strategy_exit_trigger=request.strategy_exit_trigger,
                ),
                cancel_rule=request.cancel_rule,
                expected_net_edge_bps=round(
                    decision.net_expected_return * 10_000.0, 6
                ),
                cost_snapshot=cost_snapshot,
                risk_snapshot=risk_snapshot,
                weekday_time_context=dict(request.weekday_time_context),
                source_ids=tuple(request.source_ids),
                status=TradePlanStatus.ARMED,
                reference_price=price,
                election_context=dict(request.election_context),
                decision_id=request.decision_id,
                session_id=request.session_id,
                order_contract=dict(request.order_contract),
            )
        except TradePlanError as exc:
            return TradePlanOutcome(
                no_trade=NoTradeDecision(
                    symbol=request.symbol,
                    strategy_id=request.strategy_id,
                    decided_at=moment,
                    reason_codes=(f"PLAN_CONSTRUCTION_FAILED:{exc}",),
                    stage="plan",
                    cost_snapshot=cost_snapshot,
                    risk_snapshot=risk_snapshot,
                )
            )
        return TradePlanOutcome(plan=plan)

    # ------------------------------------------------------------------ #
    def _profitability(self, request: PlanRequest, price: float):
        gross = max(0.0, _finite(request.gross_edge_bps)) / 10_000.0
        sign = -1.0 if str(request.direction).upper() == "SHORT" else 1.0
        expected_exit = price * (1.0 + sign * gross)
        spread = _finite(request.spread_bps, -1.0)
        return self.profitability_gate.evaluate(
            ProfitabilityInput(
                symbol=str(request.symbol),
                action="SELL" if str(request.direction).upper() == "SHORT" else "BUY",
                market=str(request.market),
                venue=str(request.venue or request.market),
                instrument_type=str(request.instrument_type),
                entry_price=price,
                expected_exit_price=expected_exit,
                quantity=1,
                spread_rate=(spread / 10_000.0) if spread > 0 else None,
                liquidity_score=_finite(request.liquidity_score, 1.0),
                realized_volatility=(
                    _finite(request.realized_volatility)
                    if request.realized_volatility is not None
                    else None
                ),
                orderbook_snapshot=request.orderbook_snapshot,
                average_daily_trading_value=_finite(
                    getattr(request.market_snapshot, "average_daily_trading_value", 0.0)
                ),
                account_equity_krw=_finite(getattr(request.account, "equity", 0.0)),
            )
        )

    def _intent(
        self,
        request: PlanRequest,
        decision: Any,
        weight: float,
        confidence: float,
        moment: datetime,
    ) -> OrderIntent:
        short = str(request.direction).upper() == "SHORT"
        price = _finite(request.reference_price)
        gross = max(0.0, _finite(request.gross_edge_bps)) / 10_000.0
        return OrderIntent(
            ticker=str(request.symbol).upper(),
            market=str(request.market),
            action=OrderAction.SELL if short else OrderAction.BUY,
            suggested_weight=weight,
            confidence=confidence,
            valid_until=moment
            + timedelta(seconds=max(30.0, float(request.plan_ttl_seconds))),
            reasoning_summary=(f"pre_election_plan:{request.strategy_id}",),
            supporting_factors=(f"strategy:{request.strategy_id}",),
            contradicting_factors=(),
            source_data_ids=tuple(request.source_ids)
            or (f"election:{request.symbol}:{moment.strftime('%Y%m%d%H%M%S')}",),
            strategy_family=str(request.strategy_id),
            signal_name=f"elected:{request.strategy_id}",
            expected_exit_price=price * (1.0 + (-1.0 if short else 1.0) * gross),
            expected_holding_minutes=max(
                1, int(max(60, int(request.max_holding_seconds)) / 60)
            ),
            gross_expected_return=gross,
            target_net_return=decision.required_min_net_return,
            validation_id=f"plan:{request.strategy_id}:{request.symbol}",
            strategy_metadata={
                "elected": True,
                "strategy_id": request.strategy_id,
                "weekday_time_context": dict(request.weekday_time_context),
                "profitability_decision": decision.as_dict(),
            },
            position_direction="SHORT" if short else "LONG",
            position_effect="OPEN",
            execution_product=str(
                dict(request.order_contract).get("execution_product") or "CASH"
            ),
        )
