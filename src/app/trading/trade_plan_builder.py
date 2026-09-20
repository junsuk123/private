"""Where cost, size and risk are decided — before the strategy is elected.

With an ontology resolver, one ontology authority assessment binds the current
policy to the account, integer quantity and final cost at the worst allowed entry
price. Its receipt is frozen for downstream execution. Independent profitability
and position-sizing gates below are retained only for legacy callers.

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
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Callable

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
    #: Confirmed managed entries in this market's local day. None means the
    #: durable ledger could not establish a count, never an implicit zero.
    trades_today: int | None = 0


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
    """Freeze one ontology decision, or use the legacy pipeline without a resolver."""

    def __init__(
        self,
        *,
        cost_engine: TradingCostEngine | None = None,
        profitability_gate: ProfitabilityGate | None = None,
        position_sizer: PositionSizer | None = None,
        risk_manager: RiskManager | None = None,
        risk_rules: RiskRules | None = None,
        ontology_policy_resolver: Callable[..., Any] | None = None,
    ) -> None:
        self.ontology_policy_resolver = ontology_policy_resolver
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
        if self.ontology_policy_resolver is not None:
            return self._build_ontology_plan(request, moment, price)

        # -- 1. cost and net edge ------------------------------------------ #
        decision = self._profitability(request, price)
        cost_snapshot = decision.as_dict()
        if not decision.allowed and self.ontology_policy_resolver is None:
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

        policy = None
        if self.ontology_policy_resolver is not None:
            from app.risk.ontology_thresholds import OntologyRiskPolicy
            from app.data.market_capabilities import normalize_market_group
            expected_policy_market = normalize_market_group(request.market)
            try:
                policy = self.ontology_policy_resolver(
                    symbol=request.symbol, market=request.market, now=moment,
                    all_in_cost_rate=decision.all_in_cost_rate,
                    forecast_gross_bps=request.gross_edge_bps,
                    requested_horizon_seconds=request.max_holding_seconds,
                    account=request.account,
                )
                valid_policy = (
                    isinstance(policy, OntologyRiskPolicy) and policy.valid_for_entry
                    and policy.is_current(moment) and policy.symbol.upper() == request.symbol.upper()
                    and expected_policy_market is not None and policy.market == expected_policy_market.value
                )
            except Exception:
                valid_policy = False
            if not valid_policy:
                return TradePlanOutcome(no_trade=NoTradeDecision(
                    symbol=request.symbol, strategy_id=request.strategy_id, decided_at=moment,
                    reason_codes=tuple(getattr(policy, "reason_codes", ())) or ("ONTOLOGY_POLICY_UNAVAILABLE",),
                    stage="ontology_policy", cost_snapshot=cost_snapshot,
                    risk_snapshot={"ontology_risk_policy": policy.as_dict() if isinstance(policy, OntologyRiskPolicy) else None},
                ))

        if policy is not None:
            decision = self._profitability(request, price, ontology_policy=policy, now=moment)
            cost_snapshot = decision.as_dict()
            if not decision.allowed:
                return TradePlanOutcome(no_trade=NoTradeDecision(
                    symbol=request.symbol, strategy_id=request.strategy_id, decided_at=moment,
                    reason_codes=tuple(decision.rejection_reasons), stage="ontology_profitability",
                    cost_snapshot=cost_snapshot, risk_snapshot={"ontology_risk_policy": policy.as_dict()},
                ))

        # -- 2. size ---------------------------------------------------------- #
        confidence = (
            _finite(request.confidence, DEFAULT_CONFIDENCE)
            if request.confidence is not None
            else DEFAULT_CONFIDENCE
        )
        sizing = self.position_sizer.size(
            SizingInputs(
                market=request.market_snapshot.market,
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
        if policy is not None:
            weight = min(weight, policy.position_cap)
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
            intent, request.account, request.market_snapshot,
            **({"ontology_policy": policy, "now": moment} if policy is not None else {}),
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
        if policy is not None:
            risk_snapshot["ontology_risk_policy"] = policy.as_dict()
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
        from app.risk.position_sizing import market_position_cap
        market_cap = market_position_cap(request.market_snapshot.market)
        if policy is not None:
            market_cap = min(market_cap, policy.position_cap)
        # RiskManager can round a small account up to one share. Verify the final
        # lot against the market ceiling before election, so that rounding cannot
        # silently concentrate the entire account into one high-priced name.
        from app.market_affordability import equity_available_for_market
        market_equity = equity_available_for_market(request.account, request.market_snapshot)
        if quantity * price > market_equity * market_cap:
            return TradePlanOutcome(no_trade=NoTradeDecision(
                symbol=request.symbol, strategy_id=request.strategy_id, decided_at=moment,
                reason_codes=("MARKET_POSITION_CAP_EXCEEDED",), stage="sizing",
                cost_snapshot=cost_snapshot, risk_snapshot=risk_snapshot,
                detail={"market_position_cap": market_cap, "market_equity": market_equity, "quantity": quantity},
            ))
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
                expires_at=min(moment + timedelta(seconds=max(1., float(request.plan_ttl_seconds))), policy.expires_at) if policy is not None else moment + timedelta(seconds=max(30., float(request.plan_ttl_seconds))),
                symbol=str(request.symbol).upper(),
                market=str(request.market),
                direction=str(request.direction).upper(),
                strategy_id=str(request.strategy_id),
                quantity=quantity,
                max_notional=quantity * price,
                entry_rule=EntryRule(
                    trigger=request.entry_trigger,
                    min_price=price * (1.0 - (policy.entry_band_rate if policy is not None else ENTRY_BAND_RATE)),
                    max_price=price * (1.0 + (policy.entry_band_rate if policy is not None else ENTRY_BAND_RATE)),
                    max_wait_seconds=min(float(request.plan_ttl_seconds), (policy.expires_at - moment).total_seconds()) if policy is not None else float(request.plan_ttl_seconds),
                ),
                exit_rules=ExitRules(
                    take_profit_rate=policy.target_return_rate if policy is not None else request.take_profit_rate,
                    stop_loss_rate=policy.soft_stop_rate if policy is not None else request.stop_loss_rate,
                    trailing_rate=policy.trailing_stop_rate if policy is not None else request.trailing_rate,
                    max_holding_seconds=policy.maximum_holding_seconds if policy is not None else int(request.max_holding_seconds),
                    strategy_exit_trigger=request.strategy_exit_trigger,
                ),
                cancel_rule=request.cancel_rule,
                expected_net_edge_bps=round(
                    decision.net_expected_return * 10_000.0, 6
                ),
                cost_snapshot=cost_snapshot,
                risk_snapshot=risk_snapshot,
                weekday_time_context=dict(request.weekday_time_context),
                source_ids=tuple(dict.fromkeys((*request.source_ids, *((policy.policy_id, policy.evidence_id) if policy is not None else ())))),
                status=TradePlanStatus.ARMED,
                reference_price=price,
                election_context={**dict(request.election_context), **({"ontology_risk_policy": policy.as_dict()} if policy is not None else {})},
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
    def _build_ontology_plan(self, request: PlanRequest, moment: datetime, price: float) -> TradePlanOutcome:
        """Generate one policy and consume one ontology authority assessment.

        The approved order is priced at the most expensive allowed LONG entry.
        A later fill inside the band therefore needs execution checks only.
        The independent legacy profitability/sizing pipeline is not consulted.
        """
        from app.data.market_capabilities import normalize_market_group
        from app.risk.ontology_thresholds import OntologyRiskPolicy

        def rejected(reason: str, *, stage="ontology_policy", cost=None, risk=None):
            return TradePlanOutcome(no_trade=NoTradeDecision(
                request.symbol, request.strategy_id, moment, (reason,), stage,
                cost_snapshot=cost or {}, risk_snapshot=risk or {},
            ))

        group = normalize_market_group(request.market)
        snapshot_group = normalize_market_group(request.market_snapshot.market)
        if (group is None or group != snapshot_group
                or request.symbol.upper() != request.market_snapshot.ticker.upper()):
            return rejected("PLAN_MARKET_SNAPSHOT_SCOPE_MISMATCH", stage="input")
        if (isinstance(request.trades_today, bool) or not isinstance(request.trades_today, int)
                or request.trades_today < 0):
            return rejected("ONTOLOGY_TRADE_COUNT_UNAVAILABLE", stage="ontology_authority")
        contract = dict(request.order_contract)
        contract.setdefault("direction", "LONG")
        contract.setdefault("position_direction", "LONG")
        contract.setdefault("position_effect", "OPEN")
        contract.setdefault("execution_product", "CASH")
        if (str(request.direction).upper() != "LONG"
                or any(str(contract[key]).upper() != value for key, value in
                       (("direction", "LONG"), ("position_direction", "LONG"), ("position_effect", "OPEN"), ("execution_product", "CASH")))):
            return rejected("ONTOLOGY_PLAN_REQUIRES_CASH_LONG", stage="input")
        contract.update(direction="LONG", position_direction="LONG", position_effect="OPEN", execution_product="CASH")
        authority_fraction = _finite(request.authority_size_fraction)
        requested_weight = _finite(request.max_position_weight)
        gross = _finite(request.gross_edge_bps) / 10000.
        if authority_fraction <= 0 or requested_weight <= 0:
            return rejected("AUTHORITY_NOT_ORDERABLE", stage="sizing")
        expected_exit = price * (1. + gross)
        preliminary = self.cost_engine.estimate(
            symbol=request.symbol, market=request.market, venue=request.venue or request.market,
            instrument_type=request.instrument_type, entry_price=price,
            expected_exit_price=expected_exit, quantity=1,
            orderbook_snapshot=request.orderbook_snapshot,
            average_daily_trading_value=request.market_snapshot.average_daily_trading_value,
        )
        try:
            policy = self.ontology_policy_resolver(
                symbol=request.symbol, market=request.market, now=moment,
                all_in_cost_rate=preliminary.total_cost_rate, forecast_gross_bps=request.gross_edge_bps,
                requested_horizon_seconds=request.max_holding_seconds, account=request.account,
            )
            if not (isinstance(policy, OntologyRiskPolicy) and policy.is_current(moment)
                    and policy.symbol.upper() == request.symbol.upper() and policy.market == group.value):
                return rejected("ONTOLOGY_POLICY_UNAVAILABLE")
        except Exception:
            return rejected("ONTOLOGY_POLICY_UNAVAILABLE")
        expires = min(moment + timedelta(seconds=max(1., _finite(request.plan_ttl_seconds, 1.))), policy.expires_at)
        worst_entry = price * (1. + policy.entry_band_rate)
        plan_id = new_plan_id(request.symbol, moment)
        snapshot = replace(request.market_snapshot, last_price=worst_entry)
        source_ids = tuple(dict.fromkeys((*request.source_ids, policy.policy_id, policy.evidence_id)))
        book = request.orderbook_snapshot
        if hasattr(book, "as_dict"):
            book = book.as_dict()
        intent = OrderIntent(
            ticker=request.symbol.upper(), market=request.market, action=OrderAction.BUY,
            suggested_weight=min(1., requested_weight) * min(1., authority_fraction),
            confidence=max(0., min(1., _finite(request.confidence, DEFAULT_CONFIDENCE))),
            valid_until=expires, reasoning_summary=(f"ontology_plan:{request.strategy_id}",),
            supporting_factors=(f"strategy:{request.strategy_id}",), contradicting_factors=(),
            source_data_ids=source_ids, strategy_family=request.strategy_id,
            signal_name=f"elected:{request.strategy_id}", expected_exit_price=expected_exit,
            expected_holding_minutes=max(1, math.ceil(policy.maximum_holding_seconds / 60)),
            gross_expected_return=(expected_exit / worst_entry - 1.),
            target_net_return=policy.net_profit_floor_rate, validation_id=plan_id,
            position_direction="LONG", position_effect="OPEN", execution_product="CASH",
            strategy_metadata={"elected": True, "strategy_id": request.strategy_id,
                "orderbook_snapshot": book if isinstance(book, dict) else None,
                "stop_loss_rate": policy.hard_stop_rate, "ontology_risk_policy": policy.as_dict(),
                "approval_price_basis": "worst_allowed_entry", "signal_reference_price": price},
        )
        risk = self.risk_manager.validate(intent, request.account, snapshot,
            trades_today=request.trades_today, ontology_policy=policy, now=moment)
        metadata = dict(risk.metadata or {})
        receipt = metadata.get("ontology_risk_authority")
        risk_snapshot = {"approved": bool(risk.approved), "rejection_reasons": list(risk.rejection_reasons),
                         "ontology_risk_policy": policy.as_dict(), "metadata": metadata,
                         "entry_activity_scope": "bot_confirmed_entries", "trades_today": request.trades_today,
                         "authority_size_fraction": authority_fraction,
                         "sizing_methodology": "ontology_risk_authority"}
        if isinstance(receipt, Mapping):
            risk_snapshot["ontology_authority"] = dict(receipt)
        cost = dict(metadata.get("cost_breakdown") or {})
        if not risk.approved or risk.final_order is None:
            return TradePlanOutcome(no_trade=NoTradeDecision(
                request.symbol, request.strategy_id, moment,
                tuple(risk.rejection_reasons) or ("ONTOLOGY_AUTHORITY_REJECTED",),
                "ontology_authority", cost_snapshot=cost, risk_snapshot=risk_snapshot))
        if not isinstance(receipt, Mapping) or receipt.get("authority_id") != "ontology-risk-authority-v1":
            return rejected("ONTOLOGY_AUTHORITY_RECEIPT_MISSING", stage="ontology_authority", cost=cost, risk=risk_snapshot)
        quantity = int(risk.final_order.quantity)
        all_in = _finite(receipt.get("all_in_cost_rate"), -1.)
        assessed_price = _finite(cost.get("entry_price"), -1.)
        try:
            receipt_expiry = datetime.fromisoformat(str(receipt.get("expires_at")))
            receipt_valid = (receipt.get("approved") is True
                and receipt.get("phase") == "entry_assessment"
                and receipt.get("policy_id") == policy.policy_id
                and receipt.get("symbol") == request.symbol.upper()
                and receipt.get("market") == group.value
                and receipt.get("side") == "BUY"
                and receipt.get("position_direction") == "LONG"
                and receipt.get("position_effect") == "OPEN"
                and receipt.get("execution_product") == "CASH"
                and _finite(receipt.get("actual_quantity"), -1.) == quantity
                and assessed_price == _finite(receipt.get("authorized_price"), -1.)
                and assessed_price <= worst_entry
                and quantity * assessed_price <= _finite(receipt.get("actual_notional"), -1.)
                and receipt_expiry.tzinfo is not None and receipt_expiry > moment)
        except (TypeError, ValueError, OverflowError):
            receipt_valid = False
        if quantity <= 0 or all_in < 0 or assessed_price <= 0 or not receipt_valid:
            return rejected("ONTOLOGY_AUTHORITY_RECEIPT_INVALID", stage="ontology_authority", cost=cost, risk=risk_snapshot)
        expires = min(expires, receipt_expiry)
        # These are the authority's final-quantity economics, not a one-share quote.
        cost.update(all_in_cost_rate=all_in, gross_expected_return=expected_exit / assessed_price - 1.,
                    net_expected_return=expected_exit / assessed_price - 1. - all_in,
                    expected_exit_price=expected_exit, quantity=quantity,
                    policy_version=policy.policy_id, approval_price_basis="worst_allowed_entry",
                    signal_reference_price=price)
        try:
            return TradePlanOutcome(plan=TradePlan(
                plan_id=plan_id, created_at=moment, expires_at=expires,
                symbol=request.symbol.upper(), market=request.market, direction="LONG",
                strategy_id=request.strategy_id, quantity=quantity, max_notional=quantity * assessed_price,
                entry_rule=EntryRule(request.entry_trigger, min_price=price * (1. - policy.entry_band_rate),
                    max_price=assessed_price, max_wait_seconds=(expires - moment).total_seconds()),
                exit_rules=ExitRules(policy.target_return_rate, policy.soft_stop_rate, policy.trailing_stop_rate,
                    policy.maximum_holding_seconds, request.strategy_exit_trigger),
                cancel_rule=request.cancel_rule, expected_net_edge_bps=cost["net_expected_return"] * 10000.,
                cost_snapshot=cost, risk_snapshot=risk_snapshot, source_ids=source_ids,
                weekday_time_context=dict(request.weekday_time_context), status=TradePlanStatus.ARMED,
                reference_price=assessed_price, election_context={**dict(request.election_context),
                    "ontology_risk_policy": policy.as_dict()}, decision_id=request.decision_id,
                session_id=request.session_id, order_contract=contract,
            ))
        except TradePlanError as exc:
            return rejected(f"PLAN_CONSTRUCTION_FAILED:{exc}", stage="plan", cost=cost, risk=risk_snapshot)

    def _profitability(self, request: PlanRequest, price: float, *, ontology_policy: Any = None, now: datetime | None = None):
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
            ),
            **({"ontology_policy": ontology_policy, "now": now} if ontology_policy is not None else {}),
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
