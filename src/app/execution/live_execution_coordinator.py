from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.config import OrderExecutionConfig, load_order_execution_config
from app.config.live_config import LiveConfigError, load_live_trading_safety_config
from app.config.refactor_flags import RefactorFeatureFlags
from app.execution.causal_journal import CausalOrderJournal
from app.execution.idempotency_store import IdempotencyStore
from app.execution.kis_real import KisApiError
from app.execution.kis_auth import run_kis_health_check
from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.kis_types import LiveOrderSubmission
from app.execution.live_order_journal import LiveOrderJournal
from app.execution.order_status_tracker import OrderStatusTracker
from app.execution.execution_guard import (
    ExecutionGuard,
    GuardOrder,
    default_execution_guard,
)
from app.schemas.domain import FinalOrder, OrderType
from app.trading.contracts import (
    IntentAction,
    OrderIntent as CausalOrderIntent,
    RiskVerdict,
    RiskVerdictAction,
)
from app.trading.live_runtime_guard import evaluate_live_runtime_gates


@dataclass(frozen=True)
class _AcceptedOrder:
    order: FinalOrder
    # None after durable replay: the original ack alone cannot establish what
    # remains after a restart. A subsequent broker status can resolve it.
    remaining_quantity: int | None
    ancestor_order_ids: tuple[str, ...] = ()
    filled_baseline: int = 0
    last_filled_quantity: int = 0


class LiveExecutionCoordinator:
    """Guarded bridge from approved FinalOrder to the KIS order endpoint."""

    def __init__(
        self,
        broker: Any,
        *,
        idempotency_store: IdempotencyStore | None = None,
        journal: LiveOrderJournal | None = None,
        causal_journal: CausalOrderJournal | None = None,
        execution_config: OrderExecutionConfig | None = None,
        execution_guard: ExecutionGuard | None = None,
        plan_provider: Any | None = None,
        orderable_cash_provider: Any | None = None,
        sellable_quantity_provider: Any | None = None,
        cash_equity_only: bool = False,
    ) -> None:
        self.broker = broker
        self.idempotency_store = idempotency_store or IdempotencyStore()
        self.journal = journal or LiveOrderJournal()
        self.causal_journal = causal_journal
        self.execution_config = execution_config or load_order_execution_config(allow_example=True)
        self.status_tracker = OrderStatusTracker(broker)
        # Built lazily-but-eagerly: constructing it here means a misconfigured guard
        # fails at wiring time rather than on the first live order. This is the ONLY
        # remaining gate on the execution path, and it judges order-ability, never
        # investment quality — see app.execution.execution_guard.
        self.execution_guard = (
            execution_guard
            if execution_guard is not None
            # Process health is checked once in ``_preflight_failures``. Giving
            # the same broker to the guard repeated the full account/WebSocket
            # network probe after the quote was priced, adding ~13 seconds and
            # making the order's own fresh tick stale before submission.
            else default_execution_guard(broker=None, require_plan=False)
        )
        self.plan_provider = plan_provider
        self.orderable_cash_provider = orderable_cash_provider
        self.sellable_quantity_provider = sellable_quantity_provider
        self.cash_equity_only = cash_equity_only
        self._accepted_orders: OrderedDict[str, _AcceptedOrder] = OrderedDict()
        self._accepted_orders_lock = threading.RLock()

    def submit_final_order(self, order: FinalOrder, *, idempotency_key: str | None = None) -> LiveOrderSubmission:
        self._validate_final_order(order)
        key = idempotency_key or self._idempotency_key(order)
        payload_hash = self._payload_hash(order)
        existing = self.idempotency_store.get(key, ttl_seconds=self.execution_config.idempotency_ttl_seconds)
        if existing is not None:
            if existing.payload_hash != payload_hash:
                raise LiveExecutionBlocked(("IDEMPOTENCY_KEY_PAYLOAD_MISMATCH",))
            result = existing.result
            self._remember_accepted_order(str(result.get("broker_order_id") or ""), order,
                                          str(result.get("status") or existing.status), replay=True)
            return LiveOrderSubmission(
                execution_id=str(result.get("execution_id") or key),
                idempotency_key=key,
                status=str(result.get("status") or existing.status),
                broker_order_id=result.get("broker_order_id"),
                submitted_at=_parse_dt(result.get("submitted_at")) or existing.created_at,
                message="idempotent replay",
            )

        failures = self._preflight_failures() + self._pre_submit_failures(order)
        if failures:
            self.journal.record("live_order_blocked", {"order": order, "reason_codes": failures})
            raise LiveExecutionBlocked(tuple(failures))

        execution_id = f"LIVE-{uuid4().hex}"
        pending_result = {
            "execution_id": execution_id,
            "broker_order_id": None,
            "status": "PENDING_SUBMISSION",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "order": asdict(order),
        }
        reserved, reservation = self.idempotency_store.reserve(
            key,
            payload_hash,
            pending_result,
            ttl_seconds=self.execution_config.idempotency_ttl_seconds,
        )
        if not reserved:
            if reservation.payload_hash != payload_hash:
                raise LiveExecutionBlocked(("IDEMPOTENCY_KEY_PAYLOAD_MISMATCH",))
            result = reservation.result
            self._remember_accepted_order(str(result.get("broker_order_id") or ""), order,
                                          str(result.get("status") or reservation.status), replay=True)
            return LiveOrderSubmission(
                execution_id=str(result.get("execution_id") or key),
                idempotency_key=key,
                status=str(result.get("status") or reservation.status),
                broker_order_id=result.get("broker_order_id"),
                submitted_at=_parse_dt(result.get("submitted_at")) or reservation.created_at,
                message="idempotent replay",
            )
        self.journal.record(
            "live_order_submission_attempt",
            {"execution_id": execution_id, "idempotency_key": key, "order": order},
        )
        try:
            receipt = self.broker.place_limit_order(order)
        except Exception as exc:
            error_payload: dict[str, Any] | None = None
            if isinstance(exc, KisApiError):
                response = getattr(exc, "response", None)
                if isinstance(response, dict):
                    # 전체 KIS 응답을 보존해 msg1(한글 사유) 등 진단 정보를 남긴다.
                    error_payload = {str(k): v for k, v in response.items()}
                    error_payload.setdefault("rt_cd", response.get("rt_cd"))
                    error_payload.setdefault("msg_cd", response.get("msg_cd"))
                    error_payload.setdefault("msg1", response.get("msg1"))
            self.journal.record(
                "live_order_submission_error",
                {
                    "execution_id": execution_id,
                    "idempotency_key": key,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                    "error_payload": error_payload,
                },
            )
            self.idempotency_store.put(
                key,
                payload_hash,
                "SUBMISSION_ERROR_RECONCILIATION_REQUIRED",
                {
                    **pending_result,
                    "status": "SUBMISSION_ERROR_RECONCILIATION_REQUIRED",
                    "error_type": exc.__class__.__name__,
                },
            )
            raise

        broker_order_id = str(getattr(receipt, "order_id", ""))
        status = str(getattr(receipt, "status", "UNKNOWN"))
        self._remember_accepted_order(broker_order_id, order, status)
        result = {
            "execution_id": execution_id,
            "broker_order_id": broker_order_id,
            "status": status,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "order": asdict(order),
        }
        self.idempotency_store.put(key, payload_hash, status, result)
        self.journal.record("live_order_submitted", {**result, "idempotency_key": key})
        return LiveOrderSubmission(
            execution_id=execution_id,
            idempotency_key=key,
            status=status,
            broker_order_id=broker_order_id or None,
            submitted_at=_parse_dt(result["submitted_at"]) or datetime.now(timezone.utc),
            message=str(getattr(receipt, "message", "")),
        )

    def submit_approved_intent(
        self,
        intent: CausalOrderIntent,
        verdict: RiskVerdict,
        order: FinalOrder,
    ) -> LiveOrderSubmission:
        """Submit the strategy-owned path after durable causal validation.

        This path is deliberately separate from the legacy FinalOrder API and
        remains inaccessible until its feature flag is explicitly enabled.
        """
        flags = RefactorFeatureFlags.from_env()
        if not flags.strategy_owned_execution:
            raise LiveExecutionBlocked(("STRATEGY_OWNED_EXECUTION_DISABLED",))
        self._validate_causal_chain(intent, verdict, order)
        causal_journal = self.causal_journal or CausalOrderJournal()
        self.causal_journal = causal_journal
        causal_journal.persist_intent(intent)
        causal_journal.persist_risk_verdict(verdict)
        causal_journal.record(
            "broker_submission_authorized",
            {
                "intent_id": intent.intent_id,
                "verdict_id": verdict.verdict_id,
                "idempotency_key": intent.idempotency_key,
                "approved_quantity": verdict.approved_quantity,
            },
        )
        submission = self.submit_final_order(order, idempotency_key=intent.idempotency_key)
        causal_journal.record(
            "broker_order_linked",
            {
                "intent_id": intent.intent_id,
                "verdict_id": verdict.verdict_id,
                "idempotency_key": intent.idempotency_key,
                "execution_id": submission.execution_id,
                "broker_order_id": submission.broker_order_id,
                "status": submission.status,
                "submitted_at": submission.submitted_at,
            },
        )
        return submission

    def _validate_causal_chain(
        self,
        intent: CausalOrderIntent,
        verdict: RiskVerdict,
        order: FinalOrder,
    ) -> None:
        if verdict.intent_id != intent.intent_id:
            raise LiveExecutionBlocked(("RISK_VERDICT_INTENT_MISMATCH",))
        if verdict.action not in {
            RiskVerdictAction.APPROVE,
            RiskVerdictAction.RESIZE,
            RiskVerdictAction.EMERGENCY_EXIT,
        }:
            raise LiveExecutionBlocked(("RISK_VERDICT_NOT_EXECUTABLE",))
        if verdict.approved_quantity <= 0:
            raise LiveExecutionBlocked(("RISK_APPROVED_QUANTITY_NOT_POSITIVE",))
        if verdict.approved_quantity > intent.quantity:
            raise LiveExecutionBlocked(("RISK_QUANTITY_EXCEEDS_INTENT",))
        if order.quantity != verdict.approved_quantity:
            raise LiveExecutionBlocked(("FINAL_ORDER_QUANTITY_DIFFERS_FROM_RISK",))
        if order.ticker.upper() != intent.symbol.upper():
            raise LiveExecutionBlocked(("FINAL_ORDER_SYMBOL_DIFFERS_FROM_INTENT",))
        expected_side = {
            IntentAction.BUY: "BUY",
            IntentAction.SELL: "SELL",
        }.get(intent.action)
        if expected_side is None:
            raise LiveExecutionBlocked(("INTENT_ACTION_NOT_SUBMITTABLE",))
        if order.side.value != expected_side:
            raise LiveExecutionBlocked(("FINAL_ORDER_SIDE_DIFFERS_FROM_INTENT",))

    def poll_status(self, broker_order_id: str) -> Any:
        snapshot = self.status_tracker.poll(
            broker_order_id,
            interval_seconds=self.execution_config.poll_order_status_interval_seconds,
            timeout_seconds=self.execution_config.max_order_status_poll_seconds,
        )
        self.journal.record("live_order_status", snapshot)
        self._record_accepted_order_status(broker_order_id, snapshot)
        return snapshot

    def amend_final_order(self, broker_order_id: str, replacement: FinalOrder) -> LiveOrderSubmission:
        self._validate_final_order(replacement)
        failures = self._preflight_failures() + self._pre_submit_failures(
            replacement, amending_broker_order_id=str(broker_order_id),
        )
        if failures:
            self.journal.record(
                "live_order_amend_blocked",
                {"broker_order_id": broker_order_id, "order": replacement, "reason_codes": failures},
            )
            raise LiveExecutionBlocked(tuple(failures))
        execution_id = f"LIVE-AMEND-{uuid4().hex}"
        self.journal.record(
            "live_order_amend_attempt",
            {"execution_id": execution_id, "broker_order_id": broker_order_id, "order": replacement},
        )
        try:
            receipt = self.broker.amend_limit_order(broker_order_id, replacement)
        except Exception as exc:
            # A timeout may have amended the order at the broker. Its prior
            # residual is no longer safe to reuse until status reconciliation.
            with self._accepted_orders_lock:
                previous = self._accepted_orders.get(str(broker_order_id))
                if previous is not None:
                    self._accepted_orders[str(broker_order_id)] = replace(previous, remaining_quantity=None)
            self.journal.record(
                "live_order_amend_error",
                {
                    "execution_id": execution_id,
                    "broker_order_id": broker_order_id,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                },
            )
            raise
        amended_order_id = str(getattr(receipt, "order_id", "") or broker_order_id)
        with self._accepted_orders_lock:
            previous = self._accepted_orders.pop(str(broker_order_id), None)
        ancestors = tuple(dict.fromkeys((*(previous.ancestor_order_ids if previous else ()), str(broker_order_id))))[-16:]
        # Some venues retain the original ID and its cumulative fill counter.
        # In that case only fills after this acknowledgement reduce the revised
        # residual. A new ID starts its own fill counter.
        baseline = (previous.last_filled_quantity
                    if previous is not None and amended_order_id == str(broker_order_id) else 0)
        self._remember_accepted_order(amended_order_id, replacement,
                                      str(getattr(receipt, "status", "UNKNOWN")),
                                      ancestors=ancestors, filled_baseline=baseline)
        self.journal.record(
            "live_order_amended",
            {
                "execution_id": execution_id,
                "broker_order_id": amended_order_id,
                "previous_broker_order_id": broker_order_id,
                "status": str(getattr(receipt, "status", "ACCEPTED")),
                "order": asdict(replacement),
            },
        )
        return LiveOrderSubmission(
            execution_id=execution_id,
            idempotency_key=f"amend:{broker_order_id}",
            status=str(getattr(receipt, "status", "ACCEPTED")),
            broker_order_id=amended_order_id,
            submitted_at=datetime.now(timezone.utc),
            message=str(getattr(receipt, "message", "")),
        )

    def cancel_final_order(self, broker_order_id: str, order: FinalOrder) -> LiveOrderSubmission:
        self._validate_final_order(order)
        failures = self._preflight_failures()
        if failures:
            self.journal.record(
                "live_order_cancel_blocked",
                {"broker_order_id": broker_order_id, "order": order, "reason_codes": failures},
            )
            raise LiveExecutionBlocked(tuple(failures))
        execution_id = f"LIVE-CANCEL-{uuid4().hex}"
        self.journal.record(
            "live_order_cancel_attempt",
            {"execution_id": execution_id, "broker_order_id": broker_order_id, "order": order},
        )
        # A failed cancel has to leave a trace, exactly as a failed amend does.
        # Without this the journal recorded 4,549 cancel attempts against 3
        # completions and ZERO errors, so 4,546 outcomes were simply absent: the
        # single most important failure mode on the exit path -- "the order is still
        # resting and we could not withdraw it" -- was unobservable, and reading the
        # journal suggested cancels were being silently ignored rather than refused.
        try:
            receipt = self.broker.cancel_order(broker_order_id, order)
        except Exception as exc:
            self.journal.record(
                "live_order_cancel_error",
                {
                    "execution_id": execution_id,
                    "broker_order_id": broker_order_id,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                },
            )
            raise
        canceled_order_id = str(getattr(receipt, "order_id", "") or broker_order_id)
        with self._accepted_orders_lock:
            self._accepted_orders.pop(str(broker_order_id), None)
            self._accepted_orders.pop(canceled_order_id, None)
        self.journal.record(
            "live_order_canceled",
            {
                "execution_id": execution_id,
                "broker_order_id": canceled_order_id,
                "previous_broker_order_id": broker_order_id,
                "status": str(getattr(receipt, "status", "CANCELED")),
                "order": asdict(order),
            },
        )
        return LiveOrderSubmission(
            execution_id=execution_id,
            idempotency_key=f"cancel:{broker_order_id}",
            status=str(getattr(receipt, "status", "CANCELED")),
            broker_order_id=canceled_order_id,
            submitted_at=datetime.now(timezone.utc),
            message=str(getattr(receipt, "message", "")),
        )

    def _preflight_failures(self) -> list[str]:
        require_manual_arming = _require_manual_arming()
        failures = list(
            evaluate_live_runtime_gates(require_manual_arming=require_manual_arming).failures
        )
        # Account reconciliation, orderable cash and feed freshness are checked
        # per order by ExecutionGuard. The preflight only needs credentials,
        # mode and a valid/cached token; repeating slow portfolio and WebSocket
        # approval calls here made fresh orders stale before the guard saw them.
        health = run_kis_health_check(
            self.broker,
            include_account=False,
            include_websocket=False,
        )
        if not health.ok:
            failures.extend(f"KIS_HEALTH_{name.upper()}_FAILED" for name in health.failures)
        return failures

    def _pre_submit_failures(
        self, order: FinalOrder, *, amending_broker_order_id: str | None = None,
    ) -> list[str]:
        """Order-specific re-verification, one step before the broker call.

        Separate from :meth:`_preflight_failures` because the two answer different
        questions: that one asks "is this process allowed to trade at all", this one asks
        "is THIS order still sendable right now". Kept apart so a test that disables the
        process-level gates does not silently disable the per-order ones too.

        Technical only. The guard has no access to the strategy's edge, confidence or
        ranking, and cannot form an opinion about whether the trade is a good one.
        """
        guard = self.execution_guard
        if guard is None:
            return []
        amendment = {}
        if amending_broker_order_id is not None:
            with self._accepted_orders_lock:
                origin = self._accepted_orders.get(amending_broker_order_id)
            original_order = (replace(origin.order, quantity=origin.remaining_quantity)
                              if origin is not None and origin.remaining_quantity is not None else None)
            amendment = {"amending_broker_order_id": amending_broker_order_id,
                         "amending_order": self._guard_order(original_order) if original_order else None,
                         "amending_ancestor_order_ids": origin.ancestor_order_ids if origin else ()}
        decision = guard.evaluate(
            self._guard_order(order),
            plan=self._plan_for(order),
            orderable_cash=self._orderable_cash(order),
            sellable_quantity=self._sellable_quantity(order),
            **amendment,
        )
        self._last_guard_decision = decision
        if decision.allowed and decision.permitted_quantity < order.quantity:
            # A technical affordability clip is not permission to send the old,
            # oversized quantity. A fresh plan must own the reduced order.
            return ["GUARD_QUANTITY_EXCEEDS_ORDERABLE"]
        if decision.allowed:
            return []
        self.journal.record(
            "live_order_execution_guard_blocked",
            {"order": order, "guard": decision.as_dict()},
        )
        return list(decision.reason_codes)

    def _remember_accepted_order(
        self, broker_order_id: str, order: FinalOrder, status: str, *, replay: bool = False,
        ancestors: tuple[str, ...] = (), filled_baseline: int = 0,
    ) -> None:
        if not broker_order_id or str(status).upper() not in {"ACCEPTED", "OPEN", "SUBMITTED", "PENDING"}:
            return
        with self._accepted_orders_lock:
            if replay and broker_order_id in self._accepted_orders:
                return
            self._accepted_orders[broker_order_id] = _AcceptedOrder(
                order, None if replay else order.quantity, ancestors, filled_baseline, filled_baseline,
            )
            self._accepted_orders.move_to_end(broker_order_id)
            while len(self._accepted_orders) > 256:
                self._accepted_orders.popitem(last=False)

    def _record_accepted_order_status(self, broker_order_id: str, snapshot: Any) -> None:
        with self._accepted_orders_lock:
            previous = self._accepted_orders.get(str(broker_order_id))
            if previous is None:
                return
            status = str(getattr(snapshot, "status", "UNKNOWN")).upper()
            raw = getattr(snapshot, "raw", None)
            matched = (
                str(getattr(raw, "order_id", "")) == str(broker_order_id)
                and str(getattr(raw, "ticker", "")).upper() == previous.order.ticker.upper()
                and str(getattr(getattr(raw, "side", None), "value", getattr(raw, "side", ""))).upper()
                == str(getattr(previous.order.side, "value", previous.order.side)).upper()
            )
            if matched and status in {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}:
                self._accepted_orders.pop(str(broker_order_id), None)
                return
            try:
                filled = float(getattr(raw, "quantity", None))
                valid = (not isinstance(getattr(raw, "quantity", None), bool)
                         and math.isfinite(filled) and filled.is_integer()
                         and filled >= previous.last_filled_quantity
                         and 0 <= filled - previous.filled_baseline < previous.order.quantity
                         and (status == "PARTIALLY_FILLED") == (filled > 0)
                         and matched)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if status not in {"ACCEPTED", "OPEN", "SUBMITTED", "PENDING", "PARTIALLY_FILLED"} or not valid:
                self._accepted_orders[str(broker_order_id)] = replace(previous, remaining_quantity=None)
                return
            residual = previous.order.quantity - (int(filled) - previous.filled_baseline)
            if previous.remaining_quantity is not None and residual > previous.remaining_quantity:
                self._accepted_orders[str(broker_order_id)] = replace(previous, remaining_quantity=None)
                return
            self._accepted_orders[str(broker_order_id)] = _AcceptedOrder(previous.order, residual,
                previous.ancestor_order_ids, previous.filled_baseline, int(filled))

    @staticmethod
    def _guard_order(order: FinalOrder) -> GuardOrder:
        return GuardOrder(
            symbol=str(order.ticker),
            market=str(order.market),
            side=str(getattr(order.side, "value", order.side)),
            quantity=int(order.quantity),
            limit_price=float(order.limit_price),
            direction=str(getattr(order, "position_direction", "LONG") or "LONG"),
            position_effect=str(getattr(order, "position_effect", "") or ""),
            execution_product=str(getattr(order, "execution_product", "CASH") or "CASH"),
        )

    def _plan_for(self, order: FinalOrder) -> Any | None:
        provider = self.plan_provider
        if provider is None:
            return None
        try:
            return provider(str(order.ticker))
        except Exception:  # noqa: BLE001 - an unreadable plan is no plan.
            return None

    def _orderable_cash(self, order: FinalOrder) -> float | None:
        provider = self.orderable_cash_provider
        if provider is None:
            return None
        try:
            return provider(order)
        except Exception:  # noqa: BLE001 - unknown cash is handled by the guard.
            return None

    def _sellable_quantity(self, order: FinalOrder) -> int | None:
        provider = self.sellable_quantity_provider
        if provider is None:
            return None
        try:
            return provider(order)
        except Exception:  # noqa: BLE001
            return None

    def _validate_final_order(self, order: FinalOrder) -> None:
        if not isinstance(order, FinalOrder):
            raise LiveExecutionBlocked(("FINAL_ORDER_REQUIRED",))
        if self.cash_equity_only and order.resolved_position_effect == "OPEN":
            if str(order.position_direction).upper() != "LONG" or str(order.execution_product).upper() != "CASH":
                raise LiveExecutionBlocked(("CASH_LONG_ENTRY_ONLY",))
        if order.order_type != OrderType.LIMIT:
            raise LiveExecutionBlocked(("LIMIT_ORDER_REQUIRED",))
        if order.quantity <= 0:
            raise LiveExecutionBlocked(("QUANTITY_NOT_POSITIVE",))
        if order.limit_price <= 0:
            raise LiveExecutionBlocked(("LIMIT_PRICE_NOT_POSITIVE",))
        if not _supported_live_symbol(order):
            raise LiveExecutionBlocked(("INVALID_LIVE_SYMBOL",))

    @staticmethod
    def _payload_hash(order: FinalOrder) -> str:
        payload = json.dumps(asdict(order), ensure_ascii=True, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _idempotency_key(self, order: FinalOrder) -> str:
        return "final-order:" + self._payload_hash(order)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _supported_live_symbol(order: FinalOrder) -> bool:
    ticker = str(order.ticker or "")
    market = str(order.market or "").upper()
    if ticker.isdigit() and len(ticker) == 6:
        return True
    overseas_market = any(
        token in market
        for token in ("US", "NASDAQ", "NASD", "NYSE", "AMEX", "SEHK", "SHAA", "SZAA", "TKSE", "HASE", "VNSE", "OVERSEAS")
    )
    return overseas_market and ticker.replace(".", "").replace("-", "").isalnum()


def _require_manual_arming() -> bool:
    env_value = os.getenv("REQUIRE_MANUAL_ARMING")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return bool(load_live_trading_safety_config().require_manual_arming)
    except LiveConfigError:
        return True
