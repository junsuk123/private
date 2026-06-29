from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.config import OrderExecutionConfig, load_order_execution_config
from app.execution.idempotency_store import IdempotencyStore
from app.execution.kis_auth import run_kis_health_check
from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.kis_types import LiveOrderSubmission
from app.execution.live_order_journal import LiveOrderJournal
from app.execution.order_status_tracker import OrderStatusTracker
from app.schemas.domain import FinalOrder, OrderType
from app.trading.live_runtime_guard import evaluate_live_runtime_gates


class LiveExecutionCoordinator:
    """Guarded bridge from approved FinalOrder to the KIS order endpoint."""

    def __init__(
        self,
        broker: Any,
        *,
        idempotency_store: IdempotencyStore | None = None,
        journal: LiveOrderJournal | None = None,
        execution_config: OrderExecutionConfig | None = None,
    ) -> None:
        self.broker = broker
        self.idempotency_store = idempotency_store or IdempotencyStore()
        self.journal = journal or LiveOrderJournal()
        self.execution_config = execution_config or load_order_execution_config(allow_example=True)
        self.status_tracker = OrderStatusTracker(broker)

    def submit_final_order(self, order: FinalOrder, *, idempotency_key: str | None = None) -> LiveOrderSubmission:
        self._validate_final_order(order)
        key = idempotency_key or self._idempotency_key(order)
        payload_hash = self._payload_hash(order)
        existing = self.idempotency_store.get(key, ttl_seconds=self.execution_config.idempotency_ttl_seconds)
        if existing is not None:
            if existing.payload_hash != payload_hash:
                raise LiveExecutionBlocked(("IDEMPOTENCY_KEY_PAYLOAD_MISMATCH",))
            result = existing.result
            return LiveOrderSubmission(
                execution_id=str(result.get("execution_id") or key),
                idempotency_key=key,
                status=str(result.get("status") or existing.status),
                broker_order_id=result.get("broker_order_id"),
                submitted_at=_parse_dt(result.get("submitted_at")) or existing.created_at,
                message="idempotent replay",
            )

        failures = self._preflight_failures()
        if failures:
            self.journal.record("live_order_blocked", {"order": order, "reason_codes": failures})
            raise LiveExecutionBlocked(tuple(failures))

        execution_id = f"LIVE-{uuid4().hex}"
        self.journal.record(
            "live_order_submission_attempt",
            {"execution_id": execution_id, "idempotency_key": key, "order": order},
        )
        try:
            receipt = self.broker.place_limit_order(order)
        except Exception as exc:
            self.journal.record(
                "live_order_submission_error",
                {"execution_id": execution_id, "idempotency_key": key, "error_type": exc.__class__.__name__},
            )
            raise

        broker_order_id = str(getattr(receipt, "order_id", ""))
        status = str(getattr(receipt, "status", "UNKNOWN"))
        result = {
            "execution_id": execution_id,
            "broker_order_id": broker_order_id,
            "status": status,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
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

    def poll_status(self, broker_order_id: str) -> Any:
        snapshot = self.status_tracker.poll(
            broker_order_id,
            interval_seconds=self.execution_config.poll_order_status_interval_seconds,
            timeout_seconds=self.execution_config.max_order_status_poll_seconds,
        )
        self.journal.record("live_order_status", snapshot)
        return snapshot

    def _preflight_failures(self) -> list[str]:
        failures = list(
            evaluate_live_runtime_gates(require_manual_arming=True).failures
        )
        health = run_kis_health_check(self.broker, include_account=True, include_websocket=True)
        if not health.ok:
            failures.extend(f"KIS_HEALTH_{name.upper()}_FAILED" for name in health.failures)
        return failures

    def _validate_final_order(self, order: FinalOrder) -> None:
        if not isinstance(order, FinalOrder):
            raise LiveExecutionBlocked(("FINAL_ORDER_REQUIRED",))
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
        for token in ("US", "NASDAQ", "NYSE", "AMEX", "SEHK", "SHAA", "SZAA", "TKSE", "HASE", "VNSE", "OVERSEAS")
    )
    return overseas_market and ticker.replace(".", "").replace("-", "").isalnum()
