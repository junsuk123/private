from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.execution.causal_journal import CausalJournalError, CausalOrderJournal
from app.execution.idempotency_store import IdempotencyStore
from app.trading.contracts import IntentAction, OrderIntent, RiskVerdict, RiskVerdictAction


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _intent(*, quantity: int = 2) -> OrderIntent:
    return OrderIntent(
        intent_id="intent-1",
        idempotency_key="owner-1:decision-1",
        strategy_instance_id="owner-1",
        position_id_if_any=None,
        symbol="005930",
        action=IntentAction.BUY,
        quantity=quantity,
        limit_or_price_policy={"kind": "limit", "price": 80000},
        urgency="NORMAL",
        reason_code="ENTRY_TRIGGERED",
        created_at=NOW,
        expires_at=NOW + timedelta(seconds=5),
    )


def test_same_idempotency_payload_is_persisted_once(tmp_path) -> None:
    journal = CausalOrderJournal(tmp_path / "causal.jsonl")
    journal.persist_intent(_intent())
    journal.persist_intent(_intent())
    assert len(journal.read_all()) == 1


def test_same_idempotency_key_with_different_payload_fails_closed(tmp_path) -> None:
    journal = CausalOrderJournal(tmp_path / "causal.jsonl")
    journal.persist_intent(_intent())
    with pytest.raises(CausalJournalError, match="PAYLOAD_MISMATCH"):
        journal.persist_intent(_intent(quantity=3))


def test_risk_verdict_requires_persisted_intent(tmp_path) -> None:
    journal = CausalOrderJournal(tmp_path / "causal.jsonl")
    verdict = RiskVerdict(
        verdict_id="verdict-1",
        intent_id="intent-1",
        action=RiskVerdictAction.APPROVE,
        approved_quantity=2,
        limits_evaluated={"position_limit": True},
        reason_codes=(),
        account_snapshot_id="account-1",
        timestamp=NOW,
    )
    with pytest.raises(CausalJournalError, match="WITHOUT_PERSISTED_INTENT"):
        journal.persist_risk_verdict(verdict)
    journal.persist_intent(_intent())
    journal.persist_risk_verdict(verdict)
    assert [row["event_type"] for row in journal.read_all()] == [
        "order_intent_persisted",
        "risk_verdict_persisted",
    ]


def test_idempotency_reservation_is_atomic_and_precedes_result(tmp_path) -> None:
    store = IdempotencyStore(tmp_path / "idempotency.jsonl")
    reserved, first = store.reserve(
        "same-key",
        "hash-1",
        {"status": "PENDING_SUBMISSION"},
        ttl_seconds=60,
    )
    reserved_again, second = store.reserve(
        "same-key",
        "hash-1",
        {"status": "PENDING_SUBMISSION"},
        ttl_seconds=60,
    )
    assert reserved is True
    assert reserved_again is False
    assert first.status == second.status == "PENDING_SUBMISSION"
