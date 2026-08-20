"""Last-line re-verification, immediately before the broker call.

Why check again here
--------------------
Everything upstream — the strategy selector, the risk manager, the ``FinalTradeGate`` —
decided at *decision time*. Between that instant and the socket write there is a queue, a
pricing step, a rate limiter and, in the worst case, a restart. The conditions that
authorised the order can have lapsed: the session closed, the feed went quiet, an earlier
order for the same symbol came back filled, the account snapshot aged out.

So the invariants an order cannot be sent without are checked once more, here, at the one
place every live order passes through. This is deliberately a *small* set — only the
things that can be established independently at this moment, without re-running the whole
decision. It does not duplicate the gate's sizing, its soft gates or its exposure limits;
those were correct when they were computed and are not re-derivable here.

Exits are checked differently, and less
----------------------------------------
A reduce or close is exempt from staleness, account reconciliation and session-entry
permission. The reasoning is the same one the gate uses: being unable to close a position
because the feed went quiet is worse than any condition that would have stopped the entry.
An exit is still blocked when it cannot be *routed* — an unknown order state or an
identical working order — because sending it would create the duplicate.

Absent evidence
---------------
A check whose input is unavailable resolves according to ``strict``. Live trading sets
``strict=True``: no evidence means no order. Paper and shadow set it False, where the
absence is recorded as a reason but does not block, because those paths routinely run
without a populated freshness registry and blocking them would remove the only way to
exercise this code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

__all__ = [
    "PreSubmitDecision",
    "PreSubmitGuard",
    "default_pre_submit_guard",
]

#: How old the most recent reconciled account snapshot may be before a BUY is refused.
#: Matches ``config/data_freshness.yaml``'s kis_rest/account degraded bound.
ACCOUNT_SNAPSHOT_MAX_AGE_SECONDS = 600.0

_EXIT_SIDES = {"SELL", "REDUCE", "CLOSE", "SHORT_COVER"}


@dataclass(frozen=True)
class PreSubmitDecision:
    allowed: bool
    reason_codes: tuple[str, ...] = ()
    checked: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "checked": list(self.checked),
            "detail": dict(self.detail),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_exit(side: Any) -> bool:
    value = getattr(side, "value", side)
    return str(value or "").strip().upper() in _EXIT_SIDES


class PreSubmitGuard:
    """Re-verifies session, order state, data freshness and account reconciliation."""

    def __init__(
        self,
        *,
        state_machine: Any | None = None,
        freshness: Any | None = None,
        store: Any | None = None,
        strict: bool = True,
        session_service: Any | None = None,
    ) -> None:
        self._state_machine = state_machine
        self._freshness = freshness
        self._store = store
        self._strict = bool(strict)
        self._session_service = session_service

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        *,
        ticker: str,
        side: Any,
        market: str,
        now: datetime | None = None,
    ) -> PreSubmitDecision:
        """Verdict for one order about to be submitted.

        Never raises. An error inside a check produces a reason code and, under
        ``strict``, a refusal — a guard that can be bypassed by making it throw is not a
        guard.
        """
        moment = now or _utcnow()
        exit_order = _is_exit(side)
        reasons: list[str] = []
        checked: list[str] = []
        detail: dict[str, Any] = {"exit": exit_order}

        for name, check in (
            ("order_state", self._check_order_state),
            ("session", self._check_session),
            ("data_freshness", self._check_freshness),
            ("account_reconciliation", self._check_account),
        ):
            if exit_order and name in {"session", "data_freshness", "account_reconciliation"}:
                # An exit must remain possible when the feed is quiet, the account
                # snapshot is old, or new entries are closed for the session.
                continue
            checked.append(name)
            try:
                found = check(ticker=ticker, side=side, market=market, now=moment, detail=detail)
            except Exception as exc:  # noqa: BLE001 - a failing check blocks under strict.
                found = (f"PRESUBMIT_CHECK_FAILED:{name}:{type(exc).__name__}",)
            reasons.extend(found)

        # Exits still get the routing check, above; this adds the one that applies to
        # every side.
        if exit_order:
            checked.append("order_state")
            try:
                reasons.extend(
                    self._check_order_state(
                        ticker=ticker, side=side, market=market, now=moment, detail=detail
                    )
                )
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"PRESUBMIT_CHECK_FAILED:order_state:{type(exc).__name__}")

        deduped = tuple(dict.fromkeys(reasons))
        blocking = tuple(
            code for code in deduped if self._strict or not code.startswith("PRESUBMIT_NO_EVIDENCE")
        )
        return PreSubmitDecision(
            allowed=not blocking,
            reason_codes=deduped,
            checked=tuple(dict.fromkeys(checked)),
            detail=detail,
        )

    # ------------------------------------------------------------------ #
    def _check_order_state(
        self, *, ticker: str, side: Any, now: datetime, detail: dict[str, Any], **_: Any
    ) -> tuple[str, ...]:
        machine = self._state_machine
        if machine is None:
            return ("PRESUBMIT_NO_EVIDENCE:ORDER_STATE",)
        reasons: list[str] = []
        unknown = [
            record.intent_id
            for record in machine.unknown_intents()
            if record.ticker == str(ticker).strip().upper()
        ]
        if unknown:
            detail["unknown_intent_ids"] = unknown
            reasons.append("UNKNOWN_ORDER_STATE")
        resolved_side = str(getattr(side, "value", side) or "").strip().upper()
        if machine.has_duplicate_risk(ticker, resolved_side):
            reasons.append("DUPLICATE_ORDER_RISK")
        return tuple(reasons)

    def _check_session(
        self, *, market: str, now: datetime, detail: dict[str, Any], **_: Any
    ) -> tuple[str, ...]:
        from app.data.market_capabilities import default_service, normalize_market_group

        service = self._session_service or default_service()
        group = normalize_market_group(str(market))
        if group is None:
            detail["session_market"] = str(market)
            return ("UNKNOWN_SESSION",)
        allowed = service.new_entry_allowed(group, now)
        detail["session"] = service.primary_capability(group, now).session.value
        if not allowed:
            detail["session_block_reasons"] = list(
                service.new_entry_block_reasons(group, now)
            )
            return ("SESSION_NOT_TRADEABLE",)
        return ()

    def _check_freshness(
        self,
        *,
        ticker: str,
        market: str,
        now: datetime,
        detail: dict[str, Any],
        **_: Any,
    ) -> tuple[str, ...]:
        registry = self._freshness
        if registry is None:
            return ("PRESUBMIT_NO_EVIDENCE:DATA_FRESHNESS",)
        all_blocking = tuple(registry.blocking_reasons(now=now))
        # Freshness rows are retained for rotating subscription symbols.  A stale
        # Samsung row must not veto a fresh INTC order (and vice versa). Keep
        # unscoped infrastructure failures plus rows scoped to this order's symbol
        # or market; every other symbol is irrelevant to this broker call.
        symbol = str(ticker or "").strip().upper()
        raw_market = str(market or "").strip().upper()
        market_scopes = {
            raw_market,
            "KRX" if raw_market in {"KR", "KRX", "KOREA"} else "US",
        }
        blocking: tuple[str, ...] = tuple(
            reason
            for reason in all_blocking
            if _freshness_reason_applies(
                reason,
                symbol=symbol,
                market_scopes=market_scopes,
            )
        )
        if blocking:
            detail["stale_streams"] = list(blocking)
            return ("STALE_DATA",)
        if all_blocking:
            detail["ignored_stale_stream_count"] = len(all_blocking)
        return ()

    def _check_account(
        self, *, now: datetime, detail: dict[str, Any], **_: Any
    ) -> tuple[str, ...]:
        store = self._store
        if store is None:
            return ("PRESUBMIT_NO_EVIDENCE:ACCOUNT_RECONCILIATION",)
        row = store.fetch_one(
            "select captured_at, reconciled from account_snapshot"
            " order by captured_at desc limit 1"
        )
        if row is None:
            return ("ACCOUNT_RECONCILIATION_FAIL",)
        detail["account_snapshot_at"] = row["captured_at"]
        if not int(row["reconciled"] or 0):
            return ("ACCOUNT_RECONCILIATION_FAIL",)
        captured = _parse(row["captured_at"])
        if captured is None or now - captured > timedelta(
            seconds=ACCOUNT_SNAPSHOT_MAX_AGE_SECONDS
        ):
            detail["account_snapshot_age_seconds"] = (
                (now - captured).total_seconds() if captured else None
            )
            return ("ACCOUNT_RECONCILIATION_FAIL",)
        return ()


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(
        timezone.utc
    )


def _freshness_reason_applies(
    reason: str,
    *,
    symbol: str,
    market_scopes: set[str],
) -> bool:
    """Whether ``STALE_DATA:source/type[:scope]`` applies to this order."""
    parts = str(reason or "").split(":", 2)
    if len(parts) < 3:
        return True
    scope = parts[2].strip().upper()
    return not scope or scope == symbol or scope in market_scopes


def default_pre_submit_guard(*, strict: bool | None = None) -> PreSubmitGuard:
    """The guard the live coordinator uses when none is injected.

    ``strict`` defaults to "are we actually trading live": in paper and shadow the
    absence of a populated freshness registry is normal and must not block, while in live
    it is exactly the missing evidence the guard exists to refuse on.
    """
    from app.execution.order_state_machine import OrderStateMachine
    from app.storage.trading_state_store import default_trading_state_store

    if strict is None:
        from app.trading.live_runtime_guard import env_bool

        strict = env_bool("LIVE_TRADING_ENABLED", False) and env_bool(
            "KIS_LIVE_ENABLED", False
        )

    store = default_trading_state_store()
    freshness = None
    try:
        from app.data.freshness import default_freshness_registry

        freshness = default_freshness_registry()
    except Exception:  # noqa: BLE001 - absence is itself a reason code under strict.
        freshness = None
    return PreSubmitGuard(
        state_machine=OrderStateMachine(store),
        freshness=freshness,
        store=store,
        strict=bool(strict),
    )
