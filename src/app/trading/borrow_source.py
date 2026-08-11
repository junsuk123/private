"""Where 대주 availability actually comes from.

Why this is a pluggable source and not a KIS method
---------------------------------------------------
Three endpoints were originally guessed and all three were wrong. KIS now publishes
the actual inventory query, ``CTSC2702R /lendable-by-company``.  It reports KIS-wide
lendable inventory rather than an account-specific final order quantity, so account,
collateral and strategy limits still reduce size downstream.

* ``TTTC8909R`` / ``inquire-credit-psamount`` — exists, but answers 융자 (margin BUY)
  purchasing power, not 대주 (stock-loan) availability;
* ``CTSC0271R`` / ``credit-by-company`` — "잘못된 TR 코드 입니다", no such TR;
* ``CTRP6504R`` / ``inquire-credit-balance`` — HTTP 404, no such path.

Guessing a fourth id would repeat the mistake, and a wrong-but-successful response
would be worse than a failure: it would populate the journal with a number that means
something else, and every downstream gate would trust it.

So availability became an interface with an explicit "not configured" default. Two
consequences worth stating:

* **Nothing silently degrades.** :class:`NullBorrowSource` reports unavailability with a
  reason, rather than returning ``available=False`` — which would look like a normal
  market state and hide the fact that no source exists at all.
* :class:`KisBorrowSource` is the production default when KIS credentials exist.
* :class:`FileBorrowSource` remains an explicit operator/testing override.

Every source returns a :class:`~app.trading.borrow.BorrowSnapshot`, so the freshness,
quantity and fee rules in ``evaluate_borrow`` apply identically no matter where the
answer came from. A file that has not been updated today goes stale on exactly the same
clock as a live query would.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.trading.borrow import BorrowSnapshot

logger = logging.getLogger(__name__)

DEFAULT_BORROW_FILE = "config/borrow_availability.json"

# Why a source could not answer. Distinguished from "the broker said no" because the
# two need different operator responses: one is an outage or a missing config, the
# other is normal market state.
REASON_NO_SOURCE = "BORROW_SOURCE_NOT_CONFIGURED"
REASON_SOURCE_UNREADABLE = "BORROW_SOURCE_UNREADABLE"
REASON_SYMBOL_ABSENT = "BORROW_SYMBOL_NOT_LISTED"
REASON_SOURCE_STALE = "BORROW_SOURCE_STALE"

# KIS's public credit policy currently quotes 4.5% for KOSPI200 and 6.0% for other
# names.  Unknown index membership deliberately receives the higher rate.  The values
# remain environment-overridable so a policy change can be deployed without code.
DEFAULT_KIS_KOSPI200_FEE_BPS_ANNUALISED = 450.0
DEFAULT_KIS_OTHER_FEE_BPS_ANNUALISED = 600.0


class BorrowDataSource(Protocol):
    """Answers "could this be borrowed, how much, at what rate, as of when"."""

    name: str

    def available(self) -> bool:
        """Is this source usable at all right now?"""

    def snapshot(self, symbol: str, *, now: datetime) -> BorrowSnapshot | None:
        """Point-in-time locate, or ``None`` when unanswerable.

        ``None`` is NOT ``available=False``. A caller must treat it as no-locate, but
        an operator must be able to tell "we could not ask" from "the answer was no".
        """

    def universe(self) -> tuple[str, ...]:
        """Symbols this source can speak about. Empty when unknown."""

    def status(self) -> dict[str, Any]:
        ...


@dataclass
class NullBorrowSource:
    """The default: no source configured, and it says so.

    Deliberately not a source that returns ``available=False``. That would be
    indistinguishable from a market where nothing is borrowable, and the system would
    look like it was working correctly while actually having no data path at all.
    """

    name: str = "null"
    reason: str = REASON_NO_SOURCE

    def available(self) -> bool:
        return False

    def snapshot(self, symbol: str, *, now: datetime) -> BorrowSnapshot | None:
        return None

    def universe(self) -> tuple[str, ...]:
        return ()

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": False,
            "reason": self.reason,
            "detail": (
                "No 대주 availability source is configured. Short strategies will "
                "record signals but every one will be SIGNAL_VALID_BUT_UNEXECUTABLE, "
                "so no shadow evidence accumulates and no arm can be promoted."
            ),
        }


class FileBorrowSource:
    """Operator-maintained 대주 availability, read from a JSON file.

    Format::

        {
          "observed_at": "2026-08-02T09:05:00+09:00",
          "source": "kis-web-ui",
          "symbols": {
            "SYMBOL_A": {"available": true, "quantity": 500, "fee_bps_annualised": 800},
            "000660": {"available": false, "reason": "no inventory"}
          }
        }

    ``observed_at`` is mandatory and applies to the whole file. It is what makes the
    normal freshness rule work: a file left over from yesterday goes stale on exactly
    the same clock a live query would, rather than being trusted because it is local.
    A file without it is refused outright — an undated locate cannot be point-in-time
    evaluated, and pretending it was observed "now" is the same look-ahead leak the
    shadow evaluator exists to prevent.

    Reloaded on mtime change so an operator can update it without a restart.
    """

    name = "file"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("BORROW_AVAILABILITY_FILE", DEFAULT_BORROW_FILE))
        self._lock = threading.RLock()
        self._mtime: float | None = None
        self._observed_at: datetime | None = None
        self._symbols: dict[str, dict[str, Any]] = {}
        self._source_label: str = "file"
        self._error: str = ""

    # -- loading ------------------------------------------------------------- #
    def _reload_if_changed(self) -> None:
        with self._lock:
            try:
                stat = self.path.stat()
            except OSError:
                self._symbols = {}
                self._observed_at = None
                self._error = f"{REASON_SOURCE_UNREADABLE}:{self.path}"
                return
            if self._mtime == stat.st_mtime:
                return
            self._mtime = stat.st_mtime
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._symbols = {}
                self._observed_at = None
                self._error = f"{REASON_SOURCE_UNREADABLE}:{exc}"
                return
            observed = _parse_iso(payload.get("observed_at"))
            if observed is None:
                # No timestamp == no point-in-time claim. Refused rather than stamped
                # with "now", which would silently make a stale file look fresh.
                self._symbols = {}
                self._observed_at = None
                self._error = "BORROW_SOURCE_MISSING_OBSERVED_AT"
                return
            raw = payload.get("symbols")
            self._symbols = {
                str(key).strip().upper(): dict(value)
                for key, value in (raw or {}).items()
                if isinstance(value, Mapping)
            }
            self._observed_at = observed
            self._source_label = str(payload.get("source") or "file")
            self._error = ""

    # -- BorrowDataSource ----------------------------------------------------- #
    def available(self) -> bool:
        self._reload_if_changed()
        with self._lock:
            return bool(self._symbols) and self._observed_at is not None

    def snapshot(self, symbol: str, *, now: datetime) -> BorrowSnapshot | None:
        self._reload_if_changed()
        key = str(symbol or "").strip().upper()
        with self._lock:
            entry = self._symbols.get(key)
            observed_at = self._observed_at
            label = self._source_label
        if entry is None or observed_at is None:
            # A symbol the file does not mention is UNANSWERED, not unavailable.
            return None
        quantity = entry.get("quantity")
        fee = entry.get("fee_bps_annualised")
        return BorrowSnapshot(
            symbol=key,
            observed_at=observed_at,
            available=bool(entry.get("available")),
            available_quantity=(None if quantity is None else int(quantity)),
            borrow_fee_bps_annualised=(None if fee is None else float(fee)),
            return_deadline=_parse_iso(entry.get("return_deadline")),
            source=f"file:{label}",
            source_payload_hash=BorrowSnapshot.payload_hash(entry),
            reject_reason=str(entry.get("reason") or ""),
        )

    def universe(self) -> tuple[str, ...]:
        self._reload_if_changed()
        with self._lock:
            return tuple(
                symbol
                for symbol, entry in self._symbols.items()
                if bool(entry.get("available"))
            )

    def status(self) -> dict[str, Any]:
        self._reload_if_changed()
        with self._lock:
            return {
                "name": self.name,
                "path": str(self.path),
                "available": bool(self._symbols) and self._observed_at is not None,
                "reason": self._error,
                "observed_at": (
                    self._observed_at.isoformat() if self._observed_at else None
                ),
                "symbol_count": len(self._symbols),
                "borrowable_count": len(self.universe()),
                "source_label": self._source_label,
            }


class CallableBorrowSource:
    """Adapter for a verified broker query, once one exists.

    Kept so that fixing the KIS endpoint is a one-line wiring change rather than a
    rewrite: implement the query, pass it here, and the freshness/quantity/fee rules
    apply unchanged.
    """

    name = "callable"

    def __init__(self, provider: Any, *, label: str = "broker") -> None:
        self._provider = provider
        self._label = label
        self._last_error = ""

    def available(self) -> bool:
        return self._provider is not None

    def snapshot(self, symbol: str, *, now: datetime) -> BorrowSnapshot | None:
        if self._provider is None:
            return None
        try:
            return self._provider(symbol)
        except Exception as exc:  # noqa: BLE001 - an error is "unanswered", not "no".
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("borrow source %s failed for %s: %s", self._label, symbol, exc)
            return None

    def universe(self) -> tuple[str, ...]:
        return ()

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self._label,
            "available": self._provider is not None,
            "reason": self._last_error,
        }


class KisBorrowSource:
    """Live KIS inventory backed by the official ``CTSC2702R`` query.

    The API answer is deliberately combined only with a *conservative public policy
    fee*.  It is not treated as account-specific buying power.  The account/risk
    gates continue to cap the requested quantity, and KIS remains the final authority
    when the credit order is submitted.
    """

    name = "kis"

    def __init__(
        self,
        client: Any | None = None,
        *,
        kospi200_symbols: Sequence[str] | None = None,
        kospi200_fee_bps_annualised: float | None = None,
        other_fee_bps_annualised: float | None = None,
    ) -> None:
        if client is None:
            from app.execution.kis_real import KisDevelopersApiClient

            client = KisDevelopersApiClient(enabled=False)
        self._client = client
        self._kospi200 = {
            str(item).strip().upper() for item in (kospi200_symbols or ()) if str(item).strip()
        }
        self._kospi200_fee = float(
            kospi200_fee_bps_annualised
            if kospi200_fee_bps_annualised is not None
            else os.getenv(
                "KIS_BORROW_KOSPI200_FEE_BPS_ANNUALISED",
                str(DEFAULT_KIS_KOSPI200_FEE_BPS_ANNUALISED),
            )
        )
        self._other_fee = float(
            other_fee_bps_annualised
            if other_fee_bps_annualised is not None
            else os.getenv(
                "KIS_BORROW_OTHER_FEE_BPS_ANNUALISED",
                str(DEFAULT_KIS_OTHER_FEE_BPS_ANNUALISED),
            )
        )
        self._last_error = ""
        self._last_observed_at: datetime | None = None
        self._last_symbol = ""
        self._known_borrowable: set[str] = set()

    def available(self) -> bool:
        credentials = getattr(self._client, "credentials", None)
        if credentials is None:
            return self._client is not None
        return bool(
            getattr(credentials, "app_key", "")
            and getattr(credentials, "app_secret", "")
            and getattr(credentials, "account_no", "")
        )

    def snapshot(self, symbol: str, *, now: datetime) -> BorrowSnapshot | None:
        del now  # Observation time is the broker response time, never caller-supplied.
        code = str(symbol or "").strip().upper()
        if not code or not self.available():
            return None
        try:
            answer = self._client.get_lendable_by_company(code)
        except Exception as exc:  # noqa: BLE001 - unanswered must remain distinct from no.
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("KIS borrow inventory failed for %s: %s", code, exc)
            return None

        observed_at = datetime.now(timezone.utc)
        available = bool(answer.get("available"))
        quantity = answer.get("available_quantity")
        if available and quantity is not None and int(quantity) > 0:
            self._known_borrowable.add(code)
        else:
            self._known_borrowable.discard(code)
        self._last_error = ""
        self._last_observed_at = observed_at
        self._last_symbol = code
        fee = self._kospi200_fee if code in self._kospi200 else self._other_fee
        raw = answer.get("raw") if isinstance(answer.get("raw"), Mapping) else answer
        return BorrowSnapshot(
            symbol=code,
            observed_at=observed_at,
            available=available,
            available_quantity=(None if quantity is None else max(0, int(quantity))),
            borrow_fee_bps_annualised=fee,
            source="kis:CTSC2702R+public-credit-policy",
            source_payload_hash=BorrowSnapshot.payload_hash(raw),
            reject_reason=str(answer.get("reject_reason") or ""),
        )

    def universe(self) -> tuple[str, ...]:
        # Demand-driven queries populate this cache. An empty cache means "not queried
        # yet", not a broker declaration that no name is lendable.
        return tuple(sorted(self._known_borrowable))

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available(),
            "reason": self._last_error,
            "endpoint": "/uapi/domestic-stock/v1/quotations/lendable-by-company",
            "tr_id": "CTSC2702R",
            "last_observed_at": (
                self._last_observed_at.isoformat() if self._last_observed_at else None
            ),
            "last_symbol": self._last_symbol or None,
            "known_borrowable_count": len(self._known_borrowable),
            "fee_policy": {
                "kospi200_bps_annualised": self._kospi200_fee,
                "other_bps_annualised": self._other_fee,
                "unknown_membership_uses": "other",
            },
        }


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


_DEFAULT_SOURCE: BorrowDataSource | None = None
_DEFAULT_SOURCE_LOCK = threading.Lock()


def default_borrow_source() -> BorrowDataSource:
    """The configured source, or :class:`NullBorrowSource`.

    ``BORROW_SOURCE`` may force ``kis``, ``file`` or ``none``. In ``auto`` mode the
    verified KIS source wins when credentials exist; a readable file is retained only
    as a development/operator fallback.
    """
    global _DEFAULT_SOURCE
    if _DEFAULT_SOURCE is None:
        with _DEFAULT_SOURCE_LOCK:
            if _DEFAULT_SOURCE is None:
                requested = str(os.getenv("BORROW_SOURCE", "auto") or "auto").strip().lower()
                if requested in {"none", "null", "off", "disabled"}:
                    _DEFAULT_SOURCE = NullBorrowSource(reason=REASON_NO_SOURCE)
                else:
                    kis = KisBorrowSource()
                    file_source = FileBorrowSource()
                    candidate: BorrowDataSource | None = None
                    if requested == "file":
                        candidate = file_source if file_source.available() else None
                    elif requested == "kis":
                        candidate = kis if kis.available() else None
                    else:
                        candidate = (
                            kis if kis.available() else file_source if file_source.available() else None
                        )
                    if candidate is not None:
                        logger.info("borrow source: %s", candidate.status())
                        _DEFAULT_SOURCE = candidate
                    else:
                        reason = (
                            f"BORROW_SOURCE_{requested.upper()}_UNAVAILABLE"
                            if requested != "auto"
                            else file_source.status().get("reason") or REASON_NO_SOURCE
                        )
                        logger.info(
                            "borrow source unavailable (%s); short strategies stay inert",
                            reason,
                        )
                        _DEFAULT_SOURCE = NullBorrowSource(reason=str(reason))
    return _DEFAULT_SOURCE


def set_default_borrow_source(source: BorrowDataSource | None) -> None:
    """Install a source explicitly (tests, or a verified broker query)."""
    global _DEFAULT_SOURCE
    with _DEFAULT_SOURCE_LOCK:
        _DEFAULT_SOURCE = source


def reset_default_borrow_source() -> None:
    set_default_borrow_source(None)
