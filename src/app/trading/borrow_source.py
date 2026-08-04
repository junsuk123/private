"""Where 대주 availability actually comes from.

Why this is a pluggable source and not a KIS method
---------------------------------------------------
It was a KIS method. Three endpoints were guessed and all three were wrong, proven by
read-only probes against the live account on 2026-08-02:

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
* **The ladder is still exercisable.** :class:`FileBorrowSource` reads an
  operator-maintained file. For a retail account the 대주 list is checked on the
  broker's web UI anyway, so a file is an honest primary source, not a stopgap.

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

    Resolution order is deliberately short and explicit: a file if one is configured
    AND readable, otherwise nothing. There is no "try the broker and fall back",
    because a silent fallback is how an operator ends up believing a data path exists
    when it does not.
    """
    global _DEFAULT_SOURCE
    if _DEFAULT_SOURCE is None:
        with _DEFAULT_SOURCE_LOCK:
            if _DEFAULT_SOURCE is None:
                candidate = FileBorrowSource()
                if candidate.available():
                    logger.info("borrow source: %s", candidate.status())
                    _DEFAULT_SOURCE = candidate
                else:
                    logger.info(
                        "borrow source unavailable (%s); short strategies stay inert",
                        candidate.status().get("reason") or REASON_NO_SOURCE,
                    )
                    _DEFAULT_SOURCE = NullBorrowSource(
                        reason=candidate.status().get("reason") or REASON_NO_SOURCE
                    )
    return _DEFAULT_SOURCE


def set_default_borrow_source(source: BorrowDataSource | None) -> None:
    """Install a source explicitly (tests, or a verified broker query)."""
    global _DEFAULT_SOURCE
    with _DEFAULT_SOURCE_LOCK:
        _DEFAULT_SOURCE = source


def reset_default_borrow_source() -> None:
    set_default_borrow_source(None)
