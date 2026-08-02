"""Point-in-time borrow (대주) availability: the observation, and its journal.

Why a durable, timestamped store rather than a cached lookup
-----------------------------------------------------------
"Can I borrow this stock, how much, and at what fee" is the one input in the
short path whose *staleness* is unrecoverable. A stale price produces a bad fill;
a stale locate produces an order the broker rejects, or accepts and then force-
closes at whatever price it likes. So the answer is recorded as an observation
with a timestamp and a payload hash, and every consumer has to decide explicitly
whether that observation is fresh enough to act on.

It is also what makes forward shadow evaluation honest. When a short signal is
scored weeks later, the question is not "can stock be borrowed now" but "could it
have been borrowed *at the moment the signal fired*". Only a journal can answer
that. Re-querying at evaluation time is a look-ahead leak, and it is the exact
leak that would make a shadow track record unachievable live: recall waves are
correlated with the drops these strategies want to short.

Fail-closed rules encoded here
------------------------------
* A missing observation is NOT "available" — :meth:`BorrowSnapshotStore.latest`
  returns ``None`` and every caller treats that as no-locate.
* A missing fee is NOT a zero fee. ``borrow_fee_bps_annualised=None`` fails the cost gate.
* An observation timestamped AFTER the signal is refused, not clamped: it is
  future information, and silently accepting it is how a leak hides.
* Availability is never estimated, interpolated, or carried over from a related
  symbol. There is no "probably borrowable".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

DEFAULT_BORROW_STORE_PATH = "data/store/borrow-snapshots.sqlite3"

# Beyond this an observation is a memory. Borrow availability moves intraday and
# recall waves cluster exactly when these strategies want to be short, so the
# default is deliberately short.
DEFAULT_MAX_SNAPSHOT_AGE_SECONDS = 120.0

# UNITS. Borrow fee is carried in ANNUALISED basis points, because that is what a
# broker quotes (KIS 대주 rates are annual percentages). Everything downstream that
# needs a per-trade figure must pro-rate it through ``borrow_cost_bps``.
#
# This is spelled out because mixing the two units is not a rounding error, it is
# a factor of ~10,000 in either direction. Comparing an annualised 800bps rate
# against a per-trade 40bps ceiling rejects every borrowable name; pro-rating a
# per-trade figure as if annual prices an 8%/yr borrow at 0.0009bps, i.e. free.
# The first failure is visible (nothing trades); the second is invisible and
# accepts negative-expectancy shorts. Field names carry the unit for that reason.
DEFAULT_MAX_BORROW_FEE_BPS_ANNUALISED = 1500.0  # 15%/yr

# Days per year used to pro-rate. Calendar, not trading: borrow accrues on days the
# market is shut.
_DAYS_PER_YEAR = 365.0


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class BorrowSnapshot:
    """One broker answer to "can I borrow this, how much, at what fee".

    ``available`` is a tri-state through the constructor's callers: a snapshot that
    does not exist is represented by the ABSENCE of a snapshot, never by
    ``available=False`` with fabricated quantities. That keeps "the broker said no"
    distinguishable from "we never asked", which matters because the second is an
    operational fault worth alerting on and the first is normal.
    """

    symbol: str
    observed_at: datetime
    available: bool
    available_quantity: int | None = None
    borrow_fee_bps_annualised: float | None = None
    return_deadline: datetime | None = None
    source: str = "kis"
    source_payload_hash: str = ""
    snapshot_id: str = ""
    # Why the broker refused, verbatim where available. Kept for the
    # borrow_rejection_rate metric and for operator diagnosis.
    reject_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol or "").strip().upper())
        object.__setattr__(self, "observed_at", _aware(self.observed_at))
        if self.return_deadline is not None:
            object.__setattr__(self, "return_deadline", _aware(self.return_deadline))
        if not self.snapshot_id:
            object.__setattr__(self, "snapshot_id", f"borrow-{uuid4().hex}")

    def age_seconds(self, now: datetime) -> float:
        """Seconds between the observation and ``now``. Negative == future."""
        return (_aware(now) - self.observed_at).total_seconds()

    def is_fresh(
        self, now: datetime, max_age_seconds: float = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS
    ) -> bool:
        """Fresh means recent AND not from the future.

        The lower bound is the anti-leak check. An observation timestamped after the
        decision moment is future information; it is refused rather than treated as
        "very fresh", because clamping it to zero age is precisely how a
        look-ahead leak passes a freshness test.
        """
        age = self.age_seconds(now)
        return 0.0 <= age <= max(0.0, float(max_age_seconds))

    def covers_quantity(self, quantity: int) -> bool:
        if not self.available or quantity <= 0:
            return False
        # Unknown quantity does not cover any quantity. "Available: yes, amount:
        # unknown" is not a locate you can size against.
        return self.available_quantity is not None and self.available_quantity >= quantity

    def hours_to_deadline(self, now: datetime) -> float | None:
        if self.return_deadline is None:
            return None
        return (self.return_deadline - _aware(now)).total_seconds() / 3600.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "observed_at": self.observed_at.isoformat(),
            "available": self.available,
            "available_quantity": self.available_quantity,
            "borrow_fee_bps_annualised": self.borrow_fee_bps_annualised,
            "return_deadline": (
                self.return_deadline.isoformat() if self.return_deadline else None
            ),
            "source": self.source,
            "source_payload_hash": self.source_payload_hash,
            "reject_reason": self.reject_reason,
        }

    @staticmethod
    def payload_hash(payload: Mapping[str, Any] | str | None) -> str:
        """Stable hash of the raw broker payload.

        Recorded so a disputed locate can be traced back to the exact response that
        produced it, without storing (possibly large, possibly sensitive) payloads
        in the trading store.
        """
        if payload is None:
            return ""
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class BorrowVerdict:
    """Whether a specific short order may proceed on borrow grounds."""

    allowed: bool
    snapshot: BorrowSnapshot | None
    reason_codes: tuple[str, ...] = ()
    borrow_fee_bps_annualised: float | None = None
    available_quantity: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "borrow_fee_bps_annualised": self.borrow_fee_bps_annualised,
            "available_quantity": self.available_quantity,
            "snapshot": self.snapshot.as_dict() if self.snapshot else None,
        }


def evaluate_borrow(
    snapshot: BorrowSnapshot | None,
    *,
    quantity: int,
    now: datetime,
    max_fee_bps_annualised: float = DEFAULT_MAX_BORROW_FEE_BPS_ANNUALISED,
    max_age_seconds: float = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    min_hours_before_deadline: float | None = None,
) -> BorrowVerdict:
    """The one borrow admissibility rule, used by preflight and by the simulator.

    Shared deliberately: if the shadow simulator and the live preflight applied
    different borrow rules, shadow results would not predict live executability,
    and the whole promotion ladder would be measuring the wrong thing.

    ``max_fee_bps_annualised`` is an ANNUALISED ceiling, matching the snapshot's
    unit. It answers "is this name too expensive/crowded to short at all", which is
    a rate question. Whether a *specific* trade clears its own pro-rated borrow cost
    is a different question, and it belongs to the ProfitabilityGate.
    """
    from app.trading.directional import ShortReasonCodes as R

    if snapshot is None:
        return BorrowVerdict(False, None, (R.BORROW_LOOKUP_FAILED,))
    if not snapshot.is_fresh(now, max_age_seconds):
        return BorrowVerdict(
            False,
            snapshot,
            (R.BORROW_SNAPSHOT_STALE,),
            borrow_fee_bps_annualised=snapshot.borrow_fee_bps_annualised,
            available_quantity=snapshot.available_quantity,
        )
    if not snapshot.available:
        return BorrowVerdict(
            False,
            snapshot,
            (R.BORROW_UNAVAILABLE,),
            available_quantity=snapshot.available_quantity,
        )
    if not snapshot.covers_quantity(quantity):
        return BorrowVerdict(
            False,
            snapshot,
            (R.BORROW_QUANTITY_INSUFFICIENT,),
            borrow_fee_bps_annualised=snapshot.borrow_fee_bps_annualised,
            available_quantity=snapshot.available_quantity,
        )
    fee = snapshot.borrow_fee_bps_annualised
    if fee is None or fee > max_fee_bps_annualised:
        # Unknown fee lands here on purpose: an unpriced borrow cannot be shown to
        # clear its own cost, and pricing it at zero is how a negative-expectancy
        # short passes a cost gate.
        return BorrowVerdict(
            False,
            snapshot,
            (R.BORROW_COST_TOO_HIGH,),
            borrow_fee_bps_annualised=fee,
            available_quantity=snapshot.available_quantity,
        )
    if min_hours_before_deadline is not None:
        remaining = snapshot.hours_to_deadline(now)
        if remaining is not None and remaining < min_hours_before_deadline:
            return BorrowVerdict(
                False,
                snapshot,
                (R.RECALL_DEADLINE_NEAR,),
                borrow_fee_bps_annualised=fee,
                available_quantity=snapshot.available_quantity,
            )
    return BorrowVerdict(
        True,
        snapshot,
        (),
        borrow_fee_bps_annualised=fee,
        available_quantity=snapshot.available_quantity,
    )


def borrow_cost_bps(
    fee_bps_annualised: float | None, holding_seconds: float
) -> float | None:
    """Pro-rate an ANNUALISED borrow rate into the bps actually accrued.

    The unit conversion that everything short-side depends on. The broker quotes a
    yearly rate; the position is held for minutes. Both shortcuts are wrong by
    orders of magnitude in opposite directions — charging the annual figure directly
    rejects every borrowable name, and charging zero prices an 8%/yr borrow as free
    and lets negative-expectancy shorts through a cost gate.

    365-day basis: borrow accrues on calendar days, including days the market is
    shut, so a Friday-afternoon short pays for the weekend.

    Returns ``None`` for an unknown rate so callers must decide explicitly rather
    than silently receive a fabricated 0.0.
    """
    if fee_bps_annualised is None:
        return None
    rate = max(0.0, float(fee_bps_annualised))
    seconds = max(0.0, float(holding_seconds))
    return rate * seconds / (_DAYS_PER_YEAR * 24.0 * 3600.0)


class BorrowSnapshotStore:
    """Append-only journal of borrow observations.

    Append-only rather than a key/value cache because the historical series IS the
    product: ``borrow_availability_rate`` over the promotion window, and the
    point-in-time locate that a forward evaluation has to consult, are both queries
    over history. A cache that overwrites the previous answer can support neither.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path or os.getenv("BORROW_SNAPSHOT_STORE_PATH", DEFAULT_BORROW_STORE_PATH)
        )
        self._lock = threading.RLock()
        self._available = True
        self._migrate()

    # -- writes ------------------------------------------------------------- #
    def record(self, snapshot: BorrowSnapshot) -> bool:
        if not self._available or not snapshot.symbol:
            return False
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    """
                    insert or replace into borrow_snapshots(
                        snapshot_id, symbol, observed_at, available, available_quantity,
                        borrow_fee_bps_annualised, return_deadline, source, source_payload_hash,
                        reject_reason
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.symbol,
                        snapshot.observed_at.isoformat(),
                        int(bool(snapshot.available)),
                        snapshot.available_quantity,
                        _finite(snapshot.borrow_fee_bps_annualised),
                        (
                            snapshot.return_deadline.isoformat()
                            if snapshot.return_deadline
                            else None
                        ),
                        snapshot.source,
                        snapshot.source_payload_hash,
                        snapshot.reject_reason,
                    ),
                )
                conn.commit()
        except sqlite3.Error:
            return False
        return True

    # -- reads -------------------------------------------------------------- #
    def latest(self, symbol: str, *, as_of: datetime | None = None) -> BorrowSnapshot | None:
        """Most recent observation at or before ``as_of``.

        ``as_of`` is the anti-leak parameter and the reason this is not a simple
        "latest" lookup. A forward evaluation of a signal from 14:03 must see the
        borrow world as of 14:03; handing it today's locate would let it short names
        that were unborrowable at the time, which is exactly the population these
        strategies target.
        """
        moment = _aware(as_of) if as_of is not None else datetime.now(timezone.utc)
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    select snapshot_id, symbol, observed_at, available, available_quantity,
                           borrow_fee_bps_annualised, return_deadline, source, source_payload_hash,
                           reject_reason
                    from borrow_snapshots
                    where symbol = ? and observed_at <= ?
                    order by observed_at desc, rowid desc limit 1
                    """,
                    (str(symbol or "").strip().upper(), moment.isoformat()),
                ).fetchone()
        except sqlite3.Error:
            return None
        return _snapshot_from_row(row) if row else None

    def history(
        self,
        symbol: str | None = None,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> tuple[BorrowSnapshot, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(str(symbol).strip().upper())
        if since is not None:
            clauses.append("observed_at >= ?")
            params.append(_aware(since).isoformat())
        where = f"where {' and '.join(clauses)} " if clauses else ""
        params.append(max(1, int(limit)))
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    "select snapshot_id, symbol, observed_at, available, available_quantity, "
                    "borrow_fee_bps_annualised, return_deadline, source, source_payload_hash, reject_reason "
                    f"from borrow_snapshots {where}order by observed_at desc, rowid desc limit ?",
                    params,
                ).fetchall()
        except sqlite3.Error:
            return ()
        return tuple(_snapshot_from_row(row) for row in rows)

    def health(self, *, window_seconds: float = 3600.0) -> dict[str, Any]:
        """Borrow-desk health over a recent window, for the promotion controller.

        ``availability_rate`` and ``rejection_rate`` are complements over ANSWERED
        lookups. ``lookup_count == 0`` reports rates as ``None`` rather than 0.0,
        because "we asked nothing" must not read as "nothing was available" — that
        would demote a strategy for an outage in the polling loop.
        """
        since = datetime.now(timezone.utc) - timedelta(seconds=max(1.0, window_seconds))
        snapshots = self.history(since=since, limit=5000)
        total = len(snapshots)
        available = sum(1 for item in snapshots if item.available)
        fees = [item.borrow_fee_bps_annualised for item in snapshots if item.borrow_fee_bps_annualised is not None]
        return {
            "store_path": str(self.path),
            "store_available": self._available,
            "window_seconds": window_seconds,
            "lookup_count": total,
            "distinct_symbols": len({item.symbol for item in snapshots}),
            "availability_rate": (available / total) if total else None,
            "rejection_rate": ((total - available) / total) if total else None,
            "mean_borrow_fee_bps_annualised": (sum(fees) / len(fees)) if fees else None,
            "max_borrow_fee_bps_annualised": max(fees) if fees else None,
            "last_observed_at": snapshots[0].observed_at.isoformat() if snapshots else None,
            "reject_reasons": sorted(
                {item.reject_reason for item in snapshots if item.reject_reason}
            )[:10],
        }

    def availability_rate(
        self, symbols: Sequence[str] | None = None, *, since: datetime | None = None
    ) -> float | None:
        snapshots = self.history(since=since, limit=5000)
        if symbols:
            wanted = {str(item).strip().upper() for item in symbols}
            snapshots = tuple(item for item in snapshots if item.symbol in wanted)
        if not snapshots:
            return None
        return sum(1 for item in snapshots if item.available) / len(snapshots)

    def prune(self, *, keep_rows: int = 200_000) -> int:
        try:
            with self._lock, closing(self._connect()) as conn:
                cursor = conn.execute(
                    """
                    delete from borrow_snapshots where rowid not in (
                        select rowid from borrow_snapshots
                        order by observed_at desc, rowid desc limit ?
                    )
                    """,
                    (max(1000, int(keep_rows)),),
                )
                conn.commit()
                return int(cursor.rowcount or 0)
        except sqlite3.Error:
            return 0

    # -- internals ---------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("pragma journal_mode = wal")
        return conn

    def _migrate(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, closing(self._connect()) as conn:
                conn.executescript(
                    """
                    create table if not exists borrow_snapshots (
                        snapshot_id text primary key,
                        symbol text not null,
                        observed_at text not null,
                        available integer not null,
                        available_quantity integer,
                        borrow_fee_bps_annualised real,
                        return_deadline text,
                        source text,
                        source_payload_hash text,
                        reject_reason text
                    );
                    create index if not exists idx_borrow_symbol_time
                        on borrow_snapshots(symbol, observed_at desc);
                    create index if not exists idx_borrow_time
                        on borrow_snapshots(observed_at desc);
                    """
                )
                conn.commit()
        except (OSError, sqlite3.Error):
            # An unusable store must not crash the engine, but it MUST NOT read as
            # "borrow available" either: every ``latest`` then returns None, which
            # every consumer treats as no-locate. Shorting stops; longs are
            # unaffected.
            self._available = False


def _snapshot_from_row(row: Sequence[Any]) -> BorrowSnapshot:
    return BorrowSnapshot(
        snapshot_id=str(row[0]),
        symbol=str(row[1]),
        observed_at=_parse_iso(row[2]) or datetime.now(timezone.utc),
        available=bool(row[3]),
        available_quantity=None if row[4] is None else int(row[4]),
        borrow_fee_bps_annualised=_finite(row[5]),
        return_deadline=_parse_iso(row[6]),
        source=str(row[7] or ""),
        source_payload_hash=str(row[8] or ""),
        reject_reason=str(row[9] or ""),
    )


_DEFAULT_STORE: BorrowSnapshotStore | None = None
_DEFAULT_STORE_LOCK = threading.Lock()


def default_borrow_store() -> BorrowSnapshotStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _DEFAULT_STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = BorrowSnapshotStore()
    return _DEFAULT_STORE


def reset_default_borrow_store() -> None:
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = None
