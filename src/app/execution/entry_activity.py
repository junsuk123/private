"""Read-only count of this bot's confirmed cash-long entries.

The active coordinator writes LiveOrderJournal, not OrderStateMachine's SQLite
tables. A status timestamp is when the bot observed the fill; it is NOT a claim
about the exchange execution time or all trades in the brokerage account.
"""
from __future__ import annotations

from datetime import datetime, time as day_time, timezone
import json
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

from app.data.market_capabilities import normalize_market_group
from app.execution.entry_activity_projection import EntryActivityProjection, HistoryUnavailable


_Unavailable = HistoryUnavailable


def _timestamp(value):
    try:
        result = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        raise _Unavailable("ENTRY_ACTIVITY_TIMESTAMP_INVALID") from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise _Unavailable("ENTRY_ACTIVITY_TIMESTAMP_INVALID")
    return result.astimezone(timezone.utc)


class BotConfirmedEntryActivity:
    """Callable ``(market, now) -> int | None``, without network or file writes.

    Partial fills count once, on their first observed fill day. Later completion
    or cancellation never adds another entry. Existing corrupt or incomplete
    history fails closed; an absent journal with no rotations is a fresh install.
    Timestamp order is not guaranteed by the concurrent journal writer. The
    bounded, resumable scan covers retained history and joins amendment ancestry
    before assigning the first fill day. Subsequent reads process appended bytes
    only. Unchanged successful history is reused until local midnight or a
    previously future record becomes observable. The projection is memory-only.
    """
    scope = "bot_confirmed_entries"

    def __init__(self, journal_path: str | Path, *, maximum_bytes=64 * 1024 * 1024,
                 maximum_records=100_000, maximum_seconds=1.0, cache_seconds=1.0):
        self.path = Path(journal_path)
        self.maximum_bytes = max(1, int(maximum_bytes))
        self.maximum_records = max(1, int(maximum_records))
        self.maximum_seconds = max(.01, float(maximum_seconds))
        # Retained for constructor compatibility. Successful immutable-source
        # counts no longer expire merely because this interval elapsed.
        self.cache_seconds = max(0., float(cache_seconds))
        self._lock = threading.RLock()
        self._cache = {}
        self._next_future_at = None
        self._projection = EntryActivityProjection(_timestamp)
        self._history_observed = False
        self._latest = {"scope": self.scope, "available": False, "reason": "NOT_READ"}

    def snapshot(self):
        with self._lock:
            return dict(self._latest)

    def __call__(self, market: str, now: datetime) -> int | None:
        group = normalize_market_group(market)
        if group is None or not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            return self._finish(market, now, None, "ENTRY_ACTIVITY_SCOPE_OR_CLOCK_INVALID")
        region = group.value
        zone = ZoneInfo("Asia/Seoul" if region == "KR" else "America/New_York")
        moment = now.astimezone(timezone.utc)
        day = moment.astimezone(zone).date()
        start = datetime.combine(day, day_time.min, tzinfo=zone).astimezone(timezone.utc)
        with self._lock:
            try:
                paths, signature = self._paths()
                key = (region, day.isoformat(), signature)
                cached = self._cache.get(region)
                if (cached is not None and cached[0] == key and moment >= cached[1]
                        and cached[3] is not None
                        and (cached[5] is None or moment < cached[5])):
                    return self._finish(region, moment, cached[3], cached[4])
                self._next_future_at = None
                count = self._read(paths, region, start, moment) if paths else 0
                if self._paths()[1] != signature:
                    raise _Unavailable("ENTRY_ACTIVITY_HISTORY_CHANGED_DURING_READ")
                reason = "CONFIRMED_ENTRY_HISTORY" if paths else "NO_BOT_JOURNAL_YET"
            except (OSError, UnicodeError, json.JSONDecodeError, _Unavailable) as exc:
                count, reason = None, str(exc) if isinstance(exc, _Unavailable) else "ENTRY_ACTIVITY_HISTORY_UNREADABLE"
                key = None
            self._cache[region] = (key, moment, time.monotonic(), count, reason, self._next_future_at)
            return self._finish(region, moment, count, reason)

    def _finish(self, market, now, count, reason):
        with self._lock:
            self._latest = {"scope": self.scope, "market": str(market),
                "as_of": now.isoformat() if isinstance(now, datetime) else None,
                "available": count is not None, "trades_today": count, "reason": reason,
                "time_basis": "first_bot_observed_fill",
                "history_records_indexed": self._projection.sequence,
                "history_bytes_read": sum(cursor["offset"] for cursor in self._projection.files.values())}
        return count

    def _paths(self):
        rotations = []
        if self.path.parent.exists():
            for path in self.path.parent.glob(self.path.name + ".*"):
                suffix = path.name[len(self.path.name) + 1:]
                if suffix.isdigit():
                    rotations.append((int(suffix), path))
        rotations.sort()
        if rotations:
            self._history_observed = True
        if len(rotations) > 16 or any(number != index + 1 for index, (number, _) in enumerate(rotations)):
            raise _Unavailable("ENTRY_ACTIVITY_ROTATION_HISTORY_INCOMPLETE")
        if not self.path.exists():
            if rotations:
                self._history_observed = True
                raise _Unavailable("ENTRY_ACTIVITY_ROTATION_IN_PROGRESS")
            if self._history_observed:
                raise _Unavailable("ENTRY_ACTIVITY_HISTORY_MISSING")
            return (), ()
        self._history_observed = True
        paths = (self.path, *(path for _, path in rotations))
        signature = tuple((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
                          for path in paths for stat in (path.stat(),))
        return paths, signature

    def _read(self, paths, market, day_start, now):
        deadline = time.monotonic() + self.maximum_seconds
        signature = tuple((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
                          for path in paths for stat in (path.stat(),))
        complete = self._projection.update(
            paths, signature, maximum_bytes=self.maximum_bytes,
            maximum_records=self.maximum_records, deadline=deadline,
        )
        if not complete:
            raise _Unavailable("ENTRY_ACTIVITY_HISTORY_LOADING")
        count, self._next_future_at = self._projection.count(
            market, day_start, now, deadline=deadline,
        )
        return count
