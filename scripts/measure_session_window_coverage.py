"""Count KR symbol-days that carry BOTH session-structure windows.

``market_intraday_momentum`` (and its short twin) compare the 09:00-09:30 window
with 14:50-15:20 **for the same symbol on the same day**. The collector rotates
its subscription pool during the session, so a symbol observed at the open is
usually gone by the close unless it is pinned as a session anchor.

That is the whole reason both strategies read
``STRUCTURALLY_UNREACHABLE:CONTEXT_UNAVAILABLE:FIRST_HALF_HOUR`` in every
strategy-utility checkpoint: the label builder never finds a symbol-day with
both ends, so neither strategy can produce a single training row.

Measured 2026-08-08 over the stored 40 days, with
``REALTIME_SESSION_ANCHOR_MAX=2``::

    09:00-09:30 present : 182 symbol-days
    14:50-15:20 present : 303 symbol-days
    BOTH                :  49 symbol-days
    2026-08-06          : open30=0
    2026-08-07          : open30=28  close30=73  both=0

``run.ps1`` was pinning the anchor count to 2 while the code default is 8. Run
this after a trading session to check whether raising it actually widened the
overlap: the ``both`` column for recent dates is the number that matters, and it
must be non-zero for the session-structure strategies to be evaluable at all.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# KRX continuous trading is 09:00-15:30 KST. The closing window deliberately
# stops at 15:20 rather than 15:30: the last ten minutes are the closing
# auction, and entering into that mechanism is not modelled anywhere here.
KST = timezone(timedelta(hours=9))
SESSION_OPEN_MINUTES = 9 * 60
SESSION_CLOSE_MINUTES = 15 * 60 + 30
FIRST_HALF_HOUR = (9 * 60, 9 * 60 + 30)
LAST_CONTINUOUS_HALF_HOUR = (14 * 60 + 50, 15 * 60 + 20)


def _minutes_of_day(raw: object) -> tuple[str, int] | None:
    """``(KST date, minute-of-day)`` for a stored ``minute_start``."""
    try:
        moment = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    local = moment.astimezone(KST)
    return local.date().isoformat(), local.hour * 60 + local.minute


def measure(database: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "select symbol, minute_start, market_group from realtime_minute_bars"
        ).fetchall()

    opened: set[tuple[str, str]] = set()
    closed: set[tuple[str, str]] = set()
    for symbol, minute_start, market_group in rows:
        # Market is taken from stored metadata, never guessed from the ticker
        # shape. Legacy v1 rows carry an empty market_group, so they are kept
        # and filtered by the session clock instead of being discarded.
        if (market_group or "") not in ("KR", "KRX", ""):
            continue
        parsed = _minutes_of_day(minute_start)
        if parsed is None:
            continue
        day, minutes = parsed
        if not SESSION_OPEN_MINUTES <= minutes <= SESSION_CLOSE_MINUTES:
            continue
        key = (str(symbol), day)
        if FIRST_HALF_HOUR[0] <= minutes < FIRST_HALF_HOUR[1]:
            opened.add(key)
        if LAST_CONTINUOUS_HALF_HOUR[0] <= minutes < LAST_CONTINUOUS_HALF_HOUR[1]:
            closed.add(key)

    both = opened & closed
    per_day: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for _, day in opened:
        per_day[day][0] += 1
    for _, day in closed:
        per_day[day][1] += 1
    for _, day in both:
        per_day[day][2] += 1

    return {
        "open_window_symbol_days": len(opened),
        "close_window_symbol_days": len(closed),
        "both_windows_symbol_days": len(both),
        "per_day": {day: tuple(counts) for day, counts in sorted(per_day.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/store/realtime_market_data.sqlite3"),
    )
    parser.add_argument(
        "--trailing-days",
        type=int,
        default=5,
        help="How many trailing dates to print as context under the verdict.",
    )
    args = parser.parse_args()

    report = measure(args.database)
    print(f"09:00-09:30 present : {report['open_window_symbol_days']} symbol-days")
    print(f"14:50-15:20 present : {report['close_window_symbol_days']} symbol-days")
    print(f"BOTH windows        : {report['both_windows_symbol_days']} symbol-days")
    print()
    print("date          open30  close30    both")
    per_day: dict[str, tuple[int, int, int]] = report["per_day"]  # type: ignore[assignment]
    for day, (open30, close30, both) in per_day.items():
        print(f"  {day}  {open30:>6}  {close30:>7}  {both:>6}")

    # The verdict is the MOST RECENT trading date alone, never a trailing sum.
    # A sum passes on stale history: measured 2026-08-08 the last five dates
    # totalled 14 both-window symbol-days and would have read "OK", while the two
    # most recent sessions were 0 and 0 -- exactly the starvation being checked
    # for. What matters is whether the session that just ended captured both ends.
    if not per_day:
        print()
        print("NO DATA: the store holds no KR session minute bars at all.")
        return

    latest_day, (latest_open, latest_close, latest_both) = list(per_day.items())[-1]
    trailing = list(per_day.items())[-max(1, args.trailing_days) :]

    print()
    print(f"latest session: {latest_day}")
    if latest_both:
        print(
            f"OK: {latest_both} symbol-days on {latest_day} carry both windows, so "
            "the session-structure strategies are evaluable for that session."
        )
    else:
        # Silence would read as success, so name the failure and where to look.
        print(
            f"STARVED: no symbol-day on {latest_day} carries both windows "
            f"(open30={latest_open}, close30={latest_close}). "
            "market_intraday_momentum stays STRUCTURALLY_UNREACHABLE. Check "
            "REALTIME_SESSION_ANCHOR_MAX (run.ps1 must not pin it below the code "
            "default of 8) and that the server was restarted BEFORE the open -- an "
            "anchor set chosen after 09:30 cannot recover the opening window."
        )
    print(
        "  trailing: "
        + ", ".join(f"{day}={counts[2]}" for day, counts in trailing)
    )


if __name__ == "__main__":
    main()
