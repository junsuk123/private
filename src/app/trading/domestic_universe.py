from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence


#: Where the chosen universe lives so a restart does not reshuffle it.
DEFAULT_STATE_PATH = Path("data/runtime/domestic_universe.json")

#: How many KRX names to hold for a session. KIS returns at most 30 rows per
#: ranking endpoint, and 30 names at ~390 bars/day is ~11,700 bars per day —
#: against the 15,190 bars the churning selection produced across an entire week.
DEFAULT_UNIVERSE_SIZE = 30

#: An incumbent is only dropped once it falls this far past the cut. Without the
#: band a name oscillating around rank 30 is added and removed on alternating
#: refreshes, which is the churn this module exists to stop, just slower.
DEFAULT_HYSTERESIS_MULTIPLIER = 2.0


@dataclass(frozen=True)
class UniverseDecision:
    symbols: tuple[str, ...]
    session_date: str
    source: str
    added: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()
    note: str = ""


def _today(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).date().isoformat()


def load_state(path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def save_state(decision: UniverseDecision, path: Path = DEFAULT_STATE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "session_date": decision.session_date,
                    "symbols": list(decision.symbols),
                    "source": decision.source,
                    "note": decision.note,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        # Persistence is an optimisation. A universe that cannot be written is
        # still a valid universe for this process.
        pass


def resolve_universe(
    ranked_symbols: Sequence[str],
    *,
    now: datetime | None = None,
    state: dict[str, Any] | None = None,
    size: int = DEFAULT_UNIVERSE_SIZE,
    hysteresis_multiplier: float = DEFAULT_HYSTERESIS_MULTIPLIER,
) -> UniverseDecision:
    """Hold one KRX universe for the whole session instead of re-picking it.

    The selection this replaces refreshed every 30 seconds from the volume,
    fluctuation and volume-power rankings — that is, from whatever was spiking at
    that moment. Measured over 2026-07-28..08-07: day-over-day membership survival
    was 4.7-15%, 960 distinct names appeared, and exactly ONE of them (005930)
    recurred often enough to build history. 005930 averaged 2,097 minute bars; the
    other 959 averaged 16.6, against the 20 a symbol needs before it can even be
    considered. So the engine held data on 870 names and could evaluate almost none
    of them, which is why days passed with zero candidates.

    Depth is what strategies actually need, and depth is only bought with
    stability. Three choices follow from that:

    * pick once per session, not per refresh — a name has to survive long enough to
      accumulate the bars it will be judged on;
    * rank by turnover rather than by fluctuation or volume-power — the movers
      lists are the least stable inputs available AND select wide-spread names,
      which is the wrong direction when a KRX round trip already costs ~28bps;
    * apply hysteresis so a name near the boundary is not traded in and out.

    ``ranked_symbols`` is expected most-liquid-first. Returns the previous session's
    set unchanged when the ranking is unavailable: a stale universe still has depth,
    an empty one has nothing.
    """
    session = _today(now)
    state = state if state is not None else {}
    previous = tuple(str(s) for s in (state.get("symbols") or ()))
    ranked = tuple(dict.fromkeys(str(s).strip() for s in ranked_symbols if str(s).strip()))

    if not ranked:
        if previous:
            return UniverseDecision(
                previous,
                str(state.get("session_date") or session),
                "carried_over",
                retained=previous,
                note="ranking unavailable; kept previous universe for its history",
            )
        return UniverseDecision((), session, "empty", note="no ranking and no prior universe")

    if previous and str(state.get("session_date") or "") == session:
        # Same session: the universe is already decided. Re-picking mid-session is
        # exactly the behaviour being removed.
        return UniverseDecision(
            previous,
            session,
            "session_locked",
            retained=previous,
            note="universe already chosen for this session",
        )

    head = list(ranked[:size])
    if previous:
        # Hysteresis: an incumbent still inside the widened band keeps its slot even
        # if it slipped out of the strict cut, because its accumulated history is
        # worth more than a marginal ranking difference.
        band = ranked[: max(size, int(size * hysteresis_multiplier))]
        survivors = [s for s in previous if s in band]
        chosen: list[str] = []
        for symbol in survivors:
            if symbol not in chosen:
                chosen.append(symbol)
        for symbol in head:
            if len(chosen) >= size:
                break
            if symbol not in chosen:
                chosen.append(symbol)
        head = chosen[:size]

    selected = tuple(head)
    previous_set, selected_set = set(previous), set(selected)
    return UniverseDecision(
        selected,
        session,
        "reselected",
        added=tuple(s for s in selected if s not in previous_set),
        dropped=tuple(s for s in previous if s not in selected_set),
        retained=tuple(s for s in selected if s in previous_set),
        note=f"held for session {session}",
    )


def universe_size() -> int:
    try:
        return max(1, int(float(os.getenv("REALTIME_KRX_UNIVERSE_SIZE", str(DEFAULT_UNIVERSE_SIZE)))))
    except ValueError:
        return DEFAULT_UNIVERSE_SIZE
