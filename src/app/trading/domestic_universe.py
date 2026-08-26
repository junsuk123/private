from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


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
    #: Listed names for the chosen symbols, persisted so a restart can re-apply the
    #: instrument filter without a live ranking fetch. Without this the session-locked
    #: path has no name to classify by and an untradable incumbent survives the day.
    names: dict[str, str] = field(default_factory=dict)
    #: Instruments removed because the account may not trade them, as
    #: ``InstrumentVerdict.as_dict()`` rows. Reported rather than silently dropped.
    excluded: tuple[dict[str, Any], ...] = ()


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
                    # Names for the chosen symbols only, so the file does not grow
                    # into a rolling cache of everything ever ranked.
                    "names": {
                        symbol: decision.names[symbol]
                        for symbol in decision.symbols
                        if symbol in decision.names
                    },
                    "excluded": list(decision.excluded),
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
    names: Mapping[str, str] | None = None,
    derivatives_allowed: bool = False,
    etf_allowed: bool = False,
    leverage_etf_allowed: bool = False,
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

    Instrument eligibility is applied to the INCUMBENTS as well as to the fresh
    ranking, and that is the load-bearing half. Session locking and hysteresis both
    exist to keep a name once it is chosen, so an untradable instrument that got in
    before the filter existed would otherwise hold its slot for the rest of the
    session — which is exactly the state the 2026-08-11 universe was found in, six
    leveraged ETPs deep. Filtering only the fresh ranking would have left it there.
    """
    from app.data.instrument_eligibility import filter_tradable

    session = _today(now)
    state = state if state is not None else {}
    # Names from this refresh win; the persisted ones cover incumbents that are no
    # longer in the ranking and would otherwise be unclassifiable.
    resolved_names: dict[str, str] = {
        str(k).strip().upper(): str(v)
        for k, v in (state.get("names") or {}).items()
        if str(v or "").strip()
    }
    resolved_names.update(
        {
            str(k).strip().upper(): str(v)
            for k, v in (names or {}).items()
            if str(v or "").strip()
        }
    )

    def _permitted(symbols: Sequence[str]) -> tuple[tuple[str, ...], tuple[Any, ...]]:
        return filter_tradable(
            symbols,
            resolved_names,
            market="KR",
            derivatives_allowed=derivatives_allowed,
            etf_allowed=etf_allowed,
            leverage_etf_allowed=leverage_etf_allowed,
        )

    previous_raw = tuple(str(s) for s in (state.get("symbols") or ()))
    previous, previous_excluded = _permitted(previous_raw)
    ranked_raw = tuple(dict.fromkeys(str(s).strip() for s in ranked_symbols if str(s).strip()))
    ranked, ranked_excluded = _permitted(ranked_raw)
    excluded_rows = tuple(
        verdict.as_dict()
        for verdict in {
            verdict.symbol: verdict for verdict in (*previous_excluded, *ranked_excluded)
        }.values()
    )

    purged = tuple(s for s in previous_raw if s not in set(previous))

    if not ranked:
        if previous:
            return UniverseDecision(
                previous,
                str(state.get("session_date") or session),
                "carried_over",
                retained=previous,
                dropped=purged,
                note="ranking unavailable; kept previous universe for its history",
                names=resolved_names,
                excluded=excluded_rows,
            )
        return UniverseDecision(
            (), session, "empty",
            dropped=purged,
            note="no ranking and no prior universe",
            names=resolved_names,
            excluded=excluded_rows,
        )

    if previous and str(state.get("session_date") or "") == session:
        # Same session: the universe is already decided. Re-picking mid-session is
        # exactly the behaviour being removed.
        #
        # One exception, and only one: slots vacated by an instrument the account
        # cannot trade. Those slots were never going to produce a candidate, so
        # refilling them is not the churn this module fights — it is recovering
        # capacity that was only ever nominal. The refill is written back
        # immediately, so the added names become incumbents and the next refresh
        # takes the plain locked path instead of topping up again.
        if purged and len(previous) < size:
            backfilled = list(previous)
            for symbol in ranked:
                if len(backfilled) >= size:
                    break
                if symbol not in backfilled:
                    backfilled.append(symbol)
            if len(backfilled) > len(previous):
                return UniverseDecision(
                    tuple(backfilled),
                    session,
                    "session_locked",
                    added=tuple(s for s in backfilled if s not in set(previous)),
                    dropped=purged,
                    retained=previous,
                    note=(
                        f"purged {len(purged)} untradable instrument(s) and refilled "
                        f"the vacated slots for session {session}"
                    ),
                    names=resolved_names,
                    excluded=excluded_rows,
                )
        return UniverseDecision(
            previous,
            session,
            "session_locked",
            retained=previous,
            dropped=purged,
            note="universe already chosen for this session",
            names=resolved_names,
            excluded=excluded_rows,
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
        dropped=tuple(dict.fromkeys((*purged, *(s for s in previous if s not in selected_set)))),
        retained=tuple(s for s in selected if s in previous_set),
        note=f"held for session {session}",
        names=resolved_names,
        excluded=excluded_rows,
    )


def universe_size(default: int | None = None) -> int:
    """How many KRX names to hold for a session.

    ``default`` lets the caller supply a value derived from what the realtime
    subscription budget can actually feed with DEPTH. That matters because a KRX
    name without a fresh orderbook cannot produce a LiveFeatureFrame at all — six of
    the model's inputs are depth-derived, and ``live_feature_frame`` raises
    MISSING_SOURCE_RECORDS without one — so a universe larger than the depth budget
    is over-committed by construction. Its surplus members hold slots in the
    evaluated set while being structurally unable to produce a candidate.

    Measured 2026-08-11 with the bare 30: 7 of 23 members were not subscribed at all
    and 12 had received no orderbook all day, while 363 non-universe symbols had.

    ``REALTIME_KRX_UNIVERSE_SIZE`` still wins when set, so an operator can override.
    """
    fallback = DEFAULT_UNIVERSE_SIZE if default is None else max(1, int(default))
    try:
        return max(1, int(float(os.getenv("REALTIME_KRX_UNIVERSE_SIZE", str(fallback)))))
    except ValueError:
        return fallback
