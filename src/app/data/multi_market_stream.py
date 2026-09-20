"""Fair KR/US subscription planning for one account's persistent KIS socket.

The websocket already parses both protocols. Multiplexing changes registrations,
not quote provenance or order permissions. No broker or database calls belong here.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import datetime


def market_group(symbol: str) -> str:
    text = str(symbol).strip().upper().split(".")[0]
    return "KR" if len(text) == 6 and text[:1].isdigit() else "US"


def fair_market_symbols(symbols: Iterable[str], limit: int, *, first: str = "KR") -> tuple[str, ...]:
    """Preserve each market's rank while preventing one list starving the other."""
    queues: dict[str, deque[str]] = {"KR": deque(), "US": deque()}
    seen: set[str] = set()
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            queues[market_group(symbol)].append(symbol)
    order = ("US", "KR") if first == "US" else ("KR", "US")
    selected: list[str] = []
    while len(selected) < max(0, limit) and any(queues.values()):
        for group in order:
            if queues[group] and len(selected) < limit:
                selected.append(queues[group].popleft())
    return tuple(selected)


def subscription_symbols(
    domestic: Iterable[str], overseas: Iterable[str], *,
    held: Iterable[str] = (), max_subscriptions: int = 40,
) -> tuple[str, ...]:
    """Two registrations per symbol, holdings first, with one shared hard budget."""
    candidates = tuple(dict.fromkeys(
        str(s).strip().upper() for s in (*tuple(domestic), *tuple(overseas)) if str(s).strip()
    ))
    available = set(candidates)
    pinned = tuple(s for s in dict.fromkeys(str(x).strip().upper() for x in held) if s in available)
    budget = max(0, int(max_subscriptions)) // 2
    prefix = fair_market_symbols(pinned, budget)
    return prefix + fair_market_symbols((s for s in candidates if s not in prefix), budget - len(prefix))


def subscription_tr_ids(symbol: str, *, now: datetime | None = None) -> tuple[str, ...]:
    from app.data.kis_realtime import _domestic_subscription_tr_ids
    from app.data.market_session import MarketPhase, market_phase
    import os

    if market_group(symbol) == "US":
        return ("HDFSCNT0", "HDFSASP0")
    if not os.getenv("KIS_REALTIME_FEED", "").strip() and market_phase("KRX", now) is MarketPhase.REGULAR:
        return ("H0STCNT0", "H0STASP0")
    return _domestic_subscription_tr_ids()


def subscription_key(symbol: str) -> str:
    from app.data.kis_realtime import overseas_realtime_subscription_key

    return overseas_realtime_subscription_key(symbol) if market_group(symbol) == "US" else symbol
