"""One authority for "what does a round trip on this symbol actually cost".

The defect this fixes
---------------------
Three layers priced the same round trip three different ways, and each one was
light in a way the layer below could not see:

* the algorithm TRIGGER floor (``strategy_algorithms.round_trip_cost_bps``) read the
  fee policy alone -- 33.8bps on KRX, 51.2 on US;
* the session ELECTION read a configured 28bps constant for every KR symbol;
* the training LABELS read the fee policy floored by the realized tape -- 51.9 on
  KRX, 73.6 on US -- which is the only one of the three that was right.

``config/profitability_policy.yaml`` sets ``spread_rate`` to 0 on both venues,
deliberately: the spread belongs to the symbol and the moment, not to the fee
schedule. Nothing put it back, so a trigger fired on a 40bps edge that the labels
scored as a loss and the executor paid 53bps to take.

How the number is built
-----------------------
Two independent lower bounds on the truth, and the answer is whichever is larger:

``policy + this symbol's measured spread``
    Symbol-specific and point-in-time. A buy crosses the spread to get in and the
    sell crosses it to get out, which is one full spread over the round trip --
    charged once, matching the "spread and impact are charged once each" contract
    the fee policy is written against. This is the sharper estimate when a spread
    reading exists, and it distinguishes a 6.5bps name from a 24.7bps one.

``the venue's realized round trip``
    Market-level and blunt, but it is what the tape actually charged: the p75 of
    resolved shadow round trips. It covers the symbol with no spread reading, and
    it covers everything the spread does not -- queue position, partial fills,
    impact.

Taking the max double-counts nothing (the two are alternatives, not addends) and
can only move the estimate toward the number the executor pays.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

#: Percentile of the realized tape used as the venue's round trip. The p75 rather
#: than the median for the reason the labelling pipeline already records: both
#: terms of a coverage test are estimates, so equality with cost is a loss, and the
#: conservative tail is the one that does not get the executor into a trade it
#: cannot pay for.
REALIZED_COST_PERCENTILE = 0.75

#: Resolved round trips a venue needs before its tape is allowed to speak.
MINIMUM_RESOLVED_ROUND_TRIPS = 30

#: The tape moves slowly, and this runs in a per-tick entry path.
_MEASURED_TTL_SECONDS = 900.0

_KRX_SYMBOL_LENGTH = 6

_policy_cache: dict[tuple[str, str], float] = {}
_measured_cache: dict[str, tuple[float, float | None]] = {}
_cost_engine: Any = None


def resolve_venue(symbol: str) -> tuple[str, str]:
    """``(venue, instrument_type)``, by the same 6-digit rule as everywhere else."""
    normalized = str(symbol or "").upper().strip()
    if normalized.isdigit() and len(normalized) == _KRX_SYMBOL_LENGTH:
        return "KRX", "domestic_stock"
    return "NASD", "overseas_stock"


def market_of(symbol: str) -> str:
    return "KR" if resolve_venue(symbol)[0] == "KRX" else "US"


def policy_round_trip_bps(symbol: str) -> float | None:
    """The fee schedule's round trip: commission, tax, slippage, safety margin.

    Explicitly NOT the spread, which the policy prices at zero because it is not a
    property of the schedule. ``None`` when the cost configuration is unreadable,
    which every caller must treat as "unknown", never as "free".
    """
    global _cost_engine
    venue, instrument_type = resolve_venue(symbol)
    cached = _policy_cache.get((venue, instrument_type))
    if cached is not None:
        return cached
    try:
        if _cost_engine is None:
            from app.cost.trading_cost_engine import TradingCostEngine

            _cost_engine = TradingCostEngine()
        policy = _cost_engine.policy_for(venue=venue, instrument_type=instrument_type)
    except Exception:  # noqa: BLE001 - an unreadable cost config must not crash entry.
        return None
    total_rate = (
        policy.buy_fee_rate
        + policy.sell_fee_rate
        + policy.sell_tax_rate
        # Slippage is paid on both legs; spread and impact are charged once each.
        + 2.0 * policy.slippage_rate
        + policy.spread_rate
        + policy.market_impact_rate
        + policy.safety_margin_rate
    )
    cost_bps = max(0.0, total_rate * 10_000.0)
    _policy_cache[(venue, instrument_type)] = cost_bps
    return cost_bps


def measured_round_trip_bps(market: str) -> float | None:
    """What this venue's resolved shadow round trips actually cost, in bps.

    ``None`` when the tape has too few round trips to speak, which makes the caller
    fall back to the policy rather than to a number invented from three fills.
    """
    key = str(market or "").strip().upper()
    now = time.monotonic()
    cached = _measured_cache.get(key)
    if cached is not None and now - cached[0] < _MEASURED_TTL_SECONDS:
        return cached[1]
    value = _read_measured_round_trip_bps(key)
    _measured_cache[key] = (now, value)
    return value


def _read_measured_round_trip_bps(market: str) -> float | None:
    # The same env the shadow store itself is opened with. Reading a hardcoded
    # repo-relative path would make this depend on the process's working directory
    # -- and, in a test run, on whatever the operator's live tape happened to hold
    # that morning.
    from app.trading.directional_shadow import DEFAULT_SHADOW_STORE_PATH

    path = Path(
        os.getenv("DIRECTIONAL_SHADOW_STORE_PATH", DEFAULT_SHADOW_STORE_PATH)
    )
    if not path.exists():
        return None
    try:
        with closing(
            sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        ) as conn:
            values = sorted(
                float(row[0])
                for row in conn.execute(
                    "select trading_cost_bps from shadow_outcomes "
                    "where market = ? and trading_cost_bps is not null",
                    (market,),
                )
                if row[0] is not None
            )
    except Exception:  # noqa: BLE001 - a tape read must never break the entry path.
        return None
    if len(values) < MINIMUM_RESOLVED_ROUND_TRIPS:
        return None
    return values[min(len(values) - 1, int(len(values) * REALIZED_COST_PERCENTILE))]


def all_in_round_trip_bps(
    symbol: str,
    *,
    spread_bps: float | None = None,
    fallback_bps: float = 0.0,
) -> float:
    """Everything one round trip on this symbol costs, in bps.

    ``max`` of the symbol-specific estimate (policy + its own spread) and the
    venue's realized round trip, never below ``fallback_bps``. See the module
    docstring for why those two are alternatives rather than addends.
    """
    policy = policy_round_trip_bps(symbol)
    candidates = [max(0.0, float(fallback_bps))]
    if policy is not None:
        candidates.append(policy + max(0.0, float(spread_bps or 0.0)))
    measured = measured_round_trip_bps(market_of(symbol))
    if measured is not None:
        candidates.append(measured)
    return max(candidates)


def reset_caches() -> None:
    """Drop the cached policy and tape reads (tests, and config edits in place)."""
    global _cost_engine
    _cost_engine = None
    _policy_cache.clear()
    _measured_cache.clear()
