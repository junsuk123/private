"""Can any strategy's barrier actually be reached on this symbol?

Why discovery needs this
------------------------
Election has produced ``(symbol, strategy)`` pairs for a while, and the pair's score
is now conditioned on the symbol too. But *discovery* still chose symbols by turnover
alone, so the joint election could only ever pick the best pair out of whatever the
turnover ranking happened to hand it — and turnover says nothing about whether a
strategy's exit target is reachable on that chart.

That gap is measurable, and it is large. Once ``exit_geometry`` sizes the target
against the symbol's real round-trip cost, the target becomes a concrete number of
basis points, and the symbol's own bars say how far it typically travels in the
strategy's horizon. Subtracting the two gives headroom, and headroom computed on
2026-08-11 over 61 symbols with >=60 stored minute bars says only 12 of them can
reach the barrier at all.

The validation that justifies shipping this
-------------------------------------------
The six US names that produced 775 of the 776 stored realized outcomes — at a mean of
-123bps — are the WORST names in the ranking:

    INTC  -155      SOFI  -175      T     -198
    RIVN  -160      F     -185      BAC   -199
    NIO   -166      PFE   -186      LCID  -315

The metric uses no outcome data at all: it is bar dispersion against a cost-derived
target. It reproduces the measured catastrophe from arithmetic alone, which is the
strongest evidence available that discovery was selecting for the wrong thing.

Note what this does NOT do
--------------------------
It does not rank by "how many strategies have fired here". That number is dominated
by data coverage, not by chart suitability — the six symbols carrying all five
strategies are exactly the six with the most stored bars, because the US subscription
budget anchors a handful of names. Ranking on it would re-learn the subscription
budget and call it alpha.

It also does not hard-filter. A symbol with too few bars is UNKNOWN and scores
neutral, so a new listing can still enter and accumulate history; truncation to the
universe size is what removes the hopeless tail. Emptying the universe on a data gap
would be a worse failure than trading a marginal name.
"""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from app.strategy.exit_geometry import resolve_exit_geometry

#: Bars needed before the dispersion estimate means anything. Below this the symbol
#: is UNKNOWN rather than infeasible.
DEFAULT_MINIMUM_BARS = 30

#: Non-overlapping horizon-length windows the history must cover before a
#: sqrt-of-time extrapolation to that horizon is allowed. Five is the smallest count
#: at which the estimate is interpolative rather than a projection.
DEFAULT_MINIMUM_HORIZON_WINDOWS = 5

#: How far back to read bars. Long enough to span several sessions, short enough that
#: a regime from a month ago does not decide today's universe.
DEFAULT_LOOKBACK_DAYS = 5

VERDICT_FEASIBLE = "FEASIBLE"
VERDICT_INFEASIBLE = "INFEASIBLE"
VERDICT_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SymbolFeasibility:
    """Whether some strategy's cost-sized target is reachable on this symbol."""

    symbol: str
    market: str
    verdict: str
    bar_count: int
    spread_bps: float | None
    sigma_1m_bps: float | None
    #: Typical favourable excursion over the best strategy's horizon.
    attainable_bps: float | None
    #: That strategy's cost-relative take-profit.
    required_bps: float | None
    headroom_bps: float | None
    strategy_id: str = ""
    #: Per-strategy headroom, best first. The pair matters, not just the symbol.
    per_strategy: tuple[tuple[str, float], ...] = ()

    @property
    def score(self) -> float:
        """Sort key. UNKNOWN scores neutral so a data gap is not read as a verdict."""
        return 0.0 if self.headroom_bps is None else float(self.headroom_bps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "verdict": self.verdict,
            "bar_count": self.bar_count,
            "spread_bps": _round(self.spread_bps),
            "sigma_1m_bps": _round(self.sigma_1m_bps),
            "attainable_bps": _round(self.attainable_bps),
            "required_bps": _round(self.required_bps),
            "headroom_bps": _round(self.headroom_bps),
            "strategy_id": self.strategy_id,
            "per_strategy": [
                {"strategy_id": name, "headroom_bps": round(value, 1)}
                for name, value in self.per_strategy
            ],
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def sigma_1m_bps(closes: Sequence[float]) -> float | None:
    """Mean absolute one-minute return, in bps.

    The MEAN, not the median, and for the reason recorded in
    ``stored_counterfactual._causal_horizon_sigma_bps``: in a thin name more than half
    the bars print no price change, so the median absolute return is exactly 0 and the
    estimator returns nothing for precisely the symbols that need judging. The mean is
    non-zero whenever any bar moved, and for roughly normal returns it is
    ``sigma * sqrt(2/pi)`` — a scale estimate rather than a quantile.
    """
    usable = [float(close) for close in closes if close and float(close) > 0]
    if len(usable) < 5:
        return None
    returns = [
        abs(usable[index] / usable[index - 1] - 1.0) * 10_000.0
        for index in range(1, len(usable))
        if usable[index - 1] > 0
    ]
    if not returns:
        return None
    return statistics.fmean(returns)


def attainable_move_bps(sigma_bps: float, horizon_seconds: float) -> float:
    """Typical excursion over the horizon, by square-root-of-time scaling.

    ``sqrt`` scaling rather than measuring overlapping horizon-length windows
    directly: a 40-minute horizon over five sessions of bars leaves far too few
    non-overlapping windows to estimate from, and overlapping ones inflate the sample
    without adding information (see the overlapping-window note in
    ``docs/validation.md``).
    """
    minutes = max(1.0, float(horizon_seconds) / 60.0)
    return float(sigma_bps) * math.sqrt(minutes)


def evaluate_symbol(
    symbol: str,
    *,
    closes: Sequence[float],
    spread_bps: float | None,
    round_trip_cost_bps: float | None,
    strategies: Mapping[str, float],
    market: str = "",
    minimum_bars: int = DEFAULT_MINIMUM_BARS,
    minimum_horizon_windows: int = DEFAULT_MINIMUM_HORIZON_WINDOWS,
) -> SymbolFeasibility:
    """Headroom for the best-fitting strategy on this symbol.

    ``strategies`` maps strategy id -> horizon seconds. The BEST headroom across them
    is the symbol's score: a symbol is worth holding if *some* strategy can work
    there, which is the pairing question asked one layer earlier than election.
    """
    bar_count = len(closes)
    sigma = sigma_1m_bps(closes)
    if bar_count < minimum_bars or sigma is None:
        return SymbolFeasibility(
            symbol=symbol,
            market=market,
            verdict=VERDICT_UNKNOWN,
            bar_count=bar_count,
            spread_bps=spread_bps,
            sigma_1m_bps=sigma,
            attainable_bps=None,
            required_bps=None,
            headroom_bps=None,
        )

    # The target is sized against the spread, so an unmeasured spread makes the
    # required move unknowable rather than small. Claiming FEASIBLE without it is how
    # a 70bps-spread name reads as reachable: measured while building this, LCID
    # scored +55 with its spread missing and -315 with it present.
    if spread_bps is None:
        return SymbolFeasibility(
            symbol=symbol,
            market=market,
            verdict=VERDICT_UNKNOWN,
            bar_count=bar_count,
            spread_bps=None,
            sigma_1m_bps=sigma,
            attainable_bps=None,
            required_bps=None,
            headroom_bps=None,
        )

    scored: list[tuple[str, float, float, float]] = []
    for strategy_id, horizon_seconds in strategies.items():
        geometry = resolve_exit_geometry(
            strategy_id,
            round_trip_cost_bps=round_trip_cost_bps,
            spread_bps=spread_bps,
        )
        # The geometry may stretch the horizon to make room for a bigger target, and
        # the extra time is exactly what makes the bigger target reachable. Scoring
        # the stretched target against the unstretched horizon would refuse a pair
        # the executor would actually hold long enough to resolve.
        horizon = max(float(horizon_seconds or 0.0), float(geometry.max_holding_seconds))
        horizon_minutes = max(1.0, horizon / 60.0)
        # The history must cover several NON-OVERLAPPING horizon windows before a
        # horizon-scaled dispersion estimate means anything. Overlapping windows
        # inflate the sample without adding information, which is the same trap the
        # forward-return analysis hit (stride must be >= horizon).
        #
        # Without this the extrapolation does not fail quietly: at one bar per horizon
        # minute, two thin names with 32 and 40 bars scored +607 and +164 headroom and
        # took the top of the ranking from every name with real coverage. A wild
        # microcap topping the universe is precisely the outcome to avoid, and it
        # arrives as a data artefact rather than as a signal.
        if bar_count < horizon_minutes * minimum_horizon_windows:
            continue
        attainable = attainable_move_bps(sigma, horizon)
        scored.append(
            (strategy_id, attainable - geometry.take_profit_bps, attainable, geometry.take_profit_bps)
        )
    if not scored:
        # Bars enough to estimate dispersion, but not enough to reach any strategy's
        # horizon. Neither feasible nor refuted.
        return SymbolFeasibility(
            symbol=symbol,
            market=market,
            verdict=VERDICT_UNKNOWN,
            bar_count=bar_count,
            spread_bps=spread_bps,
            sigma_1m_bps=sigma,
            attainable_bps=None,
            required_bps=None,
            headroom_bps=None,
        )
    scored.sort(key=lambda row: row[1], reverse=True)
    best_id, best_headroom, best_attainable, best_required = scored[0]
    return SymbolFeasibility(
        symbol=symbol,
        market=market,
        verdict=VERDICT_FEASIBLE if best_headroom > 0.0 else VERDICT_INFEASIBLE,
        bar_count=bar_count,
        spread_bps=spread_bps,
        sigma_1m_bps=sigma,
        attainable_bps=best_attainable,
        required_bps=best_required,
        headroom_bps=best_headroom,
        strategy_id=best_id,
        per_strategy=tuple((row[0], row[1]) for row in scored),
    )


def rank_by_feasibility(
    symbols: Sequence[str],
    feasibility: Mapping[str, SymbolFeasibility],
) -> tuple[str, ...]:
    """Reorder a turnover ranking by strategy headroom, keeping it stable.

    Turnover is not discarded — it is the liquidity and cost prerequisite that
    produced this list, and it breaks ties. What changes is that a name whose chart
    cannot reach any strategy's target no longer outranks one that can merely because
    more shares changed hands.
    """
    order = {symbol: index for index, symbol in enumerate(symbols)}

    def key(symbol: str) -> tuple[float, int]:
        entry = feasibility.get(symbol)
        return (-(entry.score if entry is not None else 0.0), order.get(symbol, 0))

    return tuple(sorted(symbols, key=key))


def enabled_strategy_horizons() -> dict[str, float]:
    """Catalogued LONG strategies that may actually be deployed, with their horizons.

    Reads the same authority election does, so a strategy turned off in config stops
    influencing which symbols are held.
    """
    try:
        from app.strategy.catalog import STRATEGY_IDS, is_short_strategy
        from app.strategy.exit_geometry import exit_geometry
        from app.technical.strategy_algorithms import strategy_live_authorized
    except Exception:  # noqa: BLE001 - discovery must not break on an import error.
        return {}
    horizons: dict[str, float] = {}
    for strategy_id in STRATEGY_IDS:
        if is_short_strategy(strategy_id):
            continue
        try:
            if not strategy_live_authorized(strategy_id):
                continue
            horizons[strategy_id] = float(exit_geometry(strategy_id).max_holding_seconds)
        except Exception:  # noqa: BLE001
            continue
    return horizons


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def measure(
    symbols: Iterable[str],
    *,
    market_store: Any = None,
    cost_engine: Any = None,
    strategies: Mapping[str, float] | None = None,
    now: datetime | None = None,
    lookback_days: int | None = None,
    minimum_bars: int | None = None,
    minimum_horizon_windows: int | None = None,
) -> dict[str, SymbolFeasibility]:
    """Feasibility for each symbol, read from stored minute bars.

    Best-effort throughout: a symbol whose bars cannot be read is UNKNOWN, never
    infeasible. Discovery is not allowed to fail because a measurement did.
    """
    horizons = dict(strategies or enabled_strategy_horizons())
    if not horizons:
        return {}
    moment = now or datetime.now(timezone.utc)
    since = moment - timedelta(days=lookback_days or _env_int("FEASIBILITY_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS))
    floor_bars = minimum_bars or _env_int("FEASIBILITY_MINIMUM_BARS", DEFAULT_MINIMUM_BARS)
    floor_windows = minimum_horizon_windows or _env_int(
        "FEASIBILITY_MINIMUM_HORIZON_WINDOWS", DEFAULT_MINIMUM_HORIZON_WINDOWS
    )

    if market_store is None:
        try:
            from app.data.realtime_store import RealtimeMarketDataStore

            market_store = RealtimeMarketDataStore()
        except Exception:  # noqa: BLE001
            return {}
    if cost_engine is None:
        try:
            from app.cost.trading_cost_engine import TradingCostEngine

            cost_engine = TradingCostEngine()
        except Exception:  # noqa: BLE001
            cost_engine = None

    results: dict[str, SymbolFeasibility] = {}
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if not symbol or symbol in results:
            continue
        try:
            bars = market_store.recent_minute_bars(symbol, since, limit=2000)
        except Exception:  # noqa: BLE001
            bars = ()
        closes = [float(getattr(bar, "close", 0.0) or 0.0) for bar in bars]
        spreads = [
            float(getattr(bar, "spread_bps", 0.0) or 0.0)
            for bar in bars
            if (getattr(bar, "spread_bps", None) or 0.0) > 0
        ]
        spread = statistics.median(spreads) if spreads else _latest_book_spread(
            market_store, symbol
        )
        market = _market_of(symbol)
        results[symbol] = evaluate_symbol(
            symbol,
            closes=closes,
            spread_bps=spread,
            round_trip_cost_bps=_round_trip_cost(
                cost_engine, symbol, market, closes[-1] if closes else 0.0, spread
            ),
            strategies=horizons,
            market=market,
            minimum_bars=floor_bars,
            minimum_horizon_windows=floor_windows,
        )
    return results


def _latest_book_spread(market_store: Any, symbol: str) -> float | None:
    """Top-of-book spread when the bar series did not carry one.

    A single live quote is a worse estimate than the median of a session of bars, but
    it is enormously better than nothing: without any spread the symbol is scored
    UNKNOWN and drops to a neutral rank, which for a well-covered name throws away a
    real measurement over a missing column.
    """
    try:
        book = market_store.latest_orderbook(symbol)
    except Exception:  # noqa: BLE001
        return None
    value = float(getattr(book, "spread_bps", 0.0) or 0.0) if book is not None else 0.0
    return value if value > 0 else None


def _market_of(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    return "KR" if len(text) == 6 and text[:1].isdigit() else "US"


def _round_trip_cost(
    cost_engine: Any, symbol: str, market: str, price: float, spread_bps: float | None
) -> float | None:
    """Modelled round trip for this symbol, or ``None`` to fall back to the reference.

    ``None`` rather than a guess: ``resolve_exit_geometry`` floors an absent cost at
    the strategy's own reference, which is the conservative direction.
    """
    if cost_engine is None or price <= 0:
        return None
    try:
        estimate = cost_engine.estimate(
            symbol=symbol,
            market=market,
            venue="KRX" if market == "KR" else "NASD",
            instrument_type="domestic_stock" if market == "KR" else "overseas_stock",
            entry_price=price,
            expected_exit_price=price * 1.01,
            quantity=max(1, int((300_000 if market == "KR" else 1_000) / price)),
        )
        return float(estimate.total_cost_rate) * 10_000.0
    except Exception:  # noqa: BLE001
        return None


def summary(feasibility: Mapping[str, SymbolFeasibility]) -> dict[str, Any]:
    """Counts plus the extremes, for a one-line report of what discovery reordered."""
    counts: dict[str, int] = {}
    for entry in feasibility.values():
        counts[entry.verdict] = counts.get(entry.verdict, 0) + 1
    ranked = sorted(
        (entry for entry in feasibility.values() if entry.headroom_bps is not None),
        key=lambda entry: entry.headroom_bps or 0.0,
        reverse=True,
    )
    return {
        "counts": counts,
        "best": [entry.as_dict() for entry in ranked[:5]],
        "worst": [entry.as_dict() for entry in ranked[-5:]],
    }
