"""Short-side context indicators, computed from data this repository actually has.

Scope discipline
----------------
Three of the inputs the short algorithms read have a real source here and are computed
below: execution spread, execution liquidity, and market alignment. Two do not:
``short_interest_ratio`` and ``days_to_cover``.

KRX publishes 공매도 잔고 daily, but nothing in this repository collects it — the only
``short_net_change`` field in the codebase is produced by the synthetic demo pipeline
(``app.trading_pipeline``), which is explicitly rejected as a live decision input by the
source policy. Deriving a crowding metric from it would be fabricating a risk measure
from sample data, which is worse than not having one: it would look like a working
squeeze filter while filtering nothing.

So this module computes what it can and returns ``None`` for what it cannot, and
:func:`short_indicator_gaps` names the missing ones so the gap is visible in the
dashboard rather than silently absent.

Why the missing ones matter
---------------------------
A squeeze is where a short's unbounded loss actually materialises, and crowding is the
best available predictor of one. With no source, the short algorithms cannot exclude a
crowded name — their ``max_days_to_cover`` / ``max_short_interest_ratio`` gates pass
vacuously. The fail-closed burden therefore falls entirely on the borrow gates, which is
a real reduction in defence-in-depth and is recorded as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

# Indicators the short algorithms consume that have NO data source in this repository.
# Named so the dashboard can show "not measured" rather than the operator inferring
# from an absent key that the check passed.
UNSOURCED_SHORT_INDICATORS: tuple[str, ...] = (
    "short_interest_ratio",
    "days_to_cover",
)

# ADTV at which liquidity scores 1.0. Matches the log-scaled mapping already used by
# ``app.risk.manager._liquidity_score_from_market`` so the two cannot disagree.
_LIQUIDITY_REFERENCE_ADTV = 10_000_000_000.0


@dataclass(frozen=True)
class ShortIndicators:
    """Computed short-side context. ``None`` means "no source", never "zero"."""

    spread_bps: float | None = None
    liquidity_score: float | None = None
    market_alignment: float | None = None
    short_interest_ratio: float | None = None
    days_to_cover: float | None = None

    def as_context(self) -> dict[str, Any]:
        """Only the fields that were actually measured.

        Omitting rather than emitting ``None`` matters: ``ElectionContext`` treats an
        absent field as unresolved and the consuming algorithm decides what to do about
        it, whereas an explicit ``None`` would look like a measurement that came back
        empty.
        """
        return {
            name: value
            for name, value in (
                ("spread_bps", self.spread_bps),
                ("liquidity_score", self.liquidity_score),
                ("market_alignment", self.market_alignment),
                ("short_interest_ratio", self.short_interest_ratio),
                ("days_to_cover", self.days_to_cover),
            )
            if value is not None
        }

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in (
                ("spread_bps", self.spread_bps),
                ("liquidity_score", self.liquidity_score),
                ("market_alignment", self.market_alignment),
                ("short_interest_ratio", self.short_interest_ratio),
                ("days_to_cover", self.days_to_cover),
            )
            if value is None
        )


def spread_bps_from_book(orderbook: Any) -> float | None:
    """Top-of-book spread in bps. ``None`` when the book is absent or crossed.

    A crossed or empty book is refused rather than reported as a huge spread, because
    the caller's job here is to measure execution quality, not to invent a worst case —
    the cost engine already has its own (deliberately punitive) handling for that.
    """
    if orderbook is None:
        return None
    bid = _positive(getattr(orderbook, "best_bid", None))
    ask = _positive(getattr(orderbook, "best_ask", None))
    if bid is None or ask is None or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10_000.0


def liquidity_score_from_adtv(average_daily_trading_value: Any) -> float | None:
    """Log-scaled 0..1 liquidity score, matching the RiskManager's mapping."""
    adtv = _positive(average_daily_trading_value)
    if adtv is None:
        return None
    return min(1.0, math.log1p(adtv) / math.log1p(_LIQUIDITY_REFERENCE_ADTV))


def market_alignment(symbol_return: Any, market_return: Any) -> float | None:
    """How much this symbol's move agrees with the market's, in [-1, 1].

    +1 means it is moving with the market, -1 against it. A directional short wants
    ALIGNMENT with a falling market; a beta-neutral short (``residual_relative_weakness``)
    deliberately does not care, which is why this is context rather than a gate.

    ``None`` when either leg is unknown — an unmeasured alignment must not read as
    "perfectly neutral", which 0.0 would.
    """
    symbol = _finite(symbol_return)
    market = _finite(market_return)
    if symbol is None or market is None:
        return None
    if symbol == 0.0 or market == 0.0:
        # No move on one side means agreement is undefined, not zero.
        return None
    same_direction = (symbol > 0) == (market > 0)
    magnitude = min(abs(symbol), abs(market)) / max(abs(symbol), abs(market))
    return magnitude if same_direction else -magnitude


def compute_short_indicators(
    *,
    orderbook: Any = None,
    average_daily_trading_value: Any = None,
    symbol_return: Any = None,
    market_return: Any = None,
    short_interest_ratio: Any = None,
    days_to_cover: Any = None,
) -> ShortIndicators:
    """Assemble whatever short-side context is measurable from the given inputs.

    ``short_interest_ratio`` / ``days_to_cover`` are accepted as PARAMETERS rather than
    derived, because no source for them exists here. A caller that acquires one (a KRX
    공매도 잔고 feed, or an operator-maintained file like the borrow source) can pass
    them through and every downstream gate starts working with no further change.
    """
    return ShortIndicators(
        spread_bps=spread_bps_from_book(orderbook),
        liquidity_score=liquidity_score_from_adtv(average_daily_trading_value),
        market_alignment=market_alignment(symbol_return, market_return),
        short_interest_ratio=_finite(short_interest_ratio),
        days_to_cover=_finite(days_to_cover),
    )


def short_indicator_gaps(indicators: ShortIndicators) -> dict[str, Any]:
    """Which short indicators have no source, and what that costs.

    Surfaced so the dashboard can state the reduction in defence-in-depth explicitly. A
    silently-absent crowding metric looks identical to a crowding check that passed.
    """
    unsourced = [
        name for name in UNSOURCED_SHORT_INDICATORS if getattr(indicators, name) is None
    ]
    return {
        "unsourced": unsourced,
        "squeeze_filter_active": not unsourced,
        "detail": (
            ""
            if not unsourced
            else (
                "공매도 잔고 데이터 소스가 없어 혼잡도(스퀴즈) 필터가 비활성입니다. "
                "해당 게이트는 무조건 통과하며, fail-closed 부담은 대주 게이트가 집니다."
            )
        ),
    }


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
