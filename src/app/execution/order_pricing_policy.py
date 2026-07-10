"""Executable limit-price policy.

The decision layer (:mod:`app.risk.manager`) stamps a *reference* price onto a
``FinalOrder`` — historically ``market.last_price`` — but the last trade price is
NOT an executable price: a BUY posted at the last print may sit behind the ask and
never fill, and a stop SELL posted at the last print may fail to exit while the book
runs away. Live execution must therefore re-price the order from the current book,
the side, and the exit urgency BEFORE submission.

This module is the single authority for that final limit price. It is deterministic,
side- and reason-aware, and never fabricates a zero-spread price when the book is
missing:

* BUY uses ``best_ask`` but never chases beyond ``EXEC_BUY_MAX_CHASE_BPS`` above the
  reference; with no book it declines to price (the caller must block the BUY).
* TAKE_PROFIT / model / reduce SELL uses ``best_bid`` (marketable to buyers).
* STOP_LOSS / HARD_STOP / EMERGENCY SELL uses ``best_bid`` minus a configurable tick
  offset so the exit actually fills; with no book but an urgent exit it falls back to
  a reference-price discount rather than refusing to sell.

All prices are snapped to the venue tick size (KRX price bands / US 0.01) so the
broker never rejects an off-tick limit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Normalized action reasons the pricing policy understands.
ENTRY = "ENTRY"
TAKE_PROFIT = "TAKE_PROFIT"
STOP_LOSS = "STOP_LOSS"
HARD_STOP = "HARD_STOP"
EMERGENCY = "EMERGENCY"
REDUCE = "REDUCE"
MODEL_EXIT = "MODEL_EXIT"
TIME_STOP = "TIME_STOP"

# Urgent exits must fill even when the order book is unavailable.
_URGENT_SELL_REASONS = frozenset({STOP_LOSS, HARD_STOP, EMERGENCY})

_EPSILON = 1e-9


def classify_action_reason(side: str, exit_reason: str | None, reason_codes: tuple[str, ...] = ()) -> str:
    """Map a decision-engine exit_reason / reason_codes to a normalized action reason.

    The shared decision engine encodes the exit motive as a free-form
    ``diagnostics["exit_reason"]`` string (e.g. ``"stop_loss:-1.20%"``,
    ``"quick_take_profit:0.80%"``). Pricing only needs the coarse category.
    """
    if str(side).upper() == "BUY":
        return ENTRY
    text = f"{exit_reason or ''} {' '.join(reason_codes or ())}".strip().lower()
    if text.startswith("hard_stop") or "hard_stop" in text:
        return HARD_STOP
    if "emergency" in text or text.startswith("loss_exit") or "loss_exit" in text:
        return EMERGENCY
    if text.startswith("stop_loss") or "stop_loss" in text:
        return STOP_LOSS
    if "drawdown_reduce" in text or "concentration_reduce" in text or "trailing_exit" in text or "reduce" in text:
        return REDUCE
    if "invalid_signal" in text or text.startswith("model") or "model_exit" in text:
        return MODEL_EXIT
    if "time_exit" in text or "time_stop" in text:
        return TIME_STOP
    if "profit" in text or "take_profit" in text:
        return TAKE_PROFIT
    # Unknown SELL motive: treat as a routine (non-urgent) model exit.
    return MODEL_EXIT


def is_urgent_sell(action_reason: str) -> bool:
    return str(action_reason).upper() in _URGENT_SELL_REASONS


def krx_tick_size(price: float) -> float:
    """KRX price-band tick size (2023 unified bands)."""
    p = float(price)
    if p < 2_000:
        return 1.0
    if p < 5_000:
        return 5.0
    if p < 20_000:
        return 10.0
    if p < 50_000:
        return 50.0
    if p < 200_000:
        return 100.0
    if p < 500_000:
        return 500.0
    return 1_000.0


def us_tick_size(price: float) -> float:
    """US sub-penny rule: 0.0001 below $1, else 0.01 (matches KIS APTR0057)."""
    return 0.0001 if float(price) < 1.0 else 0.01


def tick_size_for(price: float, is_domestic: bool) -> float:
    return krx_tick_size(price) if is_domestic else us_tick_size(price)


def _round_to_tick(price: float, tick: float, *, mode: str) -> float:
    """Snap ``price`` to a tick boundary. mode: 'down' (buy cap), 'up', 'nearest'."""
    if tick <= 0 or price <= 0:
        return max(0.0, float(price))
    units = price / tick
    if mode == "down":
        snapped = int(units + _EPSILON)
    elif mode == "up":
        snapped = int(units - _EPSILON) + (0 if abs(units - round(units)) < _EPSILON else 1)
    else:  # nearest
        snapped = int(units + 0.5)
    value = snapped * tick
    if value <= 0:
        value = tick
    # US prices carry cents/sub-cents; KRX ticks are integers.
    return round(value, 4)


@dataclass(frozen=True)
class PricingContext:
    symbol: str
    side: str                       # BUY / SELL
    action_reason: str              # ENTRY / TAKE_PROFIT / STOP_LOSS / HARD_STOP / EMERGENCY / REDUCE / MODEL_EXIT / TIME_STOP
    reference_price: float          # decision reference (last_price / model) — NOT necessarily executable
    best_bid: float = 0.0
    best_ask: float = 0.0
    is_domestic: bool = True
    min_net_exit_return: float = 0.0
    expected_net_return: float = 0.0
    orderbook_age_sec: float | None = None

    @property
    def has_valid_book(self) -> bool:
        return self.best_bid > 0 and self.best_ask > 0 and self.best_ask >= self.best_bid


@dataclass(frozen=True)
class PricingDecision:
    priced: bool                    # False => caller must NOT submit (e.g. BUY without a book)
    limit_price: float
    pricing_policy: str             # e.g. BUY_BEST_ASK, SELL_TP_BEST_BID, SELL_STOP_MARKETABLE_BID, SELL_EMERGENCY_FALLBACK
    reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ExecutionPricingPolicy:
    def __init__(self) -> None:
        self.max_chase_bps = _env_float("EXEC_BUY_MAX_CHASE_BPS", 20.0)
        self.sell_emergency_offset_ticks = max(0, _env_int("EXEC_SELL_EMERGENCY_OFFSET_TICKS", 1))
        self.sell_stop_offset_ticks = max(0, _env_int("EXEC_SELL_STOP_OFFSET_TICKS", 1))
        self.sell_emergency_fallback_offset_rate = _env_float("EXEC_SELL_EMERGENCY_FALLBACK_OFFSET_RATE", 0.003)

    def price(self, ctx: PricingContext) -> PricingDecision:
        ref = max(0.0, float(ctx.reference_price))
        tick = tick_size_for(ref if ref > 0 else max(ctx.best_ask, ctx.best_bid), ctx.is_domestic)
        base_diag: dict[str, Any] = {
            "action_reason": ctx.action_reason,
            "reference_price": ref,
            "best_bid": ctx.best_bid,
            "best_ask": ctx.best_ask,
            "tick_size": tick,
            "orderbook_age_sec": ctx.orderbook_age_sec,
            "has_valid_book": ctx.has_valid_book,
        }
        if str(ctx.side).upper() == "BUY":
            return self._price_buy(ctx, ref, tick, base_diag)
        return self._price_sell(ctx, ref, tick, base_diag)

    # -- BUY -------------------------------------------------------------
    def _price_buy(self, ctx: PricingContext, ref: float, tick: float, diag: dict[str, Any]) -> PricingDecision:
        if not ctx.has_valid_book:
            # A BUY must never be priced as if the spread were zero. Decline to
            # price; the execution-quality gate blocks the order.
            return PricingDecision(
                priced=False,
                limit_price=0.0,
                pricing_policy="BUY_NO_ORDERBOOK",
                reason_codes=("EXEC_NO_ORDERBOOK_BLOCKED",),
                warnings=("NO_ORDERBOOK_FOR_BUY",),
                diagnostics=diag,
            )
        chase_cap = ref * (1.0 + self.max_chase_bps / 10_000.0) if ref > 0 else ctx.best_ask
        capped = ref > 0 and ctx.best_ask > chase_cap + _EPSILON
        if capped:
            # Chasing beyond the cap: bind at the cap, rounded DOWN so we never exceed it.
            limit = _round_to_tick(chase_cap, tick, mode="down")
        else:
            # Post marketable at the ask (round to nearest tick so an off-tick ask still
            # lands on a fillable price at/above the ask).
            limit = _round_to_tick(ctx.best_ask, tick, mode="nearest")
        if limit <= 0:
            limit = _round_to_tick(ctx.best_ask, tick, mode="nearest")
        policy = "BUY_ASK_CHASE_CAPPED" if capped else "BUY_BEST_ASK"
        warnings = ("BUY_CHASE_CAPPED",) if capped else ()
        diag = {**diag, "chase_cap": chase_cap, "limit_price": limit}
        return PricingDecision(
            priced=True,
            limit_price=limit,
            pricing_policy=policy,
            reason_codes=(),
            warnings=warnings,
            diagnostics=diag,
        )

    # -- SELL ------------------------------------------------------------
    def _price_sell(self, ctx: PricingContext, ref: float, tick: float, diag: dict[str, Any]) -> PricingDecision:
        reason = str(ctx.action_reason).upper()
        urgent = is_urgent_sell(reason)
        if ctx.has_valid_book:
            if urgent:
                offset = self.sell_emergency_offset_ticks if reason in {HARD_STOP, EMERGENCY} else self.sell_stop_offset_ticks
                target = ctx.best_bid - offset * tick
                limit = _round_to_tick(max(target, tick), tick, mode="down")
                policy = "SELL_STOP_MARKETABLE_BID"
                warnings: tuple[str, ...] = ()
            else:
                limit = _round_to_tick(ctx.best_bid, tick, mode="down")
                policy = "SELL_TP_BEST_BID"
                warnings = ()
                if reason == TAKE_PROFIT and ctx.expected_net_return + _EPSILON < ctx.min_net_exit_return:
                    warnings = ("TP_NET_BELOW_MIN",)
            diag = {**diag, "limit_price": limit}
            return PricingDecision(
                priced=True,
                limit_price=limit,
                pricing_policy=policy,
                reason_codes=(),
                warnings=warnings,
                diagnostics=diag,
            )
        # No order book.
        if urgent:
            # Allow the exit with a discounted reference price so it still fills.
            target = ref * (1.0 - self.sell_emergency_fallback_offset_rate)
            limit = _round_to_tick(max(target, tick), tick, mode="down")
            diag = {**diag, "limit_price": limit, "fallback_offset_rate": self.sell_emergency_fallback_offset_rate}
            return PricingDecision(
                priced=True,
                limit_price=limit,
                pricing_policy="SELL_EMERGENCY_FALLBACK",
                reason_codes=(),
                warnings=("NO_ORDERBOOK_EMERGENCY_SELL_ALLOWED",),
                diagnostics=diag,
            )
        # Non-urgent SELL with no book: price at reference (execution-quality gate
        # decides whether to allow it) but flag the missing book.
        limit = _round_to_tick(ref, tick, mode="down") if ref > 0 else 0.0
        diag = {**diag, "limit_price": limit}
        return PricingDecision(
            priced=limit > 0,
            limit_price=limit,
            pricing_policy="SELL_REFERENCE_NO_ORDERBOOK",
            reason_codes=(),
            warnings=("NO_ORDERBOOK_SELL",),
            diagnostics=diag,
        )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)
