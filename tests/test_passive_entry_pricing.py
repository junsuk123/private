"""Passive entry pricing: stop paying the spread we were never charged for.

The defect this addresses is a contradiction between three layers:

* the strategy plan declares ``entry_price_policy = {"kind": "passive_limit"}``;
* the cost model charges ``spread_rate = 0`` and fills at the signal bar's close,
  i.e. it prices a PASSIVE fill;
* execution crossed at ``best_ask`` and exited at ``best_bid``, paying the FULL
  round-trip spread -- 13-50bps on the live KRX tape against a 27.8bps modelled
  round-trip cost.

So the model scored trades at a fill price execution never attempted. These tests
pin the invariants that make passive pricing safe, above all that a "passive" order
can never quietly become a crossing one.
"""

from __future__ import annotations

import pytest

from app.execution.order_pricing_policy import (
    EMERGENCY,
    ENTRY,
    HARD_STOP,
    STOP_LOSS,
    TAKE_PROFIT,
    ExecutionPricingPolicy,
    PricingContext,
    _round_to_tick,
)


def test_rounding_up_an_exact_tick_multiple_returns_itself() -> None:
    """Regression: ``mode="up"`` returned one tick LOW for exact multiples.

    258,000 / 500 = 516.0 exactly, and the old form produced 515 -> 257,500. A SELL
    priced "up to the ask" was therefore posted a tick below the ask. The bug was
    dormain only because nothing used "up" yet.
    """
    assert _round_to_tick(258_000.0, 500.0, mode="up") == 258_000.0
    assert _round_to_tick(257_600.0, 500.0, mode="up") == 258_000.0
    assert _round_to_tick(7_990.0, 10.0, mode="up") == 7_990.0
    assert _round_to_tick(7_991.0, 10.0, mode="up") == 8_000.0
    # The other modes must be untouched by the fix.
    assert _round_to_tick(258_000.0, 500.0, mode="down") == 258_000.0
    assert _round_to_tick(257_900.0, 500.0, mode="down") == 257_500.0


def _ctx(**overrides) -> PricingContext:
    values = {
        "symbol": "005930",
        "side": "BUY",
        "action_reason": ENTRY,
        "best_bid": 257_500.0,
        "best_ask": 258_000.0,
        "is_domestic": True,
    }
    values.update(overrides)
    # The tick size is derived from the reference price, so a fixture that moves the
    # book without moving the reference tests a tick that does not belong to that
    # book. Default it to the mid unless a test is deliberately probing the mismatch.
    values.setdefault(
        "reference_price", (values["best_bid"] + values["best_ask"]) / 2.0
    )
    return PricingContext(**values)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "EXEC_PASSIVE_ENTRY",
        "EXEC_PASSIVE_ENTRY_OFFSET_TICKS",
        "EXEC_PASSIVE_TAKE_PROFIT",
        "EXEC_PASSIVE_TAKE_PROFIT_OFFSET_TICKS",
        "EXEC_BUY_MAX_CHASE_BPS",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #
def test_entry_posts_at_the_bid_by_default() -> None:
    decision = ExecutionPricingPolicy().price(_ctx())
    assert decision.priced is True
    assert decision.pricing_policy == "BUY_PASSIVE_BID"
    assert decision.limit_price <= 257_500.0
    assert "BUY_PASSIVE_MAY_NOT_FILL" in decision.warnings


def test_passive_entry_is_never_at_or_above_the_ask() -> None:
    """The invariant that makes the feature honest rather than cosmetic."""
    policy = ExecutionPricingPolicy()
    for bid, ask in (
        (257_500.0, 258_000.0),   # one tick apart
        (7_010.0, 7_020.0),
        (23_300.0, 23_350.0),
        (7_950.0, 7_990.0),       # wide, 50bps
        (100.0, 100.01),          # US-style
    ):
        decision = policy.price(
            _ctx(best_bid=bid, best_ask=ask, reference_price=(bid + ask) / 2)
        )
        assert decision.limit_price < ask, f"crossed the ask for {bid}/{ask}"
        assert decision.limit_price > 0


def test_offset_ticks_improve_the_bid_but_stay_inside_the_spread(monkeypatch) -> None:
    monkeypatch.setenv("EXEC_PASSIVE_ENTRY_OFFSET_TICKS", "50")
    # A deliberately absurd offset must still be clamped below the ask.
    decision = ExecutionPricingPolicy().price(_ctx(best_bid=7_950.0, best_ask=7_990.0))
    assert decision.limit_price < 7_990.0
    assert decision.limit_price >= 7_950.0


def test_stale_reference_price_cannot_push_the_limit_below_the_bid() -> None:
    """The tick comes from the reference price, which can disagree with the book.

    An oversized tick used to drag the passive limit far under the bid — unfillable,
    while the diagnostics reported a huge fictitious spread saving.
    """
    decision = ExecutionPricingPolicy().price(
        # Reference in the 250k tick band, book down at 7,950/7,990.
        _ctx(best_bid=7_950.0, best_ask=7_990.0, reference_price=257_750.0)
    )
    assert decision.limit_price <= 7_990.0
    assert decision.limit_price >= 7_000.0, "must not collapse far below the bid"
    assert decision.diagnostics["spread_bps_saved_vs_crossing"] < 200.0


def test_saving_is_reported_for_audit() -> None:
    decision = ExecutionPricingPolicy().price(_ctx())
    saved = decision.diagnostics["spread_bps_saved_vs_crossing"]
    # Crossing at 258,000 vs posting at 257,500 is ~19bps on this book.
    assert saved > 15.0
    assert decision.diagnostics["spread_bps"] > 15.0


def test_crossing_remains_available_when_disabled(monkeypatch) -> None:
    """The old behaviour must stay reachable — this is a live-money change."""
    monkeypatch.setenv("EXEC_PASSIVE_ENTRY", "false")
    decision = ExecutionPricingPolicy().price(_ctx())
    assert decision.pricing_policy == "BUY_BEST_ASK"
    assert decision.limit_price >= 258_000.0


def test_no_orderbook_still_blocks_the_buy(monkeypatch) -> None:
    """Passive pricing must not become a way to price a BUY with no book."""
    decision = ExecutionPricingPolicy().price(_ctx(best_bid=0.0, best_ask=0.0))
    assert decision.priced is False
    assert "EXEC_NO_ORDERBOOK_BLOCKED" in decision.reason_codes


# --------------------------------------------------------------------------- #
# Exits                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reason", [STOP_LOSS, HARD_STOP, EMERGENCY])
def test_urgent_exits_are_never_passive(monkeypatch, reason: str) -> None:
    """An unfilled stop is an unbounded loss, so stops must stay marketable
    even with every passive switch turned on."""
    monkeypatch.setenv("EXEC_PASSIVE_TAKE_PROFIT", "true")
    monkeypatch.setenv("EXEC_PASSIVE_ENTRY", "true")
    decision = ExecutionPricingPolicy().price(
        _ctx(side="SELL", action_reason=reason)
    )
    assert decision.pricing_policy == "SELL_STOP_MARKETABLE_BID"
    assert decision.limit_price <= 257_500.0


def test_take_profit_crosses_by_default() -> None:
    """Not filling a take-profit means still holding, so this is opt-in only."""
    decision = ExecutionPricingPolicy().price(
        _ctx(side="SELL", action_reason=TAKE_PROFIT)
    )
    assert decision.pricing_policy == "SELL_TP_BEST_BID"
    assert decision.limit_price <= 257_500.0


def test_passive_take_profit_posts_at_the_ask_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("EXEC_PASSIVE_TAKE_PROFIT", "true")
    decision = ExecutionPricingPolicy().price(
        _ctx(side="SELL", action_reason=TAKE_PROFIT)
    )
    assert decision.pricing_policy == "SELL_TP_PASSIVE_ASK"
    assert decision.limit_price >= 258_000.0
    assert "SELL_PASSIVE_MAY_NOT_FILL" in decision.warnings


def test_passive_take_profit_is_never_below_the_bid(monkeypatch) -> None:
    monkeypatch.setenv("EXEC_PASSIVE_TAKE_PROFIT", "true")
    monkeypatch.setenv("EXEC_PASSIVE_TAKE_PROFIT_OFFSET_TICKS", "99")
    policy = ExecutionPricingPolicy()
    for bid, ask in ((257_500.0, 258_000.0), (7_950.0, 7_990.0)):
        decision = policy.price(
            _ctx(side="SELL", action_reason=TAKE_PROFIT, best_bid=bid, best_ask=ask)
        )
        assert decision.limit_price > bid, "a passive TP must beat simply crossing"
