"""Acceptance tests for the live-execution risk fixes:

* SELL pricing (fix 5) — take-profit uses best_bid; stops use a marketable bid.
* No-orderbook handling (fix 8) — a BUY without a book is blocked; an urgent SELL
  is still allowed to exit.
* US exchange routing (fix 10) — unknown US BUY is blocked in live strict mode; a
  known ticker routes to its real exchange.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from app.execution.exchange_resolver import ExchangeResolver
from app.execution.execution_quality import ExecutionQualityEngine, ExecutionQualityInput
from app.execution.order_pricing_policy import (
    EMERGENCY,
    ENTRY,
    HARD_STOP,
    STOP_LOSS,
    TAKE_PROFIT,
    ExecutionPricingPolicy,
    PricingContext,
    classify_action_reason,
)
from app.schemas.domain import FinalOrder, OrderSide, OrderType
from app.trading.realtime_trading_engine import RealtimeTradingEngine


class BuyNoOrderbookTest(unittest.TestCase):
    """Acceptance: BUY without orderbook is blocked."""

    def test_buy_without_orderbook_is_blocked(self) -> None:
        engine = ExecutionQualityEngine()
        a = engine.assess(
            ExecutionQualityInput(
                symbol="005930",
                strategy_family="live_short_horizon",
                decision_reference_price=10_000.0,
                gross_expected_return=0.02,
                net_expected_return=0.015,
                required_min_net_return=0.008,
                best_bid=0.0,
                best_ask=0.0,
                side="BUY",
            )
        )
        self.assertFalse(a.allowed)
        self.assertEqual(a.reject_reason, "EXEC_NO_ORDERBOOK_BLOCKED")
        # And the spread must NOT be reported as zero (that hid the risk).
        self.assertGreater(a.spread_rate, 0.0)

    def test_buy_with_stale_orderbook_is_blocked(self) -> None:
        engine = ExecutionQualityEngine()
        a = engine.assess(
            ExecutionQualityInput(
                symbol="005930",
                strategy_family="live_short_horizon",
                decision_reference_price=10_000.0,
                gross_expected_return=0.02,
                net_expected_return=0.015,
                required_min_net_return=0.008,
                best_bid=9_995.0,
                best_ask=10_005.0,
                side="BUY",
                orderbook_age_sec=999.0,  # far beyond EXEC_MAX_ORDERBOOK_AGE_SEC default
            )
        )
        self.assertFalse(a.allowed)
        self.assertEqual(a.reject_reason, "EXEC_NO_ORDERBOOK_BLOCKED")


class SellPricingTest(unittest.TestCase):
    """Acceptance: SELL pricing by exit reason."""

    def setUp(self) -> None:
        self.policy = ExecutionPricingPolicy()

    def test_take_profit_uses_best_bid_when_profitable(self) -> None:
        d = self.policy.price(
            PricingContext(
                symbol="005930",
                side="SELL",
                action_reason=TAKE_PROFIT,
                reference_price=10_000.0,
                best_bid=9_990.0,
                best_ask=10_010.0,
                is_domestic=True,
                min_net_exit_return=0.005,
                expected_net_return=0.012,
            )
        )
        self.assertTrue(d.priced)
        self.assertEqual(d.pricing_policy, "SELL_TP_BEST_BID")
        self.assertEqual(d.limit_price, 9_990.0)
        self.assertNotIn("TP_NET_BELOW_MIN", d.warnings)

    def test_stop_loss_uses_marketable_bid(self) -> None:
        # KRX tick at ~10,000 is 10; one tick below the bid.
        d = self.policy.price(
            PricingContext(
                symbol="005930",
                side="SELL",
                action_reason=STOP_LOSS,
                reference_price=10_000.0,
                best_bid=9_990.0,
                best_ask=10_010.0,
                is_domestic=True,
            )
        )
        self.assertTrue(d.priced)
        self.assertEqual(d.pricing_policy, "SELL_STOP_MARKETABLE_BID")
        self.assertEqual(d.limit_price, 9_980.0)

    def test_emergency_sell_allowed_without_orderbook(self) -> None:
        d = self.policy.price(
            PricingContext(
                symbol="AAPL",
                side="SELL",
                action_reason=HARD_STOP,
                reference_price=100.0,
                best_bid=0.0,
                best_ask=0.0,
                is_domestic=False,
            )
        )
        self.assertTrue(d.priced)
        self.assertEqual(d.pricing_policy, "SELL_EMERGENCY_FALLBACK")
        self.assertIn("NO_ORDERBOOK_EMERGENCY_SELL_ALLOWED", d.warnings)
        # Discounted reference (default 0.3%): 100 * 0.997 = 99.7.
        self.assertAlmostEqual(d.limit_price, 99.7, places=2)

    def test_emergency_sell_allowed_by_quality_without_orderbook(self) -> None:
        engine = ExecutionQualityEngine()
        a = engine.assess(
            ExecutionQualityInput(
                symbol="AAPL",
                strategy_family="live_short_horizon",
                decision_reference_price=100.0,
                gross_expected_return=0.0,
                net_expected_return=0.0,
                required_min_net_return=0.0,
                best_bid=0.0,
                best_ask=0.0,
                side="SELL",
                action_reason="HARD_STOP",
            )
        )
        self.assertTrue(a.allowed)
        self.assertIn("NO_ORDERBOOK_EMERGENCY_SELL_ALLOWED", a.warnings)

    def test_non_urgent_sell_without_orderbook_is_blocked_by_quality(self) -> None:
        engine = ExecutionQualityEngine()
        a = engine.assess(
            ExecutionQualityInput(
                symbol="AAPL",
                strategy_family="live_short_horizon",
                decision_reference_price=100.0,
                gross_expected_return=0.0,
                net_expected_return=0.0,
                required_min_net_return=0.0,
                best_bid=0.0,
                best_ask=0.0,
                side="SELL",
                action_reason="TAKE_PROFIT",
            )
        )
        self.assertFalse(a.allowed)
        self.assertEqual(a.reject_reason, "EXEC_NO_ORDERBOOK_SELL_BLOCKED")


class BuyPricingTest(unittest.TestCase):
    """Entry pricing. Crossing at the ask is now OPT-IN, not the default.

    Entries post passively at the bid so a round trip stops paying the full spread —
    13-50bps on the live KRX tape against a 27.8bps modelled cost, a spread the cost
    model never charged. See tests/test_passive_entry_pricing.py for that contract;
    this class keeps the crossing path covered because it is still reachable.
    """

    def setUp(self) -> None:
        self.policy = ExecutionPricingPolicy()

    def test_buy_crosses_at_best_ask_when_passive_entry_disabled(self) -> None:
        previous = os.environ.get("EXEC_PASSIVE_ENTRY")
        os.environ["EXEC_PASSIVE_ENTRY"] = "false"
        try:
            policy = ExecutionPricingPolicy()
            d = policy.price(
                PricingContext("005930", "BUY", ENTRY, 10_000.0, best_bid=9_990.0, best_ask=10_010.0, is_domestic=True)
            )
        finally:
            if previous is None:
                os.environ.pop("EXEC_PASSIVE_ENTRY", None)
            else:
                os.environ["EXEC_PASSIVE_ENTRY"] = previous
        self.assertTrue(d.priced)
        self.assertEqual(d.pricing_policy, "BUY_BEST_ASK")
        self.assertEqual(d.limit_price, 10_010.0)

    def test_buy_posts_passively_by_default(self) -> None:
        d = self.policy.price(
            PricingContext("005930", "BUY", ENTRY, 10_000.0, best_bid=9_990.0, best_ask=10_010.0, is_domestic=True)
        )
        self.assertTrue(d.priced)
        self.assertEqual(d.pricing_policy, "BUY_PASSIVE_BID")
        self.assertLess(d.limit_price, 10_010.0)

    def test_buy_no_orderbook_not_priced(self) -> None:
        d = self.policy.price(PricingContext("005930", "BUY", ENTRY, 10_000.0, best_bid=0.0, best_ask=0.0))
        self.assertFalse(d.priced)
        self.assertIn("EXEC_NO_ORDERBOOK_BLOCKED", d.reason_codes)


class ExchangeResolverTest(unittest.TestCase):
    """Acceptance: US exchange routing."""

    def setUp(self) -> None:
        self._saved = os.environ.get("KIS_US_EXCHANGE_MAP")
        os.environ.pop("KIS_US_EXCHANGE_MAP", None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("KIS_US_EXCHANGE_MAP", None)
        else:
            os.environ["KIS_US_EXCHANGE_MAP"] = self._saved

    def test_us_buy_unknown_exchange_blocked_in_live_strict_mode(self) -> None:
        resolver = ExchangeResolver(strict=True, allow_default_in_live=False, csv_path="__nonexistent_map__.csv")
        r = resolver.resolve("ZZQXW", "BUY", account=None, live=True)
        self.assertFalse(r.allowed)
        self.assertEqual(r.reason_code, "US_EXCHANGE_UNKNOWN")

    def test_us_buy_known_nyse_routes_to_nyse(self) -> None:
        os.environ["KIS_US_EXCHANGE_MAP"] = '{"TESTX": "NYSE"}'
        resolver = ExchangeResolver(strict=True, csv_path="__nonexistent_map__.csv")
        r = resolver.resolve("TESTX", "BUY", account=None, live=True)
        self.assertTrue(r.allowed)
        self.assertEqual(r.exchange, "NYSE")
        self.assertEqual(r.source, "env_map")

    def test_domestic_symbol_resolves_kr(self) -> None:
        resolver = ExchangeResolver()
        r = resolver.resolve("005930", "BUY", live=True)
        self.assertTrue(r.allowed)
        self.assertEqual(r.exchange, "KR")

    def test_unknown_us_sell_is_never_blocked(self) -> None:
        resolver = ExchangeResolver(strict=True, csv_path="__nonexistent_map__.csv")
        r = resolver.resolve("ZZQXW", "SELL", account=None, live=True)
        self.assertTrue(r.allowed)  # exiting a position must not be blocked by routing

    def test_paper_default_marked_paper_only(self) -> None:
        resolver = ExchangeResolver(strict=True, csv_path="__nonexistent_map__.csv")
        r = resolver.resolve("ZZQXW", "BUY", account=None, live=False)
        self.assertTrue(r.allowed)
        self.assertEqual(r.reason_code, "US_EXCHANGE_DEFAULTED_PAPER_ONLY")


class ActionReasonClassifierTest(unittest.TestCase):
    def test_classifies_exit_reasons(self) -> None:
        self.assertEqual(classify_action_reason("BUY", None), ENTRY)
        self.assertEqual(classify_action_reason("SELL", "stop_loss:-1.2%"), STOP_LOSS)
        self.assertEqual(classify_action_reason("SELL", "hard_stop_loss:-8.0%"), HARD_STOP)
        self.assertEqual(classify_action_reason("SELL", "loss_exit:-5.0%"), EMERGENCY)
        self.assertEqual(classify_action_reason("SELL", "quick_take_profit:0.8%"), TAKE_PROFIT)


class PrepareOrderFailClosedTest(unittest.TestCase):
    """Acceptance: _prepare_order_for_execution must FAIL-CLOSED for BUY and non-urgent
    SELL when the book source is missing or preparation raises; only an urgent stop/
    hard-stop/emergency SELL may still exit. Regression for the two fail-open escape
    hatches (EXEC_NO_BOOK_SOURCE / EXEC_PREPARE_SKIPPED) that submitted at last_price.
    """

    def _order(self, side: str) -> FinalOrder:
        return FinalOrder(
            ticker="005930",
            market="KR",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            quantity=1,
            limit_price=10_000.0,
        )

    def _prep(self, stub, side, order, diagnostics=None, reason_codes=()):  # type: ignore[no-untyped-def]
        return RealtimeTradingEngine._prepare_order_for_execution(
            stub,
            order.ticker,
            side,
            order,
            diagnostics,
            reason_codes,
            None,
            datetime.now(timezone.utc),
        )

    def test_no_book_source_blocks_buy(self) -> None:
        stub = SimpleNamespace(decision_engine=SimpleNamespace(), _last_failed_entry_price={})
        priced, ok, reason, _ = self._prep(stub, "BUY", self._order("BUY"))
        self.assertFalse(ok)
        self.assertIsNone(priced)
        self.assertEqual(reason, "EXEC_NO_BOOK_SOURCE")

    def test_no_book_source_blocks_non_urgent_sell(self) -> None:
        stub = SimpleNamespace(decision_engine=SimpleNamespace(), _last_failed_entry_price={})
        priced, ok, reason, _ = self._prep(
            stub, "SELL", self._order("SELL"), diagnostics={"exit_reason": "take_profit:0.8%"}
        )
        self.assertFalse(ok)
        self.assertIsNone(priced)
        self.assertEqual(reason, "EXEC_NO_BOOK_SOURCE")

    def test_no_book_source_allows_urgent_sell(self) -> None:
        stub = SimpleNamespace(decision_engine=SimpleNamespace(), _last_failed_entry_price={})
        order = self._order("SELL")
        priced, ok, reason, _ = self._prep(
            stub, "SELL", order, diagnostics={"exit_reason": "stop_loss:-1.2%"}
        )
        self.assertTrue(ok)
        self.assertIs(priced, order)  # original order priced at reference so the exit can fill
        self.assertEqual(reason, "EXEC_NO_BOOK_SOURCE_EMERGENCY_SELL")

    def _raising_stub(self):  # type: ignore[no-untyped-def]
        def _raise(*_a, **_k):
            raise RuntimeError("boom")

        return SimpleNamespace(
            decision_engine=SimpleNamespace(store=SimpleNamespace(latest_orderbook=lambda _s: None)),
            exchange_resolver=SimpleNamespace(resolve=_raise),
            _last_failed_entry_price={},
            _live_mode=lambda: True,
            _record=lambda _e: None,
        )

    def test_prepare_exception_blocks_buy(self) -> None:
        priced, ok, reason, _ = self._prep(self._raising_stub(), "BUY", self._order("BUY"))
        self.assertFalse(ok)
        self.assertIsNone(priced)
        self.assertEqual(reason, "EXEC_PREPARE_FAILED")

    def test_prepare_exception_allows_urgent_sell(self) -> None:
        order = self._order("SELL")
        priced, ok, reason, _ = self._prep(
            self._raising_stub(), "SELL", order, diagnostics={"exit_reason": "hard_stop_loss:-3.0%"}
        )
        self.assertTrue(ok)
        self.assertIs(priced, order)
        self.assertEqual(reason, "EXEC_PREPARE_SKIPPED_EMERGENCY_SELL")

    def test_prepare_exception_blocks_non_urgent_sell(self) -> None:
        priced, ok, reason, _ = self._prep(
            self._raising_stub(), "SELL", self._order("SELL"), diagnostics={"exit_reason": "model_exit"}
        )
        self.assertFalse(ok)
        self.assertIsNone(priced)
        self.assertEqual(reason, "EXEC_PREPARE_FAILED")


if __name__ == "__main__":
    unittest.main()
