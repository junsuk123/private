"""Acceptance tests for the theory-driven trading-domain ontology reasoner."""

from __future__ import annotations

import unittest

from app.ontology.trading_domain_ontology import IntentType, ValidationState
from app.ontology.trading_domain_ontology import ReasonCodes as RC
from app.ontology.trading_fact_builder import build_trading_facts
from app.ontology.trading_reasoner import TradingDomainReasoner


def _good_buy(**overrides):
    """A BUY that should pass: fresh book, known exchange, positive net edge, validated."""
    base = dict(
        symbol="005930",
        side="BUY",
        reference_price=10_000.0,
        best_bid=9_995.0,
        best_ask=10_005.0,
        orderbook_fresh=True,
        exchange="KR",
        exchange_known=True,
        profitability={
            "gross_expected_return": 0.02,
            "net_expected_return": 0.012,
            "required_min_net_return": 0.008,
            "spread_rate": 0.001,
        },
        model_confidence=0.7,
        signal_family="momentum",
        validation={"live_expectancy": 0.004, "sample_size": 200, "oos_positive": True, "parameter_count": 5},
    )
    base.update(overrides)
    return build_trading_facts(**base)


class TradingOntologyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reasoner = TradingDomainReasoner()

    def test_good_buy_is_approved_with_explanation(self) -> None:
        r = self.reasoner.reason(_good_buy())
        self.assertEqual(r.intent, IntentType.BUY)
        self.assertTrue(r.approved)
        self.assertIn(RC.EXECUTION_FEASIBLE, r.theory_support)
        self.assertIn(RC.NET_EDGE_POSITIVE, r.theory_support)
        self.assertIn("EXECUTABLE_PRICE_FROM_ORDER_BOOK", r.required_conditions)
        self.assertEqual(r.recommended_order_policy.price_policy, "BUY_BEST_ASK")

    def test_buy_cannot_pass_solely_on_momentum(self) -> None:
        # Momentum + confidence but NO net edge (net below required min) => not a BUY.
        r = self.reasoner.reason(
            _good_buy(profitability={
                "gross_expected_return": 0.02,
                "net_expected_return": 0.001,
                "required_min_net_return": 0.008,
                "spread_rate": 0.001,
            })
        )
        self.assertNotEqual(r.intent, IntentType.BUY)
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.COST_DOMINATED, r.blocked_by)

    def test_buy_requires_positive_cost_adjusted_edge(self) -> None:
        r = self.reasoner.reason(
            _good_buy(profitability={
                "gross_expected_return": -0.001,
                "net_expected_return": -0.005,
                "required_min_net_return": 0.008,
                "spread_rate": 0.001,
            })
        )
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.NET_EDGE_INSUFFICIENT, r.blocked_by)

    def test_buy_without_orderbook_blocked(self) -> None:
        r = self.reasoner.reason(_good_buy(best_bid=0.0, best_ask=0.0))
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.NO_ORDERBOOK, r.blocked_by)

    def test_buy_with_unknown_exchange_blocked(self) -> None:
        r = self.reasoner.reason(_good_buy(symbol="AAPL", exchange="", exchange_known=False, is_domestic=False))
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.UNKNOWN_EXCHANGE, r.blocked_by)

    def test_stale_orderbook_blocked(self) -> None:
        r = self.reasoner.reason(_good_buy(orderbook_fresh=False))
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.STALE_ORDERBOOK, r.blocked_by)

    def test_negative_expectancy_strategy_disabled(self) -> None:
        r = self.reasoner.reason(
            _good_buy(validation={"live_expectancy": -0.002, "sample_size": 200, "parameter_count": 5})
        )
        self.assertEqual(r.validation_state, ValidationState.NEGATIVE_EXPECTANCY)
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.NEGATIVE_EXPECTANCY_DISABLED, r.blocked_by)

    def test_backtest_only_cannot_produce_live_buy(self) -> None:
        r = self.reasoner.reason(
            _good_buy(validation={"backtest_expectancy": 0.01, "sample_size": 5, "parameter_count": 5})
        )
        self.assertEqual(r.validation_state, ValidationState.BACKTEST_ONLY)
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.BACKTEST_ONLY_NO_LIVE_BUY, r.blocked_by)

    def test_unvalidated_signal_low_reliability_capped(self) -> None:
        r = self.reasoner.reason(_good_buy(validation={}))
        self.assertEqual(r.validation_state, ValidationState.UNVALIDATED)
        self.assertIn(RC.LOW_VALIDATION_RELIABILITY, r.theory_conflict)
        self.assertLessEqual(r.confidence, 0.4 + 1e-9)

    def test_news_sentiment_not_standalone_buy(self) -> None:
        r = self.reasoner.reason(_good_buy(signal_family="sentiment", primary_data_tier="T4"))
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.SIGNAL_ONLY_SENTIMENT, r.blocked_by)

    def test_high_inventory_blocks_new_buy(self) -> None:
        r = self.reasoner.reason(_good_buy(inventory_weight=0.5))
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.INVENTORY_RISK_HIGH, r.blocked_by)

    def test_macro_block_buy(self) -> None:
        r = self.reasoner.reason(_good_buy(diagnostics={"macro_block_buy": True}))
        self.assertEqual(r.intent, IntentType.BLOCK)
        self.assertIn(RC.MACRO_BLOCK_BUY, r.blocked_by)

    def test_sell_take_profit_distinguished(self) -> None:
        facts = build_trading_facts("005930", "SELL", exit_reason="quick_take_profit:0.80%", best_bid=9_990, best_ask=10_010)
        r = self.reasoner.reason(facts)
        self.assertEqual(r.intent, IntentType.SELL)
        self.assertIn(RC.SELL_TAKE_PROFIT, r.reason_codes)
        self.assertEqual(r.recommended_order_policy.price_policy, "SELL_TP_BEST_BID")

    def test_sell_stop_loss_distinguished_urgent(self) -> None:
        facts = build_trading_facts("005930", "SELL", exit_reason="stop_loss:-1.20%", best_bid=9_990, best_ask=10_010)
        r = self.reasoner.reason(facts)
        self.assertEqual(r.intent, IntentType.SELL)
        self.assertIn(RC.SELL_STOP_LOSS, r.reason_codes)
        self.assertEqual(r.recommended_order_policy.price_policy, "SELL_STOP_MARKETABLE_BID")
        self.assertEqual(r.recommended_order_policy.urgency, "URGENT")

    def test_sell_hard_stop_distinguished(self) -> None:
        facts = build_trading_facts("005930", "SELL", exit_reason="hard_stop_loss:-8.0%", best_bid=9_990, best_ask=10_010)
        r = self.reasoner.reason(facts)
        self.assertIn(RC.SELL_HARD_STOP, r.reason_codes)

    def test_sell_time_stop_distinguished(self) -> None:
        facts = build_trading_facts("005930", "SELL", exit_reason="time_exit:0.10%", best_bid=9_990, best_ask=10_010)
        r = self.reasoner.reason(facts)
        self.assertIn(RC.SELL_TIME_STOP, r.reason_codes)

    def test_sell_is_never_blocked_by_ontology(self) -> None:
        # Even with a wide spread / no other support, an exit is allowed (RiskManager gates).
        facts = build_trading_facts("005930", "SELL", exit_reason="stop_loss:-2%", best_bid=0.0, best_ask=0.0)
        r = self.reasoner.reason(facts)
        self.assertIn(r.intent, (IntentType.SELL, IntentType.REDUCE))
        self.assertEqual(r.blocked_by, ())

    def test_result_is_json_serializable(self) -> None:
        import json
        r = self.reasoner.reason(_good_buy())
        json.dumps(r.as_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
