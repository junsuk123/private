from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.sample_collectors import collect_sample_account, collect_sample_market
from app.indicators import build_sample_indicators
from app.risk import RiskManager
from app.schemas import AccountSnapshot, RiskRules
from app.strategy.rule_based import generate_order_intents, generate_strategy_signals
from app.graph.builders import build_market_graph


class RiskManagerTest(unittest.TestCase):
    def test_keeps_live_trading_disabled_but_allows_paper_final_order(self) -> None:
        account = AccountSnapshot(cash=10_000_000, holdings=())
        markets = collect_sample_market()
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators)
        signals = generate_strategy_signals(markets, indicators, graph)
        intents = generate_order_intents(markets, indicators, signals)

        rules = RiskRules(max_sector_weight=0.50)
        result = RiskManager(rules).validate(intents[0], account, markets[0])

        self.assertTrue(result.checks["live_trading_disabled"])
        self.assertIsNotNone(result.final_order)
        self.assertTrue(result.checks["deposit_limit_check"])
        self.assertTrue(result.final_order.manual_approval_required)

    def test_rejects_when_deposit_is_too_small_for_one_share(self) -> None:
        account = collect_sample_account()
        markets = collect_sample_market()
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators)
        signals = generate_strategy_signals(markets, indicators, graph)
        intent = generate_order_intents(markets, indicators, signals)[0]

        result = RiskManager().validate(intent, account, markets[0])

        self.assertFalse(result.approved)
        self.assertIn("quantity_positive", result.rejection_reasons)

    def test_rejects_duplicate_pending_order(self) -> None:
        account = collect_sample_account()
        markets = collect_sample_market()
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators)
        signals = generate_strategy_signals(markets, indicators, graph)
        intent = generate_order_intents(markets, indicators, signals)[0]

        result = RiskManager().validate(
            intent,
            account,
            markets[0],
            existing_pending_tickers={intent.ticker},
        )

        self.assertFalse(result.approved)
        self.assertIn("duplicate_order_check", result.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
