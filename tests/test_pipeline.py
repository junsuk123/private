from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.pipeline import build_analysis_context
from app.cli import run_demo
from app.schemas.domain import AccountSnapshot, MarketSnapshot, RiskRules, SourceMetadata
from app.storage import StoredResearch


class PipelineTest(unittest.TestCase):
    def test_demo_pipeline_produces_auditable_outputs(self) -> None:
        result = run_demo()

        self.assertGreater(result["portfolio_report"].equity, 0)
        self.assertEqual(result["portfolio_report"].equity, 1_000_000)
        self.assertGreater(len(result["graph_triples"]), 0)
        self.assertGreater(len(result["strategy_signals"]), 0)
        self.assertGreater(len(result["order_intents"]), 0)
        self.assertGreater(len(result["risk_results"]), 0)
        self.assertEqual(result["audit_log"], "logs/audit.jsonl")

    def test_live_context_filters_domestic_and_overseas_by_one_share_cash(self) -> None:
        now = datetime.now(timezone.utc)
        source = SourceMetadata(
            source_name="KIS broker quote",
            retrieved_at=now,
            source_type="broker_api",
            trust_level=5,
            observed_at=now,
            is_realtime=True,
            quality_score=1.0,
        )
        stored = StoredResearch(
            events=(),
            raw_records=(),
            market_snapshots=(
                MarketSnapshot("000001", "KOSPI", "Affordable KR", "Technology", 4_000.0, 10_000_000_000, 0.02, source),
                MarketSnapshot("005930", "KOSPI", "Expensive KR", "Technology", 70_000.0, 10_000_000_000, 0.02, source),
                MarketSnapshot("PENNY", "NASDAQ", "Affordable US", "Technology", 2.5, 10_000_000_000, 0.02, source),
                MarketSnapshot("MSFT", "NASDAQ", "Microsoft", "Technology", 367.6, 10_000_000_000, 0.02, source),
            ),
            macro_metrics=(),
            realtime_quotes=(),
            realtime_executions=(),
            graph_triples=(),
            reasoning_paths=(),
        )
        account = AccountSnapshot(
            cash=5_000.0,
            holdings=(),
            cash_by_currency={"KRW": 5_000.0, "USD": 3.22},
            cash_equivalent_krw=9_963.0,
        )

        context = build_analysis_context(
            stored_research=stored,
            account_override=account,
            risk_rules=RiskRules(live_trading_enabled=True),
        )

        self.assertEqual({market.ticker for market in context.markets}, {"000001", "PENNY"})
        self.assertTrue(context.affordability_filter["enabled"])
        filtered_tickers = {item["ticker"] for item in context.affordability_filter["filtered_examples"]}
        self.assertIn("005930", filtered_tickers)
        self.assertIn("MSFT", filtered_tickers)
        self.assertGreaterEqual(context.affordability_filter["by_currency"]["KRW"]["filtered"], 1)
        self.assertEqual(context.affordability_filter["by_currency"]["USD"]["filtered"], 1)


if __name__ == "__main__":
    unittest.main()
