from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading_pipeline import (
    LightweightMarketSnapshot,
    build_lightweight_market_snapshots_from_markets,
    build_lightweight_market_snapshots,
    ontology_filter_1,
    universe_from_tickers,
    validate_lightweight_snapshots_for_live,
)
from app.schemas.domain import MarketSnapshot, SourceMetadata
from datetime import datetime, timezone


class OntologyFilterPipelineTest(unittest.TestCase):
    def test_filter_1_reduces_universe_before_chart_scope(self) -> None:
        tickers = tuple(f"SIM{i:04d}" for i in range(500))
        snapshots = build_lightweight_market_snapshots(universe_from_tickers(tickers), seed=11)

        result = ontology_filter_1(snapshots, target_count=80)

        self.assertEqual(result.full_universe_count, 500)
        self.assertGreaterEqual(len(result.candidate_stocks), 20)
        self.assertLessEqual(len(result.candidate_stocks), 80)
        self.assertEqual(result.chart_fetch_scope, result.candidate_stocks)
        self.assertEqual(result.api_call_count, 0)
        self.assertTrue(all(trace.stage == "ontology_filter_1" for trace in result.traces))

    def test_pseudo_features_marked_synthetic_and_rejected_for_live(self) -> None:
        snapshots = build_lightweight_market_snapshots(universe_from_tickers(("SIM0001",)), seed=11)

        self.assertTrue(snapshots[0].is_synthetic)
        self.assertIn("price_change_rate", snapshots[0].synthetic_fields)
        allowed, reasons = validate_lightweight_snapshots_for_live(snapshots)

        self.assertFalse(allowed)
        self.assertTrue(any("synthetic_fields" in reason for reason in reasons))

    def test_market_based_lightweight_snapshot_marks_hash_fields_estimated(self) -> None:
        market = MarketSnapshot(
            ticker="AAPL",
            market="US",
            company_name="Apple",
            sector="Technology",
            last_price=100.0,
            average_daily_trading_value=5_000_000_000,
            volatility_20d=0.03,
            source=SourceMetadata(
                source_name="kis_broker_api",
                retrieved_at=datetime.now(timezone.utc),
                source_type="broker_api",
                trust_level=5,
                quality_score=0.95,
                is_realtime=True,
            ),
        )

        snapshot = build_lightweight_market_snapshots_from_markets((market,))[0]

        self.assertFalse(snapshot.is_synthetic)
        self.assertIn("price_change_rate", snapshot.estimated_fields)
        self.assertIn("current_price", snapshot.measured_fields)
        self.assertGreater(snapshot.quality_score, 0.0)

    def test_krx_sub_1000_won_rule_rejects_domestic_low_price_candidates(self) -> None:
        snapshots = (
            _snapshot("005930", "KRX", 900.0),
            _snapshot("AAPL", "US", 900.0),
            _snapshot("000660", "KRX", 50_000.0),
        )

        with patch.dict(
            "os.environ",
            {
                "KRX_SUB_1000_WON_RULE_ENABLED": "true",
                "ONTOLOGY_NPU_ENABLED": "false",
            },
            clear=False,
        ):
            result = ontology_filter_1(snapshots, target_count=3, min_trading_value=1, min_liquidity_score=0)

        self.assertNotIn("005930", result.candidate_stocks)
        self.assertIn("005930", result.rejected_stocks)
        self.assertIn("AAPL", result.candidate_stocks)
        self.assertEqual(result.metrics["krx_sub_1000_won_rejected"], 1)
        low_price_trace = next(trace for trace in result.traces if trace.stock_code == "005930")
        self.assertIn("KRX_SUB_1000_WON_DELISTING_RISK", low_price_trace.fired_rules)
        self.assertIn("KRW 1,000", low_price_trace.reason)


if __name__ == "__main__":
    unittest.main()


def _snapshot(ticker: str, market: str, price: float) -> LightweightMarketSnapshot:
    return LightweightMarketSnapshot(
        ticker=ticker,
        market=market,
        sector="Technology",
        current_price=price,
        price_change_rate=0.03,
        trading_value=5_000_000_000,
        trading_volume=5_000_000,
        volume_change_rate=1.2,
        market_cap=1_000_000_000_000,
        foreign_net_buy=200_000_000,
        institution_net_buy=100_000_000,
        retail_net_buy=-50_000_000,
        program_net_buy=20_000_000,
        short_net_change=0.0,
        upper_limit_near=False,
        new_52week_high=True,
        halt_status=False,
        management_stock_status=False,
        liquidity_score=1.0,
        quality_score=1.0,
    )
