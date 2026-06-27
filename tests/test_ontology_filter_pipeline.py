from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading_pipeline import (
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


if __name__ == "__main__":
    unittest.main()
