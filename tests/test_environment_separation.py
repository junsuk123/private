from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import web
from app.runtime import DataEnvironment, default_environment
from app.schemas.domain import MarketSnapshot, RawSourceRecord, SourceMetadata
from app.storage import LocalResearchStore, ModelArtifactStore


class RealtimeUnifiedEnvironmentTest(unittest.TestCase):
    def test_default_environment_uses_single_realtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_root = os.environ.get("DATA_ROOT")
            old_env = os.environ.get("DATA_ENV")
            try:
                os.environ["DATA_ROOT"] = str(Path(tmp) / "data")
                os.environ["DATA_ENV"] = "realtime"
                env = default_environment()

                self.assertEqual(env.mode, "realtime")
                self.assertEqual(env.store_dir, Path(tmp) / "data" / "store")
                self.assertEqual(env.model_dir, Path(tmp) / "data" / "models")
            finally:
                if old_root is None:
                    os.environ.pop("DATA_ROOT", None)
                else:
                    os.environ["DATA_ROOT"] = old_root
                if old_env is None:
                    os.environ.pop("DATA_ENV", None)
                else:
                    os.environ["DATA_ENV"] = old_env

    def test_legacy_live_and_simulation_helpers_alias_realtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            self.assertEqual(DataEnvironment.live(root).store_dir, root / "store")
            self.assertEqual(DataEnvironment.simulation(root).store_dir, root / "store")
            self.assertEqual(DataEnvironment.live(root).mode, "realtime")
            self.assertEqual(DataEnvironment.simulation(root).mode, "realtime")

    def test_realtime_store_rejects_simulated_market_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp) / "store", mode="realtime")
            simulated = MarketSnapshot(
                ticker="SIM_A",
                market="SIM",
                company_name="Synthetic SIM_A",
                sector="Synthetic",
                last_price=100.0,
                average_daily_trading_value=1_000_000,
                volatility_20d=0.02,
                source=SourceMetadata(
                    source_name="synthetic_ohlcv",
                    retrieved_at=datetime.now(timezone.utc),
                    raw_url="local://synthetic/ohlcv/SIM_A",
                    source_id="synthetic:ohlcv:SIM_A",
                ),
            )

            with self.assertRaises(ValueError):
                store.save_research_result(
                    SimpleNamespace(events=(), raw_records=(), market_snapshots=(simulated,), macro_metrics=())
                )

    def test_model_store_rejects_simulated_artifacts_and_uses_family_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelArtifactStore(Path(tmp) / "models", mode="realtime")

            with self.assertRaises(ValueError):
                store.save_json("synthetic_model", {"weights": [1, 2]}, simulated=True)

            path = store.save_json("ai_semantic:centroid", {"weights": [1, 2]}, model_family="ai_semantic")
            self.assertTrue(path.exists())
            self.assertTrue((Path(tmp) / "models" / "ai_semantic" / "ai_semantic_centroid.latest.json").exists())

    def test_analysis_loads_only_unified_realtime_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp) / "data" / "store", mode="realtime")
            now = datetime.now(timezone.utc)
            record = RawSourceRecord(
                source=SourceMetadata(
                    source_name="rss",
                    retrieved_at=now,
                    raw_url="https://example.test/live",
                    source_id="rss:live",
                ),
                content_type="text/plain",
                payload="realtime input",
            )
            store.save_research_result(
                SimpleNamespace(events=(), raw_records=(record,), market_snapshots=(), macro_metrics=())
            )

            loaded = web._analysis_research_for_current_mode(store)

            self.assertEqual([item.payload for item in loaded.raw_records], ["realtime input"])
            self.assertEqual(web._current_data_policy()["analysis_input_stores"], ["data/store"])


if __name__ == "__main__":
    unittest.main()
