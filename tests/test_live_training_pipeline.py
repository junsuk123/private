from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REALTIME_SOURCE, OrderbookLevel, RealtimeOrderbookSnapshot, RealtimeTradeTick
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.live_training_pipeline import (
    build_live_training_rows_from_feature_journal,
    collect_live_feature_frames_from_realtime_store,
    live_training_status,
    train_live_short_horizon_from_collected_features,
)
from app.models.model_artifact_registry import ModelArtifactRegistry


class LiveTrainingPipelineTest(unittest.TestCase):
    def test_builds_rows_from_collected_live_feature_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=8)

            rows = build_live_training_rows_from_feature_journal(journal)

        self.assertEqual(len(rows), 7)
        self.assertEqual(set(rows[0]["features"]), set(LIVE_SHORT_HORIZON_SCHEMA.feature_names))
        self.assertIn(rows[0]["label"], {0, 1})

    def test_training_from_collected_frames_creates_latest_when_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=80)
            registry = ModelArtifactRegistry(Path(tmp) / "models")

            # 이 합성 데이터는 20bps 라벨 기준으로 깔끔히 분리되도록 설계됨(운영 기본값은 5bps).
            with patch.dict(os.environ, {"LIVE_LABEL_MIN_NET_RETURN_BPS": "20"}):
                artifact = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )

            self.assertTrue(artifact["live_eligible"], artifact["reason_codes"])
            self.assertTrue(registry.latest_path.exists())

    def test_insufficient_collected_frames_do_not_create_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=1)
            registry = ModelArtifactRegistry(Path(tmp) / "models")

            artifact = train_live_short_horizon_from_collected_features(
                journal_path=journal,
                registry=registry,
            )

            self.assertFalse(artifact["live_eligible"])
            self.assertFalse(registry.latest_path.exists())

    def test_collects_live_feature_frames_from_realtime_store(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rt.sqlite3"
            journal = Path(tmp) / "features.jsonl"
            store = RealtimeMarketDataStore(db_path)
            _seed_realtime_store(store, now)

            result = collect_live_feature_frames_from_realtime_store(
                db_path=db_path,
                journal_path=journal,
            )

            self.assertEqual(result["built"], 1)
            self.assertTrue(journal.exists())

    def test_status_explains_missing_live_training_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = live_training_status(
                db_path=Path(tmp) / "missing.sqlite3",
                journal_path=Path(tmp) / "missing.jsonl",
                registry=ModelArtifactRegistry(Path(tmp) / "models"),
            )

        self.assertFalse(status["realtime_store_exists"])
        self.assertEqual(status["training_rows"], 0)
        self.assertEqual(status["feature_frame_lines"], 0)


def _write_frames(path: Path, *, count: int) -> None:
    names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
    with path.open("w", encoding="utf-8") as file:
        for index in range(count):
            positive_phase = index % 4 in {1, 2}
            values = {name: 0.0 for name in names}
            values["return_30s"] = 0.004 if positive_phase else -0.004
            values["return_1m"] = 0.006 if positive_phase else -0.006
            values["return_3m"] = 0.01 if positive_phase else -0.01
            values["distance_from_vwap"] = 0.003 if positive_phase else -0.003
            values["spread_bps"] = 3.0 if positive_phase else 35.0
            values["orderbook_imbalance"] = 0.4 if positive_phase else -0.4
            values["bid_depth"] = 300000.0 if positive_phase else 60000.0
            values["ask_depth"] = 100000.0 if positive_phase else 200000.0
            values["depth_ratio"] = values["bid_depth"] / values["ask_depth"]
            values["liquidity_score"] = 0.9 if positive_phase else 0.2
            values["realized_volatility_3m"] = 0.002
            values["max_drop_3m"] = 0.0 if positive_phase else -0.01
            values["cost_to_volatility_ratio"] = 0.15 if positive_phase else 2.0
            values["principal_cushion_ratio"] = 1.0
            payload = {
                "symbol": "005930",
                "decision_time": f"2026-06-29T09:{index:02d}:00+00:00",
                "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
                "source_record_ids": [f"tick-{index}"],
                "values": values,
            }
            file.write(json.dumps(payload, sort_keys=True) + "\n")


def _seed_realtime_store(store: RealtimeMarketDataStore, now: datetime) -> None:
    store.save_ticks(
        tuple(
            RealtimeTradeTick(
                symbol="005930",
                exchange_timestamp=now - timedelta(seconds=120 - index * 10),
                received_at=now - timedelta(seconds=120 - index * 10),
                source=KIS_REALTIME_SOURCE,
                price=70000 + index * 10,
                volume=100 + index,
                sequence_key=f"tick:{index}",
            )
            for index in range(13)
        )
    )
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                symbol="005930",
                exchange_timestamp=now,
                received_at=now,
                source=KIS_REALTIME_SOURCE,
                levels=(OrderbookLevel(70100, 1000, 70150, 800),),
                sequence_key="book:1",
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
