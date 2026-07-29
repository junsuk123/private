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
    _merge_materialized_training_rows,
    _thin_training_rows,
    backfill_live_feature_frames_from_realtime_store,
    build_live_training_rows_from_feature_journal,
    collect_live_feature_frames_from_realtime_store,
    live_training_status,
    train_live_short_horizon_from_collected_features,
)
from app.models.model_artifact_registry import ModelArtifactRegistry


class LiveTrainingPipelineTest(unittest.TestCase):
    def test_materialized_rows_replace_matured_label_for_same_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "training.sqlite3"
            base = _training_row(
                ticker="F",
                as_of="2026-07-28T13:00:00+00:00",
                label=0,
                label_source="triple_barrier_terminal",
            )
            _merge_materialized_training_rows(store, [base])
            matured = {
                **base,
                "label": 1,
                "label_source": "triple_barrier_take_profit",
                "forward_net_return_bps": 25.0,
            }

            rows, stats = _merge_materialized_training_rows(store, [matured])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["label_source"], "triple_barrier_take_profit")
        self.assertEqual(stats["new_rows"], 0)

    def test_training_rows_collapse_sub_five_second_refreshes_per_symbol(self) -> None:
        rows = [
            _training_row(
                ticker="F",
                as_of=f"2026-07-28T13:00:0{second}+00:00",
                label=second % 2,
                label_source="triple_barrier_terminal",
            )
            for second in range(5)
        ]
        rows.append(
            _training_row(
                ticker="396500",
                as_of="2026-07-28T13:00:01+00:00",
                label=1,
                label_source="triple_barrier_take_profit",
            )
        )

        with patch.dict(os.environ, {"LIVE_TRAINING_MIN_ROW_SPACING_SECONDS": "5"}):
            thinned = _thin_training_rows(rows)

        self.assertEqual(len(thinned), 2)
        self.assertEqual({row["ticker"] for row in thinned}, {"F", "396500"})
        self.assertEqual(
            next(row for row in thinned if row["ticker"] == "F")["as_of"],
            "2026-07-28T13:00:04+00:00",
        )

    def test_builds_rows_from_collected_live_feature_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=8)

            rows = build_live_training_rows_from_feature_journal(journal)

        self.assertEqual(len(rows), 7)
        self.assertEqual(set(rows[0]["features"]), set(LIVE_SHORT_HORIZON_SCHEMA.feature_names))
        self.assertIn(rows[0]["label"], {0, 1})

    def test_flat_quote_refresh_frames_are_excluded_from_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=8, flat=True)

            rows = build_live_training_rows_from_feature_journal(journal)

        self.assertEqual(rows, [])

    def test_training_from_collected_frames_creates_latest_when_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=200)
            registry = ModelArtifactRegistry(Path(tmp) / "models")

            # 이 합성 데이터는 20bps 라벨 기준으로 깔끔히 분리되도록 설계됨(운영 기본값은 5bps).
            # 시간기반 홀드아웃(30%)에서 top-k 선택이 동작할 만큼 충분한 표본을 준다.
            with patch.dict(os.environ, {"LIVE_LABEL_MIN_NET_RETURN_BPS": "20"}):
                artifact = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )

            self.assertTrue(artifact["live_eligible"], artifact["reason_codes"])
            self.assertTrue(registry.latest_path.exists())

    def test_unchanged_labelled_dataset_skips_duplicate_training_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=200)
            registry = ModelArtifactRegistry(Path(tmp) / "models")
            with patch.dict(os.environ, {"LIVE_LABEL_MIN_NET_RETURN_BPS": "20"}):
                first = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )
                second = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )

            candidates = tuple(registry.root.glob("live_short_horizon.*.json"))

        self.assertFalse(first.get("training_skipped", False))
        self.assertTrue(second["training_skipped"])
        self.assertEqual(second["skip_reason"], "UNCHANGED_LABELLED_DATASET")
        self.assertEqual(len(candidates), 1)

    def test_changed_feature_values_force_retraining_even_when_labels_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=200)
            registry = ModelArtifactRegistry(Path(tmp) / "models")
            with patch.dict(os.environ, {"LIVE_LABEL_MIN_NET_RETURN_BPS": "20"}):
                first = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )
                frames = [
                    json.loads(line)
                    for line in journal.read_text(encoding="utf-8").splitlines()
                ]
                frames[0]["values"]["orderbook_imbalance"] += 0.01
                journal.write_text(
                    "".join(json.dumps(frame) + "\n" for frame in frames),
                    encoding="utf-8",
                )
                second = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )

        self.assertNotEqual(first["artifact_id"], second["artifact_id"])
        self.assertFalse(second.get("training_skipped", False))

    def test_new_mature_labels_accumulate_in_materialized_training_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=200)
            registry = ModelArtifactRegistry(Path(tmp) / "models")
            with patch.dict(os.environ, {"LIVE_LABEL_MIN_NET_RETURN_BPS": "20"}):
                first = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )
                _write_frames(journal, count=220)
                second = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )

        self.assertGreater(
            second["metrics"]["example_count"],
            first["metrics"]["example_count"],
        )
        self.assertGreater(second["training_data"]["fresh_row_count"], 0)
        self.assertGreater(second["training_data"]["new_materialized_row_count"], 0)
        self.assertEqual(first["training_state"]["mode"], "full")
        self.assertEqual(second["training_state"]["mode"], "incremental")
        self.assertEqual(
            second["training_state"]["parent_artifact_id"],
            first["artifact_id"],
        )
        self.assertLess(
            second["training_state"]["incremental_example_count"],
            second["training_state"]["cumulative_example_count"],
        )

    def test_data_format_change_forces_full_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "frames.jsonl"
            _write_frames(journal, count=200)
            registry = ModelArtifactRegistry(Path(tmp) / "models")
            with patch.dict(
                os.environ,
                {
                    "LIVE_LABEL_MIN_NET_RETURN_BPS": "20",
                    "LIVE_TRAINING_MIN_ROW_SPACING_SECONDS": "5",
                },
            ):
                first = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )
            _write_frames(journal, count=220)
            with patch.dict(
                os.environ,
                {
                    "LIVE_LABEL_MIN_NET_RETURN_BPS": "20",
                    "LIVE_TRAINING_MIN_ROW_SPACING_SECONDS": "15",
                },
            ):
                second = train_live_short_horizon_from_collected_features(
                    journal_path=journal,
                    registry=registry,
                )

        self.assertEqual(first["training_state"]["mode"], "full")
        self.assertEqual(second["training_state"]["mode"], "full")
        self.assertEqual(
            second["training_state"]["full_retrain_reason"],
            "NO_COMPATIBLE_PARENT_STATE",
        )

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

    def test_backfills_multiple_causal_frames_from_retained_events(self) -> None:
        now = datetime.now(timezone.utc) - timedelta(minutes=1)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rt.sqlite3"
            journal = Path(tmp) / "features.jsonl"
            store = RealtimeMarketDataStore(db_path)
            for index in range(12):
                moment = now + timedelta(seconds=index * 5)
                store.save_ticks(
                    (
                        RealtimeTradeTick(
                            symbol="396500",
                            exchange_timestamp=moment,
                            received_at=moment,
                            source=KIS_REALTIME_SOURCE,
                            price=10000 + index,
                            volume=100 + index,
                            sequence_key=f"backfill-tick:{index}",
                        ),
                    )
                )
                store.save_orderbooks(
                    (
                        RealtimeOrderbookSnapshot(
                            symbol="396500",
                            exchange_timestamp=moment,
                            received_at=moment,
                            source=KIS_REALTIME_SOURCE,
                            levels=(
                                OrderbookLevel(
                                    9999 + index,
                                    1000,
                                    10001 + index,
                                    900,
                                ),
                            ),
                            sequence_key=f"backfill-book:{index}",
                        ),
                    )
                )

            result = backfill_live_feature_frames_from_realtime_store(
                db_path=db_path,
                journal_path=journal,
                stride_seconds=5,
                maximum_frames=20,
                lookback_hours=1,
            )

            self.assertGreaterEqual(result["built"], 10)
            frames = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(
                {frame["feature_schema_hash"] for frame in frames},
                {LIVE_SHORT_HORIZON_SCHEMA.schema_hash},
            )

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


def _write_frames(path: Path, *, count: int, flat: bool = False) -> None:
    names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
    with path.open("w", encoding="utf-8") as file:
        for index in range(count):
            # 균형(50/50)·명확 분리 데이터: 시간기반 홀드아웃 top-k 평가에서도 적격이 되도록.
            # 스프레드(=비용)는 양쪽 동일하게 낮게 두어 라벨이 순수히 전방수익 부호로 결정된다.
            positive_phase = index % 2 == 0
            values = {name: 0.0 for name in names}
            values["return_30s"] = 0.0 if flat else (0.005 if positive_phase else -0.005)
            values["return_1m"] = 0.0 if flat else (0.010 if positive_phase else -0.010)
            values["return_3m"] = 0.0 if flat else (0.015 if positive_phase else -0.015)
            values["distance_from_vwap"] = 0.0 if flat else (0.003 if positive_phase else -0.003)
            values["spread_bps"] = 3.0
            values["orderbook_imbalance"] = 0.4 if positive_phase else -0.4
            values["bid_depth"] = 300000.0 if positive_phase else 60000.0
            values["ask_depth"] = 100000.0 if positive_phase else 200000.0
            values["depth_ratio"] = values["bid_depth"] / values["ask_depth"]
            values["liquidity_score"] = 0.9 if positive_phase else 0.2
            values["realized_volatility_3m"] = 0.0 if flat else 0.002
            values["max_drop_3m"] = 0.0 if flat else (0.0 if positive_phase else -0.01)
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


def _training_row(
    *,
    ticker: str,
    as_of: str,
    label: int,
    label_source: str,
) -> dict:
    return {
        "features": {name: 0.0 for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names},
        "label": label,
        "forward_net_return_bps": 10.0 if label else -10.0,
        "gross_forward_return_bps": 12.0 if label else -8.0,
        "label_source": label_source,
        "ticker": ticker,
        "as_of": as_of,
        "source": "test",
        "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
    }


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
