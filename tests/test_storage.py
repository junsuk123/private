from __future__ import annotations

import sys
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.pipeline import build_analysis_context
from app.research import ResearchService
from app.schemas.domain import (
    MacroMetricRecord,
    RawSourceRecord,
    RealtimeExecution,
    RealtimeQuote,
    SourceMetadata,
)
from app.storage import LocalResearchStore


class StorageTest(unittest.TestCase):
    def test_transient_sqlite_writer_lock_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp))
            attempts = 0

            def operation(_connection):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise sqlite3.OperationalError("database is locked")
                return 7

            with patch("app.storage.local_store.time.sleep") as sleep:
                result = store._write_with_retry(operation)

            self.assertEqual(result, 7)
            self.assertEqual(attempts, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_low_frequency_macro_metric_uses_long_retention_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp), retention_days=7)
            observed_at = datetime.now(timezone.utc) - timedelta(days=90)
            metric = MacroMetricRecord(
                name="monthly_cpi",
                value=123.4,
                observed_at=observed_at,
                source=SourceMetadata(
                    source_name="fred_public_csv",
                    retrieved_at=datetime.now(timezone.utc),
                    raw_url="https://fred.example/cpi.csv",
                    source_id="fred:CPI",
                ),
            )

            saved = store.save_research_result(
                SimpleNamespace(
                    events=(),
                    raw_records=(),
                    market_snapshots=(),
                    macro_metrics=(metric,),
                )
            )

            self.assertEqual(saved["macro_metrics"], 1)
            self.assertEqual(store.summary()["macro_metrics"], 1)

    def test_live_analysis_loader_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp))
            now = datetime.now(timezone.utc)
            records = tuple(
                RawSourceRecord(
                    source=SourceMetadata(
                        source_name="unit",
                        retrieved_at=now + timedelta(seconds=index),
                        raw_url=f"https://example.test/{index}",
                        source_id=f"raw:{index}",
                    ),
                    content_type="text/plain",
                    payload=str(index),
                )
                for index in range(8)
            )
            store.save_research_result(
                SimpleNamespace(
                    events=(),
                    raw_records=records,
                    market_snapshots=(),
                    macro_metrics=(),
                )
            )

            with patch.dict("os.environ", {"LIVE_RESEARCH_RAW_LIMIT": "3"}):
                loaded = store.load_live_analysis_inputs()

            self.assertEqual(len(loaded.raw_records), 3)
            self.assertEqual(loaded.raw_records[0].payload, "7")

    def test_store_saves_and_loads_research_for_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp))
            result = ResearchService().run_from_config(Path("config/research_sources.demo.json"))
            saved = store.save_research_result(result)
            loaded = store.load()
            context = build_analysis_context(stored_research=loaded)
            graph_saved = store.save_graph_and_reasoning(context.graph.triples(), context.reasoning_paths)

            self.assertGreater(saved["events"], 0)
            self.assertGreater(len(loaded.events), 0)
            self.assertGreater(len(context.graph.triples()), 0)
            self.assertGreater(graph_saved["graph_triples"], 0)

    def test_raw_records_saved_per_retrieved_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp))
            base = datetime.now(timezone.utc)
            record1 = RawSourceRecord(
                source=SourceMetadata(
                    source_name="dynamic_html",
                    retrieved_at=base,
                    raw_url="https://example.test/dynamic",
                    source_id="dynamic:example",
                ),
                content_type="text/html",
                payload="alpha",
            )
            record2 = RawSourceRecord(
                source=SourceMetadata(
                    source_name="dynamic_html",
                    retrieved_at=base + timedelta(seconds=3),
                    raw_url="https://example.test/dynamic",
                    source_id="dynamic:example",
                ),
                content_type="text/html",
                payload="beta",
            )

            result1 = SimpleNamespace(events=(), raw_records=(record1,), market_snapshots=(), macro_metrics=())
            result2 = SimpleNamespace(events=(), raw_records=(record2,), market_snapshots=(), macro_metrics=())
            save1 = store.save_research_result(result1)
            save2 = store.save_research_result(result2)

            self.assertEqual(save1["raw_records"], 1)
            self.assertEqual(save2["raw_records"], 1)
            self.assertEqual(store.summary()["raw_records"], 2)
            self.assertTrue((Path(tmp) / "research.sqlite3").exists())

    def test_store_deduplicates_and_prunes_stale_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp), retention_days=7)
            fresh = datetime.now(timezone.utc)
            old = fresh - timedelta(days=30)
            fresh_record = RawSourceRecord(
                source=SourceMetadata(
                    source_name="unit",
                    retrieved_at=fresh,
                    raw_url="https://example.test/fresh",
                    source_id="raw:fresh",
                ),
                content_type="text/plain",
                payload="fresh",
            )
            old_record = RawSourceRecord(
                source=SourceMetadata(
                    source_name="unit",
                    retrieved_at=old,
                    raw_url="https://example.test/old",
                    source_id="raw:old",
                ),
                content_type="text/plain",
                payload="old",
            )
            result = SimpleNamespace(
                events=(),
                raw_records=(fresh_record, fresh_record, old_record),
                market_snapshots=(),
                macro_metrics=(),
            )

            saved = store.save_research_result(result)
            loaded = store.load()

            self.assertEqual(saved["raw_records"], 1)
            self.assertEqual(len(loaded.raw_records), 1)
            self.assertEqual(loaded.raw_records[0].payload, "fresh")

    def test_store_saves_realtime_quotes_and_executions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(Path(tmp))
            observed_at = datetime.now(timezone.utc)
            source = SourceMetadata(
                source_name="unit_realtime",
                retrieved_at=observed_at,
                raw_url="local://unit/realtime",
                source_id="unit:realtime",
            )
            quote = RealtimeQuote(
                ticker="TEST",
                market="KRX",
                observed_at=observed_at,
                last_price=123.45,
                bid_price=123.4,
                ask_price=123.5,
                volume=1000,
                source=source,
            )
            execution = RealtimeExecution(
                ticker="TEST",
                market="KRX",
                executed_at=observed_at,
                price=123.45,
                quantity=7,
                side="BUY",
                trade_id="trade-1",
                source=source,
            )

            saved1 = store.save_realtime_records((quote,), (execution,))
            saved2 = store.save_realtime_records((quote,), (execution,))
            loaded = store.load()
            summary = store.summary()

            self.assertEqual(saved1["realtime_quotes"], 1)
            self.assertEqual(saved1["realtime_executions"], 1)
            self.assertEqual(saved2["realtime_quotes"], 0)
            self.assertEqual(saved2["realtime_executions"], 0)
            self.assertEqual(summary["realtime_quotes"], 1)
            self.assertEqual(summary["realtime_executions"], 1)
            self.assertEqual(loaded.realtime_quotes[0].last_price, 123.45)
            self.assertEqual(loaded.realtime_executions[0].trade_id, "trade-1")


if __name__ == "__main__":
    unittest.main()
