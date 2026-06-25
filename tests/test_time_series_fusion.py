from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.pipeline import build_analysis_context
from app.schemas.domain import (
    ClassifiedEvent,
    EventType,
    MarketSnapshot,
    RawSourceRecord,
    RealtimeExecution,
    RealtimeQuote,
    SentimentDirection,
    SourceMetadata,
)
from app.time_series import build_time_synchronized_frames


class TimeSeriesFusionTest(unittest.TestCase):
    def test_records_are_fused_by_ticker_and_time_bucket(self) -> None:
        base = datetime(2026, 6, 16, 9, 30, tzinfo=timezone.utc)
        market_source = SourceMetadata("unit_market", base, source_id="market:TEST")
        article_source = SourceMetadata("unit_news", base + timedelta(minutes=2), "https://example.test/news", "news:TEST")
        realtime_source = SourceMetadata("unit_realtime", base + timedelta(minutes=3), source_id="rt:TEST")
        market = MarketSnapshot("TEST", "SIM", "Test Corp", "Tech", 100.0, 5_000_000_000, 0.03, market_source)
        event = ClassifiedEvent(
            event_id="event-1",
            event_type=EventType.NEWS,
            title="TEST receives major order",
            summary="Material positive order flow.",
            companies=("Test Corp",),
            tickers=("TEST",),
            sectors=("Tech",),
            sentiment=SentimentDirection.POSITIVE,
            event_date=base + timedelta(minutes=2),
            source=article_source,
            classification_confidence=0.9,
        )
        raw = RawSourceRecord(article_source, "text/html", "full article text")
        quote = RealtimeQuote("TEST", "SIM", base + timedelta(minutes=4), 103.0, change_rate=0.03, source=realtime_source)
        execution = RealtimeExecution("TEST", "SIM", base + timedelta(minutes=5), 103.0, 10, "BUY", "trade-1", realtime_source)

        frames = build_time_synchronized_frames(
            markets=(market,),
            events=(event,),
            raw_records=(raw,),
            realtime_quotes=(quote,),
            realtime_executions=(execution,),
            bucket_minutes=15,
        )

        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame.ticker, "TEST")
        self.assertEqual(frame.bucket_start, base)
        self.assertEqual(frame.market_snapshot, market)
        self.assertEqual(frame.events, (event,))
        self.assertEqual(frame.raw_records, (raw,))
        self.assertEqual(frame.realtime_quotes, (quote,))
        self.assertEqual(frame.realtime_executions, (execution,))
        self.assertGreater(frame.impact_score, 0)

    def test_analysis_context_exposes_time_synchronized_frames_and_graph_edges(self) -> None:
        context = build_analysis_context()
        self.assertGreater(len(context.temporal_frames), 0)
        triples = context.graph.triples()
        self.assertTrue(any(triple.predicate == "hasTimeFrame" for triple in triples))
        self.assertTrue(any(triple.predicate == "containsEvent" for triple in triples))


if __name__ == "__main__":
    unittest.main()
