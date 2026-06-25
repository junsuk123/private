from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.classifier import classify_text_event, source_now
from app.data.public_collectors import RssNewsCollector
from app.graph import OntologyReasoner
from app.graph.runtime import get_ontology_runtime, reset_ontology_runtime_cache
from app.graph.builders import build_market_graph
from app.indicators import build_sample_indicators
from app.data.sample_collectors import collect_sample_market


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, url: str, params: dict | None = None):
        class Response:
            def __init__(self, url: str, text: str) -> None:
                self.url = url
                self.status = 200
                self.text = text

        return Response(url, self.text)


class MappingClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses

    def get_text(self, url: str, params: dict | None = None):
        class Response:
            def __init__(self, url: str, text: str) -> None:
                self.url = url
                self.status = 200
                self.text = text

        return Response(url, self.responses[url])


class ResearchAndReasoningTest(unittest.TestCase):
    def test_classifier_extracts_ticker_and_sentiment(self) -> None:
        event = classify_text_event(
            title="005930 reports profit growth",
            body="Samsung Electronics 005930 semiconductor profit growth is strong.",
            source=source_now("unit", "local://unit", "unit:1"),
            known_tickers={"005930": "Samsung Electronics"},
        )

        self.assertEqual(event.tickers, ("005930",))
        self.assertEqual(event.companies, ("Samsung Electronics",))
        self.assertEqual(event.sentiment, "POSITIVE")
        self.assertIn("Semiconductor", event.sectors)

    def test_classifier_detects_global_ticker_formats(self) -> None:
        event = classify_text_event(
            title="NVIDIA and Apple lead Nasdaq higher",
            body="NVDA gained after AI demand, while Apple (AAPL) also moved up.",
            source=source_now("unit", "local://unit", "unit:global-1"),
            known_tickers={
                "NVDA": "NVIDIA",
                "AAPL": "Apple",
                "005930.KS": "Samsung Electronics",
            },
        )

        self.assertEqual(event.tickers, ("AAPL", "NVDA"))
        self.assertEqual(event.companies, ("Apple", "NVIDIA"))

    def test_rss_collector_classifies_items(self) -> None:
        rss = """<?xml version="1.0"?>
        <rss><channel><item>
          <title>000660 memory profit growth</title>
          <description>SK hynix 000660 reports semiconductor growth.</description>
          <link>https://example.test/news/1</link>
          <pubDate>Wed, 10 Jun 2026 09:00:00 GMT</pubDate>
        </item></channel></rss>"""
        events = RssNewsCollector(FakeClient(rss)).collect(
            "https://example.test/rss",
            {"000660": "SK hynix"},
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tickers, ("000660",))
        self.assertEqual(events[0].sentiment, "POSITIVE")

    def test_rss_collector_can_fetch_article_body_and_raw_record(self) -> None:
        rss_url = "https://example.test/rss"
        article_url = "https://example.test/news/1"
        rss = f"""<?xml version="1.0"?>
        <rss><channel><item>
          <title>NVDA market update</title>
          <description>NVIDIA shares move.</description>
          <link>{article_url}</link>
          <pubDate>Wed, 10 Jun 2026 09:00:00 GMT</pubDate>
        </item></channel></rss>"""
        article = "<html><body><main>NVIDIA NVDA semiconductor profit growth and record demand.</main></body></html>"
        result = RssNewsCollector(MappingClient({rss_url: rss, article_url: article})).collect_with_articles(
            rss_url,
            {"NVDA": "NVIDIA"},
            fetch_articles=True,
            article_limit=1,
        )

        self.assertEqual(len(result.events), 1)
        self.assertEqual(len(result.raw_records), 1)
        self.assertEqual(result.events[0].tickers, ("NVDA",))
        self.assertEqual(result.events[0].sentiment, "POSITIVE")
        self.assertEqual(result.events[0].source.source_name, "html")

    def test_reasoner_builds_paths_from_graph(self) -> None:
        markets = collect_sample_market()
        indicators = build_sample_indicators(markets)
        event = classify_text_event(
            title="005930 wins major HBM contract",
            body="Samsung Electronics 005930 semiconductor contract and profit growth.",
            source=source_now("unit", "local://unit", "unit:2"),
            known_tickers={"005930": "Samsung Electronics"},
        )
        graph = build_market_graph(markets, indicators, (event,))
        reasoner = OntologyReasoner(graph)
        reasoner.infer()
        paths = reasoner.build_reasoning_paths(("005930",))

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].ticker, "005930")
        self.assertGreater(len(paths[0].supporting_triples), 0)

    def test_ontology_runtime_requests_npu_and_falls_back_without_openvino(self) -> None:
        reset_ontology_runtime_cache()
        with patch.dict(os.environ, {"ONTOLOGY_ACCELERATOR": "NPU"}), patch(
            "importlib.util.find_spec", return_value=None
        ):
            runtime = get_ontology_runtime()

        self.assertEqual(runtime.requested_backend, "NPU")
        self.assertEqual(runtime.active_backend, "CPU")
        self.assertFalse(runtime.uses_npu)
        self.assertIsNotNone(runtime.fallback_reason)
        reset_ontology_runtime_cache()


if __name__ == "__main__":
    unittest.main()
