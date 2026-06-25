from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.research import ResearchService
from app.data import extract_focus_sections


class ResearchServiceTest(unittest.TestCase):
    def test_extract_focus_sections_builds_numeric_highlights(self) -> None:
        sections = extract_focus_sections(
            "S&P 500 rose 1.2% today. CPI came at 3.1%. Treasury yield moved to 4.2%."
        )

        self.assertIn("S&P 500 rose 1.2% today.", sections["headline"])
        self.assertGreater(len(sections["numeric_highlights"]), 0)

    def test_empty_config_runs_without_external_sources(self) -> None:
        result = ResearchService().run(
            {
                "known_tickers": {"005930": "Samsung Electronics"},
                "rss_feeds": [],
                "html_pages": [],
                "stooq_symbols": [],
                "fred_series": [],
                "ecos_series": [],
                "opendart_disclosures": [],
            }
        )

        self.assertEqual(result.events, ())
        self.assertEqual(result.skipped_sources, ())

    def test_dynamic_pages_file_source_is_classified(self) -> None:
        result = ResearchService().run(
            {
                "known_tickers": {"005930": "Samsung Electronics"},
                "rss_feeds": [],
                "html_pages": [],
                "dynamic_pages": [
                    {
                        "url": "../data/fixtures/semiconductor_research.html",
                        "title": "Semiconductor dynamic research",
                        "scroll_steps": 1,
                        "wait_ms": 1,
                        "timeout_ms": 1000,
                    }
                ],
                "stooq_symbols": [],
                "yahoo_chart_symbols": [],
                "fred_series": [],
                "ecos_series": [],
                "opendart_disclosures": [],
            },
            base_dir=Path("config").resolve(),
        )

        self.assertGreater(len(result.raw_records), 0)
        self.assertGreater(len(result.events), 0)
        self.assertEqual(result.skipped_sources, ())

    def test_html_page_can_set_non_news_event_type(self) -> None:
        result = ResearchService().run(
            {
                "known_tickers": {"AAPL": "Apple"},
                "rss_feeds": [],
                "html_pages": [
                    {
                        "url": "../data/fixtures/semiconductor_research.html",
                        "title": "Market research page",
                        "event_type": "MARKET",
                    }
                ],
                "dynamic_pages": [],
                "stooq_symbols": [],
                "yahoo_chart_symbols": [],
                "fred_series": [],
                "ecos_series": [],
                "opendart_disclosures": [],
            },
            base_dir=Path("config").resolve(),
        )

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].event_type, "MARKET")

    def test_retry_disabled_records_skipped_without_crashing(self) -> None:
        result = ResearchService().run(
            {
                "known_tickers": {"AAPL": "Apple"},
                "rss_feeds": [],
                "html_pages": [],
                "dynamic_pages": ["https://invalid.invalid/market"],
                "retry_failed_sources": False,
                "stooq_symbols": [],
                "yahoo_chart_symbols": [],
                "fred_series": [],
                "ecos_series": [],
                "opendart_disclosures": [],
            }
        )

        self.assertEqual(result.events, ())
        self.assertEqual(len(result.skipped_sources), 1)
        self.assertIn("dynamic:", result.skipped_sources[0])


if __name__ == "__main__":
    unittest.main()
