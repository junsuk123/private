from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.http_client import _robots_can_fetch


def test_robots_prefers_more_specific_allow_rule() -> None:
    rules = """User-agent: *
Disallow: /
Allow: /portal/
"""

    assert _robots_can_fetch(rules, "Mozilla/5.0", "https://example.test/portal/news.rss")
    assert not _robots_can_fetch(rules, "Mozilla/5.0", "https://example.test/private")


def test_robots_equal_specificity_prefers_allow() -> None:
    rules = """User-agent: *
Disallow: /feed
Allow: /feed
"""

    assert _robots_can_fetch(rules, "Mozilla/5.0", "https://example.test/feed")
