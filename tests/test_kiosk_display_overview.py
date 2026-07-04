from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import web as web_module


def _idle_snapshot() -> dict:
    return {
        "progress": {"active": False, "message": ""},
        "is_refreshing": False,
        "learning": {"active": False, "next_collection_at": None},
        "collection_log": [],
    }


def test_kiosk_overview_reports_both_markets_closed_on_weekend() -> None:
    sunday_kst = datetime(2026, 7, 4, 15, 0, tzinfo=timezone.utc)

    with patch("app.web._live_snapshot", return_value=_idle_snapshot()):
        overview = web_module._kiosk_market_overview(sunday_kst)

    assert overview["primary"]["label"] == "양쪽 휴장"
    assert overview["markets"][0]["label"] == "국내 휴장"
    assert overview["markets"][1]["label"] == "미국 휴장"


def test_kiosk_overview_reports_krx_core_session() -> None:
    krx_open = datetime(2026, 6, 30, 0, 30, tzinfo=timezone.utc)

    with patch("app.web._live_snapshot", return_value=_idle_snapshot()):
        overview = web_module._kiosk_market_overview(krx_open)

    assert overview["primary"]["label"] == "국내 정규장"
    assert overview["markets"][0]["label"] == "국내 장 오픈"


def test_kiosk_overview_reports_us_premarket() -> None:
    us_premarket = datetime(2026, 6, 30, 10, 30, tzinfo=timezone.utc)

    with patch("app.web._live_snapshot", return_value=_idle_snapshot()):
        overview = web_module._kiosk_market_overview(us_premarket)

    assert overview["primary"]["label"] == "미국 프리마켓"
    assert overview["markets"][1]["label"] == "미국 프리마켓"


def test_kiosk_overview_reports_news_analysis_busy() -> None:
    snapshot = _idle_snapshot()
    snapshot["progress"] = {"active": True, "message": "뉴스와 차트 데이터를 수집하는 중입니다."}

    with patch("app.web._live_snapshot", return_value=snapshot):
        overview = web_module._kiosk_market_overview(datetime(2026, 7, 4, 15, 0, tzinfo=timezone.utc))

    assert overview["work"]["label"] == "뉴스·데이터 분석 중"
    assert overview["work"]["tone"] == "busy"
