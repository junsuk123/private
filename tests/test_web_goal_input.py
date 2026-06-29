from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.web import _parse_goal_request, app


class WebGoalInputTest(unittest.TestCase):
    def test_accepts_rate_mode_only(self) -> None:
        request = _parse_goal_request(
            {"goal_mode": "rate", "target_return_rate": "5", "target_profit_amount": "", "period_days": "90"}
        )

        self.assertEqual(request.target_return_rate, 0.05)
        self.assertIsNone(request.target_profit_amount)

    def test_accepts_amount_mode_only(self) -> None:
        request = _parse_goal_request(
            {
                "goal_mode": "amount",
                "target_return_rate": "",
                "target_profit_amount": "50000",
                "period_days": "90",
            }
        )

        self.assertIsNone(request.target_return_rate)
        self.assertEqual(request.target_profit_amount, 50_000)

    def test_rejects_both_values(self) -> None:
        with self.assertRaises(HTTPException):
            _parse_goal_request(
                {
                    "goal_mode": "rate",
                    "target_return_rate": "5",
                    "target_profit_amount": "50000",
                    "period_days": "90",
                }
            )

    def test_assess_goal_returns_quick_provisional_result_without_live_context(self) -> None:
        client = TestClient(app)
        snapshot = {
            "context": None,
            "research_result": None,
            "store_summary": {},
            "stored_new_records": {},
            "last_updated": None,
            "last_error": None,
            "is_refreshing": False,
            "progress": {},
            "learning": {},
            "collection_log": [],
            "graph_payload": None,
            "graph_payload_context_id": None,
        }
        with (
            patch("app.web._live_snapshot", return_value=snapshot),
            patch("app.web._get_or_refresh_live", side_effect=AssertionError("assessment should not block on live cache")),
        ):
            response = client.post(
                "/api/assess-goal",
                json={
                    "goal_mode": "rate",
                    "target_return_rate": 2,
                    "target_profit_amount": "",
                    "period_minutes": 20,
                    "period_days": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["provisional"])
        self.assertIn("assessment", payload)


if __name__ == "__main__":
    unittest.main()
