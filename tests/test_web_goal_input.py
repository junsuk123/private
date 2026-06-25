from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import HTTPException

from app.web import _parse_goal_request


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


if __name__ == "__main__":
    unittest.main()
