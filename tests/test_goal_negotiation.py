from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.goals import GoalRequest, assess_goal, build_compromise_goals
from app.pipeline import build_analysis_context


class GoalNegotiationTest(unittest.TestCase):
    def test_assesses_return_goal_and_builds_compromises(self) -> None:
        context = build_analysis_context()
        assessment = assess_goal(
            GoalRequest(target_return_rate=0.05, target_profit_amount=None, period_days=90),
            context.account,
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )
        compromises = build_compromise_goals(assessment)

        self.assertGreaterEqual(assessment.feasibility_percent, 3)
        self.assertLessEqual(assessment.feasibility_percent, 96)
        self.assertEqual(len(compromises), 4)
        self.assertTrue(any(goal.label == "Balanced compromise" for goal in compromises))

    def test_assesses_profit_amount_goal(self) -> None:
        context = build_analysis_context()
        assessment = assess_goal(
            GoalRequest(target_return_rate=None, target_profit_amount=500_000, period_days=120),
            context.account,
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )

        self.assertAlmostEqual(assessment.requested_profit_amount, 500_000)
        self.assertGreater(len(assessment.ontology_relations), 0)

    def test_intraday_feasibility_changes_with_goal_difficulty(self) -> None:
        context = build_analysis_context()
        easy = assess_goal(
            GoalRequest(target_return_rate=0.001, target_profit_amount=None, period_days=1, period_minutes=390),
            context.account,
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )
        aggressive = assess_goal(
            GoalRequest(target_return_rate=0.02, target_profit_amount=None, period_days=1, period_minutes=390),
            context.account,
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )

        self.assertGreater(easy.feasibility_percent, aggressive.feasibility_percent)
        self.assertGreater(easy.feasibility_percent, 3)
        self.assertEqual(easy.period_minutes, 390)

    def test_extreme_return_goal_does_not_overflow(self) -> None:
        context = build_analysis_context()
        assessment = assess_goal(
            GoalRequest(target_return_rate=1_000_000.0, target_profit_amount=None, period_days=1),
            context.account,
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )

        self.assertEqual(assessment.annualized_required_return, 1_000_000.0)
        self.assertGreaterEqual(assessment.feasibility_percent, 3)


if __name__ == "__main__":
    unittest.main()
