from __future__ import annotations

import unittest

from app.graph.action_aggregator import ActionAggregator
from app.graph.theory_registry import get_theory_registry
from app.graph.theory_vote import PositionContext, TheoryVote


class TheoryAwareOntologyVotingTest(unittest.TestCase):
    def test_conflicting_reversal_and_momentum_do_not_double_count_buy(self) -> None:
        decision = ActionAggregator().decide(
            "005930",
            (
                _vote("005930", "jegadeesh_1990_short_term_reversal", "short_term_reversal", "contrarian", "BUY", "short_intraday", "reversal_cluster", 0.9, 0.8, 0.5),
                _vote("005930", "gao_2018_intraday_momentum", "intraday_momentum", "continuation", "BUY", "late_intraday", "momentum_cluster", 0.9, 0.8, 0.5),
            ),
        )

        self.assertTrue(decision.conflicts)
        self.assertLess(decision.buy_score, 0.72)

    def test_sell_signal_becomes_first_class_action_for_existing_position(self) -> None:
        decision = ActionAggregator().decide(
            "005930",
            (
                _vote("005930", "profit_taking_exit", "risk_management", "profit_taking", "SELL", "position_dependent", "risk_cluster", 0.9, 0.95, 1.0),
            ),
            position_context=PositionContext(has_position=True, current_quantity=10, average_price=50000),
        )

        self.assertEqual(decision.selected_action, "SELL")
        self.assertGreater(decision.sell_score, decision.buy_score)

    def test_sell_signal_without_position_becomes_watch(self) -> None:
        decision = ActionAggregator().decide(
            "005930",
            (
                _vote("005930", "profit_taking_exit", "risk_management", "profit_taking", "SELL", "position_dependent", "risk_cluster", 0.9, 0.95, 1.0),
            ),
        )

        self.assertEqual(decision.selected_action, "WATCH")

    def test_correlated_trend_features_clustered(self) -> None:
        decision = ActionAggregator().decide(
            "005930",
            (
                _vote("005930", "gao_2018_intraday_momentum", "intraday_momentum", "continuation", "BUY", "late_intraday", "trend_cluster", 0.6, 0.8, 0.5),
                _vote("005930", "gao_2018_intraday_momentum", "intraday_momentum", "continuation", "BUY", "late_intraday", "trend_cluster", 0.6, 0.8, 0.5),
            ),
        )

        self.assertEqual(len(decision.evidence_clusters), 1)
        self.assertLess(decision.evidence_clusters[0].compressed_score, 0.48)

    def test_unvalidated_and_validated_theory_weights(self) -> None:
        registry = get_theory_registry()

        self.assertEqual(registry.weight_for("brock_1992_technical_breakout"), 0.1)
        self.assertEqual(registry.weight_for("risk_reduction_exit"), 1.0)

    def test_hold_when_scores_are_too_close(self) -> None:
        decision = ActionAggregator().decide(
            "005930",
            (
                _vote("005930", "gao_2018_intraday_momentum", "intraday_momentum", "continuation", "BUY", "late_intraday", "momentum_cluster", 0.3, 0.8, 0.5),
                _vote("005930", "profit_taking_exit", "risk_management", "profit_taking", "SELL", "position_dependent", "risk_cluster", 0.3, 0.8, 0.5),
            ),
            position_context=PositionContext(has_position=True, current_quantity=1, average_price=1000),
        )

        self.assertEqual(decision.selected_action, "HOLD")


def _vote(
    ticker: str,
    theory_id: str,
    family: str,
    style: str,
    action: str,
    horizon: str,
    cluster: str,
    raw: float,
    confidence: float,
    validation: float,
) -> TheoryVote:
    return TheoryVote(
        ticker=ticker,
        theory_id=theory_id,
        theory_family=family,
        style=style,
        action=action,
        horizon_bucket=horizon,
        expected_holding_minutes=30,
        raw_signal=raw,
        confidence=confidence,
        validation_weight=validation,
        evidence_cluster_id=cluster,
    )


if __name__ == "__main__":
    unittest.main()
