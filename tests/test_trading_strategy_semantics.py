from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.graph.reasoning_rules import build_semantic_reasoning_paths
from app.graph.semantic_builder import build_semantic_feature_graph
from app.graph.trading_strategy_semantics import (
    risk_manager_ontology_tags,
    semantic_features_from_strategy_candidates,
)
from app.strategy.candidates import StrategyCandidate


class TradingStrategySemanticsTest(unittest.TestCase):
    def test_short_term_reversal_tags_support_buy_candidate(self) -> None:
        candidate = _candidate(
            ontology_tags=[
                "ShortTermOverreaction",
                "LiquiditySupportedReversal",
                "CostEfficientReversal",
            ],
            features={"target_net_return": 0.003, "net_expected_return_after_cost": 0.006},
        )

        features = semantic_features_from_strategy_candidates((candidate,))
        names = {feature.feature_name for feature in features}
        graph = build_semantic_feature_graph((), features)
        paths = build_semantic_reasoning_paths(features)

        self.assertIn("ShortTermReversalBuy", names)
        self.assertIn("CostEfficientTrade", names)
        self.assertTrue(graph.matching(subject="005930", predicate="supportsSignal", object_="BuyCandidate"))
        self.assertTrue(graph.matching(subject="005930", predicate="supportsSignal", object_="TradeAllowed"))
        self.assertEqual(paths[0].strategy_signal, "TradeAllowed")

    def test_cost_and_spread_risks_create_trade_forbidden_tags(self) -> None:
        candidate = _candidate(
            ontology_tags=["ShortTermOverreaction", "LiquiditySupportedReversal", "SpreadTooWide"],
            features={"target_net_return": 0.003, "net_expected_return_after_cost": 0.001},
        )

        features = semantic_features_from_strategy_candidates((candidate,))
        names = {feature.feature_name for feature in features}
        graph = build_semantic_feature_graph((), features)
        paths = build_semantic_reasoning_paths(features)
        risk_tags = risk_manager_ontology_tags(features)

        self.assertIn("SpreadTooWide", names)
        self.assertIn("CostBurdenHigh", names)
        self.assertTrue(graph.matching(subject="005930", predicate="increasesRiskOf", object_="TradeForbidden"))
        self.assertIn("TradeForbidden", risk_tags)
        self.assertIn("SpreadTooWide", risk_tags)
        self.assertEqual(paths[0].strategy_signal, "TradeForbidden")

    def test_live_without_reality_check_creates_no_validation_risk(self) -> None:
        candidate = _candidate(
            strategy_family="intraday_momentum",
            signal_name="gao_2018_opening_return_momentum",
            ontology_tags=[
                "OpeningReturnStrength",
                "VolumeConfirmedMomentum",
                "MarketDirectionAligned",
            ],
            features={"target_net_return": 0.003, "net_expected_return_after_cost": 0.006},
        )

        features = semantic_features_from_strategy_candidates((candidate,), live_trading_requested=True)
        names = {feature.feature_name for feature in features}
        risk_tags = risk_manager_ontology_tags(features)
        paths = build_semantic_reasoning_paths(features)

        self.assertIn("IntradayMomentumBuy", names)
        self.assertIn("NoOutOfSampleValidation", names)
        self.assertIn("TradeForbidden", risk_tags)
        self.assertEqual(paths[0].strategy_signal, "TradeForbidden")

    def test_pair_and_technical_strategy_signals_are_mapped(self) -> None:
        pair = _candidate(
            strategy_family="pair_relative_value",
            signal_name="gatev_2006_long_only_mean_reversion",
            ontology_tags=[
                "CloseSubstitutePair",
                "PairSpreadDivergence",
                "RelativeUndervaluation",
            ],
            features={"target_net_return": 0.004, "net_expected_return_after_cost": 0.008},
        )
        technical = _candidate(
            strategy_family="technical_rule",
            signal_name="brock_1992_technical_breakout",
            ontology_tags=[
                "MovingAverageBreakout",
                "VolumeConfirmedBreakout",
                "BreakoutWatch",
            ],
            features={"target_net_return": 0.003, "net_expected_return_after_cost": 0.007},
        )

        features = semantic_features_from_strategy_candidates((pair, technical))
        names = {feature.feature_name for feature in features}

        self.assertIn("PairMeanReversionBuy", names)
        self.assertIn("TechnicalBreakoutBuy", names)


def _candidate(
    *,
    strategy_family: str = "short_term_reversal",
    signal_name: str = "jegadeesh_1990_short_term_reversal",
    ontology_tags: list[str],
    features: dict[str, float],
) -> StrategyCandidate:
    return StrategyCandidate(
        ticker="005930",
        strategy_family=strategy_family,
        signal_name=signal_name,
        entry_price=10_000,
        expected_exit_price=10_080,
        expected_holding_minutes=30,
        gross_expected_return=0.008,
        confidence=0.82,
        features=features,
        ontology_tags=ontology_tags,
        created_at=datetime(2026, 1, 2, 9, 30, tzinfo=timezone.utc),
    )


if __name__ == "__main__":
    unittest.main()
