from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.routing import StrategyRouter
from app.trading.contracts import OntologyDecision, Position, StrategyUtilityEvidence


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _ontology(*allowed: str) -> OntologyDecision:
    return OntologyDecision(
        snapshot_id="ontology-1",
        as_of=NOW,
        symbol="005930",
        allowed_strategy_ids=allowed,
        blocked_strategy_reasons={},
        compatibility_scores={name: 1.0 for name in allowed},
        explanation_paths={name: ("unit",) for name in allowed},
        valid_until=NOW + timedelta(seconds=5),
    )


def _evidence(
    strategy: str,
    utility: float,
    net: float = 10,
    compatibility: float = 1.0,
) -> StrategyUtilityEvidence:
    return StrategyUtilityEvidence(
        evidence_id=f"evidence-{strategy}",
        as_of=NOW,
        symbol="005930",
        strategy_id=strategy,
        ontology_allowed=True,
        hard_block_reasons=(),
        compatibility_score=compatibility,
        probability_success=0.6,
        expected_gross_return_bps=net + 5,
        expected_cost_bps=5,
        expected_net_return_bps=net,
        expected_adverse_excursion_bps=5,
        expected_favorable_excursion_bps=15,
        fill_probability=0.8,
        expected_holding_seconds=60,
        aleatoric_uncertainty=0.1,
        epistemic_uncertainty_or_proxy=0.1,
        utility=utility,
        model_version="unit",
        feature_snapshot_id="feature-1",
        ontology_snapshot_id="ontology-1",
        explanation_paths=("unit",),
    )


def test_router_selects_highest_utility_only_from_ontology_allowed() -> None:
    decision = StrategyRouter().route(
        as_of=NOW,
        symbol="005930",
        ontology=_ontology("momentum"),
        evidence=(_evidence("reversal", 100), _evidence("momentum", 2)),
    )
    assert decision.selected is not None
    assert decision.selected.strategy_id == "momentum"


def test_router_ranks_gnn_utility_weighted_by_ontology_compatibility() -> None:
    decision = StrategyRouter().route(
        as_of=NOW,
        symbol="005930",
        ontology=_ontology("momentum", "breakout"),
        evidence=(
            _evidence("momentum", 10, compatibility=0.1),
            _evidence("breakout", 4, compatibility=0.9),
        ),
    )

    assert decision.selected is not None
    assert decision.selected.strategy_id == "breakout"
    assert decision.weighted_utility == 3.6
    assert decision.reason_codes == ("MAX_ONTOLOGY_WEIGHTED_NET_UTILITY",)


def test_no_trade_is_first_class_for_non_positive_net_edge() -> None:
    decision = StrategyRouter().route(
        as_of=NOW,
        symbol="005930",
        ontology=_ontology("momentum"),
        evidence=(_evidence("momentum", 2, net=-1),),
    )
    assert decision.is_no_trade
    assert decision.reason_codes == ("NON_POSITIVE_NET_EDGE:momentum",)


def test_router_rejects_positive_edge_below_execution_floor() -> None:
    decision = StrategyRouter(minimum_net_edge_bps=5.0).route(
        as_of=NOW,
        symbol="005930",
        ontology=_ontology("momentum"),
        evidence=(_evidence("momentum", 2, net=1.0),),
    )

    assert decision.is_no_trade
    assert decision.reason_codes == (
        "NET_EDGE_BELOW_EXECUTION_FLOOR:momentum",
    )


def test_router_cannot_transfer_owned_position() -> None:
    position = Position(
        position_id="position-1",
        symbol="005930",
        quantity=1,
        average_price=80000,
        origin_strategy_id="reversal",
        strategy_instance_id="reversal-1",
        opened_at=NOW,
    )
    decision = StrategyRouter().route(
        as_of=NOW,
        symbol="005930",
        ontology=_ontology("momentum"),
        evidence=(_evidence("momentum", 2),),
        open_positions=(position,),
    )
    assert decision.is_no_trade
    assert decision.reason_codes == ("POSITION_ALREADY_OWNED",)
