"""StrategyProposal: a thesis statement, structurally incapable of being an order."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

import pytest

from app.context import MarketContextBuilder, SymbolContextInputs
from app.strategy.proposal import StrategyProposal, new_proposal_id
from app.strategy.proposal_engine import (
    PROPOSAL_ALGORITHM_MISSING,
    PROPOSAL_NO_FEATURES,
    StrategyProposalEngine,
)
from app.technical.signals import TechnicalFeatureSet

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)

#: Fields that would make a proposal an order. None of them may exist.
_ORDER_FIELDS = {
    "quantity",
    "shares",
    "side",
    "order_type",
    "price_policy",
    "venue",
    "exchange",
    "broker",
    "account",
    "notional",
    "position_size",
}


def _proposal(**overrides) -> StrategyProposal:
    values = {
        "proposal_id": new_proposal_id(),
        "context_id": "ctx-1",
        "strategy_id": "intraday_momentum",
        "symbol": "005930",
        "eligible": True,
        "entry_ready": True,
        "reference_entry_price": 70_000.0,
        "target_price": 71_120.0,
        "stop_price": 69_580.0,
        "expected_horizon_seconds": 600,
        "expected_gross_edge_bps": 120.0,
        "confidence": 0.6,
    }
    values.update(overrides)
    return StrategyProposal(**values)


def test_proposal_has_no_order_fields() -> None:
    names = {member.name for member in fields(StrategyProposal)}
    assert not (names & _ORDER_FIELDS), names & _ORDER_FIELDS


def test_target_and_stop_moves_are_direction_aware() -> None:
    long_side = _proposal()
    assert long_side.target_move_bps == pytest.approx(160.0, abs=0.1)
    assert long_side.stop_move_bps == pytest.approx(60.0, abs=0.1)

    short_side = _proposal(
        direction="SHORT", target_price=68_880.0, stop_price=70_420.0
    )
    # A short's target sits BELOW its entry; reporting that as a loss would be the sign bug.
    assert short_side.target_move_bps == pytest.approx(160.0, abs=0.1)
    assert short_side.stop_move_bps == pytest.approx(60.0, abs=0.1)


def test_unpriceable_proposal_reports_none_not_zero() -> None:
    proposal = _proposal(reference_entry_price=None, target_price=None, stop_price=None)
    assert proposal.target_move_bps is None
    assert proposal.stop_move_bps is None
    assert proposal.reference_entry_price is None


def test_selectable_requires_both_eligibility_and_readiness() -> None:
    assert _proposal().selectable
    assert not _proposal(eligible=False).selectable
    assert not _proposal(entry_ready=False).selectable


def test_non_positive_prices_are_rejected_at_construction() -> None:
    assert _proposal(reference_entry_price=0.0).reference_entry_price is None
    assert _proposal(target_price=-5.0).target_price is None


def test_confidence_is_clamped_to_unit_range() -> None:
    assert _proposal(confidence=5.0).confidence == 1.0
    assert _proposal(confidence=-1.0).confidence == 0.0


def test_engine_carries_the_context_id_onto_every_proposal() -> None:
    context = MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(
                symbol="005930",
                price=70_000.0,
                ema_fast=70_100.0,
                ema_slow=69_900.0,
                macd_histogram=4.0,
                aggressor_imbalance_5s=0.3,
                return_5s=0.0008,
                realized_volatility=0.004,
                realized_volatility_10s=0.003,
                tick_count_5s=9.0,
                second_data_ready=1.0,
                spread_bps=9.0,
                orderbook_imbalance=0.2,
                liquidity_score=0.6,
            ),
            context_id="ctx-engine",
            tick_freshness_sec=0.4,
            orderbook_freshness_sec=0.6,
            history_bar_count=40,
        ),
        captured_at=AT,
    )
    result = StrategyProposalEngine().evaluate(
        context, eligible_strategy_ids=("intraday_momentum",)
    )
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.context_id == "ctx-engine"
    assert proposal.proposal_id
    assert proposal.symbol == "005930"
    assert proposal.eligible is True
    assert proposal.proposed_at == context.captured_at


def test_engine_only_evaluates_the_masked_subset() -> None:
    """The cheap mask runs first; the algorithms are the expensive part."""
    context = MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(symbol="005930", price=70_000.0, second_data_ready=1.0),
            context_id="ctx-mask",
        ),
        captured_at=AT,
    )
    result = StrategyProposalEngine().evaluate(
        context, eligible_strategy_ids=("intraday_momentum", "vwap_mean_reversion")
    )
    assert {item.strategy_id for item in result.proposals} <= {
        "intraday_momentum",
        "vwap_mean_reversion",
    }
    assert len(result.proposals) <= 2


def test_engine_reports_an_unknown_strategy_rather_than_dropping_it() -> None:
    context = MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(symbol="005930", price=70_000.0),
            context_id="ctx-unknown",
        ),
        captured_at=AT,
    )
    result = StrategyProposalEngine().evaluate(
        context, eligible_strategy_ids=("no_such_strategy",)
    )
    assert result.proposals == ()
    assert result.skipped, "a skip must be reported, not silent"


def test_engine_without_features_reports_the_reason() -> None:
    context = MarketContextBuilder().build(
        SymbolContextInputs(symbol="005930", features=None, context_id="ctx-nofeat"),
        captured_at=AT,
    )
    result = StrategyProposalEngine().evaluate(
        context, eligible_strategy_ids=("intraday_momentum",)
    )
    assert result.skipped.get("intraday_momentum") == PROPOSAL_NO_FEATURES


def test_broken_algorithm_yields_a_not_ready_proposal_not_an_exception() -> None:
    class _Exploding:
        strategy_id = "intraday_momentum"
        horizon_seconds = 180

        def entry(self, *_args, **_kwargs):
            raise RuntimeError("algorithm blew up")

    context = MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(symbol="005930", price=70_000.0),
            context_id="ctx-explode",
        ),
        captured_at=AT,
    )
    engine = StrategyProposalEngine(algorithm_registry={"intraday_momentum": _Exploding()})
    result = engine.evaluate(context, eligible_strategy_ids=("intraday_momentum",))
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.entry_ready is False
    assert any(
        code.startswith("STRATEGY_ENTRY_EVALUATION_ERROR")
        for code in proposal.strategy_reason_codes
    )


def test_missing_algorithm_is_reported() -> None:
    context = MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(symbol="005930", price=70_000.0),
            context_id="ctx-missing",
        ),
        captured_at=AT,
    )
    engine = StrategyProposalEngine(algorithm_registry={})
    result = engine.evaluate(context, eligible_strategy_ids=("intraday_momentum",))
    assert result.skipped.get("intraday_momentum") == PROPOSAL_ALGORITHM_MISSING
