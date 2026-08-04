"""A trigger must clear the cost of its own market, not a constant.

The algorithm layer used one floor, ``min_expected_edge_bps: 8``, for every
strategy in every market, while a round trip costs ~34bp on KRX and ~51bp on US.
The measured consequence was strategies with a positive gross edge and a
negative net one - gap_context booked +12.1bp gross and -15.8bp net across 65
fills - because the trigger layer and the ProfitabilityGate were answering the
same question with different arithmetic.
"""

from __future__ import annotations

import pytest

from app.technical.strategy_algorithms import (
    AlgorithmConfig,
    IntradayMomentumAlgorithm,
    reset_cost_floor_cache,
    round_trip_cost_bps,
)


@pytest.fixture(autouse=True)
def _clear_cost_cache():
    reset_cost_floor_cache()
    yield
    reset_cost_floor_cache()


def test_krx_round_trip_is_dominated_by_the_sell_tax() -> None:
    cost = round_trip_cost_bps("005930")

    # 20bp sell tax + ~2.8bp fees + 10bp two-leg slippage + 1bp safety margin.
    assert cost is not None
    assert 28.0 <= cost <= 40.0


def test_us_round_trip_is_far_higher_than_krx() -> None:
    krx = round_trip_cost_bps("005930")
    us = round_trip_cost_bps("AAPL")

    # 20bp per side of commission is the whole story: US day trading has to clear
    # roughly double the KRX bar for the same trade to be worth taking.
    assert us > krx * 1.4
    assert 45.0 <= us <= 80.0


def test_floor_is_per_market_not_constant() -> None:
    algorithm = IntradayMomentumAlgorithm()

    krx_floor, krx_diagnostics = algorithm.entry_floor_bps("005930")
    us_floor, us_diagnostics = algorithm.entry_floor_bps("AAPL")

    assert krx_diagnostics["floor_basis"] == "round_trip_cost"
    assert krx_diagnostics["venue"] == "KRX"
    assert us_diagnostics["venue"] == "NASD"
    assert us_floor > krx_floor
    # Both must be far above the legacy constant; that is the entire point.
    assert krx_floor > 4 * algorithm.config.shared("min_expected_edge_bps")


def test_edge_that_cannot_cover_its_cost_is_rejected_with_a_naming_reason() -> None:
    algorithm = IntradayMomentumAlgorithm()

    # 12bp used to clear the 8bp floor and then die at the gate.
    decision = algorithm._fire(
        score=0.9,
        confidence=0.8,
        edge_bps=12.0,
        reasons=("TEST",),
        symbol="005930",
    )

    assert decision.triggered is False
    assert "EDGE_BELOW_COST_FLOOR" in decision.reason_codes
    # The numbers travel with the rejection so an operator can see the shortfall
    # rather than re-deriving it.
    assert decision.diagnostics["round_trip_cost_bps"] > 12.0
    assert decision.diagnostics["minimum_edge_bps"] > 12.0


def test_edge_above_the_cost_floor_still_fires() -> None:
    algorithm = IntradayMomentumAlgorithm()
    floor, _ = algorithm.entry_floor_bps("005930")

    decision = algorithm._fire(
        score=0.9,
        confidence=0.8,
        edge_bps=floor + 1.0,
        reasons=("TEST",),
        symbol="005930",
    )

    assert decision.triggered is True
    assert decision.expected_edge_bps == pytest.approx(floor + 1.0)


def test_disabling_the_cost_floor_restores_the_constant() -> None:
    algorithm = IntradayMomentumAlgorithm()
    algorithm.config._values["shared"]["cost_aware_floor_enabled"] = 0.0

    floor, diagnostics = algorithm.entry_floor_bps("005930")

    assert diagnostics["floor_basis"] == "absolute_only"
    assert floor == algorithm.config.shared("min_expected_edge_bps")


def test_unreadable_cost_config_falls_back_rather_than_treating_cost_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.technical.strategy_algorithms as module

    monkeypatch.setattr(module, "round_trip_cost_bps", lambda symbol: None)
    algorithm = IntradayMomentumAlgorithm(AlgorithmConfig())

    floor, diagnostics = algorithm.entry_floor_bps("005930")

    # A missing cost model must never read as "free".
    assert diagnostics["floor_basis"] == "cost_config_unreadable"
    assert floor == algorithm.config.shared("min_expected_edge_bps")
