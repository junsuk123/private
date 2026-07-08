from __future__ import annotations

from dataclasses import dataclass

from app.technical import reason_codes as rc
from app.technical.prediction import (
    PredictionAction,
    PredictionConfig,
    TechnicalPredictionEngine,
)
from app.technical.signals import TechnicalFeatureSet
from tests.test_technical_signals import trend_up_features


@dataclass
class FakeModelPrediction:
    probability_success: float = 0.7
    expected_net_return_bps: float = 20.0
    uncertainty_score: float = 0.1
    approved: bool = True
    is_fallback: bool = False


def _engine():
    return TechnicalPredictionEngine(config=PredictionConfig(min_confidence=0.3))


class TestNoTrade:
    def test_no_trade_when_regime_blocks(self):
        p = _engine().predict(trend_up_features(liquidity_score=0.05))
        assert p.action == PredictionAction.NO_TRADE
        assert not p.tradable

    def test_no_trade_when_no_edge(self):
        # Below VWAP -> composite HOLD -> no trade.
        p = _engine().predict(trend_up_features(vwap_distance_bps=-15, vwap_slope=-3))
        assert p.action == PredictionAction.NO_TRADE

    def test_never_approves_order(self):
        # The prediction object exposes no approval field / order; only advisory.
        p = _engine().predict(trend_up_features())
        assert not hasattr(p, "approved")
        assert not hasattr(p, "final_order")

    def test_low_confidence_is_no_trade(self):
        eng = TechnicalPredictionEngine(config=PredictionConfig(min_confidence=0.99))
        p = eng.predict(trend_up_features())
        assert p.action == PredictionAction.NO_TRADE
        assert rc.TECHNICAL_CONFIDENCE_TOO_LOW in p.reason_codes


class TestTradablePrediction:
    def test_buy_prediction_has_exit_price(self):
        p = _engine().predict(trend_up_features())
        assert p.action == PredictionAction.BUY
        assert p.tradable
        assert p.expected_exit_price is not None and p.expected_exit_price > p.entry_price
        assert p.expected_net_return_bps is not None
        assert p.downside_risk_bps is not None and p.downside_risk_bps >= 0
        assert p.explanation

    def test_cost_reduces_net_return(self):
        f = trend_up_features()
        cheap = _engine().predict(f, all_in_cost_bps=1.0)
        expensive = _engine().predict(f, all_in_cost_bps=8.0)
        assert cheap.expected_net_return_bps > expensive.expected_net_return_bps

    def test_high_cost_forces_no_trade(self):
        p = _engine().predict(trend_up_features(), all_in_cost_bps=100_000.0)
        assert p.action == PredictionAction.NO_TRADE
        assert rc.TECHNICAL_EDGE_NON_POSITIVE in p.reason_codes


class TestModelBlend:
    def test_model_approval_boosts_confidence(self):
        f = trend_up_features()
        base = _engine().predict(f)
        with_model = _engine().predict(f, model_prediction=FakeModelPrediction(probability_success=0.95))
        assert with_model.confidence >= base.confidence

    def test_model_disagreement_penalizes(self):
        f = trend_up_features()
        base = _engine().predict(f)
        disagree = _engine().predict(
            f, model_prediction=FakeModelPrediction(approved=False, probability_success=0.2)
        )
        assert disagree.confidence < base.confidence

    def test_model_net_return_taken_conservatively(self):
        # Model reports a lower net than the technical estimate -> take the lower.
        f = trend_up_features()
        p = _engine().predict(
            f, model_prediction=FakeModelPrediction(expected_net_return_bps=-5.0)
        )
        # A negative model net drives net below zero -> no trade.
        assert p.action == PredictionAction.NO_TRADE

    def test_fallback_model_ignored(self):
        f = trend_up_features()
        base = _engine().predict(f)
        fb = _engine().predict(f, model_prediction=FakeModelPrediction(is_fallback=True, probability_success=0.99))
        assert fb.confidence == base.confidence


class TestDeterminism:
    def test_deterministic(self):
        f = trend_up_features()
        a = _engine().predict(f)
        b = _engine().predict(f)
        assert a.as_dict() == b.as_dict()
