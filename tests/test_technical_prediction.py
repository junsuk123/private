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
        # A cost far above any plausible edge is now classified as UNVIABLE rather
        # than "no edge": the setup is real, the economics are not. Keeping both
        # under one code hid candidates that could never have qualified.
        p = _engine().predict(trend_up_features(), all_in_cost_bps=100_000.0)
        assert p.action == PredictionAction.NO_TRADE
        assert rc.HORIZON_COST_UNVIABLE in p.reason_codes
        assert p.diagnostics["all_in_cost_bps"] == 100_000.0
        assert p.diagnostics["required_net_bps"] is not None


class TestModelFusionAuthority:
    """A model must EARN expected-return authority; it is not granted by default.

    The engine used to take min(rule_net, model_net) unconditionally, which handed
    a veto to any pessimistic estimator — including one that was unapproved,
    fallback, or fitted on a retired feature set. A +30bps rule edge became the
    model's -40bps, and the router then dropped it as NON_POSITIVE_NET_EDGE.
    """

    def test_unreliable_negative_model_does_not_clamp_a_positive_rule_edge(self):
        f = trend_up_features()
        baseline = _engine().predict(f, all_in_cost_bps=1.0)
        assert baseline.tradable

        hostile = FakeModelPrediction(
            expected_net_return_bps=-400.0, approved=False, is_fallback=False
        )
        fused = _engine().predict(f, all_in_cost_bps=1.0, model_prediction=hostile)

        assert fused.diagnostics["model_reliability_weight"] == 0.0
        assert fused.expected_net_return_bps == baseline.expected_net_return_bps
        assert fused.expected_net_return_bps > 0

    def test_fallback_model_gets_no_expected_return_authority(self):
        f = trend_up_features()
        baseline = _engine().predict(f, all_in_cost_bps=1.0)
        fallback = FakeModelPrediction(
            expected_net_return_bps=-400.0, approved=True, is_fallback=True
        )
        fused = _engine().predict(f, all_in_cost_bps=1.0, model_prediction=fallback)
        assert fused.diagnostics["model_reliability_weight"] == 0.0
        assert fused.expected_net_return_bps == baseline.expected_net_return_bps

    def test_reliable_negative_model_lowers_net_conservatively(self):
        f = trend_up_features()
        baseline = _engine().predict(f, all_in_cost_bps=1.0)
        reliable_bear = FakeModelPrediction(
            expected_net_return_bps=-30.0, approved=True, is_fallback=False,
            uncertainty_score=0.0,
        )
        fused = _engine().predict(f, all_in_cost_bps=1.0, model_prediction=reliable_bear)

        assert fused.diagnostics["model_reliability_weight"] > 0.0
        assert fused.diagnostics["uncertainty_penalty_bps"] > 0.0
        # Reflected, but bounded: not a wholesale replacement by the model's number.
        assert fused.expected_net_return_bps < baseline.expected_net_return_bps
        assert fused.expected_net_return_bps > -30.0

    def test_diagnostics_expose_the_full_decomposition(self):
        f = trend_up_features()
        p = _engine().predict(
            f, all_in_cost_bps=2.0, model_prediction=FakeModelPrediction()
        )
        for key in (
            "rule_gross_bps",
            "all_in_cost_bps",
            "rule_net_bps",
            "model_net_bps",
            "model_reliability_weight",
            "uncertainty_penalty_bps",
            "fused_net_bps",
            "required_net_bps",
            "cost_coverage_ratio",
        ):
            assert key in p.diagnostics, key
        d = p.diagnostics
        assert d["rule_net_bps"] == d["rule_gross_bps"] - d["all_in_cost_bps"]

    def test_fused_net_below_required_still_blocks(self):
        # Weakening the model's veto must not weaken the cost bar.
        f = trend_up_features()
        p = _engine().predict(f, all_in_cost_bps=40.0)
        assert p.action == PredictionAction.NO_TRADE


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
