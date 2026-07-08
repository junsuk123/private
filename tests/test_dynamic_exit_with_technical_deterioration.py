from __future__ import annotations

from types import SimpleNamespace

from app.technical import reason_codes as rc
from app.trading.shared_decision_engine import SharedLiveDecisionEngine
from tests.test_shared_decision_engine_technical_prediction import _trend_up_frame


def _frame_with(**over):
    frame = _trend_up_frame(10_000.0)
    base = frame.as_feature_dict()
    base.update(over)
    return SimpleNamespace(
        symbol="005930",
        mark_price=10_000.0,
        feature_schema_hash="unit",
        provenance=SimpleNamespace(source_record_ids=("f",)),
        as_feature_dict=lambda: dict(base),
    )


def _engine():
    return SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None))


class TestDeteriorationHelper:
    def test_healthy_frame_no_deterioration(self):
        codes, penalty = _engine()._technical_exit_deterioration(_frame_with(), "005930")
        assert codes == ()
        assert penalty == 0.0

    def test_vwap_breakdown_and_momentum_loss(self):
        frame = _frame_with(distance_from_vwap=-0.001, macd_histogram=-0.3)
        codes, penalty = _engine()._technical_exit_deterioration(frame, "005930")
        assert rc.VWAP_BREAKDOWN in codes
        assert rc.MOMENTUM_WEAKENED in codes
        assert rc.TECHNICAL_EXIT_DETERIORATION in codes
        assert penalty > 0.0

    def test_penalty_capped(self):
        frame = _frame_with(distance_from_vwap=-0.001, macd_histogram=-0.3, liquidity_score=0.1)
        _, penalty = _engine()._technical_exit_deterioration(frame, "005930")
        assert penalty <= 0.5

    def test_none_frame(self):
        codes, penalty = _engine()._technical_exit_deterioration(None, "005930")
        assert codes == () and penalty == 0.0

    def test_disabled_engine(self):
        engine = _engine()
        engine._technical_enabled = False
        codes, penalty = engine._technical_exit_deterioration(
            _frame_with(distance_from_vwap=-0.001, macd_histogram=-0.3), "005930"
        )
        assert codes == () and penalty == 0.0

    def test_deterioration_only_strengthens_never_forces_loss(self):
        # The penalty is bounded well below the magnitude needed to override the
        # loss-exit gate; it only nudges the (profit-gated) invalid-signal branch.
        frame = _frame_with(distance_from_vwap=-0.001, macd_histogram=-0.3)
        _, penalty = _engine()._technical_exit_deterioration(frame, "005930")
        assert 0.0 < penalty <= 0.5
