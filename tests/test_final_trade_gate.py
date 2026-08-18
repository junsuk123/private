"""FinalTradeGate: fail-closed hard gates, compounding soft gates, immovable limits."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from app.risk.final_trade_gate import (
    HARD_GATES,
    SOFT_GATES,
    FinalTradeGate,
    GateInputs,
    load_gate_config,
)

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)
CONFIG = load_gate_config()


def _clean(**overrides) -> GateInputs:
    """Inputs on which every hard gate is satisfied and no soft gate triggers."""
    base = dict(
        ticker="005930",
        side="BUY",
        evaluated_at=NOW,
        stale_data_reasons=(),
        websocket_connected=True,
        price_feed_divergence_bps=2.0,
        session_id="KRX_REGULAR",
        session_allows_new_entry=True,
        trading_halted=False,
        account_reconciled=True,
        unknown_order_ids=(),
        duplicate_order_risk=False,
        model_health_state="HEALTHY",
        risk_engine_ok=True,
        realized_volatility=0.001,
        liquidity_score=0.85,
        global_agreement=0.4,
        sector_relative_strength=0.01,
        model_confidence=0.8,
        session_phase="MIDDAY",
        spread_bps=6.0,
        dominant_regime="TREND_UP",
        account_equity=100_000_000.0,
        current_position_value=0.0,
        current_sector_exposure=0.0,
        current_market_exposure=0.0,
        session_pnl_ratio=0.001,
        drawdown_ratio=0.01,
        requested_position_fraction=0.05,
    )
    base.update(overrides)
    return GateInputs(**base)


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def test_clean_inputs_are_approved_at_full_size() -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean())
    assert decision.approved
    assert not decision.hard_failures
    assert not decision.soft_failures
    assert not decision.limit_failures
    assert decision.approved_position_fraction > 0.0


def test_decision_serialises_every_term() -> None:
    payload = FinalTradeGate(CONFIG).evaluate(_clean()).as_dict()
    assert set(payload) >= {
        "approved",
        "hard_failures",
        "soft_failures",
        "limit_failures",
        "position_multiplier",
        "approved_position_fraction",
        "factors",
    }
    assert set(payload["factors"]) >= {
        "model_confidence",
        "regime_factor",
        "liquidity_factor",
        "risk_factor",
    }


# --------------------------------------------------------------------------- #
# Hard gates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"stale_data_reasons": ("STALE_DATA:kis_realtime/trade",)}, "STALE_DATA"),
        ({"websocket_connected": False}, "WS_DISCONNECTED"),
        ({"price_feed_divergence_bps": 400.0}, "PRICE_FEED_CONFLICT"),
        ({"session_id": "KR_CLOSED"}, "UNKNOWN_SESSION"),
        ({"session_allows_new_entry": False}, "UNKNOWN_SESSION"),
        ({"trading_halted": True}, "TRADING_HALT"),
        ({"account_reconciled": False}, "ACCOUNT_RECONCILIATION_FAIL"),
        ({"unknown_order_ids": ("ORD-1",)}, "UNKNOWN_ORDER_STATE"),
        ({"duplicate_order_risk": True}, "DUPLICATE_ORDER_RISK"),
        ({"model_health_state": "OFFLINE"}, "MODEL_INFERENCE_FAIL"),
        ({"risk_engine_ok": False}, "RISK_ENGINE_FAIL"),
    ],
)
def test_every_hard_gate_blocks(overrides: dict, expected: str) -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean(**overrides))
    assert not decision.approved
    assert expected in decision.hard_failures
    assert decision.approved_position_fraction == 0.0


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("websocket_connected", "WS_DISCONNECTED"),
        ("session_id", "UNKNOWN_SESSION"),
        ("session_allows_new_entry", "UNKNOWN_SESSION"),
        ("trading_halted", "TRADING_HALT"),
        ("account_reconciled", "ACCOUNT_RECONCILIATION_FAIL"),
        ("duplicate_order_risk", "DUPLICATE_ORDER_RISK"),
        ("model_health_state", "MODEL_INFERENCE_FAIL"),
        ("risk_engine_ok", "RISK_ENGINE_FAIL"),
    ],
)
def test_unsupplied_hard_gate_input_blocks_rather_than_defaults_open(
    field: str, expected: str
) -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean(**{field: None}))
    assert not decision.approved
    assert expected in decision.hard_failures


def test_price_cross_check_only_required_when_asked() -> None:
    gate = FinalTradeGate(CONFIG)
    optional = gate.evaluate(_clean(price_feed_divergence_bps=None))
    required = gate.evaluate(
        _clean(price_feed_divergence_bps=None, require_price_cross_check=True)
    )
    assert optional.approved
    assert "PRICE_FEED_CONFLICT" in required.hard_failures


def test_the_hard_gate_set_is_exactly_the_declared_one() -> None:
    declared = set(HARD_GATES)
    seen: set[str] = set()
    for overrides in (
        {"stale_data_reasons": ("x",)},
        {"websocket_connected": False},
        {"price_feed_divergence_bps": 900.0},
        {"session_id": None},
        {"trading_halted": True},
        {"account_reconciled": False},
        {"unknown_order_ids": ("a",)},
        {"duplicate_order_risk": True},
        {"model_health_state": "OFFLINE"},
        {"risk_engine_ok": False},
    ):
        seen.update(FinalTradeGate(CONFIG).evaluate(_clean(**overrides)).hard_failures)
    assert seen == declared


def test_gate_never_raises_and_fails_closed_on_an_internal_error() -> None:
    class Exploding(FinalTradeGate):
        def _soft_failures(self, inputs):  # type: ignore[override]
            raise RuntimeError("boom")

    decision = Exploding(CONFIG).evaluate(_clean())
    assert not decision.approved
    assert decision.hard_failures == ("RISK_ENGINE_FAIL",)
    assert "boom" in decision.reasons[0]


# --------------------------------------------------------------------------- #
# Soft gates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"realized_volatility": 0.02}, "HIGH_VOLATILITY"),
        ({"liquidity_score": 0.10}, "LOW_LIQUIDITY"),
        ({"global_agreement": -0.9}, "GLOBAL_CONFLICT"),
        ({"sector_relative_strength": -0.02}, "SECTOR_CONFLICT"),
        ({"model_confidence": 0.30}, "LOW_MODEL_CONFIDENCE"),
        (
            {"session_phase": "OPENING", "opening_volatility_multiple": 4.0},
            "OPENING_EXTREME_VOL",
        ),
        ({"spread_bps": 120.0}, "ABNORMAL_SPREAD"),
    ],
)
def test_soft_gates_reduce_size_without_necessarily_blocking(
    overrides: dict, expected: str
) -> None:
    gate = FinalTradeGate(CONFIG)
    baseline = gate.evaluate(_clean())
    degraded = gate.evaluate(_clean(**overrides))
    assert expected in degraded.soft_failures
    assert degraded.position_multiplier < baseline.position_multiplier
    assert expected in set(SOFT_GATES)


def test_soft_gates_compound_rather_than_taking_the_worst() -> None:
    gate = FinalTradeGate(CONFIG)
    one = gate.evaluate(_clean(spread_bps=120.0))
    two = gate.evaluate(_clean(spread_bps=120.0, realized_volatility=0.02))
    assert len(two.soft_failures) == 2
    assert two.position_multiplier < one.position_multiplier
    assert two.factors["risk_factor"] == pytest.approx(
        CONFIG.soft_gates["ABNORMAL_SPREAD"].multiplier
        * CONFIG.soft_gates["HIGH_VOLATILITY"].multiplier
    )


def test_enough_compounding_blocks_the_trade() -> None:
    decision = FinalTradeGate(CONFIG).evaluate(
        _clean(
            spread_bps=120.0,
            realized_volatility=0.02,
            liquidity_score=0.10,
            global_agreement=-0.9,
            model_confidence=0.25,
        )
    )
    assert not decision.approved
    assert "POSITION_BELOW_MINIMUM" in decision.limit_failures
    assert decision.position_multiplier < CONFIG.sizing.block_below


def test_sector_conflict_is_side_aware() -> None:
    gate = FinalTradeGate(CONFIG)
    weak_sector = {"sector_relative_strength": -0.02}
    buy = gate.evaluate(_clean(side="BUY", **weak_sector))
    sell = gate.evaluate(_clean(side="SELL", **weak_sector))
    assert "SECTOR_CONFLICT" in buy.soft_failures
    assert "SECTOR_CONFLICT" not in sell.soft_failures


def test_opening_volatility_gate_only_applies_in_the_opening_phases() -> None:
    gate = FinalTradeGate(CONFIG)
    midday = gate.evaluate(_clean(session_phase="MIDDAY", opening_volatility_multiple=9.0))
    opening = gate.evaluate(
        _clean(session_phase="OPEN_TRANSITION", opening_volatility_multiple=9.0)
    )
    assert "OPENING_EXTREME_VOL" not in midday.soft_failures
    assert "OPENING_EXTREME_VOL" in opening.soft_failures


# --------------------------------------------------------------------------- #
# Sizing and limits
# --------------------------------------------------------------------------- #
def test_position_formula_is_the_declared_product() -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean(spread_bps=120.0))
    expected = 1.0
    for value in decision.factors.values():
        expected *= value
    assert decision.position_multiplier == pytest.approx(expected, rel=1e-6)


def test_model_confidence_cannot_size_above_the_ceiling() -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean(model_confidence=99.0))
    assert decision.factors["model_confidence"] == pytest.approx(
        CONFIG.sizing.model_confidence_ceiling
    )


def test_absent_model_confidence_sizes_at_the_floor_not_at_full() -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean(model_confidence=None))
    assert decision.factors["model_confidence"] == pytest.approx(
        CONFIG.sizing.model_confidence_floor
    )


def test_degraded_model_halves_size_and_offline_blocks() -> None:
    gate = FinalTradeGate(CONFIG)
    healthy = gate.evaluate(_clean(model_health_state="HEALTHY"))
    degraded = gate.evaluate(_clean(model_health_state="DEGRADED"))
    offline = gate.evaluate(_clean(model_health_state="OFFLINE"))
    assert degraded.approved
    assert degraded.position_multiplier == pytest.approx(
        healthy.position_multiplier * 0.5
    )
    assert not offline.approved


def test_model_cannot_exceed_the_per_stock_limit() -> None:
    decision = FinalTradeGate(CONFIG).evaluate(
        _clean(requested_position_fraction=0.90, model_confidence=1.0)
    )
    assert decision.approved
    assert decision.approved_position_fraction <= CONFIG.limits.max_position_per_stock


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"current_position_value": 50_000_000.0}, "MAX_POSITION_PER_STOCK"),
        ({"current_sector_exposure": 90_000_000.0}, "MAX_SECTOR_EXPOSURE"),
        ({"current_market_exposure": 95_000_000.0}, "MAX_MARKET_EXPOSURE"),
        ({"session_pnl_ratio": -0.05}, "MAX_DAILY_LOSS"),
        ({"drawdown_ratio": -0.25}, "MAX_DRAWDOWN"),
    ],
)
def test_each_exposure_limit_blocks(overrides: dict, expected: str) -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean(**overrides))
    assert not decision.approved
    assert expected in decision.limit_failures


def test_missing_equity_cannot_authorise_exposure() -> None:
    decision = FinalTradeGate(CONFIG).evaluate(_clean(account_equity=None))
    assert not decision.approved
    assert decision.approved_position_fraction == 0.0


def test_headroom_shrinks_the_approved_fraction() -> None:
    gate = FinalTradeGate(CONFIG)
    empty = gate.evaluate(_clean(requested_position_fraction=0.10))
    partial = gate.evaluate(
        _clean(requested_position_fraction=0.10, current_position_value=9_000_000.0)
    )
    assert partial.approved
    assert partial.approved_position_fraction < empty.approved_position_fraction
    assert partial.approved_position_fraction == pytest.approx(
        CONFIG.limits.max_position_per_stock - 0.09
    )


# --------------------------------------------------------------------------- #
# Exits
# --------------------------------------------------------------------------- #
def test_stale_data_blocks_entry_but_never_an_exit() -> None:
    gate = FinalTradeGate(CONFIG)
    stale = _clean(
        side="SELL",
        stale_data_reasons=("STALE_DATA:kis_realtime/trade",),
        websocket_connected=False,
        model_health_state="OFFLINE",
        account_reconciled=False,
    )
    assert not gate.evaluate(stale).approved
    exit_decision = gate.evaluate_exit(stale)
    assert exit_decision.approved
    assert exit_decision.hard_failures == ()


def test_exit_is_still_blocked_when_it_cannot_be_routed() -> None:
    gate = FinalTradeGate(CONFIG)
    for overrides, expected in (
        ({"session_id": "KR_CLOSED"}, "UNKNOWN_SESSION"),
        ({"unknown_order_ids": ("ORD-9",)}, "UNKNOWN_ORDER_STATE"),
        ({"duplicate_order_risk": True}, "DUPLICATE_ORDER_RISK"),
        ({"trading_halted": True}, "TRADING_HALT"),
    ):
        decision = gate.evaluate_exit(_clean(side="SELL", **overrides))
        assert not decision.approved
        assert expected in decision.hard_failures


def test_exposure_limits_never_block_an_exit() -> None:
    decision = FinalTradeGate(CONFIG).evaluate_exit(
        _clean(
            side="SELL",
            current_position_value=99_000_000.0,
            session_pnl_ratio=-0.5,
            drawdown_ratio=-0.5,
        )
    )
    assert decision.approved
    assert decision.limit_failures == ()


# --------------------------------------------------------------------------- #
# Policy authority
# --------------------------------------------------------------------------- #
def test_limits_are_read_from_policy_not_from_the_caller() -> None:
    tighter = replace(CONFIG, limits=replace(CONFIG.limits, max_position_per_stock=0.01))
    decision = FinalTradeGate(tighter).evaluate(
        _clean(requested_position_fraction=0.50, model_confidence=1.0)
    )
    assert decision.approved_position_fraction <= 0.01


def test_shipped_policy_parses_with_every_declared_soft_gate() -> None:
    assert CONFIG.source_path is not None
    assert set(CONFIG.soft_gates) == set(SOFT_GATES)
    assert CONFIG.limits.max_position_per_stock > 0.0
