from __future__ import annotations

from datetime import datetime, timezone

from app.strategy.catalog import STRATEGY_IDS
from app.strategy.exit_geometry import all_geometries, exit_bps, exit_geometry, max_holding_seconds
from app.technical.signals import TechnicalFeatureSet
from app.technical.strategy_algorithms import (
    ALGORITHM_IDS,
    AlgorithmConfig,
    ElectionContext,
    build_algorithm_registry,
    get_algorithm,
    macro_strategy_permitted,
    strategy_live_authorized,
    strategy_shadow_authorized,
)


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)


def _features(**overrides) -> TechnicalFeatureSet:
    base = dict(
        symbol="005930",
        price=70_000.0,
        second_data_ready=1.0,
        tick_count_5s=8.0,
        return_1s=0.0002,
        return_5s=0.0008,
        return_10s=0.0015,
        aggressor_imbalance_5s=0.3,
        realized_volatility_10s=0.002,
        realized_volatility=0.003,
        spread_change_5s=-0.0001,
        orderbook_imbalance=0.4,
        orderbook_imbalance_change_5s=0.05,
        spread_bps=8.0,
        depth_ratio=1.5,
        relative_volume=1.8,
        short_return=0.002,
        vwap=70_500.0,
        vwap_distance_bps=-71.0,
    )
    base.update(overrides)
    return TechnicalFeatureSet(**base)


def _context(strategy_id, **overrides) -> ElectionContext:
    base = dict(strategy_id=strategy_id, elected_at=NOW)
    base.update(overrides)
    return ElectionContext(**base)


# --------------------------------------------------------------------------- #
# Derived microstructure                                                       #
# --------------------------------------------------------------------------- #
def test_microprice_edge_follows_the_book_tilt():
    # microprice - mid == (spread / 2) * depth imbalance
    assert _features(spread_bps=10.0, orderbook_imbalance=0.5).microprice_edge_bps == 2.5
    assert _features(spread_bps=10.0, orderbook_imbalance=-0.5).microprice_edge_bps == -2.5
    # Unknown inputs stay unknown rather than becoming a neutral 0.0.
    assert _features(spread_bps=None).microprice_edge_bps is None
    assert _features(orderbook_imbalance=None).microprice_edge_bps is None


def test_vwap_zscore_normalises_displacement_by_volatility():
    calm = _features(vwap_distance_bps=-60.0, realized_volatility=0.001)
    violent = _features(vwap_distance_bps=-60.0, realized_volatility=0.01)
    assert calm.vwap_zscore == -6.0
    assert violent.vwap_zscore == -0.6
    # The same 60bps displacement is a real dislocation in a calm tape and noise
    # in a violent one — which is exactly what the fixed-25bps rule could not say.
    assert abs(calm.vwap_zscore) > abs(violent.vwap_zscore)
    assert _features(realized_volatility=None, realized_volatility_10s=None).vwap_zscore is None


# --------------------------------------------------------------------------- #
# Catalog / deployment                                                         #
# --------------------------------------------------------------------------- #
def test_new_strategies_are_registered_and_shadow_only():
    added = (
        "residual_relative_strength",
        "adaptive_anchored_vwap_reversion",
        "ofi_microprice_exhaustion_reversal",
    )
    assert ALGORITHM_IDS == STRATEGY_IDS
    registry = build_algorithm_registry()
    for strategy_id in added:
        assert strategy_id in registry
        # Shadow first: they must accumulate per-regime samples before the
        # conservative bandit can give them a positive lower bound.
        assert strategy_live_authorized(strategy_id) is False
        assert strategy_shadow_authorized(strategy_id) is True


def test_residual_strength_is_a_relative_strength_family_not_momentum():
    # This matters: a falling index blocks momentum but still allows
    # relative_strength, and the residual thesis belongs in the latter.
    assert (
        macro_strategy_permitted(
            "residual_relative_strength",
            ("sell", "reduce_risk", "hold", "relative_strength"),
            ("momentum", "breakout"),
        )
        is True
    )
    assert (
        macro_strategy_permitted(
            "adaptive_anchored_vwap_reversion",
            ("mean_reversion", "vwap_reversion"),
            ("momentum",),
        )
        is True
    )


# --------------------------------------------------------------------------- #
# Residual relative strength                                                   #
# --------------------------------------------------------------------------- #
def _residual_context(**overrides) -> ElectionContext:
    base = dict(
        residual_return_short_bps=20.0,
        residual_return_long_bps=15.0,
        sector_rank=1,
        sector_candidate_count=5,
        foreign_flow_zscore=1.2,
        institution_flow_zscore=0.8,
        market_beta=0.9,
        change_point_probability=0.1,
    )
    base.update(overrides)
    return _context("residual_relative_strength", **base)


def test_residual_strength_fires_on_confirmed_idiosyncratic_leadership():
    algorithm = get_algorithm("residual_relative_strength")
    decision = algorithm.entry(_features(), _residual_context())
    assert decision.triggered, decision.reason_codes
    assert "RESIDUAL_STRENGTH_CONFIRMED" in decision.reason_codes


def test_residual_strength_fails_closed_without_residuals():
    """Falling back to raw return here would rebuild the market-beta defect."""
    algorithm = get_algorithm("residual_relative_strength")
    decision = algorithm.entry(
        _features(), _residual_context(residual_return_short_bps=None)
    )
    assert not decision.triggered
    assert "RESIDUAL_STRENGTH_CONTEXT_ABSENT" in decision.reason_codes


def test_residual_strength_requires_a_real_sector_rank():
    algorithm = get_algorithm("residual_relative_strength")
    decision = algorithm.entry(
        _features(), _residual_context(sector_rank=None, sector_candidate_count=None)
    )
    assert not decision.triggered
    assert "RESIDUAL_SECTOR_RANK_ABSENT" in decision.reason_codes


def test_residual_strength_requires_investor_flow_confirmation():
    algorithm = get_algorithm("residual_relative_strength")
    absent = algorithm.entry(
        _features(),
        _residual_context(foreign_flow_zscore=None, institution_flow_zscore=None),
    )
    assert not absent.triggered
    assert "RESIDUAL_INVESTOR_FLOW_ABSENT" in absent.reason_codes

    negative = algorithm.entry(
        _features(),
        _residual_context(foreign_flow_zscore=-2.0, institution_flow_zscore=-1.0),
    )
    assert not negative.triggered
    assert "RESIDUAL_INVESTOR_FLOW_NEGATIVE" in negative.reason_codes


def test_residual_strength_stands_down_on_a_structural_break():
    algorithm = get_algorithm("residual_relative_strength")
    decision = algorithm.entry(
        _features(), _residual_context(change_point_probability=0.95)
    )
    assert not decision.triggered
    assert "RESIDUAL_STRENGTH_REGIME_UNSTABLE" in decision.reason_codes


def test_residual_strength_rejects_an_offered_book():
    algorithm = get_algorithm("residual_relative_strength")
    decision = algorithm.entry(
        _features(orderbook_imbalance=-0.6), _residual_context()
    )
    assert not decision.triggered
    assert "RESIDUAL_MICROPRICE_NOT_SUPPORTIVE" in decision.reason_codes


# --------------------------------------------------------------------------- #
# Adaptive anchored VWAP reversion                                             #
# --------------------------------------------------------------------------- #
def _vwap_features(**overrides) -> TechnicalFeatureSet:
    base = dict(
        price=69_300.0,
        vwap=70_000.0,
        realized_volatility=0.004,      # 40bps per observation
        realized_volatility_10s=0.004,
        return_1s=0.0001,
        orderbook_imbalance_change_5s=0.06,
        spread_change_5s=-0.0002,
        orderbook_imbalance=0.3,
        spread_bps=9.0,
    )
    base.update(overrides)
    return _features(**base)


def test_adaptive_vwap_fires_on_a_normalised_displacement_with_liquidity_returning():
    algorithm = get_algorithm("adaptive_anchored_vwap_reversion")
    decision = algorithm.entry(
        _vwap_features(), _context("adaptive_anchored_vwap_reversion")
    )
    assert decision.triggered, decision.reason_codes
    assert "LIQUIDITY_RECOVERY_CONFIRMED" in decision.reason_codes
    assert decision.diagnostics["anchor_basis"] == "session_vwap"


def test_adaptive_vwap_ignores_a_displacement_that_is_only_noise():
    """A 25bps band is noise when the tape moves 4% a day; the z-score says so."""
    algorithm = get_algorithm("adaptive_anchored_vwap_reversion")
    decision = algorithm.entry(
        _vwap_features(price=69_800.0, realized_volatility=0.02),
        _context("adaptive_anchored_vwap_reversion"),
    )
    assert not decision.triggered
    assert "ADAPTIVE_VWAP_DISPLACEMENT_TOO_SMALL" in decision.reason_codes


def test_adaptive_vwap_refuses_a_dislocation_that_is_no_longer_a_displacement():
    algorithm = get_algorithm("adaptive_anchored_vwap_reversion")
    decision = algorithm.entry(
        _vwap_features(price=50_000.0), _context("adaptive_anchored_vwap_reversion")
    )
    assert not decision.triggered
    assert "ADAPTIVE_VWAP_DISLOCATION_NOT_REVERSION" in decision.reason_codes


def test_adaptive_vwap_fails_closed_without_a_volatility_scale():
    algorithm = get_algorithm("adaptive_anchored_vwap_reversion")
    decision = algorithm.entry(
        _vwap_features(realized_volatility=None, realized_volatility_10s=None),
        _context("adaptive_anchored_vwap_reversion"),
    )
    assert not decision.triggered
    assert "ADAPTIVE_VWAP_VOLATILITY_UNAVAILABLE" in decision.reason_codes


def test_adaptive_vwap_will_not_buy_while_liquidity_is_still_leaving():
    algorithm = get_algorithm("adaptive_anchored_vwap_reversion")
    widening = algorithm.entry(
        _vwap_features(spread_change_5s=0.0005),
        _context("adaptive_anchored_vwap_reversion"),
    )
    assert not widening.triggered
    assert "ADAPTIVE_VWAP_SPREAD_STILL_WIDENING" in widening.reason_codes

    falling = algorithm.entry(
        _vwap_features(return_1s=-0.001), _context("adaptive_anchored_vwap_reversion")
    )
    assert not falling.triggered
    assert "ADAPTIVE_VWAP_STILL_FALLING" in falling.reason_codes


def test_adaptive_vwap_uses_the_election_anchor_when_supplied():
    algorithm = get_algorithm("adaptive_anchored_vwap_reversion")
    decision = algorithm.entry(
        _vwap_features(),
        _context(
            "adaptive_anchored_vwap_reversion",
            anchored_vwap=70_200.0,
            anchor_basis="volatility_spike",
        ),
    )
    assert decision.triggered, decision.reason_codes
    assert decision.diagnostics["anchor_basis"] == "volatility_spike"
    assert decision.diagnostics["anchored_vwap"] == 70_200.0


def test_adaptive_vwap_target_never_exceeds_the_anchor():
    algorithm = get_algorithm("adaptive_anchored_vwap_reversion")
    rule = algorithm.exit_rule(
        69_300.0, _vwap_features(), _context("adaptive_anchored_vwap_reversion")
    )
    assert rule.target_price is not None
    assert 69_300.0 < rule.target_price <= 70_000.0
    assert rule.stop_price is not None and rule.stop_price < 69_300.0


# --------------------------------------------------------------------------- #
# OFI / microprice exhaustion reversal                                         #
# --------------------------------------------------------------------------- #
def _ofi_features(**overrides) -> TechnicalFeatureSet:
    base = dict(
        return_10s=-0.006,
        orderbook_imbalance=0.35,
        orderbook_imbalance_change_5s=0.08,
        depth_ratio=1.6,
        spread_change_5s=-0.0003,
        aggressor_imbalance_5s=-0.1,
        spread_bps=12.0,
    )
    base.update(overrides)
    return _features(**base)


def test_ofi_exhaustion_fires_when_the_book_turns_not_when_price_falls():
    algorithm = get_algorithm("ofi_microprice_exhaustion_reversal")
    decision = algorithm.entry(
        _ofi_features(), _context("ofi_microprice_exhaustion_reversal")
    )
    assert decision.triggered, decision.reason_codes
    assert "MICROPRICE_ABOVE_MID" in decision.reason_codes
    assert "DEPTH_RECOVERING" in decision.reason_codes


def test_ofi_exhaustion_requires_more_than_a_price_drop():
    algorithm = get_algorithm("ofi_microprice_exhaustion_reversal")
    flat_ofi = algorithm.entry(
        _ofi_features(orderbook_imbalance_change_5s=-0.02),
        _context("ofi_microprice_exhaustion_reversal"),
    )
    assert not flat_ofi.triggered
    assert "OFI_SLOPE_NOT_POSITIVE" in flat_ofi.reason_codes

    thin_bid = algorithm.entry(
        _ofi_features(depth_ratio=0.6),
        _context("ofi_microprice_exhaustion_reversal"),
    )
    assert not thin_bid.triggered
    assert "OFI_ASK_NOT_DEPLETED" in thin_bid.reason_codes


def test_ofi_exhaustion_treats_toxicity_as_risk_never_as_a_buy_reason():
    algorithm = get_algorithm("ofi_microprice_exhaustion_reversal")
    decision = algorithm.entry(
        _ofi_features(),
        _context("ofi_microprice_exhaustion_reversal", flow_toxicity=0.95),
    )
    assert not decision.triggered
    assert "OFI_EXHAUSTION_FLOW_TOXIC" in decision.reason_codes


def test_ofi_exhaustion_needs_an_actual_selloff():
    algorithm = get_algorithm("ofi_microprice_exhaustion_reversal")
    decision = algorithm.entry(
        _ofi_features(return_10s=0.001), _context("ofi_microprice_exhaustion_reversal")
    )
    assert not decision.triggered
    assert "OFI_EXHAUSTION_NO_SELLOFF" in decision.reason_codes


def test_ofi_exhaustion_has_the_shortest_thesis_driven_holding_window():
    """Shortest among theses whose horizon is set by the THESIS.

    ``market_intraday_momentum`` is excluded because its horizon is set by session
    structure, not by how long its edge persists: it is entered in the last
    continuous half-hour and must be flat before the 15:20 KRX closing auction, so
    1500s is a deadline rather than a claim about decay speed.
    """
    holding = {
        strategy_id: max_holding_seconds(strategy_id)
        for strategy_id in STRATEGY_IDS
        if strategy_id != "market_intraday_momentum"
    }
    assert holding["ofi_microprice_exhaustion_reversal"] == min(holding.values())


def test_every_algorithm_rejects_a_cold_tick_window():
    """No thesis in this module may fire before the tick window is populated."""
    cold = _features(second_data_ready=0.0, tick_count_5s=0.0)
    for strategy_id in STRATEGY_IDS:
        algorithm = get_algorithm(strategy_id)
        decision = algorithm.entry(cold, _context(strategy_id))
        assert not decision.triggered, strategy_id


# --------------------------------------------------------------------------- #
# Exit geometry is one table                                                    #
# --------------------------------------------------------------------------- #
def test_exit_geometry_covers_every_catalogued_strategy():
    geometries = all_geometries()
    for strategy_id in STRATEGY_IDS:
        assert strategy_id in geometries
        stop, target, trailing = exit_bps(strategy_id)
        assert stop > 0 and target > 0 and trailing > 0
        # Every strategy targets more than it risks; a sub-1 reward:risk ratio on a
        # small account is how the loss-accumulation pathology started.
        assert target > stop


def test_unknown_strategy_gets_the_tight_fallback_not_the_widest_rope():
    """An unidentified thesis must never get more rope than a known one.

    Asserted as a relation against the real table rather than as literals: the
    numbers are retuned when measured costs move, and pinning them here only
    produced a failure that says "the table changed", not "the table is wrong".
    """
    fallback = exit_geometry("no_such_strategy")
    known = tuple(all_geometries().values())
    assert fallback.stop_loss_bps <= min(g.stop_loss_bps for g in known)
    assert fallback.max_holding_seconds <= min(g.max_holding_seconds for g in known)
    # Least rope still has to be a viable trade, not a guaranteed cost loss.
    assert fallback.take_profit_bps > fallback.stop_loss_bps


def test_label_barriers_match_the_executor_not_the_old_inverted_defaults():
    """The core label/execution mismatch: labels must not tolerate a drawdown the
    executor would never sit through."""
    geometry = exit_geometry("intraday_momentum")
    barriers = geometry.as_label_barriers()
    # The point is that both sides read one table, whatever its current values.
    assert barriers["take_profit_bps"] == geometry.take_profit_bps
    assert barriers["stop_loss_bps"] == geometry.stop_loss_bps
    assert barriers["horizon_seconds"] == float(geometry.max_holding_seconds)
    # The old generic label was tp=25 / sl=100 — the opposite assignment.
    assert barriers["take_profit_bps"] > barriers["stop_loss_bps"]


def test_geometry_is_environment_overridable(monkeypatch):
    monkeypatch.setenv("EXIT_GEOMETRY_INTRADAY_MOMENTUM_STOP_LOSS_BPS", "30")
    assert exit_geometry("intraday_momentum").stop_loss_bps == 30.0


def test_algorithm_config_exposes_the_new_sections():
    config = AlgorithmConfig()
    resolved = config.as_dict()
    for strategy_id in (
        "residual_relative_strength",
        "adaptive_anchored_vwap_reversion",
        "ofi_microprice_exhaustion_reversal",
    ):
        assert resolved[strategy_id]["live_authorized"] == 0.0
        assert resolved[strategy_id]["shadow_enabled"] == 1.0
