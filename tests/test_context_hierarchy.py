"""Global -> domestic -> sector context hierarchy, and the cross-market primitives."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.context.cross_market import (
    estimate_beta,
    estimate_lead_lag,
    relative_strength,
    resample_returns,
)
from app.context.domestic_context import (
    DOMESTIC_INSUFFICIENT_BREADTH,
    DOMESTIC_NO_INDEX_DATA,
    DOMESTIC_VENUE_DIVERGENCE,
    DomesticContextBuilder,
    DomesticContextInputs,
    VenueQuote,
)
from app.context.global_context import (
    GLOBAL_NO_COVERAGE,
    GlobalContextBuilder,
    IndicatorObservation,
    load_global_indicator_config,
)
from app.context.sector_context import (
    SECTOR_NO_MEMBERS,
    SECTOR_THIN_MEMBERSHIP,
    SectorContextBuilder,
    SectorMemberObservation,
)

NOW = datetime(2026, 8, 19, 0, 10, tzinfo=timezone.utc)
CONFIG = load_global_indicator_config()


def _observation(name: str, value: float, change: float, *, minutes: int = 5, **kwargs):
    return IndicatorObservation(
        name=name,
        value=value,
        observed_at=NOW - timedelta(minutes=minutes),
        change_ratio=change,
        **kwargs,
    )


def _risk_off_world() -> list[IndicatorObservation]:
    return [
        _observation("SP500", 5400.0, -0.012, change_ratio_long=-0.02),
        _observation("NASDAQ", 18000.0, -0.018),
        _observation("SOX", 5200.0, -0.030),
        _observation("VIX", 24.0, 0.15),
        _observation("US10Y", 4.3, 0.02),
        _observation("USDKRW", 1380.0, 0.005),
        _observation("NIKKEI", 39000.0, -0.008),
        _observation("ES", 5405.0, -0.010, minutes=2),
        _observation("WTI", 78.0, -0.010),
    ]


# --------------------------------------------------------------------------- #
# Global context
# --------------------------------------------------------------------------- #
def test_risk_off_world_reads_negative_across_the_declared_outputs() -> None:
    context = GlobalContextBuilder(CONFIG).build(_risk_off_world(), captured_at=NOW)
    assert context.direction is not None and context.direction < -0.5
    assert context.risk_sentiment is not None and context.risk_sentiment < -0.5
    assert context.rates_pressure is not None and context.rates_pressure > 0.0
    assert context.fx_pressure is not None and context.fx_pressure > 0.0
    assert context.volatility is not None and context.volatility > 1.0
    assert context.confidence > 0.8


def test_vix_orientation_is_inverted_relative_to_its_raw_move() -> None:
    context = GlobalContextBuilder(CONFIG).build(_risk_off_world(), captured_at=NOW)
    risk = context.groups["risk"]
    assert risk.raw_score is not None and risk.raw_score > 0.0
    assert risk.score is not None and risk.score < 0.0


def test_alignment_is_plus_one_when_every_group_agrees() -> None:
    aligned = GlobalContextBuilder(CONFIG).build(
        [
            _observation("SP500", 5400.0, 0.012),
            _observation("SOX", 5200.0, 0.020),
            _observation("VIX", 15.0, -0.10),
            _observation("NIKKEI", 39000.0, 0.010),
            _observation("ES", 5405.0, 0.011, minutes=2),
            _observation("US10Y", 4.3, -0.01),
            _observation("USDKRW", 1380.0, -0.004),
            _observation("WTI", 78.0, 0.02),
        ],
        captured_at=NOW,
    )
    assert aligned.global_alignment == pytest.approx(1.0)


def test_alignment_cancels_when_groups_oppose() -> None:
    split = GlobalContextBuilder(CONFIG).build(
        [
            _observation("SP500", 5400.0, 0.012),
            _observation("SOX", 5200.0, 0.020),
            _observation("NIKKEI", 39000.0, -0.011),
            _observation("ES", 5405.0, -0.009, minutes=2),
            _observation("VIX", 24.0, 0.12),
            _observation("USDKRW", 1380.0, -0.006),
        ],
        captured_at=NOW,
    )
    assert split.global_alignment is not None
    assert abs(split.global_alignment) < 0.25


def test_thin_coverage_reports_no_direction_rather_than_guessing() -> None:
    context = GlobalContextBuilder(CONFIG).build(
        [_observation("WTI", 78.0, -0.03)], captured_at=NOW
    )
    assert context.direction is None
    assert GLOBAL_NO_COVERAGE in context.reason_codes
    assert context.confidence < CONFIG.minimum_group_coverage


def test_stale_observation_is_excluded_and_flagged() -> None:
    stale = IndicatorObservation(
        name="ES",
        value=5405.0,
        observed_at=NOW - timedelta(hours=6),  # futures allow only 3600s
        change_ratio=-0.05,
    )
    context = GlobalContextBuilder(CONFIG).build(
        [*_risk_off_world()[:-2], stale], captured_at=NOW
    )
    futures = context.groups["futures"]
    assert futures.stale_members == ("ES",)
    assert futures.score is None
    assert "GLOBAL_OBSERVATIONS_STALE" in context.reason_codes


def test_unknown_indicator_is_reported_not_silently_dropped() -> None:
    context = GlobalContextBuilder(CONFIG).build(
        [*_risk_off_world(), _observation("BITCOIN", 50000.0, 0.1)], captured_at=NOW
    )
    assert "GLOBAL_UNKNOWN_INDICATOR" in context.reason_codes


def test_global_context_takes_no_ticker_and_emits_no_action() -> None:
    payload = GlobalContextBuilder(CONFIG).build(_risk_off_world(), captured_at=NOW).as_dict()
    assert not {"ticker", "symbol", "action", "side"} & set(payload)


# --------------------------------------------------------------------------- #
# Domestic context
# --------------------------------------------------------------------------- #
def _domestic(**overrides) -> DomesticContextInputs:
    base = dict(
        kospi_return=-0.004,
        kosdaq_return=-0.002,
        advancing_count=300,
        declining_count=520,
        total_trading_value=9e12,
        average_trading_value=1.1e13,
        realized_volatility=0.011,
        foreign_flow=-3.2e11,
        institution_flow=-1.0e11,
        retail_flow=4.0e11,
        average_spread_bps=12.0,
        sector_returns={"semi": 0.004, "bio": -0.01, "auto": -0.005, "bank": 0.001},
        venues=(VenueQuote("KRX", mid=2600.0), VenueQuote("NXT", mid=2600.9)),
    )
    base.update(overrides)
    return DomesticContextInputs(**base)


def test_domestic_direction_is_measured_from_domestic_prices_only() -> None:
    world = GlobalContextBuilder(CONFIG).build(_risk_off_world(), captured_at=NOW)
    weak_global = DomesticContextBuilder().build(
        _domestic(kospi_return=0.008, kosdaq_return=0.010, advancing_count=600,
                  declining_count=200, foreign_flow=4e11, institution_flow=1e11),
        captured_at=NOW,
        global_context=world,
    )
    # The world is risk-off; the domestic tape is not, and the domestic direction says so.
    assert weak_global.direction is not None and weak_global.direction > 0.0
    assert weak_global.global_conflict


def test_global_weakness_needs_three_domestic_witnesses() -> None:
    builder = DomesticContextBuilder()
    all_negative = builder.build(_domestic(), captured_at=NOW)
    assert all_negative.confirms_global_weakness()

    flow_positive = builder.build(
        _domestic(foreign_flow=5e11, institution_flow=2e11), captured_at=NOW
    )
    assert not flow_positive.confirms_global_weakness()

    breadth_positive = builder.build(
        _domestic(advancing_count=600, declining_count=200), captured_at=NOW
    )
    assert not breadth_positive.confirms_global_weakness()


def test_missing_witness_cannot_confirm_weakness() -> None:
    context = DomesticContextBuilder().build(
        _domestic(foreign_flow=None, institution_flow=None), captured_at=NOW
    )
    assert context.flow is None
    assert not context.confirms_global_weakness()
    assert "DOMESTIC_NO_FLOW_DATA" in context.reason_codes


def test_absent_index_data_yields_none_not_zero() -> None:
    context = DomesticContextBuilder().build(
        _domestic(kospi_return=None, kosdaq_return=None), captured_at=NOW
    )
    assert context.direction is None
    assert DOMESTIC_NO_INDEX_DATA in context.reason_codes


def test_tiny_breadth_sample_is_suppressed() -> None:
    context = DomesticContextBuilder().build(
        _domestic(advancing_count=3, declining_count=2), captured_at=NOW
    )
    assert context.breadth is None
    assert DOMESTIC_INSUFFICIENT_BREADTH in context.reason_codes


def test_venue_divergence_lowers_confidence_and_is_flagged() -> None:
    builder = DomesticContextBuilder()
    tight = builder.build(_domestic(), captured_at=NOW)
    dislocated = builder.build(
        _domestic(venues=(VenueQuote("KRX", mid=2600.0), VenueQuote("NXT", mid=2620.0))),
        captured_at=NOW,
    )
    assert dislocated.venue_divergence == pytest.approx(1.0)
    assert DOMESTIC_VENUE_DIVERGENCE in dislocated.reason_codes
    assert dislocated.confidence < tight.confidence


def test_single_venue_reports_no_divergence_rather_than_zero() -> None:
    context = DomesticContextBuilder().build(
        _domestic(venues=(VenueQuote("KRX", mid=2600.0),)), captured_at=NOW
    )
    assert context.venue_divergence is None
    assert "DOMESTIC_SINGLE_VENUE" in context.reason_codes


# --------------------------------------------------------------------------- #
# Sector context
# --------------------------------------------------------------------------- #
#: A market return path the member histories co-move with, so the sector beta is
#: estimable rather than degenerate.
MARKET_HISTORY = [0.004, -0.006, 0.002, -0.003, 0.005, -0.002] * 8


def _members(returns, *, member_beta: float = 1.2, **kwargs) -> list[SectorMemberObservation]:
    return [
        SectorMemberObservation(
            ticker=f"00000{index}",
            session_return=value,
            volume=kwargs.get("volume", 1500.0),
            average_volume=1000.0,
            realized_volatility=0.01,
            trading_value=1e9,
            foreign_flow=kwargs.get("foreign_flow", 1e7),
            return_history=[member_beta * item for item in MARKET_HISTORY],
        )
        for index, value in enumerate(returns)
    ]


def test_sector_relative_strength_is_net_of_market() -> None:
    context = SectorContextBuilder().build(
        "semiconductor",
        _members([0.02, 0.01, -0.005, 0.003, -0.001], member_beta=1.2),
        captured_at=NOW,
        market_return=-0.003,
        market_return_history=MARKET_HISTORY,
    )
    assert context.beta is not None and context.beta.estimated
    assert context.beta.beta == pytest.approx(1.2, rel=1e-6)
    assert context.sector_return is not None
    assert context.relative_strength is not None
    # RS = R_sector - beta * R_market; the market fell, so RS exceeds the raw return.
    assert context.relative_strength == pytest.approx(
        context.sector_return - 1.2 * -0.003
    )
    assert context.relative_strength > context.sector_return


def test_thin_sector_suppresses_cross_sectional_statistics() -> None:
    context = SectorContextBuilder().build(
        "shipbuilding", _members([0.01, -0.01]), captured_at=NOW, market_return=0.0
    )
    assert SECTOR_THIN_MEMBERSHIP in context.reason_codes
    assert context.breadth is None
    assert context.leader_concentration is None
    assert context.confidence < 1.0


def test_empty_sector_reports_no_members() -> None:
    context = SectorContextBuilder().build("empty", [], captured_at=NOW)
    assert SECTOR_NO_MEMBERS in context.reason_codes
    assert context.sector_return is None
    assert context.confidence == 0.0


def test_leader_concentration_separates_broad_from_narrow_advances() -> None:
    builder = SectorContextBuilder()
    narrow = builder.build(
        "narrow", _members([0.05, 0.001, 0.0005, 0.0005]), captured_at=NOW, market_return=0.0
    )
    broad = builder.build(
        "broad", _members([0.012, 0.011, 0.010, 0.009]), captured_at=NOW, market_return=0.0
    )
    assert narrow.leader_concentration is not None and broad.leader_concentration is not None
    assert narrow.leader_concentration > broad.leader_concentration


def test_volume_z_is_not_dominated_by_one_outlier() -> None:
    builder = SectorContextBuilder()
    ordinary = builder.build(
        "a",
        [
            SectorMemberObservation(f"{i}", session_return=0.001, volume=1100.0, average_volume=1000.0)
            for i in range(5)
        ],
        captured_at=NOW,
    )
    one_spike = builder.build(
        "b",
        [
            SectorMemberObservation("0", session_return=0.001, volume=40000.0, average_volume=1000.0),
            *[
                SectorMemberObservation(f"{i}", session_return=0.001, volume=1100.0, average_volume=1000.0)
                for i in range(1, 5)
            ],
        ],
        captured_at=NOW,
    )
    assert one_spike.volume_z is not None and ordinary.volume_z is not None
    assert one_spike.volume_z > ordinary.volume_z
    # A single 40x print must not read as the whole sector printing 8x its average.
    assert one_spike.volume_z < math.log(8.0)


def test_sector_global_alignment_uses_the_linked_group() -> None:
    world = GlobalContextBuilder(CONFIG).build(_risk_off_world(), captured_at=NOW)
    strong_sector = SectorContextBuilder().build(
        "semiconductor",
        _members([0.02, 0.015, 0.01, 0.012]),
        captured_at=NOW,
        market_return=0.0,
        global_context=world,
        global_group="semiconductor",
    )
    # Semis are up domestically while SOX fell: the alignment must be negative.
    assert strong_sector.global_alignment is not None
    assert strong_sector.global_alignment < 0.0


# --------------------------------------------------------------------------- #
# Cross-market primitives
# --------------------------------------------------------------------------- #
def test_beta_recovers_a_known_slope() -> None:
    market = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01] * 6
    local = [1.5 * value for value in market]
    estimate = estimate_beta(local, market)
    assert estimate.estimated
    assert estimate.beta == pytest.approx(1.5, rel=1e-6)
    assert estimate.r_squared == pytest.approx(1.0, rel=1e-6)


def test_beta_falls_back_visibly_on_a_thin_window() -> None:
    estimate = estimate_beta([0.01, 0.02], [0.01, 0.02])
    assert not estimate.estimated
    assert estimate.beta == 1.0
    assert "BETA_INSUFFICIENT_SAMPLES" in estimate.reason_codes


def test_beta_is_clamped_and_says_so() -> None:
    market = [0.001, -0.001] * 20
    local = [50.0 * value for value in market]
    estimate = estimate_beta(local, market)
    assert estimate.clamped
    assert estimate.beta == 3.0
    assert "BETA_CLAMPED" in estimate.reason_codes


def test_relative_strength_neutralises_beta_exposure() -> None:
    market = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01] * 6
    high_beta = [2.0 * value for value in market]
    # A 2-beta name down 2% on a market down 1% has added nothing of its own.
    result = relative_strength(
        -0.02, -0.01, local_history=high_beta, reference_history=market
    )
    assert result.beta.beta == pytest.approx(2.0, rel=1e-6)
    assert result.value == pytest.approx(0.0, abs=1e-9)


def test_lead_lag_finds_a_planted_lead() -> None:
    import random

    random.seed(11)
    leader = [random.gauss(0.0, 0.01) for _ in range(240)]
    follower = [0.0, 0.0, 0.0, *leader[:-3]]
    estimate = estimate_lead_lag(leader, follower, leader_name="SOX", follower_name="KR_SEMI")
    assert estimate is not None
    assert estimate.lag_minutes == 3
    assert estimate.leads
    assert estimate.correlation > 0.9


def test_lead_lag_reports_no_lead_for_contemporaneous_series() -> None:
    import random

    random.seed(3)
    series = [random.gauss(0.0, 0.01) for _ in range(240)]
    estimate = estimate_lead_lag(series, list(series))
    assert estimate is not None
    assert estimate.lag_minutes == 0
    assert not estimate.leads


def test_resample_returns_puts_two_clocks_on_one_grid() -> None:
    start = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)
    points = [
        (start, 100.0),
        (start + timedelta(seconds=30), 101.0),
        (start + timedelta(minutes=2, seconds=10), 102.0),
        (start + timedelta(minutes=5), 103.0),
    ]
    returns = resample_returns(points, step_minutes=1)
    assert len(returns) == 5
    # Minutes with no new print carry the previous level forward and therefore produce a
    # zero return, never an interpolated move.
    assert returns[1] == pytest.approx(0.0)
    assert returns[3] == pytest.approx(0.0)
    assert sum(returns) == pytest.approx(math.log(103.0 / 100.0), rel=1e-9)
