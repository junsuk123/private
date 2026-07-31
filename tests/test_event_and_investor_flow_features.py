"""Features that made event_momentum and residual_relative_strength trainable.

Both strategies reported zero fills for the whole life of the training set, and
neither was unprofitable -- both were UNEVALUABLE, because the labeller hardcoded
their inputs to never-fire constants. These tests pin the three things that were
actually wrong, so none of them can quietly return.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.data.investor_flow_store import (
    InvestorFlowDay,
    InvestorFlowStore,
    business_date_for,
)
from app.evaluation.stored_counterfactual import (
    _event_quantiles,
    _investor_flow_quantile,
    _rolling_mean_percentile,
)

NOW = datetime(2026, 7, 31, 3, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# 1. A mean must be ranked against means, not against single observations.      #
# --------------------------------------------------------------------------- #
def test_rolling_mean_is_ranked_against_rolling_means() -> None:
    """The defect: averaging shrinks variance, so a window mean compared with a
    distribution of singles lands near the middle almost always. That capped
    ``residual_strength_long`` below its 0.65 threshold permanently."""
    # Strictly rising series: the latest window mean is the highest window mean,
    # so a correct comparison must put it at the very top.
    series = [float(i) for i in range(60)]
    value = _rolling_mean_percentile(series, index=59, history_start=29, window=15)
    assert value == pytest.approx(1.0)

    # Strictly falling series: the latest window mean is the lowest.
    falling = [float(-i) for i in range(60)]
    assert _rolling_mean_percentile(falling, index=59, history_start=29, window=15) == 0.0


def test_rolling_mean_percentile_needs_enough_history() -> None:
    """Too little history must read as "no evidence", not as a strong signal."""
    series = [1.0, 2.0, 3.0]
    assert _rolling_mean_percentile(series, index=2, history_start=0, window=15) == 0.0


def test_flat_series_ranks_at_the_top_by_percentile_convention() -> None:
    """Documents a real sharp edge rather than asserting a tautology.

    ``causal_percentile`` counts observations <= the value, so on a CONSTANT series
    every window mean ties and the rank is 1.0 — a flat tape reads as maximum
    residual strength. That is inherent to the convention used by every feature
    here, not specific to this one, and it is why residual strength alone cannot
    arm a trade: the expert additionally requires investor flow and liquidity
    confirmation, neither of which a flat series satisfies.
    """
    series = [5.0] * 60
    assert _rolling_mean_percentile(series, index=59, history_start=29, window=15) == 1.0


# --------------------------------------------------------------------------- #
# 2. Event features from a saturated sentiment feed.                           #
# --------------------------------------------------------------------------- #
def _series(*offsets_and_scores: tuple[float, float]):
    return tuple(
        (NOW - timedelta(seconds=offset), score) for offset, score in offsets_and_scores
    )


def test_event_relevance_measures_a_burst_not_a_level() -> None:
    """96.4% of stored scores are exactly +1.0, so the level gates nothing.

    Coverage INTENSITY is what indicates an event actually happened.
    """
    # Sparse background coverage across the 6h baseline, no recent burst.
    quiet = _series(*[(float(seconds), 1.0) for seconds in range(20_000, 1_000, -2_000)])
    # Same baseline plus a cluster inside the last 15 minutes.
    burst = quiet + _series(*[(float(seconds), 1.0) for seconds in (500, 400, 300, 200, 100)])

    quiet_relevance = _event_quantiles(sorted(quiet), NOW)["event_relevance"]
    burst_relevance = _event_quantiles(sorted(burst), NOW)["event_relevance"]
    assert burst_relevance > quiet_relevance


def test_negative_coverage_blocks_bullish_confirmation() -> None:
    """The 3.6% negative rows are the only discriminating observations."""
    clean = _series((600.0, 1.0), (300.0, 1.0), (100.0, 1.0))
    mixed = _series((600.0, 1.0), (300.0, -1.0), (100.0, 1.0))
    assert _event_quantiles(sorted(clean), NOW)["event_direction"] == pytest.approx(1.0)
    assert _event_quantiles(sorted(mixed), NOW)["event_direction"] < 1.0


def test_event_features_are_causal() -> None:
    """News published AFTER the decision moment must never be consulted."""
    future_only = tuple(
        (NOW + timedelta(seconds=offset), 1.0) for offset in (60.0, 120.0, 180.0)
    )
    result = _event_quantiles(future_only, NOW)
    assert result == {"event_relevance": 0.0, "event_direction": 0.0}


def test_no_coverage_does_not_fire() -> None:
    assert _event_quantiles((), NOW) == {"event_relevance": 0.0, "event_direction": 0.0}


# --------------------------------------------------------------------------- #
# 3. Investor flow: real broker data, ranked causally.                         #
# --------------------------------------------------------------------------- #
def _day(business_date: str, foreign: float, institution: float) -> InvestorFlowDay:
    return InvestorFlowDay(
        symbol="005930",
        business_date=business_date,
        close_price=70_000.0,
        retail_net_buy_value=-(foreign + institution),
        foreign_net_buy_value=foreign,
        institution_net_buy_value=institution,
    )


def test_informed_flow_excludes_retail() -> None:
    """Retail is not subtracted into the magnitude: a heavily retail-sold day must
    not be indistinguishable from an institution-bought one."""
    day = _day("20260731", foreign=100.0, institution=50.0)
    assert day.informed_net_buy_value == pytest.approx(150.0)


def test_investor_flow_quantile_never_ranks_against_its_own_future() -> None:
    history = {
        "20260727": _day("20260727", 10.0, 0.0),
        "20260728": _day("20260728", 20.0, 0.0),
        "20260729": _day("20260729", 30.0, 0.0),
        "20260730": _day("20260730", 40.0, 0.0),
        # A huge future day must not drag today's rank down.
        "20260731": _day("20260731", 9_999.0, 0.0),
    }
    # Ranked on 07-30 against 27/28/29 only -> the highest of those.
    assert _investor_flow_quantile(history, "20260730") == pytest.approx(1.0)


def test_investor_flow_requires_a_comparison_set() -> None:
    history = {"20260731": _day("20260731", 10.0, 0.0)}
    assert _investor_flow_quantile(history, "20260731") == 0.0
    assert _investor_flow_quantile(None, "20260731") == 0.0
    # A date with no row at all cannot be ranked.
    assert _investor_flow_quantile(history, "20260730") == 0.0


def test_business_date_uses_korean_calendar_not_utc() -> None:
    """A KRX session opening at 00:00 UTC belongs to the next Korean day.

    Getting this wrong would pair every morning bar with the previous day's flow.
    """
    midnight_utc = datetime(2026, 7, 31, 0, 30, tzinfo=timezone.utc)  # 09:30 KST
    assert business_date_for(midnight_utc) == "20260731"
    late_utc = datetime(2026, 7, 30, 22, 0, tzinfo=timezone.utc)  # 07:00 KST 7/31
    assert business_date_for(late_utc) == "20260731"


def test_store_roundtrip_and_upsert(tmp_path) -> None:
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    written = store.upsert_many(
        [
            {
                "symbol": "005930",
                "business_date": "20260731",
                "close_price": 262_500.0,
                "retail_net_buy_value": -2_958_439.0,
                "foreign_net_buy_value": 2_102_954.0,
                "institution_net_buy_value": 931_837.0,
            }
        ]
    )
    assert written == 1
    history = store.history("005930")
    assert len(history) == 1
    assert history[0].foreign_net_buy_value == pytest.approx(2_102_954.0)

    # The newest business day is still trading when first fetched, so a re-read
    # must overwrite rather than be ignored.
    store.upsert_many(
        [
            {
                "symbol": "005930",
                "business_date": "20260731",
                "close_price": 263_000.0,
                "retail_net_buy_value": -3_000_000.0,
                "foreign_net_buy_value": 2_500_000.0,
                "institution_net_buy_value": 1_000_000.0,
            }
        ]
    )
    history = store.history("005930")
    assert len(history) == 1, "upsert must not duplicate the day"
    assert history[0].foreign_net_buy_value == pytest.approx(2_500_000.0)

    coverage = store.coverage()
    assert coverage["rows"] == 1 and coverage["symbols"] == 1


def test_store_ignores_malformed_rows(tmp_path) -> None:
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    assert store.upsert_many([{"symbol": "", "business_date": "20260731"}]) == 0
    assert store.upsert_many([{"symbol": "005930", "business_date": "bad"}]) == 0
