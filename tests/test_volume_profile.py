"""Volume profile: the position information every existing feature lacks.

Across 1,114 simulated fills the fill-weighted gross edge is -0.2bps against 58.8bps
of cost. The signals are not wrong-signed, they are absent — and one thing no current
feature expresses is WHERE price sits in traded-volume terms. These tests pin the two
properties that make the layer usable rather than decorative:

* it refuses to produce a confident profile from data that cannot support one;
* ``structural_room_bps`` reports the distance to the next real barrier, so a
  volatility forecast can be capped by what is actually standing above the entry.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.technical.volume_profile import (
    ABOVE_VALUE,
    AT_POC,
    AT_VAH,
    BELOW_VALUE,
    INSIDE_VALUE,
    SESSION_PROFILE,
    build_profile,
    build_profile_from_store,
    cost_covered_room,
    next_barrier_above,
    next_barrier_below,
    structural_room_bps,
    value_area_position,
)

START = datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc)
END = START + timedelta(hours=6)


def _profile(trades, **kwargs):
    return build_profile(
        "005930",
        trades,
        range_type=SESSION_PROFILE,
        range_start=START,
        range_end=END,
        computed_at=END,
        **kwargs,
    )


def _bell(center: float, width: float, count: int = 400):
    """Volume concentrated around ``center`` — a single clean distribution."""
    trades = []
    for index in range(count):
        offset = ((index % 21) - 10) / 10.0  # -1..1
        price = center + offset * width
        # Heaviest volume at the centre.
        volume = 100.0 * (1.0 - abs(offset)) + 1.0
        trades.append((price, volume))
    return trades


# --------------------------------------------------------------------------- #
# Refusing to build what the data cannot support                                #
# --------------------------------------------------------------------------- #
def test_too_few_trades_yields_no_profile() -> None:
    assert _profile([(100.0, 5.0)] * 19) is None


def test_zero_volume_yields_no_profile() -> None:
    assert _profile([(100.0 + i * 0.1, 0.0) for i in range(50)]) is None


def test_flat_price_yields_no_profile() -> None:
    """A single price is not a distribution over price."""
    assert _profile([(100.0, 5.0) for _ in range(50)]) is None


def test_burst_in_one_bin_scores_low_quality() -> None:
    """Volume piled into one place is a burst, not a market structure."""
    trades = [(100.0, 50.0) for _ in range(200)] + [(101.0, 1.0) for _ in range(30)]
    profile = _profile(trades)
    assert profile is not None
    broad = _profile(_bell(100.0, 2.0))
    assert broad is not None
    assert profile.profile_quality < broad.profile_quality


# --------------------------------------------------------------------------- #
# Structure                                                                    #
# --------------------------------------------------------------------------- #
def test_poc_lands_where_volume_concentrates() -> None:
    profile = _profile(_bell(100.0, 2.0))
    assert profile is not None
    assert profile.poc == pytest.approx(100.0, abs=0.4)


def test_value_area_brackets_the_poc() -> None:
    profile = _profile(_bell(100.0, 2.0))
    assert profile is not None
    assert profile.val < profile.poc < profile.vah
    # 70% of volume must not span the entire range, or it says nothing.
    assert (profile.vah - profile.val) < (2.0 * 2.0)


def test_value_area_fraction_widens_the_area() -> None:
    trades = _bell(100.0, 2.0)
    narrow = _profile(trades, value_area_fraction=0.5)
    wide = _profile(trades, value_area_fraction=0.95)
    assert narrow is not None and wide is not None
    assert (wide.vah - wide.val) >= (narrow.vah - narrow.val)


@pytest.mark.parametrize(
    "price,expected",
    [(100.0, AT_POC), (94.0, BELOW_VALUE), (106.0, ABOVE_VALUE)],
)
def test_value_area_position(price, expected) -> None:
    profile = _profile(_bell(100.0, 2.0))
    assert profile is not None
    assert value_area_position(profile, price) == expected


def test_inside_value_is_reported_between_the_edges() -> None:
    profile = _profile(_bell(100.0, 4.0))
    assert profile is not None
    midpoint = (profile.poc + profile.vah) / 2.0
    assert value_area_position(profile, midpoint) in {INSIDE_VALUE, AT_POC, AT_VAH}


# --------------------------------------------------------------------------- #
# Structural room — the number that decides whether a trade can pay            #
# --------------------------------------------------------------------------- #
def _two_shelf_profile():
    """Volume at 100 and again at 102, with a gap between: a real barrier above."""
    trades = []
    for index in range(300):
        trades.append((100.0 + (index % 5) * 0.02, 80.0))
    for index in range(300):
        trades.append((102.0 + (index % 5) * 0.02, 80.0))
    # Sparse traversal through the gap.
    for index in range(40):
        trades.append((100.6 + index * 0.02, 1.0))
    return trades


def test_structural_room_measures_distance_to_the_next_barrier() -> None:
    profile = _profile(_two_shelf_profile())
    assert profile is not None
    room = structural_room_bps(profile, 100.1)
    assert room is not None
    # The upper shelf is ~1.9% above, so room must be materially positive and
    # nowhere near an unbounded volatility extrapolation.
    assert 30.0 < room < 400.0


def test_no_room_above_reports_none_not_zero() -> None:
    """Above everything there is no measured barrier; that is unknown, not 0bps."""
    profile = _profile(_bell(100.0, 1.0))
    assert profile is not None
    assert structural_room_bps(profile, 200.0) is None


def test_barriers_are_directional() -> None:
    profile = _profile(_two_shelf_profile())
    assert profile is not None
    above = next_barrier_above(profile, 101.0)
    below = next_barrier_below(profile, 101.0)
    assert above is not None and above > 101.0
    assert below is not None and below < 101.0


# --------------------------------------------------------------------------- #
# Cost coverage                                                                #
# --------------------------------------------------------------------------- #
def test_room_below_cost_multiple_is_blocked() -> None:
    """An 18bps trade cannot clear a 28bps KRX round trip."""
    verdict, ratio = cost_covered_room(18.0, 28.0)
    assert verdict == "BLOCKED"
    assert ratio < 1.0


def test_marginal_room_is_reduced_not_full_size() -> None:
    verdict, _ = cost_covered_room(28.0 * 1.5, 28.0)
    assert verdict == "REDUCED"


def test_ample_room_is_clear() -> None:
    verdict, ratio = cost_covered_room(28.0 * 2.0, 28.0)
    assert verdict == "CLEAR"
    assert ratio == pytest.approx(2.0)


def test_absent_profile_is_unknown_never_permission() -> None:
    """No profile must not read as approval to trade."""
    assert cost_covered_room(None, 28.0) == ("UNKNOWN", 0.0)
    assert cost_covered_room(50.0, 0.0) == ("UNKNOWN", 0.0)


# --------------------------------------------------------------------------- #
# Causality via the store adapter                                              #
# --------------------------------------------------------------------------- #
class _Tick:
    def __init__(self, price: float, volume: float) -> None:
        self.price = price
        self.volume = volume


class _Store:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def recent_ticks(self, symbol, since, *, until=None):
        self.calls.append({"symbol": symbol, "since": since, "until": until})
        return tuple(_Tick(100.0 + (i % 21) * 0.05, 10.0) for i in range(300))


def test_store_adapter_bounds_the_query_at_as_of() -> None:
    """Look-ahead would silently invent an edge; the bound must be passed through."""
    store = _Store()
    profile = build_profile_from_store(
        "005930", store, range_start=START, as_of=END, range_type=SESSION_PROFILE
    )
    assert profile is not None
    assert store.calls[0]["until"] == END
    assert profile.range_end == END


def test_unreadable_store_yields_no_profile() -> None:
    class _Broken:
        def recent_ticks(self, *a, **k):
            raise RuntimeError("db gone")

    assert (
        build_profile_from_store("005930", _Broken(), range_start=START, as_of=END)
        is None
    )
