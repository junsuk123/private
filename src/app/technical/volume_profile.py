"""Fixed-range volume profile: where price is, in traded-volume terms.

Why this exists
---------------
Every feature this system already computes describes MOVEMENT (returns, VWAP
deviation, RSI, order-flow imbalance) or REGIME (volatility, liquidity, breadth).
None of them describes POSITION — whether price is sitting at a level where large
volume previously traded, or in a gap where it did not.

That omission is measurable. Across 1,114 simulated fills the fill-weighted gross
edge is -0.2bps against 58.8bps of cost: the signals are not wrong-signed, they are
absent, and the losses are almost entirely cost. A strategy has no way to tell these
two situations apart today:

    price falls to the value-area low, where volume clusters and prior sessions
    found support, and is rejected downward  -> mean reversion has an edge

    price falls INTO a low-volume node with empty space beneath it and the market
    accepts the lower prices                 -> mean reversion is standing in front
                                                of a move

RSI, VWAP distance and a tick down-return look identical in both. A volume profile
separates them.

What this module is NOT
----------------------
It is not a fourteenth strategy. The catalogue already has thirteen and adding a
cold bandit arm costs a long shadow period before its lower bound can turn positive.
This is a shared market-structure layer: it tells EXISTING strategies whether the
current price location makes their thesis executable, and — via
:func:`structural_room_bps` — whether there is enough distance to the next
volume barrier to cover the round-trip cost at all.

Causality
---------
:func:`build_profile` consumes only trades at or before ``as_of``. The store's
``recent_ticks(symbol, since, until=...)`` supports that directly, so a profile can
be reconstructed for any past moment without look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import Any, Iterable, Sequence

# Share of total traded volume that defines the value area. 70% is the convention
# the practitioner literature settled on; exposed so it can be varied in research
# rather than being a hidden constant.
DEFAULT_VALUE_AREA_FRACTION = 0.70

# A node counts as high-volume when its bin holds this multiple of the mean bin
# volume, and low-volume below the low multiple. Two thresholds rather than one so
# the middle of the distribution is neither, which is the honest classification.
DEFAULT_HVN_MULTIPLE = 1.5
DEFAULT_LVN_MULTIPLE = 0.5

# Value-area positions.
BELOW_VALUE = "BELOW_VALUE"
AT_VAL = "AT_VAL"
INSIDE_VALUE = "INSIDE_VALUE"
AT_POC = "AT_POC"
AT_VAH = "AT_VAH"
ABOVE_VALUE = "ABOVE_VALUE"

# Range types. Mechanical only — a hand-drawn range cannot be backtested.
SESSION_PROFILE = "SESSION_PROFILE"
PREVIOUS_SESSION_PROFILE = "PREVIOUS_SESSION_PROFILE"
CAUSAL_SWING_PROFILE = "CAUSAL_SWING_PROFILE"

POC = "POC"
HVN = "HVN"
LVN = "LVN"


@dataclass(frozen=True)
class VolumeNode:
    node_type: str
    low: float
    high: float
    center: float
    volume: float
    normalized_volume: float


@dataclass(frozen=True)
class VolumeProfileSnapshot:
    symbol: str
    range_type: str
    range_start: datetime
    range_end: datetime
    computed_at: datetime

    poc: float
    vah: float
    val: float
    hvns: tuple[VolumeNode, ...]
    lvns: tuple[VolumeNode, ...]

    tick_count: int
    total_volume: float
    bin_size: float
    profile_quality: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "range_type": self.range_type,
            "range_start": self.range_start.isoformat(),
            "range_end": self.range_end.isoformat(),
            "computed_at": self.computed_at.isoformat(),
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val,
            "hvn_count": len(self.hvns),
            "lvn_count": len(self.lvns),
            "tick_count": self.tick_count,
            "total_volume": self.total_volume,
            "bin_size": self.bin_size,
            "profile_quality": self.profile_quality,
        }


def build_profile(
    symbol: str,
    trades: Sequence[tuple[float, float]],
    *,
    range_type: str,
    range_start: datetime,
    range_end: datetime,
    computed_at: datetime,
    bin_count: int = 60,
    value_area_fraction: float = DEFAULT_VALUE_AREA_FRACTION,
    hvn_multiple: float = DEFAULT_HVN_MULTIPLE,
    lvn_multiple: float = DEFAULT_LVN_MULTIPLE,
) -> VolumeProfileSnapshot | None:
    """Build a profile from ``(price, volume)`` pairs.

    Returns ``None`` when the input cannot support a profile — too few trades, no
    volume, or a degenerate price range. A profile computed from three prints is not
    a market structure, and returning a confident-looking object for it is how a
    feature comes to mean something other than its name.
    """
    usable = [
        (float(price), float(volume))
        for price, volume in trades
        if price and float(price) > 0 and volume and float(volume) > 0
    ]
    if len(usable) < 20:
        return None
    prices = [price for price, _ in usable]
    low, high = min(prices), max(prices)
    total_volume = sum(volume for _, volume in usable)
    if total_volume <= 0 or high <= low:
        return None

    bins = max(8, int(bin_count))
    bin_size = (high - low) / bins
    if bin_size <= 0:
        return None

    buckets = [0.0] * bins
    for price, volume in usable:
        index = int((price - low) / bin_size)
        if index >= bins:
            index = bins - 1
        buckets[index] += volume

    def bin_center(index: int) -> float:
        return low + (index + 0.5) * bin_size

    poc_index = max(range(bins), key=lambda i: buckets[i])
    poc = bin_center(poc_index)

    # Value area: expand outward from the POC, always taking the richer neighbour,
    # until the configured share of volume is enclosed. This is the standard
    # construction and is deliberately not a symmetric percentile of price.
    target = total_volume * max(0.05, min(0.99, value_area_fraction))
    lower = upper = poc_index
    captured = buckets[poc_index]
    while captured < target and (lower > 0 or upper < bins - 1):
        below = buckets[lower - 1] if lower > 0 else -1.0
        above = buckets[upper + 1] if upper < bins - 1 else -1.0
        if above >= below:
            upper += 1
            captured += buckets[upper]
        else:
            lower -= 1
            captured += buckets[lower]
    val = low + lower * bin_size
    vah = low + (upper + 1) * bin_size

    mean_volume = fmean(buckets) if buckets else 0.0
    hvns: list[VolumeNode] = []
    lvns: list[VolumeNode] = []
    if mean_volume > 0:
        for index, volume in enumerate(buckets):
            normalized = volume / mean_volume
            node = VolumeNode(
                node_type=HVN if normalized >= hvn_multiple else LVN,
                low=low + index * bin_size,
                high=low + (index + 1) * bin_size,
                center=bin_center(index),
                volume=volume,
                normalized_volume=normalized,
            )
            if normalized >= hvn_multiple:
                hvns.append(node)
            elif normalized <= lvn_multiple:
                lvns.append(node)

    return VolumeProfileSnapshot(
        symbol=str(symbol),
        range_type=str(range_type),
        range_start=range_start,
        range_end=range_end,
        computed_at=computed_at,
        poc=poc,
        vah=vah,
        val=val,
        hvns=tuple(hvns),
        lvns=tuple(lvns),
        tick_count=len(usable),
        total_volume=total_volume,
        bin_size=bin_size,
        profile_quality=_profile_quality(len(usable), buckets),
    )


def _profile_quality(tick_count: int, buckets: Sequence[float]) -> float:
    """0..1 confidence that this profile describes a real distribution.

    Two things make a profile untrustworthy: too few prints, and volume piled into
    one or two bins (which is a single burst, not a distribution). Both are reported
    rather than hidden, so a consumer can require quality before acting.
    """
    if tick_count <= 0:
        return 0.0
    sample = min(1.0, tick_count / 500.0)
    occupied = sum(1 for volume in buckets if volume > 0)
    spread = min(1.0, occupied / max(1, len(buckets) * 0.25))
    return round(max(0.0, min(1.0, 0.5 * sample + 0.5 * spread)), 4)


def value_area_position(profile: VolumeProfileSnapshot, price: float) -> str:
    """Where ``price`` sits relative to the value area."""
    if price <= 0:
        return INSIDE_VALUE
    tolerance = max(profile.bin_size, price * 0.0005)
    if abs(price - profile.poc) <= tolerance:
        return AT_POC
    if abs(price - profile.val) <= tolerance:
        return AT_VAL
    if abs(price - profile.vah) <= tolerance:
        return AT_VAH
    if price < profile.val:
        return BELOW_VALUE
    if price > profile.vah:
        return ABOVE_VALUE
    return INSIDE_VALUE


def next_barrier_above(
    profile: VolumeProfileSnapshot, price: float, *, minimum_normalized: float = DEFAULT_HVN_MULTIPLE
) -> float | None:
    """Nearest high-volume node above ``price`` — the first real resistance.

    This is the number that decides whether a long has room to pay for itself.
    """
    candidates = [
        node.low
        for node in profile.hvns
        if node.low > price and node.normalized_volume >= minimum_normalized
    ]
    if not candidates:
        # The value-area high is the structural boundary when no HVN sits above.
        return profile.vah if profile.vah > price else None
    return min(candidates)


def next_barrier_below(
    profile: VolumeProfileSnapshot, price: float, *, minimum_normalized: float = DEFAULT_HVN_MULTIPLE
) -> float | None:
    """Nearest high-volume node below ``price`` — the first real support."""
    candidates = [
        node.high
        for node in profile.hvns
        if node.high < price and node.normalized_volume >= minimum_normalized
    ]
    if not candidates:
        return profile.val if profile.val < price else None
    return max(candidates)


def structural_room_bps(profile: VolumeProfileSnapshot, entry_price: float) -> float | None:
    """Distance in bps from ``entry_price`` to the next barrier above.

    The system's expected move is a volatility extrapolation
    (``sigma * sqrt(horizon) * capture``) that knows nothing about what is standing
    above the entry. A 45bps forecast into an HVN 18bps away is an 18bps trade, and
    on KRX an 18bps trade cannot clear a ~28bps round trip. Returning the real room
    lets the caller cap the forecast instead of trading a number that cannot happen.
    """
    if entry_price <= 0:
        return None
    barrier = next_barrier_above(profile, entry_price)
    if barrier is None or barrier <= entry_price:
        return None
    return (barrier - entry_price) / entry_price * 10_000.0


def cost_covered_room(
    room_bps: float | None,
    expected_cost_bps: float,
    *,
    block_below: float = 1.3,
    full_size_above: float = 1.7,
) -> tuple[str, float]:
    """Classify structural room against cost. Returns ``(verdict, ratio)``.

    Thresholds mirror the existing cost-coverage policy rather than inventing a
    second scale: below 1.3x the round trip a trade is not worth taking, and full
    size waits for 1.7x. ``UNKNOWN`` when there is no usable profile — which must be
    treated as "no information", never as permission.
    """
    if room_bps is None or expected_cost_bps <= 0:
        return "UNKNOWN", 0.0
    ratio = room_bps / expected_cost_bps
    if ratio < block_below:
        return "BLOCKED", ratio
    if ratio < full_size_above:
        return "REDUCED", ratio
    return "CLEAR", ratio


def build_profile_from_store(
    symbol: str,
    store: Any,
    *,
    range_start: datetime,
    as_of: datetime,
    range_type: str = SESSION_PROFILE,
    **kwargs: Any,
) -> VolumeProfileSnapshot | None:
    """Adapter over ``RealtimeMarketDataStore``. Strictly causal via ``until=as_of``."""
    try:
        ticks = store.recent_ticks(symbol, range_start, until=as_of)
    except Exception:  # noqa: BLE001 - an unreadable store yields no profile, not a crash
        return None
    trades: list[tuple[float, float]] = [
        (float(getattr(tick, "price", 0.0) or 0.0), float(getattr(tick, "volume", 0.0) or 0.0))
        for tick in ticks or ()
    ]
    return build_profile(
        symbol,
        trades,
        range_type=range_type,
        range_start=range_start,
        range_end=as_of,
        computed_at=as_of,
        **kwargs,
    )
