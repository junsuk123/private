"""Indicator families as independent confirmation scores, not an AND gate.

Requiring every indicator to agree makes the system strictly less tradable with
each indicator added -- nineteen indicators would mean nineteen veto points, and
the honest result would be permanent NO_TRADE. Requiring none of them to agree
makes a single noisy oscillator sufficient.

The middle position, implemented here: each family produces one signed score in
[-1, 1] from whichever of its members are computable, and a family whose members
are all unavailable reports ``available=False`` instead of 0.0. A neutral reading
and a missing reading are then distinguishable downstream, which is exactly what
a model cannot recover once both have been flattened to the same number.

Evidence rule (see :func:`confirmation`): at least two INDEPENDENT families must
agree in direction. Agreement inside one family is not independent evidence --
RSI, Stochastic and Williams %R are three views of the same oscillator idea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.features.schemas import OHLCVBar
from app.technical import indicators as ti

#: Family identifiers, also used as the availability-mask keys.
TREND = "trend"
MOMENTUM = "momentum"
MEAN_REVERSION = "mean_reversion"
STRUCTURE = "structure"
VOLUME_FLOW = "volume_flow"
VOLATILITY_RISK = "volatility_risk"

FAMILY_NAMES: tuple[str, ...] = (
    TREND, MOMENTUM, MEAN_REVERSION, STRUCTURE, VOLUME_FLOW, VOLATILITY_RISK,
)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _scale(value: Any, span: float) -> float | None:
    """Map a raw reading onto [-1, 1] by a span that means 'clearly significant'."""
    number = _finite(value)
    if number is None or span <= 0:
        return None
    return _clamp(number / span)


@dataclass(frozen=True)
class FamilyScore:
    """One family's verdict. ``available`` is the mask; ``score`` is the reading."""

    name: str
    score: float = 0.0
    available: bool = False
    #: How many member indicators actually contributed.
    contributing: int = 0
    #: Raw member readings, for audit. Never fed to the model directly.
    members: Mapping[str, float] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        if not self.available:
            return 0
        if self.score > 0:
            return 1
        return -1 if self.score < 0 else 0


def _combine(name: str, members: dict[str, float | None]) -> FamilyScore:
    """Mean of available members. Absent members abstain; they do not vote 0."""
    present = {key: value for key, value in members.items() if value is not None}
    if not present:
        return FamilyScore(name=name, score=0.0, available=False, contributing=0)
    score = sum(present.values()) / len(present)
    return FamilyScore(
        name=name,
        score=_clamp(score),
        available=True,
        contributing=len(present),
        members=dict(present),
    )


# --------------------------------------------------------------------------- #
# Families                                                                     #
# --------------------------------------------------------------------------- #
def trend_family(bars: Sequence[OHLCVBar]) -> FamilyScore:
    """MA alignment/slope, MACD, DMI/ADX, TRIX, Ichimoku."""
    dmi = ti.dmi_adx(bars)
    ichimoku = ti.ichimoku(bars)
    macd = ti.macd(ti.closes(bars))
    trix = ti.trix(bars)
    close_values = ti.closes(bars)
    price = close_values[-1] if close_values else None

    macd_score = None
    if macd.ok and price:
        macd_score = _scale((macd.histogram or 0.0) / price * 10_000.0, 8.0)

    # ADX weights the DMI direction: a spread without trend strength is noise.
    dmi_score = None
    if dmi.ok and dmi.dmi_spread is not None and dmi.adx is not None:
        strength = _clamp(dmi.adx / 40.0, 0.0, 1.0)
        dmi_score = _clamp(dmi.dmi_spread / 30.0) * strength

    return _combine(
        TREND,
        {
            "ma_alignment": _finite(ti.ma_alignment_score(bars)),
            "ma_slope": _scale(ti.ma_slope_bps(bars, 20), 6.0),
            "macd_histogram": macd_score,
            "dmi": dmi_score,
            "trix": _scale(trix.trix if trix.ok else None, 4.0),
            "ichimoku_cloud": _finite(ichimoku.price_vs_cloud) if ichimoku.ok else None,
        },
    )


def momentum_family(bars: Sequence[OHLCVBar]) -> FamilyScore:
    """Momentum, ROC, RSI slope, Stochastic, Williams %R."""
    close_values = ti.closes(bars)
    rsi_now = ti.rsi(close_values, 14)
    rsi_prev = ti.rsi(close_values[:-1], 14) if len(close_values) > 15 else None
    rsi_slope = (
        (rsi_now - rsi_prev) if rsi_now is not None and rsi_prev is not None else None
    )
    williams = ti.williams_r(bars, 14)
    return _combine(
        MOMENTUM,
        {
            # RSI/Williams are re-centred so 50 / -50 is neutral.
            "rsi_level": _scale((rsi_now - 50.0) if rsi_now is not None else None, 25.0),
            "rsi_slope": _scale(rsi_slope, 8.0),
            "roc": _scale(ti.roc_bps(bars, 10), 25.0),
            "stochastic_diff": _scale(ti.stochastic_diff(bars), 15.0),
            "williams_r": _scale(
                (williams + 50.0) if williams is not None else None, 30.0
            ),
        },
    )


def mean_reversion_family(bars: Sequence[OHLCVBar]) -> FamilyScore:
    """Envelope, Bollinger, RSI, CCI, Stochastic, Williams %R, MFI.

    Sign convention is REVERSION: stretched-low reads positive (buy the dip), so
    this family opposes ``trend_family`` by construction. The regime weighting is
    what stops both from being trusted at once.
    """
    envelope = ti.envelope(bars)
    bollinger = ti.bollinger(ti.closes(bars))
    rsi_now = ti.rsi(ti.closes(bars), 14)
    cci = ti.cci(bars, 20)
    williams = ti.williams_r(bars, 14)

    envelope_score = None
    if envelope.ok and envelope.position is not None:
        envelope_score = _clamp((0.5 - envelope.position) * 2.0)
    bollinger_score = None
    if bollinger.ok and bollinger.percent_b is not None:
        bollinger_score = _clamp((0.5 - bollinger.percent_b) * 2.0)
    return _combine(
        MEAN_REVERSION,
        {
            "envelope": envelope_score,
            "bollinger_percent_b": bollinger_score,
            "rsi_extreme": _scale(
                (50.0 - rsi_now) if rsi_now is not None else None, 25.0
            ),
            "cci": _scale(-cci if cci is not None else None, 150.0),
            "williams_r": _scale(
                (-50.0 - williams) if williams is not None else None, 30.0
            ),
        },
    )


def structure_family(bars: Sequence[OHLCVBar]) -> FamilyScore:
    """Donchian/box, trendline resistance, Bollinger squeeze, ATR expansion."""
    donchian = ti.donchian(bars, 20)
    trendline = ti.trendline(bars, 60)
    bollinger = ti.bollinger(ti.closes(bars))
    close_values = ti.closes(bars)
    price = close_values[-1] if close_values else None

    breakout = None
    if price and getattr(donchian, "ok", False) and donchian.high:
        breakout = _scale((price - donchian.high) / donchian.high * 10_000.0, 20.0)

    # A positive residual z means price is ABOVE its own regression line, i.e.
    # pressing resistance -- structurally a breakout attempt, not a fade.
    residual = _scale(trendline.residual_zscore if trendline.ok else None, 1.5)
    squeeze = None
    if bollinger.ok and bollinger.bandwidth is not None:
        expansion = ti.atr_expansion(bars, 14)
        if expansion is not None:
            squeeze = _clamp((expansion - 1.0) / 0.5)
    return _combine(
        STRUCTURE,
        {
            "donchian_breakout": breakout,
            "trendline_residual": residual,
            "volatility_expansion": squeeze,
        },
    )


def volume_flow_family(
    bars: Sequence[OHLCVBar],
    *,
    aggressor_imbalance: float | None = None,
    orderbook_imbalance: float | None = None,
) -> FamilyScore:
    """Relative volume, volume z-score, OBV slope, MFI, book/aggressor imbalance."""
    return _combine(
        VOLUME_FLOW,
        {
            "relative_volume": _scale(
                (value - 1.0) if (value := ti.relative_volume(bars, 20)) is not None else None,
                1.0,
            ),
            "volume_zscore": _scale(ti.volume_zscore(bars, 20), 2.0),
            "obv_slope": _scale(ti.obv_slope(bars, 20), 0.5),
            "obv_zscore": _scale(ti.obv_zscore(bars, 20), 2.0),
            "aggressor_imbalance": _scale(aggressor_imbalance, 0.5),
            "orderbook_imbalance": _scale(orderbook_imbalance, 0.5),
        },
    )


def volatility_risk_family(
    bars: Sequence[OHLCVBar],
    *,
    spread_bps: float | None = None,
    liquidity_score: float | None = None,
) -> FamilyScore:
    """RISK, not direction: +1 is calm and tradable, -1 is hostile.

    Kept on the same [-1, 1] scale as the directional families so one weighting
    scheme covers all six, but it never contributes to direction -- it only damps
    conviction and feeds the existing risk blocks.
    """
    atr_pct = ti.atr_percent(bars, 14)
    expansion = ti.atr_expansion(bars, 14)
    members: dict[str, float | None] = {
        # 1% ATR is treated as clearly elevated for a short-horizon entry.
        "atr_pct": _clamp(1.0 - (atr_pct / 0.01)) if atr_pct is not None else None,
        "expansion": _clamp(1.0 - (expansion - 1.0) / 0.5) if expansion is not None else None,
    }
    if (spread := _finite(spread_bps)) is not None:
        members["spread"] = _clamp(1.0 - spread / 30.0)
    if (liquidity := _finite(liquidity_score)) is not None:
        members["liquidity"] = _clamp(liquidity * 2.0 - 1.0)
    return _combine(VOLATILITY_RISK, members)


# --------------------------------------------------------------------------- #
# Bundle + regime weighting                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilyBundle:
    scores: Mapping[str, FamilyScore]

    def score(self, name: str) -> float:
        item = self.scores.get(name)
        return item.score if item and item.available else 0.0

    def available(self, name: str) -> bool:
        item = self.scores.get(name)
        return bool(item and item.available)

    def availability_mask(self) -> dict[str, float]:
        """1.0/0.0 per family. Always emitted beside the value, never instead."""
        return {
            f"{name}_available": (1.0 if self.available(name) else 0.0)
            for name in FAMILY_NAMES
        }

    def as_model_features(self) -> dict[str, float]:
        """Compact, scale-free features. Value AND mask, because the live tensor
        forbids NaN and a neutral 0.0 would otherwise be indistinguishable from
        'not computable'."""
        payload = {f"{name}_family_score": self.score(name) for name in FAMILY_NAMES}
        payload.update(self.availability_mask())
        return payload

    def as_dict(self) -> dict[str, Any]:
        return {
            name: {
                "score": item.score,
                "available": item.available,
                "contributing": item.contributing,
                "members": dict(item.members),
            }
            for name, item in self.scores.items()
        }


def build_families(
    bars: Sequence[OHLCVBar],
    *,
    aggressor_imbalance: float | None = None,
    orderbook_imbalance: float | None = None,
    spread_bps: float | None = None,
    liquidity_score: float | None = None,
) -> FamilyBundle:
    return FamilyBundle(
        {
            TREND: trend_family(bars),
            MOMENTUM: momentum_family(bars),
            MEAN_REVERSION: mean_reversion_family(bars),
            STRUCTURE: structure_family(bars),
            VOLUME_FLOW: volume_flow_family(
                bars,
                aggressor_imbalance=aggressor_imbalance,
                orderbook_imbalance=orderbook_imbalance,
            ),
            VOLATILITY_RISK: volatility_risk_family(
                bars, spread_bps=spread_bps, liquidity_score=liquidity_score
            ),
        }
    )


#: The compact columns the model consumes, in schema order. Deliberately small:
#: feeding every raw indicator would multiply dimensionality without adding
#: independent information, and most raw readings are per-symbol levels (the
#: instrument-identity failure mode). Everything here is scale-free.
COMPACT_MODEL_FEATURES: tuple[str, ...] = (
    "trend_family_score",
    "momentum_family_score",
    "mean_reversion_family_score",
    "breakout_structure_score",
    "volume_flow_score",
    "volatility_risk_score",
    "trend_available",
    "momentum_available",
    "mean_reversion_available",
    "structure_available",
    "volume_flow_available",
    "adx_14",
    "dmi_spread",
    "cci_20_scaled",
    "roc_10_bps",
    "stochastic_diff",
    "williams_r_14_scaled",
    "trix_histogram_normalized",
    "obv_slope_zscore",
    "envelope_position",
    "ichimoku_cloud_position",
    "trendline_residual_zscore",
)

#: Neutral substitutes used ONLY when the paired ``*_available`` flag is 0, or
#: when a scalar has no family mask. The live tensor forbids NaN, so a number must
#: be present -- the mask is what tells the model the number means nothing.
_NEUTRAL: Mapping[str, float] = {
    "adx_14": 0.0,
    "dmi_spread": 0.0,
    "cci_20_scaled": 0.0,
    "roc_10_bps": 0.0,
    "stochastic_diff": 0.0,
    "williams_r_14_scaled": 0.0,
    "trix_histogram_normalized": 0.0,
    "obv_slope_zscore": 0.0,
    "envelope_position": 0.5,
    "ichimoku_cloud_position": 0.0,
    "trendline_residual_zscore": 0.0,
}


def compact_model_features(
    bars: Sequence[OHLCVBar], bundle: FamilyBundle
) -> dict[str, float]:
    """The scale-free model vector contributed by the indicator layer.

    Every value is finite. Family scores are paired with an availability flag;
    the scalars fall back to a documented neutral, which is safe here because each
    is already centred (0 = no signal) rather than a per-symbol level.
    """
    dmi = ti.dmi_adx(bars)
    trix = ti.trix(bars)
    envelope = ti.envelope(bars)
    ichimoku = ti.ichimoku(bars)
    trendline = ti.trendline(bars, 60)

    def value(raw: Any, key: str, scale: float = 1.0) -> float:
        number = _finite(raw)
        if number is None:
            return float(_NEUTRAL.get(key, 0.0))
        return number / scale if scale != 1.0 else number

    payload: dict[str, float] = {
        "trend_family_score": bundle.score(TREND),
        "momentum_family_score": bundle.score(MOMENTUM),
        "mean_reversion_family_score": bundle.score(MEAN_REVERSION),
        "breakout_structure_score": bundle.score(STRUCTURE),
        "volume_flow_score": bundle.score(VOLUME_FLOW),
        "volatility_risk_score": bundle.score(VOLATILITY_RISK),
        "trend_available": 1.0 if bundle.available(TREND) else 0.0,
        "momentum_available": 1.0 if bundle.available(MOMENTUM) else 0.0,
        "mean_reversion_available": 1.0 if bundle.available(MEAN_REVERSION) else 0.0,
        "structure_available": 1.0 if bundle.available(STRUCTURE) else 0.0,
        "volume_flow_available": 1.0 if bundle.available(VOLUME_FLOW) else 0.0,
        # ADX is 0-100; scale to ~[0,1] so no column dominates standardisation.
        "adx_14": value(dmi.adx if dmi.ok else None, "adx_14", 100.0),
        "dmi_spread": value(dmi.dmi_spread if dmi.ok else None, "dmi_spread", 100.0),
        "cci_20_scaled": _clamp(value(ti.cci(bars, 20), "cci_20_scaled", 200.0)),
        "roc_10_bps": value(ti.roc_bps(bars, 10), "roc_10_bps"),
        "stochastic_diff": value(ti.stochastic_diff(bars), "stochastic_diff", 100.0),
        # Williams %R is [-100, 0]; re-centre to [-1, 1].
        "williams_r_14_scaled": value(
            (w + 50.0) if (w := ti.williams_r(bars, 14)) is not None else None,
            "williams_r_14_scaled",
            50.0,
        ),
        "trix_histogram_normalized": _clamp(
            value(trix.histogram if trix.ok else None, "trix_histogram_normalized", 10.0)
        ),
        "obv_slope_zscore": _clamp(
            value(ti.obv_zscore(bars, 20), "obv_slope_zscore", 3.0)
        ),
        "envelope_position": value(
            envelope.position if envelope.ok else None, "envelope_position"
        ),
        "ichimoku_cloud_position": value(
            ichimoku.price_vs_cloud if ichimoku.ok else None, "ichimoku_cloud_position"
        ),
        "trendline_residual_zscore": _clamp(
            value(
                trendline.residual_zscore if trendline.ok else None,
                "trendline_residual_zscore",
                3.0,
            )
        ),
    }
    # Contract check: the schema tuple and this payload must never drift apart.
    missing = set(COMPACT_MODEL_FEATURES) - set(payload)
    if missing:  # pragma: no cover - guards a future edit, not runtime input
        raise KeyError(f"compact feature(s) missing: {sorted(missing)}")
    return {name: float(payload[name]) for name in COMPACT_MODEL_FEATURES}


#: Directional families only. volatility_risk damps, it does not point.
_DIRECTIONAL: tuple[str, ...] = (TREND, MOMENTUM, MEAN_REVERSION, STRUCTURE, VOLUME_FLOW)

#: Regime -> per-family weight. Absent means weight 0 in that regime.
REGIME_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "TREND_UP": {TREND: 1.0, MOMENTUM: 0.9, STRUCTURE: 0.5, VOLUME_FLOW: 0.6,
                 MEAN_REVERSION: 0.1},
    "TREND_DOWN": {TREND: 1.0, MOMENTUM: 0.9, STRUCTURE: 0.4, VOLUME_FLOW: 0.5,
                   MEAN_REVERSION: 0.1},
    "RANGE_BOUND": {MEAN_REVERSION: 1.0, MOMENTUM: 0.4, VOLUME_FLOW: 0.4,
                    TREND: 0.2, STRUCTURE: 0.2},
    "BREAKOUT_CANDIDATE": {STRUCTURE: 1.0, VOLUME_FLOW: 0.9, TREND: 0.7,
                           MOMENTUM: 0.6, MEAN_REVERSION: 0.0},
    "MEAN_REVERSION_CANDIDATE": {MEAN_REVERSION: 1.0, VOLUME_FLOW: 0.4,
                                 MOMENTUM: 0.3, TREND: 0.2, STRUCTURE: 0.1},
}

_DEFAULT_WEIGHTS: Mapping[str, float] = {
    TREND: 0.6, MOMENTUM: 0.6, MEAN_REVERSION: 0.4, STRUCTURE: 0.5, VOLUME_FLOW: 0.5,
}

#: ADX above this means a trend is present, so fading it is discouraged.
TRENDING_ADX = 25.0


@dataclass(frozen=True)
class Confirmation:
    """Directional evidence built from INDEPENDENT families."""

    direction: int
    weighted_score: float
    agreeing_families: tuple[str, ...]
    opposing_families: tuple[str, ...]
    #: Conviction multiplier in [0, 1] applied to expected capture downstream.
    conviction: float
    reason_codes: tuple[str, ...]

    @property
    def confirmed(self) -> bool:
        return self.direction != 0 and len(self.agreeing_families) >= 2


def confirmation(
    bundle: FamilyBundle,
    *,
    regime: str = "",
    adx: float | None = None,
    minimum_families: int = 2,
) -> Confirmation:
    """Weighted family vote. Two independent families must agree to confirm.

    Conflicting families reduce ``conviction`` rather than hard-blocking, because
    a genuine turn always shows disagreement between trend and reversion families
    at the moment it happens; the cost of treating that as a veto is missing every
    turn, and the cost of ignoring it is sizing a contested setup as if it were
    clean.
    """
    weights = dict(REGIME_WEIGHTS.get(str(regime or "").upper(), _DEFAULT_WEIGHTS))
    reason_codes: list[str] = []

    # A strong ADX means the tape is trending; fading it is the classic way a
    # mean-reversion book bleeds. Suppress rather than delete the family.
    strength = _finite(adx)
    if strength is not None and strength >= TRENDING_ADX:
        weights[MEAN_REVERSION] = weights.get(MEAN_REVERSION, 0.0) * 0.25
        reason_codes.append("MEAN_REVERSION_SUPPRESSED_BY_ADX")

    weighted = 0.0
    total_weight = 0.0
    agreeing: list[str] = []
    opposing: list[str] = []
    for name in _DIRECTIONAL:
        if not bundle.available(name):
            continue
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        score = bundle.score(name)
        weighted += weight * score
        total_weight += weight
    if total_weight <= 0.0:
        return Confirmation(0, 0.0, (), (), 0.0, ("FAMILY_EVIDENCE_UNAVAILABLE",))

    weighted_score = _clamp(weighted / total_weight)
    direction = 1 if weighted_score > 0 else (-1 if weighted_score < 0 else 0)
    for name in _DIRECTIONAL:
        if not bundle.available(name) or float(weights.get(name, 0.0)) <= 0.0:
            continue
        family_direction = bundle.scores[name].direction
        if family_direction == 0:
            continue
        (agreeing if family_direction == direction else opposing).append(name)

    conviction = abs(weighted_score)
    if opposing:
        # Each opposing independent family costs conviction, floored so a single
        # dissenter cannot zero out otherwise strong evidence.
        conviction *= max(0.4, 1.0 - 0.25 * len(opposing))
        reason_codes.append("FAMILY_CONFLICT")
    risk = bundle.scores.get(VOLATILITY_RISK)
    if risk and risk.available:
        # risk score is +1 calm .. -1 hostile -> multiplier in [0.5, 1.0].
        conviction *= _clamp(0.75 + 0.25 * risk.score, 0.5, 1.0)

    if len(agreeing) < max(1, minimum_families):
        reason_codes.append("INSUFFICIENT_INDEPENDENT_FAMILIES")
        direction = 0
        conviction = 0.0
    else:
        reason_codes.append("FAMILY_CONFIRMED")
    return Confirmation(
        direction=direction,
        weighted_score=weighted_score,
        agreeing_families=tuple(agreeing),
        opposing_families=tuple(opposing),
        conviction=_clamp(conviction, 0.0, 1.0),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
    )
