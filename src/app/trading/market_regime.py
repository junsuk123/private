from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class MarketRegime(StrEnum):
    CAPITAL_PROTECTION = "capital_protection"
    CONSERVATIVE = "conservative"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class MarketRegimeEstimate:
    regime: MarketRegime
    volatility_state: str
    liquidity_state: str
    spread_state: str
    regime_score: float
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_market_regime(*, volatility: float, liquidity: float, spread_bps: float, recent_performance: float, model_ok: bool) -> MarketRegimeEstimate:
    volatility = max(0.0, float(volatility))
    liquidity = max(0.0, float(liquidity))
    spread_bps = max(0.0, float(spread_bps))
    recent_performance = float(recent_performance)

    notes: list[str] = []
    score = 0.0
    if volatility >= 0.03:
        notes.append("HIGH_VOLATILITY")
        score -= 1.0
    elif volatility <= 0.012:
        notes.append("LOW_VOLATILITY")
        score += 0.5

    if liquidity >= 3_000_000_000:
        notes.append("HIGH_LIQUIDITY")
        score += 1.0
    elif liquidity <= 300_000_000:
        notes.append("LOW_LIQUIDITY")
        score -= 0.8

    if spread_bps >= 40:
        notes.append("WIDE_SPREAD")
        score -= 0.8
    elif spread_bps <= 10:
        notes.append("TIGHT_SPREAD")
        score += 0.4

    if recent_performance >= 0.01:
        notes.append("POSITIVE_RECENT_PERFORMANCE")
        score += 0.6
    elif recent_performance <= -0.01:
        notes.append("NEGATIVE_RECENT_PERFORMANCE")
        score -= 0.6

    if not model_ok:
        notes.append("MODEL_UNAVAILABLE")
        score -= 0.5

    if score <= -1.0:
        regime = MarketRegime.CAPITAL_PROTECTION
        volatility_state = "elevated"
        liquidity_state = "thin"
    elif score < 0.0:
        regime = MarketRegime.CONSERVATIVE
        volatility_state = "moderate"
        liquidity_state = "mixed"
    elif score < 1.5:
        regime = MarketRegime.NORMAL
        volatility_state = "stable"
        liquidity_state = "healthy"
    else:
        regime = MarketRegime.AGGRESSIVE
        volatility_state = "compressed"
        liquidity_state = "deep"

    spread_state = "tight" if spread_bps <= 15 else "normal" if spread_bps <= 30 else "wide"
    return MarketRegimeEstimate(regime=regime, volatility_state=volatility_state, liquidity_state=liquidity_state, spread_state=spread_state, regime_score=round(score, 4), notes=tuple(notes))
