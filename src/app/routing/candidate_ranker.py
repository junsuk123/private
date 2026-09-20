"""Explainable cross-sectional candidate ranking (WHAT, never WHEN).

The ranker is deliberately pure and has no broker dependency.  It turns
point-in-time observations into a stable score and a component ledger; strategy
entry algorithms remain the only owners of timing.  A 200-bar return excluding
the most recent 20 bars is the intraday, timeframe-normalised counterpart of
12-1 momentum.  Daily/weekly replays can supply conventional 12-1 values through
the same field without changing the ranking code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "multi_strategy_routing.yaml"
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CandidateObservation:
    symbol: str
    momentum_12_1: float | None = None
    realized_volatility: float | None = None
    relative_strength: float | None = None
    long_trend_score: float | None = None
    liquidity_score: float | None = None
    regime_fit: float | None = None
    ontology_fit: float | None = None
    gnn_suitability: float | None = None
    penalties: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRank:
    symbol: str
    score: float
    rank: int
    components: Mapping[str, float]
    penalties: Mapping[str, float]
    missing: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": self.score,
            "rank": self.rank,
            "components": dict(self.components),
            "penalties": dict(self.penalties),
            "missing": list(self.missing),
        }


def load_ranker_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, float]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(payload.get("candidate_ranker") or {})


def _percentiles(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) <= 1:
        return {symbol: 0.5 for symbol, _ in ordered}
    return {
        symbol: index / (len(ordered) - 1)
        for index, (symbol, _) in enumerate(ordered)
    }


def rank_candidates(
    observations: Sequence[CandidateObservation],
    *,
    config: Mapping[str, float] | None = None,
) -> tuple[CandidateRank, ...]:
    """Rank one point-in-time cross-section and retain every score component."""
    cfg = dict(config or load_ranker_config())
    neutral = _clamp(float(cfg.get("missing_neutral", 0.5)))
    by_symbol = {str(item.symbol).upper().strip(): item for item in observations}
    momentum = {
        symbol: value
        for symbol, item in by_symbol.items()
        if (value := _finite(item.momentum_12_1)) is not None
    }
    risk_adjusted = {}
    relative = {}
    for symbol, item in by_symbol.items():
        mom = _finite(item.momentum_12_1)
        vol = _finite(item.realized_volatility)
        if mom is not None and vol is not None and vol > 0.0:
            risk_adjusted[symbol] = mom / max(vol, 1e-6)
        rs = _finite(item.relative_strength)
        if rs is not None:
            relative[symbol] = rs
    momentum_pct = _percentiles(momentum)
    risk_pct = _percentiles(risk_adjusted)
    relative_pct = _percentiles(relative)
    weights = {
        name: max(0.0, float(cfg.get(name, 0.0)))
        for name in (
            "momentum_12_1", "risk_adjusted_momentum", "relative_strength",
            "long_trend_filter", "liquidity", "regime_fit", "ontology_fit",
            "gnn_suitability",
        )
    }
    total_weight = sum(weights.values()) or 1.0
    scored: list[tuple[str, float, dict[str, float], dict[str, float], tuple[str, ...]]] = []
    for symbol, item in by_symbol.items():
        raw = {
            "momentum_12_1": momentum_pct.get(symbol),
            "risk_adjusted_momentum": risk_pct.get(symbol),
            "relative_strength": relative_pct.get(symbol),
            "long_trend_filter": _finite(item.long_trend_score),
            "liquidity": _finite(item.liquidity_score),
            "regime_fit": _finite(item.regime_fit),
            "ontology_fit": _finite(item.ontology_fit),
            "gnn_suitability": _finite(item.gnn_suitability),
        }
        missing = tuple(name for name, value in raw.items() if value is None)
        components = {
            name: _clamp(neutral if value is None else value)
            for name, value in raw.items()
        }
        gross = sum(components[name] * weights[name] for name in weights) / total_weight
        penalties = {
            str(name): max(0.0, float(value))
            for name, value in dict(item.penalties or {}).items()
            if _finite(value) is not None and float(value) > 0.0
        }
        score = _clamp(gross - sum(penalties.values()))
        scored.append((symbol, score, components, penalties, missing))
    scored.sort(key=lambda row: (-row[1], row[0]))
    return tuple(
        CandidateRank(
            symbol=symbol,
            score=round(score, 6),
            rank=index,
            components=components,
            penalties=penalties,
            missing=missing,
        )
        for index, (symbol, score, components, penalties, missing) in enumerate(scored, 1)
    )


def observation_from_closes(
    symbol: str,
    closes: Sequence[float],
    *,
    liquidity_score: float | None = None,
    relative_strength: float | None = None,
    penalties: Mapping[str, float] | None = None,
    long_window: int = 200,
    skip_recent: int = 20,
    volatility_window: int = 60,
) -> CandidateObservation:
    """Build causal ranking inputs from closes available at the decision time."""
    usable = [float(value) for value in closes if _finite(value) is not None and float(value) > 0]
    momentum = None
    if len(usable) > long_window and len(usable) > skip_recent:
        momentum = usable[-skip_recent - 1] / usable[-long_window - 1] - 1.0
    returns = [
        usable[index] / usable[index - 1] - 1.0
        for index in range(max(1, len(usable) - volatility_window), len(usable))
        if usable[index - 1] > 0.0
    ]
    volatility = None
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        volatility = math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))
    long_trend = None
    if len(usable) >= long_window + 5:
        ma_now = sum(usable[-long_window:]) / long_window
        ma_prev = sum(usable[-long_window - 5:-5]) / long_window
        long_trend = _clamp(
            0.5 * (1.0 if usable[-1] > ma_now else 0.0)
            + 0.5 * (1.0 if ma_now > ma_prev else 0.0)
        )
    return CandidateObservation(
        symbol=str(symbol).upper().strip(),
        momentum_12_1=momentum,
        realized_volatility=volatility,
        relative_strength=relative_strength,
        long_trend_score=long_trend,
        liquidity_score=liquidity_score,
        regime_fit=0.5,
        ontology_fit=0.5,
        gnn_suitability=None,
        penalties=dict(penalties or {}),
    )
