"""Macro market ontology reasoning.

MacroMarketReasoner reads market-level context (index trend, breadth, volatility,
liquidity/trading value, macro news/disclosure, candidate universe) and produces
a market regime, macro risk level, sector ranking, the candidate symbols to hand
to micro reasoning, and the allowed/blocked micro strategies for the current
regime.

ADVISORY ONLY: it selects candidates and strategy permissions; it never creates
a buy order (`important_rule` in the spec). The authoritative gates are untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.graph.macro_micro_common import (
    MACRO_CANDIDATE_SELECTED,
    MACRO_HIGH_VOLATILITY,
    MACRO_INSUFFICIENT_DATA,
    MACRO_LOW_LIQUIDITY,
    MACRO_NEWS_SHOCK,
    MACRO_RANGE_BOUND,
    MACRO_SECTOR_STRONG,
    MACRO_TREND_DOWN,
    MACRO_TREND_UP,
    MacroRiskLevel,
    MarketRegime,
    explanation,
)

# Default per-regime micro-strategy permissions (overridden by
# config/macro_micro_ontology.yaml at wiring time — Phase 7).
DEFAULT_STRATEGY_PERMISSIONS: dict[str, dict[str, tuple[str, ...]]] = {
    MarketRegime.TREND_UP.value: {"allow": ("momentum", "breakout", "vwap_pullback"), "block": ("aggressive_countertrend_reversion",)},
    MarketRegime.BREAKOUT_MARKET.value: {"allow": ("breakout", "momentum", "vwap_pullback"), "block": ("late_breakout_chasing",)},
    MarketRegime.TREND_DOWN.value: {"allow": ("sell", "reduce_risk", "defensive_hold"), "block": ("weak_breakout_buy", "low_volume_momentum_buy")},
    MarketRegime.RANGE_BOUND.value: {"allow": ("mean_reversion", "vwap_reversion"), "block": ("late_breakout_chasing",)},
    MarketRegime.HIGH_VOLATILITY_RISK.value: {"allow": ("sell", "reduce_risk", "hold"), "block": ("new_buy",)},
    MarketRegime.LOW_LIQUIDITY_MARKET.value: {"allow": ("sell", "reduce_risk", "hold"), "block": ("new_buy",)},
    MarketRegime.NEWS_SHOCK.value: {"allow": ("sell", "reduce_risk", "hold"), "block": ("new_buy",)},
    MarketRegime.NO_TRADE_MARKET.value: {"allow": ("sell", "reduce_risk", "hold"), "block": ("new_buy",)},
}


@dataclass(frozen=True)
class MacroReasonerConfig:
    candidate_limit: int = 30
    minimum_macro_confidence: float = 0.5
    high_volatility_threshold: float = 0.02          # market realized volatility
    low_liquidity_total_value: float = 0.0           # 0 disables (unknown-safe)
    weak_breadth_threshold: float = 0.45             # advancers / total
    strong_breadth_threshold: float = 0.55
    news_shock_severity_threshold: float = 0.7
    block_buy_on_high_volatility: bool = True
    block_buy_on_news_shock: bool = True
    block_buy_on_low_liquidity_market: bool = True
    strategy_permissions: Mapping[str, Mapping[str, Sequence[str]]] = field(
        default_factory=lambda: DEFAULT_STRATEGY_PERMISSIONS
    )

    @classmethod
    def from_env(cls) -> "MacroReasonerConfig":
        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            try:
                return float(raw) if raw not in (None, "") else default
            except ValueError:
                return default

        def _i(name: str, default: int) -> int:
            raw = os.getenv(name)
            try:
                return int(raw) if raw not in (None, "") else default
            except ValueError:
                return default

        return cls(
            candidate_limit=_i("MACRO_CANDIDATE_LIMIT", cls.candidate_limit),
            minimum_macro_confidence=_f("MACRO_MIN_CONFIDENCE", cls.minimum_macro_confidence),
            high_volatility_threshold=_f("MACRO_HIGH_VOLATILITY", cls.high_volatility_threshold),
        )


@dataclass(frozen=True)
class SectorRanking:
    sector: str
    score: float
    confidence: float
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"sector": self.sector, "score": round(self.score, 4), "confidence": round(self.confidence, 4), "reason_codes": list(self.reason_codes)}


@dataclass(frozen=True)
class MacroReasoningInput:
    timestamp: datetime
    market: str = "KR"
    index_snapshots: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)   # name -> {return, trend, ...}
    sector_snapshots: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # sector -> {strength, volume_change, ...}
    market_breadth: float | None = None            # advancers / total in [0,1]
    market_volatility: float | None = None
    total_trading_value: float | None = None
    macro_news_evidence: tuple[Mapping[str, Any], ...] = ()   # each {severity, sentiment, ...}
    disclosure_evidence: tuple[Mapping[str, Any], ...] = ()
    global_market_context: Mapping[str, Any] = field(default_factory=dict)
    currency_rate: float | None = None
    interest_rate: float | None = None
    candidate_universe: tuple[str, ...] = ()
    source_freshness: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MacroReasoningResult:
    timestamp: datetime
    market_regime: MarketRegime
    macro_risk_level: MacroRiskLevel
    sector_rankings: tuple[SectorRanking, ...]
    candidate_symbols: tuple[str, ...]
    allowed_micro_strategies: tuple[str, ...]
    blocked_micro_strategies: tuple[str, ...]
    macro_confidence: float
    reason_codes: tuple[str, ...]
    explanation_paths: tuple[dict, ...]
    rdf_graph_id: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def blocks_buy(self) -> bool:
        return self.macro_risk_level == MacroRiskLevel.BLOCK_BUY or self.market_regime in (
            MarketRegime.NO_TRADE_MARKET,
            MarketRegime.HIGH_VOLATILITY_RISK,
            MarketRegime.LOW_LIQUIDITY_MARKET,
            MarketRegime.NEWS_SHOCK,
        )

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_regime": self.market_regime.value,
            "macro_risk_level": self.macro_risk_level.value,
            "sector_rankings": [s.as_dict() for s in self.sector_rankings],
            "candidate_symbols": list(self.candidate_symbols),
            "allowed_micro_strategies": list(self.allowed_micro_strategies),
            "blocked_micro_strategies": list(self.blocked_micro_strategies),
            "macro_confidence": round(self.macro_confidence, 4),
            "reason_codes": list(self.reason_codes),
            "explanation_paths": list(self.explanation_paths),
            "rdf_graph_id": self.rdf_graph_id,
            "blocks_buy": self.blocks_buy,
            "diagnostics": dict(self.diagnostics),
        }


class MacroMarketReasoner:
    def __init__(self, config: MacroReasonerConfig | None = None) -> None:
        self.config = config or MacroReasonerConfig()

    def _permissions(self, regime: MarketRegime) -> tuple[tuple[str, ...], tuple[str, ...]]:
        perms = self.config.strategy_permissions.get(regime.value, {})
        allow = tuple(str(s) for s in (perms.get("allow") or ()))
        block = tuple(str(s) for s in (perms.get("block") or ()))
        return allow, block

    def reason(self, data: MacroReasoningInput) -> MacroReasoningResult:
        cfg = self.config
        reasons: list[str] = []
        paths: list[dict] = []

        # --- Data sufficiency: conservative NO_TRADE when we cannot see the market.
        if not data.index_snapshots and not data.candidate_universe:
            return self._result(
                data, MarketRegime.NO_TRADE_MARKET, MacroRiskLevel.BLOCK_BUY,
                (), (), reasons=[MACRO_INSUFFICIENT_DATA],
                paths=[explanation(MACRO_INSUFFICIENT_DATA, "No market context available; conservative NO_TRADE.")],
                confidence=0.0,
            )

        index_trend = self._index_trend(data.index_snapshots)
        breadth = data.market_breadth
        vol = data.market_volatility
        news_severity = max((float(n.get("severity", 0.0) or 0.0) for n in data.macro_news_evidence), default=0.0)

        # --- Regime + risk (risk gates first). ---
        regime = MarketRegime.RANGE_BOUND
        risk = MacroRiskLevel.NORMAL
        if news_severity >= cfg.news_shock_severity_threshold:
            regime, risk = MarketRegime.NEWS_SHOCK, (MacroRiskLevel.BLOCK_BUY if cfg.block_buy_on_news_shock else MacroRiskLevel.HIGH)
            reasons.append(MACRO_NEWS_SHOCK)
            paths.append(explanation(MACRO_NEWS_SHOCK, f"Macro news shock severity {news_severity:.2f}.", {"severity": news_severity}))
        elif vol is not None and vol > cfg.high_volatility_threshold:
            regime, risk = MarketRegime.HIGH_VOLATILITY_RISK, (MacroRiskLevel.BLOCK_BUY if cfg.block_buy_on_high_volatility else MacroRiskLevel.HIGH)
            reasons.append(MACRO_HIGH_VOLATILITY)
            paths.append(explanation(MACRO_HIGH_VOLATILITY, f"Market volatility {vol:.4f} exceeds {cfg.high_volatility_threshold:.4f}.", {"volatility": vol}))
        elif cfg.low_liquidity_total_value > 0 and data.total_trading_value is not None and data.total_trading_value < cfg.low_liquidity_total_value:
            regime, risk = MarketRegime.LOW_LIQUIDITY_MARKET, (MacroRiskLevel.BLOCK_BUY if cfg.block_buy_on_low_liquidity_market else MacroRiskLevel.ELEVATED)
            reasons.append(MACRO_LOW_LIQUIDITY)
            paths.append(explanation(MACRO_LOW_LIQUIDITY, "Market-wide trading value below liquidity floor."))
        elif index_trend is not None and index_trend > 0 and (breadth is None or breadth >= cfg.weak_breadth_threshold):
            regime, risk = MarketRegime.TREND_UP, MacroRiskLevel.LOW
            reasons.append(MACRO_TREND_UP)
            paths.append(explanation(MACRO_TREND_UP, f"Index trend positive ({index_trend:.4f}) with acceptable breadth.", {"index_trend": index_trend, "breadth": breadth}))
        elif index_trend is not None and index_trend < 0:
            regime, risk = MarketRegime.TREND_DOWN, MacroRiskLevel.ELEVATED
            reasons.append(MACRO_TREND_DOWN)
            paths.append(explanation(MACRO_TREND_DOWN, f"Index trend negative ({index_trend:.4f}).", {"index_trend": index_trend}))
        elif index_trend is None and breadth is None and vol is None:
            # No realtime market signal at all (no live ticks/orderbooks for the
            # universe) — be honest rather than implying a tradeable range.
            regime, risk = MarketRegime.NO_TRADE_MARKET, MacroRiskLevel.NORMAL
            reasons.append(MACRO_INSUFFICIENT_DATA)
            paths.append(explanation(MACRO_INSUFFICIENT_DATA, "No realtime market data for the tracked universe; awaiting a live feed."))
        else:
            regime, risk = MarketRegime.RANGE_BOUND, MacroRiskLevel.NORMAL
            reasons.append(MACRO_RANGE_BOUND)
            paths.append(explanation(MACRO_RANGE_BOUND, "Flat index trend; range-bound regime."))

        sector_rankings = self._sector_rankings(data.sector_snapshots)
        strong_sectors = {s.sector for s in sector_rankings if s.score > 0}

        # --- Candidate selection (explainable). Blocked-buy regimes select none.
        candidates: tuple[str, ...] = ()
        if risk != MacroRiskLevel.BLOCK_BUY:
            candidates = self._select_candidates(data, strong_sectors)
            for sym in candidates:
                paths.append(explanation(MACRO_CANDIDATE_SELECTED, f"{sym} selected for micro reasoning.", {"symbol": sym}))
            if candidates:
                reasons.append(MACRO_CANDIDATE_SELECTED)
        allow, block = self._permissions(regime)
        confidence = self._confidence(data, index_trend, sector_rankings)
        return self._result(data, regime, risk, tuple(sector_rankings), candidates,
                            reasons=reasons, paths=paths, confidence=confidence,
                            allow=allow, block=block)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _index_trend(index_snapshots: Mapping[str, Mapping[str, Any]]) -> float | None:
        vals: list[float] = []
        for snap in index_snapshots.values():
            for key in ("trend", "return", "change_rate", "pct_change"):
                if key in snap and snap[key] is not None:
                    try:
                        vals.append(float(snap[key]))
                    except (TypeError, ValueError):
                        pass
                    break
        return sum(vals) / len(vals) if vals else None

    def _sector_rankings(self, sector_snapshots: Mapping[str, Mapping[str, Any]]) -> list[SectorRanking]:
        rankings: list[SectorRanking] = []
        for sector, snap in sector_snapshots.items():
            strength = _num(snap.get("strength"))
            vol_change = _num(snap.get("volume_change"))
            score = 0.6 * strength + 0.4 * vol_change
            codes = (MACRO_SECTOR_STRONG,) if score > 0 else ()
            rankings.append(SectorRanking(str(sector), score, min(1.0, abs(score)), codes))
        rankings.sort(key=lambda r: r.score, reverse=True)
        return rankings

    def _select_candidates(self, data: MacroReasoningInput, strong_sectors: set[str]) -> tuple[str, ...]:
        universe = [str(s) for s in data.candidate_universe if str(s)]
        # Prefer symbols in strong sectors when sector mapping is available.
        sector_of = {str(s): str((data.provenance.get("sector_of") or {}).get(str(s), "")) for s in universe}
        if strong_sectors and any(sector_of.values()):
            preferred = [s for s in universe if sector_of.get(s) in strong_sectors]
            rest = [s for s in universe if s not in preferred]
            ordered = preferred + rest
        else:
            ordered = universe
        return tuple(ordered[: self.config.candidate_limit])

    def _confidence(self, data: MacroReasoningInput, index_trend: float | None, sectors: list[SectorRanking]) -> float:
        coverage = 0.0
        coverage += 0.4 if index_trend is not None else 0.0
        coverage += 0.2 if data.market_breadth is not None else 0.0
        coverage += 0.2 if data.market_volatility is not None else 0.0
        coverage += 0.2 if sectors else 0.0
        return max(0.0, min(1.0, coverage))

    def _result(self, data, regime, risk, sector_rankings, candidates, *, reasons, paths,
                confidence, allow=None, block=None) -> MacroReasoningResult:
        if allow is None or block is None:
            allow, block = self._permissions(regime)
        return MacroReasoningResult(
            timestamp=data.timestamp,
            market_regime=regime,
            macro_risk_level=risk,
            sector_rankings=tuple(sector_rankings),
            candidate_symbols=tuple(candidates),
            allowed_micro_strategies=tuple(allow),
            blocked_micro_strategies=tuple(block),
            macro_confidence=confidence,
            reason_codes=tuple(dict.fromkeys(reasons)),
            explanation_paths=tuple(paths),
            diagnostics={"index_count": len(data.index_snapshots), "sector_count": len(data.sector_snapshots)},
        )


def _num(value: Any) -> float:
    try:
        v = float(value)
        return v if v == v else 0.0  # NaN-safe
    except (TypeError, ValueError):
        return 0.0
