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
    HIGH_VOLATILITY_REGIMES,
    MACRO_BREADTH_DETERIORATING,
    MACRO_BREADTH_IMPROVING,
    MACRO_CANDIDATE_SELECTED,
    MACRO_CHANGE_POINT_BLOCKS_ENTRY,
    MACRO_CORRELATION_SPIKE,
    MACRO_FOREIGN_OUTFLOW_SHOCK,
    MACRO_HIGH_VOL_DISLOCATED,
    MACRO_HIGH_VOL_MEAN_REVERTING,
    MACRO_HIGH_VOL_RECOVERY,
    MACRO_HIGH_VOL_TRENDING,
    MACRO_HIGH_VOL_UNCLASSIFIED,
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
    # A falling index is not itself a closed-world ban on every long entry.
    # Permit only strategies whose thesis can still be valid in a weak tape:
    # liquid mean reversion and stock-specific relative strength.  Directional
    # momentum/breakout remains excluded unless the regime changes.
    MarketRegime.TREND_DOWN.value: {
        "allow": (
            "sell",
            "reduce_risk",
            "defensive_hold",
            "mean_reversion",
            "vwap_reversion",
            "relative_strength",
        ),
        "block": (
            "weak_breakout_buy",
            "low_volume_momentum_buy",
            "intraday_momentum",
            "event_momentum",
            "gap_context",
            "breakout_volume",
            "rvgi_box_breakout",
        ),
    },
    MarketRegime.RANGE_BOUND.value: {"allow": ("mean_reversion", "vwap_reversion"), "block": ("late_breakout_chasing",)},
    MarketRegime.HIGH_VOLATILITY_RISK.value: {"allow": ("sell", "reduce_risk", "hold"), "block": ("new_buy",)},
    # --- High-volatility sub-regimes -------------------------------------------
    # High volatility with a persistent direction. A confirmed relative-strength
    # or momentum thesis can still be valid; countertrend reversion cannot.
    MarketRegime.HIGH_VOL_TRENDING.value: {
        "allow": ("sell", "reduce_risk", "hold", "relative_strength", "momentum"),
        "block": (
            "mean_reversion",
            "vwap_reversion",
            "aggressive_countertrend_reversion",
            "late_breakout_chasing",
        ),
    },
    # Overshoot followed by liquidity coming back: this is where a normalised VWAP
    # reversion and an order-flow exhaustion reversal belong, and where directional
    # momentum is most likely to be buying the top of a bounce.
    MarketRegime.HIGH_VOL_MEAN_REVERTING.value: {
        "allow": ("sell", "reduce_risk", "hold", "mean_reversion", "vwap_reversion"),
        "block": ("momentum", "breakout", "late_breakout_chasing"),
    },
    # Widening spreads, evaporating depth, a detected structural break. No long
    # thesis in this repository is valid here; this is the honest NO_TRADE.
    MarketRegime.HIGH_VOL_DISLOCATED.value: {
        "allow": ("sell", "reduce_risk", "hold"),
        "block": ("new_buy",),
    },
    # Index bouncing, breadth improving, foreign flow no longer bleeding: small
    # exploratory long risk is permitted, momentum chasing is not.
    MarketRegime.HIGH_VOL_RECOVERY.value: {
        "allow": ("sell", "reduce_risk", "hold", "relative_strength", "vwap_reversion", "mean_reversion"),
        "block": ("late_breakout_chasing", "low_volume_momentum_buy"),
    },
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
    # A market-wide shock should be corroborated. Requiring a single item to
    # clear the threshold let one mislabelled headline block every buy for a
    # full TTL; requiring two independent items is the cheapest real guard.
    news_shock_minimum_events: int = 2
    block_buy_on_high_volatility: bool = True
    block_buy_on_news_shock: bool = True
    block_buy_on_low_liquidity_market: bool = True
    # --- High-volatility sub-regime classification ------------------------------
    # When True, a high-volatility tape is classified into one of the four
    # sub-regimes instead of collapsing to a single blanket buy ban. Only the
    # DISLOCATED sub-regime then blocks every new buy. Set False to restore the
    # previous single-state behaviour.
    classify_high_volatility_subregimes: bool = True
    # Directional persistence: |index trend| above this makes the tape TRENDING.
    high_vol_trend_threshold: float = 0.004
    # Breadth extremes that separate a one-sided tape from an overshoot.
    high_vol_weak_breadth: float = 0.35
    high_vol_recovery_breadth: float = 0.55
    # Dislocation evidence. Any ONE of these is enough: a market with a 90th
    # percentile spread, a correlation spike (everything moving as one name) or a
    # detected structural break has no cross-sectional edge left to harvest.
    dislocation_spread_percentile: float = 0.9
    dislocation_correlation: float = 0.85
    dislocation_change_point_probability: float = 0.5
    # Foreign-flow z-score below this is an outflow shock (KR-specific but generic).
    foreign_outflow_zscore: float = -1.5
    foreign_recovery_zscore: float = 0.0
    # A detected change point suppresses new entries regardless of sub-regime.
    block_buy_on_change_point: bool = True
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

        def _b(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw in (None, ""):
                return default
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            candidate_limit=_i("MACRO_CANDIDATE_LIMIT", cls.candidate_limit),
            minimum_macro_confidence=_f("MACRO_MIN_CONFIDENCE", cls.minimum_macro_confidence),
            high_volatility_threshold=_f("MACRO_HIGH_VOLATILITY", cls.high_volatility_threshold),
            news_shock_severity_threshold=_f(
                "MACRO_NEWS_SHOCK_SEVERITY", cls.news_shock_severity_threshold
            ),
            news_shock_minimum_events=_i(
                "MACRO_NEWS_SHOCK_MIN_EVENTS", cls.news_shock_minimum_events
            ),
            classify_high_volatility_subregimes=_b(
                "MACRO_CLASSIFY_HIGH_VOL_SUBREGIMES",
                cls.classify_high_volatility_subregimes,
            ),
            high_vol_trend_threshold=_f(
                "MACRO_HIGH_VOL_TREND_THRESHOLD", cls.high_vol_trend_threshold
            ),
            high_vol_weak_breadth=_f("MACRO_HIGH_VOL_WEAK_BREADTH", cls.high_vol_weak_breadth),
            high_vol_recovery_breadth=_f(
                "MACRO_HIGH_VOL_RECOVERY_BREADTH", cls.high_vol_recovery_breadth
            ),
            dislocation_spread_percentile=_f(
                "MACRO_DISLOCATION_SPREAD_PERCENTILE", cls.dislocation_spread_percentile
            ),
            dislocation_correlation=_f(
                "MACRO_DISLOCATION_CORRELATION", cls.dislocation_correlation
            ),
            dislocation_change_point_probability=_f(
                "MACRO_DISLOCATION_CHANGE_POINT_PROBABILITY",
                cls.dislocation_change_point_probability,
            ),
            foreign_outflow_zscore=_f(
                "MACRO_FOREIGN_OUTFLOW_ZSCORE", cls.foreign_outflow_zscore
            ),
            foreign_recovery_zscore=_f(
                "MACRO_FOREIGN_RECOVERY_ZSCORE", cls.foreign_recovery_zscore
            ),
            block_buy_on_change_point=_b(
                "MACRO_BLOCK_BUY_ON_CHANGE_POINT", cls.block_buy_on_change_point
            ),
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
class SectorRankTable:
    """Real within-sector ranking by residual (market/sector-neutral) return.

    The previous ``sector_rank`` handed to ``cross_sectional_relative_strength``
    was the arbiter's global BUY rank ordered by expected net return — not a
    sector rank at all. A name could be "rank 1" while being the weakest stock in
    its own sector, so the strategy's entire thesis ("the strongest name in a
    supportive sector") was never actually tested.

    This table answers the question the thesis asks: among the tracked names in
    THIS symbol's sector, where does it sit on residual strength? A symbol with no
    known sector, or a sector with only one tracked name, has no answer — and the
    consuming algorithm then fails closed rather than trading on a fake rank.
    """

    ranks: Mapping[str, int] = field(default_factory=dict)
    sector_sizes: Mapping[str, int] = field(default_factory=dict)
    sector_of: Mapping[str, str] = field(default_factory=dict)
    residual_returns: Mapping[str, float] = field(default_factory=dict)
    long_residual_returns: Mapping[str, float] = field(default_factory=dict)
    market_betas: Mapping[str, float] = field(default_factory=dict)

    def rank_for(self, symbol: str) -> tuple[int, int] | None:
        """``(rank, sector_size)`` 1-based from the STRONG end, or ``None``."""
        key = str(symbol or "").upper()
        rank = self.ranks.get(key)
        sector = self.sector_of.get(key)
        if rank is None or not sector:
            return None
        size = int(self.sector_sizes.get(sector, 0))
        if size <= 1:
            return None
        return int(rank), size

    def weakness_rank_for(self, symbol: str) -> tuple[int, int] | None:
        """``(rank, sector_size)`` 1-based from the WEAK end.

        ``residual_relative_weakness`` asks "is this the weakest name in its
        sector?", which is rank 1 counted from the bottom. Deriving it as
        ``size - rank + 1`` rather than storing a second ranking keeps the two
        views provably consistent — a separately-built weak ranking could disagree
        with the strong one after a tie-break change, and then a symbol could be
        simultaneously the strongest and the weakest in its sector.
        """
        resolved = self.rank_for(symbol)
        if resolved is None:
            return None
        rank, size = resolved
        return size - rank + 1, size

    def residual_for(self, symbol: str) -> float | None:
        return self.residual_returns.get(str(symbol or "").upper())

    def long_residual_for(self, symbol: str) -> float | None:
        return self.long_residual_returns.get(str(symbol or "").upper())

    def beta_for(self, symbol: str) -> float | None:
        return self.market_betas.get(str(symbol or "").upper())

    def as_dict(self) -> dict:
        return {
            "ranks": {key: int(value) for key, value in sorted(self.ranks.items())},
            "sector_sizes": {key: int(value) for key, value in sorted(self.sector_sizes.items())},
            "sector_of": {key: str(value) for key, value in sorted(self.sector_of.items())},
            "residual_returns": {
                key: round(float(value), 6) for key, value in sorted(self.residual_returns.items())
            },
        }


def build_sector_rank_table(
    *,
    sector_of: Mapping[str, str],
    residual_returns: Mapping[str, float],
    long_residual_returns: Mapping[str, float] | None = None,
    market_betas: Mapping[str, float] | None = None,
) -> SectorRankTable:
    normalized_sector = {
        str(symbol).upper(): str(sector).strip()
        for symbol, sector in (sector_of or {}).items()
        if str(sector or "").strip() and str(sector).strip().lower() != "unknown"
    }
    normalized_residual = {
        str(symbol).upper(): float(value)
        for symbol, value in (residual_returns or {}).items()
        if _is_number(value)
    }
    grouped: dict[str, list[tuple[str, float]]] = {}
    for symbol, sector in normalized_sector.items():
        if symbol in normalized_residual:
            grouped.setdefault(sector, []).append((symbol, normalized_residual[symbol]))
    ranks: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for sector, members in grouped.items():
        members.sort(key=lambda item: item[1], reverse=True)
        sizes[sector] = len(members)
        for position, (symbol, _value) in enumerate(members, start=1):
            ranks[symbol] = position
    return SectorRankTable(
        ranks=ranks,
        sector_sizes=sizes,
        sector_of={symbol: sector for symbol, sector in normalized_sector.items() if symbol in ranks},
        residual_returns=normalized_residual,
        long_residual_returns={
            str(symbol).upper(): float(value)
            for symbol, value in (long_residual_returns or {}).items()
            if _is_number(value)
        },
        market_betas={
            str(symbol).upper(): float(value)
            for symbol, value in (market_betas or {}).items()
            if _is_number(value)
        },
    )


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number  # NaN-safe


@dataclass(frozen=True)
class MacroReasoningInput:
    timestamp: datetime
    market: str = "KR"
    index_snapshots: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)   # name -> {return, trend, ...}
    sector_snapshots: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # sector -> {strength, volume_change, ...}
    market_breadth: float | None = None            # advancers / total in [0,1]
    market_volatility: float | None = None
    total_trading_value: float | None = None
    # --- Additional market-state channels ---------------------------------------
    # All optional. ``None`` means "not measured", which the classifier treats as
    # unknown, never as benign. Supplied by MacroFeatureFrame where the realtime
    # data allows it to be computed honestly.
    breadth_momentum: float | None = None           # change in breadth, signed
    cross_sectional_dispersion: float | None = None  # stdev of per-symbol returns
    average_market_correlation: float | None = None  # mean pairwise correlation
    volatility_percentile: float | None = None     # current vs its own history
    volatility_of_volatility: float | None = None
    jump_ratio: float | None = None                # jump share of total variation
    spread_percentile: float | None = None
    foreign_flow_zscore: float | None = None
    change_point_probability: float | None = None
    regime_stability: float | None = None
    # Per-symbol residuals from MacroFeatureFrame, used to build a REAL within-sector
    # ranking (see SectorRankTable) instead of reusing the arbiter's global rank.
    symbol_residual_returns: Mapping[str, float] = field(default_factory=dict)
    symbol_long_residual_returns: Mapping[str, float] = field(default_factory=dict)
    symbol_market_betas: Mapping[str, float] = field(default_factory=dict)
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
    sector_rank_table: SectorRankTable = field(default_factory=SectorRankTable)
    change_point_probability: float | None = None
    regime_stability: float | None = None
    volatility_percentile: float | None = None
    spread_percentile: float | None = None
    foreign_flow_zscore: float | None = None

    @property
    def blocks_buy(self) -> bool:
        # Only the unclassifiable and the dislocated high-volatility states are a
        # blanket ban. The trending / mean-reverting / recovery sub-regimes carry
        # an elevated risk level and a NARROWED allow-list instead — a violent tape
        # with a functioning book and a cost-covering thesis is tradeable, and
        # treating it as untradeable was itself an expensive decision.
        return self.macro_risk_level == MacroRiskLevel.BLOCK_BUY or self.market_regime in (
            MarketRegime.NO_TRADE_MARKET,
            MarketRegime.HIGH_VOLATILITY_RISK,
            MarketRegime.HIGH_VOL_DISLOCATED,
            MarketRegime.LOW_LIQUIDITY_MARKET,
            MarketRegime.NEWS_SHOCK,
        )

    @property
    def is_high_volatility(self) -> bool:
        return self.market_regime in HIGH_VOLATILITY_REGIMES

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
            "is_high_volatility": self.is_high_volatility,
            "sector_rank_table": self.sector_rank_table.as_dict(),
            "change_point_probability": self.change_point_probability,
            "regime_stability": self.regime_stability,
            "volatility_percentile": self.volatility_percentile,
            "spread_percentile": self.spread_percentile,
            "foreign_flow_zscore": self.foreign_flow_zscore,
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
        news_severities = sorted(
            (float(n.get("severity", 0.0) or 0.0) for n in data.macro_news_evidence),
            reverse=True,
        )
        news_severity = news_severities[0] if news_severities else 0.0
        # Corroboration: how many independent macro items clear the bar. One
        # mislabelled headline must not be able to declare a market-wide shock.
        corroborating = sum(
            1 for value in news_severities if value >= cfg.news_shock_severity_threshold
        )
        required_events = max(1, int(cfg.news_shock_minimum_events))

        # --- Regime + risk (risk gates first). ---
        regime = MarketRegime.RANGE_BOUND
        risk = MacroRiskLevel.NORMAL
        if corroborating >= required_events:
            regime, risk = MarketRegime.NEWS_SHOCK, (MacroRiskLevel.BLOCK_BUY if cfg.block_buy_on_news_shock else MacroRiskLevel.HIGH)
            reasons.append(MACRO_NEWS_SHOCK)
            paths.append(
                explanation(
                    MACRO_NEWS_SHOCK,
                    f"Macro news shock: {corroborating} corroborating items, peak severity {news_severity:.2f}.",
                    {
                        "severity": news_severity,
                        "corroborating_events": corroborating,
                        "required_events": required_events,
                    },
                )
            )
        elif vol is not None and vol > cfg.high_volatility_threshold:
            regime, risk = self._high_volatility_regime(data, index_trend, reasons, paths)
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
    def _high_volatility_regime(
        self,
        data: MacroReasoningInput,
        index_trend: float | None,
        reasons: list[str],
        paths: list[dict],
    ) -> tuple[MarketRegime, MacroRiskLevel]:
        """Which KIND of high volatility is this, and what may still trade?

        Volatility magnitude alone cannot answer "is any long thesis still valid".
        A one-sided repricing, an overshoot that is being bought back, a market
        whose book has evaporated, and a broad recovery are four different states
        that happen to share a volatility reading. Collapsing them into one
        blanket ban is what made every violent session untradeable regardless of
        whether a valid, cost-covering thesis existed.

        Order matters: dislocation is checked FIRST, because a market that has
        stopped functioning must not be reclassified as a nice mean-reverting
        opportunity just because the index happens to be flat.
        """
        cfg = self.config
        if not cfg.classify_high_volatility_subregimes:
            return MarketRegime.HIGH_VOLATILITY_RISK, (
                MacroRiskLevel.BLOCK_BUY
                if cfg.block_buy_on_high_volatility
                else MacroRiskLevel.HIGH
            )

        breadth = data.market_breadth
        breadth_momentum = data.breadth_momentum
        correlation = data.average_market_correlation
        spread_percentile = data.spread_percentile
        change_point = data.change_point_probability
        foreign_flow = data.foreign_flow_zscore

        # --- 1. Dislocation: the market itself is impaired. --------------------
        dislocation_evidence: list[str] = []
        if (
            spread_percentile is not None
            and spread_percentile >= cfg.dislocation_spread_percentile
        ):
            dislocation_evidence.append("spread_percentile")
        if correlation is not None and correlation >= cfg.dislocation_correlation:
            dislocation_evidence.append("average_market_correlation")
        if (
            change_point is not None
            and change_point >= cfg.dislocation_change_point_probability
        ):
            dislocation_evidence.append("change_point_probability")
        if dislocation_evidence:
            reasons.append(MACRO_HIGH_VOL_DISLOCATED)
            if "average_market_correlation" in dislocation_evidence:
                reasons.append(MACRO_CORRELATION_SPIKE)
            if "change_point_probability" in dislocation_evidence:
                reasons.append(MACRO_CHANGE_POINT_BLOCKS_ENTRY)
            paths.append(
                explanation(
                    MACRO_HIGH_VOL_DISLOCATED,
                    "High volatility with an impaired market: "
                    f"{', '.join(dislocation_evidence)}. No long thesis is admissible.",
                    {
                        "spread_percentile": spread_percentile,
                        "average_market_correlation": correlation,
                        "change_point_probability": change_point,
                    },
                )
            )
            return MarketRegime.HIGH_VOL_DISLOCATED, MacroRiskLevel.BLOCK_BUY

        # A detected structural break suppresses new entries even when the tape
        # otherwise looks orderly — the models scoring it are the stale ones.
        if (
            cfg.block_buy_on_change_point
            and change_point is not None
            and change_point >= cfg.dislocation_change_point_probability
        ):
            reasons.append(MACRO_CHANGE_POINT_BLOCKS_ENTRY)
            return MarketRegime.HIGH_VOL_DISLOCATED, MacroRiskLevel.BLOCK_BUY

        # --- 2. Recovery: bouncing with widening participation. ---------------
        recovering = (
            index_trend is not None
            and index_trend > 0
            and breadth is not None
            and breadth >= cfg.high_vol_recovery_breadth
            and (breadth_momentum is None or breadth_momentum >= 0.0)
            and (foreign_flow is None or foreign_flow >= cfg.foreign_recovery_zscore)
        )
        if recovering:
            reasons.append(MACRO_HIGH_VOL_RECOVERY)
            if breadth_momentum is not None and breadth_momentum > 0:
                reasons.append(MACRO_BREADTH_IMPROVING)
            paths.append(
                explanation(
                    MACRO_HIGH_VOL_RECOVERY,
                    "High volatility recovery: index up, breadth broad, flow no longer bleeding. "
                    "Small exploratory long risk permitted.",
                    {
                        "index_trend": index_trend,
                        "breadth": breadth,
                        "breadth_momentum": breadth_momentum,
                        "foreign_flow_zscore": foreign_flow,
                    },
                )
            )
            return MarketRegime.HIGH_VOL_RECOVERY, MacroRiskLevel.ELEVATED

        # --- 3. Trending: one-sided and persistent. ---------------------------
        trending = (
            index_trend is not None
            and abs(index_trend) >= cfg.high_vol_trend_threshold
        )
        one_sided_breadth = breadth is not None and (
            breadth <= cfg.high_vol_weak_breadth or breadth >= cfg.high_vol_recovery_breadth
        )
        if trending and (one_sided_breadth or breadth is None):
            reasons.append(MACRO_HIGH_VOL_TRENDING)
            if breadth is not None and breadth <= cfg.high_vol_weak_breadth:
                reasons.append(MACRO_BREADTH_DETERIORATING)
            if foreign_flow is not None and foreign_flow <= cfg.foreign_outflow_zscore:
                reasons.append(MACRO_FOREIGN_OUTFLOW_SHOCK)
            paths.append(
                explanation(
                    MACRO_HIGH_VOL_TRENDING,
                    f"High volatility with persistent direction (index trend {index_trend:.4f}). "
                    "Only confirmed relative-strength / momentum theses remain admissible.",
                    {"index_trend": index_trend, "breadth": breadth},
                )
            )
            # A one-sided DOWN tape with an outflow shock is not a place to add
            # long risk, even though relative strength is theoretically valid.
            risk = (
                MacroRiskLevel.HIGH
                if index_trend is not None and index_trend < 0
                else MacroRiskLevel.ELEVATED
            )
            return MarketRegime.HIGH_VOL_TRENDING, risk

        # --- 4. Mean reverting: volatile but not directional. -----------------
        if index_trend is not None and abs(index_trend) < cfg.high_vol_trend_threshold:
            reasons.append(MACRO_HIGH_VOL_MEAN_REVERTING)
            paths.append(
                explanation(
                    MACRO_HIGH_VOL_MEAN_REVERTING,
                    f"High volatility without direction (index trend {index_trend:.4f}) and a "
                    "functioning book: normalised reversion theses remain admissible.",
                    {
                        "index_trend": index_trend,
                        "breadth": breadth,
                        "cross_sectional_dispersion": data.cross_sectional_dispersion,
                    },
                )
            )
            return MarketRegime.HIGH_VOL_MEAN_REVERTING, MacroRiskLevel.ELEVATED

        # --- 5. Unclassifiable: fall back to the blanket ban. -----------------
        # We cannot see enough of the market to say which sub-regime this is, so
        # the conservative single-state behaviour is the honest answer.
        reasons.append(MACRO_HIGH_VOL_UNCLASSIFIED)
        paths.append(
            explanation(
                MACRO_HIGH_VOL_UNCLASSIFIED,
                "High volatility with insufficient market context to classify a sub-regime; "
                "falling back to a conservative buy block.",
                {"index_trend": index_trend, "breadth": breadth},
            )
        )
        return MarketRegime.HIGH_VOLATILITY_RISK, (
            MacroRiskLevel.BLOCK_BUY
            if cfg.block_buy_on_high_volatility
            else MacroRiskLevel.HIGH
        )

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
        sector_rank_table = build_sector_rank_table(
            sector_of=dict((data.provenance.get("sector_of") or {})),
            residual_returns=data.symbol_residual_returns,
            long_residual_returns=data.symbol_long_residual_returns,
            market_betas=data.symbol_market_betas,
        )
        return MacroReasoningResult(
            timestamp=data.timestamp,
            market_regime=regime,
            macro_risk_level=risk,
            sector_rank_table=sector_rank_table,
            change_point_probability=data.change_point_probability,
            regime_stability=data.regime_stability,
            volatility_percentile=data.volatility_percentile,
            spread_percentile=data.spread_percentile,
            foreign_flow_zscore=data.foreign_flow_zscore,
            sector_rankings=tuple(sector_rankings),
            candidate_symbols=tuple(candidates),
            allowed_micro_strategies=tuple(allow),
            blocked_micro_strategies=tuple(block),
            macro_confidence=confidence,
            reason_codes=tuple(dict.fromkeys(reasons)),
            explanation_paths=tuple(paths),
            diagnostics={
                "index_count": len(data.index_snapshots),
                "sector_count": len(data.sector_snapshots),
                "macro_event_count": len(data.macro_news_evidence),
                "market_breadth": data.market_breadth,
                "breadth_momentum": data.breadth_momentum,
                "market_volatility": data.market_volatility,
                "volatility_percentile": data.volatility_percentile,
                "volatility_of_volatility": data.volatility_of_volatility,
                "jump_ratio": data.jump_ratio,
                "cross_sectional_dispersion": data.cross_sectional_dispersion,
                "average_market_correlation": data.average_market_correlation,
                "spread_percentile": data.spread_percentile,
                "foreign_flow_zscore": data.foreign_flow_zscore,
                "change_point_probability": data.change_point_probability,
                "regime_stability": data.regime_stability,
                "market_context_symbols": list(
                    data.provenance.get("market_context_symbols") or ()
                ),
                "market_context_symbol_count": int(
                    data.provenance.get("market_context_symbol_count") or 0
                ),
                "trading_candidate_input_count": int(
                    data.provenance.get("trading_candidate_input_count") or 0
                ),
                "market_context_source": data.provenance.get(
                    "market_context_source"
                ),
                "max_macro_event_severity": max(
                    (
                        float(item.get("severity", 0.0) or 0.0)
                        for item in data.macro_news_evidence
                    ),
                    default=0.0,
                ),
            },
        )


def _num(value: Any) -> float:
    try:
        v = float(value)
        return v if v == v else 0.0  # NaN-safe
    except (TypeError, ValueError):
        return 0.0
