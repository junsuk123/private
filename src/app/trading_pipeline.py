from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import yaml
import numpy as np

from app.data.source_policy import compute_quality_score
from app.features.schemas import OHLCVBar
from app.features.short_horizon_features import ShortHorizonFeatures
from app.graph.npu_classifier import OntologyNpuLinearScorer
from app.native.screening import reason_mask_to_names, screen_candidates_vectorized
from app.schemas.domain import InvestorFlowSnapshot, MarketSnapshot
from app.strategy.candidate_factory import (
    StrategyCandidateFactory,
    StrategyCandidateFactoryInput,
    StrategyCandidateFactoryResult,
    StrategyFactoryConfig,
)
from app.strategy.pairs_relative_value import PairRelativeValueConfig, PairRelativeValueEngine, PairUniverseBuilder
from app.strategy.short_horizon import (
    IntradayMomentumConfig,
    IntradayMomentumEngine,
    ShortTermReversalConfig,
    ShortTermReversalEngine,
    TechnicalRuleConfig,
    TechnicalRuleEngine,
)
from app.strategy.investor_flow import assess_domestic_investor_flow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseStock:
    ticker: str
    market: str
    sector: str
    company_name: str


@dataclass(frozen=True)
class LightweightMarketSnapshot:
    ticker: str
    market: str
    sector: str
    current_price: float
    price_change_rate: float
    trading_value: float
    trading_volume: int
    volume_change_rate: float
    market_cap: float
    foreign_net_buy: float
    institution_net_buy: float
    retail_net_buy: float
    program_net_buy: float
    short_net_change: float
    upper_limit_near: bool
    new_52week_high: bool
    halt_status: bool
    management_stock_status: bool
    liquidity_score: float
    is_synthetic: bool = False
    synthetic_fields: tuple[str, ...] = ()
    estimated_fields: tuple[str, ...] = ()
    measured_fields: tuple[str, ...] = ()
    quality_score: float = 0.0


@dataclass(frozen=True)
class ReasoningTrace:
    stock_code: str
    stage: str
    input_features: dict[str, float | int | bool | str]
    fired_rules: tuple[str, ...]
    decision: str
    score: float
    reason: str


@dataclass(frozen=True)
class CandidateSelectionResult:
    candidate_stocks: tuple[str, ...]
    rejected_stocks: tuple[str, ...]
    traces: tuple[ReasoningTrace, ...]
    latency_ms: int
    api_call_count: int
    full_universe_count: int
    chart_fetch_scope: tuple[str, ...]
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class CandidateCache:
    ttl_seconds: int = 30
    _entries: dict[str, tuple[datetime, CandidateSelectionResult]] = field(default_factory=dict)

    def get(self, key: str) -> CandidateSelectionResult | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if datetime.now(timezone.utc) - stored_at > timedelta(seconds=self.ttl_seconds):
            self._entries.pop(key, None)
            return None
        return value

    def set(self, key: str, value: CandidateSelectionResult) -> None:
        self._entries[key] = (datetime.now(timezone.utc), value)


_candidate_cache = CandidateCache()
SHORT_HORIZON_STRATEGY_CONFIG_PATH = Path("config/short_horizon_strategies.yaml")


def universe_from_tickers(tickers: Iterable[str]) -> tuple[UniverseStock, ...]:
    return tuple(
        UniverseStock(
            ticker=ticker,
            market="KRX" if ticker.endswith(".KS") or ticker.isdigit() else "US",
            sector=_sector_for(ticker),
            company_name=ticker,
        )
        for ticker in dict.fromkeys(str(item).strip() for item in tickers if str(item).strip())
    )


def build_lightweight_market_snapshots(
    universe: tuple[UniverseStock, ...],
    *,
    seed: int = 42,
) -> tuple[LightweightMarketSnapshot, ...]:
    """Create low-cost market snapshots without minute chart data.

    In live mode this layer is where broker quote/ranking APIs should be wired.
    For local simulation we derive deterministic quote-like features from the
    ticker universe so the chart provider is never called for rejected names.
    """
    snapshots: list[LightweightMarketSnapshot] = []
    for stock in universe:
        unit = _stable_unit(f"{seed}:{stock.ticker}")
        price = 20 + unit * 480
        if stock.market == "KRX":
            price = 2_000 + unit * 198_000
        trading_volume = int(20_000 + _stable_unit(f"{stock.ticker}:volume") * 8_000_000)
        volume_change = -0.45 + _stable_unit(f"{stock.ticker}:volume-change") * 2.4
        price_change = -0.08 + _stable_unit(f"{stock.ticker}:price-change") * 0.18
        trading_value = price * trading_volume
        liquidity_score = min(1.0, trading_value / 3_000_000_000)
        snapshots.append(
            LightweightMarketSnapshot(
                ticker=stock.ticker,
                market=stock.market,
                sector=stock.sector,
                current_price=round(price, 2),
                price_change_rate=round(price_change, 5),
                trading_value=round(trading_value, 2),
                trading_volume=trading_volume,
                volume_change_rate=round(volume_change, 5),
                market_cap=round(trading_value * (20 + _stable_unit(f"{stock.ticker}:cap") * 900), 2),
                foreign_net_buy=round((-0.5 + _stable_unit(f"{stock.ticker}:foreign")) * trading_value * 0.08, 2),
                institution_net_buy=round((-0.5 + _stable_unit(f"{stock.ticker}:institution")) * trading_value * 0.08, 2),
                retail_net_buy=round((-0.5 + _stable_unit(f"{stock.ticker}:retail")) * trading_value * 0.08, 2),
                program_net_buy=round((-0.5 + _stable_unit(f"{stock.ticker}:program")) * trading_value * 0.05, 2),
                short_net_change=round((-0.5 + _stable_unit(f"{stock.ticker}:short")) * trading_value * 0.02, 2),
                upper_limit_near=price_change > 0.12,
                new_52week_high=_stable_unit(f"{stock.ticker}:high52") > 0.93,
                halt_status=_stable_unit(f"{stock.ticker}:halt") > 0.997,
                management_stock_status=_stable_unit(f"{stock.ticker}:management") > 0.996,
                liquidity_score=round(liquidity_score, 5),
                is_synthetic=True,
                synthetic_fields=(
                    "current_price",
                    "price_change_rate",
                    "trading_value",
                    "trading_volume",
                    "volume_change_rate",
                    "market_cap",
                    "foreign_net_buy",
                    "institution_net_buy",
                    "retail_net_buy",
                    "program_net_buy",
                    "short_net_change",
                    "upper_limit_near",
                    "new_52week_high",
                    "halt_status",
                    "management_stock_status",
                    "liquidity_score",
                ),
                quality_score=0.0,
            )
        )
    return tuple(snapshots)


def build_lightweight_market_snapshots_from_markets(
    markets: tuple[MarketSnapshot, ...],
) -> tuple[LightweightMarketSnapshot, ...]:
    snapshots: list[LightweightMarketSnapshot] = []
    for market in markets:
        trading_value = max(0.0, float(market.average_daily_trading_value or 0.0))
        price = max(0.01, float(market.last_price or 0.0))
        volume = int(trading_value / price) if price else 0
        liquidity_score = min(1.0, trading_value / 3_000_000_000)
        momentum = _stable_unit(f"{market.ticker}:analysis-momentum") * 0.18 - 0.06
        volume_change = _stable_unit(f"{market.ticker}:analysis-volume") * 1.8 - 0.35
        flow = market.investor_flow
        measured_fields = ("current_price", "trading_value", "trading_volume", "liquidity_score")
        estimated_fields = [
            "price_change_rate",
            "volume_change_rate",
            "market_cap",
            "new_52week_high",
        ]
        if flow is None:
            estimated_fields.extend(
                (
                    "foreign_net_buy",
                    "institution_net_buy",
                    "retail_net_buy",
                    "program_net_buy",
                    "short_net_change",
                )
            )
        else:
            measured_fields = measured_fields + (
                "foreign_net_buy",
                "institution_net_buy",
                "retail_net_buy",
                "program_net_buy",
                "short_net_change",
            )
        quality_score = compute_quality_score(market.source, missing_ratio=len(estimated_fields) / 16)
        snapshots.append(
            LightweightMarketSnapshot(
                ticker=market.ticker,
                market=market.market,
                sector=market.sector,
                current_price=price,
                price_change_rate=round(momentum, 5),
                trading_value=trading_value,
                trading_volume=volume,
                volume_change_rate=round(volume_change, 5),
                market_cap=trading_value * (20 + _stable_unit(f"{market.ticker}:analysis-cap") * 900),
                foreign_net_buy=_flow_value(flow, "foreign_net_buy", market.ticker, "analysis-foreign", trading_value, 0.08),
                institution_net_buy=_flow_value(flow, "institution_net_buy", market.ticker, "analysis-institution", trading_value, 0.08),
                retail_net_buy=_flow_value(flow, "retail_net_buy", market.ticker, "analysis-retail", trading_value, 0.08),
                program_net_buy=_flow_value(flow, "program_net_buy", market.ticker, "analysis-program", trading_value, 0.05),
                short_net_change=_flow_value(flow, "short_net_change", market.ticker, "analysis-short", trading_value, 0.02),
                upper_limit_near=momentum > 0.10,
                new_52week_high=_stable_unit(f"{market.ticker}:analysis-high52") > 0.94,
                halt_status=False,
                management_stock_status=False,
                liquidity_score=round(liquidity_score, 5),
                is_synthetic=market.source.is_synthetic,
                synthetic_fields=tuple(estimated_fields) if market.source.is_synthetic else (),
                estimated_fields=tuple(estimated_fields),
                measured_fields=measured_fields,
                quality_score=quality_score,
            )
        )
    return tuple(snapshots)


def validate_lightweight_snapshots_for_live(
    snapshots: tuple[LightweightMarketSnapshot, ...],
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    for snapshot in snapshots:
        if snapshot.is_synthetic or snapshot.synthetic_fields:
            reasons.append(f"{snapshot.ticker}:synthetic_fields")
        if snapshot.quality_score <= 0:
            reasons.append(f"{snapshot.ticker}:data_quality")
    return not reasons, tuple(reasons)


def ontology_filter_1(
    snapshots: tuple[LightweightMarketSnapshot, ...],
    *,
    target_count: int = 80,
    min_trading_value: float = 500_000_000,
    min_liquidity_score: float = 0.12,
    cache_key: str | None = None,
) -> CandidateSelectionResult:
    if os.getenv("ONTOLOGY_FILTER1_BACKEND", "vectorized").strip().lower() in {"loop", "python_loop", "legacy"}:
        return _ontology_filter_1_python_loop(
            snapshots,
            target_count=target_count,
            min_trading_value=min_trading_value,
            min_liquidity_score=min_liquidity_score,
            cache_key=cache_key,
        )

    cached = _candidate_cache.get(cache_key) if cache_key else None
    if cached is not None:
        return cached

    started = time.perf_counter()
    target_count = max(1, min(_max_top_k(), int(target_count)))
    arrays_started = time.perf_counter()
    tickers = tuple(snapshot.ticker for snapshot in snapshots)
    markets = tuple(snapshot.market for snapshot in snapshots)
    vectorized = screen_candidates_vectorized(
        trading_value=np.fromiter((snapshot.trading_value for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        liquidity_score=np.fromiter((snapshot.liquidity_score for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        volume_change_rate=np.fromiter((snapshot.volume_change_rate for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        price_change_rate=np.fromiter((snapshot.price_change_rate for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        foreign_net_buy=np.fromiter((snapshot.foreign_net_buy for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        institution_net_buy=np.fromiter((snapshot.institution_net_buy for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        retail_net_buy=np.fromiter((snapshot.retail_net_buy for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        program_net_buy=np.fromiter((snapshot.program_net_buy for snapshot in snapshots), dtype=np.float32, count=len(snapshots)),
        upper_limit_near=np.fromiter((snapshot.upper_limit_near for snapshot in snapshots), dtype=bool, count=len(snapshots)),
        new_52week_high=np.fromiter((snapshot.new_52week_high for snapshot in snapshots), dtype=bool, count=len(snapshots)),
        halt_status=np.fromiter((snapshot.halt_status for snapshot in snapshots), dtype=bool, count=len(snapshots)),
        management_stock_status=np.fromiter((snapshot.management_stock_status for snapshot in snapshots), dtype=bool, count=len(snapshots)),
        domestic_market=np.fromiter((_is_domestic_snapshot_market(ticker, market) for ticker, market in zip(tickers, markets)), dtype=bool, count=len(snapshots)),
        min_trading_value=min_trading_value,
        min_liquidity_score=min_liquidity_score,
        top_k=target_count,
    )
    array_build_ms = (time.perf_counter() - arrays_started) * 1000.0
    accepted = tuple(
        sorted(
            (int(index) for index in vectorized.accepted_indices),
            key=lambda index: (float(vectorized.scores[index]), tickers[index]),
            reverse=True,
        )
    )
    candidates = tuple(tickers[index] for index in accepted[:target_count])
    rejected = tuple(tickers[int(index)] for index in vectorized.rejected_indices)

    npu_metrics: dict[str, float | int | str] = {
        **vectorized.profile,
        "candidate_array_build_ms": round(array_build_ms, 3),
        "candidate_count_input": len(snapshots),
        "candidate_count_after_hard_filter": len(accepted),
    }
    if _npu_candidate_scoring_enabled() and accepted:
        npu_candidates, npu_metrics_update = _rank_accepted_with_npu(
            snapshots,
            tuple(tickers[index] for index in accepted),
            top_k=target_count,
        )
        candidates = npu_candidates
        npu_metrics.update(npu_metrics_update)
    else:
        npu_metrics.update(
            {
                "candidate_count_after_npu_topk": len(candidates),
                "npu_enabled": 0,
                "device": "CPU_RULES",
                "top_k": target_count,
            }
        )

    trace_indices = _trace_indices_for_screening(tickers, vectorized, candidates)
    traces = tuple(
        _trace_from_vectorized_snapshot(
            snapshots[int(index)],
            float(vectorized.scores[int(index)]),
            int(vectorized.reason_masks[int(index)]),
            accepted=bool(vectorized.hard_reject_masks[int(index)] == 0),
        )
        for index in trace_indices
    )
    result = CandidateSelectionResult(
        candidate_stocks=candidates,
        rejected_stocks=rejected,
        traces=traces,
        latency_ms=int((time.perf_counter() - started) * 1000),
        api_call_count=0,
        full_universe_count=len(snapshots),
        chart_fetch_scope=candidates,
        metrics=npu_metrics,
    )
    if cache_key:
        _candidate_cache.set(cache_key, result)
    logger.info(
        "ontology_filter_1 universe=%s candidates=%s rejected=%s latency_ms=%s api_calls=%s backend=%s",
        result.full_universe_count,
        len(result.candidate_stocks),
        len(result.rejected_stocks),
        result.latency_ms,
        result.api_call_count,
        result.metrics.get("backend", "unknown"),
    )
    return result


def _ontology_filter_1_python_loop(
    snapshots: tuple[LightweightMarketSnapshot, ...],
    *,
    target_count: int = 80,
    min_trading_value: float = 500_000_000,
    min_liquidity_score: float = 0.12,
    cache_key: str | None = None,
) -> CandidateSelectionResult:
    cached = _candidate_cache.get(cache_key) if cache_key else None
    if cached is not None:
        return cached

    started = time.perf_counter()
    target_count = max(1, min(_max_top_k(), int(target_count)))
    accepted: list[tuple[float, str]] = []
    rejected: list[str] = []
    traces: list[ReasoningTrace] = []
    hard_filter_count = 0

    for snapshot in snapshots:
        score, decision, fired_rules, reason = _score_lightweight_snapshot(
            snapshot,
            min_trading_value=min_trading_value,
            min_liquidity_score=min_liquidity_score,
        )
        trace = ReasoningTrace(
            stock_code=snapshot.ticker,
            stage="ontology_filter_1",
            input_features={
                "current_price": snapshot.current_price,
                "price_change_rate": snapshot.price_change_rate,
                "trading_value": snapshot.trading_value,
                "trading_volume": snapshot.trading_volume,
                "volume_change_rate": snapshot.volume_change_rate,
                "market_cap": snapshot.market_cap,
                "foreign_net_buy": snapshot.foreign_net_buy,
                "institution_net_buy": snapshot.institution_net_buy,
                "retail_net_buy": snapshot.retail_net_buy,
                "program_net_buy": snapshot.program_net_buy,
                "short_net_change": snapshot.short_net_change,
                "upper_limit_near": snapshot.upper_limit_near,
                "new_52week_high": snapshot.new_52week_high,
                "halt_status": snapshot.halt_status,
                "management_stock_status": snapshot.management_stock_status,
                "liquidity_score": snapshot.liquidity_score,
            },
            fired_rules=tuple(fired_rules),
            decision=decision,
            score=round(score, 6),
            reason=reason,
        )
        traces.append(trace)
        if decision == "CandidateStock":
            accepted.append((score, snapshot.ticker))
            hard_filter_count += 1
        else:
            rejected.append(snapshot.ticker)

    npu_metrics: dict[str, float | int | str] = {
        "candidate_count_input": len(snapshots),
        "candidate_count_after_hard_filter": hard_filter_count,
    }
    if _npu_candidate_scoring_enabled() and accepted:
        npu_candidates, npu_metrics_update = _rank_accepted_with_npu(
            snapshots,
            tuple(ticker for _score, ticker in accepted),
            top_k=target_count,
        )
        candidates = npu_candidates
        npu_metrics.update(npu_metrics_update)
    else:
        candidates = tuple(ticker for _score, ticker in sorted(accepted, reverse=True)[:target_count])
        npu_metrics.update(
            {
                "candidate_count_after_npu_topk": len(candidates),
                "npu_enabled": 0,
                "device": "CPU_RULES",
                "top_k": target_count,
            }
        )
    result = CandidateSelectionResult(
        candidate_stocks=candidates,
        rejected_stocks=tuple(rejected),
        traces=tuple(traces),
        latency_ms=int((time.perf_counter() - started) * 1000),
        api_call_count=0,
        full_universe_count=len(snapshots),
        chart_fetch_scope=candidates,
        metrics=npu_metrics,
    )
    if cache_key:
        _candidate_cache.set(cache_key, result)
    logger.info(
        "ontology_filter_1 universe=%s candidates=%s rejected=%s latency_ms=%s api_calls=%s",
        result.full_universe_count,
        len(result.candidate_stocks),
        len(result.rejected_stocks),
        result.latency_ms,
        result.api_call_count,
    )
    return result


def _trace_indices_for_screening(
    tickers: tuple[str, ...],
    vectorized: Any,
    candidates: tuple[str, ...],
    reject_sample_size: int = 24,
) -> tuple[int, ...]:
    candidate_set = set(candidates)
    selected = [index for index, ticker in enumerate(tickers) if ticker in candidate_set]
    rejected = [int(index) for index in vectorized.rejected_indices[:reject_sample_size]]
    return tuple(dict.fromkeys((*selected, *rejected)))


def _trace_from_vectorized_snapshot(
    snapshot: LightweightMarketSnapshot,
    score: float,
    reason_mask: int,
    *,
    accepted: bool,
) -> ReasoningTrace:
    fired_rules = reason_mask_to_names(reason_mask)
    decision = "CandidateStock" if accepted else "RejectStock"
    reason = (
        "Passed lightweight ontology liquidity and momentum screening."
        if accepted
        else _reject_reason_from_mask(reason_mask)
    )
    return ReasoningTrace(
        stock_code=snapshot.ticker,
        stage="ontology_filter_1",
        input_features={
            "current_price": snapshot.current_price,
            "price_change_rate": snapshot.price_change_rate,
            "trading_value": snapshot.trading_value,
            "trading_volume": snapshot.trading_volume,
            "volume_change_rate": snapshot.volume_change_rate,
            "market_cap": snapshot.market_cap,
            "foreign_net_buy": snapshot.foreign_net_buy,
            "institution_net_buy": snapshot.institution_net_buy,
            "retail_net_buy": snapshot.retail_net_buy,
            "program_net_buy": snapshot.program_net_buy,
            "short_net_change": snapshot.short_net_change,
            "upper_limit_near": snapshot.upper_limit_near,
            "new_52week_high": snapshot.new_52week_high,
            "halt_status": snapshot.halt_status,
            "management_stock_status": snapshot.management_stock_status,
            "liquidity_score": snapshot.liquidity_score,
        },
        fired_rules=fired_rules,
        decision=decision,
        score=round(score, 6),
        reason=reason,
    )


def _reject_reason_from_mask(reason_mask: int) -> str:
    names = set(reason_mask_to_names(reason_mask))
    if "halt_status" in names:
        return "Trading is halted."
    if "management_stock_status" in names:
        return "Management stock status blocks trading."
    if "insufficient_liquidity" in names:
        return "Trading value or liquidity score is too low."
    return "Rejected by lightweight ontology screening."


def _is_domestic_snapshot_market(ticker: str, market: str) -> bool:
    market_name = str(market).upper()
    return market_name in {"KRX", "KOSPI", "KOSDAQ", "KONEX"} or str(ticker).endswith(".KS") or str(ticker).isdigit()


def load_short_horizon_strategy_config(
    path: Path | str = SHORT_HORIZON_STRATEGY_CONFIG_PATH,
) -> dict[str, dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Short-horizon strategy config must be a mapping: {config_path}")
    return {
        str(section): dict(values or {})
        for section, values in loaded.items()
        if isinstance(values, dict) or values is None
    }


def build_strategy_candidate_factory_from_config(
    config: dict[str, dict[str, Any]] | None = None,
) -> StrategyCandidateFactory:
    loaded = config or load_short_horizon_strategy_config()
    factory_cfg = loaded.get("strategy_candidate_factory", {})
    reversal_cfg = loaded.get("short_term_reversal", {})
    intraday_cfg = loaded.get("intraday_momentum", {})
    technical_cfg = loaded.get("technical_rule", loaded.get("technical_breakout", {}))
    pair_cfg = loaded.get("pair_relative_value", {})
    return StrategyCandidateFactory(
        StrategyFactoryConfig(
            enabled=bool(factory_cfg.get("enabled", True)),
            paper_only=bool(factory_cfg.get("paper_only", True)),
            enable_short_term_reversal=bool(reversal_cfg.get("enabled", True)),
            enable_intraday_momentum=bool(intraday_cfg.get("enabled", True)),
            enable_technical_rule=bool(technical_cfg.get("enabled", True)),
            enable_pair_relative_value=bool(pair_cfg.get("enabled", True)),
            target_net_return=float(factory_cfg.get("target_net_return", 0.003)),
            max_cost_to_alpha_ratio=float(factory_cfg.get("max_cost_to_alpha_ratio", 0.5)),
            max_spread_rate=float(factory_cfg.get("max_spread_rate", 0.0015)),
            min_liquidity_score=float(factory_cfg.get("min_liquidity_score", 0.5)),
            venue=str(factory_cfg.get("venue", "KRX")),
            market=str(factory_cfg.get("market", "KR")),
            instrument_type=str(factory_cfg.get("instrument_type", "domestic_stock")),
        ),
        short_term_reversal=ShortTermReversalEngine(
            ShortTermReversalConfig(
                enabled=bool(reversal_cfg.get("enabled", True)),
                paper_only=bool(reversal_cfg.get("paper_only", True)),
                rebound_ratio=float(reversal_cfg.get("rebound_ratio", 0.35)),
                max_rebound_cap=float(reversal_cfg.get("max_rebound_cap", 0.006)),
                min_shock_vol_multiple=float(reversal_cfg.get("min_shock_vol_multiple", 1.2)),
                target_net_return=float(reversal_cfg.get("target_net_return", 0.003)),
                max_spread_rate=float(reversal_cfg.get("max_spread_rate", 0.0015)),
                min_liquidity_score=float(reversal_cfg.get("min_liquidity_score", 0.5)),
                expected_holding_minutes=int(reversal_cfg.get("expected_holding_minutes", 30)),
            )
        ),
        intraday_momentum=IntradayMomentumEngine(
            IntradayMomentumConfig(
                enabled=bool(intraday_cfg.get("enabled", True)),
                paper_only=bool(intraday_cfg.get("paper_only", True)),
                opening_window_minutes=int(intraday_cfg.get("opening_window_minutes", 30)),
                beta_r_open_to_late=float(intraday_cfg.get("beta_r_open_to_late", 0.25)),
                target_net_return=float(intraday_cfg.get("target_net_return", 0.003)),
                min_open_return=float(intraday_cfg.get("min_open_return", 0.002)),
                min_volume_zscore=float(intraday_cfg.get("min_volume_zscore", 0.5)),
                min_market_alignment_score=float(intraday_cfg.get("min_market_alignment_score", 0.5)),
                requires_market_alignment=bool(intraday_cfg.get("requires_market_alignment", True)),
                expected_holding_minutes=int(intraday_cfg.get("expected_holding_minutes", 180)),
            )
        ),
        technical_rule=TechnicalRuleEngine(
            TechnicalRuleConfig(
                enabled=bool(technical_cfg.get("enabled", True)),
                paper_only=bool(technical_cfg.get("paper_only", True)),
                ma_fast=int(technical_cfg.get("ma_fast", 5)),
                ma_slow=int(technical_cfg.get("ma_slow", 20)),
                range_window=int(technical_cfg.get("range_window", 20)),
                breakout_buffer=float(technical_cfg.get("breakout_buffer", 0.001)),
                volume_multiplier=float(technical_cfg.get("volume_multiplier", 1.5)),
                breakout_capture_ratio=float(technical_cfg.get("breakout_capture_ratio", 0.4)),
                volatility_target=float(technical_cfg.get("volatility_target", 0.006)),
                target_net_return=float(technical_cfg.get("target_net_return", 0.003)),
                max_spread_rate=float(technical_cfg.get("max_spread_rate", 0.0015)),
                min_liquidity_score=float(technical_cfg.get("min_liquidity_score", 0.5)),
                expected_holding_minutes=int(technical_cfg.get("expected_holding_minutes", 60)),
            )
        ),
        pair_relative_value=PairRelativeValueEngine(
            PairRelativeValueConfig(
                enabled=bool(pair_cfg.get("enabled", True)),
                paper_only=bool(pair_cfg.get("paper_only", True)),
                formation_window_days=int(pair_cfg.get("formation_window_days", 60)),
                trading_window_days=int(pair_cfg.get("trading_window_days", 20)),
                max_pair_distance=float(pair_cfg.get("max_pair_distance", 0.15)),
                spread_z_entry=float(pair_cfg.get("spread_z_entry", -2.0)),
                convergence_ratio=float(pair_cfg.get("convergence_ratio", 0.4)),
                target_net_return=float(pair_cfg.get("target_net_return", 0.004)),
                min_liquidity_score=float(pair_cfg.get("min_liquidity_score", 0.5)),
                max_spread_rate=float(pair_cfg.get("max_spread_rate", 0.0015)),
                expected_holding_minutes=int(pair_cfg.get("expected_holding_minutes", 7200)),
            )
        ),
        pair_universe_builder=PairUniverseBuilder(
            PairRelativeValueConfig(
                enabled=bool(pair_cfg.get("enabled", True)),
                formation_window_days=int(pair_cfg.get("formation_window_days", 60)),
                trading_window_days=int(pair_cfg.get("trading_window_days", 20)),
                max_pair_distance=float(pair_cfg.get("max_pair_distance", 0.15)),
            )
        ),
    )


def generate_short_horizon_strategy_candidates(
    *,
    features_by_ticker: dict[str, ShortHorizonFeatures],
    price_history_by_ticker: dict[str, tuple[OHLCVBar, ...]] | None = None,
    entry_prices: dict[str, float] | None = None,
    mode: str | None = None,
    config: dict[str, dict[str, Any]] | None = None,
) -> StrategyCandidateFactoryResult:
    loaded = config or load_short_horizon_strategy_config()
    execution = loaded.get("execution", {})
    requested_mode = mode or str(execution.get("default_mode", "paper_trading"))
    live_enabled = bool(execution.get("live_trading_enabled", False))
    if requested_mode == "live_trading" and not live_enabled:
        logger.warning("short_horizon_factory_live_blocked defaulting to no candidates")
        return StrategyCandidateFactoryResult(candidates=(), filtered_candidates=())
    trading_mode = "paper" if requested_mode in {"paper", "paper_trading", "dry_run"} else requested_mode
    factory = build_strategy_candidate_factory_from_config(loaded)
    return factory.build(
        StrategyCandidateFactoryInput(
            features_by_ticker=features_by_ticker,
            price_history_by_ticker=price_history_by_ticker or {},
            entry_prices=entry_prices or {},
        ),
        trading_mode=trading_mode,
    )


def _rank_accepted_with_npu(
    snapshots: tuple[LightweightMarketSnapshot, ...],
    accepted_tickers: tuple[str, ...],
    *,
    top_k: int,
) -> tuple[tuple[str, ...], dict[str, float | int | str]]:
    snapshot_by_ticker = {snapshot.ticker: snapshot for snapshot in snapshots}
    rows = []
    tickers = []
    for ticker in accepted_tickers:
        snapshot = snapshot_by_ticker[ticker]
        tickers.append(ticker)
        rows.append(
            (
                max(0.0, snapshot.price_change_rate),
                max(0.0, snapshot.volume_change_rate),
                max(0.0, snapshot.market_cap / 10_000_000_000_000),
                0.0 if snapshot.trading_value > 0 else 1.0,
                max(0.0, min(1.0, abs(snapshot.price_change_rate) * 2)),
                max(0.0, snapshot.price_change_rate + snapshot.volume_change_rate * 0.05),
                0.50 + max(-0.30, min(0.30, snapshot.price_change_rate)),
                max(0.0, snapshot.liquidity_score),
            )
        )
    scorer = OntologyNpuLinearScorer(batch_size=os.getenv("ONTOLOGY_NPU_BATCH_SIZE", "auto"))
    scored = scorer.score_candidates(tickers, rows, top_k=top_k)
    metrics = dict(scored.profile)
    metrics["candidate_count_after_npu_topk"] = len(scored.tickers)
    metrics["npu_enabled"] = 1
    return scored.tickers, metrics


def _npu_candidate_scoring_enabled() -> bool:
    return os.getenv("ONTOLOGY_NPU_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _max_top_k() -> int:
    try:
        return max(1, int(os.getenv("ONTOLOGY_NPU_TOP_K", "50")))
    except ValueError:
        return 50


def _score_lightweight_snapshot(
    snapshot: LightweightMarketSnapshot,
    *,
    min_trading_value: float,
    min_liquidity_score: float,
) -> tuple[float, str, list[str], str]:
    fired: list[str] = []
    if snapshot.halt_status:
        return 0.0, "RejectStock", ["halt_status"], "Trading is halted."
    if snapshot.management_stock_status:
        return 0.0, "RejectStock", ["management_stock_status"], "Management stock status blocks trading."
    if snapshot.trading_value < min_trading_value or snapshot.liquidity_score < min_liquidity_score:
        return 0.0, "RejectStock", ["insufficient_liquidity"], "Trading value or liquidity score is too low."

    score = 0.0
    score += min(1.0, snapshot.liquidity_score) * 0.35
    score += max(0.0, snapshot.volume_change_rate) * 0.18
    score += max(0.0, snapshot.price_change_rate) * 3.0
    flow_assessment = assess_domestic_investor_flow(
        MarketSnapshot(
            ticker=snapshot.ticker,
            market=snapshot.market,
            company_name=snapshot.ticker,
            sector=snapshot.sector,
            last_price=snapshot.current_price,
            average_daily_trading_value=snapshot.trading_value,
            volatility_20d=0.0,
            source=_pipeline_source(),
            investor_flow=InvestorFlowSnapshot(
                ticker=snapshot.ticker,
                market=snapshot.market,
                foreign_net_buy=snapshot.foreign_net_buy,
                institution_net_buy=snapshot.institution_net_buy,
                retail_net_buy=snapshot.retail_net_buy,
                program_net_buy=snapshot.program_net_buy,
                short_net_change=snapshot.short_net_change,
                volume_change_rate=snapshot.volume_change_rate,
                price_change_rate=snapshot.price_change_rate,
                trading_value=snapshot.trading_value,
            ),
        )
    )
    score += flow_assessment.score_adjustment * 0.25
    fired.extend(factor.lower() for factor in flow_assessment.supporting_factors[:3])
    fired.extend(factor.lower() for factor in flow_assessment.contradicting_factors[:3])
    if snapshot.trading_value >= min_trading_value and snapshot.volume_change_rate > 0.25:
        fired.append("high_trading_value_and_volume_surge")
        score += 0.18
    if snapshot.foreign_net_buy > 0 and snapshot.institution_net_buy > 0 and snapshot.price_change_rate > 0:
        fired.append("foreign_institution_buying_with_momentum")
        score += 0.2
    if snapshot.upper_limit_near or snapshot.new_52week_high:
        fired.append("breakout_priority")
        score += 0.12
    if not fired:
        fired.append("baseline_liquidity_candidate")
    return score, "CandidateStock", fired, "Passed lightweight ontology liquidity and momentum screening."


def _sector_for(ticker: str) -> str:
    sectors = ("Technology", "Financials", "Healthcare", "Industrials", "Consumer", "Energy", "Materials")
    return sectors[int(_stable_unit(f"{ticker}:sector") * len(sectors)) % len(sectors)]


def _flow_value(
    flow: InvestorFlowSnapshot | None,
    field_name: str,
    ticker: str,
    salt: str,
    trading_value: float,
    scale: float,
) -> float:
    if flow is not None:
        return round(float(getattr(flow, field_name)), 2)
    return round((_stable_unit(f"{ticker}:{salt}") - 0.5) * trading_value * scale, 2)


def _pipeline_source():
    from app.schemas.domain import SourceMetadata

    return SourceMetadata("pipeline_lightweight_flow", datetime.now(timezone.utc), source_id="pipeline-flow")


def _stable_unit(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)
