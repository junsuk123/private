from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.schemas.domain import MarketSnapshot

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
    upper_limit_near: bool
    new_52week_high: bool
    halt_status: bool
    management_stock_status: bool
    liquidity_score: float


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
                upper_limit_near=price_change > 0.12,
                new_52week_high=_stable_unit(f"{stock.ticker}:high52") > 0.93,
                halt_status=_stable_unit(f"{stock.ticker}:halt") > 0.997,
                management_stock_status=_stable_unit(f"{stock.ticker}:management") > 0.996,
                liquidity_score=round(liquidity_score, 5),
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
                foreign_net_buy=round((_stable_unit(f"{market.ticker}:analysis-foreign") - 0.5) * trading_value * 0.08, 2),
                institution_net_buy=round((_stable_unit(f"{market.ticker}:analysis-institution") - 0.5) * trading_value * 0.08, 2),
                retail_net_buy=round((_stable_unit(f"{market.ticker}:analysis-retail") - 0.5) * trading_value * 0.08, 2),
                upper_limit_near=momentum > 0.10,
                new_52week_high=_stable_unit(f"{market.ticker}:analysis-high52") > 0.94,
                halt_status=False,
                management_stock_status=False,
                liquidity_score=round(liquidity_score, 5),
            )
        )
    return tuple(snapshots)


def ontology_filter_1(
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
    target_count = max(1, min(100, int(target_count)))
    accepted: list[tuple[float, str]] = []
    rejected: list[str] = []
    traces: list[ReasoningTrace] = []

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
        else:
            rejected.append(snapshot.ticker)

    candidates = tuple(ticker for _score, ticker in sorted(accepted, reverse=True)[:target_count])
    result = CandidateSelectionResult(
        candidate_stocks=candidates,
        rejected_stocks=tuple(rejected),
        traces=tuple(traces),
        latency_ms=int((time.perf_counter() - started) * 1000),
        api_call_count=0,
        full_universe_count=len(snapshots),
        chart_fetch_scope=candidates,
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


def _stable_unit(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)
