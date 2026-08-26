"""Leak-aware validation for the long-only bear-market cash-equity submode.

This evaluator deliberately calls the production ``residual_relative_strength``
entry/exit rules.  It reconstructs only facts available at each stored minute,
enters on the next bar, charges the venue cost plus observed spread, and excludes
ETFs and non-common instruments before forming the cross section.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable

from app.cost.round_trip import all_in_round_trip_bps
from app.data.instrument_catalog import us_instrument
from app.data.instrument_eligibility import CATEGORY_EQUITY, classify
from app.evaluation.stored_counterfactual import load_minute_bars, load_minute_microstructure
from app.technical.signals import TechnicalFeatureSet
from app.technical.strategy_algorithms import (
    AlgorithmConfig,
    ElectionContext,
    build_algorithm_registry,
    round_trip_cost_bps,
)
from app.trading.contracts import Bar


@dataclass(frozen=True)
class BearValidationConfig:
    database: Path = Path("data/store/realtime_market_data.sqlite3")
    investor_flow_database: Path = Path("data/store/investor_flow.sqlite3")
    stride_bars: int = 5
    history_bars: int = 30
    maximum_gap_seconds: float = 120.0
    minimum_cross_section: int = 5
    minimum_samples: int = 30
    minimum_test_samples: int = 10
    minimum_positive_day_fraction: float = 0.60
    cost_stress_multiple: float = 1.25
    pessimism_z: float = 1.64


@dataclass(frozen=True)
class _Snapshot:
    symbol: str
    market: str
    moment: datetime
    index: int
    one_return: float
    short_return: float
    long_return: float
    price: float
    ema_fast: float
    ema_slow: float
    vwap: float
    vwap_distance_bps: float
    momentum_persistence: float
    relative_volume: float
    realized_volatility: float
    spread_bps: float | None
    orderbook_imbalance: float | None
    liquidity_score: float | None


@dataclass(frozen=True)
class BearValidationTrade:
    symbol: str
    market: str
    signal_at: datetime
    exit_at: datetime
    gross_bps: float
    net_bps: float
    cost_bps: float
    stressed_net_bps: float
    exit_reason: str
    holding_seconds: float


def _known_kr_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for path in (Path("config/instrument_names.json"), Path("data/runtime/domestic_universe.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        source = payload.get("names") if isinstance(payload, dict) and "names" in payload else payload
        if isinstance(source, dict):
            names.update(
                {
                    str(key).split(".", 1)[0].upper(): str(value)
                    for key, value in source.items()
                    if str(value or "").strip() and not str(key).startswith("_")
                }
            )
    return names


def _eligible_cash_equity(symbol: str, kr_names: dict[str, str]) -> bool:
    normalized = str(symbol or "").upper()
    market = "KR" if normalized.isdigit() and len(normalized) == 6 else "US"
    if market == "KR":
        name = kr_names.get(normalized)
        # Historical KRX bars without a name cannot be proven not to be an ETF.
        return bool(name and classify(normalized, name, market="KR").category == CATEGORY_EQUITY)
    record = us_instrument(normalized)
    if record is None or record.is_etf:
        return False
    descriptor = record.security_name.upper()
    if any(token in descriptor for token in (" WARRANT", " UNIT", " RIGHT")):
        return False
    return classify(normalized, record.security_name, market="US").category == CATEGORY_EQUITY


def _contiguous(left: Bar, right: Bar, maximum_gap_seconds: float) -> bool:
    gap = (right.start_time - left.end_time).total_seconds()
    return 0.0 <= gap <= maximum_gap_seconds


def _ema(values: list[float], span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _snapshots(
    bars_by_symbol: dict[str, tuple[Bar, ...]],
    micro_by_symbol: dict[str, dict[datetime, Any]],
    cfg: BearValidationConfig,
    kr_names: dict[str, str],
) -> tuple[dict[tuple[str, datetime], _Snapshot], dict[str, tuple[Bar, ...]], dict[str, int]]:
    rows: dict[tuple[str, datetime], _Snapshot] = {}
    eligible_bars: dict[str, tuple[Bar, ...]] = {}
    exclusions = {"etf_or_non_common": 0, "unclassified_kr": 0, "insufficient_history": 0}
    for symbol, bars in bars_by_symbol.items():
        if not _eligible_cash_equity(symbol, kr_names):
            if str(symbol).isdigit() and str(symbol) not in kr_names:
                exclusions["unclassified_kr"] += 1
            else:
                exclusions["etf_or_non_common"] += 1
            continue
        if len(bars) < cfg.history_bars + 2:
            exclusions["insufficient_history"] += 1
            continue
        eligible_bars[symbol] = bars
        market = "KR" if str(symbol).isdigit() and len(str(symbol)) == 6 else "US"
        day_start = 0
        for index, current in enumerate(bars):
            if index == 0 or current.start_time.date() != bars[index - 1].start_time.date():
                day_start = index
            if index - day_start < cfg.history_bars or index + 1 >= len(bars):
                continue
            history = list(bars[index - cfg.history_bars : index + 1])
            if any(
                not _contiguous(left, right, cfg.maximum_gap_seconds)
                for left, right in zip(history, history[1:])
            ):
                continue
            if (index - day_start) % max(1, cfg.stride_bars):
                continue
            closes = [float(item.close) for item in history]
            if min(closes) <= 0.0:
                continue
            one_returns = [later / earlier - 1.0 for earlier, later in zip(closes, closes[1:])]
            volumes = [max(0.0, float(item.volume)) for item in history]
            prior_volume = fmean(volumes[:-1]) if any(volumes[:-1]) else 0.0
            session = bars[day_start : index + 1]
            total_volume = sum(max(0.0, float(item.volume)) for item in session)
            vwap = (
                sum(float(item.close) * max(0.0, float(item.volume)) for item in session) / total_volume
                if total_volume > 0.0
                else 0.0
            )
            if vwap <= 0.0:
                continue
            micro = micro_by_symbol.get(symbol, {}).get(current.start_time)
            rows[(symbol, current.start_time)] = _Snapshot(
                symbol=symbol,
                market=market,
                moment=current.start_time,
                index=index,
                one_return=one_returns[-1],
                short_return=closes[-1] / closes[-6] - 1.0,
                long_return=closes[-1] / closes[0] - 1.0,
                price=closes[-1],
                ema_fast=_ema(closes, 12),
                ema_slow=_ema(closes, 26),
                vwap=vwap,
                vwap_distance_bps=(closes[-1] / vwap - 1.0) * 10_000.0,
                momentum_persistence=sum(value > 0.0 for value in one_returns[-10:]) / 10.0,
                relative_volume=(volumes[-1] / prior_volume if prior_volume > 0.0 else 0.0),
                realized_volatility=(stdev(one_returns) if len(one_returns) >= 2 else 0.0),
                spread_bps=getattr(micro, "spread_bps", None),
                orderbook_imbalance=getattr(micro, "orderbook_imbalance", None),
                liquidity_score=getattr(micro, "liquidity_score", None),
            )
    return rows, eligible_bars, exclusions


def _flow_scores(path: Path) -> dict[tuple[str, date], tuple[float | None, float | None]]:
    if not path.exists():
        return {}
    series: dict[str, list[tuple[date, float, float]]] = defaultdict(list)
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        for symbol, business_date, foreign, institution in connection.execute(
            "select symbol,business_date,foreign_net_buy_value,institution_net_buy_value "
            "from investor_flow_daily order by symbol,business_date"
        ):
            series[str(symbol)].append(
                (date.fromisoformat(str(business_date)[:10]), float(foreign or 0.0), float(institution or 0.0))
            )
        connection.close()
    except (sqlite3.Error, ValueError):
        return {}

    def zscore(value: float, history: list[float]) -> float | None:
        if len(history) < 5:
            return None
        sigma = stdev(history)
        return (value - fmean(history)) / sigma if sigma > 0.0 else 0.0

    output: dict[tuple[str, date], tuple[float | None, float | None]] = {}
    for symbol, values in series.items():
        for index, (day, foreign, institution) in enumerate(values):
            prior = values[max(0, index - 20) : index]
            output[(symbol, day)] = (
                zscore(foreign, [item[1] for item in prior]),
                zscore(institution, [item[2] for item in prior]),
            )
    return output


def _covariance_beta(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 10:
        return None
    stock = [item[0] for item in pairs]
    market = [item[1] for item in pairs]
    market_mean = fmean(market)
    variance = sum((value - market_mean) ** 2 for value in market) / (len(market) - 1)
    if variance <= 0.0:
        return None
    stock_mean = fmean(stock)
    covariance = sum((x - stock_mean) * (y - market_mean) for x, y in pairs) / (len(pairs) - 1)
    return covariance / variance


def _simulate(
    snapshot: _Snapshot,
    bars: tuple[Bar, ...],
    decision: Any,
    rule: Any,
    cfg: BearValidationConfig,
) -> BearValidationTrade | None:
    entry_index = snapshot.index + 1
    if entry_index >= len(bars) or not _contiguous(bars[snapshot.index], bars[entry_index], cfg.maximum_gap_seconds):
        return None
    entry = float(bars[entry_index].open)
    target = float(rule.target_price or 0.0)
    stop = float(rule.stop_price or 0.0)
    if entry <= 0.0 or target <= entry or stop <= 0.0 or stop >= entry:
        return None
    fallback = float(round_trip_cost_bps(snapshot.symbol) or 0.0)
    cost = all_in_round_trip_bps(
        snapshot.symbol,
        spread_bps=snapshot.spread_bps,
        fallback_bps=fallback,
    )
    target_bps = (target / entry - 1.0) * 10_000.0
    if target_bps <= cost + 25.0:
        return None
    maximum = max(1, math.ceil(float(decision.horizon_seconds) / 60.0))
    future = bars[entry_index : entry_index + maximum]
    contiguous: list[Bar] = []
    previous: Bar | None = None
    for bar in future:
        if previous is not None and not _contiguous(previous, bar, cfg.maximum_gap_seconds):
            break
        contiguous.append(bar)
        previous = bar
    if not contiguous:
        return None
    watermark = entry
    exit_price = float(contiguous[-1].close)
    exit_reason = "TIME_EXIT"
    exited = contiguous[-1]
    trailing_bps = max(0.0, float(rule.trailing_bps or 0.0))
    for bar in contiguous:
        # Pessimistic ordering when a minute touches both barriers.
        if float(bar.low) <= stop:
            exit_price, exit_reason, exited = stop, "STOP", bar
            break
        if float(bar.high) >= target:
            exit_price, exit_reason, exited = target, "TARGET", bar
            break
        watermark = max(watermark, float(bar.high))
        trailing = watermark * (1.0 - trailing_bps / 10_000.0)
        if watermark > entry and trailing_bps > 0.0 and float(bar.low) <= trailing:
            exit_price, exit_reason, exited = trailing, "TRAILING", bar
            break
    gross = (exit_price / entry - 1.0) * 10_000.0
    net = gross - cost
    return BearValidationTrade(
        symbol=snapshot.symbol,
        market=snapshot.market,
        signal_at=snapshot.moment,
        exit_at=exited.end_time,
        gross_bps=gross,
        net_bps=net,
        cost_bps=cost,
        stressed_net_bps=gross - cfg.cost_stress_multiple * cost,
        exit_reason=exit_reason,
        holding_seconds=max(0.0, (exited.end_time - bars[entry_index].start_time).total_seconds()),
    )


def _metrics(trades: Iterable[BearValidationTrade], cfg: BearValidationConfig) -> dict[str, Any]:
    rows = tuple(trades)
    values = [item.net_bps for item in rows]
    mean = fmean(values) if values else float("-inf")
    lower = (
        mean - cfg.pessimism_z * stdev(values) / math.sqrt(len(values))
        if len(values) >= 2
        else float("-inf")
    )
    by_day: dict[date, list[float]] = defaultdict(list)
    for item in rows:
        by_day[item.signal_at.date()].append(item.net_bps)
    positive_days = (
        sum(fmean(day_values) > 0.0 for day_values in by_day.values()) / len(by_day)
        if by_day
        else 0.0
    )
    return {
        "samples": len(rows),
        "distinct_days": len(by_day),
        "mean_net_bps": mean if values else None,
        "lower_confidence_bound_bps": lower if values else None,
        "positive_trade_rate": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
        "positive_day_fraction": positive_days,
        "cost_stressed_mean_net_bps": fmean(item.stressed_net_bps for item in rows) if rows else None,
        "mean_cost_bps": fmean(item.cost_bps for item in rows) if rows else None,
        "exit_reasons": {
            reason: sum(item.exit_reason == reason for item in rows)
            for reason in sorted({item.exit_reason for item in rows})
        },
    }


def build_bear_cash_equity_report(cfg: BearValidationConfig | None = None) -> dict[str, Any]:
    config = cfg or BearValidationConfig()
    bars_by_symbol = load_minute_bars(config.database)
    micro = load_minute_microstructure(config.database)
    snapshots, eligible_bars, exclusions = _snapshots(
        bars_by_symbol, micro, config, _known_kr_names()
    )
    by_market_time: dict[tuple[str, datetime], list[_Snapshot]] = defaultdict(list)
    by_symbol: dict[str, list[_Snapshot]] = defaultdict(list)
    for row in snapshots.values():
        by_market_time[(row.market, row.moment)].append(row)
        by_symbol[row.symbol].append(row)
    market_returns: dict[tuple[str, datetime], tuple[float, float, float, float]] = {}
    for key, rows in by_market_time.items():
        if len(rows) < config.minimum_cross_section:
            continue
        market_returns[key] = (
            fmean(item.one_return for item in rows),
            fmean(item.short_return for item in rows),
            fmean(item.long_return for item in rows),
            sum(item.long_return > 0.0 for item in rows) / len(rows),
        )
    flow = _flow_scores(config.investor_flow_database)
    registries = {
        market: build_algorithm_registry(AlgorithmConfig(market=market))
        for market in ("KR", "US")
    }
    trades: list[BearValidationTrade] = []
    diagnostics = defaultdict(int)
    next_allowed: dict[str, datetime] = {}

    for (market, moment), cross_section in sorted(by_market_time.items(), key=lambda item: item[0][1]):
        market_row = market_returns.get((market, moment))
        if market_row is None:
            diagnostics["cross_section_insufficient"] += 1
            continue
        _m1, market_short, market_long, breadth = market_row
        if market_long >= 0.0 or breadth > 0.45:
            diagnostics["not_bear_regime"] += 1
            continue
        ranked: list[tuple[float, _Snapshot, float]] = []
        for row in cross_section:
            history_pairs: list[tuple[float, float]] = []
            for prior in by_symbol[row.symbol]:
                if prior.moment >= moment:
                    break
                market_prior = market_returns.get((market, prior.moment))
                if market_prior is not None:
                    history_pairs.append((prior.one_return, market_prior[0]))
            beta = _covariance_beta(history_pairs[-30:])
            if beta is None:
                diagnostics["beta_unavailable"] += 1
                continue
            residual_short = (row.short_return - beta * market_short) * 10_000.0
            ranked.append((residual_short, row, beta))
        ranked.sort(key=lambda item: (item[0], item[1].symbol), reverse=True)
        universe = len(ranked)
        for rank, (_residual_short, row, beta) in enumerate(ranked[:3], start=1):
            if row.symbol in next_allowed and moment < next_allowed[row.symbol]:
                continue
            residual_long = (row.long_return - beta * market_long) * 10_000.0
            foreign, institution = flow.get((row.symbol, moment.date()), (None, None))
            features = TechnicalFeatureSet(
                symbol=row.symbol,
                price=row.price,
                ema_fast=row.ema_fast,
                ema_slow=row.ema_slow,
                short_return=row.short_return,
                momentum_persistence=row.momentum_persistence,
                vwap=row.vwap,
                vwap_distance_bps=row.vwap_distance_bps,
                relative_volume=row.relative_volume,
                realized_volatility=row.realized_volatility,
                spread_bps=row.spread_bps,
                orderbook_imbalance=row.orderbook_imbalance,
                liquidity_score=row.liquidity_score,
            )
            context = ElectionContext(
                strategy_id="residual_relative_strength",
                elected_at=moment,
                reference_price=row.price,
                sector_rank=rank,
                sector_candidate_count=universe,
                residual_return_short_bps=_residual_short,
                residual_return_long_bps=residual_long,
                market_beta=beta,
                foreign_flow_zscore=foreign,
                institution_flow_zscore=institution,
                market_trend="TREND_DOWN",
                market_breadth=breadth,
            )
            algorithm = registries[market]["residual_relative_strength"]
            decision = algorithm.entry(features, context)
            if not decision.triggered:
                for reason in decision.reason_codes:
                    diagnostics[str(reason)] += 1
                continue
            rule = algorithm.exit_rule(row.price, features, context)
            trade = _simulate(row, eligible_bars[row.symbol], decision, rule, config)
            if trade is None:
                diagnostics["unexecutable_or_cost_unviable"] += 1
                continue
            trades.append(trade)
            next_allowed[row.symbol] = trade.exit_at

    dates = sorted({item.signal_at.date() for item in trades})
    split_day = dates[max(0, math.ceil(len(dates) * 0.60) - 1)] if dates else None
    train = [item for item in trades if split_day is not None and item.signal_at.date() <= split_day]
    test = [item for item in trades if split_day is not None and item.signal_at.date() > split_day]
    by_market: dict[str, Any] = {}
    for market in ("KR", "US"):
        rows = [item for item in trades if item.market == market]
        metrics = _metrics(rows, config)
        market_test = _metrics([item for item in test if item.market == market], config)
        qualified = bool(
            metrics["samples"] >= config.minimum_samples
            and (metrics["lower_confidence_bound_bps"] or float("-inf")) > 0.0
            and (metrics["cost_stressed_mean_net_bps"] or float("-inf")) > 0.0
            and metrics["positive_day_fraction"] >= config.minimum_positive_day_fraction
            and market_test["samples"] >= config.minimum_test_samples
            and (market_test["mean_net_bps"] or float("-inf")) > 0.0
        )
        by_market[market] = {"all": metrics, "test": market_test, "research_qualified": qualified}
    return {
        "strategy_id": "residual_relative_strength",
        "submode": "TREND_DOWN_LONG_ONLY_CASH_EQUITY",
        "generated_at": datetime.now().astimezone().isoformat(),
        "data": {
            "database": str(config.database),
            "bars_loaded": sum(len(items) for items in bars_by_symbol.values()),
            "symbols_loaded": len(bars_by_symbol),
            "eligible_cash_equity_symbols": len(eligible_bars),
            "first_event": min((bar.start_time for bars in bars_by_symbol.values() for bar in bars), default=None),
            "last_event": max((bar.end_time for bars in bars_by_symbol.values() for bar in bars), default=None),
            "exclusions": exclusions,
        },
        "configuration": {**asdict(config), "database": str(config.database), "investor_flow_database": str(config.investor_flow_database)},
        "split_day": split_day,
        "overall": _metrics(trades, config),
        "train": _metrics(train, config),
        "test": _metrics(test, config),
        "markets": by_market,
        "diagnostics": dict(sorted(diagnostics.items(), key=lambda item: (-item[1], item[0]))),
        "promotion_eligible": any(value["research_qualified"] for value in by_market.values()),
        "deployment_effect": "SHADOW_ONLY; forward regime-scoped promotion remains mandatory",
        "limitations": [
            "Minute bars cannot recover queue position or intraminute barrier order; same-bar ambiguity is resolved stop-first.",
            "KRX symbols without a resolvable listed name are excluded because they cannot be proven not to be ETFs.",
            "Historical sector membership is unavailable, so cross-sectional rank substitutes only for offline screening; live inference retains sector-neutral residuals.",
        ],
    }

