from __future__ import annotations

import csv
import json
import math
import random
import os
import urllib.request
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.cost import TradingCostEngine
from app.goals import NegotiatedGoal
from app.graph import OntologyReasoner
from app.graph.builders import build_market_graph
from app.risk import RiskManager
from app.simulation import MarketCalendar, market_session_for
from app.schemas.domain import (
    AccountSnapshot,
    Holding,
    IndicatorSnapshot,
    MarketSnapshot,
    OrderAction,
    OrderSide,
    RiskRules,
    SourceMetadata,
)
from app.strategy import build_goal_execution_plan
from app.trading_pipeline import (
    build_lightweight_market_snapshots,
    ontology_filter_1,
    universe_from_tickers,
)


@dataclass(frozen=True)
class ChartBar:
    ticker: str
    timestamp: datetime
    close: float
    volume: int


@dataclass(frozen=True)
class SimulatedTrade:
    timestamp: datetime
    ticker: str
    side: str
    quantity: int
    price: float
    value: float
    reason: str
    currency: str = "KRW"
    fx_rate: float = 1.0
    value_krw: float | None = None
    trading_cost: float = 0.0
    net_value: float | None = None


@dataclass(frozen=True)
class AcceleratedDemoResult:
    initial_equity: float
    final_equity: float
    profit_amount: float
    return_rate: float
    target_return_rate: float
    target_profit_amount: float
    target_achieved: bool
    simulated_minutes: int
    bars_per_ticker: int
    ticker_count: int
    accelerated_steps: int
    trade_count: int
    final_cash: float
    final_positions: dict[str, float]
    sample_trades: tuple[SimulatedTrade, ...]
    report_path: str
    trades_path: str
    charts_path: str


DEMO_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "AMD",
    "INTC",
    "QCOM",
    "MU",
    "ASML",
    "TSM",
    "ARM",
    "ORCL",
    "CRM",
    "ADBE",
    "NOW",
    "PANW",
    "SNOW",
    "SHOP",
    "UBER",
    "NFLX",
    "DIS",
    "V",
    "MA",
    "JPM",
    "BAC",
    "GS",
    "XOM",
    "CVX",
    "COP",
    "CAT",
    "GE",
    "DE",
    "LLY",
    "UNH",
    "JNJ",
    "MRK",
    "COST",
    "WMT",
    "HD",
    "MCD",
    "NKE",
    "SPY",
    "QQQ",
    "IBM",
    "TXN",
    "PEP",
)


def _simulated_trade_cost(
    engine: TradingCostEngine,
    side: OrderSide,
    ticker: str,
    price: float,
    quantity: int,
    currency: str = "KRW",
) -> float:
    if currency != "KRW" or price <= 0 or quantity <= 0:
        return 0.0
    instrument_type = "domestic_stock" if ticker.replace(".", "").isdigit() else "domestic_stock"
    policy = engine.policy_for(instrument_type=instrument_type, venue="KRX")
    value = price * quantity
    variable_cost = value * (policy.slippage_rate + policy.spread_rate + policy.market_impact_rate)
    if side == OrderSide.BUY:
        return value * policy.buy_fee_rate + variable_cost
    if side == OrderSide.SELL:
        return value * (policy.sell_fee_rate + policy.sell_tax_rate) + variable_cost
    return 0.0

NASDAQ_TRADER_LISTING_URLS = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
)
SIM_UNIVERSE_CACHE = Path("data/universe/us_listed_symbols.csv")
KRX_KIND_LISTED_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
KRX_UNIVERSE_CACHE = Path("data/universe/krx_listed_symbols.csv")


def load_us_listed_universe(
    cache_path: Path = SIM_UNIVERSE_CACHE,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Load US-listed symbols for simulation.

    The cache is used first. If it is missing, NASDAQ Trader symbol directories are
    downloaded and cached. If the network is unavailable, the demo universe remains
    the deterministic fallback so simulation still works offline.
    """
    configured_limit = os.getenv("SIM_UNIVERSE_LIMIT", "").strip()
    if limit is None and configured_limit:
        try:
            limit = max(0, int(configured_limit))
        except ValueError:
            limit = None

    symbols = _read_symbol_cache(cache_path)
    if not symbols:
        symbols = _download_us_listed_symbols()
        if symbols:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=("symbol",))
                writer.writeheader()
                for symbol in symbols:
                    writer.writerow({"symbol": symbol})
    if not symbols:
        symbols = list(DEMO_TICKERS)
    if limit and limit > 0:
        symbols = symbols[:limit]
    return tuple(symbols)


def load_krx_listed_universe(
    cache_path: Path = KRX_UNIVERSE_CACHE,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Load KRX/KOSDAQ/KONEX-listed symbols for simulation."""
    configured_limit = os.getenv("SIM_KRX_UNIVERSE_LIMIT", "").strip()
    if limit is None and configured_limit:
        try:
            limit = max(0, int(configured_limit))
        except ValueError:
            limit = None

    symbols = _read_symbol_cache(cache_path)
    if symbols and not any(symbol.endswith(".KS") for symbol in symbols):
        refreshed = _download_krx_listed_symbols()
        if refreshed:
            symbols = refreshed
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=("symbol",))
                writer.writeheader()
                for symbol in symbols:
                    writer.writerow({"symbol": symbol})
    if not symbols:
        symbols = _download_krx_listed_symbols()
        if symbols:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=("symbol",))
                writer.writeheader()
                for symbol in symbols:
                    writer.writerow({"symbol": symbol})
    if limit and limit > 0:
        symbols = symbols[:limit]
    return tuple(symbols)


def load_global_listed_universe() -> tuple[str, ...]:
    """Load the configured overseas/US and domestic listed simulation universe."""
    markets = {
        item.strip().upper()
        for item in os.getenv("SIM_MARKETS", "US,KR").split(",")
        if item.strip()
    }
    groups: list[tuple[str, ...]] = []
    if "US" in markets or "OVERSEAS" in markets or "GLOBAL" in markets:
        groups.append(load_us_listed_universe())
    if "KR" in markets or "KOREA" in markets or "DOMESTIC" in markets:
        groups.append(load_krx_listed_universe())
    symbols = _interleave_symbol_groups(groups)
    if not symbols:
        symbols.extend(DEMO_TICKERS)
    return tuple(_unique_symbols(symbols))


def _interleave_symbol_groups(groups: list[tuple[str, ...]]) -> list[str]:
    if not groups:
        return []
    max_len = max((len(group) for group in groups), default=0)
    interleaved: list[str] = []
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                interleaved.append(group[index])
    return interleaved


def _read_symbol_cache(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        symbols = [_normalize_symbol(row.get("symbol", "")) for row in reader]
    return _unique_symbols(symbol for symbol in symbols if symbol)


def _download_us_listed_symbols() -> list[str]:
    symbols: list[str] = []
    for url in NASDAQ_TRADER_LISTING_URLS:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        symbols.extend(_parse_nasdaq_trader_symbols(text))
    return _unique_symbols(symbols)


def _download_krx_listed_symbols() -> list[str]:
    try:
        with urllib.request.urlopen(KRX_KIND_LISTED_URL, timeout=15) as response:
            text = response.read().decode("euc-kr", errors="replace")
    except Exception:
        return []
    rows = _KrxKindTableParser.parse(text)
    symbols: list[str] = []
    for row in rows:
        if len(row) < 3:
            continue
        market = row[1].strip()
        code = row[2].strip().upper()
        if not code:
            continue
        suffix = ".KS" if "유가" in market or "코스피" in market else ".KQ"
        symbols.append(f"{code}{suffix}")
    return _unique_symbols(symbols)


class _KrxKindTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_cell = False
        self._current_cell: list[str] = []
        self._current_row: list[str] = []
        self.rows: list[list[str]] = []

    @classmethod
    def parse(cls, text: str) -> list[list[str]]:
        parser = cls()
        parser.feed(text)
        return parser.rows

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"td", "th"}:
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"td", "th"} and self._in_cell:
            value = " ".join(part.strip() for part in self._current_cell if part.strip())
            self._current_row.append(value)
            self._in_cell = False
        elif lowered == "tr":
            if self._current_row and self._current_row[0] != "회사명":
                self.rows.append(self._current_row)
            self._current_row = []


def _parse_nasdaq_trader_symbols(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("|")
    rows = []
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        values = line.split("|")
        row = {headers[index]: values[index] for index in range(min(len(headers), len(values)))}
        if row.get("Test Issue", "N") == "Y":
            continue
        if row.get("Financial Status") == "D":
            continue
        symbol = _normalize_symbol(row.get("Symbol") or row.get("ACT Symbol") or "")
        if symbol:
            rows.append(symbol)
    return rows


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return ""
    symbol = symbol.replace("/", ".")
    if any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for char in symbol):
        return ""
    return symbol


def _calendar_for_ticker(ticker: str) -> MarketCalendar:
    normalized = ticker.upper()
    if normalized.endswith((".KS", ".KQ")) or normalized.isdigit():
        return MarketCalendar.krx()
    return MarketCalendar.us()


def _session_timestamps(
    calendar: MarketCalendar,
    total_bars: int,
    interval_minutes: int,
) -> tuple[datetime, ...]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(calendar.timezone_name)
    day = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps: list[datetime] = []
    while len(timestamps) < total_bars:
        if _demo_session_is_trading_day(day, calendar):
            for session in calendar.sessions:
                start = datetime.combine(day.date(), session.start_time, tzinfo=tz)
                end = datetime.combine(day.date(), session.end_time, tzinfo=tz)
                if session.end_time <= session.start_time:
                    end += timedelta(days=1)
                current = start
                while current < end and len(timestamps) < total_bars:
                    timestamps.append(current.astimezone(timezone.utc))
                    current += timedelta(minutes=interval_minutes)
        day += timedelta(days=1)
    return tuple(timestamps)


def _demo_session_is_trading_day(local_midnight: datetime, calendar: MarketCalendar) -> bool:
    if calendar.name.upper() == "US" and local_midnight.weekday() == 6:
        return any(session.start_time > session.end_time for session in calendar.sessions)
    return local_midnight.weekday() < 5


def _session_name_for_timestamp(calendar: MarketCalendar, timestamp: datetime) -> str:
    return market_session_for(timestamp, calendar) or "regular"


def _demo_session_volume_factor(session_name: str) -> float:
    return {
        "day_market": 0.35,
        "premarket": 0.55,
        "regular": 1.0,
        "aftermarket": 0.45,
    }.get(session_name, 1.0)


def _demo_session_volatility_factor(session_name: str) -> float:
    return {
        "day_market": 1.25,
        "premarket": 1.20,
        "regular": 1.0,
        "aftermarket": 1.15,
    }.get(session_name, 1.0)


def _unique_symbols(symbols: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for symbol in symbols:
        normalized = _normalize_symbol(str(symbol))
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def run_accelerated_demo(
    target_return_rate: float = 0.02,
    period_minutes: int = 390,
    initial_cash: float = 10_000_000,
    output_dir: Path = Path("data/reports"),
    seed: int = 42,
) -> AcceleratedDemoResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_tickers = load_global_listed_universe()
    universe = universe_from_tickers(universe_tickers)
    snapshots = build_lightweight_market_snapshots(universe, seed=seed)
    candidate_selection = ontology_filter_1(
        snapshots,
        target_count=50,
        cache_key=f"accelerated:{seed}:{len(universe_tickers)}:50",
    )
    tickers = candidate_selection.candidate_stocks or tuple(universe_tickers[:50])
    bars_by_ticker = generate_synthetic_charts(tickers, period_minutes, seed)
    timestamps = tuple(bar.timestamp for bar in bars_by_ticker[tickers[0]])
    cash = initial_cash
    holdings: dict[str, int] = {}
    trades: list[SimulatedTrade] = []
    cost_engine = TradingCostEngine()
    period_days = max(1, math.ceil(period_minutes / 390))
    warmup_steps = min(15, max(0, len(timestamps) - 1))
    goal = NegotiatedGoal(
        target_return_rate=target_return_rate,
        target_profit_amount=initial_cash * target_return_rate,
        period_days=period_days,
        feasibility_percent=68,
        label="Accelerated demo target",
    )
    rules = RiskRules(
        max_single_stock_weight=0.06,
        max_sector_weight=0.55,
        minimum_cash_reserve=0.08,
        max_trades_per_day=80,
        min_average_daily_trading_value=max(1_000.0, initial_cash * 0.02),
        max_volatility=0.12,
    )

    for step, timestamp in enumerate(timestamps):
        if step < warmup_steps:
            continue
        prices = _prices_at(bars_by_ticker, step)
        account = _account_from_state(cash, holdings, prices, timestamp)
        markets = _markets_at(bars_by_ticker, step)
        indicators = _indicators_at(bars_by_ticker, step)
        graph = build_market_graph(markets, indicators, account=account)
        OntologyReasoner(graph).infer()
        plan = build_goal_execution_plan(goal, account, markets, indicators, graph)
        market_by_ticker = {market.ticker: market for market in markets}
        pending: set[str] = set()

        ranked_intents = sorted(
            plan.intents,
            key=lambda intent: (
                0 if intent.action in {OrderAction.SELL, OrderAction.REDUCE} else 1,
                -intent.confidence,
            ),
        )
        for intent in ranked_intents[:10]:
            if intent.ticker in pending:
                continue
            market = market_by_ticker[intent.ticker]
            current_account = _account_from_state(cash, holdings, prices, timestamp)
            result = RiskManager(rules).validate(
                intent,
                current_account,
                market,
                trades_today=0,
                existing_pending_tickers=pending,
            )
            if not result.approved or result.final_order is None:
                continue
            order = result.final_order
            value = order.quantity * order.limit_price
            quantity = order.quantity
            trading_cost = _simulated_trade_cost(cost_engine, order.side, order.ticker, order.limit_price, quantity)
            if order.side == OrderSide.BUY and cash >= value + trading_cost:
                cash -= value + trading_cost
                holdings[order.ticker] = holdings.get(order.ticker, 0) + order.quantity
                net_value = value + trading_cost
            elif order.side == OrderSide.SELL:
                owned = holdings.get(order.ticker, 0)
                quantity = min(owned, order.quantity)
                if quantity <= 0:
                    continue
                value = quantity * order.limit_price
                trading_cost = _simulated_trade_cost(cost_engine, order.side, order.ticker, order.limit_price, quantity)
                cash += value - trading_cost
                holdings[order.ticker] = owned - quantity
                if holdings[order.ticker] <= 0:
                    del holdings[order.ticker]
                net_value = value - trading_cost
            else:
                continue
            pending.add(order.ticker)
            trades.append(
                SimulatedTrade(
                    timestamp=timestamp,
                    ticker=order.ticker,
                    side=order.side.value,
                    quantity=quantity,
                    price=order.limit_price,
                    value=value,
                    reason="; ".join(intent.reasoning_summary),
                    trading_cost=round(trading_cost, 4),
                    net_value=round(net_value, 4),
                )
            )

    final_prices = _prices_at(bars_by_ticker, len(timestamps) - 1)
    final_timestamp = timestamps[-1]
    for ticker, quantity in list(holdings.items()):
        if quantity <= 0:
            holdings.pop(ticker, None)
            continue
        price = float(final_prices.get(ticker, 0.0) or 0.0)
        if price <= 0:
            continue
        value = quantity * price
        trading_cost = _simulated_trade_cost(cost_engine, OrderSide.SELL, ticker, price, quantity)
        cash += value - trading_cost
        del holdings[ticker]
        trades.append(
            SimulatedTrade(
                timestamp=final_timestamp,
                ticker=ticker,
                side=OrderSide.SELL.value,
                quantity=quantity,
                price=price,
                value=value,
                reason="mandatory final liquidation",
                trading_cost=round(trading_cost, 4),
                net_value=round(value - trading_cost, 4),
            )
        )
    final_positions = {
        ticker: quantity * final_prices[ticker]
        for ticker, quantity in holdings.items()
        if quantity > 0
    }
    final_equity = cash + sum(final_positions.values())
    profit = final_equity - initial_cash
    return_rate = profit / initial_cash

    charts_path = output_dir / "accelerated_demo_charts.csv"
    trades_path = output_dir / "accelerated_demo_trades.csv"
    report_path = output_dir / "accelerated_demo_report.json"
    _write_charts(charts_path, bars_by_ticker)
    _write_trades(trades_path, trades)

    result = AcceleratedDemoResult(
        initial_equity=initial_cash,
        final_equity=round(final_equity, 2),
        profit_amount=round(profit, 2),
        return_rate=round(return_rate, 6),
        target_return_rate=target_return_rate,
        target_profit_amount=round(initial_cash * target_return_rate, 2),
        target_achieved=return_rate >= target_return_rate,
        simulated_minutes=period_minutes,
        bars_per_ticker=len(timestamps),
        ticker_count=len(tickers),
        accelerated_steps=len(timestamps),
        trade_count=len(trades),
        final_cash=round(cash, 2),
        final_positions={key: round(value, 2) for key, value in sorted(final_positions.items())},
        sample_trades=tuple(trades[:20]),
        report_path=str(report_path),
        trades_path=str(trades_path),
        charts_path=str(charts_path),
    )
    report_path.write_text(
        json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def generate_synthetic_charts(
    tickers: tuple[str, ...],
    period_minutes: int,
    seed: int,
) -> dict[str, tuple[ChartBar, ...]]:
    rng = random.Random(seed)
    bar_interval_minutes = 1
    total_bars = max(1, period_minutes)
    charts: dict[str, tuple[ChartBar, ...]] = {}
    for index, ticker in enumerate(tickers):
        calendar = _calendar_for_ticker(ticker)
        timestamps = _session_timestamps(calendar, total_bars, bar_interval_minutes)
        price = rng.uniform(30, 450)
        if ticker in {"005930", "000660"}:
            price = rng.uniform(70_000, 210_000)
        trend_bucket = index % 10
        upward_bias = os.getenv("SIM_DEMO_UPWARD_BIAS", "1").strip().lower() not in {"0", "false", "no"}
        if upward_bias:
            drift = 0.0036 if trend_bucket in {0, 1, 2, 3, 4, 5} else 0.00155
            volatility = 0.0034 + (index % 7) * 0.00055
        else:
            drift = 0.0018 if trend_bucket in {0, 1, 2, 3} else 0.00045
            if trend_bucket in {8, 9}:
                drift = -0.00065
            volatility = 0.004 + (index % 7) * 0.0008
        bars: list[ChartBar] = []
        for step, timestamp in enumerate(timestamps):
            session_name = _session_name_for_timestamp(calendar, timestamp)
            session_volatility = _demo_session_volatility_factor(session_name)
            session_volume = _demo_session_volume_factor(session_name)
            cycle = math.sin(step / 8 + index * 0.7) * 0.0015
            shock = rng.gauss(drift + cycle, volatility * session_volatility)
            if step in {18, 37, 69} and trend_bucket in {0, 1, 2, 3}:
                shock += 0.018
            if not upward_bias and step in {44, 92} and trend_bucket in {8, 9}:
                shock -= 0.022
            price = max(1.0, price * (1 + shock))
            volume_base = 1_000_000 + (index + 1) * 80_000
            volume = int(volume_base * session_volume * (1 + abs(shock) * 35) * rng.uniform(0.75, 1.35))
            bars.append(ChartBar(ticker=ticker, timestamp=timestamp, close=round(price, 4), volume=volume))
        charts[ticker] = tuple(bars)
    return charts


def _markets_at(charts: dict[str, tuple[ChartBar, ...]], step: int) -> tuple[MarketSnapshot, ...]:
    markets = []
    for index, (ticker, bars) in enumerate(charts.items()):
        window = bars[max(0, step - 20) : step + 1]
        returns = [
            (window[i].close - window[i - 1].close) / window[i - 1].close
            for i in range(1, len(window))
            if window[i - 1].close
        ]
        average_value = sum(bar.close * bar.volume for bar in window) / max(1, len(window))
        markets.append(
            MarketSnapshot(
                ticker=ticker,
                market="SIM",
                company_name=f"Demo {ticker}",
                sector=_sector_for(index),
                last_price=bars[step].close,
                average_daily_trading_value=average_value,
                volatility_20d=_stddev(returns),
                source=SourceMetadata(
                    source_name="accelerated_demo_chart",
                    retrieved_at=bars[step].timestamp,
                    raw_url="local://accelerated-demo",
                    source_id=f"demo-chart:{ticker}:{step}",
                ),
            )
        )
    return tuple(markets)


def _indicators_at(
    charts: dict[str, tuple[ChartBar, ...]],
    step: int,
) -> dict[str, IndicatorSnapshot]:
    indicators = {}
    for index, (ticker, bars) in enumerate(charts.items()):
        window = bars[max(0, step - 30) : step + 1]
        closes = [bar.close for bar in window]
        volumes = [bar.volume for bar in window]
        first = closes[0]
        last = closes[-1]
        momentum = (last - first) / first if first else 0.0
        indicators[ticker] = IndicatorSnapshot(
            ticker=ticker,
            revenue_growth=max(-0.10, min(0.35, 0.08 + momentum * 1.8 + (index % 5) * 0.01)),
            operating_income_growth=max(-0.15, min(0.55, 0.12 + momentum * 2.4 + (index % 4) * 0.015)),
            operating_margin=max(0.05, min(0.30, 0.14 + (index % 6) * 0.012 + momentum * 0.30)),
            roe=max(0.02, min(0.24, 0.09 + momentum * 0.40)),
            debt_ratio=0.25 + (index % 8) * 0.035,
            per=max(8.0, min(34.0, 18.0 + (index % 9) * 1.4 - momentum * 18)),
            pbr=1.0 + (index % 7) * 0.22,
            rsi_14d=_rsi(closes[-15:]),
            volume_ratio=_volume_ratio(volumes),
            macro_risk_score=0.30 + (index % 6) * 0.035,
            source_ids=(f"demo-indicator:{ticker}:{step}",),
        )
    return indicators


def _account_from_state(
    cash: float,
    holdings: dict[str, int],
    prices: dict[str, float],
    timestamp: datetime,
) -> AccountSnapshot:
    holding_rows = tuple(
        Holding(
            ticker=ticker,
            market="SIM",
            company_name=f"Demo {ticker}",
            sector="Simulated",
            quantity=quantity,
            average_price=prices[ticker],
            last_price=prices[ticker],
            opened_at=timestamp,
        )
        for ticker, quantity in sorted(holdings.items())
        if quantity > 0
    )
    return AccountSnapshot(cash=cash, holdings=holding_rows, captured_at=timestamp)


def _prices_at(charts: dict[str, tuple[ChartBar, ...]], step: int) -> dict[str, float]:
    return {ticker: bars[step].close for ticker, bars in charts.items()}


def _sector_for(index: int) -> str:
    sectors = ("Technology", "Semiconductor", "Consumer", "Finance", "Energy", "Healthcare")
    return sectors[index % len(sectors)]


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _rsi(closes: list[float]) -> float:
    if len(closes) < 3:
        return 50.0
    gains = []
    losses = []
    for index in range(1, len(closes)):
        delta = closes[index] - closes[index - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
    avg_gain = sum(gains) / max(1, len(gains))
    avg_loss = sum(losses) / max(1, len(losses))
    if avg_loss == 0:
        return 72.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _volume_ratio(volumes: list[int]) -> float:
    if len(volumes) < 2:
        return 1.0
    recent = sum(volumes[-3:]) / min(3, len(volumes))
    base = sum(volumes[:-3] or volumes) / max(1, len(volumes[:-3] or volumes))
    return recent / base if base else 1.0


def _write_charts(path: Path, charts: dict[str, tuple[ChartBar, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("timestamp", "ticker", "close", "volume"))
        writer.writeheader()
        for bars in charts.values():
            for bar in bars:
                writer.writerow(_to_jsonable(bar))


def _write_trades(path: Path, trades: list[SimulatedTrade]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "timestamp",
                "ticker",
                "side",
                "quantity",
                "price",
                "value",
                "reason",
                "currency",
                "fx_rate",
                "value_krw",
                "trading_cost",
                "net_value",
            ),
        )
        writer.writeheader()
        for trade in trades:
            writer.writerow(_to_jsonable(trade))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
