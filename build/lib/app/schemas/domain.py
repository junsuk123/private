from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class OrderAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    REBALANCE = "REBALANCE"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class EventType(StrEnum):
    DISCLOSURE = "DISCLOSURE"
    NEWS = "NEWS"
    MACRO = "MACRO"
    MARKET = "MARKET"
    FINANCIAL = "FINANCIAL"


class SentimentDirection(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    retrieved_at: datetime
    raw_url: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class Holding:
    ticker: str
    market: str
    company_name: str
    sector: str
    quantity: int
    average_price: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.last_price - self.average_price)


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    holdings: tuple[Holding, ...]
    realized_pnl_today: float = 0.0
    unrealized_pnl_today: float = 0.0
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def invested_value(self) -> float:
        return sum(holding.market_value for holding in self.holdings)

    @property
    def equity(self) -> float:
        return self.cash + self.invested_value

    def holdings_by_ticker(self) -> dict[str, float]:
        return {holding.ticker: holding.market_value for holding in self.holdings}


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    market: str
    company_name: str
    sector: str
    last_price: float
    average_daily_trading_value: float
    volatility_20d: float
    source: SourceMetadata


@dataclass(frozen=True)
class IndicatorSnapshot:
    ticker: str
    revenue_growth: float | None
    operating_income_growth: float | None
    operating_margin: float | None
    roe: float | None
    debt_ratio: float | None
    per: float | None
    pbr: float | None
    rsi_14d: float | None
    volume_ratio: float | None
    macro_risk_score: float
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawSourceRecord:
    source: SourceMetadata
    content_type: str
    payload: str


@dataclass(frozen=True)
class FinancialMetricRecord:
    ticker: str
    company_name: str
    fiscal_year: int
    revenue: float | None
    operating_income: float | None
    net_income: float | None
    total_assets: float | None
    total_liabilities: float | None
    source: SourceMetadata


@dataclass(frozen=True)
class MacroMetricRecord:
    name: str
    value: float
    observed_at: datetime
    source: SourceMetadata


@dataclass(frozen=True)
class ClassifiedEvent:
    event_id: str
    event_type: EventType
    title: str
    summary: str
    companies: tuple[str, ...]
    tickers: tuple[str, ...]
    sectors: tuple[str, ...]
    sentiment: SentimentDirection
    event_date: datetime
    source: SourceMetadata


@dataclass(frozen=True)
class ReasoningPath:
    path_id: str
    ticker: str
    conclusion: str
    confidence: float
    supporting_triples: tuple[str, ...]
    contradicting_triples: tuple[str, ...]
    risk_triples: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class PortfolioStatusReport:
    equity: float
    cash_weight: float
    position_weights: dict[str, float]
    sector_weights: dict[str, float]
    daily_pnl_ratio: float


@dataclass(frozen=True)
class StrategySignal:
    ticker: str
    action: OrderAction
    confidence: float
    score: float
    supporting_factors: tuple[str, ...]
    contradicting_factors: tuple[str, ...]
    reasoning_path_ids: tuple[str, ...]


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    market: str
    action: OrderAction
    suggested_weight: float
    confidence: float
    valid_until: datetime
    reasoning_summary: tuple[str, ...]
    supporting_factors: tuple[str, ...]
    contradicting_factors: tuple[str, ...]
    source_data_ids: tuple[str, ...]


@dataclass(frozen=True)
class FinalOrder:
    ticker: str
    market: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    limit_price: float
    time_in_force: str = "DAY"
    manual_approval_required: bool = True


@dataclass(frozen=True)
class RiskRules:
    max_single_stock_weight: float = 0.05
    max_sector_weight: float = 0.25
    minimum_cash_reserve: float = 0.30
    daily_loss_stop: float = 0.01
    max_trades_per_day: int = 5
    min_average_daily_trading_value: float = 1_000_000_000
    max_volatility: float = 0.08
    order_type: OrderType = OrderType.LIMIT
    manual_approval_required: bool = True
    live_trading_enabled: bool = False
    margin_trading_allowed: bool = False
    short_selling_allowed: bool = False
    derivatives_allowed: bool = False
    leverage_etf_allowed: bool = False
    credit_loan_allowed: bool = False
    llm_direct_order_execution_allowed: bool = False


@dataclass(frozen=True)
class RiskManagerResult:
    ticker: str
    action: OrderAction
    approved: bool
    adjusted_weight: float | None
    checks: dict[str, bool]
    rejection_reasons: tuple[str, ...]
    final_order: FinalOrder | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
