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


class PrincipalProtectionMode(StrEnum):
    NORMAL_GROWTH = "NORMAL_GROWTH"
    PROFIT_ONLY = "PROFIT_ONLY"
    DE_RISK = "DE_RISK"
    PRINCIPAL_LOCKDOWN = "PRINCIPAL_LOCKDOWN"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class PrincipalProtectionDecisionAction(StrEnum):
    ALLOW = "ALLOW"
    REDUCE_SIZE = "REDUCE_SIZE"
    BLOCK = "BLOCK"
    SELL_ONLY = "SELL_ONLY"
    LOCKDOWN = "LOCKDOWN"


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


class InvestorGroup(StrEnum):
    RETAIL = "RETAIL"
    INSTITUTION = "INSTITUTION"
    FOREIGN = "FOREIGN"
    SUSPECTED_SMART_MONEY = "SUSPECTED_SMART_MONEY"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    retrieved_at: datetime
    raw_url: str | None = None
    source_id: str | None = None
    source_type: str = "unknown"
    trust_level: int = 0
    observed_at: datetime | None = None
    latency_sec: float | None = None
    is_realtime: bool = False
    is_delayed: bool = False
    is_synthetic: bool = False
    is_backfilled: bool = False
    license_policy: str = "unknown"
    quality_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "trust_level", max(0, min(5, int(self.trust_level))))
        object.__setattr__(self, "quality_score", max(0.0, min(1.0, float(self.quality_score))))
        if self.latency_sec is not None:
            object.__setattr__(self, "latency_sec", max(0.0, float(self.latency_sec)))


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
    base_currency: str = "KRW"
    cash_by_currency: dict[str, float] = field(default_factory=dict)
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
    investor_flow: InvestorFlowSnapshot | None = None


@dataclass(frozen=True)
class InvestorFlowSnapshot:
    ticker: str
    market: str
    foreign_net_buy: float = 0.0
    institution_net_buy: float = 0.0
    retail_net_buy: float = 0.0
    program_net_buy: float = 0.0
    short_net_change: float = 0.0
    volume_change_rate: float = 0.0
    price_change_rate: float = 0.0
    trading_value: float = 0.0
    observed_at: datetime | None = None
    source: SourceMetadata | None = None

    @property
    def net_buy_total(self) -> float:
        return self.foreign_net_buy + self.institution_net_buy + self.retail_net_buy


@dataclass(frozen=True)
class RealtimeQuote:
    ticker: str
    market: str
    observed_at: datetime
    last_price: float
    bid_price: float | None = None
    ask_price: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None
    change: float | None = None
    change_rate: float | None = None
    source: SourceMetadata | None = None


@dataclass(frozen=True)
class RealtimeExecution:
    ticker: str
    market: str
    executed_at: datetime
    price: float
    quantity: int
    side: str | None = None
    trade_id: str | None = None
    source: SourceMetadata | None = None


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
    key_facts: tuple[str, ...] = ()
    event_labels: tuple[str, ...] = ()
    classification_confidence: float = 0.0
    classification_model: str = "keyword_v1"


@dataclass(frozen=True)
class TimeSynchronizedTickerFrame:
    ticker: str
    market: str
    bucket_start: datetime
    bucket_end: datetime
    market_snapshot: MarketSnapshot | None = None
    realtime_quotes: tuple[RealtimeQuote, ...] = ()
    realtime_executions: tuple[RealtimeExecution, ...] = ()
    events: tuple[ClassifiedEvent, ...] = ()
    raw_records: tuple[RawSourceRecord, ...] = ()
    macro_metrics: tuple[MacroMetricRecord, ...] = ()
    impact_score: float = 0.0
    data_source_ids: tuple[str, ...] = ()

    @property
    def frame_id(self) -> str:
        return f"TemporalFrame:{self.ticker}:{self.bucket_start.isoformat()}"


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
    model_uncertainty: float | None = None

    strategy_family: str | None = None
    signal_name: str | None = None
    expected_exit_price: float | None = None
    expected_holding_minutes: int | None = None
    gross_expected_return: float | None = None
    target_net_return: float | None = None
    validation_id: str | None = None
    cost_breakdown: dict[str, Any] | None = None
    ontology_tags: tuple[str, ...] = ()
    strategy_metadata: dict[str, Any] = field(default_factory=dict)


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
class PrincipalProtectionConfig:
    initial_principal: float = 0.0
    principal_floor_enabled: bool = True
    principal_floor_ratio: float = 1.0
    profit_lockin_enabled: bool = True
    profit_lockin_ratio: float = 0.30
    cppi_enabled: bool = True
    cppi_multiplier: float = 2.0
    max_gap_loss_assumption: float = 0.12
    cost_buffer_ratio: float = 0.003
    per_trade_risk_budget_ratio: float = 0.0025
    daily_risk_budget_ratio: float = 0.005
    weekly_risk_budget_ratio: float = 0.015
    max_total_drawdown: float = 0.05
    fractional_kelly_enabled: bool = False
    fractional_kelly_ratio: float = 0.25
    cvar_enabled: bool = False
    cvar_confidence: float = 0.95
    principal_lockdown_enabled: bool = True
    count_unrealized_profit_as_growth: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class PrincipalProtectionState:
    initial_principal: float
    current_equity: float
    protected_floor: float
    high_watermark: float
    locked_profit: float
    cushion: float
    risk_budget: float
    available_growth_capital: float
    current_mode: PrincipalProtectionMode
    floor_breach_status: bool
    drawdown_from_high_watermark: float
    cost_buffer: float
    gap_risk_buffer: float
    active_risky_exposure: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrincipalProtectionDecision:
    action: PrincipalProtectionDecisionAction
    state: PrincipalProtectionState
    allowed: bool
    reason_codes: tuple[str, ...]
    explanations: tuple[str, ...]
    estimated_trade_loss: float = 0.0
    suggested_quantity: int | None = None
    max_risky_exposure: float = 0.0


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
    max_intraday_position_weight: float = 0.025
    max_short_horizon_downside_risk: float = 0.012
    emergency_exit_loss: float = 0.018
    min_source_trust_level: int = 4
    min_data_quality_score: float = 0.80
    max_quote_age_seconds: float = 5.0
    max_model_uncertainty: float = 0.60
    synthetic_live_data_allowed: bool = False
    unknown_source_live_allowed: bool = False
    principal_protection: PrincipalProtectionConfig = field(default_factory=PrincipalProtectionConfig)


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
