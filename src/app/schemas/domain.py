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
    WATCH = "WATCH"
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
    opened_at: datetime | None = None
    sellable_quantity: int | None = None
    # --- Direction and borrow metadata -------------------------------------- #
    # ``quantity`` stays a positive magnitude for both directions; direction is
    # never encoded as a sign, because a negative quantity survives one refactor
    # and then silently becomes a buy somewhere.
    #
    # Defaults describe the only kind of position that existed before shorts, so
    # every existing construction site keeps its exact meaning (a cash long).
    direction: str = "LONG"
    execution_product: str = "CASH"
    # Broker-authoritative borrow facts. A SHORT position that cannot produce a
    # ``loan_date`` cannot be closed through the 매수상환 (buy-to-cover) contract,
    # so its absence is a fail-closed condition rather than a missing nicety.
    loan_date: str | None = None
    borrow_reference: str | None = None
    borrow_fee_rate: float | None = None
    return_deadline: datetime | None = None

    @property
    def is_short(self) -> bool:
        return str(self.direction or "LONG").upper() == "SHORT"

    @property
    def direction_sign(self) -> int:
        return -1 if self.is_short else 1

    @property
    def market_value(self) -> float:
        """Absolute market value of the exposure (always non-negative).

        Kept unsigned so every existing consumer — portfolio weights, sector
        concentration, affordability — keeps working unchanged. Signed exposure is
        a separate question; ask :attr:`signed_exposure` for it.
        """
        return self.quantity * self.last_price

    @property
    def signed_exposure(self) -> float:
        """Direction-signed exposure, for gross/net exposure limits."""
        return self.direction_sign * self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        """PnL of the exposure as held. A short gains when price falls."""
        return self.direction_sign * self.quantity * (self.last_price - self.average_price)


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    holdings: tuple[Holding, ...]
    realized_pnl_today: float = 0.0
    unrealized_pnl_today: float = 0.0
    base_currency: str = "KRW"
    cash_by_currency: dict[str, float] = field(default_factory=dict)
    orderable_cash_by_currency: dict[str, float] = field(default_factory=dict)
    fx_rate_by_currency: dict[str, float] = field(default_factory=dict)
    position_opened_at_by_ticker: dict[str, datetime] = field(default_factory=dict)
    cash_equivalent_krw: float | None = None
    foreign_cash_krw: float | None = None
    total_equity_krw: float | None = None
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def invested_value(self) -> float:
        return sum(holding.market_value for holding in self.holdings)

    @property
    def securities_market_value(self) -> float:
        return self.invested_value

    @property
    def pure_cash(self) -> float:
        return self.cash if self.cash_equivalent_krw is None else self.cash_equivalent_krw

    @property
    def equity(self) -> float:
        if self.total_equity_krw is not None and self.total_equity_krw > 0:
            return self.total_equity_krw
        return self.pure_cash + self.invested_value

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

    # --- Direction contract -------------------------------------------------- #
    # ``action`` alone is ambiguous once shorts exist: a SELL is either "close the
    # long I hold" or "open a new short", and those two differ in whether the
    # account ends up flat or ends up owing shares. These three fields disambiguate
    # it, and the execution layer refuses any order where they are not all resolved.
    position_direction: str = "LONG"
    # Empty means "infer from action and direction" — see
    # ``app.risk.manager._parse_effect``. Deliberately NOT defaulted to "OPEN": every
    # intent that predates shorts leaves this unset, and a hard "OPEN" default would
    # have relabelled every existing long SELL/REDUCE exit as an ENTRY. That
    # contradicts its own broker side, so the contract-consistency check would have
    # rejected every de-risking order in the system.
    position_effect: str = ""
    execution_product: str = "CASH"

    @property
    def resolved_position_effect(self) -> str:
        """OPEN / CLOSE, inferring from ``action`` when not stated.

        Direction-aware: a BUY opens a LONG but CLOSES a short, and a SELL is the
        mirror. Inferring "SELL == CLOSE" unconditionally is the exact ambiguity this
        contract exists to remove.
        """
        stated = str(self.position_effect or "").strip().upper()
        if stated in {"OPEN", "CLOSE"}:
            return stated
        opening = (
            OrderAction.SELL
            if str(self.position_direction or "").upper() == "SHORT"
            else OrderAction.BUY
        )
        return "OPEN" if self.action == opening else "CLOSE"

    @property
    def is_short_entry(self) -> bool:
        return (
            str(self.position_direction or "").upper() == "SHORT"
            and self.resolved_position_effect == "OPEN"
        )


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
    # --- Direction contract -------------------------------------------------- #
    # ``side`` is what the BROKER sees; these say what the order MEANS. The four
    # combinations are enumerated in ``app.trading.directional.broker_side``.
    # Defaults reproduce the pre-short semantics exactly (cash long open/close),
    # so every existing construction site is unchanged.
    position_direction: str = "LONG"
    # Empty == infer from ``side`` and ``position_direction``. Not defaulted to
    # "OPEN" for the same reason as on ``OrderIntent``: every long exit order built by
    # existing code leaves this unset, and labelling those as entries would make each
    # one contradict its own SELL side.
    position_effect: str = ""
    execution_product: str = "CASH"
    # 대주 (credit borrow) routing metadata. ``credit_type`` is the broker's
    # 신용거래구분; ``loan_date`` (대출일) identifies WHICH borrow lot a buy-to-cover
    # repays and is mandatory on a SHORT CLOSE — lots opened on different dates are
    # separate positions to the broker and cannot be netted.
    credit_type: str | None = None
    loan_date: str | None = None
    # --- Session / venue routing (optional) ---------------------------------- #
    # 비어 있으면 KisSessionOrderRouter 가 현재 capability 와 symbol exchange map 으로
    # 계산한다. 계산 결과가 모호하면 (예: 썸머타임의 미국 주간거래·프리마켓 중첩)
    # 주문은 차단된다 — 임의로 하나를 고르지 않는다.
    #
    # 값을 채워 두면 라우터는 그 세션/venue 로만 라우팅하며, 현재 세션과 맞지 않으면
    # ``SESSION_MISMATCH`` 로 fail-closed 한다. 정정·취소가 원주문 route family 를
    # 유지해야 하기 때문에 필요한 필드다.
    market_session: str = ""
    execution_venue: str = ""
    exchange_code: str = ""
    order_condition: str = ""

    @property
    def resolved_position_effect(self) -> str:
        stated = str(self.position_effect or "").strip().upper()
        if stated in {"OPEN", "CLOSE"}:
            return stated
        opening = (
            OrderSide.SELL
            if str(self.position_direction or "").upper() == "SHORT"
            else OrderSide.BUY
        )
        return "OPEN" if self.side == opening else "CLOSE"

    @property
    def is_credit_borrow(self) -> bool:
        return str(self.execution_product or "").upper() == "CREDIT_BORROW"


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
    # Block BUYs of warrants / units / rights (non-common-equity instruments).
    # These are typically illiquid SPAC/IPO securities (e.g. NASDAQ 5th-letter
    # "W"/"U"/"R", or ".WS"/"-UN"/".RT" suffixes) that cannot be reliably exited.
    # SELL/REDUCE of already-held positions is never blocked by this flag.
    warrant_unit_buys_allowed: bool = False
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
    # --- Short-side policy --------------------------------------------------- #
    # ``short_selling_allowed`` / ``credit_loan_allowed`` above remain the account
    # -level master switches and still default to False. These are the *additional*
    # constraints that apply once an operator turns those on; they are never
    # sufficient on their own, because a per-strategy deployment state (see
    # ``app.trading.short_strategy_promotion``) still has to authorise the arm.
    #
    # Deliberately NOT one "shorts enabled" flag: a single flag is exactly how an
    # unvalidated strategy reaches a live order.
    max_open_short_positions: int = 1
    max_single_short_weight: float = 0.01
    max_total_short_weight: float = 0.05
    max_gross_exposure: float = 1.0
    max_net_short_exposure: float = 0.10
    max_daily_short_entries: int = 1
    max_short_holding_minutes: int = 30
    overnight_short_allowed: bool = False
    # A short's risk budget starts at half a long's, and 0.5 is a ceiling rather
    # than a tuning target: the loss is unbounded above and accelerates.
    short_risk_budget_ratio_of_long: float = 0.5
    short_daily_loss_stop: float = 0.005
    max_borrow_fee_bps: float = 40.0
    min_borrow_snapshot_freshness_seconds: float = 30.0
    min_hours_before_recall_deadline: float = 24.0
    require_short_stop_order_capability: bool = True
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
