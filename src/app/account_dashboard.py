from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from app.cost import TradingCostEngine
from app.account_snapshot_store import AccountSnapshotStore


_COST_ENGINE = TradingCostEngine()


@dataclass(frozen=True)
class HoldingDashboardRow:
    market_group: str
    market: str
    exchange: str
    ticker: str
    name: str
    currency: str
    quantity: float
    available_quantity: float
    average_price: float
    current_price: float
    purchase_amount_original: float
    evaluation_amount_original: float
    purchase_amount_krw: float
    evaluation_amount_krw: float
    weight_of_total_asset: float
    unrealized_pnl_original: float
    unrealized_pnl_krw: float
    unrealized_pnl_rate: float
    realized_pnl_krw: float = 0.0
    last_price_source: str = "account"
    updated_at: str = ""
    is_stale: bool = False
    round_trip_cost_rate: float = 0.0
    break_even_price: float = 0.0
    estimated_net_pnl_krw: float = 0.0


@dataclass(frozen=True)
class CashCurrencyRow:
    currency: str
    cash_balance: float
    orderable_amount: float
    withdrawable_amount: float
    fx_rate_to_krw: float
    krw_equivalent: float
    updated_at: str
    source: str


@dataclass(frozen=True)
class TradeHistoryRow:
    occurred_at: str
    market_group: str
    market: str
    exchange: str
    ticker: str
    name: str
    side: str
    order_type: str
    order_id: str
    order_status: str
    ordered_quantity: float
    filled_quantity: float
    average_fill_price: float
    amount_original: float
    amount_krw: float
    fee_krw: float
    tax_krw: float
    realized_pnl_krw: float
    currency: str
    source: str


@dataclass(frozen=True)
class HoldingOrderStatusRow:
    ticker: str
    name: str
    market_group: str
    market: str
    exchange: str
    currency: str
    quantity: float
    average_price: float
    current_price: float
    order_state: str
    order_status: str
    order_action: str
    order_type: str
    order_id: str
    order_quantity: float
    filled_quantity: float
    order_price: float
    order_summary: str
    occurred_at: str
    source: str


@dataclass(frozen=True)
class AccountDashboardSnapshot:
    snapshot_id: str
    created_at: str
    updated_at: str
    source: str
    is_live: bool
    is_stale: bool
    stale_seconds: float
    base_currency: str
    total_asset_krw: float
    net_asset_krw: float
    cash_equivalent_krw: float
    krw_cash: float
    foreign_cash_krw: float
    settlement_cash_krw: float
    cash_by_currency: dict[str, float]
    orderable_cash_by_currency: dict[str, float]
    domestic_stock_value_krw: float
    overseas_stock_value_krw: float
    domestic_unrealized_pnl_krw: float
    overseas_unrealized_pnl_krw: float
    realized_pnl_today_krw: float
    realized_pnl_period_krw: float
    unrealized_pnl_krw: float
    total_pnl_krw: float
    total_pnl_rate: float
    asset_allocations: list[dict[str, Any]]
    principal_protection: dict[str, Any] = field(default_factory=dict)
    data_quality_warnings: list[str] = field(default_factory=list)


class AccountDashboardService:
    def __init__(
        self,
        *,
        status_provider: Callable[[], dict[str, Any] | None] | None = None,
        logs_provider: Callable[[], dict[str, Any] | None] | None = None,
        technical_provider: Callable[[], list[dict[str, Any]] | None] | None = None,
        macro_micro_provider: Callable[[], dict[str, Any] | None] | None = None,
        store: AccountSnapshotStore | None = None,
    ) -> None:
        self.status_provider = status_provider
        self.logs_provider = logs_provider
        # Returns the most recent technical decision diagnostics (advisory).
        self.technical_provider = technical_provider
        # Returns the most recent macro–micro reasoning bundle (advisory).
        self.macro_micro_provider = macro_micro_provider
        self.store = store or AccountSnapshotStore()

    def build_dashboard(self, *, persist: bool = True) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        previous_dashboard = self.store.latest_dashboard() or {}
        previous_snapshot = dict(previous_dashboard.get("snapshot") or {})
        status = self._status_payload()
        logs = self._logs_payload()
        authoritative_live = bool(
            status.get("account_checked")
            or status.get("basis_source") == "kis_live_account"
            or status.get("source") == "kis_live_account"
        )
        updated_at = _parse_time(status.get("updated_at")) or now
        stale_seconds = max(0.0, (now - updated_at).total_seconds())
        is_stale = stale_seconds > 90
        positions = list(status.get("positions") or [])
        holdings = [
            row
            for item in positions
            if isinstance(item, dict)
            if (row := _holding_from_position(item, updated_at.isoformat())) is not None
        ]
        cash_rows = _cash_rows(status, updated_at.isoformat())

        krw_cash = _num(status.get("krw_cash") or status.get("actual_deposit") or status.get("cash"))
        foreign_cash_krw = _num(status.get("foreign_cash_krw"))
        cash_equivalent_krw = _num(status.get("cash_equivalent_krw") or (krw_cash + foreign_cash_krw))
        settlement_cash_krw = max(0.0, cash_equivalent_krw - krw_cash - foreign_cash_krw)
        domestic_value = sum(row.evaluation_amount_krw for row in holdings if row.market_group == "domestic")
        overseas_value = sum(row.evaluation_amount_krw for row in holdings if row.market_group == "overseas")
        if domestic_value <= 0 and overseas_value <= 0:
            invested = _num(status.get("invested") or status.get("invested_value"))
            domestic_value = invested
        total_asset_krw = _num(status.get("equity") or status.get("actual_equity") or status.get("account_value"))
        if total_asset_krw <= 0:
            total_asset_krw = cash_equivalent_krw + domestic_value + overseas_value
        if total_asset_krw <= 0 and previous_snapshot and not authoritative_live:
            total_asset_krw = _num(previous_snapshot.get("total_asset_krw") or previous_snapshot.get("net_asset_krw"))
        if cash_equivalent_krw <= 0 and previous_snapshot and not authoritative_live:
            cash_equivalent_krw = _num(previous_snapshot.get("cash_equivalent_krw"))
        if krw_cash <= 0 and previous_snapshot and not authoritative_live:
            krw_cash = _num(previous_snapshot.get("krw_cash"))
        if foreign_cash_krw <= 0 and previous_snapshot and not authoritative_live:
            foreign_cash_krw = _num(previous_snapshot.get("foreign_cash_krw"))
        if settlement_cash_krw <= 0 and previous_snapshot and not authoritative_live:
            settlement_cash_krw = _num(previous_snapshot.get("settlement_cash_krw"))
        holdings = [
            HoldingDashboardRow(**{**asdict(row), "weight_of_total_asset": _ratio(row.evaluation_amount_krw, total_asset_krw)})
            for row in holdings
        ]
        domestic_unrealized = sum(row.unrealized_pnl_krw for row in holdings if row.market_group == "domestic")
        overseas_unrealized = sum(row.unrealized_pnl_krw for row in holdings if row.market_group == "overseas")
        unrealized = domestic_unrealized + overseas_unrealized
        realized_today = _num(status.get("realized_pnl_today_krw") or status.get("realized_pnl_today"))
        realized_period = _num(status.get("realized_pnl_period_krw") or realized_today)
        total_pnl = realized_period + unrealized
        purchase_total = sum(max(0.0, row.purchase_amount_krw) for row in holdings)
        snapshot = AccountDashboardSnapshot(
            snapshot_id=uuid4().hex,
            created_at=now.isoformat(),
            updated_at=updated_at.isoformat(),
            source=str(status.get("basis_source") or status.get("source") or "local_status"),
            is_live=authoritative_live,
            is_stale=is_stale,
            stale_seconds=stale_seconds,
            base_currency=str(status.get("base_currency") or "KRW"),
            total_asset_krw=total_asset_krw,
            net_asset_krw=total_asset_krw,
            cash_equivalent_krw=cash_equivalent_krw,
            krw_cash=krw_cash,
            foreign_cash_krw=foreign_cash_krw,
            settlement_cash_krw=settlement_cash_krw,
            cash_by_currency={str(k).upper(): _num(v) for k, v in dict(status.get("cash_by_currency") or {"KRW": krw_cash}).items()},
            orderable_cash_by_currency=_orderable_by_currency(status),
            domestic_stock_value_krw=domestic_value,
            overseas_stock_value_krw=overseas_value,
            domestic_unrealized_pnl_krw=domestic_unrealized,
            overseas_unrealized_pnl_krw=overseas_unrealized,
            realized_pnl_today_krw=realized_today,
            realized_pnl_period_krw=realized_period,
            unrealized_pnl_krw=unrealized,
            total_pnl_krw=total_pnl,
            total_pnl_rate=_ratio(total_pnl, purchase_total),
            asset_allocations=_allocations(
                total_asset_krw,
                domestic_value,
                overseas_value,
                krw_cash,
                foreign_cash_krw,
                settlement_cash_krw,
            ),
            principal_protection=dict(status.get("principal_protection") or {}),
            data_quality_warnings=_warnings(status, logs, is_stale),
        )
        trade_rows = _trade_rows(logs)
        holding_order_rows = _holding_order_rows(holdings, logs)
        dashboard = {
            "snapshot": asdict(snapshot),
            "holdings": [asdict(row) for row in holdings],
            "cash": [asdict(row) for row in cash_rows],
            "trades": trade_rows,
            "holding_orders": holding_order_rows,
            "profitability": _profitability_summary(trade_rows, snapshot),
            "technical": build_technical_panel(self._technical_payload()),
            "macro_micro": build_macro_micro_panel(self._macro_micro_payload()),
            "logs": {
                "collection_log": list(logs.get("collection_log") or []),
                "last_error": logs.get("last_error"),
                "live_execution_summary": logs.get("live_execution_summary"),
                "warnings": snapshot.data_quality_warnings,
            },
        }
        if persist:
            self.store.save_dashboard(dashboard)
        return dashboard

    def cached_asset_summary(self) -> dict[str, Any]:
        """Return a fast, broker-call-free asset summary for the trading terminal."""
        latest = self.store.latest_dashboard() or {}
        latest_snapshot = dict(latest.get("snapshot") or {})
        latest_live = self.store.latest_dashboard_for_source("kis_live_account") or {}
        live_snapshot = dict(latest_live.get("snapshot") or {})
        latest_is_live = bool(
            latest_snapshot.get("is_live")
            and latest_snapshot.get("source") == "kis_live_account"
        )
        display_snapshot = latest_snapshot if latest_is_live else live_snapshot
        if latest_is_live:
            status = "live"
            message = "KIS 실계좌 확인값"
        elif display_snapshot:
            status = "last_known"
            message = "KIS 연결을 확인할 수 없어 마지막 실계좌 확인값을 표시합니다."
        else:
            status = "unavailable"
            message = "확인된 KIS 실계좌 자산 정보가 없습니다."
        verified_at = display_snapshot.get("updated_at") or display_snapshot.get("created_at")
        return {
            "status": status,
            "authoritative": latest_is_live,
            "message": message,
            "current_source": latest_snapshot.get("source") or "unavailable",
            "last_verified_at": verified_at,
            "snapshot": display_snapshot if display_snapshot else None,
            "holdings": list((latest if latest_is_live else latest_live).get("holdings") or []),
        }

    def technical(self) -> dict[str, Any]:
        return build_technical_panel(self._technical_payload())

    def macro_micro(self) -> dict[str, Any]:
        return build_macro_micro_panel(self._macro_micro_payload())

    def holdings(self) -> list[dict[str, Any]]:
        return list(self.build_dashboard(persist=False).get("holdings") or [])

    def cash(self) -> list[dict[str, Any]]:
        return list(self.build_dashboard(persist=False).get("cash") or [])

    def trades(self) -> list[dict[str, Any]]:
        return list(self.build_dashboard(persist=False).get("trades") or [])

    def holding_orders(self) -> list[dict[str, Any]]:
        return list(self.build_dashboard(persist=False).get("holding_orders") or [])

    def logs(self) -> dict[str, Any]:
        return dict(self.build_dashboard(persist=False).get("logs") or {})

    def asset_history(self, range_name: str = "1D") -> list[dict[str, Any]]:
        history = self.store.asset_history(range_name)
        if history:
            return history
        latest = self.store.latest_dashboard() or self.build_dashboard(persist=False)
        snapshot = dict(latest.get("snapshot") or {})
        if not snapshot:
            return []
        return [
            {
                "created_at": snapshot.get("created_at"),
                "total_asset_krw": _num(snapshot.get("total_asset_krw")),
                "cash_equivalent_krw": _num(snapshot.get("cash_equivalent_krw")),
                "domestic_stock_value_krw": _num(snapshot.get("domestic_stock_value_krw")),
                "overseas_stock_value_krw": _num(snapshot.get("overseas_stock_value_krw")),
                "unrealized_pnl_krw": _num(snapshot.get("unrealized_pnl_krw")),
                "realized_pnl_krw": _num(snapshot.get("realized_pnl_period_krw")),
                "total_pnl_krw": _num(snapshot.get("total_pnl_krw")),
            }
        ]

    def _status_payload(self) -> dict[str, Any]:
        if self.status_provider is None:
            return {}
        try:
            payload = self.status_provider() or {}
        except Exception as exc:  # noqa: BLE001 - dashboard should degrade, not break trading.
            return {"last_error": str(exc), "basis_source": "status_provider_error"}
        if "status" in payload and isinstance(payload.get("status"), dict):
            base = dict(payload["status"])
            base.setdefault("positions", payload.get("positions"))
            return base
        return dict(payload)

    def _logs_payload(self) -> dict[str, Any]:
        if self.logs_provider is None:
            return {}
        try:
            return dict(self.logs_provider() or {})
        except Exception as exc:  # noqa: BLE001
            return {"last_error": str(exc), "collection_log": []}

    def _technical_payload(self) -> list[dict[str, Any]]:
        if self.technical_provider is None:
            return []
        try:
            return list(self.technical_provider() or [])
        except Exception:  # noqa: BLE001 - advisory panel must never break the dashboard.
            return []

    def _macro_micro_payload(self) -> dict[str, Any] | None:
        if self.macro_micro_provider is None:
            return None
        try:
            return self.macro_micro_provider()
        except Exception:  # noqa: BLE001 - advisory panel must never break the dashboard.
            return None


def _holding_from_position(position: dict[str, Any], updated_at: str) -> HoldingDashboardRow | None:
    currency = str(position.get("currency") or ("KRW" if str(position.get("market") or "").upper() in {"KR", "KRX", "KOSPI", "KOSDAQ"} else "USD")).upper()
    market = str(position.get("market") or ("KRX" if currency == "KRW" else "US")).upper()
    market_group = "domestic" if currency == "KRW" or market in {"KR", "KRX", "KOSPI", "KOSDAQ"} else "overseas"
    quantity = _num(position.get("quantity"))
    if quantity <= 0:
        return None
    average_price = _num(position.get("average_price") or position.get("avg_price"))
    current_price = _num(position.get("last_price") or position.get("current_price"))
    evaluation_krw = _num(position.get("market_value_krw") or position.get("market_value") or (quantity * current_price))
    pnl_krw = _num(position.get("unrealized_pnl_krw"))
    purchase_krw = _num(position.get("purchase_amount_krw"))
    if purchase_krw <= 0 and pnl_krw != 0:
        purchase_krw = max(0.0, evaluation_krw - pnl_krw)
    if purchase_krw <= 0:
        purchase_krw = quantity * average_price if market_group == "domestic" else evaluation_krw
    if pnl_krw == 0:
        pnl_krw = evaluation_krw - purchase_krw
    # Exit-cost aware fields. round_trip_cost_rate is attached by the realtime exit
    # decision engine to exit intents; account snapshots rarely carry it, so we
    # fall back to a current-price estimate here so the dashboard can show a
    # realistic break-even and net PnL instead of n/a for every live holding.
    round_trip_cost_rate = _num(
        position.get("round_trip_cost_rate") or position.get("all_in_cost_rate")
    )
    if round_trip_cost_rate <= 0 and quantity > 0 and average_price > 0 and current_price > 0:
        round_trip_cost_rate = _estimate_round_trip_cost_rate(
            market_group=market_group,
            market=market,
            exchange=str(position.get("exchange") or market),
            currency=currency,
            ticker=str(position.get("ticker") or ""),
            name=str(position.get("name") or position.get("company_name") or ""),
            sector=str(position.get("sector") or ""),
            quantity=quantity,
            average_price=average_price,
            current_price=current_price,
        )
    break_even_price = average_price * (1.0 + round_trip_cost_rate) if (average_price > 0 and round_trip_cost_rate > 0) else 0.0
    estimated_net_pnl_krw = pnl_krw - (round_trip_cost_rate * evaluation_krw) if round_trip_cost_rate > 0 else 0.0
    return HoldingDashboardRow(
        market_group=market_group,
        market=market,
        exchange=str(position.get("exchange") or market),
        ticker=str(position.get("ticker") or "").upper(),
        name=str(position.get("name") or position.get("company_name") or position.get("ticker") or ""),
        currency=currency,
        quantity=quantity,
        available_quantity=_num(position.get("available_quantity") or position.get("ord_psbl_qty") or quantity),
        average_price=average_price,
        current_price=current_price,
        purchase_amount_original=_num(position.get("purchase_amount_original") or purchase_krw),
        evaluation_amount_original=_num(position.get("evaluation_amount_original") or evaluation_krw),
        purchase_amount_krw=purchase_krw,
        evaluation_amount_krw=evaluation_krw,
        weight_of_total_asset=0.0,
        unrealized_pnl_original=_num(position.get("unrealized_pnl_original") or pnl_krw),
        unrealized_pnl_krw=pnl_krw,
        unrealized_pnl_rate=_ratio(pnl_krw, purchase_krw),
        realized_pnl_krw=_num(position.get("realized_pnl_krw")),
        last_price_source=str(position.get("last_price_source") or "account"),
        updated_at=updated_at,
        is_stale=False,
        round_trip_cost_rate=round_trip_cost_rate,
        break_even_price=break_even_price,
        estimated_net_pnl_krw=estimated_net_pnl_krw,
    )


def _estimate_round_trip_cost_rate(
    *,
    market_group: str,
    market: str,
    exchange: str,
    currency: str,
    ticker: str,
    quantity: float,
    average_price: float,
    current_price: float,
    name: str = "",
    sector: str = "",
) -> float:
    venue = _cost_venue_for_position(market_group=market_group, market=market, exchange=exchange, ticker=ticker)
    instrument_type = _cost_instrument_for_position(currency=currency, ticker=ticker, name=name, sector=sector)
    try:
        cost = _COST_ENGINE.estimate(
            symbol=ticker,
            market=market,
            venue=venue,
            instrument_type=instrument_type,
            entry_price=average_price,
            expected_exit_price=current_price,
            quantity=max(1, int(quantity)),
        )
    except Exception:
        return 0.0
    return _num(cost.total_cost_rate)


def _cost_instrument_for_position(*, currency: str, ticker: str, name: str = "", sector: str = "") -> str:
    if str(currency or "").upper() != "KRW":
        return "overseas_stock"
    descriptor = f"{ticker} {name} {sector}".lower()
    if any(token in descriptor for token in ("etf", "etn", "elw", "상장지수", "인버스", "레버리지")):
        return "domestic_etf"
    return "domestic_stock"


def _cost_venue_for_position(*, market_group: str, market: str, exchange: str, ticker: str) -> str:
    market_name = str(market or exchange or "").upper().strip()
    ticker_text = str(ticker or "").strip()
    if market_name in {"NXT", "KOSPI", "KOSDAQ", "KONEX"}:
        return market_name
    if market_name in {"KR", "KRX"} or market_group == "domestic" or ticker_text.isdigit():
        return "KRX"
    if market_name in {"NASD", "NYSE", "AMEX", "OVERSEAS", "US"}:
        return "NASD"
    return "KRX" if market_group == "domestic" else "NASD"


def _cash_rows(status: dict[str, Any], updated_at: str) -> list[CashCurrencyRow]:
    cash_by_currency = dict(status.get("cash_by_currency") or {})
    if "KRW" not in cash_by_currency:
        cash_by_currency["KRW"] = _num(status.get("krw_cash") or status.get("cash"))
    orderable = _orderable_by_currency(status)
    rows: list[CashCurrencyRow] = []
    for currency, amount in sorted(cash_by_currency.items()):
        code = str(currency).upper()
        krw_equivalent = _num(amount) if code == "KRW" else _num(status.get("foreign_cash_krw")) if code == "USD" else _num(amount)
        fx_rate = 1.0 if code == "KRW" else _ratio(krw_equivalent, _num(amount)) or 0.0
        rows.append(
            CashCurrencyRow(
                currency=code,
                cash_balance=_num(amount),
                orderable_amount=_num(orderable.get(code, amount)),
                withdrawable_amount=_num(amount),
                fx_rate_to_krw=fx_rate,
                krw_equivalent=krw_equivalent,
                updated_at=updated_at,
                source=str(status.get("basis_source") or "account"),
            )
        )
    settlement_cash = max(
        0.0,
        _num(status.get("cash_equivalent_krw"))
        - _num(status.get("krw_cash") or status.get("actual_deposit") or status.get("cash"))
        - _num(status.get("foreign_cash_krw")),
    )
    if settlement_cash > 0.5:
        rows.append(
            CashCurrencyRow(
                currency="KRW_SETTLEMENT",
                cash_balance=settlement_cash,
                orderable_amount=0.0,
                withdrawable_amount=0.0,
                fx_rate_to_krw=1.0,
                krw_equivalent=settlement_cash,
                updated_at=updated_at,
                source=str(status.get("basis_source") or "account"),
            )
        )
    return rows


def _orderable_by_currency(status: dict[str, Any]) -> dict[str, float]:
    source = status.get("orderable_cash_by_currency") or status.get("cash_by_currency") or {}
    if isinstance(source, dict):
        return {str(key).upper(): _num(value) for key, value in source.items()}
    return {}


def _allocations(total: float, domestic: float, overseas: float, krw_cash: float, foreign_cash: float) -> list[dict[str, Any]]:
    rows = [
        ("domestic_stock", "국내주식", domestic),
        ("overseas_stock", "해외주식", overseas),
        ("krw_cash", "원화 예수금", krw_cash),
        ("foreign_cash", "외화 예수금", foreign_cash),
    ]
    used = sum(max(0.0, value) for _, _, value in rows)
    if total > used:
        rows.append(("other", "기타/미분류", total - used))
    return [{"key": key, "label": label, "value_krw": value, "weight": _ratio(value, total)} for key, label, value in rows]


def _allocations(
    total: float,
    domestic: float,
    overseas: float,
    krw_cash: float,
    foreign_cash: float,
    settlement_cash: float = 0.0,
) -> list[dict[str, Any]]:
    rows = [
        ("domestic_stock", "국내주식", domestic),
        ("overseas_stock", "해외주식", overseas),
        ("krw_cash", "원화 예수금", krw_cash),
        ("foreign_cash", "외화 예수금", foreign_cash),
        ("settlement_cash", "결제예정/미분류 현금", settlement_cash),
    ]
    used = sum(max(0.0, value) for _, _, value in rows)
    if total > used:
        rows.append(("other", "기타/미분류", total - used))
    return [{"key": key, "label": label, "value_krw": value, "weight": _ratio(value, total)} for key, label, value in rows]


def _trade_rows(logs: dict[str, Any]) -> list[dict[str, Any]]:
    journal = logs.get("live_order_journal")
    if isinstance(journal, dict):
        rows = list(journal.get("submitted_orders") or ()) + list(journal.get("recent_executions") or ())
        normalized = [_trade_from_dict(item) for item in rows if isinstance(item, dict)]
        return [asdict(row) for row in _dedupe_trade_rows(normalized)][-50:]
    summary = logs.get("live_execution_summary")
    if not isinstance(summary, dict):
        return []
    rows = summary.get("orders") or summary.get("submitted_orders") or summary.get("fills") or []
    if not isinstance(rows, list):
        return []
    return [asdict(_trade_from_dict(item)) for item in rows if isinstance(item, dict)][:50]


def _holding_order_rows(holdings: list[HoldingDashboardRow], logs: dict[str, Any]) -> list[dict[str, Any]]:
    journal = logs.get("live_order_journal")
    if not isinstance(journal, dict):
        return [asdict(_holding_order_from_holding(row, None)) for row in holdings]
    events = []
    for key in ("recent_orders", "submitted_orders", "recent_executions"):
        for item in journal.get(key) or []:
            if isinstance(item, dict):
                events.append(item)
    latest_by_ticker: dict[str, dict[str, Any]] = {}
    latest_key_by_ticker: dict[str, tuple[datetime, int, str]] = {}
    for item in events:
        ticker = str(item.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        occurred_at = _parse_time(item.get("occurred_at") or item.get("recorded_at") or item.get("submitted_at") or item.get("filled_at")) or datetime.min.replace(tzinfo=timezone.utc)
        key = (occurred_at, _order_event_priority(item), str(item.get("broker_order_id") or item.get("order_id") or ""))
        if ticker not in latest_key_by_ticker or key >= latest_key_by_ticker[ticker]:
            latest_key_by_ticker[ticker] = key
            latest_by_ticker[ticker] = item
    rows: list[HoldingOrderStatusRow] = []
    for holding in holdings:
        rows.append(_holding_order_from_holding(holding, latest_by_ticker.get(holding.ticker.upper())))
    return [asdict(row) for row in rows]


def _holding_order_from_holding(
    holding: HoldingDashboardRow,
    event: dict[str, Any] | None,
) -> HoldingOrderStatusRow:
    if event is None:
        return HoldingOrderStatusRow(
            ticker=holding.ticker,
            name=holding.name,
            market_group=holding.market_group,
            market=holding.market,
            exchange=holding.exchange,
            currency=holding.currency,
            quantity=holding.quantity,
            average_price=holding.average_price,
            current_price=holding.current_price,
            order_state="주문 없음",
            order_status="NO_ORDER",
            order_action="",
            order_type="",
            order_id="",
            order_quantity=0.0,
            filled_quantity=0.0,
            order_price=0.0,
            order_summary="현재 걸린 주문 없음",
            occurred_at=holding.updated_at,
            source="holding_snapshot",
        )
    order_status = str(event.get("status") or event.get("order_status") or "").upper()
    event_type = str(event.get("event_type") or "")
    order_action = str(event.get("side") or "").upper()
    order_quantity = _num(event.get("quantity") or event.get("ordered_quantity"))
    filled_quantity = _num(event.get("filled_quantity"))
    order_price = _num(event.get("limit_price") or event.get("average_fill_price"))
    order_state = _humanize_order_state(event_type, order_status)
    order_summary = _build_order_summary(order_action, order_state, order_quantity, filled_quantity, order_price)
    return HoldingOrderStatusRow(
        ticker=holding.ticker,
        name=holding.name,
        market_group=holding.market_group,
        market=holding.market,
        exchange=holding.exchange,
        currency=holding.currency,
        quantity=holding.quantity,
        average_price=holding.average_price,
        current_price=holding.current_price,
        order_state=order_state,
        order_status=order_status or event_type.upper() or "UNKNOWN",
        order_action=order_action,
        order_type=str(event.get("order_type") or ""),
        order_id=str(event.get("order_id") or event.get("broker_order_id") or ""),
        order_quantity=order_quantity,
        filled_quantity=filled_quantity,
        order_price=order_price,
        order_summary=order_summary,
        occurred_at=str(event.get("occurred_at") or event.get("recorded_at") or event.get("submitted_at") or event.get("filled_at") or holding.updated_at),
        source=str(event.get("source") or event.get("event_type") or "live_order_journal"),
    )


def _order_event_priority(item: dict[str, Any]) -> int:
    event_type = str(item.get("event_type") or "").lower()
    status = str(item.get("status") or item.get("order_status") or "").upper()
    if event_type.endswith("amended"):
        return 5
    if status in {"FILLED"}:
        return 4
    if status in {"PARTIALLY_FILLED"}:
        return 3
    if event_type.endswith("submitted"):
        return 2
    if status in {"ACCEPTED", "OPEN", "WORKING"}:
        return 1
    return 0


def _humanize_order_state(event_type: str, order_status: str) -> str:
    event_type = str(event_type or "").lower()
    order_status = str(order_status or "").upper()
    if event_type.endswith("amended"):
        return "정정"
    if order_status == "PARTIALLY_FILLED":
        return "일부 체결"
    if order_status == "FILLED":
        return "체결"
    if event_type.endswith("submitted"):
        return "주문"
    if order_status in {"CANCELED", "CANCELLED"}:
        return "취소"
    if order_status in {"REJECTED", "BLOCKED"}:
        return "차단"
    if order_status:
        return order_status
    return "주문 상태 없음"


def _build_order_summary(order_action: str, order_state: str, order_quantity: float, filled_quantity: float, order_price: float) -> str:
    parts = [part for part in (order_action, order_state) if part]
    quantity_text = f"{order_quantity:.0f}주" if order_quantity > 0 else ""
    filled_text = f"체결 {filled_quantity:.0f}주" if filled_quantity > 0 else ""
    price_text = f"@ {order_price:,.0f}" if order_price > 0 else ""
    tail = " · ".join(part for part in (quantity_text, filled_text, price_text) if part)
    if tail:
        parts.append(tail)
    return " / ".join(parts) if parts else "현재 걸린 주문 없음"


def _trade_from_dict(item: dict[str, Any]) -> TradeHistoryRow:
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    status = str(item.get("order_status") or item.get("status") or raw.get("status") or "").upper()
    ordered_quantity = _num(item.get("ordered_quantity") or item.get("quantity"))
    filled_quantity = _num(item.get("filled_quantity") or item.get("filled_qty"))
    if filled_quantity <= 0 and status in {"FILLED", "PARTIALLY_FILLED"}:
        filled_quantity = _num(raw.get("quantity") or item.get("quantity"))
    price = _num(item.get("average_fill_price") or item.get("price") or raw.get("price") or item.get("limit_price"))
    amount = _num(item.get("amount_krw") or item.get("notional") or item.get("filled_amount") or raw.get("executed_value"))
    if amount <= 0 and filled_quantity > 0 and price > 0:
        amount = filled_quantity * price
    ticker = str(item.get("ticker") or raw.get("ticker") or "")
    market = str(item.get("market") or raw.get("market") or "")
    side = item.get("side") or raw.get("side") or ""
    side_value = getattr(side, "value", side)
    currency = str(item.get("currency") or ("USD" if _is_us_market(market, ticker) else "KRW")).upper()
    return TradeHistoryRow(
        occurred_at=str(item.get("occurred_at") or item.get("submitted_at") or item.get("filled_at") or item.get("recorded_at") or raw.get("executed_at") or datetime.now(timezone.utc).isoformat()),
        market_group=str(item.get("market_group") or ("domestic" if currency == "KRW" else "overseas")),
        market=market,
        exchange=str(item.get("exchange") or market),
        ticker=ticker,
        name=str(item.get("name") or ticker),
        side=str(side_value or ""),
        order_type=str(item.get("order_type") or ""),
        order_id=str(item.get("order_id") or item.get("broker_order_id") or raw.get("order_id") or ""),
        order_status=status,
        ordered_quantity=ordered_quantity,
        filled_quantity=filled_quantity,
        average_fill_price=price,
        amount_original=_num(item.get("amount_original") or amount),
        amount_krw=amount,
        fee_krw=_num(item.get("fee_krw")),
        tax_krw=_num(item.get("tax_krw")),
        realized_pnl_krw=_num(item.get("realized_pnl_krw")),
        currency=currency,
        source=str(item.get("source") or item.get("event_type") or "live_execution_summary"),
    )


def _dedupe_trade_rows(rows: list[TradeHistoryRow]) -> list[TradeHistoryRow]:
    seen: set[tuple[str, str, str, str, str, float, float]] = set()
    deduped: list[TradeHistoryRow] = []
    for row in rows:
        key = (
            row.source,
            row.order_id,
            row.order_status,
            row.ticker,
            row.side,
            row.ordered_quantity,
            row.filled_quantity,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _is_us_market(market: str, ticker: str) -> bool:
    upper_market = str(market or "").upper()
    if upper_market in {"US", "NASDAQ", "NASD", "NYSE", "AMEX", "OVERSEAS"}:
        return True
    return bool(str(ticker or "").strip()) and not str(ticker or "").isdigit()


def _profitability_summary(trades: list[dict[str, Any]], snapshot: AccountDashboardSnapshot) -> dict[str, Any]:
    """Derive win-rate / payoff / expectancy from realized-trade rows.

    Uses the same trade rows that feed the trade table. Values that cannot be
    computed from available data are returned as ``None`` so the front-end can
    show ``n/a`` instead of a fabricated number. No trading logic here — this is
    a pure read-model over already-realized fills.
    """
    realized = [_num(t.get("realized_pnl_krw")) for t in trades if _num(t.get("realized_pnl_krw")) != 0.0]
    wins = [v for v in realized if v > 0]
    losses = [v for v in realized if v < 0]
    closed = len(realized)
    fees = sum(_num(t.get("fee_krw")) for t in trades)
    tax = sum(_num(t.get("tax_krw")) for t in trades)
    win_rate = (len(wins) / closed) if closed else None
    avg_win = (sum(wins) / len(wins)) if wins else None
    avg_loss = (sum(losses) / len(losses)) if losses else None  # negative
    payoff = (avg_win / abs(avg_loss)) if (avg_win is not None and avg_loss not in (None, 0.0)) else None
    if win_rate is not None and avg_win is not None and avg_loss is not None:
        expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    else:
        expectancy = None
    return {
        "closed_trade_count": closed,
        "win_count": len(wins),
        "loss_count": len(losses),
        "gross_realized_pnl_krw": sum(realized),
        "realized_pnl_today_krw": snapshot.realized_pnl_today_krw,
        "unrealized_pnl_krw": snapshot.unrealized_pnl_krw,
        "trade_cost_krw": fees + tax,
        "fees_krw": fees,
        "tax_krw": tax,
        "net_after_cost_krw": snapshot.realized_pnl_today_krw - (fees + tax),
        "win_rate": win_rate,
        "avg_win_krw": avg_win,
        "avg_loss_krw": avg_loss,
        "payoff_ratio": payoff,
        "expectancy_krw": expectancy,
    }


def _warnings(status: dict[str, Any], logs: dict[str, Any], is_stale: bool) -> list[str]:
    warnings: list[str] = []
    if is_stale:
        warnings.append("ACCOUNT_DATA_STALE")
    if status.get("last_error"):
        warnings.append(f"ACCOUNT_STATUS_ERROR:{status['last_error']}")
    if logs.get("last_error"):
        warnings.append(f"LOG_ERROR:{logs['last_error']}")
    if not status:
        warnings.append("ACCOUNT_STATUS_EMPTY")
    return warnings


def _num(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Technical prediction panel (advisory GUI). Pure transform of the latest
# per-symbol technical decision diagnostics into a render-ready structure.
# ---------------------------------------------------------------------------
_REJECT_CARD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("high_volatility", ("HIGH_VOLATILITY_TECHNICAL_BLOCK", "HIGH_VOLATILITY_RISK")),
    ("low_liquidity", ("LOW_LIQUIDITY_TECHNICAL_BLOCK", "LIQUIDITY_TOO_LOW", "LOW_LIQUIDITY_RISK")),
    ("spread_consumes_alpha", ("SPREAD_CONSUMES_TECHNICAL_ALPHA", "SPREAD_CONSUMES_ALPHA", "SPREAD_TOO_WIDE")),
    ("model_feature_unavailable", ("MODEL_UNAVAILABLE", "MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE", "MISSING_MARKET_DATA", "TECHNICAL_SIGNAL_UNAVAILABLE")),
    ("no_ontology_support", ("ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK", "ONTOLOGY_BELOW_ADAPTIVE_THRESHOLD")),
    ("below_net_edge", ("PROFITABILITY_GATE_REJECTED", "BELOW_TARGET_NET_RETURN_AFTER_COST", "BELOW_BREAK_EVEN_WITH_MARGIN", "TECHNICAL_EDGE_NON_POSITIVE", "BUY_SIGNAL_TOO_WEAK")),
)


def _reject_reason(reason_codes: list[str]) -> str:
    codes = set(reason_codes or ())
    for label, triggers in _REJECT_CARD_RULES:
        if codes & set(triggers):
            return label
    return "other"


def _technical_card(decision: dict[str, Any]) -> dict[str, Any]:
    tech = dict(decision.get("technical") or decision.get("technical_prediction") or {})
    prof = dict(decision.get("profitability") or decision.get("profitability_decision") or {})
    reason_codes = [str(c) for c in (decision.get("reason_codes") or ())]
    action = str(decision.get("action") or ("BUY" if decision.get("approved") else "HOLD")).upper()
    approved = bool(decision.get("approved"))
    if action in ("SELL", "REDUCE"):
        category = action.lower()
    elif approved:
        category = "buy_approved"
    elif action == "BUY" or reason_codes:
        category = "buy_rejected"
    else:
        category = "hold"
    return {
        "symbol": decision.get("symbol"),
        "category": category,
        "reject_reason": _reject_reason(reason_codes) if category == "buy_rejected" else None,
        "regime": decision.get("technical_regime") or tech.get("regime"),
        "methodology": decision.get("technical_methodology") or tech.get("methodology"),
        "expected_edge_bps": tech.get("expected_net_return_bps"),
        "expected_horizon_seconds": tech.get("expected_horizon_seconds"),
        "expected_exit_price": tech.get("expected_exit_price"),
        "downside_risk_bps": tech.get("downside_risk_bps"),
        "confidence": tech.get("confidence"),
        "vwap_distance_bps": (tech.get("diagnostics") or {}).get("vwap_distance_bps"),
        "net_expected_return": prof.get("net_expected_return"),
        "required_min_net_return": prof.get("required_min_net_return"),
        "spread_rate": prof.get("spread_rate"),
        "cost_to_alpha_ratio": prof.get("cost_to_alpha_ratio"),
        "gate_allowed": prof.get("allowed"),
        "reason_codes": reason_codes,
        "explanation": tech.get("explanation"),
    }


def build_technical_panel(decisions: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Transform latest technical decision diagnostics into a GUI panel payload.

    Advisory only: this describes why the (authoritative) gates approved,
    rejected, held, sold, or reduced — it never itself decides anything.
    """
    cards = [_technical_card(d) for d in (decisions or []) if isinstance(d, dict)]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        by_category.setdefault(card["category"], []).append(card)
    return {
        "available": bool(cards),
        "count": len(cards),
        "cards": cards,
        "buy_approved": by_category.get("buy_approved", []),
        "buy_rejected": by_category.get("buy_rejected", []),
        "sell": by_category.get("sell", []),
        "reduce": by_category.get("reduce", []),
        "hold": by_category.get("hold", []),
    }


# ---------------------------------------------------------------------------
# Macro–micro ontology panel (advisory GUI). Pure transform of the latest
# MacroMicroReasoningBundle dict into a render-ready structure. Describes the
# market regime, candidate selection, and per-symbol micro reasoning; it never
# itself decides anything (the gates remain authoritative).
# ---------------------------------------------------------------------------
def build_macro_micro_panel(bundle: dict[str, Any] | None) -> dict[str, Any]:
    if not bundle:
        return {"available": False, "market_regime": None, "micro": [], "ranked_intents": []}
    macro = dict(bundle.get("macro_result") or {})
    micro_rows = []
    for m in bundle.get("micro_results") or []:
        micro_rows.append({
            "symbol": m.get("symbol"),
            "micro_regime": m.get("micro_regime"),
            "selected_strategy": m.get("selected_strategy"),
            "entry_signal": m.get("entry_signal"),
            "exit_signal": m.get("exit_signal"),
            "expected_exit_price": m.get("expected_exit_price"),
            "expected_net_return_bps": m.get("expected_net_return_bps"),
            "downside_risk_bps": m.get("downside_risk_bps"),
            "execution_quality": m.get("execution_quality"),
            "confidence": m.get("confidence"),
            "reason_codes": list(m.get("reason_codes") or []),
        })
    ranked = [
        {
            "rank": i.get("rank"), "symbol": i.get("symbol"), "side": i.get("side"),
            "expected_net_return_bps": i.get("expected_net_return_bps"),
            "micro_regime": i.get("micro_regime"), "selected_strategy": i.get("selected_strategy"),
            "confidence": i.get("confidence"),
        }
        for i in (bundle.get("ranked_trade_intents") or [])
    ]
    macro_reason_codes = [str(c) for c in (macro.get("reason_codes") or [])]
    # No live market data anywhere -> every symbol lacks a price series so the
    # macro regime is NO_TRADE_MARKET(insufficient data) and micro is unavailable.
    # Surface this explicitly so the panel reads as "awaiting feed" rather than
    # a wall of blocked rows.
    micro_all_unavailable = bool(micro_rows) and all(
        "MICRO_SIGNAL_UNAVAILABLE" in (r.get("reason_codes") or []) for r in micro_rows
    )
    no_live_data = "MACRO_INSUFFICIENT_DATA" in macro_reason_codes or (
        (macro.get("macro_confidence") or 0) == 0 and (not micro_rows or micro_all_unavailable)
    )
    return {
        "available": True,
        "data_status": "no_live_data" if no_live_data else "live",
        "timestamp": bundle.get("timestamp"),
        "market_regime": macro.get("market_regime"),
        "macro_risk_level": macro.get("macro_risk_level"),
        "macro_confidence": macro.get("macro_confidence"),
        "macro_reason_codes": macro_reason_codes,
        "blocks_buy": macro.get("blocks_buy"),
        "sector_rankings": macro.get("sector_rankings") or [],
        "candidate_symbols": macro.get("candidate_symbols") or [],
        "allowed_micro_strategies": macro.get("allowed_micro_strategies") or [],
        "blocked_micro_strategies": macro.get("blocked_micro_strategies") or [],
        "micro": micro_rows,
        "ranked_intents": ranked,
        "sell_reduce_candidates": list(bundle.get("sell_reduce_candidates") or []),
        "buy_candidates": list(bundle.get("buy_candidates") or []),
        "blocked_candidates": list(bundle.get("blocked_candidates") or []),
        "failed_symbols": list(bundle.get("failed_symbols") or []),
    }
