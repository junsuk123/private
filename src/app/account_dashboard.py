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
        store: AccountSnapshotStore | None = None,
    ) -> None:
        self.status_provider = status_provider
        self.logs_provider = logs_provider
        self.store = store or AccountSnapshotStore()

    def build_dashboard(self, *, persist: bool = True) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        previous_dashboard = self.store.latest_dashboard() or {}
        previous_snapshot = dict(previous_dashboard.get("snapshot") or {})
        status = self._status_payload()
        logs = self._logs_payload()
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
        if total_asset_krw <= 0 and previous_snapshot:
            total_asset_krw = _num(previous_snapshot.get("total_asset_krw") or previous_snapshot.get("net_asset_krw"))
        if cash_equivalent_krw <= 0 and previous_snapshot:
            cash_equivalent_krw = _num(previous_snapshot.get("cash_equivalent_krw"))
        if krw_cash <= 0 and previous_snapshot:
            krw_cash = _num(previous_snapshot.get("krw_cash"))
        if foreign_cash_krw <= 0 and previous_snapshot:
            foreign_cash_krw = _num(previous_snapshot.get("foreign_cash_krw"))
        if settlement_cash_krw <= 0 and previous_snapshot:
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
            is_live=bool(status.get("account_checked") or status.get("basis_source") == "kis_live_account"),
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
        dashboard = {
            "snapshot": asdict(snapshot),
            "holdings": [asdict(row) for row in holdings],
            "cash": [asdict(row) for row in cash_rows],
            "trades": trade_rows,
            "profitability": _profitability_summary(trade_rows, snapshot),
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

    def holdings(self) -> list[dict[str, Any]]:
        return list(self.build_dashboard(persist=False).get("holdings") or [])

    def cash(self) -> list[dict[str, Any]]:
        return list(self.build_dashboard(persist=False).get("cash") or [])

    def trades(self) -> list[dict[str, Any]]:
        return list(self.build_dashboard(persist=False).get("trades") or [])

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
) -> float:
    venue = _cost_venue_for_position(market_group=market_group, market=market, exchange=exchange, ticker=ticker)
    instrument_type = "domestic_stock" if currency == "KRW" else "overseas_stock"
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
