from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, OrderSide


@dataclass(frozen=True)
class MockKisOrderReceipt:
    order_id: str
    accepted: bool
    status: str
    message: str
    order: FinalOrder
    submitted_at: datetime


@dataclass(frozen=True)
class MockKisExecution:
    order_id: str
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    executed_value: float
    status: str
    message: str
    executed_at: datetime


@dataclass(frozen=True)
class MockKisPortfolio:
    account: AccountSnapshot
    market_prices: dict[str, float]
    updated_at: datetime


class MockKisDevelopersApi:
    def __init__(
        self,
        account: AccountSnapshot,
        market_prices: dict[str, float],
        sectors: dict[str, str] | None = None,
        company_names: dict[str, str] | None = None,
    ) -> None:
        self.account = account
        self.market_prices = dict(market_prices)
        self.sectors = sectors or {}
        self.company_names = company_names or {}
        self._orders: dict[str, MockKisOrderReceipt] = {}
        self._executions: dict[str, MockKisExecution] = {}

    def place_limit_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        order_id = f"KISMOCK-{uuid4().hex[:12]}"
        receipt = MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="ACCEPTED",
            message="Mock KIS accepted the limit order.",
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )
        self._orders[order_id] = receipt
        self._executions[order_id] = self._match_limit_order(order_id, order)
        return receipt

    def get_order_status(self, order_id: str) -> MockKisExecution:
        if order_id not in self._executions:
            raise KeyError(f"unknown mock KIS order_id: {order_id}")
        return self._executions[order_id]

    def amend_limit_order(self, order_id: str, replacement: FinalOrder) -> MockKisOrderReceipt:
        if order_id not in self._orders:
            raise KeyError(f"unknown mock KIS order_id: {order_id}")
        receipt = MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="ACCEPTED",
            message="Mock KIS accepted the limit order amendment.",
            order=replacement,
            submitted_at=datetime.now(timezone.utc),
        )
        self._orders[order_id] = receipt
        self._executions[order_id] = self._match_limit_order(order_id, replacement)
        return receipt

    def cancel_order(self, order_id: str, order: FinalOrder) -> MockKisOrderReceipt:
        if order_id not in self._orders:
            raise KeyError(f"unknown mock KIS order_id: {order_id}")
        receipt = MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="CANCELED",
            message="Mock KIS canceled the limit order.",
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )
        self._orders[order_id] = receipt
        self._executions[order_id] = MockKisExecution(
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=0,
            price=order.limit_price,
            executed_value=0.0,
            status="CANCELED",
            message="Mock KIS canceled the limit order.",
            executed_at=datetime.now(timezone.utc),
        )
        return receipt

    def get_portfolio(self) -> MockKisPortfolio:
        return MockKisPortfolio(
            account=self.account,
            market_prices=dict(self.market_prices),
            updated_at=datetime.now(timezone.utc),
        )

    def list_orders(self) -> tuple[MockKisOrderReceipt, ...]:
        return tuple(self._orders.values())

    def list_executions(self) -> tuple[MockKisExecution, ...]:
        return tuple(self._executions.values())

    def _match_limit_order(self, order_id: str, order: FinalOrder) -> MockKisExecution:
        market_price = self.market_prices.get(order.ticker, order.limit_price)
        can_fill = (
            order.side == OrderSide.BUY
            and order.limit_price >= market_price
            or order.side == OrderSide.SELL
            and order.limit_price <= market_price
        )
        if not can_fill:
            return MockKisExecution(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=0,
                price=market_price,
                executed_value=0.0,
                status="OPEN",
                message="Limit price has not crossed the mock market price.",
                executed_at=datetime.now(timezone.utc),
            )

        if order.side == OrderSide.BUY:
            return self._fill_buy(order_id, order, market_price)
        return self._fill_sell(order_id, order, market_price)

    def _fill_buy(self, order_id: str, order: FinalOrder, price: float) -> MockKisExecution:
        value = order.quantity * price
        if value > self.account.cash:
            return MockKisExecution(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=0,
                price=price,
                executed_value=0.0,
                status="REJECTED",
                message="Insufficient mock cash.",
                executed_at=datetime.now(timezone.utc),
            )

        existing = {holding.ticker: holding for holding in self.account.holdings}
        current = existing.get(order.ticker)
        if current is None:
            existing[order.ticker] = Holding(
                ticker=order.ticker,
                market=order.market,
                company_name=self.company_names.get(order.ticker, order.ticker),
                sector=self.sectors.get(order.ticker, "Unknown"),
                quantity=order.quantity,
                average_price=price,
                last_price=price,
                opened_at=datetime.now(timezone.utc),
            )
        else:
            total_quantity = current.quantity + order.quantity
            average_price = (
                current.average_price * current.quantity + value
            ) / max(1, total_quantity)
            existing[order.ticker] = replace(
                current,
                quantity=total_quantity,
                average_price=average_price,
                last_price=price,
            )

        self.account = replace(
            self.account,
            cash=self.account.cash - value,
            holdings=tuple(existing.values()),
            captured_at=datetime.now(timezone.utc),
        )
        return MockKisExecution(
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=price,
            executed_value=value,
            status="FILLED",
            message="Mock KIS filled the buy limit order.",
            executed_at=datetime.now(timezone.utc),
        )

    def _fill_sell(self, order_id: str, order: FinalOrder, price: float) -> MockKisExecution:
        holdings = {holding.ticker: holding for holding in self.account.holdings}
        current = holdings.get(order.ticker)
        if current is None or current.quantity <= 0:
            return MockKisExecution(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=0,
                price=price,
                executed_value=0.0,
                status="REJECTED",
                message="No mock holding to sell.",
                executed_at=datetime.now(timezone.utc),
            )

        quantity = min(order.quantity, current.quantity)
        value = quantity * price
        remaining = current.quantity - quantity
        if remaining > 0:
            holdings[order.ticker] = replace(current, quantity=remaining, last_price=price)
        else:
            del holdings[order.ticker]

        self.account = replace(
            self.account,
            cash=self.account.cash + value,
            holdings=tuple(holdings.values()),
            captured_at=datetime.now(timezone.utc),
        )
        return MockKisExecution(
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=quantity,
            price=price,
            executed_value=value,
            status="FILLED",
            message="Mock KIS filled the sell limit order.",
            executed_at=datetime.now(timezone.utc),
        )
