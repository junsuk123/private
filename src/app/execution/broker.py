from __future__ import annotations

from typing import Protocol

from app.execution.kis_mock import MockKisExecution, MockKisOrderReceipt, MockKisPortfolio
from app.schemas.domain import FinalOrder


class BrokerClient(Protocol):
    def place_limit_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        """Submit a limit order and return the broker receipt."""

    def get_order_status(self, order_id: str) -> MockKisExecution:
        """Return the latest broker-side order or execution status."""

    def get_portfolio(self) -> MockKisPortfolio:
        """Return the broker-side portfolio snapshot."""
