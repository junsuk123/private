from __future__ import annotations

from typing import Protocol

from app.schemas.domain import AccountSnapshot, OrderIntent, PortfolioStatusReport


class PortfolioAgent(Protocol):
    def analyze(self, account: AccountSnapshot) -> PortfolioStatusReport:
        """Return a structured portfolio report without executing orders."""


class StrategyAgent(Protocol):
    def propose_orders(self) -> tuple[OrderIntent, ...]:
        """Return structured order intents for deterministic risk validation."""
