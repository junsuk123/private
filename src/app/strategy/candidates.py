from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.schemas.domain import OrderAction, OrderIntent


@dataclass(frozen=True)
class StrategyCandidate:
    """A strategy-produced opportunity, not an executable order."""

    ticker: str
    strategy_family: str
    signal_name: str
    entry_price: float
    expected_exit_price: float
    expected_holding_minutes: int
    gross_expected_return: float
    confidence: float
    features: dict[str, float] = field(default_factory=dict)
    ontology_tags: list[str] = field(default_factory=list)
    validation_id: str | None = None
    reason: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.expected_exit_price <= 0:
            raise ValueError("expected_exit_price must be positive")
        if self.expected_holding_minutes <= 0:
            raise ValueError("expected_holding_minutes must be positive")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        """Return the candidate as a dictionary."""
        return {
            "ticker": self.ticker,
            "strategy_family": self.strategy_family,
            "signal_name": self.signal_name,
            "entry_price": self.entry_price,
            "expected_exit_price": self.expected_exit_price,
            "expected_holding_minutes": self.expected_holding_minutes,
            "gross_expected_return": self.gross_expected_return,
            "confidence": self.confidence,
            "features": self.features,
            "ontology_tags": self.ontology_tags,
            "validation_id": self.validation_id,
            "reason": self.reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_order_intent(
        self,
        *,
        market: str,
        action: OrderAction = OrderAction.BUY,
        suggested_weight: float,
        valid_until: datetime,
        source_data_ids: tuple[str, ...],
        target_net_return: float = 0.0,
        model_uncertainty: float | None = None,
    ) -> OrderIntent:
        """Convert a candidate into an OrderIntent for RiskManager review."""
        return OrderIntent(
            ticker=self.ticker,
            market=market,
            action=action,
            suggested_weight=suggested_weight,
            confidence=self.confidence,
            valid_until=valid_until,
            reasoning_summary=(self.reason or f"{self.strategy_family}:{self.signal_name}",),
            supporting_factors=tuple(self.ontology_tags),
            contradicting_factors=(),
            source_data_ids=source_data_ids,
            model_uncertainty=model_uncertainty,
            strategy_family=self.strategy_family,
            signal_name=self.signal_name,
            expected_exit_price=self.expected_exit_price,
            expected_holding_minutes=self.expected_holding_minutes,
            gross_expected_return=self.gross_expected_return,
            target_net_return=target_net_return,
            validation_id=self.validation_id,
            ontology_tags=tuple(self.ontology_tags),
            strategy_metadata={"features": dict(self.features), "entry_price": self.entry_price},
        )
