from __future__ import annotations

from dataclasses import dataclass

from app.trading.contracts import Position


class PositionOwnershipError(RuntimeError):
    pass


@dataclass(frozen=True)
class OwnershipGuard:
    """Enforces that only the origin strategy instance manages a position."""

    def assert_owner(self, position: Position, strategy_instance_id: str) -> None:
        if position.strategy_instance_id != strategy_instance_id:
            raise PositionOwnershipError(
                "POSITION_OWNED_BY_DIFFERENT_STRATEGY_INSTANCE:"
                f"{position.strategy_instance_id}"
            )

    def assert_strategy(self, position: Position, strategy_id: str) -> None:
        if position.origin_strategy_id != strategy_id:
            raise PositionOwnershipError(
                f"POSITION_ORIGIN_STRATEGY_MISMATCH:{position.origin_strategy_id}"
            )
