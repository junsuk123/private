from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionPolicy:
    buy_threshold: float
    sell_target: float
    stop_loss: float
    trailing_stop: float
    max_position_size: int
    quote_ttl_seconds: float
    min_expected_net_return: float
    max_spread_bps: float
    max_slippage_bps: float
    allowed_fallback_mode: str
    order_type: str = "LIMIT"
    time_exit_seconds: int = 300
    confidence_floor: float = 0.5
    risk_mode: str = "normal"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
