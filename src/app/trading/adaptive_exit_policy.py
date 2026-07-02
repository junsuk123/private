from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from app.cost import CostBreakdown, TradingCostEngine
from app.schemas.domain import AccountSnapshot, Holding, MarketSnapshot


@dataclass(frozen=True)
class AdaptiveExitPolicy:
    sell_target: float
    stop_loss: float
    trailing_stop: float
    time_exit_seconds: int
    confidence_floor: float
    min_expected_net_return: float
    allow_loss_exit: bool
    exit_mode: str
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_exit_policy(
    *,
    holding: Holding,
    account: AccountSnapshot,
    market: MarketSnapshot,
    take_profit: float,
    stop_loss: float,
    ontology_score: float,
    decision_time: datetime,
    target_net_return: float,
    cost_engine: TradingCostEngine | None = None,
) -> tuple[AdaptiveExitPolicy, CostBreakdown]:
    engine = cost_engine or TradingCostEngine()
    quantity = max(1, int(getattr(holding, "quantity", 0) or 0))
    venue = "KRX" if str(holding.market or market.market or "").upper().startswith("K") else "NASD"
    instrument_type = "domestic_stock" if venue == "KRX" else "overseas_stock"
    cost_floor = engine.estimate(
        symbol=holding.ticker,
        market=holding.market or market.market,
        venue=venue,
        instrument_type=instrument_type,
        entry_price=float(getattr(holding, "average_price", 0.0) or 0.0),
        expected_exit_price=float(getattr(holding, "last_price", 0.0) or 0.0),
        quantity=quantity,
        target_net_return=max(0.0, float(target_net_return)),
    )
    volatility = max(0.0, float(getattr(market, "volatility_20d", 0.0) or 0.0))
    held_age_seconds = 0.0
    opened_at = getattr(holding, "opened_at", None)
    if opened_at is not None:
        try:
            held_age_seconds = max(0.0, (decision_time - opened_at).total_seconds())
        except Exception:  # noqa: BLE001
            held_age_seconds = 0.0
    allow_loss_exit = os.getenv("REALTIME_ALLOW_LOSS_EXIT", "false").strip().lower() in {"1", "true", "yes", "on"}
    exit_mode = "profit_exit"
    if ontology_score <= -0.6:
        exit_mode = "risk_exit"
    elif ontology_score <= -0.1:
        exit_mode = "invalid_signal_exit"
    elif held_age_seconds >= 3600:
        exit_mode = "time_exit"

    base_average_price = max(0.01, float(getattr(holding, "average_price", 0.0) or 0.0))
    dynamic_profit = max(
        take_profit,
        cost_floor.required_exit_price / base_average_price - 1.0,
        max(0.0, target_net_return) + cost_floor.net_expected_return,
        0.0005 + volatility * 0.05,
    )
    dynamic_stop = max(0.0025, min(0.04, stop_loss + volatility * 0.1 + cost_floor.total_cost_rate * 0.5))
    trailing_stop = max(0.0015, min(dynamic_stop, dynamic_stop * 0.7 + volatility * 0.3))
    time_exit_seconds = max(120, min(7200, int(1200 + held_age_seconds * 0.4 + volatility * 10_000)))
    confidence_floor = max(0.35, min(0.85, 0.5 + volatility * 3.0 - cost_floor.total_cost_rate))
    policy = AdaptiveExitPolicy(
        sell_target=round(dynamic_profit, 6),
        stop_loss=round(dynamic_stop, 6),
        trailing_stop=round(trailing_stop, 6),
        time_exit_seconds=time_exit_seconds,
        confidence_floor=round(confidence_floor, 4),
        min_expected_net_return=round(max(max(0.0, target_net_return), cost_floor.net_expected_return), 6),
        allow_loss_exit=allow_loss_exit,
        exit_mode=exit_mode,
        diagnostics={
            "ontology_score": round(float(ontology_score), 4),
            "held_age_seconds": round(held_age_seconds, 2),
            "volatility": round(volatility, 4),
            "decision_time": decision_time.isoformat(),
        },
    )
    return policy, cost_floor
