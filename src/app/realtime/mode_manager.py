from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from app.runtime import DataEnvironment


class OperationMode(StrEnum):
    LEARNING = "learning"
    TESTING = "testing"
    LIVE_TRADING = "live_trading"


@dataclass(frozen=True)
class OperationModeState:
    mode: OperationMode
    data_environment: str
    data_root: str
    model_root: str
    synthetic_data_allowed: bool
    live_orders_allowed: bool
    training_allowed: bool
    testing_allowed: bool
    execution_label: str
    guardrails: tuple[str, ...]
    started_at: datetime


class OperationModeManager:
    def start(self, mode: OperationMode | str) -> OperationModeState:
        selected = OperationMode(mode)
        env = DataEnvironment.realtime()
        return OperationModeState(
            mode=selected,
            data_environment=env.mode,
            data_root=str(env.root),
            model_root=str(env.model_dir),
            synthetic_data_allowed=False,
            live_orders_allowed=selected == OperationMode.LIVE_TRADING,
            training_allowed=selected == OperationMode.LEARNING,
            testing_allowed=selected == OperationMode.TESTING,
            execution_label={
                OperationMode.LEARNING: "Realtime learning with supervised PnL labels",
                OperationMode.TESTING: "Realtime hypothetical trading test",
                OperationMode.LIVE_TRADING: "Realtime trading gate",
            }[selected],
            guardrails=(
                "Use one unified realtime data store only: data/store.",
                "Synthetic and simulation data are not valid inputs for learning, testing, or live trading.",
                "Learning may update model artifacts under data/models/<model_family>/.",
                "Testing must not submit broker orders; it records hypothetical realized PnL only.",
                "Live trading is the only mode that may reach brokerage execution gates.",
            ),
            started_at=datetime.now(timezone.utc),
        )
