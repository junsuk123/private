from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from app.runtime import DataEnvironment


class OperationMode(StrEnum):
    LEARNING = "learning"
    TESTING = "testing"
    PAPER_TRADING = "paper_trading"
    PAPER_TRADING_TEST = "paper_trading_test"
    LIVE_READINESS = "live_readiness"
    LIVE_TRADING_TEST = "live_trading_test"
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
    paper_trading_allowed: bool
    live_readiness_allowed: bool
    execution_label: str
    guardrails: tuple[str, ...]
    started_at: datetime

    @property
    def testing_allowed(self) -> bool:
        """Backward-compatible alias for older callers."""
        return self.paper_trading_allowed or self.live_readiness_allowed


class OperationModeManager:
    def start(self, mode: OperationMode | str) -> OperationModeState:
        requested = str(getattr(mode, "value", mode))
        if requested in {
            OperationMode.TESTING.value,
            OperationMode.PAPER_TRADING.value,
            OperationMode.PAPER_TRADING_TEST.value,
        }:
            requested = OperationMode.LIVE_TRADING.value
        selected = OperationMode(requested)
        env = DataEnvironment.realtime()
        return OperationModeState(
            mode=selected,
            data_environment=env.mode,
            data_root=str(env.root),
            model_root=str(env.model_dir),
            synthetic_data_allowed=False,
            live_orders_allowed=selected == OperationMode.LIVE_TRADING,
            training_allowed=selected in {OperationMode.LEARNING, OperationMode.LIVE_TRADING},
            paper_trading_allowed=False,
            live_readiness_allowed=selected in {
                OperationMode.LIVE_READINESS,
                OperationMode.LIVE_TRADING_TEST,
            },
            execution_label={
                OperationMode.LEARNING: "Realtime learning with supervised PnL labels",
                OperationMode.TESTING: "Legacy paper trading replay",
                OperationMode.PAPER_TRADING: "Deprecated; normalized to live trading",
                OperationMode.PAPER_TRADING_TEST: "Deprecated; normalized to live trading",
                OperationMode.LIVE_READINESS: "KIS live readiness check",
                OperationMode.LIVE_TRADING_TEST: "KIS live readiness check",
                OperationMode.LIVE_TRADING: "Realtime trading gate",
            }[selected],
            guardrails=(
                "Use one unified realtime data store only: data/store.",
                "Synthetic and simulation data are not valid inputs for learning or live trading.",
                "Learning and information collection continue while the server is running.",
                "Paper trading modes are removed and normalized to live trading.",
                "KIS live readiness may verify authentication but must not submit orders.",
                "Live trading is the only mode that may reach brokerage execution gates.",
            ),
            started_at=datetime.now(timezone.utc),
        )
