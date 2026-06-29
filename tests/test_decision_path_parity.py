from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REALTIME_SOURCE, OrderbookLevel, RealtimeOrderbookSnapshot, RealtimeTradeTick
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.live_signal_predictor import LiveSignalPredictor
from app.models.model_artifact_registry import ModelArtifactRegistry
from app.schemas.domain import AccountSnapshot, PrincipalProtectionConfig, RiskRules
from app.risk import RiskManager
from app.trading.shared_decision_engine import SharedLiveDecisionEngine
from tests.test_model_training_artifacts import _rows


class DecisionPathParityTest(unittest.TestCase):
    def test_model_prediction_connects_to_risk_and_final_order(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(Path(tmp) / "models")
            train_live_short_horizon_model(_rows(), registry=registry)
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            _seed(store, now)
            risk = RiskManager(
                RiskRules(
                    min_average_daily_trading_value=1,
                    max_volatility=1.0,
                    minimum_cash_reserve=0.0,
                    principal_protection=PrincipalProtectionConfig(
                        initial_principal=1_000_000,
                        profit_lockin_ratio=0.0,
                        per_trade_risk_budget_ratio=0.05,
                        daily_risk_budget_ratio=0.05,
                    ),
                )
            )
            result = SharedLiveDecisionEngine(store, predictor=LiveSignalPredictor(registry), risk_manager=risk).evaluate_buy(
                "005930",
                AccountSnapshot(cash=10_000_000, holdings=(), realized_pnl_today=9_000_000),
                suggested_weight=0.10,
                decision_time=now,
            )

        self.assertTrue(result.approved, result.reason_codes)
        self.assertIsNotNone(result.final_order)


def _seed(store: RealtimeMarketDataStore, now: datetime) -> None:
    store.save_ticks(
        tuple(
            RealtimeTradeTick("005930", now - timedelta(seconds=120 - i * 10), now - timedelta(seconds=120 - i * 10), KIS_REALTIME_SOURCE, 70000 + i * 30, 1000, sequence_key=f"t{i}")
            for i in range(13)
        )
    )
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot("005930", now, now, KIS_REALTIME_SOURCE, (OrderbookLevel(70380, 500000, 70400, 100000),), sequence_key="b"),
        )
    )


if __name__ == "__main__":
    unittest.main()
