from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.schemas.domain import AccountSnapshot, Holding, MarketSnapshot, OrderAction, PrincipalProtectionConfig, SourceMetadata
from app.trading.auto_tuning_engine import AutoTuningEngine
from app.trading.adaptive_exit_policy import derive_exit_policy


class _Graph:
    def __init__(self, supports: tuple[str, ...]) -> None:
        self.supports = supports

    def matching(self, subject=None, predicate=None):
        if predicate == "supportsSignal":
            return [SimpleNamespace(object=item) for item in self.supports]
        if predicate == "increasesRiskOf":
            return []
        return []


class AutoTuningEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        now = datetime.now(timezone.utc)
        self.market = MarketSnapshot(
            ticker="005930",
            market="KR",
            company_name="Samsung",
            sector="Tech",
            last_price=100.0,
            average_daily_trading_value=10_000_000_000,
            volatility_20d=0.02,
            source=SourceMetadata(
                source_name="unit",
                retrieved_at=now,
                observed_at=now,
                source_type="broker_api",
                trust_level=5,
                quality_score=1.0,
                is_realtime=True,
            ),
        )
        self.account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0}, cash_equivalent_krw=1_000_000.0)
        self.engine = AutoTuningEngine()

    def test_fallback_buy_score_needs_support(self) -> None:
        weak = self.engine.fallback_buy_score(
            ontology_score=0.0,
            technical_momentum=0.0,
            liquidity_score=0.2,
            spread_bps=30.0,
            volatility=0.02,
            recent_performance=0.0,
        )
        strong = self.engine.fallback_buy_score(
            ontology_score=2.0,
            technical_momentum=0.3,
            liquidity_score=0.9,
            spread_bps=5.0,
            volatility=0.01,
            recent_performance=0.02,
        )

        self.assertLess(weak, 0.5)
        self.assertGreater(strong, weak)

    def test_derive_exit_policy_keeps_dynamic_sell_target_reasonable(self) -> None:
        holding = Holding(ticker="005930", market="KR", company_name="Samsung", sector="Tech", quantity=10, average_price=100.0, last_price=100.0)
        policy, cost_floor = derive_exit_policy(
            holding=holding,
            account=self.account,
            market=self.market,
            take_profit=0.006,
            stop_loss=0.01,
            ontology_score=0.0,
            decision_time=datetime.now(timezone.utc),
            target_net_return=0.0015,
        )

        self.assertGreater(policy.sell_target, 0.0)
        self.assertLess(policy.sell_target, 0.02)
        self.assertGreaterEqual(cost_floor.required_exit_price, holding.average_price)

    def test_stale_quote_refresh_attempted_before_reject(self) -> None:
        stale_tick_time = datetime.now(timezone.utc) - timedelta(seconds=90)

        class Store:
            def latest_tick(self, symbol: str):
                return SimpleNamespace(price=99.0, received_at=stale_tick_time, exchange_timestamp=stale_tick_time, sequence_key="stale")

            def latest_orderbook(self, symbol: str):
                return None

        refreshed_market = MarketSnapshot(
            ticker="LAB",
            market="NASD",
            company_name="LAB",
            sector="Unknown",
            last_price=100.0,
            average_daily_trading_value=2_000_000_000,
            volatility_20d=0.015,
            source=SourceMetadata(
                source_name="KIS broker quote",
                retrieved_at=datetime.now(timezone.utc),
                observed_at=datetime.now(timezone.utc),
                source_type="broker_api",
                trust_level=5,
                quality_score=1.0,
                is_realtime=True,
            ),
        )

        class DummyPredictor:
            def predict(self, frame):
                raise RuntimeError("model down")

        engine = AutoTuningEngine()
        from app.trading.shared_decision_engine import SharedLiveDecisionEngine

        decision_engine = SharedLiveDecisionEngine(
            Store(),
            predictor=DummyPredictor(),
            market_refresher=lambda symbol, market, decision_time: refreshed_market,
        )
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"USD": 1_000_000.0}, cash_equivalent_krw=130_000_000.0)
        result = decision_engine.evaluate_buy("LAB", account, suggested_weight=0.01, ontology_graph=_Graph(("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying")))

        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.diagnostics.get("quote_refresh_status"), "quote_refresh_ok")


if __name__ == "__main__":
    unittest.main()
