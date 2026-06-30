from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.schemas.domain import AccountSnapshot, Holding, OrderSide
from app.execution.kis_types import LiveOrderSubmission
from app.schemas.domain import FinalOrder, OrderType
from app.trading.realtime_trading_engine import RealtimeTradingEngine, RealtimeTradingConfig
from app.trading.shared_decision_engine import SharedLiveDecisionEngine


class _FakeStore:
    """Minimal store exposing only latest_tick; recent_ticks intentionally absent
    so the model-exit path fails fast and falls back to HOLD."""

    def __init__(self, price: float) -> None:
        now = datetime.now(timezone.utc)
        self._tick = SimpleNamespace(
            price=price,
            received_at=now,
            exchange_timestamp=now,
            sequence_key=f"test:{price}",
        )

    def latest_tick(self, symbol: str):
        return self._tick


class _DummyPredictor:
    def predict(self, frame):  # pragma: no cover - not reached in these tests
        raise AssertionError("predictor should not be called for TP/SL exits")


def _engine(price: float) -> SharedLiveDecisionEngine:
    return SharedLiveDecisionEngine(_FakeStore(price), predictor=_DummyPredictor())


def _holding(avg: float, last_price: float | None = None) -> Holding:
    return Holding(
        ticker="005930",
        market="KR",
        company_name="Samsung",
        sector="Tech",
        quantity=10,
        average_price=avg,
        last_price=avg if last_price is None else last_price,
    )


class _FakeGraph:
    """Minimal KnowledgeGraph stand-in for _holding_exit_adjustment."""

    def __init__(self, risk_objects=(), support_objects=()) -> None:
        self._risk = tuple(risk_objects)
        self._support = tuple(support_objects)

    def matching(self, subject=None, predicate=None):
        if predicate == "increasesRiskOf":
            return [SimpleNamespace(object=o) for o in self._risk]
        if predicate == "supportsSignal":
            return [SimpleNamespace(object=o) for o in self._support]
        return []


def _account(holding: Holding, cash: float = 0.0) -> AccountSnapshot:
    # Low cash on purpose: de-risking sells must not be blocked by the cash reserve gate.
    return AccountSnapshot(cash=cash, holdings=(holding,))


class RealtimeExitDecisionTest(unittest.TestCase):
    def test_stop_loss_triggers_sell(self) -> None:
        engine = _engine(price=98.0)  # -2% vs avg 100, below 1% stop
        result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)
        self.assertTrue(result.approved, result.reason_codes)
        self.assertIsNotNone(result.final_order)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertEqual(result.final_order.quantity, 10)

    def test_take_profit_triggers_sell(self) -> None:
        engine = _engine(price=101.0)  # +1% vs avg 100, above 0.6% take-profit
        result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)

    def test_within_bands_holds(self) -> None:
        engine = _engine(price=100.2)  # +0.2%, inside both bands; model exit unavailable -> HOLD
        result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved)
        self.assertIsNone(result.final_order)
        self.assertIn("HOLD_WITHIN_BANDS", result.reason_codes)

    def test_no_tick_falls_back_to_broker_balance_mark(self) -> None:
        # 실시간 틱이 없어도 브로커 잔고가(last_price)로 손절을 판단해야 한다.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=98.0)  # -2% via broker mark, below 1% stop
        result = engine.evaluate_exit_for_holding(holding, _account(holding), take_profit=0.006, stop_loss=0.01)
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)

    def test_no_price_anywhere_returns_missing_market_data(self) -> None:
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=0.0)  # no tick and no broker mark
        result = engine.evaluate_exit_for_holding(holding, _account(holding))
        self.assertFalse(result.approved)
        self.assertIn("MISSING_MARKET_DATA", result.reason_codes)

    def test_ontology_risk_triggers_sell_within_bands(self) -> None:
        # 가격은 밴드 안(평단 근처)이지만 온톨로지가 매도 신호면 매도해야 한다.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=100.0)  # flat -> within TP/SL bands
        # 현금을 충분히 둬 포지션 비중을 작게 만들어 온톨로지 효과만 분리한다.
        account = _account(holding, cash=1_000_000.0)
        graph = _FakeGraph(risk_objects=("SellCandidate",))
        result = engine.evaluate_exit_for_holding(
            holding, account, take_profit=0.006, stop_loss=0.01, ontology_graph=graph
        )
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)

    def test_neutral_ontology_keeps_hold_within_bands(self) -> None:
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=100.0)
        account = _account(holding, cash=1_000_000.0)
        graph = _FakeGraph()  # no risk/support evidence
        result = engine.evaluate_exit_for_holding(
            holding, account, take_profit=0.006, stop_loss=0.01, ontology_graph=graph
        )
        self.assertFalse(result.approved)
        self.assertIn("HOLD_WITHIN_BANDS", result.reason_codes)


class _BuyStore:
    """Store exposing a fresh tick (and no orderbook) for the buy path."""

    def __init__(self, price: float) -> None:
        now = datetime.now(timezone.utc)
        self._tick = SimpleNamespace(price=price, received_at=now, exchange_timestamp=now, sequence_key=f"buy:{price}")

    def latest_tick(self, symbol: str):
        return self._tick

    def latest_orderbook(self, symbol: str):
        return None


class RealtimeBuyDecisionTest(unittest.TestCase):
    def test_ontology_drives_buy_when_model_unavailable(self) -> None:
        # 모델이 없어도(프레임 빌드 실패) 온톨로지 매수신호가 강하면 매수가 성립해야 한다.
        engine = SharedLiveDecisionEngine(_BuyStore(price=5.0), predictor=_DummyPredictor())
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0, "USD": 100000.0}, cash_equivalent_krw=130_000_000.0)
        graph = _FakeGraph(support_objects=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying"))
        result = engine.evaluate_buy("LAB", account, suggested_weight=0.01, ontology_graph=graph)
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.BUY)

    def test_no_ontology_and_no_model_rejects_buy(self) -> None:
        engine = SharedLiveDecisionEngine(_BuyStore(price=5.0), predictor=_DummyPredictor())
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0, "USD": 100000.0}, cash_equivalent_krw=130_000_000.0)
        graph = _FakeGraph()  # no buy-supportive evidence
        result = engine.evaluate_buy("LAB", account, suggested_weight=0.01, ontology_graph=graph)
        self.assertFalse(result.approved)


class _FixedSellDecisionEngine:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_exit_for_holding(self, holding, account, **kwargs):
        self.calls += 1
        order = FinalOrder(
            ticker=holding.ticker,
            market=holding.market,
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=holding.quantity,
            limit_price=99.0 + self.calls,
        )
        return SimpleNamespace(approved=True, final_order=order, reason_codes=("unit_exit",))

    def evaluate_buy(self, *args, **kwargs):  # pragma: no cover - candidates are empty
        raise AssertionError("buy path should not be reached")


class _AmendAwareCoordinator:
    def __init__(self) -> None:
        self.submitted = []
        self.amended = []

    def submit_final_order(self, order):
        self.submitted.append(order)
        return LiveOrderSubmission(
            execution_id="submit-1",
            idempotency_key="unit",
            status="ACCEPTED",
            broker_order_id="SELL0001",
            submitted_at=datetime.now(timezone.utc),
            message="submitted",
        )

    def amend_final_order(self, broker_order_id, replacement):
        self.amended.append((broker_order_id, replacement))
        return LiveOrderSubmission(
            execution_id="amend-1",
            idempotency_key="amend",
            status="ACCEPTED",
            broker_order_id="SELL0002",
            submitted_at=datetime.now(timezone.utc),
            message="amended",
        )

    def cancel_final_order(self, broker_order_id, order):  # pragma: no cover
        raise AssertionError("cancel should not be needed")


class RealtimeSellAmendTest(unittest.TestCase):
    def test_second_sell_for_same_symbol_amends_existing_order(self) -> None:
        holding = _holding(100.0, last_price=99.0)
        account = _account(holding, cash=1_000_000.0)
        coordinator = _AmendAwareCoordinator()
        engine = RealtimeTradingEngine(
            decision_engine=_FixedSellDecisionEngine(),
            coordinator=coordinator,
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: (),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(submit_cooldown_sec=999, sell_inflight_cooldown_sec=999),
        )

        first = engine.run_once()
        second = engine.run_once()

        self.assertEqual(first["submitted"], 1)
        self.assertEqual(second["submitted"], 0)
        self.assertEqual(second["amended"], 1)
        self.assertEqual(len(coordinator.submitted), 1)
        self.assertEqual(len(coordinator.amended), 1)
        self.assertEqual(coordinator.amended[0][0], "SELL0001")
        self.assertEqual(coordinator.amended[0][1].limit_price, 101.0)


if __name__ == "__main__":
    unittest.main()
