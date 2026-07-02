from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.schemas.domain import AccountSnapshot, Holding, MarketSnapshot, OrderSide, SourceMetadata
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


class _ApprovedPredictor:
    def predict(self, frame):
        return SimpleNamespace(
            probability_success=0.74,
            expected_net_return_bps=80.0,
            uncertainty_score=0.2,
            approved=True,
            reason_codes=(),
            model_artifact_id="unit-model",
            feature_schema_hash=frame.feature_schema_hash,
        )


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
    def test_stop_loss_is_blocked_by_default(self) -> None:
        engine = _engine(price=98.0)  # -2% vs avg 100, below 1% stop
        result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved)
        self.assertIsNone(result.final_order)
        self.assertTrue(any("HOLD_LOSS_EXIT_DISABLED" in code for code in result.reason_codes))

    def test_emergency_stop_loss_requires_explicit_opt_in(self) -> None:
        engine = _engine(price=94.0)
        with patch.dict("os.environ", {"REALTIME_ALLOW_LOSS_EXIT": "true", "REALTIME_EMERGENCY_STOP_LOSS": "0.05"}):
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
        self.assertIn("HOLD_BELOW_PROFIT_TARGET", result.reason_codes)

    def test_no_tick_falls_back_to_broker_balance_mark(self) -> None:
        # 실시간 틱이 없어도 브로커 잔고가(last_price)로 손절을 판단해야 한다.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=98.0)  # -2% via broker mark, below 1% stop
        result = engine.evaluate_exit_for_holding(holding, _account(holding), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved)
        self.assertTrue(any("HOLD_LOSS_EXIT_DISABLED" in code for code in result.reason_codes))

    def test_no_price_anywhere_returns_missing_market_data(self) -> None:
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=0.0)  # no tick and no broker mark
        result = engine.evaluate_exit_for_holding(holding, _account(holding))
        self.assertFalse(result.approved)
        self.assertIn("MISSING_MARKET_DATA", result.reason_codes)

    def test_ontology_risk_does_not_sell_below_profit_floor(self) -> None:
        # 가격은 밴드 안(평단 근처)이지만 온톨로지가 매도 신호면 매도해야 한다.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=100.0)  # flat -> within TP/SL bands
        # 현금을 충분히 둬 포지션 비중을 작게 만들어 온톨로지 효과만 분리한다.
        account = _account(holding, cash=1_000_000.0)
        graph = _FakeGraph(risk_objects=("SellCandidate",))
        result = engine.evaluate_exit_for_holding(
            holding, account, take_profit=0.006, stop_loss=0.01, ontology_graph=graph
        )
        self.assertFalse(result.approved)
        self.assertTrue(any("HOLD_UNPROFITABLE_ONTOLOGY_SELL_BLOCKED" in code for code in result.reason_codes))

    def test_ontology_risk_sells_once_profit_floor_is_met(self) -> None:
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=101.0)
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
        self.assertIn("HOLD_BELOW_PROFIT_TARGET", result.reason_codes)

    def test_domestic_drawdown_reduces_before_large_loss(self) -> None:
        engine = _engine(price=98.0)
        holding = _holding(100.0)
        account = _account(holding, cash=1_000_000.0)
        with patch.dict(
            "os.environ",
            {
                "REALTIME_ALLOW_LOSS_EXIT": "true",
                "REALTIME_DOMESTIC_DRAWDOWN_REDUCE_TRIGGER": "0.015",
                "REALTIME_DOMESTIC_EMERGENCY_EXIT_TRIGGER": "0.03",
            },
        ):
            result = engine.evaluate_exit_for_holding(
                holding, account, take_profit=0.006, stop_loss=0.01, ontology_graph=_FakeGraph()
            )

        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertEqual(result.final_order.quantity, 5)
        self.assertIn("domestic_drawdown_reduce", result.reason_codes[0])

    def test_domestic_single_share_concentration_exits_instead_of_zero_reduce(self) -> None:
        engine = _engine(price=99.5)
        holding = Holding(
            ticker="476830",
            market="KR",
            company_name="Concentrated",
            sector="ETF",
            quantity=1,
            average_price=100.0,
            last_price=99.5,
        )
        account = AccountSnapshot(cash=300.0, holdings=(holding,))
        with patch.dict(
            "os.environ",
            {
                "REALTIME_ALLOW_LOSS_EXIT": "true",
                "REALTIME_DOMESTIC_CONCENTRATION_REDUCE_WEIGHT": "0.20",
            },
        ):
            result = engine.evaluate_exit_for_holding(
                holding, account, take_profit=0.006, stop_loss=0.01, ontology_graph=_FakeGraph()
            )

        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertEqual(result.final_order.quantity, 1)
        self.assertIn("domestic_concentration_reduce", result.reason_codes[0])


class _BuyStore:
    """Store exposing a fresh tick (and no orderbook) for the buy path."""

    def __init__(self, price: float) -> None:
        now = datetime.now(timezone.utc)
        self._tick = SimpleNamespace(price=price, received_at=now, exchange_timestamp=now, sequence_key=f"buy:{price}")

    def latest_tick(self, symbol: str):
        return self._tick

    def latest_orderbook(self, symbol: str):
        return None


class _NoTickBuyStore:
    def latest_tick(self, symbol: str):
        return None

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

    def test_domestic_buy_tightens_but_does_not_pause_when_domestic_book_is_in_drawdown(self) -> None:
        engine = SharedLiveDecisionEngine(_BuyStore(price=5_000.0), predictor=_DummyPredictor())
        losing_holding = Holding(
            ticker="005930",
            market="KR",
            company_name="Samsung",
            sector="Tech",
            quantity=1,
            average_price=100_000.0,
            last_price=98_000.0,
        )
        account = AccountSnapshot(
            cash=1_000_000.0,
            holdings=(losing_holding,),
            cash_by_currency={"KRW": 1_000_000.0},
            cash_equivalent_krw=1_000_000.0,
        )
        graph = _FakeGraph(support_objects=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying"))
        with patch.dict(
            "os.environ",
            {
                "REALTIME_DOMESTIC_DRAWDOWN_BUY_TIGHTEN_TRIGGER": "0.005",
                "REALTIME_DOMESTIC_DRAWDOWN_BUY_BONUS_MULTIPLIER": "6.0",
            },
        ):
            result = engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=graph)

        self.assertTrue(result.approved, result.reason_codes)
        diagnostics = engine.get_diagnostics()
        self.assertGreater(diagnostics["effective_buy_threshold"], diagnostics["buy_threshold"])
        self.assertLess(diagnostics["domestic_drawdown_rate"], 0)

    def test_model_only_buy_is_rejected_when_ai_is_auxiliary(self) -> None:
        engine = SharedLiveDecisionEngine(_BuyStore(price=5_000.0), predictor=_ApprovedPredictor())
        engine.feature_builder = SimpleNamespace(
            build=lambda symbol, decision_time=None: SimpleNamespace(
                feature_schema_hash="unit-schema",
                provenance=SimpleNamespace(source_record_ids=("unit-frame",)),
            )
        )
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})

        with patch.dict("os.environ", {"REALTIME_MODEL_AUXILIARY_ONLY": "true"}):
            result = engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=_FakeGraph())

        self.assertFalse(result.approved)
        self.assertIn("MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION", result.reason_codes)

    def test_model_can_boost_buy_when_ontology_confirms(self) -> None:
        engine = SharedLiveDecisionEngine(_BuyStore(price=5_000.0), predictor=_ApprovedPredictor())
        engine.feature_builder = SimpleNamespace(
            build=lambda symbol, decision_time=None: SimpleNamespace(
                feature_schema_hash="unit-schema",
                provenance=SimpleNamespace(source_record_ids=("unit-frame",)),
            )
        )
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        graph = _FakeGraph(support_objects=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying"))

        with patch.dict("os.environ", {"REALTIME_MODEL_AUXILIARY_ONLY": "true"}):
            result = engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=graph)

        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.BUY)

    def test_buy_cash_check_refreshes_broker_quote_before_rejecting(self) -> None:
        now = datetime.now(timezone.utc)

        def refresh(symbol: str, market: str, decision_time: datetime) -> MarketSnapshot:
            return MarketSnapshot(
                symbol,
                market,
                symbol,
                "Unknown",
                70_000.0,
                10_000_000_000,
                0.02,
                SourceMetadata(
                    "KIS broker quote",
                    decision_time,
                    source_type="broker_api",
                    trust_level=5,
                    is_realtime=True,
                    quality_score=1.0,
                ),
            )

        engine = SharedLiveDecisionEngine(
            _BuyStore(price=450_780.0),
            predictor=_DummyPredictor(),
            market_refresher=refresh,
        )
        account = AccountSnapshot(
            cash=102_413.0,
            holdings=(),
            cash_by_currency={"KRW": 102_413.0},
            cash_equivalent_krw=102_413.0,
        )
        graph = _FakeGraph(support_objects=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying"))

        result = engine.evaluate_buy("005930", account, suggested_weight=0.01, ontology_graph=graph, decision_time=now)

        self.assertNotIn("INSUFFICIENT_CASH_FOR_ONE_SHARE", result.reason_codes)
        self.assertEqual(engine.get_diagnostics()["quote_refresh_status"], "quote_refresh_ok")

    def test_missing_tick_refreshes_broker_quote_before_rejecting(self) -> None:
        now = datetime.now(timezone.utc)

        def refresh(symbol: str, market: str, decision_time: datetime) -> MarketSnapshot:
            return MarketSnapshot(
                symbol,
                market,
                symbol,
                "Unknown",
                5.0,
                10_000_000_000,
                0.02,
                SourceMetadata(
                    "KIS broker quote",
                    decision_time,
                    source_type="broker_api",
                    trust_level=5,
                    is_realtime=True,
                    quality_score=1.0,
                ),
            )

        engine = SharedLiveDecisionEngine(
            _NoTickBuyStore(),
            predictor=_DummyPredictor(),
            market_refresher=refresh,
        )
        account = AccountSnapshot(
            cash=1_000_000.0,
            holdings=(),
            cash_by_currency={"USD": 100_000.0},
            cash_equivalent_krw=130_000_000.0,
        )
        graph = _FakeGraph(support_objects=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying"))

        result = engine.evaluate_buy("LAB", account, suggested_weight=0.01, ontology_graph=graph, decision_time=now)

        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.BUY)
        self.assertEqual(engine.get_diagnostics()["quote_refresh_status"], "quote_refresh_ok")

    def test_runtime_probe_allows_small_buy_without_ontology_when_broker_quote_is_fresh(self) -> None:
        now = datetime.now(timezone.utc)

        def refresh(symbol: str, market: str, decision_time: datetime) -> MarketSnapshot:
            return MarketSnapshot(
                symbol,
                market,
                symbol,
                "Unknown",
                5_000.0,
                10_000_000_000,
                0.02,
                SourceMetadata(
                    "KIS broker quote",
                    decision_time,
                    source_type="broker_api",
                    trust_level=5,
                    is_realtime=True,
                    quality_score=1.0,
                ),
            )

        engine = SharedLiveDecisionEngine(
            _NoTickBuyStore(),
            predictor=_DummyPredictor(),
            market_refresher=refresh,
        )
        losing_holding = Holding(
            ticker="005930",
            market="KR",
            company_name="Samsung",
            sector="Tech",
            quantity=1,
            average_price=100_000.0,
            last_price=98_000.0,
        )
        account = AccountSnapshot(
            cash=1_000_000.0,
            holdings=(losing_holding,),
            cash_by_currency={"KRW": 1_000_000.0},
            cash_equivalent_krw=1_000_000.0,
        )
        with patch.dict(
            "os.environ",
            {
                "REALTIME_RUNTIME_PROBE_BUY_ENABLED": "true",
                "REALTIME_RUNTIME_PROBE_BUY_MARGIN": "1.0",
                "REALTIME_RUNTIME_PROBE_BUY_WEIGHT": "0.003",
                "REALTIME_DOMESTIC_DRAWDOWN_BUY_TIGHTEN_TRIGGER": "0.005",
                "REALTIME_DOMESTIC_DRAWDOWN_BUY_BONUS_MULTIPLIER": "20.0",
                "REALTIME_DOMESTIC_DRAWDOWN_BUY_MAX_BONUS": "0.8",
            },
        ):
            result = engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=_FakeGraph(), decision_time=now)

        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.BUY)
        diagnostics = engine.get_diagnostics()
        self.assertTrue(diagnostics["runtime_execution_ready"])
        self.assertTrue(diagnostics["runtime_probe_support"])

    def test_realtime_adaptive_fallback_can_supply_runtime_ontology_support(self) -> None:
        now = datetime.now(timezone.utc)

        def refresh(symbol: str, market: str, decision_time: datetime) -> MarketSnapshot:
            return MarketSnapshot(
                symbol,
                market,
                symbol,
                "Unknown",
                5.0,
                10_000_000_000,
                0.02,
                SourceMetadata(
                    "KIS broker quote",
                    decision_time,
                    source_type="broker_api",
                    trust_level=5,
                    is_realtime=True,
                    quality_score=1.0,
                ),
            )

        engine = SharedLiveDecisionEngine(
            _NoTickBuyStore(),
            predictor=_DummyPredictor(),
            market_refresher=refresh,
        )
        account = AccountSnapshot(
            cash=1_000_000.0,
            holdings=(),
            cash_by_currency={"USD": 100_000.0},
            cash_equivalent_krw=130_000_000.0,
        )

        result = engine.evaluate_buy("LAB", account, suggested_weight=0.01, ontology_graph=_FakeGraph(), decision_time=now)

        self.assertTrue(result.approved, result.reason_codes)
        self.assertTrue(result.final_order.quantity > 0)
        diagnostics = engine.get_diagnostics()
        self.assertTrue(diagnostics["runtime_execution_ready"])
        self.assertTrue(diagnostics["runtime_fallback_support"])


class _FixedSellDecisionEngine:
    def __init__(self, limit_price: float | None = None) -> None:
        self.calls = 0
        self.limit_price = limit_price

    def evaluate_exit_for_holding(self, holding, account, **kwargs):
        self.calls += 1
        limit_price = self.limit_price if self.limit_price is not None else 101.0
        order = FinalOrder(
            ticker=holding.ticker,
            market=holding.market,
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=holding.quantity,
            limit_price=limit_price,
        )
        return SimpleNamespace(approved=True, final_order=order, reason_codes=("unit_exit",))

    def evaluate_buy(self, *args, **kwargs):  # pragma: no cover - candidates are empty
        raise AssertionError("buy path should not be reached")


class _FixedBuyDecisionEngine:
    def evaluate_exit_for_holding(self, *args, **kwargs):  # pragma: no cover - no holdings
        raise AssertionError("sell path should not be reached")

    def evaluate_buy(self, symbol, account, **kwargs):
        order = FinalOrder(
            ticker=symbol,
            market="KR",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=1,
            limit_price=5_000.0,
        )
        return SimpleNamespace(approved=True, final_order=order, reason_codes=("unit_buy",))


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


class _FailingBuyCoordinator:
    def __init__(self) -> None:
        self.submitted = []

    def submit_final_order(self, order):
        self.submitted.append(order)
        raise RuntimeError("broker rejected")

    def amend_final_order(self, broker_order_id, replacement):  # pragma: no cover
        raise AssertionError("amend should not be needed")

    def cancel_final_order(self, broker_order_id, order):  # pragma: no cover
        raise AssertionError("cancel should not be needed")


class RealtimeSellAmendTest(unittest.TestCase):
    def test_domestic_buy_is_skipped_outside_core_session(self) -> None:
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        coordinator = _AmendAwareCoordinator()
        engine = RealtimeTradingEngine(
            decision_engine=_FixedBuyDecisionEngine(),
            coordinator=coordinator,
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: ("000660",),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(max_buy_orders_per_cycle=1),
        )

        with patch.dict("os.environ", {"REALTIME_DOMESTIC_BUY_CORE_SESSION_ONLY": "true"}):
            summary = engine.run_once(datetime(2026, 7, 2, 6, 32, tzinfo=timezone.utc))

        self.assertEqual(summary["buy_evaluated"], 0)
        self.assertEqual(summary["skipped_market_closed"], 1)
        self.assertEqual(len(coordinator.submitted), 0)

    def test_failed_buy_submission_counts_toward_cycle_attempt_limit(self) -> None:
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        coordinator = _FailingBuyCoordinator()
        engine = RealtimeTradingEngine(
            decision_engine=_FixedBuyDecisionEngine(),
            coordinator=coordinator,
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: ("000660", "005930", "035420"),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(max_orders_per_cycle=8, max_buy_orders_per_cycle=1, error_cooldown_sec=0),
        )

        with patch.dict("os.environ", {"REALTIME_DOMESTIC_BUY_CORE_SESSION_ONLY": "false"}):
            summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))

        self.assertEqual(summary["buy_submit_attempted"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(len(coordinator.submitted), 1)

    def test_buy_disabled_flag_skips_buy_path(self) -> None:
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        coordinator = _AmendAwareCoordinator()
        engine = RealtimeTradingEngine(
            decision_engine=_FixedBuyDecisionEngine(),
            coordinator=coordinator,
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: ("000660",),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(max_orders_per_cycle=8, max_buy_orders_per_cycle=1),
        )

        with patch.dict("os.environ", {"REALTIME_BUY_ENABLED": "true"}):
            engine.disable_buys("unit_shutdown")
            summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))

            self.assertEqual(summary["buy_evaluated"], 0)
            self.assertEqual(summary["buy_submitted"], 0)
            self.assertTrue(summary["buy_disabled"])
            self.assertEqual(summary["reason"], "unit_shutdown")
            self.assertEqual(len(coordinator.submitted), 0)
            self.assertFalse(engine.get_status()["buy_enabled"])

    def test_second_sell_for_same_symbol_keeps_existing_order_when_price_unchanged(self) -> None:
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
        self.assertEqual(second["amended"], 0)
        self.assertEqual(len(coordinator.submitted), 1)
        self.assertEqual(len(coordinator.amended), 0)

    def test_second_sell_for_same_symbol_amends_when_price_moves_enough(self) -> None:
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
            config=RealtimeTradingConfig(
                submit_cooldown_sec=999,
                sell_inflight_cooldown_sec=999,
                sell_amend_min_price_delta=0.0001,
            ),
        )

        first = engine.run_once()
        engine.decision_engine = _FixedSellDecisionEngine(limit_price=101.2)
        second = engine.run_once()

        self.assertEqual(first["submitted"], 1)
        self.assertEqual(second["amended"], 1)
        self.assertEqual(len(coordinator.submitted), 1)
        self.assertEqual(len(coordinator.amended), 1)
        self.assertEqual(coordinator.amended[0][0], "SELL0001")
        self.assertEqual(coordinator.amended[0][1].limit_price, 101.2)


if __name__ == "__main__":
    unittest.main()
