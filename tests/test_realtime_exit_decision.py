from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
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
            # A genuinely profitable model signal: ~2% predicted short-horizon move,
            # which clears the unified ProfitabilityGate's KR net-return floor after
            # round-trip cost. (The old 80bps signal is a marginal trade the gate now
            # correctly rejects — see the profitability refactor.)
            expected_net_return_bps=200.0,
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
    def test_small_account_one_share_loss_concentration_should_hold(self) -> None:
        holding = Holding(
            ticker="005930",
            market="KR",
            company_name="Samsung",
            sector="Tech",
            quantity=1,
            average_price=50_000.0,
            last_price=49_900.0,
        )
        account = AccountSnapshot(cash=150_000.0, holdings=(holding,))

        with patch.dict(
            "os.environ",
            {
                "REALTIME_SMALL_ACCOUNT_MODE": "true",
                "REALTIME_SMALL_ACCOUNT_EQUITY_KRW": "300000",
                "REALTIME_BLOCK_ONE_SHARE_LOSS_REDUCE": "true",
                "REALTIME_BLOCK_SELL_BELOW_BREAKEVEN": "true",
                "REALTIME_HARD_STOP_LOSS": "0.03",
                "REALTIME_STOP_LOSS_NET": "0.004",
            },
        ):
            result = _engine(price=49_900.0).evaluate_exit_for_holding(
                holding, account, take_profit=0.006, stop_loss=0.01
            )

        self.assertFalse(result.approved, result.reason_codes)
        self.assertIn("SMALL_ACCOUNT_ONE_SHARE_LOSS_BLOCK", result.reason_codes)

    def test_small_account_hard_stop_exit_allowed(self) -> None:
        holding = Holding(
            ticker="005930",
            market="KR",
            company_name="Samsung",
            sector="Tech",
            quantity=1,
            average_price=50_000.0,
            last_price=48_000.0,
        )
        account = AccountSnapshot(cash=150_000.0, holdings=(holding,))

        with patch.dict(
            "os.environ",
            {
                "REALTIME_SMALL_ACCOUNT_MODE": "true",
                "REALTIME_SMALL_ACCOUNT_EQUITY_KRW": "300000",
                "REALTIME_BLOCK_ONE_SHARE_LOSS_REDUCE": "true",
                "REALTIME_BLOCK_SELL_BELOW_BREAKEVEN": "true",
                "REALTIME_HARD_STOP_LOSS": "0.03",
                "REALTIME_STOP_LOSS_NET": "0",
            },
        ):
            result = _engine(price=48_000.0).evaluate_exit_for_holding(
                holding, account, take_profit=0.006, stop_loss=0.01
            )

        self.assertTrue(result.approved, result.reason_codes)
        self.assertTrue(any(code.startswith("hard_stop_loss") for code in result.reason_codes), result.reason_codes)

    def test_investment_mode_holds_normal_dip(self) -> None:
        # Default (investment) mode: a normal -2% dip is HELD, never dumped below entry.
        engine = _engine(price=98.0)
        result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved, result.reason_codes)
        self.assertIsNone(result.final_order)

    def test_optin_tight_stop_sells_when_enabled(self) -> None:
        # A tight stop is opt-in (scalping); enabling it cuts a -2% loser.
        engine = _engine(price=98.0)
        with patch.dict("os.environ", {"REALTIME_STOP_LOSS_NET": "0.004"}):
            result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertTrue(any(code.startswith("stop_loss") for code in result.reason_codes), result.reason_codes)

    def test_optin_tight_stop_sells_even_when_block_below_breakeven_is_on(self) -> None:
        # REGRESSION: run.ps1 pins REALTIME_STOP_LOSS_NET=0.004 AND
        # REALTIME_BLOCK_SELL_BELOW_BREAKEVEN=true together. The net tight stop is
        # documented to "fire regardless of allow_loss_exit", but the break-even
        # block used to silently veto it (only hard/emergency stops were exempt),
        # so the configured stop-loss never actually sold — losers between the net
        # stop and the -3% hard stop were held indefinitely and showed as blocked
        # sells. A deliberate stop-loss must still execute below break-even.
        engine = _engine(price=98.0)  # -2% -> below the 0.4% net stop
        with patch.dict(
            "os.environ",
            {
                "REALTIME_STOP_LOSS_NET": "0.004",
                "REALTIME_BLOCK_SELL_BELOW_BREAKEVEN": "true",
                "REALTIME_ALLOW_LOSS_EXIT": "false",
                "REALTIME_HARD_STOP_LOSS": "0.03",
            },
        ):
            result = engine.evaluate_exit_for_holding(
                _holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01
            )
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertTrue(any(code.startswith("stop_loss") for code in result.reason_codes), result.reason_codes)

    def test_block_below_breakeven_still_holds_noise_dip_without_a_stop(self) -> None:
        # Guard the other side: with NO deliberate stop configured, a slightly
        # underwater position must still be HELD by block_sell_below_breakeven
        # (the churn guard is only bypassed for real stops, not for noise dips).
        engine = _engine(price=98.7)  # ~ -1.3%, no stop configured
        with patch.dict(
            "os.environ",
            {
                "REALTIME_STOP_LOSS_NET": "0",
                "REALTIME_HARD_STOP_LOSS": "0",
                "REALTIME_BLOCK_SELL_BELOW_BREAKEVEN": "true",
                "REALTIME_ALLOW_LOSS_EXIT": "false",
            },
        ):
            result = engine.evaluate_exit_for_holding(
                _holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01
            )
        self.assertFalse(result.approved, result.reason_codes)
        self.assertIsNone(result.final_order)

    def test_all_stops_disabled_holds_loss(self) -> None:
        engine = _engine(price=98.0)  # -2%
        with patch.dict("os.environ", {"REALTIME_STOP_LOSS_NET": "0", "REALTIME_HARD_STOP_LOSS": "0"}):
            result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved)
        self.assertTrue(any("HOLD_LOSS_EXIT_DISABLED" in code for code in result.reason_codes))

    def test_emergency_stop_loss_requires_explicit_opt_in(self) -> None:
        engine = _engine(price=94.0)
        with patch.dict("os.environ", {"REALTIME_ALLOW_LOSS_EXIT": "true", "REALTIME_EMERGENCY_STOP_LOSS": "0.05", "REALTIME_STOP_LOSS_NET": "0", "REALTIME_HARD_STOP_LOSS": "0"}):
            result = engine.evaluate_exit_for_holding(_holding(100.0), _account(_holding(100.0)), take_profit=0.006, stop_loss=0.01)

        self.assertTrue(result.approved, result.reason_codes)
        self.assertIsNotNone(result.final_order)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertEqual(result.final_order.quantity, 10)

    def test_take_profit_triggers_sell(self) -> None:
        engine = _engine(price=101.6)  # +1.6% vs avg 100, above the net quick-take-profit floor
        # Disable the won-amount rule to exercise the percentage take-profit path.
        with patch.dict("os.environ", {"REALTIME_TAKE_PROFIT_AMOUNT_KRW": "0"}):
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
        # 실시간 틱이 없어도 브로커 잔고가(last_price)로 판단해야 한다. opt-in 손절을 켜고
        # -2% 잔고가로 손절이 발동하면 = 잔고가가 실제로 사용된 것.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=98.0)  # -2% via broker mark
        with patch.dict("os.environ", {"REALTIME_STOP_LOSS_NET": "0.004"}):
            result = engine.evaluate_exit_for_holding(holding, _account(holding), take_profit=0.006, stop_loss=0.01)
        self.assertTrue(result.approved, result.reason_codes)
        self.assertTrue(any(code.startswith("stop_loss") for code in result.reason_codes), result.reason_codes)

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
                "REALTIME_STOP_LOSS_NET": "0",
                "REALTIME_HARD_STOP_LOSS": "0",
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
                "REALTIME_STOP_LOSS_NET": "0",
                "REALTIME_HARD_STOP_LOSS": "0",
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


class _VolatileBuyStore(_BuyStore):
    def __init__(self, prices: tuple[float, ...]) -> None:
        super().__init__(price=prices[-1])
        now = datetime.now(timezone.utc)
        self._ticks = tuple(
            SimpleNamespace(
                price=price,
                received_at=now - timedelta(seconds=(len(prices) - index) * 10),
                exchange_timestamp=now - timedelta(seconds=(len(prices) - index) * 10),
                sequence_key=f"vol:{index}",
            )
            for index, price in enumerate(prices)
        )

    def recent_ticks(self, symbol: str, since: datetime):
        return tuple(tick for tick in self._ticks if tick.exchange_timestamp >= since)

    def active_symbols(self, since: datetime, *, limit: int = 200):
        return ("000660",)


class _NoTickBuyStore:
    def latest_tick(self, symbol: str):
        return None

    def latest_orderbook(self, symbol: str):
        return None


class RealtimeBuyDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        # These tests exercise SIGNAL WIRING (ontology/flow/probe drives a buy), not
        # profitability calibration. Under the unified ProfitabilityGate a buy must
        # clear a real net-edge floor (0.8% KR / 1.2% US after cost), so a strong
        # flow signal must map to a genuinely clearing expected move. We raise the
        # fallback-edge coefficient (default 120 bps/score) to represent that strong
        # conviction; the gate still enforces break-even, spread, and cost burden.
        self._prev_fallback_edge = os.environ.get("REALTIME_FALLBACK_EDGE_BPS_PER_SCORE")
        os.environ["REALTIME_FALLBACK_EDGE_BPS_PER_SCORE"] = "700"

    def tearDown(self) -> None:
        if self._prev_fallback_edge is None:
            os.environ.pop("REALTIME_FALLBACK_EDGE_BPS_PER_SCORE", None)
        else:
            os.environ["REALTIME_FALLBACK_EDGE_BPS_PER_SCORE"] = self._prev_fallback_edge

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

    def test_symbol_realtime_volatility_blocks_buy(self) -> None:
        engine = SharedLiveDecisionEngine(
            _VolatileBuyStore(prices=(100.0, 103.0, 97.0, 104.0, 96.0, 105.0)),
            predictor=_DummyPredictor(),
        )
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        graph = _FakeGraph(support_objects=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying"))

        with patch.dict("os.environ", {"REALTIME_MAX_SYMBOL_VOLATILITY_BUY": "0.01"}):
            result = engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=graph)

        self.assertFalse(result.approved)
        self.assertTrue(any(code.startswith("SYMBOL_VOLATILITY_TOO_HIGH") for code in result.reason_codes), result.reason_codes)
        self.assertGreater(engine.get_diagnostics()["policy_state"]["market_state"]["symbol_volatility"], 0.01)

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

    def test_buy_cash_precheck_uses_orderable_cash(self) -> None:
        engine = SharedLiveDecisionEngine(_BuyStore(price=50_000.0), predictor=_DummyPredictor())
        account = AccountSnapshot(
            cash=0.0,
            holdings=(),
            cash_by_currency={"KRW": 0.0},
            orderable_cash_by_currency={"KRW": 100_000.0},
            cash_equivalent_krw=100_000.0,
        )
        graph = _FakeGraph(support_objects=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying"))

        result = engine.evaluate_buy("005930", account, suggested_weight=0.01, ontology_graph=graph)

        self.assertTrue(result.approved, result.reason_codes)
        self.assertNotIn("INSUFFICIENT_CASH_FOR_ONE_SHARE", result.reason_codes)

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
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_exit_for_holding(self, *args, **kwargs):  # pragma: no cover - no holdings
        raise AssertionError("sell path should not be reached")

    def evaluate_buy(self, symbol, account, **kwargs):
        self.calls += 1
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


class _NoQuantityAmendCoordinator(_AmendAwareCoordinator):
    def amend_final_order(self, broker_order_id, replacement):
        self.amended.append((broker_order_id, replacement))
        raise RuntimeError("KIS overseas order revise rejected: 매매가능한 수량이 없습니다.")


class RealtimeSellAmendTest(unittest.TestCase):
    def test_ignored_holding_skips_sell_evaluation(self) -> None:
        ignored = Holding(
            ticker="LCFYW",
            market="NASDAQ",
            company_name="Locafy Warrant",
            sector="Tech",
            quantity=1,
            average_price=1.0,
            last_price=1.0,
        )
        account = AccountSnapshot(cash=1_000_000.0, holdings=(ignored,), cash_by_currency={"USD": 1_000.0})
        decision_engine = _FixedSellDecisionEngine()
        coordinator = _AmendAwareCoordinator()
        engine = RealtimeTradingEngine(
            decision_engine=decision_engine,
            coordinator=coordinator,
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: (),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
        )

        with patch.dict("os.environ", {"REALTIME_IGNORE_SYMBOLS": "lcfyw"}):
            summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))

        self.assertEqual(summary["skipped_ignored"], 1)
        self.assertEqual(summary["sell_evaluated"], 0)
        self.assertEqual(decision_engine.calls, 0)
        self.assertEqual(len(coordinator.submitted), 0)

    def test_ignored_buy_candidate_does_not_block_later_candidates(self) -> None:
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        decision_engine = _FixedBuyDecisionEngine()
        coordinator = _AmendAwareCoordinator()
        engine = RealtimeTradingEngine(
            decision_engine=decision_engine,
            coordinator=coordinator,
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: ("LCFYW", "000660"),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(max_buy_orders_per_cycle=1),
        )

        with patch.dict(
            "os.environ",
            {"REALTIME_IGNORE_SYMBOLS": "LCFYW", "REALTIME_DOMESTIC_BUY_CORE_SESSION_ONLY": "false"},
        ):
            summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))

        self.assertEqual(summary["skipped_ignored"], 1)
        self.assertEqual(summary["buy_evaluated"], 1)
        self.assertEqual(decision_engine.calls, 1)
        self.assertEqual(coordinator.submitted[0].ticker, "000660")

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

    def test_daily_realized_loss_stops_new_buys_but_keeps_runtime_alive(self) -> None:
        account = AccountSnapshot(
            cash=1_000_000.0,
            holdings=(),
            realized_pnl_today=-1_500.0,
            cash_by_currency={"KRW": 1_000_000.0},
            cash_equivalent_krw=1_000_000.0,
        )
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

        with patch.dict("os.environ", {
            "REALTIME_BUY_ENABLED": "true",
            "REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW": "1000",
            "REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE": "0",
        }):
            summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))

        self.assertEqual(summary["buy_evaluated"], 0)
        self.assertTrue(summary["buy_disabled"])
        self.assertTrue(summary["buy_disabled_reason"].startswith("DAILY_REALIZED_LOSS_BUY_STOP"))
        self.assertEqual(len(coordinator.submitted), 0)

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

    def test_sell_uses_sellable_quantity_instead_of_total_holding_quantity(self) -> None:
        holding = Holding(
            ticker="484590",
            market="KR",
            company_name="Sellable",
            sector="ETF",
            quantity=2,
            average_price=7236.5,
            last_price=7340.0,
            sellable_quantity=1,
        )
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

        result = engine.run_once()

        self.assertEqual(result["submitted"], 1)
        self.assertEqual(coordinator.submitted[0].quantity, 1)

    def test_sell_skips_when_no_sellable_quantity(self) -> None:
        holding = Holding(
            ticker="484590",
            market="KR",
            company_name="Locked",
            sector="ETF",
            quantity=2,
            average_price=7236.5,
            last_price=7340.0,
            sellable_quantity=0,
        )
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

        result = engine.run_once()

        self.assertEqual(result["submitted"], 0)
        self.assertEqual(result["sell_rejected"], 1)
        self.assertEqual(len(coordinator.submitted), 0)
        self.assertIn("NO_SELLABLE_QUANTITY", result["rejections"][0]["reason_codes"])

    def test_sell_amend_no_available_quantity_drops_stale_open_sell(self) -> None:
        holding = _holding(100.0, last_price=99.0)
        account = _account(holding, cash=1_000_000.0)
        coordinator = _NoQuantityAmendCoordinator()
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
        third = engine.run_once()

        self.assertEqual(first["submitted"], 1)
        self.assertEqual(second["submitted"], 0)
        self.assertEqual(second["errors"], 0)
        self.assertEqual(len(coordinator.amended), 1)
        self.assertEqual(third["skipped_cooldown"], 1)


class ProfitTurnoverExitTest(unittest.TestCase):
    """Turnover-first profit realization: net winners are sold decisively, and
    gains are locked on giveback — without ever realizing a net loss."""

    def test_quick_take_profit_realizes_net_winner(self) -> None:
        # +1.6% gross clears the round-trip cost by well over the 1.2% net floor,
        # so the position is realized immediately regardless of the dynamic bar.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=101.6)
        # Disable the won-amount rule to exercise the percentage take-profit path.
        with patch.dict("os.environ", {"REALTIME_TAKE_PROFIT_AMOUNT_KRW": "0"}):
            result = engine.evaluate_exit_for_holding(holding, _account(holding, cash=1_000_000.0))
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertTrue(any(code.startswith("quick_take_profit") for code in result.reason_codes), result.reason_codes)

    def test_quick_take_profit_does_not_fire_below_net_floor(self) -> None:
        # +0.2% gross is below the round-trip cost + net floor: selling here would
        # be churn at ~break-even, so the engine holds.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=100.2)
        result = engine.evaluate_exit_for_holding(holding, _account(holding, cash=1_000_000.0), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved)
        self.assertIn("HOLD_BELOW_PROFIT_TARGET", result.reason_codes)

    def test_small_loss_within_stop_is_not_sold_by_turnover_rules(self) -> None:
        # A small loss inside the primary stop band must not be sold by the profit
        # turnover rules (no phantom profit sale); it is simply held.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=99.9)  # -0.1% -> net within the -0.4% stop
        result = engine.evaluate_exit_for_holding(holding, _account(holding, cash=1_000_000.0), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved)
        self.assertIsNone(result.final_order)

    def test_moderate_loss_is_held_in_investment_mode(self) -> None:
        # Investment mode: a moderate -1.3% loss is HELD (not sold below entry).
        # Only the catastrophic backstop (default -8%) can cut a position.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        holding = _holding(100.0, last_price=98.7)  # ~ -1.3% gross
        result = engine.evaluate_exit_for_holding(holding, _account(holding, cash=1_000_000.0), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved, result.reason_codes)
        self.assertIsNone(result.final_order)

    def test_profit_time_exit_realizes_meaningful_net_winner_after_window(self) -> None:
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        now = datetime.now(timezone.utc)
        holding = Holding(
            ticker="005930", market="KR", company_name="Samsung", sector="Tech",
            quantity=10, average_price=100.0, last_price=101.1,  # +1.1% gross -> ~0.8% net, clears the floor
            opened_at=now - timedelta(seconds=600),
        )
        with patch.dict("os.environ", {
            "REALTIME_QUICK_TAKE_PROFIT_NET": "0.05",   # disable quick TP so time-exit is exercised
            "REALTIME_PROFIT_TIME_EXIT_SEC": "300",
            "REALTIME_TAKE_PROFIT_AMOUNT_KRW": "0",      # disable won-amount rule to isolate time-exit
        }):
            result = engine.evaluate_exit_for_holding(
                holding, _account(holding, cash=1_000_000.0), take_profit=0.02, stop_loss=0.01, decision_time=now
            )
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertTrue(any(code.startswith("profit_time_exit") for code in result.reason_codes), result.reason_codes)

    def test_near_breakeven_position_is_not_churned(self) -> None:
        # A position only marginally above cost (net < min floor) must NOT be sold:
        # churning it at ~break-even bleeds fees and erodes total assets.
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        now = datetime.now(timezone.utc)
        holding = Holding(
            ticker="005930", market="KR", company_name="Samsung", sector="Tech",
            quantity=10, average_price=100.0, last_price=100.4,  # +0.4% gross -> ~0.1% net (below 0.3% floor)
            opened_at=now - timedelta(seconds=600),
        )
        result = engine.evaluate_exit_for_holding(
            holding, _account(holding, cash=1_000_000.0), take_profit=0.006, stop_loss=0.01, decision_time=now
        )
        self.assertFalse(result.approved, result.reason_codes)
        self.assertIsNone(result.final_order)

    def test_hard_stop_cuts_catastrophic_loser_even_when_loss_exit_disabled(self) -> None:
        # Capital circuit-breaker: a catastrophic loser is cut to protect total assets,
        # even though routine loss exits are disabled by default.
        engine = _engine(price=94.0)  # -6% vs avg 100, past a -5% hard stop
        holding = _holding(100.0)
        # Disable the routine stop; set the catastrophic backstop to -5% so -6% cuts.
        with patch.dict("os.environ", {"REALTIME_STOP_LOSS_NET": "0", "REALTIME_HARD_STOP_LOSS": "0.05"}):
            result = engine.evaluate_exit_for_holding(holding, _account(holding, cash=1_000_000.0), take_profit=0.006, stop_loss=0.01)
        self.assertTrue(result.approved, result.reason_codes)
        self.assertEqual(result.final_order.side, OrderSide.SELL)
        self.assertEqual(result.final_order.quantity, 10)
        self.assertTrue(any(code.startswith("hard_stop_loss") for code in result.reason_codes), result.reason_codes)

    def test_all_stops_disabled_holds_catastrophic_loss(self) -> None:
        engine = _engine(price=94.0)  # -6%
        holding = _holding(100.0)
        with patch.dict("os.environ", {"REALTIME_HARD_STOP_LOSS": "0", "REALTIME_STOP_LOSS_NET": "0"}):
            result = engine.evaluate_exit_for_holding(holding, _account(holding, cash=1_000_000.0), take_profit=0.006, stop_loss=0.01)
        self.assertFalse(result.approved, result.reason_codes)
        self.assertIsNone(result.final_order)

    def test_profit_lock_sells_on_giveback_from_peak(self) -> None:
        engine = SharedLiveDecisionEngine(SimpleNamespace(latest_tick=lambda s: None), predictor=_DummyPredictor())
        env = {
            "REALTIME_QUICK_TAKE_PROFIT_NET": "0.05",   # disable quick TP
            "REALTIME_PROFIT_LOCK_ARM_NET": "0.006",
            "REALTIME_PROFIT_LOCK_GIVEBACK": "0.35",
        }
        # Call 1: arm the peak at ~+1.2% gross; take_profit high so profit_exit does not preempt.
        peak_holding = _holding(100.0, last_price=101.2)
        with patch.dict("os.environ", env):
            first = engine.evaluate_exit_for_holding(
                peak_holding, _account(peak_holding, cash=1_000_000.0), take_profit=0.02, stop_loss=0.01
            )
        self.assertFalse(first.approved, first.reason_codes)  # held, peak recorded
        # Call 2: price gives back to ~+0.7% gross (still net-positive) -> lock the gain.
        giveback_holding = _holding(100.0, last_price=100.7)
        with patch.dict("os.environ", env):
            second = engine.evaluate_exit_for_holding(
                giveback_holding, _account(giveback_holding, cash=1_000_000.0), take_profit=0.02, stop_loss=0.01
            )
        self.assertTrue(second.approved, second.reason_codes)
        self.assertEqual(second.final_order.side, OrderSide.SELL)
        self.assertTrue(any(code.startswith("profit_lock") for code in second.reason_codes), second.reason_codes)


class InvestmentModeRealHoldingsTest(unittest.TestCase):
    """Regression on real reported holdings: winners are realized, losers are held.

    Reproduces the dashboard snapshot where profitable positions were wrongly held
    while losers were sold. Investment mode must sell winners and hold losers.
    """

    def _engine_no_tick(self) -> SharedLiveDecisionEngine:
        return SharedLiveDecisionEngine(
            SimpleNamespace(latest_tick=lambda s: None, latest_orderbook=lambda s: None),
            predictor=_DummyPredictor(),
        )

    def test_winners_sell_losers_hold_on_real_holdings(self) -> None:
        rows = [
            ("342870", 3900, 3860, False),  # -1.03% -> HOLD
            ("364950", 3670, 3635, False),  # -0.95% -> HOLD
            ("388610", 4970, 5072, True),   # +2.05% -> SELL
            ("460870", 2155, 2260, True),   # +4.87% -> SELL
            ("475230", 7440, 7440, False),  # flat -> HOLD
        ]
        holdings = tuple(
            Holding(ticker=t, market="KR", company_name=t, sector="", quantity=1, average_price=a, last_price=c)
            for t, a, c, _ in rows
        )
        account = AccountSnapshot(cash=1_000_000.0, holdings=holdings, cash_by_currency={"KRW": 1_000_000.0})
        engine = self._engine_no_tick()
        for (t, a, c, should_sell), holding in zip(rows, holdings):
            result = engine.evaluate_exit_for_holding(holding, account, take_profit=0.006, stop_loss=0.01)
            self.assertEqual(
                result.approved, should_sell,
                msg=f"{t} ret={(c/a-1)*100:.2f}% expected {'SELL' if should_sell else 'HOLD'}, got {result.reason_codes}",
            )
            if should_sell:
                self.assertEqual(result.final_order.side, OrderSide.SELL)
                self.assertTrue(
                    any(code.startswith(("take_profit_amount", "quick_take_profit")) for code in result.reason_codes),
                    result.reason_codes,
                )

    def test_amount_based_take_profit_threshold(self) -> None:
        # Sell when AFTER-FEE profit clears the won threshold; hold small gains below it.
        engine = self._engine_no_tick()
        acct = lambda h: AccountSnapshot(cash=1_000_000.0, holdings=(h,), cash_by_currency={"KRW": 1_000_000.0})
        big = Holding(ticker="388610", market="KR", company_name="x", sector="", quantity=1, average_price=4970, last_price=5072)   # net ~88원
        small = Holding(ticker="100000", market="KR", company_name="x", sector="", quantity=1, average_price=4000, last_price=4020)  # gross 20원 -> net<0
        self.assertTrue(engine.evaluate_exit_for_holding(big, acct(big)).approved)     # 88 >= 40 -> SELL
        self.assertFalse(engine.evaluate_exit_for_holding(small, acct(small)).approved)  # tiny -> HOLD
        # Raising the threshold to 200원 makes the 88원 winner hold.
        with patch.dict("os.environ", {"REALTIME_TAKE_PROFIT_AMOUNT_KRW": "200", "REALTIME_QUICK_TAKE_PROFIT_NET": "1"}):
            self.assertFalse(engine.evaluate_exit_for_holding(big, acct(big)).approved)


class RebuyCooldownTest(unittest.TestCase):
    def test_recently_sold_ticker_is_not_rebought_within_cooldown(self) -> None:
        # A ticker sold moments ago must not be re-bought within the rebuy cooldown
        # (anti-churn), even if it reappears as a candidate and is no longer held.
        import time as _time

        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        engine = RealtimeTradingEngine(
            decision_engine=_FixedBuyDecisionEngine(),
            coordinator=_AmendAwareCoordinator(),
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: ("000660",),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(rebuy_cooldown_sec=3600, submit_cooldown_sec=0),
        )
        engine._recent_sell_monotonic["000660"] = _time.monotonic()  # sold just now
        summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))
        self.assertEqual(summary["buy_evaluated"], 0)
        self.assertGreaterEqual(summary["skipped_cooldown"], 1)

    def test_rebuy_allowed_after_cooldown_elapses(self) -> None:
        import time as _time

        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        engine = RealtimeTradingEngine(
            decision_engine=_FixedBuyDecisionEngine(),
            coordinator=_AmendAwareCoordinator(),
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: ("000660",),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(rebuy_cooldown_sec=1.0, submit_cooldown_sec=0),
        )
        engine._recent_sell_monotonic["000660"] = _time.monotonic() - 5.0  # sold well before cooldown
        summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))
        self.assertEqual(summary["buy_evaluated"], 1)

    def test_recent_loss_symbol_cooldown_skips_buy_before_evaluation(self) -> None:
        import time as _time

        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        engine = RealtimeTradingEngine(
            decision_engine=_FixedBuyDecisionEngine(),
            coordinator=_AmendAwareCoordinator(),
            account_provider=lambda: account,
            candidate_symbols_provider=lambda: ("000660",),
            session_open_provider=lambda: True,
            market_open_provider=lambda ticker, market: True,
            config=RealtimeTradingConfig(loss_rebuy_cooldown_sec=3600, submit_cooldown_sec=0),
        )
        engine._loss_cooldown_until["000660"] = _time.monotonic() + 3600

        summary = engine.run_once(datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc))

        self.assertEqual(summary["buy_evaluated"], 0)
        self.assertEqual(summary["skipped_cooldown"], 1)
        self.assertEqual(summary["rejections"][0]["reason_codes"], ("RECENT_LOSS_SYMBOL_COOLDOWN",))

    def test_submitted_loss_round_trip_creates_symbol_cooldown(self) -> None:
        engine = RealtimeTradingEngine(
            decision_engine=_FixedBuyDecisionEngine(),
            coordinator=_AmendAwareCoordinator(),
            account_provider=lambda: AccountSnapshot(cash=1_000_000.0, holdings=()),
            candidate_symbols_provider=lambda: (),
            session_open_provider=lambda: True,
            config=RealtimeTradingConfig(loss_rebuy_cooldown_sec=3600, loss_rebuy_return_threshold=-0.004),
        )
        buy = FinalOrder("000660", "KR", OrderType.LIMIT, OrderSide.BUY, 1, 100.0)
        sell = FinalOrder("000660", "KR", OrderType.LIMIT, OrderSide.SELL, 1, 99.0)

        engine._record_submitted_order_for_performance(buy, "BUY")
        engine._record_submitted_order_for_performance(sell, "SELL")

        self.assertIn("000660", engine.get_status()["loss_cooldown_symbols"])


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
