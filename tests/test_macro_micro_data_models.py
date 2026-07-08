from __future__ import annotations

from datetime import datetime, timezone

from app.graph.macro_micro_common import (
    EntrySignal,
    ExecutionQuality,
    ExitSignal,
    IntentType,
    MacroRiskLevel,
    MarketRegime,
    MicroRegime,
    SelectedStrategy,
)
from app.graph.macro_reasoner import (
    MacroReasoningInput,
    MacroReasoningResult,
    SectorRanking,
)
from app.graph.micro_reasoner import MicroReasoningResult
from app.graph.global_trade_arbiter import GlobalTradeArbiter, RankedTradeIntent
from app.graph.ontology_coordinator import MacroMicroReasoningBundle


def _now():
    return datetime(2026, 7, 9, tzinfo=timezone.utc)


def _macro_result(regime=MarketRegime.TREND_UP, risk=MacroRiskLevel.LOW):
    return MacroReasoningResult(
        timestamp=_now(),
        market_regime=regime,
        macro_risk_level=risk,
        sector_rankings=(SectorRanking("tech", 0.8, 0.7, ("MACRO_SECTOR_STRONG",)),),
        candidate_symbols=("005930",),
        allowed_micro_strategies=("momentum", "breakout"),
        blocked_micro_strategies=("aggressive_countertrend_reversion",),
        macro_confidence=0.7,
        reason_codes=("MACRO_TREND_UP",),
        explanation_paths=({"code": "MACRO_TREND_UP", "text": "up", "features": {}},),
    )


def _micro_result(symbol="005930", entry=EntrySignal.BUY_CANDIDATE, exit_=ExitSignal.NONE,
                  regime=MicroRegime.MOMENTUM_CANDIDATE, net=15.0, downside=40.0, conf=0.7):
    return MicroReasoningResult(
        timestamp=_now(),
        symbol=symbol,
        micro_regime=regime,
        selected_strategy=SelectedStrategy.MOMENTUM,
        entry_signal=entry,
        exit_signal=exit_,
        expected_entry_price=100.0,
        expected_exit_price=100.2,
        expected_gross_return_bps=20.0,
        expected_net_return_bps=net,
        downside_risk_bps=downside,
        confidence=conf,
        execution_quality=ExecutionQuality.GOOD,
        reason_codes=("MICRO_MOMENTUM_CONFIRMED",),
        explanation_paths=({"code": "MICRO_MOMENTUM_CONFIRMED", "text": "ok", "features": {}},),
    )


class TestEnums:
    def test_enum_values(self):
        assert MarketRegime.TREND_UP.value == "TREND_UP"
        assert MacroRiskLevel.BLOCK_BUY.value == "BLOCK_BUY"
        assert MicroRegime.EXIT_DETERIORATION.value == "EXIT_DETERIORATION"
        assert IntentType.SELL.value == "SELL"


class TestMacroResult:
    def test_as_dict_has_required_fields(self):
        d = _macro_result().as_dict()
        for key in ("market_regime", "macro_risk_level", "candidate_symbols",
                    "allowed_micro_strategies", "blocked_micro_strategies",
                    "reason_codes", "explanation_paths", "macro_confidence", "blocks_buy"):
            assert key in d

    def test_blocks_buy_logic(self):
        assert not _macro_result(MarketRegime.TREND_UP, MacroRiskLevel.LOW).blocks_buy
        assert _macro_result(MarketRegime.TREND_UP, MacroRiskLevel.BLOCK_BUY).blocks_buy
        assert _macro_result(MarketRegime.HIGH_VOLATILITY_RISK, MacroRiskLevel.HIGH).blocks_buy
        assert _macro_result(MarketRegime.NO_TRADE_MARKET, MacroRiskLevel.HIGH).blocks_buy


class TestMicroResult:
    def test_flags(self):
        buy = _micro_result(entry=EntrySignal.BUY_CANDIDATE)
        assert buy.is_buy_candidate and not buy.is_exit_candidate
        exit_ = _micro_result(entry=EntrySignal.NONE, exit_=ExitSignal.SELL_CANDIDATE, regime=MicroRegime.EXIT_DETERIORATION)
        assert exit_.is_exit_candidate and not exit_.is_buy_candidate

    def test_as_dict_has_required_fields(self):
        d = _micro_result().as_dict()
        for key in ("symbol", "micro_regime", "selected_strategy", "entry_signal", "exit_signal",
                    "expected_entry_price", "expected_exit_price", "expected_net_return_bps",
                    "downside_risk_bps", "execution_quality", "reason_codes", "explanation_paths"):
            assert key in d


class TestArbiterAdvisoryOnly:
    def test_ranked_intent_has_no_order_authority(self):
        intent = RankedTradeIntent(
            intent_type=IntentType.BUY, symbol="A", side="BUY", rank=0, score=1.0,
            expected_entry_price=100.0, expected_exit_price=101.0, expected_net_return_bps=15.0,
            downside_risk_bps=40.0, macro_regime="TREND_UP", micro_regime="MOMENTUM_CANDIDATE",
            selected_strategy="momentum", confidence=0.7, reason_codes=(), explanation_paths=(),
        )
        # Advisory only: no broker-submission surface.
        for attr in ("final_order", "submit", "broker", "place_order", "order"):
            assert not hasattr(intent, attr)

    def test_sell_reduce_ranked_before_buy(self):
        macro = _macro_result()
        micros = [
            _micro_result("BUY1", entry=EntrySignal.BUY_CANDIDATE, net=30.0),
            _micro_result("SELLX", entry=EntrySignal.NONE, exit_=ExitSignal.SELL_CANDIDATE,
                          regime=MicroRegime.EXIT_DETERIORATION),
        ]
        ranked = GlobalTradeArbiter().rank(macro, micros)["ranked_trade_intents"]
        assert ranked[0].symbol == "SELLX"  # SELL/REDUCE first
        assert ranked[0].side in ("SELL", "REDUCE")
        assert ranked[-1].side == "BUY"


class TestBundle:
    def test_bundle_as_dict(self):
        bundle = MacroMicroReasoningBundle(
            timestamp=_now(), macro_result=_macro_result(), micro_results=(_micro_result(),),
            failed_symbols=(), ranked_trade_intents=(), sell_reduce_candidates=(),
            buy_candidates=("005930",), blocked_candidates=(),
        )
        d = bundle.as_dict()
        assert d["buy_candidates"] == ["005930"]
        assert "macro_result" in d and "micro_results" in d
