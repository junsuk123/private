from __future__ import annotations

from datetime import datetime, timezone

from app.graph.global_trade_arbiter import GlobalTradeArbiter, RankedTradeIntent
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
from app.graph.macro_reasoner import MacroReasoningResult
from app.graph.micro_reasoner import MicroReasoningResult


def _now():
    return datetime(2026, 7, 9, tzinfo=timezone.utc)


def _macro():
    return MacroReasoningResult(
        timestamp=_now(), market_regime=MarketRegime.TREND_UP, macro_risk_level=MacroRiskLevel.LOW,
        sector_rankings=(), candidate_symbols=(), allowed_micro_strategies=(), blocked_micro_strategies=(),
        macro_confidence=0.7, reason_codes=(), explanation_paths=(),
    )


def _micro(symbol, *, buy=True, net=15.0, conf=0.7, downside=40.0, exit_=False, blocked=False):
    return MicroReasoningResult(
        timestamp=_now(), symbol=symbol,
        micro_regime=(MicroRegime.EXIT_DETERIORATION if exit_ else
                      MicroRegime.NO_TRADE_SYMBOL if blocked else MicroRegime.MOMENTUM_CANDIDATE),
        selected_strategy=SelectedStrategy.MOMENTUM,
        entry_signal=(EntrySignal.BLOCKED if blocked else EntrySignal.BUY_CANDIDATE if buy else EntrySignal.NONE),
        exit_signal=(ExitSignal.SELL_CANDIDATE if exit_ else ExitSignal.NONE),
        expected_entry_price=100.0, expected_exit_price=101.0,
        expected_gross_return_bps=25.0, expected_net_return_bps=net,
        downside_risk_bps=downside, confidence=conf, execution_quality=ExecutionQuality.GOOD,
        reason_codes=(), explanation_paths=(),
    )


class TestArbiter:
    def test_sell_reduce_before_buy(self):
        out = GlobalTradeArbiter().rank(_macro(), [
            _micro("BUY1", net=30.0), _micro("SELLX", buy=False, exit_=True),
        ])
        intents = out["ranked_trade_intents"]
        assert intents[0].symbol == "SELLX" and intents[0].intent_type in (IntentType.SELL, IntentType.REDUCE)
        assert intents[-1].symbol == "BUY1" and intents[-1].intent_type == IntentType.BUY

    def test_buy_ranked_by_net_return(self):
        out = GlobalTradeArbiter().rank(_macro(), [
            _micro("LOW", net=5.0), _micro("HIGH", net=40.0), _micro("MID", net=20.0),
        ])
        buys = [i for i in out["ranked_trade_intents"] if i.intent_type == IntentType.BUY]
        assert [i.symbol for i in buys] == ["HIGH", "MID", "LOW"]

    def test_downside_penalizes_score(self):
        out = GlobalTradeArbiter().rank(_macro(), [
            _micro("SAFE", net=20.0, downside=10.0), _micro("RISKY", net=20.0, downside=400.0),
        ])
        buys = [i for i in out["ranked_trade_intents"] if i.intent_type == IntentType.BUY]
        assert buys[0].symbol == "SAFE"

    def test_blocked_candidates_listed(self):
        out = GlobalTradeArbiter().rank(_macro(), [_micro("BLK", blocked=True)])
        assert "BLK" in out["blocked_candidates"]
        assert not any(i.symbol == "BLK" for i in out["ranked_trade_intents"])

    def test_ranked_intent_advisory_only(self):
        out = GlobalTradeArbiter().rank(_macro(), [_micro("A")])
        intent = out["ranked_trade_intents"][0]
        assert isinstance(intent, RankedTradeIntent)
        for attr in ("final_order", "submit", "broker", "place_order"):
            assert not hasattr(intent, attr)
