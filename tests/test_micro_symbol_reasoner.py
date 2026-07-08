from __future__ import annotations

from datetime import datetime, timezone

from app.graph.macro_micro_common import (
    EntrySignal,
    ExecutionQuality,
    ExitSignal,
    MicroRegime,
)
from app.graph.micro_reasoner import MicroReasonerConfig, MicroReasoningInput, MicroSymbolReasoner
from app.graph.rdf_adapter import attach_micro_result_rdf
from app.graph.rdf_graph import RdfTradingGraph, TR
from app.graph.shacl_validator import reset_shapes_cache, validate_graph
from tests.test_technical_signals import trend_up_features


def _now():
    return datetime(2026, 7, 9, tzinfo=timezone.utc)


def _reasoner():
    return MicroSymbolReasoner(MicroReasonerConfig(minimum_micro_confidence=0.3))


def _inp(*, allowed=("momentum", "breakout", "vwap_pullback"), blocked=("aggressive_countertrend_reversion",),
         features=None, holding=None, quote_age=None, **feat_over):
    return MicroReasoningInput(
        timestamp=_now(),
        symbol="005930",
        allowed_micro_strategies=allowed,
        blocked_micro_strategies=blocked,
        technical_features=features if features is not None else trend_up_features(symbol="005930", **feat_over),
        holding_state=holding,
        quote_age_seconds=quote_age,
    )


class TestStrategyPermission:
    def test_buy_candidate_when_allowed(self):
        r = _reasoner().reason(_inp())
        assert r.entry_signal == EntrySignal.BUY_CANDIDATE
        assert r.expected_exit_price is not None
        assert r.expected_net_return_bps is not None and r.expected_net_return_bps > 0

    def test_blocked_when_macro_blocks_new_buy(self):
        r = _reasoner().reason(_inp(blocked=("new_buy",)))
        assert r.entry_signal == EntrySignal.BLOCKED
        assert r.micro_regime == MicroRegime.NO_TRADE_SYMBOL
        assert "MICRO_STRATEGY_BLOCKED_BY_MACRO" in r.reason_codes

    def test_blocked_when_strategy_not_in_allowlist(self):
        # Only mean_reversion allowed, but a trend-up momentum/breakout signal is selected.
        r = _reasoner().reason(_inp(allowed=("mean_reversion",), blocked=()))
        assert r.entry_signal != EntrySignal.BUY_CANDIDATE


class TestExpectedNetReturnRequired:
    def test_no_signal_features_is_no_trade(self):
        from app.technical.signals import TechnicalFeatureSet
        r = _reasoner().reason(_inp(features=TechnicalFeatureSet(symbol="005930")))
        assert r.entry_signal in (EntrySignal.NONE, EntrySignal.BLOCKED)
        assert r.micro_regime in (MicroRegime.NO_TRADE_SYMBOL, MicroRegime.HOLD_OR_WATCH)

    def test_high_cost_no_positive_net_is_not_buy(self):
        # Force spread to consume alpha via a very wide spread.
        r = _reasoner().reason(_inp(spread_bps=500.0))
        assert r.entry_signal != EntrySignal.BUY_CANDIDATE


class TestFreshnessAndExit:
    def test_stale_quote_blocks(self):
        r = _reasoner().reason(_inp(quote_age=200.0))
        assert r.entry_signal == EntrySignal.BLOCKED
        assert "STALE_QUOTE" in r.reason_codes

    def test_holding_deterioration_produces_exit(self):
        r = _reasoner().reason(_inp(
            holding={"quantity": 10, "average_price": 100.0},
            vwap_distance_bps=-10.0, macd_histogram=-0.3,
        ))
        assert r.micro_regime == MicroRegime.EXIT_DETERIORATION
        assert r.exit_signal in (ExitSignal.RISK_REDUCE, ExitSignal.SELL_CANDIDATE)
        assert r.is_exit_candidate


class TestRdfProjectionAdvisory:
    def test_buy_projection_conforms(self):
        reset_shapes_cache()
        result = _reasoner().reason(_inp())
        assert result.entry_signal == EntrySignal.BUY_CANDIDATE
        rdf = RdfTradingGraph(cycle_id="micro-test")
        attach_micro_result_rdf(rdf, result)
        report = validate_graph(rdf.merged_graph(), mode="live")
        assert report.conforms, report.as_dict()

    def test_micro_never_asserts_final_order(self):
        result = _reasoner().reason(_inp())
        rdf = RdfTradingGraph(cycle_id="micro-test")
        attach_micro_result_rdf(rdf, result)
        assert (None, None, TR.FinalOrder) not in rdf.merged_graph()
        assert not hasattr(result, "final_order")

    def test_result_is_deterministic(self):
        a = _reasoner().reason(_inp())
        b = _reasoner().reason(_inp())
        assert a.as_dict() == b.as_dict()
