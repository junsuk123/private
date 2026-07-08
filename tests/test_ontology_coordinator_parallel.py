from __future__ import annotations

import time
from datetime import datetime, timezone

from app.graph.macro_micro_common import (
    EntrySignal,
    ExecutionQuality,
    ExitSignal,
    MacroRiskLevel,
    MarketRegime,
    MicroRegime,
    SelectedStrategy,
)
from app.graph.macro_reasoner import MacroReasoningInput, MacroReasoningResult
from app.graph.micro_reasoner import MicroReasoningInput, MicroReasoningResult
from app.graph.ontology_coordinator import CoordinatorConfig, OntologyCoordinator


def _now():
    return datetime(2026, 7, 9, tzinfo=timezone.utc)


def _macro_result(candidates, *, blocks=False):
    return MacroReasoningResult(
        timestamp=_now(),
        market_regime=MarketRegime.HIGH_VOLATILITY_RISK if blocks else MarketRegime.TREND_UP,
        macro_risk_level=MacroRiskLevel.BLOCK_BUY if blocks else MacroRiskLevel.LOW,
        sector_rankings=(),
        candidate_symbols=tuple(candidates),
        allowed_micro_strategies=("momentum",),
        blocked_micro_strategies=(),
        macro_confidence=0.7,
        reason_codes=("MACRO_TREND_UP",),
        explanation_paths=(),
    )


class _FakeMacro:
    def __init__(self, result):
        self._result = result

    def reason(self, macro_input):
        return self._result


def _micro_result(symbol, *, exit_=False):
    return MicroReasoningResult(
        timestamp=_now(), symbol=symbol,
        micro_regime=MicroRegime.EXIT_DETERIORATION if exit_ else MicroRegime.MOMENTUM_CANDIDATE,
        selected_strategy=SelectedStrategy.REDUCE_RISK if exit_ else SelectedStrategy.MOMENTUM,
        entry_signal=EntrySignal.NONE if exit_ else EntrySignal.BUY_CANDIDATE,
        exit_signal=ExitSignal.SELL_CANDIDATE if exit_ else ExitSignal.NONE,
        expected_entry_price=100.0, expected_exit_price=101.0,
        expected_gross_return_bps=20.0, expected_net_return_bps=15.0,
        downside_risk_bps=40.0, confidence=0.7, execution_quality=ExecutionQuality.GOOD,
        reason_codes=(), explanation_paths=(),
    )


class _FakeMicro:
    """Configurable micro reasoner: raise / sleep / normal per symbol."""

    def __init__(self, *, raise_for=(), sleep_for=(), sleep_seconds=1.0, exit_for=()):
        self.raise_for = set(raise_for)
        self.sleep_for = set(sleep_for)
        self.sleep_seconds = sleep_seconds
        self.exit_for = set(exit_for)
        self.calls = []

    def reason(self, data: MicroReasoningInput):
        self.calls.append(data.symbol)
        if data.symbol in self.raise_for:
            raise RuntimeError(f"boom {data.symbol}")
        if data.symbol in self.sleep_for:
            time.sleep(self.sleep_seconds)
        return _micro_result(data.symbol, exit_=data.symbol in self.exit_for)


def _builder(symbol, macro_result):
    return MicroReasoningInput(timestamp=_now(), symbol=symbol,
                               allowed_micro_strategies=macro_result.allowed_micro_strategies,
                               blocked_micro_strategies=macro_result.blocked_micro_strategies)


def _coordinator(macro_result, micro, config=None):
    return OntologyCoordinator(
        macro_reasoner=_FakeMacro(macro_result),
        micro_reasoner=micro,
        config=config or CoordinatorConfig(max_parallel_symbols=4, worker_timeout_seconds=2.0),
    )


class TestDispatch:
    def test_macro_first_then_micro_for_candidates(self):
        coord = _coordinator(_macro_result(["A", "B", "C"]), _FakeMicro())
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder)
        assert {r.symbol for r in bundle.micro_results} == {"A", "B", "C"}
        assert bundle.macro_result.market_regime == MarketRegime.TREND_UP

    def test_block_buy_skips_new_candidates_but_keeps_holdings(self):
        micro = _FakeMicro(exit_for=("HELD",))
        coord = _coordinator(_macro_result(["A", "B"], blocks=True), micro)
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder, held_symbols=("HELD",))
        symbols = {r.symbol for r in bundle.micro_results}
        assert symbols == {"HELD"}          # new BUY candidates A/B skipped under BLOCK_BUY
        assert "HELD" in bundle.sell_reduce_candidates

    def test_held_symbol_always_evaluated(self):
        coord = _coordinator(_macro_result(["A"]), _FakeMicro(exit_for=("HELD",)))
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder, held_symbols=("HELD",))
        assert {"A", "HELD"} <= {r.symbol for r in bundle.micro_results}


class TestFailureIsolation:
    def test_worker_exception_does_not_crash_loop(self):
        micro = _FakeMicro(raise_for=("BAD",))
        coord = _coordinator(_macro_result(["A", "BAD", "C"]), micro)
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder)
        assert "BAD" in bundle.failed_symbols
        assert {"A", "C"} <= {r.symbol for r in bundle.micro_results}

    def test_worker_timeout_is_isolated(self):
        micro = _FakeMicro(sleep_for=("SLOW",), sleep_seconds=1.0)
        coord = _coordinator(_macro_result(["A", "SLOW", "C"]), micro,
                             CoordinatorConfig(max_parallel_symbols=4, worker_timeout_seconds=0.2))
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder)
        assert "SLOW" in bundle.failed_symbols
        assert {"A", "C"} <= {r.symbol for r in bundle.micro_results}

    def test_builder_failure_isolated(self):
        def bad_builder(symbol, macro_result):
            if symbol == "X":
                raise ValueError("build fail")
            return _builder(symbol, macro_result)

        coord = _coordinator(_macro_result(["A", "X", "C"]), _FakeMicro())
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=bad_builder)
        assert "X" in bundle.failed_symbols
        assert {"A", "C"} <= {r.symbol for r in bundle.micro_results}


class TestBoundedAndDeterministic:
    def test_all_results_returned_under_bound(self):
        symbols = [f"S{i}" for i in range(12)]
        coord = _coordinator(_macro_result(symbols), _FakeMicro(),
                             CoordinatorConfig(max_parallel_symbols=3, worker_timeout_seconds=2.0))
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder)
        assert len(bundle.micro_results) == 12

    def test_results_sorted_by_symbol(self):
        coord = _coordinator(_macro_result(["C", "A", "B"]), _FakeMicro())
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder)
        symbols = [r.symbol for r in bundle.micro_results]
        assert symbols == sorted(symbols)

    def test_sell_reduce_ranked_before_buy(self):
        coord = _coordinator(_macro_result(["BUYA"]), _FakeMicro(exit_for=("HELD",)))
        bundle = coord.run(MacroReasoningInput(timestamp=_now()), micro_input_builder=_builder, held_symbols=("HELD",))
        intents = bundle.ranked_trade_intents
        assert intents[0].symbol == "HELD" and intents[0].side in ("SELL", "REDUCE")
        assert intents[-1].side == "BUY"
