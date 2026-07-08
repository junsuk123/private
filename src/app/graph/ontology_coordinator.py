"""Ontology coordinator — macro-first, then bounded-parallel micro reasoning.

Runs MacroMarketReasoner once, then dispatches MicroSymbolReasoner in bounded
parallel for macro-selected candidates (plus held symbols, which are always
evaluated for SELL/REDUCE even under macro BLOCK_BUY). Worker failures and
timeouts are isolated so a single symbol can never crash the trading loop.
Results are aggregated into a MacroMicroReasoningBundle and ranked by the
GlobalTradeArbiter into advisory intents.

CPU-only safe: uses a ThreadPoolExecutor (no NPU requirement). NPU/OpenVINO is
used only where the reused technical/model layer already supports it.
"""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, TimeoutError as FuturesTimeout, wait
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from app.graph.global_trade_arbiter import GlobalTradeArbiter, RankedTradeIntent
from app.graph.macro_reasoner import (
    MacroMarketReasoner,
    MacroReasoningInput,
    MacroReasoningResult,
)
from app.graph.micro_reasoner import (
    MicroReasoningInput,
    MicroReasoningResult,
    MicroSymbolReasoner,
)

logger = logging.getLogger(__name__)

# builder(symbol, macro_result) -> MicroReasoningInput | None  (None => skip symbol)
MicroInputBuilder = Callable[[str, MacroReasoningResult], "MicroReasoningInput | None"]


@dataclass(frozen=True)
class MacroMicroReasoningBundle:
    timestamp: datetime
    macro_result: MacroReasoningResult
    micro_results: tuple[MicroReasoningResult, ...]
    failed_symbols: tuple[str, ...]
    ranked_trade_intents: tuple[RankedTradeIntent, ...]
    sell_reduce_candidates: tuple[str, ...]
    buy_candidates: tuple[str, ...]
    blocked_candidates: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "macro_result": self.macro_result.as_dict(),
            "micro_results": [r.as_dict() for r in self.micro_results],
            "failed_symbols": list(self.failed_symbols),
            "ranked_trade_intents": [i.as_dict() for i in self.ranked_trade_intents],
            "sell_reduce_candidates": list(self.sell_reduce_candidates),
            "buy_candidates": list(self.buy_candidates),
            "blocked_candidates": list(self.blocked_candidates),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class CoordinatorConfig:
    max_parallel_symbols: int = 20
    worker_timeout_seconds: float = 3.0


class ParallelMicroReasoningPool:
    """Bounded-parallel micro reasoning with per-worker timeout + failure isolation."""

    def __init__(self, reasoner: MicroSymbolReasoner, config: CoordinatorConfig | None = None) -> None:
        self.reasoner = reasoner
        self.config = config or CoordinatorConfig()

    def run(self, inputs: Sequence[MicroReasoningInput]) -> tuple[list[MicroReasoningResult], list[str]]:
        results: list[MicroReasoningResult] = []
        failed: list[str] = []
        if not inputs:
            return results, failed
        max_workers = max(1, min(self.config.max_parallel_symbols, len(inputs)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="micro") as pool:
            future_to_symbol = {pool.submit(self.reasoner.reason, inp): inp.symbol for inp in inputs}
            for future, symbol in list(future_to_symbol.items()):
                try:
                    results.append(future.result(timeout=self.config.worker_timeout_seconds))
                except FuturesTimeout:
                    logger.warning("micro worker timeout: %s", symbol)
                    failed.append(symbol)
                    future.cancel()
                except Exception as exc:  # noqa: BLE001 - one symbol must not crash the loop.
                    logger.warning("micro worker failed: %s (%s)", symbol, exc)
                    failed.append(symbol)
        # Deterministic ordering by symbol.
        results.sort(key=lambda r: r.symbol)
        failed.sort()
        return results, failed


class OntologyCoordinator:
    def __init__(
        self,
        *,
        macro_reasoner: MacroMarketReasoner | None = None,
        micro_reasoner: MicroSymbolReasoner | None = None,
        arbiter: GlobalTradeArbiter | None = None,
        config: CoordinatorConfig | None = None,
    ) -> None:
        self.macro_reasoner = macro_reasoner or MacroMarketReasoner()
        self.micro_reasoner = micro_reasoner or MicroSymbolReasoner()
        self.arbiter = arbiter or GlobalTradeArbiter()
        self.config = config or CoordinatorConfig()
        self.pool = ParallelMicroReasoningPool(self.micro_reasoner, self.config)

    def run(
        self,
        macro_input: MacroReasoningInput,
        *,
        micro_input_builder: MicroInputBuilder,
        held_symbols: Sequence[str] = (),
    ) -> MacroMicroReasoningBundle:
        macro_result = self.macro_reasoner.reason(macro_input)

        # Held symbols are ALWAYS evaluated (for SELL/REDUCE), even under BLOCK_BUY.
        # New BUY candidates are only dispatched when macro does not block buys.
        symbols: list[str] = list(dict.fromkeys(str(s) for s in held_symbols if str(s)))
        if not macro_result.blocks_buy:
            for sym in macro_result.candidate_symbols:
                if sym not in symbols:
                    symbols.append(sym)

        inputs: list[MicroReasoningInput] = []
        build_failures: list[str] = []
        for sym in symbols:
            try:
                inp = micro_input_builder(sym, macro_result)
            except Exception as exc:  # noqa: BLE001 - builder failure isolates to the symbol.
                logger.warning("micro input build failed: %s (%s)", sym, exc)
                build_failures.append(sym)
                continue
            if inp is not None:
                inputs.append(inp)

        micro_results, worker_failures = self.pool.run(inputs)
        ranked = self.arbiter.rank(macro_result, micro_results)

        return MacroMicroReasoningBundle(
            timestamp=macro_input.timestamp,
            macro_result=macro_result,
            micro_results=tuple(micro_results),
            failed_symbols=tuple(sorted(set(build_failures) | set(worker_failures))),
            ranked_trade_intents=ranked["ranked_trade_intents"],
            sell_reduce_candidates=ranked["sell_reduce_candidates"],
            buy_candidates=ranked["buy_candidates"],
            blocked_candidates=ranked["blocked_candidates"],
            diagnostics={
                "dispatched": len(inputs),
                "macro_blocks_buy": macro_result.blocks_buy,
                "held_symbols": list(symbols),
            },
        )
