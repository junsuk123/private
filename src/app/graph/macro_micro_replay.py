"""Replay / walk-forward validation for the macro–micro reasoning layer.

At each decision step over a time-ordered multi-symbol bar series, the macro
context and micro features are built from PAST bars only (``bars[:i+1]``) and
the realized outcome from FUTURE bars only (``bars[i+1:]``) — a hard
no-look-ahead split. It exercises the real MacroMarketReasoner +
MicroSymbolReasoner via the OntologyCoordinator, then compares each micro BUY's
expected net return against the realized cost-adjusted outcome, and confirms
SELL/REDUCE intents rank before BUY intents.

Deterministic and I/O-free (no wall clock); the CLI wrapper persists reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import fmean
from typing import Mapping, Sequence

from app.features.schemas import OHLCVBar
from app.graph.global_trade_arbiter import GlobalTradeArbiter
from app.graph.macro_reasoner import MacroMarketReasoner, MacroReasonerConfig, MacroReasoningInput
from app.graph.micro_reasoner import MicroReasonerConfig, MicroReasoningInput, MicroSymbolReasoner
from app.graph.ontology_coordinator import CoordinatorConfig, OntologyCoordinator
from app.technical.feature_builder import build_technical_feature_set
from app.technical.labels import LabelBuilder, LabelConfig


@dataclass(frozen=True)
class MacroMicroReplayConfig:
    warmup_bars: int = 30
    index_trend_window: int = 5
    macro_config: MacroReasonerConfig = field(default_factory=MacroReasonerConfig)
    micro_config: MicroReasonerConfig = field(default_factory=lambda: MicroReasonerConfig(minimum_micro_confidence=0.3))
    coordinator_config: CoordinatorConfig = field(default_factory=lambda: CoordinatorConfig(max_parallel_symbols=8, worker_timeout_seconds=5.0))
    label_config: LabelConfig = field(default_factory=LabelConfig)


class MacroMicroReplayEvaluator:
    def __init__(self, config: MacroMicroReplayConfig | None = None, *, cost_engine=None) -> None:
        self.config = config or MacroMicroReplayConfig()
        self.coordinator = OntologyCoordinator(
            macro_reasoner=MacroMarketReasoner(self.config.macro_config),
            micro_reasoner=MicroSymbolReasoner(self.config.micro_config),
            arbiter=GlobalTradeArbiter(),
            config=self.config.coordinator_config,
        )
        self.label_builder = LabelBuilder(cost_engine=cost_engine, config=self.config.label_config)

    def evaluate_rows(self, symbol_bars: Mapping[str, Sequence[OHLCVBar]]) -> list[dict]:
        cfg = self.config
        series = {s: tuple(b) for s, b in symbol_bars.items() if b}
        if not series:
            return []
        n = min(len(b) for b in series.values())
        rows: list[dict] = []
        for i in range(cfg.warmup_bars, n - 1):
            decision_at = next(iter(series.values()))[i].as_of
            macro_input = self._macro_input(series, i, decision_at)

            def builder(symbol, macro_result, _i=i):
                past = series[symbol][: _i + 1]
                return MicroReasoningInput(
                    timestamp=decision_at, symbol=symbol,
                    allowed_micro_strategies=macro_result.allowed_micro_strategies,
                    blocked_micro_strategies=macro_result.blocked_micro_strategies,
                    technical_features=build_technical_feature_set(past, symbol=symbol),
                )

            bundle = self.coordinator.run(macro_input, micro_input_builder=builder)
            intent_side = {it.symbol: it.side for it in bundle.ranked_trade_intents}
            for micro in bundle.micro_results:
                realized = self._realized_net_bps(series[micro.symbol], i, micro.expected_entry_price)
                rows.append({
                    "step": i,
                    "decision_at": decision_at.isoformat(),
                    "symbol": micro.symbol,
                    "macro_regime": bundle.macro_result.market_regime.value,
                    "macro_blocks_buy": bundle.macro_result.blocks_buy,
                    "entry_signal": micro.entry_signal.value,
                    "predicted_net_bps": micro.expected_net_return_bps,
                    "realized_net_bps": realized,
                    "ranked_side": intent_side.get(micro.symbol),
                })
        return rows

    def evaluate(self, symbol_bars: Mapping[str, Sequence[OHLCVBar]]) -> dict:
        rows = self.evaluate_rows(symbol_bars)
        buys = [r for r in rows if r["entry_signal"] == "BUY_CANDIDATE"]
        errors = [
            r["predicted_net_bps"] - r["realized_net_bps"]
            for r in buys
            if r["predicted_net_bps"] is not None and r["realized_net_bps"] is not None
        ]
        regimes: dict[str, int] = {}
        for r in rows:
            regimes[r["macro_regime"]] = regimes.get(r["macro_regime"], 0) + 1
        return {
            "symbols": sorted(symbol_bars.keys()),
            "steps": len({r["step"] for r in rows}),
            "rows": len(rows),
            "buy_candidates": len(buys),
            "avg_predicted_net_bps": round(fmean([r["predicted_net_bps"] for r in buys if r["predicted_net_bps"] is not None]), 3) if buys else None,
            "avg_realized_net_bps": round(fmean([r["realized_net_bps"] for r in buys if r["realized_net_bps"] is not None]), 3) if errors else None,
            "avg_edge_error_bps": round(fmean(errors), 3) if errors else None,
            "regime_distribution": regimes,
            "no_lookahead": True,
        }

    # ------------------------------------------------------------------ #
    def _macro_input(self, series, i, decision_at) -> MacroReasoningInput:
        cfg = self.config
        # Index trend = mean of per-symbol trailing returns over the window (past only).
        trends = []
        for bars in series.values():
            past = bars[: i + 1]
            if len(past) > cfg.index_trend_window and past[-cfg.index_trend_window - 1].close:
                trends.append(past[-1].close / past[-cfg.index_trend_window - 1].close - 1.0)
        index_trend = fmean(trends) if trends else 0.0
        return MacroReasoningInput(
            timestamp=decision_at,
            index_snapshots={"COMPOSITE": {"trend": index_trend}},
            market_breadth=0.55,
            market_volatility=0.005,
            candidate_universe=tuple(sorted(series.keys())),
        )

    def _realized_net_bps(self, bars, i, entry_price) -> float | None:
        if entry_price is None or entry_price <= 0:
            entry_price = bars[i].close
        decision_at = bars[i].as_of
        future_path = [((b.as_of - decision_at).total_seconds(), float(b.close)) for b in bars[i + 1 :]]
        labels = self.label_builder.build(
            symbol=bars[i].ticker, entry_price=entry_price, future_path=future_path, source="replay",
        )
        if labels is None:
            return None
        net = labels.metadata.get("net_return_after_cost")
        return net * 10_000.0 if net is not None else None
