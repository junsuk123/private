"""Global trade arbiter — advisory ranking across many micro results.

Combines the macro result and all micro results into a ranked list of advisory
`RankedTradeIntent`s. SELL/REDUCE candidates (from existing holdings) are ranked
and processed BEFORE any new BUY candidate. BUY candidates are ranked by expected
net return, confidence, macro compatibility, liquidity, spread, and downside risk.

`RankedTradeIntent` is ADVISORY ONLY — it carries no broker-submission authority.
Final approval remains with RiskManager / ProfitabilityGate / FinalTradeGate,
invoked by the SharedLiveDecisionEngine downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.graph.macro_micro_common import IntentType
from app.graph.macro_reasoner import MacroReasoningResult
from app.graph.micro_reasoner import MicroReasoningResult
from app.graph.macro_micro_common import ExecutionQuality, ExitSignal, MicroRegime


@dataclass(frozen=True)
class RankedTradeIntent:
    intent_type: IntentType
    symbol: str
    side: str                    # "BUY" | "SELL" | "REDUCE"
    rank: int
    score: float
    expected_entry_price: float | None
    expected_exit_price: float | None
    expected_net_return_bps: float | None
    downside_risk_bps: float | None
    macro_regime: str
    micro_regime: str
    selected_strategy: str
    confidence: float
    reason_codes: tuple[str, ...]
    explanation_paths: tuple[dict, ...]
    source_result_refs: tuple[str, ...] = ()

    # RankedTradeIntent is advisory only: intentionally no order/broker fields.
    def as_dict(self) -> dict:
        def _r(v, n=3):
            return round(v, n) if isinstance(v, (int, float)) else v

        return {
            "intent_type": self.intent_type.value,
            "symbol": self.symbol,
            "side": self.side,
            "rank": self.rank,
            "score": _r(self.score, 4),
            "expected_entry_price": _r(self.expected_entry_price, 4),
            "expected_exit_price": _r(self.expected_exit_price, 4),
            "expected_net_return_bps": _r(self.expected_net_return_bps),
            "downside_risk_bps": _r(self.downside_risk_bps),
            "macro_regime": self.macro_regime,
            "micro_regime": self.micro_regime,
            "selected_strategy": self.selected_strategy,
            "confidence": _r(self.confidence, 4),
            "reason_codes": list(self.reason_codes),
            "explanation_paths": list(self.explanation_paths),
            "source_result_refs": list(self.source_result_refs),
        }


@dataclass(frozen=True)
class ArbiterConfig:
    net_return_weight: float = 1.0
    confidence_weight: float = 0.6
    liquidity_weight: float = 0.3
    downside_penalty_weight: float = 0.5
    spread_penalty_weight: float = 0.2


class GlobalTradeArbiter:
    def __init__(self, config: ArbiterConfig | None = None) -> None:
        self.config = config or ArbiterConfig()

    def rank(
        self,
        macro_result: MacroReasoningResult,
        micro_results: Sequence[MicroReasoningResult],
    ) -> dict[str, Any]:
        """Return {ranked_trade_intents, sell_reduce_candidates, buy_candidates, blocked_candidates}."""
        macro_regime = macro_result.market_regime.value

        exits: list[MicroReasoningResult] = []
        buys: list[MicroReasoningResult] = []
        blocked: list[MicroReasoningResult] = []
        for r in micro_results:
            if r.is_exit_candidate or r.micro_regime == MicroRegime.EXIT_DETERIORATION:
                exits.append(r)
            elif r.is_buy_candidate:
                # A positive net edge is NOT enough: a BUY whose execution quality is
                # WEAK/BLOCKED (thin liquidity or a spread that eats the alpha) is excluded
                # from the BUY ranking rather than ranked on edge alone.
                if r.execution_quality in (ExecutionQuality.WEAK, ExecutionQuality.BLOCKED):
                    blocked.append(r)
                else:
                    buys.append(r)
            else:
                blocked.append(r)

        ranked: list[RankedTradeIntent] = []
        rank = 0

        # --- SELL / REDUCE FIRST (capital protection precedes new risk). ---
        # Most-deteriorated / largest downside first.
        exits.sort(key=lambda r: (r.downside_risk_bps or 0.0), reverse=True)
        for r in exits:
            side = "SELL" if r.exit_signal in (ExitSignal.SELL_CANDIDATE, ExitSignal.TAKE_PROFIT, ExitSignal.TRAILING_STOP) else "REDUCE"
            intent_type = IntentType.SELL if side == "SELL" else IntentType.REDUCE
            ranked.append(self._intent(intent_type, side, r, macro_regime, rank, score=1_000_000.0 - rank))
            rank += 1

        # --- BUY SECOND, ranked by advisory score. ---
        scored = sorted(buys, key=lambda r: self._buy_score(r), reverse=True)
        for r in scored:
            ranked.append(self._intent(IntentType.BUY, "BUY", r, macro_regime, rank, score=self._buy_score(r)))
            rank += 1

        return {
            "ranked_trade_intents": tuple(ranked),
            "sell_reduce_candidates": tuple(r.symbol for r in exits),
            "buy_candidates": tuple(r.symbol for r in scored),
            "blocked_candidates": tuple(r.symbol for r in blocked),
        }

    # Execution-quality → (liquidity bonus, spread penalty) in [0,1]. The micro reasoner
    # collapses raw liquidity_score/spread_bps into this ExecutionQuality grade, so it is
    # the available proxy for both the liquidity_weight and spread_penalty_weight terms.
    _LIQUIDITY_BONUS = {
        ExecutionQuality.GOOD: 1.0,
        ExecutionQuality.ACCEPTABLE: 0.5,
        ExecutionQuality.WEAK: 0.0,
        ExecutionQuality.BLOCKED: 0.0,
    }
    _SPREAD_PENALTY = {
        ExecutionQuality.GOOD: 0.0,
        ExecutionQuality.ACCEPTABLE: 0.5,
        ExecutionQuality.WEAK: 1.0,
        ExecutionQuality.BLOCKED: 1.0,
    }

    def _buy_score(self, r: MicroReasoningResult) -> float:
        cfg = self.config
        net = float(r.expected_net_return_bps or 0.0)
        conf = float(r.confidence or 0.0)
        downside = float(r.downside_risk_bps or 0.0)
        liq = self._LIQUIDITY_BONUS.get(r.execution_quality, 0.0)
        spread = self._SPREAD_PENALTY.get(r.execution_quality, 0.5)
        return (
            cfg.net_return_weight * net
            + cfg.confidence_weight * conf * 100.0
            + cfg.liquidity_weight * liq * 100.0
            - cfg.spread_penalty_weight * spread * 100.0
            - cfg.downside_penalty_weight * downside * 0.1
        )

    def _intent(self, intent_type, side, r, macro_regime, rank, *, score) -> RankedTradeIntent:
        return RankedTradeIntent(
            intent_type=intent_type,
            symbol=r.symbol,
            side=side,
            rank=rank,
            score=score,
            expected_entry_price=r.expected_entry_price,
            expected_exit_price=r.expected_exit_price,
            expected_net_return_bps=r.expected_net_return_bps,
            downside_risk_bps=r.downside_risk_bps,
            macro_regime=macro_regime,
            micro_regime=r.micro_regime.value,
            selected_strategy=r.selected_strategy.value,
            confidence=r.confidence,
            reason_codes=r.reason_codes,
            explanation_paths=r.explanation_paths,
            source_result_refs=(r.symbol,),
        )
