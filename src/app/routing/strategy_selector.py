"""StrategySelectorV2 — compares every eligible strategy, including NO_TRADE.

    U_s = M_s * (mu_gross_s - C_s - lambda_d*D_s - lambda_u*sigma_s
                 + lambda_o*O_s + lambda_b*B_s)

    s* = argmax(U_NO_TRADE, U_1, ..., U_N)

Every term is computed by a component whose only job is that term:

===========  ==========================================================
``M_s``      ``OntologyStrategyMask``      hard eligibility, 0 or 1
``mu_gross``  utility predictor           GROSS bps, never net
``C_s``      ``TradingCostAdapter``       deterministic cost policy
``D_s``      utility predictor            expected downside bps
``sigma_s``  utility predictor            prediction uncertainty bps
``O_s``      ontology soft relations      compatibility evidence
``B_s``      ``StrategyBanditAdapter``    bounded realized-history correction
===========  ==========================================================

What this class does NOT do
--------------------------
It does not size, gate, price, or submit. It has no import path to ``RiskManager``,
``PositionSizer``, ``ProfitabilityGate``, ``ExecutionPricingPolicy``,
``LiveExecutionCoordinator`` or any broker client. Its output is a
:class:`StrategySelectionResult` — a *recommendation with its arithmetic attached* — and
whether that recommendation may become an order is decided entirely downstream.

``selected_strategy=None`` with ``decision="NO_TRADE"`` is a normal, successful result.

Term decomposition is stored, not just the total
------------------------------------------------
Every ranked candidate carries all seven terms. "the selector picked X" is not a usable
diagnosis; "X won because its bandit correction was -8.7 and its uncertainty penalty 8.2"
is, and it is also the only form in which the selector's own errors can be attributed to a
component later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from app.context.market_context import MarketContext
from app.ontology.strategy_eligibility import (
    StrategyEligibility,
    StrategyEligibilityEngine,
    StrategyEligibilityResult,
)
from app.routing.bandit_adapter import BanditCorrection, StrategyBanditAdapter
from app.routing.no_trade_policy import NO_TRADE_REASONS, NoTradePolicy, NoTradeVerdict
from app.routing.ontology_strategy_mask import OntologyStrategyMask
from app.routing.strategy_utility import (
    CompositeUtilityPredictor,
    CostEstimate,
    StrategyUtilityPrediction,
    TradingCostAdapter,
)
from app.strategy.proposal import StrategyProposal
from app.strategy.proposal_engine import StrategyProposalEngine
from app.strategy.registry import StrategyRegistry, default_strategy_registry
from app.strategy.spec import StrategyLifecycleState

__all__ = [
    "SELECTION_VERSION",
    "RankedStrategyCandidate",
    "StrategySelectionResult",
    "StrategySelectorV2",
    "UtilityWeights",
]

#: Stamped onto every result. Bump when the formula or the term set changes, so a stored
#: selection can be matched to the arithmetic that produced it.
SELECTION_VERSION = "selector-v2.0.0"

SELECTION_REASON_ENTRY_NOT_READY = "CANDIDATE_ENTRY_NOT_READY"
SELECTION_REASON_LIFECYCLE_NOT_LIVE = "CANDIDATE_LIFECYCLE_NOT_LIVE"
SELECTION_REASON_HARD_BLOCKED = "CANDIDATE_HARD_BLOCKED"
SELECTION_REASON_NO_PREDICTION = "CANDIDATE_NO_UTILITY_PREDICTION"


@dataclass(frozen=True)
class UtilityWeights:
    """The lambdas. Config, never hardcoded inside a model or an optimizer.

    ``lambda_d`` at 0.5 says a stop-out costs half its bps against the expected return —
    not 1.0, because the downside is conditional on failure while the return is an
    expectation over both outcomes, so charging it fully would double-count the loss the
    probability term already prices.

    ``lambda_o`` converts a unit-less compatibility score in ``[-1, 1]`` into bps, which is
    why it is the largest number here: 8bps of swing between "the ontology says this
    thesis fits the tape" and "it says the opposite".
    """

    lambda_downside: float = 0.5
    lambda_uncertainty: float = 1.0
    lambda_ontology_bps: float = 8.0
    lambda_bandit: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "UtilityWeights":
        payload = dict(values or {})

        def read(name: str, default: float) -> float:
            try:
                number = float(payload.get(name, default))
            except (TypeError, ValueError):
                return default
            return number if math.isfinite(number) else default

        return cls(
            lambda_downside=read("lambda_downside", 0.5),
            lambda_uncertainty=read("lambda_uncertainty", 1.0),
            lambda_ontology_bps=read("lambda_ontology_bps", 8.0),
            lambda_bandit=read("lambda_bandit", 1.0),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "lambda_downside": self.lambda_downside,
            "lambda_uncertainty": self.lambda_uncertainty,
            "lambda_ontology_bps": self.lambda_ontology_bps,
            "lambda_bandit": self.lambda_bandit,
        }


@dataclass(frozen=True)
class RankedStrategyCandidate:
    """One strategy's full utility decomposition."""

    strategy_id: str
    symbol: str
    eligible: bool
    entry_ready: bool
    expected_gross_return_bps: float
    expected_cost_bps: float
    downside_penalty_bps: float
    uncertainty_penalty_bps: float
    ontology_adjustment_bps: float
    bandit_adjustment_bps: float
    final_utility_bps: float
    expected_holding_seconds: float
    probability_profit: float
    cost_measured: bool
    utility_source: str
    model_version: str
    reason_codes: tuple[str, ...] = ()
    proposal_id: str = ""
    lifecycle_state: str = ""

    @property
    def expected_net_return_bps(self) -> float:
        return self.expected_gross_return_bps - self.expected_cost_bps

    @property
    def lower_confidence_bound_bps(self) -> float:
        """``net - uncertainty``. The number that should be compared against cost."""
        return self.expected_net_return_bps - self.uncertainty_penalty_bps

    @property
    def selectable(self) -> bool:
        """Eligible, triggered, and authorised by lifecycle for a live order.

        ``DEGRADED`` and ``SHADOW`` strategies are ranked (their numbers are evidence) but
        are not selectable, so a demoted strategy cannot be re-armed by the ranking.
        """
        return bool(
            self.eligible
            and self.entry_ready
            and SELECTION_REASON_LIFECYCLE_NOT_LIVE not in self.reason_codes
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "eligible": self.eligible,
            "entry_ready": self.entry_ready,
            "selectable": self.selectable,
            "expected_gross_return_bps": round(self.expected_gross_return_bps, 3),
            "expected_cost_bps": round(self.expected_cost_bps, 3),
            "expected_net_return_bps": round(self.expected_net_return_bps, 3),
            "downside_penalty_bps": round(self.downside_penalty_bps, 3),
            "uncertainty_penalty_bps": round(self.uncertainty_penalty_bps, 3),
            "ontology_adjustment_bps": round(self.ontology_adjustment_bps, 3),
            "bandit_adjustment_bps": round(self.bandit_adjustment_bps, 3),
            "final_utility_bps": round(self.final_utility_bps, 3),
            "lower_confidence_bound_bps": round(self.lower_confidence_bound_bps, 3),
            "expected_holding_seconds": round(self.expected_holding_seconds, 1),
            "probability_profit": round(self.probability_profit, 4),
            "cost_measured": self.cost_measured,
            "utility_source": self.utility_source,
            "model_version": self.model_version,
            "lifecycle_state": self.lifecycle_state,
            "proposal_id": self.proposal_id,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class StrategySelectionResult:
    """The selector's verdict. ``selected_strategy=None`` is a normal outcome."""

    symbol: str
    context_id: str
    selected_strategy: str | None
    decision: str  # SELECT | NO_TRADE
    utility: float | None
    ranked_candidates: tuple[RankedStrategyCandidate, ...]
    reason_codes: tuple[str, ...]
    selection_version: str = SELECTION_VERSION
    no_trade_verdict: NoTradeVerdict | None = None
    weights: UtilityWeights = field(default_factory=UtilityWeights)
    evaluated_at: datetime | None = None
    #: Hard-block reasons for everything the mask removed, so a dashboard can show why a
    #: strategy is absent instead of just not showing it.
    blocked: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The proposals that were ranked, carried by value.
    #:
    #: Needed for two reasons the ranking alone cannot serve: the data-integrity contract
    #: requires every selection to record its ``proposal_id``s, and the counterfactual engine
    #: needs each proposal's point-in-time entry reference and barriers to open a shadow
    #: position. Reconstructing those from the ranking is impossible, and re-deriving them
    #: from a later read is the leak the shadow journal exists to prevent.
    proposals: tuple[StrategyProposal, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_no_trade(self) -> bool:
        return self.decision == "NO_TRADE"

    @property
    def selected_candidate(self) -> RankedStrategyCandidate | None:
        if self.selected_strategy is None:
            return None
        return next(
            (
                item
                for item in self.ranked_candidates
                if item.strategy_id == self.selected_strategy
            ),
            None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "context_id": self.context_id,
            "selected_strategy": self.selected_strategy,
            "decision": self.decision,
            "utility": round(self.utility, 3) if self.utility is not None else None,
            "selection_version": self.selection_version,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "weights": self.weights.as_dict(),
            "no_trade": self.no_trade_verdict.as_dict() if self.no_trade_verdict else None,
            "ranked_candidates": [item.as_dict() for item in self.ranked_candidates],
            "proposals": [item.as_dict() for item in self.proposals],
            "blocked": {key: list(value) for key, value in dict(self.blocked).items()},
            "reason_codes": list(self.reason_codes),
            "diagnostics": dict(self.diagnostics),
        }


class StrategySelectorV2:
    """Eligibility -> proposals -> utility -> cost -> bandit -> ranking -> NO_TRADE."""

    def __init__(
        self,
        *,
        registry: StrategyRegistry | None = None,
        mask: OntologyStrategyMask | None = None,
        proposal_engine: StrategyProposalEngine | None = None,
        utility_predictor: Any | None = None,
        cost_adapter: TradingCostAdapter | None = None,
        bandit_adapter: StrategyBanditAdapter | None = None,
        no_trade_policy: NoTradePolicy | None = None,
        weights: UtilityWeights | None = None,
        bandit_enabled: bool = True,
    ) -> None:
        self._registry = registry or default_strategy_registry()
        self._mask = mask or OntologyStrategyMask(
            engine=StrategyEligibilityEngine(registry=self._registry)
        )
        self._proposals = proposal_engine or StrategyProposalEngine(registry=self._registry)
        self._utility = utility_predictor or CompositeUtilityPredictor()
        self._costs = cost_adapter or TradingCostAdapter()
        self._bandit = bandit_adapter or StrategyBanditAdapter()
        self._no_trade = no_trade_policy or NoTradePolicy()
        self._weights = weights or UtilityWeights()
        self._bandit_enabled = bool(bandit_enabled)

    # -- public API --------------------------------------------------------- #
    def select(
        self,
        context: MarketContext,
        *,
        election_inputs: Mapping[str, Any] | None = None,
        gnn_rows: Iterable[Any] = (),
        macro_allowed: Iterable[str] = (),
        macro_blocked: Iterable[str] = (),
        strategy_ids: Sequence[str] | None = None,
        now: datetime | None = None,
    ) -> StrategySelectionResult:
        evaluated_at = now or context.captured_at
        inputs = dict(election_inputs or {})

        eligibility = self._mask.evaluate(
            context,
            election_inputs=inputs,
            strategy_ids=strategy_ids,
            macro_allowed=macro_allowed,
            macro_blocked=macro_blocked,
        )
        by_id = eligibility.by_id()
        eligible_ids = eligibility.eligible_ids
        blocked = {
            item.strategy_id: item.hard_block_reasons
            for item in eligibility.blocked
        }

        if not eligible_ids:
            return self._no_trade_result(
                context=context,
                evaluated_at=evaluated_at,
                candidates=(),
                eligibility=eligibility,
                blocked=blocked,
                entry_ready_count=0,
                extra_reasons=(NO_TRADE_REASONS.ALL_HARD_BLOCKED,),
            )

        # Cost first: the proposal engine's geometry is sized against the measured cost of
        # the venue actually being traded, and the utility needs the same number.
        reference_cost = self._costs.estimate(
            strategy_id="__reference__",
            symbol=context.symbol_id,
            market=context.market,
            reference_price=context.symbol.reference_price,
            spread_bps=context.microstructure.spread_bps,
        )
        proposal_result = self._proposals.evaluate(
            context,
            eligible_strategy_ids=eligible_ids,
            election_inputs=inputs,
            round_trip_cost_bps=(
                reference_cost.expected_cost_bps if reference_cost.measured else None
            ),
        )
        proposals = proposal_result.proposals
        if not proposals:
            return self._no_trade_result(
                context=context,
                evaluated_at=evaluated_at,
                candidates=(),
                eligibility=eligibility,
                blocked=blocked,
                entry_ready_count=0,
                extra_reasons=(NO_TRADE_REASONS.NO_PROPOSALS,),
            )

        costs = self._costs_for(context, proposals)
        predictions = self._predict(context, proposals, costs, gnn_rows)
        corrections = self._corrections(context, predictions, proposals, evaluated_at)

        candidates = self._rank(
            context=context,
            proposals=proposals,
            predictions=predictions,
            costs=costs,
            corrections=corrections,
            eligibility=by_id,
        )
        entry_ready_count = sum(1 for item in candidates if item.entry_ready)
        selectable = tuple(item for item in candidates if item.selectable)

        verdict = self._no_trade.evaluate(
            market=context.market,
            candidates=selectable,
            feature_completeness=context.data_quality.feature_completeness,
            eligible_count=len(eligible_ids),
            entry_ready_count=entry_ready_count,
            coverage_gap=self._coverage_gap(candidates),
        )
        if verdict.no_trade or not selectable:
            return self._no_trade_result(
                context=context,
                evaluated_at=evaluated_at,
                candidates=candidates,
                eligibility=eligibility,
                blocked=blocked,
                entry_ready_count=entry_ready_count,
                verdict=verdict,
                proposals=proposals,
            )

        winner = max(selectable, key=lambda item: item.final_utility_bps)
        return StrategySelectionResult(
            symbol=context.symbol_id,
            context_id=context.context_id,
            selected_strategy=winner.strategy_id,
            decision="SELECT",
            utility=winner.final_utility_bps,
            ranked_candidates=candidates,
            reason_codes=("STRATEGY_SELECTED", f"UTILITY_SOURCE:{winner.utility_source}"),
            no_trade_verdict=verdict,
            weights=self._weights,
            evaluated_at=evaluated_at,
            blocked=blocked,
            proposals=proposals,
            diagnostics=self._diagnostics(
                context, eligibility, proposal_result, reference_cost
            ),
        )

    # -- internals ---------------------------------------------------------- #
    def _costs_for(
        self, context: MarketContext, proposals: Sequence[StrategyProposal]
    ) -> dict[str, CostEstimate]:
        return {
            proposal.strategy_id: self._costs.estimate(
                strategy_id=proposal.strategy_id,
                symbol=context.symbol_id,
                market=context.market,
                reference_price=proposal.reference_entry_price
                or context.symbol.reference_price,
                spread_bps=context.microstructure.spread_bps,
                is_short=proposal.is_short,
            )
            for proposal in proposals
        }

    def _predict(
        self,
        context: MarketContext,
        proposals: Sequence[StrategyProposal],
        costs: Mapping[str, CostEstimate],
        gnn_rows: Iterable[Any],
    ) -> dict[str, StrategyUtilityPrediction]:
        rows = tuple(gnn_rows or ())
        try:
            predicted = self._utility.predict(context, proposals, costs, rows=rows)
        except TypeError:
            # A predictor that does not accept GNN rows (the heuristic, or an injected
            # test double) still satisfies the protocol.
            predicted = self._utility.predict(context, proposals, costs)
        return {item.strategy_id: item for item in predicted}

    def _corrections(
        self,
        context: MarketContext,
        predictions: Mapping[str, StrategyUtilityPrediction],
        proposals: Sequence[StrategyProposal],
        now: datetime,
    ) -> dict[str, BanditCorrection]:
        if not self._bandit_enabled:
            return {}
        directions = {item.strategy_id: item.direction for item in proposals}
        try:
            return self._bandit.correct_all(
                tuple(predictions.values()),
                market=context.market,
                regime=context.macro.market_regime,
                volatility_percentile=context.macro.volatility_percentile,
                change_point_probability=context.macro.change_point_probability or 0.0,
                directions=directions,
                symbol=context.symbol_id,
                now=now,
            )
        except Exception:  # noqa: BLE001 - no correction rather than no selection.
            return {}

    def _rank(
        self,
        *,
        context: MarketContext,
        proposals: Sequence[StrategyProposal],
        predictions: Mapping[str, StrategyUtilityPrediction],
        costs: Mapping[str, CostEstimate],
        corrections: Mapping[str, BanditCorrection],
        eligibility: Mapping[str, StrategyEligibility],
    ) -> tuple[RankedStrategyCandidate, ...]:
        weights = self._weights
        ranked: list[RankedStrategyCandidate] = []
        for proposal in proposals:
            strategy_id = proposal.strategy_id
            prediction = predictions.get(strategy_id)
            eligible = eligibility.get(strategy_id)
            spec = self._registry.get(strategy_id)
            reasons: list[str] = list(proposal.strategy_reason_codes[:6])

            if prediction is None:
                reasons.append(SELECTION_REASON_NO_PREDICTION)
            if not proposal.entry_ready:
                reasons.append(SELECTION_REASON_ENTRY_NOT_READY)
            lifecycle = spec.lifecycle_state if spec is not None else StrategyLifecycleState.RESEARCH
            if not lifecycle.is_live_candidate:
                reasons.append(SELECTION_REASON_LIFECYCLE_NOT_LIVE)

            cost = costs.get(strategy_id)
            cost_bps = (
                prediction.expected_cost_bps
                if prediction is not None
                else (cost.expected_cost_bps if cost is not None else 0.0)
            )
            gross = prediction.expected_gross_return_bps if prediction is not None else 0.0
            downside = prediction.expected_downside_bps if prediction is not None else 0.0
            uncertainty = prediction.uncertainty_bps if prediction is not None else 0.0
            ontology_score = eligible.compatibility_score if eligible is not None else 0.0
            correction = corrections.get(strategy_id)

            downside_penalty = weights.lambda_downside * downside
            uncertainty_penalty = weights.lambda_uncertainty * uncertainty
            ontology_adjustment = weights.lambda_ontology_bps * ontology_score
            bandit_adjustment = (
                weights.lambda_bandit * correction.correction_bps
                if correction is not None
                else 0.0
            )
            mask = eligible.mask if eligible is not None else 0.0
            utility = mask * (
                gross
                - cost_bps
                - downside_penalty
                - uncertainty_penalty
                + ontology_adjustment
                + bandit_adjustment
            )
            if correction is not None:
                reasons.extend(correction.reason_codes)
            if prediction is not None:
                reasons.extend(prediction.reason_codes)

            ranked.append(
                RankedStrategyCandidate(
                    strategy_id=strategy_id,
                    symbol=context.symbol_id,
                    eligible=bool(eligible.eligible) if eligible is not None else False,
                    entry_ready=proposal.entry_ready,
                    expected_gross_return_bps=gross,
                    expected_cost_bps=cost_bps,
                    downside_penalty_bps=downside_penalty,
                    uncertainty_penalty_bps=uncertainty_penalty,
                    ontology_adjustment_bps=ontology_adjustment,
                    bandit_adjustment_bps=bandit_adjustment,
                    final_utility_bps=utility,
                    expected_holding_seconds=(
                        prediction.expected_holding_seconds
                        if prediction is not None
                        else float(proposal.expected_horizon_seconds or 0)
                    ),
                    probability_profit=(
                        prediction.probability_profit if prediction is not None else 0.0
                    ),
                    cost_measured=bool(cost.measured) if cost is not None else False,
                    utility_source=(
                        prediction.source if prediction is not None else "none"
                    ),
                    model_version=(
                        prediction.model_version if prediction is not None else "none"
                    ),
                    reason_codes=tuple(dict.fromkeys(reasons)),
                    proposal_id=proposal.proposal_id,
                    lifecycle_state=str(lifecycle),
                )
            )
        # Highest utility first, then strategy id so the order is stable across cycles
        # with identical numbers — an unstable ranking makes churn look like a signal.
        ranked.sort(key=lambda item: (-item.final_utility_bps, item.strategy_id))
        return tuple(ranked)

    @staticmethod
    def _coverage_gap(candidates: Sequence[RankedStrategyCandidate]) -> bool:
        """No entry-ready candidate is authorised for live selection.

        This is the ``STRATEGY_COVERAGE_GAP`` condition: strategies existed, some even
        fired, but none that fired has the validation standing to be selected. Forcing the
        nearest one would be exactly the ``forced_selection`` failure.
        """
        return not any(
            item.entry_ready
            and SELECTION_REASON_LIFECYCLE_NOT_LIVE not in item.reason_codes
            for item in candidates
        )

    def _no_trade_result(
        self,
        *,
        context: MarketContext,
        evaluated_at: datetime,
        candidates: tuple[RankedStrategyCandidate, ...],
        eligibility: StrategyEligibilityResult,
        blocked: Mapping[str, tuple[str, ...]],
        entry_ready_count: int,
        verdict: NoTradeVerdict | None = None,
        extra_reasons: tuple[str, ...] = (),
        proposals: tuple[StrategyProposal, ...] = (),
    ) -> StrategySelectionResult:
        if verdict is None:
            verdict = self._no_trade.evaluate(
                market=context.market,
                candidates=tuple(item for item in candidates if item.selectable),
                feature_completeness=context.data_quality.feature_completeness,
                eligible_count=len(eligibility.eligible_ids),
                entry_ready_count=entry_ready_count,
                coverage_gap=self._coverage_gap(candidates),
            )
        reasons = tuple(dict.fromkeys(("NO_TRADE", *verdict.reason_codes, *extra_reasons)))
        return StrategySelectionResult(
            symbol=context.symbol_id,
            context_id=context.context_id,
            selected_strategy=None,
            decision="NO_TRADE",
            utility=None,
            ranked_candidates=candidates,
            reason_codes=reasons,
            no_trade_verdict=verdict,
            weights=self._weights,
            evaluated_at=evaluated_at,
            blocked=blocked,
            proposals=proposals,
            diagnostics=self._diagnostics(context, eligibility, None, None),
        )

    def _diagnostics(
        self,
        context: MarketContext,
        eligibility: StrategyEligibilityResult,
        proposal_result: Any | None,
        reference_cost: CostEstimate | None,
    ) -> dict[str, Any]:
        return {
            "context": {
                "context_id": context.context_id,
                "captured_at": context.captured_at.isoformat(),
                "market": context.market,
                "market_regime": context.macro.market_regime,
                "session_phase": context.temporal.session_phase,
                "feature_completeness": context.data_quality.feature_completeness,
                "spread_bps": context.microstructure.spread_bps,
                "liquidity_score": context.microstructure.liquidity_score,
                "reason_codes": list(context.reason_codes),
            },
            "eligible_count": len(eligibility.eligible_ids),
            "blocked_count": len(eligibility.blocked),
            "reference_cost": reference_cost.as_dict() if reference_cost else None,
            "proposal_skips": dict(getattr(proposal_result, "skipped", {}) or {}),
            "no_trade_policy": self._no_trade.config.as_dict(),
            "bandit_enabled": self._bandit_enabled,
        }
