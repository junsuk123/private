"""Runs the catalogued algorithms against one context and returns proposals.

This is the only place a ``TradingAlgorithm.entry()`` is called on behalf of the V2
selector, and it is deliberately narrow:

* **Mask first.** Only strategies the caller passes as eligible are evaluated. The
  ontology mask is cheap (declared requirements, no market data, no model) and the
  algorithms are not, so shrinking the candidate set before running any of them is what
  keeps the realtime loop's latency flat as the catalogue grows.
* **One feature object.** Features are rebuilt once from ``context.feature_snapshot`` and
  shared by every algorithm in the cycle. Previously each evaluation path rebuilt its
  own ``TechnicalFeatureSet``.
* **No authority.** The engine imports no broker, no coordinator, no risk manager, no
  position sizer and no profitability gate — the module has no path to any of them. An
  algorithm raising is recorded as ``STRATEGY_ENTRY_EVALUATION_ERROR`` and yields a
  not-ready proposal; it never becomes a trade and never stops the cycle.

Exit geometry
-------------
Target and stop come from ``app.strategy.exit_geometry``, the existing single authority
that also derives the training labels. ``round_trip_cost_bps`` is passed IN by the
caller (which owns the cost engine) rather than looked up here, so a strategy still
cannot reach cost policy on its own while the geometry stays sized against the venue
actually being traded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping, Sequence

from app.context.market_context import MarketContext
from app.strategy.proposal import StrategyProposal, new_proposal_id
from app.strategy.registry import StrategyRegistry, default_strategy_registry
from app.strategy.spec import StrategySpec

__all__ = [
    "PROPOSAL_ALGORITHM_MISSING",
    "PROPOSAL_ENTRY_ERROR",
    "PROPOSAL_NOT_ELIGIBLE",
    "PROPOSAL_NO_FEATURES",
    "PROPOSAL_NO_REFERENCE_PRICE",
    "ProposalEngineResult",
    "StrategyProposalEngine",
]

PROPOSAL_ALGORITHM_MISSING = "STRATEGY_IMPLEMENTATION_MISSING"
PROPOSAL_ENTRY_ERROR = "STRATEGY_ENTRY_EVALUATION_ERROR"
PROPOSAL_NOT_ELIGIBLE = "STRATEGY_NOT_ELIGIBLE"
PROPOSAL_NO_FEATURES = "STRATEGY_NO_FEATURE_SNAPSHOT"
PROPOSAL_NO_REFERENCE_PRICE = "STRATEGY_NO_REFERENCE_PRICE"


@dataclass(frozen=True)
class ProposalEngineResult:
    context_id: str
    symbol: str
    proposals: tuple[StrategyProposal, ...]
    #: Ids the caller marked eligible but which produced no proposal at all, with the
    #: reason. Reported rather than dropped: a silent omission is indistinguishable from
    #: a strategy that declined, and those need different responses.
    skipped: Mapping[str, str]

    @property
    def entry_ready(self) -> tuple[StrategyProposal, ...]:
        return tuple(item for item in self.proposals if item.entry_ready)

    @property
    def selectable(self) -> tuple[StrategyProposal, ...]:
        return tuple(item for item in self.proposals if item.selectable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "symbol": self.symbol,
            "proposals": [item.as_dict() for item in self.proposals],
            "skipped": dict(self.skipped),
        }


class StrategyProposalEngine:
    """Evaluates eligible strategies against one :class:`MarketContext`."""

    def __init__(
        self,
        *,
        registry: StrategyRegistry | None = None,
        algorithm_registry: Mapping[str, Any] | None = None,
    ) -> None:
        self._registry = registry or default_strategy_registry()
        # Built once. ``build_algorithm_registry`` re-reads YAML and instantiates every
        # algorithm; doing that per cycle per symbol was measurable overhead in the
        # existing ``_mechanical_entry_verdict``, which calls ``get_algorithm`` (and
        # therefore rebuilds the whole registry) once per candidate strategy.
        #
        # ``is not None`` rather than truthiness: an explicitly EMPTY registry is a valid
        # injection meaning "no algorithms are available", and reading it as "not supplied"
        # would silently substitute the real one.
        self._algorithms = (
            dict(algorithm_registry) if algorithm_registry is not None else None
        )

    # -- public API --------------------------------------------------------- #
    def evaluate(
        self,
        context: MarketContext,
        *,
        eligible_strategy_ids: Sequence[str] | Iterable[str],
        election_inputs: Mapping[str, Any] | None = None,
        features: Any = None,
        round_trip_cost_bps: float | None = None,
    ) -> ProposalEngineResult:
        eligible = tuple(
            dict.fromkeys(str(item or "").strip().lower() for item in eligible_strategy_ids if item)
        )
        skipped: dict[str, str] = {}
        if not eligible:
            return ProposalEngineResult(
                context_id=context.context_id,
                symbol=context.symbol_id,
                proposals=(),
                skipped=skipped,
            )

        feature_set = features if features is not None else self._features_from(context)
        if feature_set is None:
            return ProposalEngineResult(
                context_id=context.context_id,
                symbol=context.symbol_id,
                proposals=(),
                skipped={strategy_id: PROPOSAL_NO_FEATURES for strategy_id in eligible},
            )

        algorithms = self._algorithm_registry()
        proposals: list[StrategyProposal] = []
        for strategy_id in eligible:
            spec = self._registry.get(strategy_id)
            if spec is None:
                skipped[strategy_id] = PROPOSAL_NOT_ELIGIBLE
                continue
            algorithm = algorithms.get(strategy_id)
            if algorithm is None:
                skipped[strategy_id] = PROPOSAL_ALGORITHM_MISSING
                continue
            proposals.append(
                self._propose(
                    spec=spec,
                    algorithm=algorithm,
                    context=context,
                    features=feature_set,
                    election_inputs=election_inputs or {},
                    round_trip_cost_bps=round_trip_cost_bps,
                )
            )
        return ProposalEngineResult(
            context_id=context.context_id,
            symbol=context.symbol_id,
            proposals=tuple(proposals),
            skipped=skipped,
        )

    # -- internals ---------------------------------------------------------- #
    def _algorithm_registry(self) -> Mapping[str, Any]:
        if self._algorithms is None:
            from app.technical.strategy_algorithms import build_algorithm_registry

            self._algorithms = build_algorithm_registry()
        return self._algorithms

    def _propose(
        self,
        *,
        spec: StrategySpec,
        algorithm: Any,
        context: MarketContext,
        features: Any,
        election_inputs: Mapping[str, Any],
        round_trip_cost_bps: float | None,
    ) -> StrategyProposal:
        decision, reasons = self._entry_decision(
            spec=spec,
            algorithm=algorithm,
            context=context,
            features=features,
            election_inputs=election_inputs,
        )
        triggered = bool(decision.get("triggered", False))
        horizon = int(decision.get("horizon_seconds") or spec.horizon_seconds)
        edge_bps = _finite(decision.get("expected_edge_bps"))
        reference = context.symbol.reference_price or _finite(
            election_inputs.get("reference_price")
        )
        if reference is None:
            reasons = (*reasons, PROPOSAL_NO_REFERENCE_PRICE)

        target_price, stop_price = self._geometry_prices(
            spec=spec,
            reference=reference,
            spread_bps=context.microstructure.spread_bps,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        return StrategyProposal(
            proposal_id=new_proposal_id(),
            context_id=context.context_id,
            strategy_id=spec.strategy_id,
            symbol=context.symbol_id,
            eligible=True,
            entry_ready=triggered,
            raw_signal_strength=_finite(decision.get("score")) or 0.0,
            confidence=_finite(decision.get("confidence")) or 0.0,
            expected_horizon_seconds=horizon,
            reference_entry_price=reference,
            target_price=target_price,
            stop_price=stop_price,
            expected_gross_edge_bps=edge_bps,
            strategy_reason_codes=(
                *tuple(str(code) for code in (decision.get("reason_codes") or ())),
                *reasons,
            ),
            feature_snapshot=context.feature_snapshot,
            direction=spec.direction,
            proposed_at=context.captured_at,
            diagnostics=dict(decision.get("diagnostics") or {}),
        )

    def _entry_decision(
        self,
        *,
        spec: StrategySpec,
        algorithm: Any,
        context: MarketContext,
        features: Any,
        election_inputs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        try:
            from app.technical.strategy_algorithms import ElectionContext

            payload = self._election_payload(
                spec=spec,
                context=context,
                election_inputs=election_inputs,
                election_context_type=ElectionContext,
            )
            decision = algorithm.entry(features, ElectionContext(**payload))
            return decision.as_dict(), ()
        except Exception as exc:  # noqa: BLE001 - a broken algorithm fails closed.
            return (
                {
                    "strategy_id": spec.strategy_id,
                    "triggered": False,
                    "score": 0.0,
                    "confidence": 0.0,
                    "expected_edge_bps": 0.0,
                    "horizon_seconds": spec.horizon_seconds,
                    "reason_codes": [f"{PROPOSAL_ENTRY_ERROR}:{type(exc).__name__}"],
                    "diagnostics": {},
                },
                (),
            )

    @staticmethod
    def _election_payload(
        *,
        spec: StrategySpec,
        context: MarketContext,
        election_inputs: Mapping[str, Any],
        election_context_type: Any,
    ) -> dict[str, Any]:
        """Assemble an ``ElectionContext`` from context + caller-supplied slow inputs.

        Caller inputs win over context-derived ones. The caller is the electing layer,
        which is the only producer of cross-sectional and session-structure quantities;
        where it supplied nothing, a few values are readable off the context (they are
        literally the same measurements), and everything else stays at its fail-closed
        default so the algorithm refuses rather than assumes.
        """
        allowed = set(election_context_type.__dataclass_fields__.keys())
        payload: dict[str, Any] = {}

        derived: dict[str, Any] = {
            "reference_price": context.symbol.reference_price,
            "change_point_probability": context.macro.change_point_probability,
            "relative_volume": context.symbol.relative_volume,
            "market_breadth": context.cross_sectional.market_breadth,
            "liquidity_score": context.microstructure.liquidity_score,
            "spread_bps": context.microstructure.spread_bps,
            "sector_rank": context.cross_sectional.relative_strength_rank,
            "sector_candidate_count": context.cross_sectional.sector_candidate_count,
            "minutes_to_continuous_close": context.temporal.minutes_to_close,
            "in_last_continuous_half_hour": context.temporal.is_closing_window,
            "event_recency_seconds": context.event.event_recency_seconds,
        }
        for key, value in derived.items():
            if key in allowed and value is not None:
                payload[key] = value
        for key, value in election_inputs.items():
            name = str(key)
            if name in allowed and name != "elected_at":
                payload[name] = value
        payload["strategy_id"] = spec.strategy_id
        payload["elected_at"] = context.captured_at
        return payload

    @staticmethod
    def _geometry_prices(
        *,
        spec: StrategySpec,
        reference: float | None,
        spread_bps: float | None,
        round_trip_cost_bps: float | None,
    ) -> tuple[float | None, float | None]:
        if reference is None or reference <= 0:
            return None, None
        try:
            from app.strategy.exit_geometry import resolve_exit_geometry

            geometry = resolve_exit_geometry(
                spec.strategy_id,
                round_trip_cost_bps=round_trip_cost_bps,
                spread_bps=spread_bps,
            )
        except Exception:  # noqa: BLE001 - geometry lookup must not break a proposal.
            return None, None
        # Direction-aware: a short's target is BELOW its entry and its stop ABOVE it.
        # An unconditional ``* (1 + rate)`` would hand a short a "target" that only
        # pays when the position is losing.
        sign = -1.0 if spec.is_short else 1.0
        target = reference * (1.0 + sign * geometry.take_profit_bps / 10_000.0)
        stop = reference * (1.0 - sign * geometry.stop_loss_bps / 10_000.0)
        return (
            target if math.isfinite(target) and target > 0 else None,
            stop if math.isfinite(stop) and stop > 0 else None,
        )

    @staticmethod
    def _features_from(context: MarketContext) -> Any | None:
        snapshot = context.feature_snapshot
        if not isinstance(snapshot, Mapping) or not snapshot:
            return None
        try:
            from app.technical.signals import TechnicalFeatureSet

            names = {member.name for member in fields(TechnicalFeatureSet)}
            return TechnicalFeatureSet(
                **{key: value for key, value in snapshot.items() if key in names}
            )
        except Exception:  # noqa: BLE001 - malformed snapshot yields no proposals.
            return None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
