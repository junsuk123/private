"""Regime-driven strategy family selection, with WAIT as a first-class answer.

Selection score
---------------
For each of the eight ontology strategy families::

    score(f) = regime_term(f) + ontology_term(f) + model_term(f) + micro_term(f)

``regime_term``
    ``Σ_r P(r) * (SUITABLE_FOR(r, f) - UNSUITABLE_FOR(r, f))`` over the multi-label
    regime probabilities, using the ontology edges' **effective weight** — the learned
    weight where one exists, the expert prior otherwise. Both are recorded in the trace.
``ontology_term``
    The same arithmetic over the *temporal* edges (session phase, expiry window) and the
    active risk conditions' ``CONTRADICTS`` edges.
``model_term``
    The GNN's ``strategy_suitability`` head, and only when the runtime is HEALTHY. A
    DEGRADED model contributes nothing rather than contributing less: a model whose
    outputs are suspect is not a model to weight down, it is one to stop reading.
``micro_term``
    Confirmation from the stock's own tape. A TREND family scored highly by the regime
    but contradicted by order flow is not a trade; this term is what makes the
    disagreement visible instead of averaging it away.

Hard exclusions come first
--------------------------
An active risk condition with an ``INVALIDATES`` edge removes a family outright — no
score, no ranking, no "but it was only slightly invalidated". Those edges are structural
in ``config/market_graph_ontology.yaml`` and cannot be learned away.

WAIT
----
Returned when nothing clears ``minimum_score``, when the top two families are within
``ambiguity_margin`` of each other while pointing in opposite directions, or when regime
confidence is below ``minimum_regime_confidence``. WAIT is a successful result, not a
failure: most minutes of most sessions are not trades.

This selector recommends. It does not size, gate, price or submit — those belong to
:class:`~app.risk.final_trade_gate.FinalTradeGate` and the execution layer, which this
module has no import path to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.context.regime import RegimeEstimate
from app.models.temporal_hetero_gnn import STRATEGY_FAMILIES
from app.ontology.market_graph import MarketGraph, default_market_graph

__all__ = [
    "FAMILY_STRATEGY_IDS",
    "FamilyScore",
    "MicroConfirmation",
    "RegimeStrategySelector",
    "StrategySelection",
    "SelectorPolicy",
]

WAIT = "WAIT"

#: Concrete catalogue ids per ontology family. Derived from the existing
#: ``app.strategy.registry`` family map rather than restated, so the two taxonomies cannot
#: drift; the handful of ids below are the ones whose ontology family is narrower than
#: their registry family (relative strength within trend following, gap within
#: event-driven).
_EXPLICIT_FAMILY_IDS: dict[str, tuple[str, ...]] = {
    "RELATIVE_STRENGTH": (
        "cross_sectional_relative_strength",
        "residual_relative_strength",
        "residual_relative_weakness",
    ),
    "GAP": ("gap_context", "overnight_gap_carry"),
    #: DEFENSIVE has no entry strategy by design: it is the stance of not adding risk and
    #: managing what is already held. A "defensive strategy" that opens positions would be
    #: the opposite of the thing.
    "DEFENSIVE": (),
}

#: Registry families that roll up into each ontology family.
_REGISTRY_FAMILY_ROLLUP: dict[str, tuple[str, ...]] = {
    "TREND": ("TrendFollowing", "TrendFollowingShort"),
    "BREAKOUT": ("Breakout", "BreakdownShort"),
    "MEAN_REVERSION": ("MeanReversion",),
    "ORDER_FLOW": ("MicrostructureReversal",),
    "EVENT": ("EventDriven",),
}


def _build_family_strategy_ids() -> dict[str, tuple[str, ...]]:
    """Ontology family -> catalogue ids, read from the registry's family map."""
    try:
        from app.strategy.registry import _FAMILY  # noqa: PLC2701 - single source of truth
    except Exception:  # noqa: BLE001 - selection still works without the catalogue.
        return {family: _EXPLICIT_FAMILY_IDS.get(family, ()) for family in STRATEGY_FAMILIES}

    explicit = {
        strategy_id
        for ids in _EXPLICIT_FAMILY_IDS.values()
        for strategy_id in ids
    }
    resolved: dict[str, tuple[str, ...]] = {}
    for family in STRATEGY_FAMILIES:
        if family in _EXPLICIT_FAMILY_IDS:
            resolved[family] = _EXPLICIT_FAMILY_IDS[family]
            continue
        registry_families = set(_REGISTRY_FAMILY_ROLLUP.get(family, ()))
        resolved[family] = tuple(
            strategy_id
            for strategy_id, registry_family in _FAMILY.items()
            if str(registry_family) in registry_families and strategy_id not in explicit
        )
    return resolved


FAMILY_STRATEGY_IDS: dict[str, tuple[str, ...]] = _build_family_strategy_ids()


@dataclass(frozen=True)
class SelectorPolicy:
    #: Score a family must reach before it can be selected at all.
    minimum_score: float = 0.15
    #: When the best two families are within this and disagree in direction, WAIT.
    ambiguity_margin: float = 0.05
    #: Regime confidence below this means the state is not known well enough to choose.
    minimum_regime_confidence: float = 0.30
    #: Weight on the GNN suitability head. Kept at parity with the ontology term so the
    #: model can move a decision but not dominate the expert structure on its own.
    model_weight: float = 1.0
    ontology_weight: float = 1.0
    micro_weight: float = 0.75
    regime_weight: float = 1.0


@dataclass(frozen=True)
class MicroConfirmation:
    """The stock's own tape, as agreement per family in [-1, 1].

    Built by :meth:`RegimeStrategySelector.micro_confirmation_from_context` rather than
    supplied ad hoc, so every caller confirms a family the same way.
    """

    values: Mapping[str, float] = field(default_factory=dict)

    def get(self, family: str) -> float:
        return float(self.values.get(family, 0.0))


@dataclass(frozen=True)
class FamilyScore:
    family: str
    score: float
    regime_term: float
    ontology_term: float
    model_term: float
    micro_term: float
    excluded: bool = False
    exclusion_reasons: tuple[str, ...] = ()
    supporting_factors: tuple[str, ...] = ()
    conflicting_factors: tuple[str, ...] = ()
    #: Ontology edges that contributed, with prior and learned weight both present.
    ontology_relations: tuple[Mapping[str, Any], ...] = ()

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        return FAMILY_STRATEGY_IDS.get(self.family, ())

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "score": self.score,
            "regime_term": self.regime_term,
            "ontology_term": self.ontology_term,
            "model_term": self.model_term,
            "micro_term": self.micro_term,
            "excluded": self.excluded,
            "exclusion_reasons": list(self.exclusion_reasons),
            "supporting_factors": list(self.supporting_factors),
            "conflicting_factors": list(self.conflicting_factors),
            "strategy_ids": list(self.strategy_ids),
            "ontology_relations": [dict(item) for item in self.ontology_relations],
        }


@dataclass(frozen=True)
class StrategySelection:
    """The recommendation, with its whole ranking attached."""

    decided_at: datetime
    ticker: str
    family: str
    action: str
    score: float
    ranked: tuple[FamilyScore, ...]
    reasons: tuple[str, ...] = ()
    regime_confidence: float = 0.0
    dominant_regime: str | None = None
    model_used: bool = False

    @property
    def is_wait(self) -> bool:
        return self.action == WAIT

    @property
    def selected(self) -> FamilyScore | None:
        for item in self.ranked:
            if item.family == self.family:
                return item
        return None

    def strategy_ids(self) -> tuple[str, ...]:
        return () if self.is_wait else FAMILY_STRATEGY_IDS.get(self.family, ())

    def as_dict(self) -> dict[str, Any]:
        return {
            "decided_at": self.decided_at.isoformat(),
            "ticker": self.ticker,
            "family": self.family,
            "action": self.action,
            "score": self.score,
            "reasons": list(self.reasons),
            "regime_confidence": self.regime_confidence,
            "dominant_regime": self.dominant_regime,
            "model_used": self.model_used,
            "strategy_ids": list(self.strategy_ids()),
            "ranked": [item.as_dict() for item in self.ranked],
        }


#: Families whose thesis is long-biased versus short/defensive. Used only to detect a
#: top-two disagreement worth waiting out.
_LONG_BIASED = frozenset({"TREND", "BREAKOUT", "RELATIVE_STRENGTH", "GAP"})
_DEFENSIVE_BIASED = frozenset({"DEFENSIVE", "MEAN_REVERSION"})

_PHASE_NODE = {
    "OPENING": "OPENING_PHASE",
    "OPEN_TRANSITION": "OPENING_PHASE",
    "MORNING_TREND": "OPENING_PHASE",
    "MIDDAY": "MIDDAY_PHASE",
    "AFTERNOON": "MIDDAY_PHASE",
    "CLOSING": "CLOSING_PHASE",
}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class RegimeStrategySelector:
    """Ranks strategy families from regime, ontology, model and micro evidence."""

    def __init__(
        self,
        *,
        graph: MarketGraph | None = None,
        policy: SelectorPolicy | None = None,
    ) -> None:
        self._graph = graph or default_market_graph()
        self._policy = policy or SelectorPolicy()

    @property
    def policy(self) -> SelectorPolicy:
        return self._policy

    # ------------------------------------------------------------------ #
    def select(
        self,
        *,
        ticker: str,
        regime: RegimeEstimate,
        decided_at: datetime | None = None,
        session_phase: str | None = None,
        expiry_context: str | None = None,
        active_risk_conditions: Mapping[str, float] | None = None,
        model_suitability: Mapping[str, float] | None = None,
        model_healthy: bool = False,
        micro: MicroConfirmation | None = None,
    ) -> StrategySelection:
        moment = (
            decided_at
            if decided_at is not None and decided_at.tzinfo
            else (decided_at or datetime.now(timezone.utc)).replace(tzinfo=timezone.utc)
            if decided_at is not None
            else datetime.now(timezone.utc)
        ).astimezone(timezone.utc)
        risk_conditions = dict(active_risk_conditions or {})
        confirmation = micro or MicroConfirmation()
        use_model = bool(model_healthy and model_suitability)
        reasons: list[str] = []

        scores: list[FamilyScore] = []
        for family in STRATEGY_FAMILIES:
            scores.append(
                self._score_family(
                    family,
                    regime=regime,
                    session_phase=session_phase,
                    expiry_context=expiry_context,
                    risk_conditions=risk_conditions,
                    model_suitability=model_suitability if use_model else None,
                    micro=confirmation,
                )
            )
        ranked = tuple(sorted(scores, key=lambda item: (item.excluded, -item.score)))

        available = [item for item in ranked if not item.excluded]
        if regime.confidence < self._policy.minimum_regime_confidence:
            reasons.append("WAIT_LOW_REGIME_CONFIDENCE")
            return self._wait(moment, ticker, ranked, reasons, regime, use_model)
        # DEFENSIVE has no strategies behind it, so it surviving is not the same as an
        # entry family surviving. Reported separately: "every thesis was invalidated" and
        # "the defensive stance won on points" are different diagnoses.
        if not [item for item in available if item.family != "DEFENSIVE"]:
            reasons.append("WAIT_ALL_FAMILIES_INVALIDATED")
            reasons.extend(
                dict.fromkeys(
                    reason
                    for item in ranked
                    if item.excluded
                    for reason in item.exclusion_reasons
                )
            )
            return self._wait(moment, ticker, ranked, reasons, regime, use_model)

        best = available[0]
        if best.score < self._policy.minimum_score:
            reasons.append("WAIT_BELOW_MINIMUM_SCORE")
            return self._wait(moment, ticker, ranked, reasons, regime, use_model)

        if len(available) > 1:
            runner_up = available[1]
            if (
                best.score - runner_up.score < self._policy.ambiguity_margin
                and _opposed(best.family, runner_up.family)
            ):
                reasons.append(
                    f"WAIT_AMBIGUOUS:{best.family}~{runner_up.family}"
                )
                return self._wait(moment, ticker, ranked, reasons, regime, use_model)

        if best.family == "DEFENSIVE":
            # DEFENSIVE winning is a positive statement — take no new risk — and it is
            # expressed as WAIT so nothing downstream has to special-case a family with
            # no strategies behind it.
            reasons.append("WAIT_DEFENSIVE_REGIME")
            return self._wait(moment, ticker, ranked, reasons, regime, use_model)

        return StrategySelection(
            decided_at=moment,
            ticker=str(ticker),
            family=best.family,
            action="ENTER",
            score=best.score,
            ranked=ranked,
            reasons=tuple(reasons),
            regime_confidence=regime.confidence,
            dominant_regime=regime.dominant,
            model_used=use_model,
        )

    # ------------------------------------------------------------------ #
    def _wait(
        self,
        moment: datetime,
        ticker: str,
        ranked: Sequence[FamilyScore],
        reasons: Sequence[str],
        regime: RegimeEstimate,
        model_used: bool,
    ) -> StrategySelection:
        return StrategySelection(
            decided_at=moment,
            ticker=str(ticker),
            family=WAIT,
            action=WAIT,
            score=0.0,
            ranked=tuple(ranked),
            reasons=tuple(reasons),
            regime_confidence=regime.confidence,
            dominant_regime=regime.dominant,
            model_used=model_used,
        )

    def _score_family(
        self,
        family: str,
        *,
        regime: RegimeEstimate,
        session_phase: str | None,
        expiry_context: str | None,
        risk_conditions: Mapping[str, float],
        model_suitability: Mapping[str, float] | None,
        micro: MicroConfirmation,
    ) -> FamilyScore:
        supporting: list[str] = []
        conflicting: list[str] = []
        relations: list[Mapping[str, Any]] = []

        exclusions = self._hard_exclusions(family, risk_conditions)

        regime_term = 0.0
        for label, probability in regime.probabilities.items():
            if probability <= 0.0:
                continue
            suitable = self._edge_weight(label, family, "SUITABLE_FOR", relations)
            unsuitable = self._edge_weight(label, family, "UNSUITABLE_FOR", relations)
            contribution = probability * (suitable - unsuitable)
            regime_term += contribution
            if contribution > 0.05:
                supporting.append(f"REGIME:{label}={probability:.2f}")
            elif contribution < -0.05:
                conflicting.append(f"REGIME:{label}={probability:.2f}")

        ontology_term = 0.0
        phase_node = _PHASE_NODE.get(str(session_phase or "").upper())
        if phase_node:
            suitable = self._edge_weight(phase_node, family, "SUITABLE_FOR", relations)
            unsuitable = self._edge_weight(phase_node, family, "UNSUITABLE_FOR", relations)
            ontology_term += suitable - unsuitable
            if suitable > unsuitable:
                supporting.append(f"PHASE:{session_phase}")
            elif unsuitable > suitable:
                conflicting.append(f"PHASE:{session_phase}")
        if expiry_context and str(expiry_context).upper() != "NONE":
            suitable = self._edge_weight("EXPIRY_WINDOW", family, "SUITABLE_FOR", relations)
            unsuitable = self._edge_weight(
                "EXPIRY_WINDOW", family, "UNSUITABLE_FOR", relations
            )
            ontology_term += suitable - unsuitable
            if unsuitable > suitable:
                conflicting.append(f"EXPIRY:{expiry_context}")

        for condition, severity in risk_conditions.items():
            weight = _finite(severity) or 0.0
            if weight <= 0.0:
                continue
            contradiction = self._edge_weight(condition, family, "CONTRADICTS", relations)
            increases = self._edge_weight(condition, family, "INCREASES_RISK", relations)
            decreases = self._edge_weight(condition, family, "DECREASES_RISK", relations)
            penalty = weight * (contradiction + increases - decreases)
            ontology_term -= penalty
            if penalty > 0.05:
                conflicting.append(f"RISK:{condition}={weight:.2f}")
            elif penalty < -0.05:
                supporting.append(f"RISK:{condition}={weight:.2f}")

        model_term = 0.0
        if model_suitability is not None:
            raw = _finite(model_suitability.get(family))
            if raw is not None:
                # Centred on 0.5 so an indifferent head contributes nothing rather than a
                # constant positive bias to every family.
                model_term = (raw - 0.5) * 2.0
                if model_term > 0.2:
                    supporting.append(f"MODEL:{family}={raw:.2f}")
                elif model_term < -0.2:
                    conflicting.append(f"MODEL:{family}={raw:.2f}")

        micro_term = micro.get(family)
        if micro_term > 0.2:
            supporting.append(f"MICRO:{family}={micro_term:.2f}")
        elif micro_term < -0.2:
            conflicting.append(f"MICRO:{family}={micro_term:.2f}")

        policy = self._policy
        score = (
            policy.regime_weight * regime_term
            + policy.ontology_weight * ontology_term
            + policy.model_weight * model_term
            + policy.micro_weight * micro_term
        )
        return FamilyScore(
            family=family,
            score=round(score, 6),
            regime_term=round(regime_term, 6),
            ontology_term=round(ontology_term, 6),
            model_term=round(model_term, 6),
            micro_term=round(micro_term, 6),
            excluded=bool(exclusions),
            exclusion_reasons=exclusions,
            supporting_factors=tuple(dict.fromkeys(supporting)),
            conflicting_factors=tuple(dict.fromkeys(conflicting)),
            ontology_relations=tuple(relations),
        )

    def _hard_exclusions(
        self, family: str, risk_conditions: Mapping[str, float]
    ) -> tuple[str, ...]:
        """``INVALIDATES`` edges from active risk conditions. Not scoreable, not learnable."""
        reasons: list[str] = []
        for condition, severity in risk_conditions.items():
            if (_finite(severity) or 0.0) <= 0.0:
                continue
            if self._graph.edges(
                relation="INVALIDATES", source=str(condition), target=family
            ):
                reasons.append(f"INVALIDATED_BY:{condition}")
        return tuple(reasons)

    def _edge_weight(
        self,
        source: str,
        target: str,
        relation: str,
        relations: list[Mapping[str, Any]],
    ) -> float:
        edges = self._graph.edges(relation=relation, source=source, target=target)
        total = 0.0
        for edge in edges:
            total += edge.effective_weight
            relations.append(edge.as_dict())
        return total

    # ------------------------------------------------------------------ #
    @staticmethod
    def micro_confirmation_from_context(
        *,
        trend_strength: float | None = None,
        orderflow_imbalance: float | None = None,
        breakout_state: float | None = None,
        relative_strength: float | None = None,
        vwap_distance_bps: float | None = None,
        liquidity_score: float | None = None,
        gap_bps: float | None = None,
        event_score: float | None = None,
    ) -> MicroConfirmation:
        """Per-family agreement from one symbol's tape, each in [-1, 1].

        Every term is the family's *own* thesis measured on the tape, not a generic
        bullishness score reused eight times — which is what would make micro confirmation
        unable to disagree with the regime.
        """
        values: dict[str, float] = {}

        trend = _finite(trend_strength)
        if trend is not None:
            values["TREND"] = math.tanh(trend / 20.0)
        breakout = _finite(breakout_state)
        if breakout is not None:
            values["BREAKOUT"] = max(-1.0, min(1.0, breakout))
        vwap = _finite(vwap_distance_bps)
        if vwap is not None:
            # Mean reversion agrees when price is stretched AWAY from VWAP, either side.
            values["MEAN_REVERSION"] = min(1.0, abs(vwap) / 40.0) * (
                1.0 if abs(vwap) > 10.0 else -0.5
            )
        strength = _finite(relative_strength)
        if strength is not None:
            values["RELATIVE_STRENGTH"] = math.tanh(strength / 0.01)
        flow = _finite(orderflow_imbalance)
        if flow is not None:
            liquidity = _finite(liquidity_score)
            # Order-flow theses need a book to read; thin liquidity makes the imbalance
            # a small-sample artefact rather than pressure.
            scale = 1.0 if liquidity is None else max(0.0, min(1.0, liquidity / 0.5))
            values["ORDER_FLOW"] = max(-1.0, min(1.0, flow)) * scale
        gap = _finite(gap_bps)
        if gap is not None:
            values["GAP"] = math.tanh(gap / 100.0)
        event = _finite(event_score)
        if event is not None:
            values["EVENT"] = max(-1.0, min(1.0, event))
        liquidity = _finite(liquidity_score)
        if liquidity is not None:
            # Defensive agrees when the tape is poor: thin liquidity, no direction.
            values["DEFENSIVE"] = max(-1.0, min(1.0, 1.0 - liquidity * 2.0))
        return MicroConfirmation(values=values)


def _opposed(left: str, right: str) -> bool:
    """Do these two families imply incompatible actions on the same name?"""
    if left == right:
        return False
    return (left in _LONG_BIASED and right in _DEFENSIVE_BIASED) or (
        right in _LONG_BIASED and left in _DEFENSIVE_BIASED
    )
