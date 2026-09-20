"""Multi-label market regime: probabilities, not one winning enum.

Why multi-label
---------------
A market can be ``RISK_OFF`` and ``RANGE_HIGH_VOL`` and ``INDEX_UP_BREADTH_DOWN`` at the
same time, and those three facts imply different things about which strategies are viable.
A single-label classifier has to pick one and discard the rest, which is how a selector
loses the ability to notice that the index is rising on collapsing breadth.

So every label carries an **independent** probability in [0, 1]. They do not sum to one,
and nothing here normalises them.

Two evidence sources, kept separable
------------------------------------
* **Rule evidence** — deterministic scoring from the context hierarchy. Always available,
  auditable term by term, and the only source when the GNN is DEGRADED or OFFLINE.
* **Model evidence** — the GNN's ``market_regime`` head, used only when the runtime
  reports HEALTHY.

They are combined as a confidence-weighted blend, and :attr:`RegimeEstimate.contributions`
keeps both sides visible, so "the model says BREAKDOWN but the rules do not" is a question
that can be answered from a stored decision rather than reconstructed.

This module labels the state. It never selects a strategy and never authorises anything.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from app.context.domestic_context import DomesticContext
from app.context.global_context import GlobalContext
from app.context.session_phase import SessionPhase
from app.context.temporal_context import TemporalSnapshot
from app.models.temporal_hetero_gnn import REGIME_LABELS

__all__ = [
    "REGIME_LABELS",
    "RegimeEstimate",
    "RegimeEstimator",
    "RegimeEvidence",
    "RegimeStabilizer",
]

ROUTING_REGIMES = (
    "TREND_LOW_VOL",
    "TREND_HIGH_VOL",
    "RANGE_LOW_VOL",
    "RANGE_HIGH_VOL",
    "RISK_OFF",
)
DEFAULT_ROUTING_CONFIG = (
    Path(__file__).resolve().parents[3] / "config" / "multi_strategy_routing.yaml"
)

#: Weight given to model evidence when the GNN is healthy. Below 0.5 on purpose: the rule
#: layer is the auditable one, and a model whose evidence outvotes every deterministic
#: term is a model that has quietly become the decision-maker.
MODEL_BLEND_WEIGHT = 0.45

#: Realized volatility (per-observation fraction) separating low- from high-volatility
#: ranges. 0.0025 is the "medium/high" cut already used by config/context_features.yaml.
VOLATILITY_CUT = 0.0025

#: |direction| below which the market is not trending in either direction.
DIRECTION_FLAT = 0.20

#: |direction| above which a trend claim is strong.
DIRECTION_STRONG = 0.45

#: Breadth magnitude that makes an index/breadth divergence worth naming.
BREADTH_DIVERGENCE = 0.15

#: Liquidity below this reads as stress rather than as a quiet tape.
LIQUIDITY_STRESS_CUT = 0.35


@dataclass(frozen=True)
class RegimeEvidence:
    """Inputs the estimator scores. Every field optional; absence lowers confidence."""

    direction: float | None = None
    breadth: float | None = None
    volatility: float | None = None
    liquidity: float | None = None
    flow: float | None = None
    leadership: float | None = None
    global_direction: float | None = None
    global_risk_sentiment: float | None = None
    global_alignment: float | None = None
    global_volatility: float | None = None
    venue_divergence: float | None = None
    change_point_probability: float | None = None
    event_shock_score: float | None = None
    breakout_state: float | None = None
    session_phase: SessionPhase | None = None

    @classmethod
    def from_contexts(
        cls,
        *,
        domestic: DomesticContext | None = None,
        global_context: GlobalContext | None = None,
        temporal: TemporalSnapshot | None = None,
        change_point_probability: float | None = None,
        event_shock_score: float | None = None,
        breakout_state: float | None = None,
    ) -> "RegimeEvidence":
        return cls(
            direction=domestic.direction if domestic else None,
            breadth=domestic.breadth if domestic else None,
            volatility=domestic.volatility if domestic else None,
            liquidity=domestic.liquidity if domestic else None,
            flow=domestic.flow if domestic else None,
            leadership=domestic.leadership if domestic else None,
            global_direction=global_context.direction if global_context else None,
            global_risk_sentiment=(
                global_context.risk_sentiment if global_context else None
            ),
            global_alignment=global_context.global_alignment if global_context else None,
            global_volatility=global_context.volatility if global_context else None,
            venue_divergence=domestic.venue_divergence if domestic else None,
            change_point_probability=change_point_probability,
            event_shock_score=event_shock_score,
            breakout_state=breakout_state,
            session_phase=temporal.session_phase if temporal else None,
        )

    def answered_fields(self) -> int:
        return sum(
            1
            for name, value in self.__dict__.items()
            if name != "session_phase" and value is not None
        )


@dataclass(frozen=True)
class RegimeEstimate:
    """Independent probabilities per label, with both evidence sources attached."""

    probabilities: Mapping[str, float]
    confidence: float
    evaluated_at: datetime
    source: str
    #: ``{label: {"rule": p, "model": p or None}}``.
    contributions: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    model_version: str | None = None
    routing_probabilities: Mapping[str, float] = field(default_factory=dict)
    raw_routing_regime: str | None = None
    routing_regime: str | None = None
    routing_confidence: float = 0.0
    routing_reasons: tuple[str, ...] = ()
    feature_snapshot: Mapping[str, Any] = field(default_factory=dict)

    #: Reported when no label carries any probability. An estimate with nothing behind it
    #: must not name a regime: ``max()`` over all-zeros returns whichever label happens to
    #: be first in the catalogue, which reads on a dashboard as a confident TREND_UP.
    UNKNOWN = "UNKNOWN"

    @property
    def dominant(self) -> str:
        best = max(self.probabilities, key=lambda label: self.probabilities[label])
        return best if self.probabilities.get(best, 0.0) > 0.0 else self.UNKNOWN

    @property
    def entropy(self) -> float:
        """Bernoulli entropy averaged over labels — how undecided the estimate is.

        Not Shannon entropy over a distribution: these are independent probabilities, and
        treating them as one distribution would report a confident multi-label state as
        maximally uncertain.
        """
        total = 0.0
        for probability in self.probabilities.values():
            p = min(1.0 - 1e-9, max(1e-9, float(probability)))
            total += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        return round(total / max(1, len(self.probabilities)), 6)

    def above(self, threshold: float) -> tuple[str, ...]:
        return tuple(
            label
            for label, probability in sorted(
                self.probabilities.items(), key=lambda item: -item[1]
            )
            if probability >= threshold
        )

    def probability(self, label: str) -> float:
        return float(self.probabilities.get(label, 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated_at": self.evaluated_at.isoformat(),
            "source": self.source,
            "model_version": self.model_version,
            "dominant": self.dominant,
            "entropy": self.entropy,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "contributions": {
                label: dict(values) for label, values in self.contributions.items()
            },
            "reasons": list(self.reasons),
            "routing": {
                "raw_regime": self.raw_routing_regime,
                "regime": self.routing_regime,
                "confidence": self.routing_confidence,
                "probabilities": dict(self.routing_probabilities),
                "reasons": list(self.routing_reasons),
            },
            "feature_snapshot": dict(self.feature_snapshot),
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _logistic(value: float, *, midpoint: float = 0.0, steepness: float = 6.0) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (value - midpoint)))


class RegimeEstimator:
    """Scores the regime catalogue from context evidence, optionally fused with the GNN."""

    def estimate(
        self,
        evidence: RegimeEvidence,
        *,
        evaluated_at: datetime,
        model_probabilities: Mapping[str, float] | None = None,
        model_version: str | None = None,
        model_weight: float = MODEL_BLEND_WEIGHT,
    ) -> RegimeEstimate:
        moment = (
            evaluated_at
            if evaluated_at.tzinfo
            else evaluated_at.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        rule_scores, reasons = self._rule_scores(evidence)

        contributions: dict[str, dict[str, float | None]] = {}
        blended: dict[str, float] = {}
        weight = _clamp(float(model_weight)) if model_probabilities else 0.0
        for label in REGIME_LABELS:
            rule = _clamp(rule_scores.get(label, 0.0))
            model = (
                _clamp(float(model_probabilities.get(label, 0.0)))
                if model_probabilities is not None and label in model_probabilities
                else None
            )
            value = rule if model is None else (1.0 - weight) * rule + weight * model
            blended[label] = round(value, 6)
            contributions[label] = {"rule": round(rule, 6), "model": model}

        answered = evidence.answered_fields()
        # Confidence is coverage-driven: nine of the eleven scored inputs present is a
        # confident estimate; two is not, regardless of how decisive those two look.
        confidence = round(min(1.0, answered / 9.0), 6)
        source = "rule+model" if model_probabilities else "rule"
        volatility = evidence.volatility
        high_vol = (
            _logistic(volatility, midpoint=VOLATILITY_CUT, steepness=800.0)
            if volatility is not None else 0.5
        )
        trend_mass = max(blended.get("TREND_UP", 0.0), blended.get("TREND_DOWN", 0.0))
        routing_probabilities = {
            "TREND_LOW_VOL": round(trend_mass * (1.0 - high_vol), 6),
            "TREND_HIGH_VOL": round(trend_mass * high_vol, 6),
            "RANGE_LOW_VOL": round(blended.get("RANGE_LOW_VOL", 0.0), 6),
            "RANGE_HIGH_VOL": round(blended.get("RANGE_HIGH_VOL", 0.0), 6),
            "RISK_OFF": round(blended.get("RISK_OFF", 0.0), 6),
        }
        raw_routing = max(routing_probabilities, key=routing_probabilities.get)
        routing_confidence = float(routing_probabilities[raw_routing])
        return RegimeEstimate(
            probabilities=blended,
            confidence=confidence,
            evaluated_at=moment,
            source=source,
            contributions=contributions,
            reasons=tuple(dict.fromkeys(reasons)),
            model_version=model_version,
            routing_probabilities=routing_probabilities,
            raw_routing_regime=raw_routing,
            routing_regime=raw_routing,
            routing_confidence=round(routing_confidence, 6),
            routing_reasons=("REGIME_ROUTING_RAW",),
            feature_snapshot={
                name: value.value if isinstance(value, SessionPhase) else value
                for name, value in evidence.__dict__.items()
            },
        )


    # ------------------------------------------------------------------ #
    def _rule_scores(
        self, evidence: RegimeEvidence
    ) -> tuple[dict[str, float], list[str]]:
        scores = {label: 0.0 for label in REGIME_LABELS}
        reasons: list[str] = []

        direction = evidence.direction
        breadth = evidence.breadth
        volatility = evidence.volatility
        liquidity = evidence.liquidity
        flow = evidence.flow

        if direction is None:
            reasons.append("REGIME_NO_DIRECTION")
        else:
            up = _logistic(direction, midpoint=DIRECTION_FLAT, steepness=8.0)
            down = _logistic(-direction, midpoint=DIRECTION_FLAT, steepness=8.0)
            flat = 1.0 - max(up, down)
            scores["TREND_UP"] = up
            scores["TREND_DOWN"] = down
            # A range is a market that is neither trending; volatility only decides WHICH
            # range it is, which is why the two range labels share the flat mass.
            if volatility is None:
                scores["RANGE_LOW_VOL"] = flat * 0.5
                scores["RANGE_HIGH_VOL"] = flat * 0.5
                reasons.append("REGIME_NO_VOLATILITY")
            else:
                high = _logistic(volatility, midpoint=VOLATILITY_CUT, steepness=800.0)
                scores["RANGE_LOW_VOL"] = flat * (1.0 - high)
                scores["RANGE_HIGH_VOL"] = flat * high

            breakout = evidence.breakout_state
            if breakout is not None:
                scores["BREAKOUT_UP"] = _clamp(up * _clamp(breakout))
                scores["BREAKDOWN"] = _clamp(down * _clamp(-breakout))
            else:
                # Without a breakout reading, a strong directional move with volume is
                # still evidence of a break, but weaker evidence than a confirmed one.
                strong_up = _logistic(direction, midpoint=DIRECTION_STRONG, steepness=10.0)
                strong_down = _logistic(-direction, midpoint=DIRECTION_STRONG, steepness=10.0)
                scores["BREAKOUT_UP"] = strong_up * 0.6
                scores["BREAKDOWN"] = strong_down * 0.6

        # -- risk on / off ------------------------------------------------- #
        risk_terms = [
            value
            for value in (
                evidence.global_risk_sentiment,
                direction,
                flow,
            )
            if value is not None
        ]
        if risk_terms:
            mean = sum(risk_terms) / len(risk_terms)
            scores["RISK_ON"] = _logistic(mean, midpoint=0.15, steepness=6.0)
            scores["RISK_OFF"] = _logistic(-mean, midpoint=0.15, steepness=6.0)
        else:
            reasons.append("REGIME_NO_RISK_EVIDENCE")

        # -- liquidity stress ------------------------------------------------ #
        stress_terms: list[float] = []
        if liquidity is not None:
            stress_terms.append(_logistic(-liquidity, midpoint=-LIQUIDITY_STRESS_CUT, steepness=10.0))
        if evidence.venue_divergence is not None:
            stress_terms.append(_clamp(evidence.venue_divergence))
        if evidence.global_volatility is not None:
            stress_terms.append(_clamp((evidence.global_volatility - 1.0) / 1.0))
        if stress_terms:
            scores["LIQUIDITY_STRESS"] = _clamp(max(stress_terms))
        else:
            reasons.append("REGIME_NO_LIQUIDITY_EVIDENCE")

        # -- index / breadth divergence --------------------------------------- #
        if direction is not None and breadth is not None:
            if direction > 0 and breadth < -BREADTH_DIVERGENCE:
                scores["INDEX_UP_BREADTH_DOWN"] = _clamp(
                    min(abs(direction), abs(breadth)) * 2.0
                )
            if direction < 0 and breadth > BREADTH_DIVERGENCE:
                scores["INDEX_DOWN_BREADTH_UP"] = _clamp(
                    min(abs(direction), abs(breadth)) * 2.0
                )
        elif breadth is None:
            reasons.append("REGIME_NO_BREADTH")

        # -- event shock -------------------------------------------------------- #
        shock_terms = [
            value
            for value in (
                evidence.event_shock_score,
                (
                    _clamp((evidence.global_volatility - 1.25) / 0.75)
                    if evidence.global_volatility is not None
                    else None
                ),
            )
            if value is not None
        ]
        if shock_terms:
            scores["EVENT_SHOCK"] = _clamp(max(shock_terms))

        # -- transition ---------------------------------------------------------- #
        transition_terms: list[float] = []
        if evidence.change_point_probability is not None:
            transition_terms.append(_clamp(evidence.change_point_probability))
        if evidence.global_alignment is not None:
            # A world whose groups cancel is a world mid-rotation.
            transition_terms.append(_clamp(1.0 - abs(evidence.global_alignment)))
        if direction is not None and abs(direction) < DIRECTION_FLAT:
            transition_terms.append(0.35)
        if transition_terms:
            scores["TRANSITION"] = _clamp(sum(transition_terms) / len(transition_terms))

        return {label: round(_clamp(value), 6) for label, value in scores.items()}, reasons


class RegimeStabilizer:
    """Stateful routing-regime hysteresis with immediate strong RISK_OFF."""

    def __init__(self, path: str | Path = DEFAULT_ROUTING_CONFIG) -> None:
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            cfg = dict((payload.get("regime") or {}).get("hysteresis") or {})
        except Exception:  # noqa: BLE001
            cfg = {}
        self.minimum_consecutive = max(1, int(cfg.get("minimum_consecutive_observations", 3)))
        self.minimum_hold_seconds = max(0.0, float(cfg.get("minimum_hold_seconds", 90)))
        self.switch_margin = max(0.0, float(cfg.get("switch_margin", 0.08)))
        self.risk_off_immediate_threshold = min(
            1.0, max(0.0, float(cfg.get("risk_off_immediate_threshold", 0.80)))
        )
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def stabilize(self, estimate: RegimeEstimate, *, key: str = "GLOBAL") -> RegimeEstimate:
        scores = dict(estimate.routing_probabilities)
        if not scores:
            return estimate
        raw = estimate.raw_routing_regime or max(scores, key=scores.get)
        moment = estimate.evaluated_at
        state_key = str(key or "GLOBAL").upper()
        reasons: list[str] = []
        with self._lock:
            state = self._state.get(state_key)
            if state is None:
                state = {"current": raw, "since": moment, "pending": None, "count": 0}
                self._state[state_key] = state
                reasons.append("REGIME_HYSTERESIS_INITIALIZED")
            elif raw == "RISK_OFF" and scores.get("RISK_OFF", 0.0) >= self.risk_off_immediate_threshold:
                if state["current"] != "RISK_OFF":
                    state.update(current="RISK_OFF", since=moment, pending=None, count=0)
                    reasons.append("REGIME_RISK_OFF_IMMEDIATE")
                else:
                    reasons.append("REGIME_HYSTERESIS_HELD")
            elif raw == state["current"]:
                state.update(pending=None, count=0)
                reasons.append("REGIME_HYSTERESIS_HELD")
            else:
                held = max(0.0, (moment - state["since"]).total_seconds())
                current_score = float(scores.get(state["current"], 0.0))
                margin_ok = float(scores.get(raw, 0.0)) >= current_score + self.switch_margin
                if state.get("pending") == raw:
                    state["count"] = int(state.get("count", 0)) + 1
                else:
                    state.update(pending=raw, count=1)
                if held >= self.minimum_hold_seconds and margin_ok and int(state["count"]) >= self.minimum_consecutive:
                    state.update(current=raw, since=moment, pending=None, count=0)
                    reasons.append("REGIME_HYSTERESIS_SWITCH_CONFIRMED")
                else:
                    reasons.append("REGIME_HYSTERESIS_SWITCH_PENDING")
            stable = str(state["current"])
            if stable != raw:
                reasons.append("REGIME_RAW_DIFFERS_FROM_STABLE")
        return replace(
            estimate,
            routing_regime=stable,
            routing_confidence=round(float(scores.get(stable, 0.0)), 6),
            routing_reasons=tuple(reasons),
        )


def regime_from_prediction(
    prediction: Any, *, market_group: str = "KR"
) -> dict[str, float] | None:
    """Extract the GNN's market-regime head for the market node, or ``None``.

    Returns ``None`` rather than an empty mapping when the model did not answer, so a
    caller cannot mistake "no model" for "the model said every regime is impossible".
    """
    if prediction is None:
        return None
    payload = prediction.for_market(market_group)
    if not payload:
        return None
    regimes = payload.get("market_regime")
    if not isinstance(regimes, Mapping) or not regimes:
        return None
    return {str(label): float(value) for label, value in regimes.items()}
