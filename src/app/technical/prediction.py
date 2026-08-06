"""Conservative technical prediction engine.

Fuses the composite technical signal with (optionally) the trained live model
prediction, a VWAP-anchored target, realized volatility, and a cost estimate to
produce a single conservative :class:`TechnicalPrediction`: an expected exit
price, expected gross/net return, downside-risk estimate, confidence, horizon,
and an explanation.

Boundaries (enforced):
    * This engine NEVER approves or submits an order. It emits an expected exit
      price and diagnostics; ProfitabilityGate + RiskManager remain the sole
      authorities for whether a BUY may proceed.
    * When confidence or data quality is insufficient — or the composite blocks
      the buy, or the model strongly disagrees — it returns a NO_TRADE
      prediction (``tradable=False``) rather than a fabricated number.
    * Expected edge/net is conservative and derived from measured quantities;
      there is no fabricated minimum-alpha floor.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol

from app.technical import reason_codes as rc
from app.technical.signals import (
    CompositeTechnicalSignal,
    CompositeTechnicalSignalEngine,
    SignalDirection,
    TechnicalFeatureSet,
)


class PredictionAction(str, Enum):
    BUY = "BUY"
    NO_TRADE = "NO_TRADE"


class _ModelPredictionLike(Protocol):
    probability_success: float
    expected_net_return_bps: float
    uncertainty_score: float
    approved: bool
    is_fallback: bool


@dataclass(frozen=True)
class TechnicalPrediction:
    symbol: str
    action: PredictionAction
    tradable: bool
    entry_price: float | None
    expected_exit_price: float | None
    expected_gross_return: float | None       # fraction
    expected_net_return_bps: float | None
    downside_risk_bps: float | None
    confidence: float
    expected_horizon_seconds: int
    methodology: str
    regime: str
    reason_codes: tuple[str, ...]
    explanation: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        def _r(v, n=4):
            return round(v, n) if isinstance(v, (int, float)) else v

        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "tradable": self.tradable,
            "entry_price": _r(self.entry_price, 4),
            "expected_exit_price": _r(self.expected_exit_price, 4),
            "expected_gross_return": _r(self.expected_gross_return, 6),
            "expected_net_return_bps": _r(self.expected_net_return_bps, 3),
            "downside_risk_bps": _r(self.downside_risk_bps, 3),
            "confidence": _r(self.confidence),
            "expected_horizon_seconds": self.expected_horizon_seconds,
            "methodology": self.methodology,
            "regime": self.regime,
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ExpectedMoveEstimator:
    """Turns a composite BUY signal into an expected exit price + risk band."""

    downside_multiple: float = 1.5   # adverse excursion vs expected favorable
    vwap_target_blend: float = 0.5   # weight of VWAP-anchored target vs edge target

    def estimate(
        self, features: TechnicalFeatureSet, composite: CompositeTechnicalSignal
    ) -> tuple[float | None, float | None, float | None]:
        """Return (expected_exit_price, expected_gross_return, downside_risk_bps)."""
        price = features.price
        if price is None or price <= 0 or composite.expected_edge_bps <= 0:
            return None, None, None

        edge_return = composite.expected_edge_bps / 10_000.0
        edge_target = price * (1.0 + edge_return)

        # VWAP-anchored target: if we sit below VWAP, VWAP is a natural mean-
        # reversion magnet; blend it in (never below the entry for a BUY).
        target = edge_target
        if features.vwap is not None and features.vwap > price:
            vwap_target = features.vwap
            target = (
                self.vwap_target_blend * vwap_target
                + (1.0 - self.vwap_target_blend) * edge_target
            )
        expected_gross_return = target / price - 1.0
        if expected_gross_return <= 0:
            return None, None, None

        # Downside risk from volatility over the horizon (conservative multiple).
        vol_proxy = features.realized_volatility if features.realized_volatility is not None else features.atr_pct
        if vol_proxy and vol_proxy > 0:
            horizon = max(1, composite.expected_horizon_seconds)
            vol_h = vol_proxy * math.sqrt(horizon / 60.0)
            downside_risk_bps = self.downside_multiple * vol_h * 10_000.0
        else:
            downside_risk_bps = expected_gross_return * 10_000.0  # fall back to symmetric
        return target, expected_gross_return, downside_risk_bps


@dataclass(frozen=True)
class ModelFusionPolicy:
    """How much expected-return authority a trained model actually gets.

    The engine used to take ``min(rule_net, model_net)`` unconditionally. That is
    not conservatism, it is a veto handed to whichever estimator is more
    pessimistic — including a model that is stale, schema-mismatched, fitted on a
    different feature set, or simply not economically validated. A rule edge of
    +30bps and an unreliable model saying -40bps produced -40bps, and the router
    then dropped the candidate as NON_POSITIVE_NET_EDGE. The model silently held
    veto power it had never earned.

    Now authority is explicit and bounded: ``weight`` is 0 unless the model is
    approved AND non-fallback, and a disagreeing model costs a bounded
    ``uncertainty_penalty_bps`` rather than replacing the rule estimate.
    """

    #: Blend weight ceiling for a fully reliable model. Deliberately below 1.0:
    #: no single estimator gets the whole decision.
    max_model_weight: float = 0.5
    #: Penalty applied when a reliable model disagrees in SIGN with the rule.
    #: Bounded so a model can discourage but not invert a measured rule edge.
    disagreement_penalty_bps: float = 8.0
    #: Extra penalty scaled by the model's own uncertainty score.
    uncertainty_penalty_bps: float = 6.0

    def weight_for(self, model_prediction: _ModelPredictionLike | None) -> float:
        """0 when the model has not earned expected-return authority."""
        if model_prediction is None:
            return 0.0
        if getattr(model_prediction, "is_fallback", False):
            return 0.0
        if not getattr(model_prediction, "approved", False):
            return 0.0
        uncertainty = max(0.0, min(1.0, _f(getattr(model_prediction, "uncertainty_score", 0.0))))
        return max(0.0, min(1.0, self.max_model_weight * (1.0 - uncertainty)))

    def fuse(
        self,
        *,
        rule_net_bps: float,
        model_prediction: _ModelPredictionLike | None,
    ) -> tuple[float, float, float | None, float]:
        """Return (fused_net_bps, weight, model_net_bps, penalty_bps)."""
        weight = self.weight_for(model_prediction)
        model_net: float | None = None
        if model_prediction is not None and not getattr(model_prediction, "is_fallback", False):
            model_net = _f(getattr(model_prediction, "expected_net_return_bps", None), None)
        if weight <= 0.0 or model_net is None:
            # Model is shadow evidence only. The rule estimate stands unchanged.
            return rule_net_bps, 0.0, model_net, 0.0
        penalty = 0.0
        if (model_net < 0.0) and (rule_net_bps > 0.0):
            uncertainty = max(
                0.0, min(1.0, _f(getattr(model_prediction, "uncertainty_score", 0.0)))
            )
            penalty = self.disagreement_penalty_bps + self.uncertainty_penalty_bps * uncertainty
        fused = (1.0 - weight) * rule_net_bps + weight * model_net - penalty
        return fused, weight, model_net, penalty


def _f(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass(frozen=True)
class CostViabilityPolicy:
    """The bar a candidate must clear, derived from cost — never from a target.

    ``required_gross_bps`` is the max of three independently-motivated floors so
    that lowering any single knob cannot quietly make an uneconomic trade look
    viable. When the strategy's own best plausible gross edge cannot reach it, the
    honest classification is HORIZON_COST_UNVIABLE, not a negative net number that
    downstream code reports as "no edge".
    """

    min_cost_coverage_ratio: float = 1.3
    minimum_net_buffer_bps: float = 2.0
    #: Per-strategy empirical floors, bps. Only measured values belong here.
    empirical_strategy_floor_bps: Mapping[str, float] = field(default_factory=dict)

    def required_gross_bps(self, all_in_cost_bps: float, strategy_id: str = "") -> float:
        cost = max(0.0, _f(all_in_cost_bps))
        return max(
            cost * max(1.0, self.min_cost_coverage_ratio),
            cost + max(0.0, self.minimum_net_buffer_bps),
            max(0.0, _f(self.empirical_strategy_floor_bps.get(strategy_id, 0.0))),
        )

    def required_net_bps(self, all_in_cost_bps: float, strategy_id: str = "") -> float:
        return self.required_gross_bps(all_in_cost_bps, strategy_id) - max(
            0.0, _f(all_in_cost_bps)
        )

    def cost_coverage_ratio(
        self, gross_bps: float, all_in_cost_bps: float
    ) -> float | None:
        cost = max(0.0, _f(all_in_cost_bps))
        if cost <= 0.0:
            return None
        return _f(gross_bps) / cost


@dataclass(frozen=True)
class PredictionConfig:
    min_confidence: float = 0.5
    min_net_return_bps: float = 0.0       # net must be > 0; the gate applies the real buffer
    # Additive per-horizon net-edge buffer (bps): a shorter horizon can demand a
    # larger edge to overcome noise. Keyed by horizon seconds; default empty ->
    # no extra buffer. The authoritative net check is still the ProfitabilityGate.
    horizon_edge_buffer_bps: Mapping[int, float] = field(default_factory=dict)
    model_disagreement_penalty: float = 0.5
    model_confidence_weight: float = 0.4  # blend weight of model probability
    # Warning-only threshold (not a hard block): flags setups whose adverse-
    # excursion estimate is heavily skewed vs the expected favorable move.
    max_downside_to_edge_ratio: float = 6.0
    fusion: ModelFusionPolicy = field(default_factory=ModelFusionPolicy)
    cost_viability: CostViabilityPolicy = field(default_factory=CostViabilityPolicy)

    @classmethod
    def from_env(cls) -> "PredictionConfig":
        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            try:
                return float(raw) if raw not in (None, "") else default
            except ValueError:
                return default

        return cls(
            min_confidence=_f("TECHNICAL_PREDICTION_MIN_CONFIDENCE", cls.min_confidence),
            min_net_return_bps=_f("TECHNICAL_PREDICTION_MIN_NET_BPS", cls.min_net_return_bps),
        )


class TechnicalPredictionEngine:
    def __init__(
        self,
        *,
        signal_engine: CompositeTechnicalSignalEngine | None = None,
        estimator: ExpectedMoveEstimator | None = None,
        config: PredictionConfig | None = None,
    ) -> None:
        self.signal_engine = signal_engine or CompositeTechnicalSignalEngine()
        self.estimator = estimator or ExpectedMoveEstimator()
        self.config = config or PredictionConfig()

    def predict(
        self,
        features: TechnicalFeatureSet,
        *,
        composite: CompositeTechnicalSignal | None = None,
        model_prediction: _ModelPredictionLike | None = None,
        all_in_cost_bps: float | None = None,
    ) -> TechnicalPrediction:
        composite = composite or self.signal_engine.evaluate(features)
        regime = composite.regime.value

        # ---- Hard no-trade paths ---- #
        if composite.blocks_buy:
            return self._no_trade(features, composite, list(composite.reason_codes),
                                  "Regime blocks buy (risk gate).")
        if composite.direction != SignalDirection.BUY or composite.expected_edge_bps <= 0:
            codes = list(composite.reason_codes) or [rc.TECHNICAL_EDGE_NON_POSITIVE]
            return self._no_trade(features, composite, codes,
                                  "No positive technical BUY edge.")

        reason_codes: list[str] = list(composite.reason_codes)
        confidence = composite.confidence

        # ---- Blend in the trained model prediction (advisory) ---- #
        if model_prediction is not None and not getattr(model_prediction, "is_fallback", False):
            if not model_prediction.approved:
                # The model actively disagrees -> strong contradiction.
                confidence *= self.config.model_disagreement_penalty
                reason_codes.append(rc.TECHNICAL_CONFIDENCE_TOO_LOW)
            else:
                w = self.config.model_confidence_weight
                confidence = (1.0 - w) * confidence + w * float(model_prediction.probability_success)
                # Uncertainty discounts confidence.
                unc = max(0.0, min(1.0, float(getattr(model_prediction, "uncertainty_score", 0.0))))
                confidence *= (1.0 - 0.5 * unc)

        confidence = max(0.0, min(1.0, confidence))

        # ---- Expected move / exit price ---- #
        exit_price, gross_return, downside_bps = self.estimator.estimate(features, composite)
        if exit_price is None or gross_return is None:
            return self._no_trade(features, composite, reason_codes + [rc.TECHNICAL_EDGE_NON_POSITIVE],
                                  "Expected move non-positive.")

        gross_bps = gross_return * 10_000.0
        cost_bps = all_in_cost_bps if all_in_cost_bps is not None else (features.spread_bps or 0.0)
        rule_net_bps = gross_bps - max(0.0, cost_bps)

        # Cost viability BEFORE fusion: if this strategy/horizon cannot clear the
        # cost-derived bar even at its own best plausible gross edge, the answer is
        # "uneconomic", which is a different fact from "no signal". Reporting it as
        # a negative net let a real setup and an impossible one share one code.
        viability = self.config.cost_viability
        required_net = viability.required_net_bps(cost_bps, composite.selected_methodology)
        coverage = viability.cost_coverage_ratio(gross_bps, cost_bps)
        if gross_bps < viability.required_gross_bps(cost_bps, composite.selected_methodology):
            return self._no_trade(
                features, composite,
                reason_codes + [rc.HORIZON_COST_UNVIABLE],
                (
                    f"Best plausible gross {gross_bps:.1f}bps cannot clear the "
                    f"cost-derived floor at ~{cost_bps:.1f}bps cost."
                ),
                exit_price=exit_price, gross_return=gross_return,
                net_bps=rule_net_bps, downside_bps=downside_bps, confidence=confidence,
                extra_diagnostics={
                    "rule_gross_bps": gross_bps,
                    "all_in_cost_bps": cost_bps,
                    "rule_net_bps": rule_net_bps,
                    "required_net_bps": required_net,
                    "cost_coverage_ratio": coverage,
                },
            )

        # Fuse rule and model with EXPLICIT, bounded model authority. A model that
        # is unapproved, fallback, or uncertain gets weight 0 and stays shadow
        # evidence; it can no longer replace a measured rule edge with its own.
        net_bps, model_weight, model_net_bps, uncertainty_penalty_bps = (
            self.config.fusion.fuse(
                rule_net_bps=rule_net_bps,
                model_prediction=model_prediction,
            )
        )
        fusion_diagnostics = {
            "rule_gross_bps": gross_bps,
            "all_in_cost_bps": cost_bps,
            "rule_net_bps": rule_net_bps,
            "model_net_bps": model_net_bps,
            "model_reliability_weight": model_weight,
            "uncertainty_penalty_bps": uncertainty_penalty_bps,
            "fused_net_bps": net_bps,
            "required_net_bps": required_net,
            "cost_coverage_ratio": coverage,
        }

        # ---- Quality gates -> NO_TRADE (never an approval) ---- #
        if confidence < self.config.min_confidence:
            reason_codes.append(rc.TECHNICAL_CONFIDENCE_TOO_LOW)
            return self._no_trade(features, composite, reason_codes,
                                  f"Confidence {confidence:.2f} below minimum.",
                                  exit_price=exit_price, gross_return=gross_return,
                                  net_bps=net_bps, downside_bps=downside_bps, confidence=confidence,
                                  extra_diagnostics=fusion_diagnostics)
        # The cost-derived floor is authoritative; the legacy per-horizon buffer is
        # kept as an additional (never a replacement) requirement.
        required_net_bps = max(
            required_net,
            self.config.min_net_return_bps
            + float(
                self.config.horizon_edge_buffer_bps.get(composite.expected_horizon_seconds, 0.0)
            ),
        )
        fusion_diagnostics["required_net_bps"] = required_net_bps
        if net_bps <= required_net_bps:
            reason_codes.append(rc.TECHNICAL_EDGE_NON_POSITIVE)
            return self._no_trade(features, composite, reason_codes,
                                  "Expected net return not positive after cost.",
                                  exit_price=exit_price, gross_return=gross_return,
                                  net_bps=net_bps, downside_bps=downside_bps, confidence=confidence,
                                  extra_diagnostics=fusion_diagnostics)
        # Downside risk is surfaced as INFORMATION + a warning code, not a hard
        # block: adverse excursion routinely exceeds the expected favorable move
        # over short horizons, and the authoritative risk/reward stop is set by
        # DynamicExitPolicy, not here. We only flag egregiously skewed setups.
        if downside_bps and gross_bps > 0 and downside_bps > self.config.max_downside_to_edge_ratio * gross_bps:
            reason_codes.append(rc.TECHNICAL_DOWNSIDE_RISK_HIGH)

        explanation = (
            f"{composite.selected_methodology} in {regime}: expected exit "
            f"{exit_price:.4f} (+{gross_bps:.1f}bps gross, {net_bps:.1f}bps net after "
            f"~{cost_bps:.1f}bps cost), downside ~{downside_bps:.1f}bps, "
            f"confidence {confidence:.2f}."
        )
        return TechnicalPrediction(
            symbol=features.symbol,
            action=PredictionAction.BUY,
            tradable=True,
            entry_price=features.price,
            expected_exit_price=exit_price,
            expected_gross_return=gross_return,
            expected_net_return_bps=net_bps,
            downside_risk_bps=downside_bps,
            confidence=confidence,
            expected_horizon_seconds=composite.expected_horizon_seconds,
            methodology=composite.selected_methodology,
            regime=regime,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            explanation=explanation,
            diagnostics={
                "composite": composite.as_dict(),
                "cost_bps": cost_bps,
                **fusion_diagnostics,
            },
        )

    # ------------------------------------------------------------------ #
    def _no_trade(
        self,
        features: TechnicalFeatureSet,
        composite: CompositeTechnicalSignal,
        reason_codes: list[str],
        explanation: str,
        *,
        exit_price: float | None = None,
        gross_return: float | None = None,
        net_bps: float | None = None,
        downside_bps: float | None = None,
        confidence: float = 0.0,
        extra_diagnostics: Mapping[str, object] | None = None,
    ) -> TechnicalPrediction:
        return TechnicalPrediction(
            symbol=features.symbol,
            action=PredictionAction.NO_TRADE,
            tradable=False,
            entry_price=features.price,
            expected_exit_price=exit_price,
            expected_gross_return=gross_return,
            expected_net_return_bps=net_bps,
            downside_risk_bps=downside_bps,
            confidence=confidence,
            expected_horizon_seconds=composite.expected_horizon_seconds,
            methodology=composite.selected_methodology,
            regime=composite.regime.value,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            explanation=explanation,
            diagnostics={
                "composite": composite.as_dict(),
                **dict(extra_diagnostics or {}),
            },
        )
