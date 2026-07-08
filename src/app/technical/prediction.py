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
class PredictionConfig:
    min_confidence: float = 0.5
    min_net_return_bps: float = 0.0       # net must be > 0; the gate applies the real buffer
    model_disagreement_penalty: float = 0.5
    model_confidence_weight: float = 0.4  # blend weight of model probability
    # Warning-only threshold (not a hard block): flags setups whose adverse-
    # excursion estimate is heavily skewed vs the expected favorable move.
    max_downside_to_edge_ratio: float = 6.0

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
        # Blend model net-return estimate conservatively (take the lower).
        net_bps = gross_bps - max(0.0, cost_bps)
        if model_prediction is not None and not getattr(model_prediction, "is_fallback", False):
            model_net = float(getattr(model_prediction, "expected_net_return_bps", net_bps))
            net_bps = min(net_bps, model_net)

        # ---- Quality gates -> NO_TRADE (never an approval) ---- #
        if confidence < self.config.min_confidence:
            reason_codes.append(rc.TECHNICAL_CONFIDENCE_TOO_LOW)
            return self._no_trade(features, composite, reason_codes,
                                  f"Confidence {confidence:.2f} below minimum.",
                                  exit_price=exit_price, gross_return=gross_return,
                                  net_bps=net_bps, downside_bps=downside_bps, confidence=confidence)
        if net_bps <= self.config.min_net_return_bps:
            reason_codes.append(rc.TECHNICAL_EDGE_NON_POSITIVE)
            return self._no_trade(features, composite, reason_codes,
                                  "Expected net return not positive after cost.",
                                  exit_price=exit_price, gross_return=gross_return,
                                  net_bps=net_bps, downside_bps=downside_bps, confidence=confidence)
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
            diagnostics={"composite": composite.as_dict(), "cost_bps": cost_bps},
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
            diagnostics={"composite": composite.as_dict()},
        )
