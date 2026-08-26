"""Strategy-conditioned utility prediction, with cost taken OUT of the model.

The change this module makes
---------------------------
The R-GCN's decoder (``app.models.strategy_utility.rgcn.output_from_raw``) predicts a
cost channel — ``cost = softplus(raw[..., 2]) * 10`` — and folds it into the utility it
also computes. Two consequences followed:

1. A fee, tax or FX policy change required *retraining*, because the cost the selector
   used lived in a checkpoint rather than in ``config/trading_costs.json``.
2. The model, not the selector, decided how downside and uncertainty traded off against
   return, so the weights were unauditable and unchangeable without a new checkpoint.

Here the model supplies only what a model can know — probability of profit, expected
gross move, downside, duration, uncertainty — and ``TradingCostEngine`` supplies the
cost. ``expected_net_return_bps`` is then an identity:

    expected_net_return_bps = expected_gross_return_bps - expected_cost_bps

Predictors
----------
``GnnUtilityAdapter``       reads the existing per-strategy GNN vector (live evidence rows
                            or ``StrategyUtilityEvidence`` objects) and *discards* the
                            model's cost channel.
``HeuristicUtilityPredictor`` the Phase-4 placeholder: the strategy's own measured gross
                            edge and its exit geometry, with uncertainty from the tape.
                            No trained weights, so it cannot pretend to be calibrated.
``CompositeUtilityPredictor`` GNN where a trusted vector exists, heuristic elsewhere, with
                            a reason code on every prediction saying which produced it.

Nothing here submits an order or can reach one: the module imports no broker, no
coordinator and no gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from app.context.market_context import MarketContext
from app.strategy.proposal import StrategyProposal

__all__ = [
    "CostEstimate",
    "CompositeUtilityPredictor",
    "GnnUtilityAdapter",
    "HeuristicUtilityPredictor",
    "StrategyUtilityPrediction",
    "TradingCostAdapter",
    "UTILITY_SOURCE_GNN",
    "UTILITY_SOURCE_HEURISTIC",
]

UTILITY_SOURCE_GNN = "UTILITY_SOURCE_GNN"
UTILITY_SOURCE_HEURISTIC = "UTILITY_SOURCE_HEURISTIC"

#: Reason code on a prediction whose GNN vector existed but was not trusted for this
#: (strategy, market). Distinct from "no vector at all": one is an untrusted estimator,
#: the other a missing one, and they justify different uncertainty.
UTILITY_GNN_UNTRUSTED = "UTILITY_GNN_UNTRUSTED"
UTILITY_GNN_ABSENT = "UTILITY_GNN_ABSENT"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------- #
# Cost                                                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostEstimate:
    """Deterministic round-trip cost for one candidate, in bps."""

    strategy_id: str
    symbol: str
    expected_cost_bps: float
    #: ``True`` when the estimate came from the real cost engine. ``False`` means the
    #: fallback was used, which must be visible: a cost the system guessed and a cost it
    #: computed are different evidence.
    measured: bool
    source: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "expected_cost_bps": round(self.expected_cost_bps, 3),
            "measured": self.measured,
            "source": self.source,
            "diagnostics": dict(self.diagnostics),
        }


class TradingCostAdapter:
    """One-call round-trip cost in bps, from the existing ``TradingCostEngine``.

    The engine's own API is priced-order shaped (entry price, exit price, quantity); this
    wraps it in the shape a selector needs, and caches per (venue, instrument, spread
    bucket) because the fee policy is a config lookup and the realtime loop asks the same
    question for every candidate on every cycle.

    ``fallback_bps`` is used only when the engine is unreadable. It is not a default: the
    returned estimate is flagged ``measured=False`` so the selector can charge extra
    uncertainty for it rather than treat a guess as a measurement.
    """

    def __init__(
        self,
        *,
        cost_engine: Any | None = None,
        fallback_bps: float = 28.0,
    ) -> None:
        self._engine = cost_engine
        self._fallback_bps = max(0.0, float(fallback_bps))
        self._cache: dict[tuple[str, str, int], float] = {}

    def _resolve_engine(self) -> Any | None:
        if self._engine is None:
            try:
                from app.cost.trading_cost_engine import TradingCostEngine

                self._engine = TradingCostEngine()
            except Exception:  # noqa: BLE001 - an unreadable cost config is not fatal.
                return None
        return self._engine

    def estimate(
        self,
        *,
        strategy_id: str,
        symbol: str,
        market: str,
        reference_price: float | None,
        spread_bps: float | None = None,
        is_short: bool = False,
    ) -> CostEstimate:
        is_krx = str(market or "").strip().upper() in {"KR", "KRX"}
        venue = "KRX" if is_krx else "NASD"
        instrument = "domestic_stock" if is_krx else "overseas_stock"
        # Spread bucketed to the nearest bp: the policy reads a spread RATE, so
        # sub-basis-point differences cannot change the answer but would defeat the cache.
        bucket = int(round(spread_bps)) if spread_bps is not None else -1
        key = (venue, instrument, bucket)

        cached = self._cache.get(key)
        if cached is not None:
            return CostEstimate(
                strategy_id=str(strategy_id),
                symbol=str(symbol),
                expected_cost_bps=cached + (_borrow_reference_bps() if is_short else 0.0),
                measured=True,
                source="trading_cost_engine(cached)",
                diagnostics={"venue": venue, "instrument_type": instrument},
            )

        engine = self._resolve_engine()
        price = _finite(reference_price)
        if engine is None or price is None or price <= 0:
            return CostEstimate(
                strategy_id=str(strategy_id),
                symbol=str(symbol),
                expected_cost_bps=self._fallback_bps
                + (_borrow_reference_bps() if is_short else 0.0),
                measured=False,
                source="fallback",
                diagnostics={
                    "reason": "COST_ENGINE_UNAVAILABLE"
                    if engine is None
                    else "NO_REFERENCE_PRICE",
                },
            )
        try:
            orderbook = _orderbook_from_spread(price, spread_bps)
            if is_short:
                rate = engine.short_round_trip_cost_rate(
                    venue=venue,
                    instrument_type=instrument,
                    orderbook_snapshot=orderbook,
                )
                cost_bps = float(rate) * 10_000.0 + _borrow_reference_bps()
            else:
                breakdown = engine.estimate(
                    symbol=str(symbol),
                    market="KR" if is_krx else "US",
                    venue=venue,
                    instrument_type=instrument,
                    entry_price=price,
                    # Flat exit: the question is "what does a round trip cost", not
                    # "is this trade profitable". A non-flat exit price would mix the
                    # cost estimate with an edge assumption.
                    expected_exit_price=price,
                    quantity=1,
                    orderbook_snapshot=orderbook,
                )
                cost_bps = float(breakdown.total_cost_rate) * 10_000.0
            # Floor model utility at the same p75 resolved-tape authority used by
            # mechanical entry, labels and the execution gate.  The fee engine alone
            # was materially optimistic on both venues. Borrow remains an independent
            # short-side add-on.
            from app.cost.round_trip import all_in_round_trip_bps

            borrow_bps = _borrow_reference_bps() if is_short else 0.0
            venue_cost = all_in_round_trip_bps(
                str(symbol),
                spread_bps=spread_bps,
                fallback_bps=max(0.0, cost_bps - borrow_bps),
            )
            cost_bps = max(cost_bps, venue_cost + borrow_bps)
        except Exception:  # noqa: BLE001
            return CostEstimate(
                strategy_id=str(strategy_id),
                symbol=str(symbol),
                expected_cost_bps=self._fallback_bps
                + (_borrow_reference_bps() if is_short else 0.0),
                measured=False,
                source="fallback",
                diagnostics={"reason": "COST_ENGINE_RAISED"},
            )
        if not is_short:
            self._cache[key] = cost_bps
        return CostEstimate(
            strategy_id=str(strategy_id),
            symbol=str(symbol),
            expected_cost_bps=cost_bps,
            measured=True,
            source="trading_cost_engine",
            diagnostics={"venue": venue, "instrument_type": instrument},
        )


def _orderbook_from_spread(
    price: float | None, spread_bps: float | None
) -> dict[str, float] | None:
    """Reconstruct the bid/ask pair ``TradingCostEngine`` expects, or pass nothing.

    ``policy_for`` reads ``bid_price``/``ask_price`` (or ``best_bid``/``best_ask``) and
    treats a snapshot WITHOUT them as an empty book — deliberately returning a 100%
    spread, which is the worst case for a buy. Handing it ``{"spread_bps": ...}`` therefore
    produced a ~15,000bps round trip and made every candidate lose to NO_TRADE. Passing
    ``None`` instead of a malformed snapshot keeps the engine's configured default, and
    passing a reconstructed book lets its dynamic-slippage rule work as designed.
    """
    if price is None or price <= 0 or spread_bps is None or spread_bps <= 0:
        return None
    half = price * (float(spread_bps) / 2.0) / 10_000.0
    bid = price - half
    ask = price + half
    if bid <= 0 or ask <= bid:
        return None
    return {"bid_price": bid, "ask_price": ask}


def _borrow_reference_bps() -> float:
    """Borrow leg for a short, from the existing geometry reference.

    Reused rather than restated, so a change to the borrow reference moves both the
    geometry and the cost estimate together.
    """
    try:
        from app.strategy.exit_geometry import SHORT_BORROW_REFERENCE_BPS

        return float(SHORT_BORROW_REFERENCE_BPS)
    except Exception:  # noqa: BLE001
        return 8.0


# --------------------------------------------------------------------------- #
# Prediction contract                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyUtilityPrediction:
    """What the utility model says about ONE strategy under ONE context."""

    strategy_id: str
    context_id: str
    symbol: str
    probability_profit: float
    expected_gross_return_bps: float
    expected_cost_bps: float
    expected_downside_bps: float
    expected_holding_seconds: float
    uncertainty_bps: float
    model_version: str
    source: str
    reason_codes: tuple[str, ...] = ()

    @property
    def expected_net_return_bps(self) -> float:
        """Identity, never a separate prediction. See the module docstring."""
        return self.expected_gross_return_bps - self.expected_cost_bps

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "context_id": self.context_id,
            "symbol": self.symbol,
            "probability_profit": round(self.probability_profit, 4),
            "expected_gross_return_bps": round(self.expected_gross_return_bps, 3),
            "expected_cost_bps": round(self.expected_cost_bps, 3),
            "expected_net_return_bps": round(self.expected_net_return_bps, 3),
            "expected_downside_bps": round(self.expected_downside_bps, 3),
            "expected_holding_seconds": round(self.expected_holding_seconds, 1),
            "uncertainty_bps": round(self.uncertainty_bps, 3),
            "model_version": self.model_version,
            "source": self.source,
            "reason_codes": list(self.reason_codes),
        }


class StrategyUtilityPredictor(Protocol):
    def predict(
        self,
        context: MarketContext,
        proposals: Sequence[StrategyProposal],
        costs: Mapping[str, CostEstimate],
    ) -> tuple[StrategyUtilityPrediction, ...]:
        ...


# --------------------------------------------------------------------------- #
# Heuristic predictor (Phase 4 placeholder)                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeuristicUtilityConfig:
    """Knobs for the rule-based placeholder. Every one is a stated assumption.

    ``uncertainty_floor_bps`` is not tuning. With no trained model, the honest
    uncertainty on a single-observation forecast is large, and a floor is what stops the
    placeholder from producing a confident-looking number it has no basis for.
    """

    uncertainty_floor_bps: float = 20.0
    #: Extra uncertainty charged when the cost estimate was a fallback rather than a
    #: measurement.
    unmeasured_cost_uncertainty_bps: float = 15.0
    #: Multiplier on the strategy's own stop distance to get expected downside. 1.0 =
    #: "if it fails it goes to the stop", which is what the executor would realise.
    downside_stop_multiple: float = 1.0
    #: Probability assigned when nothing supplies one. 0.5 rather than a flattering
    #: number: a coin flip is what "we do not know" means.
    default_probability: float = 0.5


class HeuristicUtilityPredictor:
    """Utility from the strategy's own measured edge and geometry — no trained weights.

    This is the Phase-4 placeholder the migration plan calls for. It exists so
    StrategySelectorV2 can run, be compared against the legacy selector and accumulate
    counterfactual outcomes BEFORE a utility GNN is trained, and it is deliberately
    conservative rather than plausible-looking:

    * gross edge is the algorithm's own ``expected_edge_bps`` — a volatility-derived
      capture estimate, already floored by the cost-aware entry floor;
    * downside is the strategy's stop distance, i.e. what the executor would actually
      realise on a failure, not a model's guess;
    * uncertainty starts at a floor and grows with the tape's own volatility and with any
      unmeasured cost, so a thin or unpriced candidate ranks below a measured one.
    """

    model_version = "heuristic-v1"

    def __init__(self, *, config: HeuristicUtilityConfig | None = None) -> None:
        self._config = config or HeuristicUtilityConfig()

    def predict(
        self,
        context: MarketContext,
        proposals: Sequence[StrategyProposal],
        costs: Mapping[str, CostEstimate],
    ) -> tuple[StrategyUtilityPrediction, ...]:
        return tuple(
            self._predict_one(context, proposal, costs.get(proposal.strategy_id))
            for proposal in proposals
        )

    def _predict_one(
        self,
        context: MarketContext,
        proposal: StrategyProposal,
        cost: CostEstimate | None,
    ) -> StrategyUtilityPrediction:
        config = self._config
        cost_bps = cost.expected_cost_bps if cost is not None else 0.0
        gross = _finite(proposal.expected_gross_edge_bps)
        if gross is None:
            # No measured edge: fall back to the geometry's own target, which is what the
            # executor is aiming at. ``None`` for both leaves gross at 0.0, and a
            # zero-gross candidate loses to NO_TRADE by construction.
            gross = _finite(proposal.target_move_bps) or 0.0

        downside = _finite(proposal.stop_move_bps)
        if downside is None:
            # Symmetric with the target when no stop was priced. Not optimistic: the
            # target is the larger number in every geometry row, so this over-states
            # downside rather than under-stating it.
            downside = abs(gross)
        downside = max(0.0, downside * config.downside_stop_multiple)

        volatility_bps = 0.0
        realized = _finite(context.symbol.realized_volatility)
        if realized is not None and realized > 0:
            # Scaled to the proposal's own horizon: sqrt(t) on a per-observation stdev.
            horizon = max(1, int(proposal.expected_horizon_seconds or 1))
            volatility_bps = realized * 10_000.0 * math.sqrt(horizon / 60.0)

        uncertainty = max(config.uncertainty_floor_bps, 0.5 * volatility_bps)
        reasons: list[str] = [UTILITY_SOURCE_HEURISTIC]
        if cost is None or not cost.measured:
            uncertainty += config.unmeasured_cost_uncertainty_bps
            reasons.append("UTILITY_COST_NOT_MEASURED")

        probability = proposal.confidence if proposal.confidence > 0 else config.default_probability
        return StrategyUtilityPrediction(
            strategy_id=proposal.strategy_id,
            context_id=context.context_id,
            symbol=context.symbol_id,
            probability_profit=probability,
            expected_gross_return_bps=gross,
            expected_cost_bps=cost_bps,
            expected_downside_bps=downside,
            expected_holding_seconds=float(proposal.expected_horizon_seconds or 0),
            uncertainty_bps=uncertainty,
            model_version=self.model_version,
            source=UTILITY_SOURCE_HEURISTIC,
            reason_codes=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# GNN adapter                                                                  #
# --------------------------------------------------------------------------- #
class GnnUtilityAdapter:
    """Turns the existing per-strategy GNN vector into cost-free predictions.

    Accepts either ``StrategyUtilityEvidence`` objects or the persisted evidence-row
    dicts the live path already writes (``path="cpu_gnn_validation"``). Both carry the
    same names, so one reader serves the live loop and a replay.

    The model's ``expected_cost_bps`` is read only to RECONSTRUCT gross where a row
    carries net but not gross. It is never used as the cost of the trade — that comes
    from ``TradingCostEngine``, which is the whole point of this layer.
    """

    def __init__(self, *, model_version: str = "gnn-vector") -> None:
        self._model_version = model_version

    def predict(
        self,
        context: MarketContext,
        proposals: Sequence[StrategyProposal],
        costs: Mapping[str, CostEstimate],
        *,
        rows: Iterable[Any] = (),
    ) -> tuple[StrategyUtilityPrediction, ...]:
        by_id = _rows_by_strategy(rows)
        predictions: list[StrategyUtilityPrediction] = []
        for proposal in proposals:
            row = by_id.get(proposal.strategy_id)
            if row is None:
                continue
            cost = costs.get(proposal.strategy_id)
            prediction = self._from_row(context, proposal, row, cost)
            if prediction is not None:
                predictions.append(prediction)
        return tuple(predictions)

    def _from_row(
        self,
        context: MarketContext,
        proposal: StrategyProposal,
        row: Mapping[str, Any],
        cost: CostEstimate | None,
    ) -> StrategyUtilityPrediction | None:
        gross = _finite(row.get("expected_gross_return_bps"))
        model_cost = _finite(row.get("expected_cost_bps"))
        if gross is None:
            net = _finite(row.get("expected_net_return_bps"))
            if net is None:
                return None
            # net == gross - model_cost is the contract in ``StrategyUtilityEvidence``,
            # so gross is recoverable. Without a model cost the row is unusable rather
            # than assumed cost-free.
            if model_cost is None:
                return None
            gross = net + model_cost

        downside = _finite(row.get("expected_adverse_excursion_bps"))
        if downside is None:
            downside = _finite(proposal.stop_move_bps) or 0.0
        aleatoric = _finite(row.get("aleatoric_uncertainty")) or 0.0
        epistemic = _finite(row.get("epistemic_uncertainty_or_proxy")) or 0.0
        total = _finite(row.get("total_uncertainty"))
        uncertainty = total if total is not None else aleatoric + epistemic

        reason_codes = tuple(str(code) for code in (row.get("reason_codes") or ()))
        trusted = "GNN_REALTIME_TRUST_PASSED" in reason_codes
        reasons = [UTILITY_SOURCE_GNN]
        if not trusted:
            reasons.append(UTILITY_GNN_UNTRUSTED)

        return StrategyUtilityPrediction(
            strategy_id=proposal.strategy_id,
            context_id=context.context_id,
            symbol=context.symbol_id,
            probability_profit=_finite(row.get("probability_success")) or 0.0,
            expected_gross_return_bps=gross,
            expected_cost_bps=cost.expected_cost_bps if cost is not None else 0.0,
            expected_downside_bps=max(0.0, downside),
            expected_holding_seconds=_finite(row.get("expected_holding_seconds"))
            or float(proposal.expected_horizon_seconds or 0),
            uncertainty_bps=max(0.0, uncertainty),
            model_version=str(row.get("model_version") or self._model_version),
            source=UTILITY_SOURCE_GNN,
            reason_codes=tuple(reasons),
        )


class CompositeUtilityPredictor:
    """GNN prediction where one exists and is usable; heuristic everywhere else.

    Not a blend. Averaging a trained forecast with a rule-of-thumb produces a number
    neither model would stand behind, and hides which one is driving the decision. Every
    prediction states its source.
    """

    def __init__(
        self,
        *,
        gnn: GnnUtilityAdapter | None = None,
        heuristic: HeuristicUtilityPredictor | None = None,
        require_trusted_gnn: bool = False,
    ) -> None:
        self._gnn = gnn or GnnUtilityAdapter()
        self._heuristic = heuristic or HeuristicUtilityPredictor()
        # When True an untrusted GNN vector is discarded and the heuristic is used
        # instead. Default False: an untrusted estimator is still an estimator, and the
        # selector charges its uncertainty explicitly rather than throwing it away.
        self._require_trusted_gnn = bool(require_trusted_gnn)

    def predict(
        self,
        context: MarketContext,
        proposals: Sequence[StrategyProposal],
        costs: Mapping[str, CostEstimate],
        *,
        rows: Iterable[Any] = (),
    ) -> tuple[StrategyUtilityPrediction, ...]:
        gnn_predictions = {
            prediction.strategy_id: prediction
            for prediction in self._gnn.predict(context, proposals, costs, rows=rows)
            if not (
                self._require_trusted_gnn
                and UTILITY_GNN_UNTRUSTED in prediction.reason_codes
            )
        }
        missing = [
            proposal
            for proposal in proposals
            if proposal.strategy_id not in gnn_predictions
        ]
        heuristic = {
            prediction.strategy_id: _with_reason(prediction, UTILITY_GNN_ABSENT)
            for prediction in self._heuristic.predict(context, missing, costs)
        }
        merged = {**heuristic, **gnn_predictions}
        # Preserve proposal order so the ranking is stable across cycles.
        return tuple(
            merged[proposal.strategy_id]
            for proposal in proposals
            if proposal.strategy_id in merged
        )


def _with_reason(
    prediction: StrategyUtilityPrediction, reason: str
) -> StrategyUtilityPrediction:
    if reason in prediction.reason_codes:
        return prediction
    return StrategyUtilityPrediction(
        strategy_id=prediction.strategy_id,
        context_id=prediction.context_id,
        symbol=prediction.symbol,
        probability_profit=prediction.probability_profit,
        expected_gross_return_bps=prediction.expected_gross_return_bps,
        expected_cost_bps=prediction.expected_cost_bps,
        expected_downside_bps=prediction.expected_downside_bps,
        expected_holding_seconds=prediction.expected_holding_seconds,
        uncertainty_bps=prediction.uncertainty_bps,
        model_version=prediction.model_version,
        source=prediction.source,
        reason_codes=(*prediction.reason_codes, reason),
    )


def _rows_by_strategy(rows: Iterable[Any]) -> dict[str, Mapping[str, Any]]:
    """Index GNN vector rows by strategy id, accepting objects or dicts."""
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows or ():
        if isinstance(row, Mapping):
            payload = dict(row)
        else:
            payload = {
                name: getattr(row, name)
                for name in (
                    "strategy_id",
                    "probability_success",
                    "expected_gross_return_bps",
                    "expected_cost_bps",
                    "expected_net_return_bps",
                    "expected_adverse_excursion_bps",
                    "expected_holding_seconds",
                    "aleatoric_uncertainty",
                    "epistemic_uncertainty_or_proxy",
                    "model_version",
                    "reason_codes",
                )
                if hasattr(row, name)
            }
        strategy_id = str(payload.get("strategy_id") or "").strip().lower()
        if strategy_id:
            indexed[strategy_id] = payload
    return indexed
