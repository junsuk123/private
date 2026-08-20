"""The decision chain, end to end, with one durable trace per candidate.

::

    Calendar/Session -> GlobalContext -> DomesticContext -> SectorContext
      -> StockMicroContext -> OntologyGraph -> TemporalHeteroGNN
      -> Regime / TradeQuality -> StrategySelector -> FinalTradeGate
      -> PositionSizing -> OrderIntent

This module owns the *sequence* and the *record*, not the arithmetic: every stage is a
component that can be tested and replayed on its own, and the pipeline's job is to run
them in one order, on one snapshot of time, and write down what happened.

One clock per cycle
-------------------
Every context in a cycle is stamped with the same ``captured_at``. Two candidates in one
election must be judged against one market state, not against two reads a few hundred
milliseconds apart — that difference is invisible in a log and decisive in a comparison.

One transaction per cycle
-------------------------
Contexts, regime, model prediction, strategy decision and gate verdict commit together.
A crash between them would otherwise leave an order intent whose authorisation cannot be
produced, which is precisely the record an audit needs.

Nothing here submits an order
-----------------------------
The pipeline's terminal output is an ``OrderIntent`` row in ``CREATED``/``GATED`` state.
Submission belongs to ``LiveExecutionCoordinator``, which this module deliberately does
not import — so no configuration mistake can make a context calculation reach a broker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from app.context.domestic_context import DomesticContext
from app.context.global_context import GlobalContext
from app.context.regime import RegimeEstimate, RegimeEstimator, RegimeEvidence
from app.context.sector_context import SectorContext
from app.context.temporal_context import TemporalSnapshot
from app.data.freshness import DataFreshnessRegistry
from app.execution.order_state_machine import OrderState, OrderStateMachine
from app.models.gnn_runtime import GnnHealth, GnnHealthState, GnnPrediction, GnnRuntime
from app.models.graph_snapshot import (
    GraphSnapshot,
    GraphSnapshotBuilder,
    StockNodeObservation,
)
from app.risk.final_trade_gate import FinalTradeGate, GateDecision, GateInputs
from app.routing.regime_strategy_selector import RegimeStrategySelector
from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
    json_column,
)

__all__ = [
    "AccountState",
    "CandidateInput",
    "ContextDecisionPipeline",
    "CycleResult",
    "DecisionTrace",
]


@dataclass(frozen=True)
class AccountState:
    """Account facts the gate sizes against. Absent fields block, never default open."""

    equity: float | None = None
    cash: float | None = None
    reconciled: bool | None = None
    session_pnl_ratio: float | None = None
    drawdown_ratio: float | None = None
    #: Current exposure by ticker and by sector, in account currency.
    position_value_by_ticker: Mapping[str, float] = field(default_factory=dict)
    exposure_by_sector: Mapping[str, float] = field(default_factory=dict)
    total_market_exposure: float = 0.0


@dataclass(frozen=True)
class CandidateInput:
    """One candidate's micro state plus everything the gate needs about it."""

    ticker: str
    side: str = "BUY"
    sector: str | None = None
    venue: str | None = None
    market_group: str = "KR"

    # micro tape
    session_return: float | None = None
    vwap_distance_bps: float | None = None
    ema_gap_bps: float | None = None
    momentum: float | None = None
    realized_volatility: float | None = None
    volume_intensity: float | None = None
    trade_intensity: float | None = None
    spread_bps: float | None = None
    depth: float | None = None
    orderbook_imbalance: float | None = None
    trade_imbalance: float | None = None
    relative_strength: float | None = None
    breakout_state: float | None = None
    liquidity_score: float | None = None
    gap_bps: float | None = None
    event_score: float | None = None
    trend_strength: float | None = None
    opening_volatility_multiple: float | None = None
    data_age_seconds: float | None = None
    price_feed_divergence_bps: float | None = None
    reference_price: float | None = None
    #: Venue-reported halt for THIS symbol. Takes precedence over the cycle-wide flag,
    #: which can only answer "is the venue trading" — a suspended symbol inside an open
    #: session is the case that actually occurs.
    halted: bool | None = None
    peers: Sequence[str] = ()
    requested_position_fraction: float | None = None

    def graph_observation(self) -> StockNodeObservation:
        return StockNodeObservation(
            ticker=self.ticker,
            sector=self.sector,
            venue=self.venue,
            market_group=self.market_group,
            session_return=self.session_return,
            vwap_distance_bps=self.vwap_distance_bps,
            ema_gap_bps=self.ema_gap_bps,
            momentum=self.momentum,
            realized_volatility=self.realized_volatility,
            volume_intensity=self.volume_intensity,
            trade_intensity=self.trade_intensity,
            spread_bps=self.spread_bps,
            depth=self.depth,
            orderbook_imbalance=self.orderbook_imbalance,
            trade_imbalance=self.trade_imbalance,
            relative_strength=self.relative_strength,
            breakout_state=self.breakout_state,
            data_age_seconds=self.data_age_seconds,
            peers=tuple(self.peers),
        )

    def micro_features(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in {
                "session_return": self.session_return,
                "vwap_distance_bps": self.vwap_distance_bps,
                "ema_gap_bps": self.ema_gap_bps,
                "momentum": self.momentum,
                "realized_volatility": self.realized_volatility,
                "volume_intensity": self.volume_intensity,
                "trade_intensity": self.trade_intensity,
                "spread_bps": self.spread_bps,
                "depth": self.depth,
                "orderbook_imbalance": self.orderbook_imbalance,
                "trade_imbalance": self.trade_imbalance,
                "relative_strength": self.relative_strength,
                "breakout_state": self.breakout_state,
                "liquidity_score": self.liquidity_score,
            }.items()
            if value is not None and math.isfinite(float(value))
        }


@dataclass(frozen=True)
class DecisionTrace:
    """Everything needed to reconstruct one decision, months later."""

    decision_id: str
    ticker: str
    timestamp: datetime
    temporal_context: Mapping[str, Any]
    global_context: Mapping[str, Any]
    domestic_context: Mapping[str, Any]
    sector_context: Mapping[str, Any]
    micro_context: Mapping[str, Any]
    regime_probabilities: Mapping[str, float]
    strategy: str
    strategy_family: str
    action: str
    supporting_factors: tuple[str, ...]
    conflicting_factors: tuple[str, ...]
    ontology_relations: tuple[Mapping[str, Any], ...]
    learned_relation_weights: Mapping[str, float]
    model_confidence: float | None
    uncertainty: float | None
    gate_result: bool
    gate_reasons: tuple[str, ...]
    position_multiplier: float
    order_intent: Mapping[str, Any] | None
    execution_result: Mapping[str, Any] | None = None
    model_prediction: Mapping[str, Any] = field(default_factory=dict)
    model_health: Mapping[str, Any] = field(default_factory=dict)
    data_health: Mapping[str, Any] = field(default_factory=dict)
    gate_id: str | None = None
    strategy_candidates: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "ticker": self.ticker,
            "timestamp": iso_column(self.timestamp),
            "temporal_context": dict(self.temporal_context),
            "global_context": dict(self.global_context),
            "domestic_context": dict(self.domestic_context),
            "sector_context": dict(self.sector_context),
            "micro_context": dict(self.micro_context),
            "regime_probabilities": dict(self.regime_probabilities),
            "strategy": self.strategy,
            "strategy_family": self.strategy_family,
            "action": self.action,
            "supporting_factors": list(self.supporting_factors),
            "conflicting_factors": list(self.conflicting_factors),
            "ontology_relations": [dict(item) for item in self.ontology_relations],
            "learned_relation_weights": dict(self.learned_relation_weights),
            "model_confidence": self.model_confidence,
            "uncertainty": self.uncertainty,
            "gate_result": self.gate_result,
            "gate_reasons": list(self.gate_reasons),
            "position_multiplier": self.position_multiplier,
            "order_intent": dict(self.order_intent) if self.order_intent else None,
            "execution_result": (
                dict(self.execution_result) if self.execution_result else None
            ),
            "model_prediction": dict(self.model_prediction),
            "model_health": dict(self.model_health),
            "data_health": dict(self.data_health),
            "gate_id": self.gate_id,
            "strategy_candidates": [dict(item) for item in self.strategy_candidates],
        }


@dataclass(frozen=True)
class CycleResult:
    """One election's worth of decisions, plus the shared state they rest on."""

    cycle_id: str
    captured_at: datetime
    temporal: TemporalSnapshot
    regime: RegimeEstimate
    decisions: tuple[DecisionTrace, ...]
    snapshot: GraphSnapshot | None
    prediction: GnnPrediction | None
    model_health: GnnHealth | None
    data_health: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def approved(self) -> tuple[DecisionTrace, ...]:
        return tuple(item for item in self.decisions if item.gate_result)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "captured_at": iso_column(self.captured_at),
            "temporal": self.temporal.as_dict(),
            "regime": self.regime.as_dict(),
            "model_health": self.model_health.as_dict() if self.model_health else None,
            "data_health": dict(self.data_health),
            "decision_count": len(self.decisions),
            "approved_count": len(self.approved),
            "reason_codes": list(self.reason_codes),
            "decisions": [item.as_dict() for item in self.decisions],
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


class ContextDecisionPipeline:
    """Runs the chain for one cycle and persists the trace."""

    def __init__(
        self,
        *,
        store: TradingStateStore | None = None,
        gnn_runtime: GnnRuntime | None = None,
        snapshot_builder: GraphSnapshotBuilder | None = None,
        selector: RegimeStrategySelector | None = None,
        gate: FinalTradeGate | None = None,
        regime_estimator: RegimeEstimator | None = None,
        freshness: DataFreshnessRegistry | None = None,
        state_machine: OrderStateMachine | None = None,
        persist: bool = True,
    ) -> None:
        self._store = store or default_trading_state_store()
        self._gnn = gnn_runtime
        self._snapshots = snapshot_builder or GraphSnapshotBuilder()
        self._selector = selector or RegimeStrategySelector()
        self._gate = gate or FinalTradeGate()
        self._regimes = regime_estimator or RegimeEstimator()
        self._freshness = freshness
        self._states = state_machine
        self._persist = bool(persist)

    # ------------------------------------------------------------------ #
    def run_cycle(
        self,
        *,
        captured_at: datetime,
        temporal: TemporalSnapshot,
        candidates: Sequence[CandidateInput],
        global_context: GlobalContext | None = None,
        domestic_context: DomesticContext | None = None,
        sector_contexts: Sequence[SectorContext] = (),
        account: AccountState | None = None,
        websocket_connected: bool | None = None,
        trading_halted: bool | None = None,
        active_risk_conditions: Mapping[str, float] | None = None,
        change_point_probability: float | None = None,
        create_order_intents: bool = False,
    ) -> CycleResult:
        moment = _aware(captured_at)
        cycle_id = f"cyc-{moment.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        reasons: list[str] = []
        account_state = account or AccountState()
        sectors_by_name = {context.sector: context for context in sector_contexts}

        # -- data health ------------------------------------------------- #
        stale_reasons: tuple[str, ...] = ()
        data_health: dict[str, Any] = {}
        if self._freshness is not None:
            stale_reasons = self._freshness.blocking_reasons(now=moment)
            data_health = self._freshness.report(now=moment)

        # -- risk conditions ----------------------------------------------- #
        risk_conditions = dict(active_risk_conditions or {})
        risk_conditions.update(
            self._derive_risk_conditions(
                # Symbol-scoped freshness is evaluated against that symbol's
                # gate below. It is not a market-wide ontology risk condition.
                stale_reasons=_global_stale_reasons(stale_reasons),
                domestic_context=domestic_context,
                global_context=global_context,
            )
        )

        # -- graph + model --------------------------------------------------- #
        snapshot = self._snapshots.build(
            captured_at=moment,
            temporal=temporal,
            global_context=global_context,
            domestic_context=domestic_context,
            sector_contexts=sector_contexts,
            stocks=[candidate.graph_observation() for candidate in candidates],
            active_risk_conditions=risk_conditions,
        )
        reasons.extend(snapshot.reason_codes)

        prediction: GnnPrediction | None = None
        health: GnnHealth | None = None
        if self._gnn is not None:
            prediction = self._gnn.predict(snapshot, now=moment)
            health = self._gnn.health(now=moment)
            if prediction is None:
                reasons.append("GNN_NO_PREDICTION")

        # -- regime ------------------------------------------------------------ #
        model_regime = None
        if prediction is not None and health is not None and health.allows_model_evidence:
            market_payload = prediction.for_market(temporal.market_group)
            if market_payload:
                model_regime = {
                    str(label): float(value)
                    for label, value in market_payload["market_regime"].items()
                }
        regime = self._regimes.estimate(
            RegimeEvidence.from_contexts(
                domestic=domestic_context,
                global_context=global_context,
                temporal=temporal,
                change_point_probability=change_point_probability,
            ),
            evaluated_at=moment,
            model_probabilities=model_regime,
            model_version=prediction.model_version if prediction else None,
        )

        # -- per-candidate decisions --------------------------------------------- #
        decisions: list[DecisionTrace] = []
        for candidate in candidates:
            candidate_stale_reasons = _stale_reasons_for_ticker(
                stale_reasons, candidate.ticker
            )
            decisions.append(
                self._decide(
                    candidate,
                    moment=moment,
                    temporal=temporal,
                    global_context=global_context,
                    domestic_context=domestic_context,
                    sector_context=sectors_by_name.get(candidate.sector or ""),
                    regime=regime,
                    prediction=prediction,
                    health=health,
                    risk_conditions=risk_conditions,
                    stale_reasons=candidate_stale_reasons,
                    account=account_state,
                    websocket_connected=websocket_connected,
                    trading_halted=trading_halted,
                    data_health=data_health,
                    create_order_intent=create_order_intents,
                )
            )

        result = CycleResult(
            cycle_id=cycle_id,
            captured_at=moment,
            temporal=temporal,
            regime=regime,
            decisions=tuple(decisions),
            snapshot=snapshot,
            prediction=prediction,
            model_health=health,
            data_health=data_health,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
        if self._persist:
            self._persist_cycle(
                result,
                global_context=global_context,
                domestic_context=domestic_context,
                sector_contexts=sector_contexts,
                candidates=candidates,
            )
        return result

    # ------------------------------------------------------------------ #
    def _decide(
        self,
        candidate: CandidateInput,
        *,
        moment: datetime,
        temporal: TemporalSnapshot,
        global_context: GlobalContext | None,
        domestic_context: DomesticContext | None,
        sector_context: SectorContext | None,
        regime: RegimeEstimate,
        prediction: GnnPrediction | None,
        health: GnnHealth | None,
        risk_conditions: Mapping[str, float],
        stale_reasons: Sequence[str],
        account: AccountState,
        websocket_connected: bool | None,
        trading_halted: bool | None,
        data_health: Mapping[str, Any],
        create_order_intent: bool,
    ) -> DecisionTrace:
        decision_id = f"dec-{uuid4().hex}"
        model_healthy = bool(health and health.allows_model_evidence)
        node_payload: Mapping[str, Any] = {}
        if prediction is not None:
            node_payload = prediction.for_ticker(candidate.ticker) or {}

        micro = self._selector.micro_confirmation_from_context(
            trend_strength=candidate.trend_strength,
            orderflow_imbalance=candidate.orderbook_imbalance,
            breakout_state=candidate.breakout_state,
            relative_strength=candidate.relative_strength,
            vwap_distance_bps=candidate.vwap_distance_bps,
            liquidity_score=candidate.liquidity_score,
            gap_bps=candidate.gap_bps,
            event_score=candidate.event_score,
        )
        selection = self._selector.select(
            ticker=candidate.ticker,
            regime=regime,
            decided_at=moment,
            session_phase=temporal.session_phase.value,
            expiry_context=temporal.expiry_context.value,
            active_risk_conditions=risk_conditions,
            model_suitability=node_payload.get("strategy_suitability"),
            model_healthy=model_healthy,
            micro=micro,
        )

        model_confidence = self._model_confidence(
            node_payload, regime=regime, model_healthy=model_healthy
        )
        uncertainty = (
            float(node_payload["uncertainty"])
            if model_healthy and "uncertainty" in node_payload
            else None
        )

        gate_inputs = self._gate_inputs(
            candidate,
            moment=moment,
            temporal=temporal,
            domestic_context=domestic_context,
            sector_context=sector_context,
            regime=regime,
            health=health,
            stale_reasons=stale_reasons,
            account=account,
            websocket_connected=websocket_connected,
            trading_halted=trading_halted,
            model_confidence=model_confidence,
            decision_id=decision_id,
        )
        # A WAIT is not routed, but the gate still runs: an operator asking "why did
        # nothing trade" needs both answers, and running only one of them makes the other
        # unknowable after the fact.
        gate = self._gate.evaluate(gate_inputs)
        approved = bool(gate.approved and not selection.is_wait)

        order_intent: Mapping[str, Any] | None = None
        if approved and create_order_intent and self._states is not None:
            order_intent = self._create_intent(
                candidate,
                gate=gate,
                decision_id=decision_id,
                moment=moment,
                equity=account.equity,
            )

        selected = selection.selected
        return DecisionTrace(
            decision_id=decision_id,
            ticker=candidate.ticker,
            timestamp=moment,
            temporal_context=temporal.as_dict(),
            global_context=global_context.as_dict() if global_context else {},
            domestic_context=domestic_context.as_dict() if domestic_context else {},
            sector_context=sector_context.as_dict() if sector_context else {},
            micro_context=candidate.micro_features(),
            regime_probabilities=dict(regime.probabilities),
            strategy=(
                ",".join(selection.strategy_ids()) if not selection.is_wait else "WAIT"
            ),
            strategy_family=selection.family,
            action=selection.action if approved else "WAIT",
            supporting_factors=selected.supporting_factors if selected else (),
            conflicting_factors=(
                selected.conflicting_factors if selected else tuple(selection.reasons)
            ),
            ontology_relations=selected.ontology_relations if selected else (),
            learned_relation_weights=self._learned_weights(selected),
            model_confidence=model_confidence,
            uncertainty=uncertainty,
            gate_result=approved,
            gate_reasons=tuple([*gate.reasons, *selection.reasons]),
            position_multiplier=gate.position_multiplier,
            order_intent=order_intent,
            model_prediction=dict(node_payload),
            model_health=health.as_dict() if health else {},
            data_health={
                "worst_state": data_health.get("worst_state"),
                # A stale quote for A must never make B look unsafe. Unscoped
                # sources (account/global context) still apply to every ticker.
                "blocking_reasons": list(stale_reasons),
            },
            gate_id=gate.gate_id,
            strategy_candidates=tuple(item.as_dict() for item in selection.ranked),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _model_confidence(
        node_payload: Mapping[str, Any],
        *,
        regime: RegimeEstimate,
        model_healthy: bool,
    ) -> float | None:
        """Confidence handed to the gate.

        With a healthy model this is ``trade_quality`` discounted by its own uncertainty
        — a high-quality call the model is unsure of is not a high-confidence call.
        Without one it falls back to the regime estimator's coverage confidence, which is
        rule-derived and therefore still auditable.
        """
        if not model_healthy or "trade_quality" not in node_payload:
            return regime.confidence
        quality = float(node_payload["trade_quality"])
        uncertainty = float(node_payload.get("uncertainty") or 0.0)
        return round(max(0.0, min(1.0, quality / (1.0 + uncertainty))), 6)

    @staticmethod
    def _learned_weights(selected: Any) -> dict[str, float]:
        if selected is None:
            return {}
        return {
            str(relation["edge_id"]): float(relation["learned_weight"])
            for relation in selected.ontology_relations
            if relation.get("learned_weight") is not None
        }

    def _derive_risk_conditions(
        self,
        *,
        stale_reasons: Sequence[str],
        domestic_context: DomesticContext | None,
        global_context: GlobalContext | None,
    ) -> dict[str, float]:
        """Ontology risk-condition nodes activated by the current context."""
        conditions: dict[str, float] = {}
        if stale_reasons:
            conditions["STALE_DATA"] = 1.0
        if domestic_context is not None:
            if domestic_context.liquidity is not None and domestic_context.liquidity < 0.35:
                conditions["LOW_LIQUIDITY"] = round(
                    1.0 - domestic_context.liquidity / 0.35, 6
                )
            if domestic_context.global_conflict:
                conditions["GLOBAL_CONFLICT"] = abs(
                    float(domestic_context.global_agreement or 0.0)
                )
            if domestic_context.venue_divergence:
                conditions["WIDE_SPREAD"] = float(domestic_context.venue_divergence)
        if global_context is not None and global_context.volatility is not None:
            if global_context.volatility > 1.0:
                conditions["HIGH_VOLATILITY"] = min(
                    1.0, (global_context.volatility - 1.0)
                )
        return conditions

    def _gate_inputs(
        self,
        candidate: CandidateInput,
        *,
        moment: datetime,
        temporal: TemporalSnapshot,
        domestic_context: DomesticContext | None,
        sector_context: SectorContext | None,
        regime: RegimeEstimate,
        health: GnnHealth | None,
        stale_reasons: Sequence[str],
        account: AccountState,
        websocket_connected: bool | None,
        trading_halted: bool | None,
        model_confidence: float | None,
        decision_id: str,
    ) -> GateInputs:
        duplicate_risk: bool | None = None
        unknown_orders: tuple[str, ...] = ()
        if self._states is not None:
            duplicate_risk = self._states.has_duplicate_risk(
                candidate.ticker, candidate.side
            )
            unknown_orders = tuple(
                record.intent_id
                for record in self._states.unknown_intents()
                if record.ticker == candidate.ticker.upper()
            )
        return GateInputs(
            ticker=candidate.ticker,
            side=candidate.side,
            evaluated_at=moment,
            stale_data_reasons=tuple(stale_reasons),
            websocket_connected=websocket_connected,
            price_feed_divergence_bps=candidate.price_feed_divergence_bps,
            session_id=temporal.phase_state.primary_session,
            session_allows_new_entry=_session_allows_entry(temporal),
            trading_halted=(
                candidate.halted if candidate.halted is not None else trading_halted
            ),
            account_reconciled=account.reconciled,
            unknown_order_ids=unknown_orders,
            duplicate_order_risk=duplicate_risk,
            model_health_state=health.state.value if health else None,
            risk_engine_ok=True,
            realized_volatility=candidate.realized_volatility,
            liquidity_score=candidate.liquidity_score,
            global_agreement=(
                domestic_context.global_agreement if domestic_context else None
            ),
            sector_relative_strength=(
                sector_context.relative_strength if sector_context else None
            ),
            model_confidence=model_confidence,
            session_phase=temporal.session_phase.value,
            opening_volatility_multiple=candidate.opening_volatility_multiple,
            spread_bps=candidate.spread_bps,
            dominant_regime=regime.dominant,
            account_equity=account.equity,
            current_position_value=float(
                account.position_value_by_ticker.get(candidate.ticker.upper(), 0.0)
            ),
            current_sector_exposure=float(
                account.exposure_by_sector.get(candidate.sector or "", 0.0)
            ),
            current_market_exposure=float(account.total_market_exposure),
            session_pnl_ratio=account.session_pnl_ratio,
            drawdown_ratio=account.drawdown_ratio,
            requested_position_fraction=candidate.requested_position_fraction,
            decision_id=decision_id,
            sector=candidate.sector,
        )

    def _create_intent(
        self,
        candidate: CandidateInput,
        *,
        gate: GateDecision,
        decision_id: str,
        moment: datetime,
        equity: float | None,
    ) -> Mapping[str, Any] | None:
        """Turn an approved decision into a durable, gated order intent.

        Quantity comes from the gate's approved fraction and the candidate's reference
        price — never from the strategy's request, which the gate has already capped.
        A fraction that rounds down to zero shares produces no intent rather than a
        one-share order nobody asked for.
        """
        if self._states is None:
            return None
        price = candidate.reference_price
        if price is None or price <= 0.0:
            return None
        if equity is None or equity <= 0.0:
            return None
        notional = gate.approved_position_fraction * float(equity)
        if notional <= 0.0:
            return None
        quantity = int(notional // price)
        if quantity <= 0:
            return None
        record = self._states.create(
            ticker=candidate.ticker,
            side=candidate.side,
            quantity=quantity,
            idempotency_key=f"{decision_id}:{candidate.ticker.upper()}:{candidate.side.upper()}",
            limit_price=price,
            market_group=candidate.market_group,
            venue=candidate.venue or "",
            decision_id=decision_id,
            gate_id=gate.gate_id,
            payload={"position_fraction": gate.approved_position_fraction},
            now=moment,
        )
        gated = self._states.transition(
            record.intent_id,
            OrderState.GATED,
            reason=f"GATE:{gate.gate_id}",
            now=moment,
        )
        return gated.as_dict()

    # ------------------------------------------------------------------ #
    def _persist_cycle(
        self,
        result: CycleResult,
        *,
        global_context: GlobalContext | None,
        domestic_context: DomesticContext | None,
        sector_contexts: Sequence[SectorContext],
        candidates: Sequence[CandidateInput],
    ) -> None:
        moment = result.captured_at
        temporal = result.temporal
        with self._store.transaction() as conn:
            conn.execute(
                "insert or replace into market_session"
                " (session_key, market_group, trading_day, session_id, venue, phase,"
                "  session_start, session_end, data_available, trade_available,"
                "  new_entry_allowed, exit_allowed, reason_codes_json, recorded_at)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{temporal.market_group}:{temporal.trading_day}:{temporal.session_phase.value}",
                    temporal.market_group,
                    str(temporal.trading_day),
                    temporal.phase_state.primary_session,
                    "",
                    temporal.session_phase.value,
                    iso_column(temporal.phase_state.continuous_open),
                    iso_column(temporal.phase_state.continuous_close),
                    1 if temporal.is_trading_day else 0,
                    1 if temporal.phase_state.is_continuous else 0,
                    1 if _session_allows_entry(temporal) else 0,
                    1,
                    json_column(list(temporal.calendar_reasons)),
                    iso_column(moment),
                ),
            )
            if global_context is not None:
                conn.execute(
                    "insert or replace into global_context"
                    " (context_id, captured_at, trading_day, session_phase, direction,"
                    "  momentum, risk_sentiment, volatility, rates_pressure, fx_pressure,"
                    "  global_alignment, confidence, group_scores_json, sources_json,"
                    "  reason_codes_json)"
                    " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        global_context.context_id,
                        iso_column(global_context.captured_at),
                        str(temporal.trading_day),
                        temporal.session_phase.value,
                        global_context.direction,
                        global_context.momentum,
                        global_context.risk_sentiment,
                        global_context.volatility,
                        global_context.rates_pressure,
                        global_context.fx_pressure,
                        global_context.global_alignment,
                        global_context.confidence,
                        json_column(
                            {
                                name: score.as_dict()
                                for name, score in global_context.groups.items()
                            }
                        ),
                        json_column({}),
                        json_column(list(global_context.reason_codes)),
                    ),
                )
            if domestic_context is not None:
                conn.execute(
                    "insert or replace into domestic_context"
                    " (context_id, captured_at, trading_day, session_phase,"
                    "  global_context_id, direction, breadth, liquidity, volatility,"
                    "  flow, leadership, venue_divergence, confidence, components_json,"
                    "  sources_json, reason_codes_json)"
                    " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        domestic_context.context_id,
                        iso_column(domestic_context.captured_at),
                        str(temporal.trading_day),
                        temporal.session_phase.value,
                        domestic_context.global_context_id,
                        domestic_context.direction,
                        domestic_context.breadth,
                        domestic_context.liquidity,
                        domestic_context.volatility,
                        domestic_context.flow,
                        domestic_context.leadership,
                        domestic_context.venue_divergence,
                        domestic_context.confidence,
                        json_column(dict(domestic_context.components)),
                        json_column({}),
                        json_column(list(domestic_context.reason_codes)),
                    ),
                )
            for sector in sector_contexts:
                conn.execute(
                    "insert or replace into sector_context"
                    " (context_id, captured_at, sector, market_group, domestic_context_id,"
                    "  return_value, breadth, volume_z, volatility, relative_strength,"
                    "  foreign_flow, leader_strength, leader_concentration,"
                    "  global_alignment, confidence, member_count, sources_json,"
                    "  reason_codes_json)"
                    " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sector.context_id,
                        iso_column(sector.captured_at),
                        sector.sector,
                        sector.market_group,
                        sector.domestic_context_id,
                        sector.sector_return,
                        sector.breadth,
                        sector.volume_z,
                        sector.volatility,
                        sector.relative_strength,
                        sector.foreign_flow,
                        sector.leader_strength,
                        sector.leader_concentration,
                        sector.global_alignment,
                        sector.confidence,
                        sector.member_count,
                        json_column({}),
                        json_column(list(sector.reason_codes)),
                    ),
                )
            sector_by_ticker = {
                candidate.ticker.upper(): candidate.sector for candidate in candidates
            }
            for candidate in candidates:
                conn.execute(
                    "insert or replace into stock_context"
                    " (context_id, captured_at, ticker, market_group, sector,"
                    "  sector_context_id, features_json, confidence, reason_codes_json)"
                    " values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"kctx-{result.cycle_id}-{candidate.ticker.upper()}",
                        iso_column(moment),
                        candidate.ticker.upper(),
                        candidate.market_group,
                        candidate.sector,
                        next(
                            (
                                sector.context_id
                                for sector in sector_contexts
                                if sector.sector == candidate.sector
                            ),
                            None,
                        ),
                        json_column(candidate.micro_features()),
                        1.0 if candidate.micro_features() else 0.0,
                        json_column([]),
                    ),
                )

            regime_id = f"rp-{result.cycle_id}"
            conn.execute(
                "insert or replace into regime_prediction"
                " (prediction_id, predicted_at, scope, scope_key, probabilities_json,"
                "  dominant_regime, entropy, source, model_version, confidence)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    regime_id,
                    iso_column(result.regime.evaluated_at),
                    "market",
                    temporal.market_group,
                    json_column(dict(result.regime.probabilities)),
                    result.regime.dominant,
                    result.regime.entropy,
                    result.regime.source,
                    result.regime.model_version or "",
                    result.regime.confidence,
                ),
            )
            if result.model_health is not None:
                conn.execute(
                    "insert or replace into model_health"
                    " (health_id, observed_at, model_name, state, reason_codes_json,"
                    "  detail_json) values (?, ?, ?, ?, ?, ?)",
                    (
                        f"mh-{result.cycle_id}",
                        iso_column(result.model_health.observed_at),
                        "temporal_hetero_gnn",
                        result.model_health.state.value,
                        json_column(list(result.model_health.reason_codes)),
                        json_column(result.model_health.as_dict()),
                    ),
                )

            for decision in result.decisions:
                prediction_id: str | None = None
                if decision.model_prediction:
                    prediction_id = f"mp-{decision.decision_id}"
                    conn.execute(
                        "insert or replace into model_prediction"
                        " (prediction_id, predicted_at, model_name, model_version,"
                        "  ticker, heads_json, uncertainty, confidence, health,"
                        "  input_context_ids_json) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            prediction_id,
                            iso_column(decision.timestamp),
                            "temporal_hetero_gnn",
                            (
                                result.prediction.model_version
                                if result.prediction
                                else ""
                            ),
                            decision.ticker,
                            json_column(dict(decision.model_prediction)),
                            decision.uncertainty,
                            decision.model_confidence,
                            (
                                result.model_health.state.value
                                if result.model_health
                                else GnnHealthState.OFFLINE.value
                            ),
                            json_column(
                                [
                                    global_context.context_id if global_context else None,
                                    (
                                        domestic_context.context_id
                                        if domestic_context
                                        else None
                                    ),
                                ]
                            ),
                        ),
                    )
                conn.execute(
                    "insert or replace into strategy_decision"
                    " (decision_id, decided_at, ticker, strategy, strategy_family,"
                    "  action, regime_prediction_id, model_prediction_id,"
                    "  global_context_id, domestic_context_id, sector_context_id,"
                    "  stock_context_id, supporting_factors_json,"
                    "  conflicting_factors_json, ontology_relations_json,"
                    "  learned_relation_weights_json, model_confidence, uncertainty,"
                    "  trace_json) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
                    "  ?, ?, ?, ?)",
                    (
                        decision.decision_id,
                        iso_column(decision.timestamp),
                        decision.ticker,
                        decision.strategy,
                        decision.strategy_family,
                        decision.action,
                        regime_id,
                        prediction_id,
                        global_context.context_id if global_context else None,
                        domestic_context.context_id if domestic_context else None,
                        next(
                            (
                                sector.context_id
                                for sector in sector_contexts
                                if sector.sector == sector_by_ticker.get(decision.ticker)
                            ),
                            None,
                        ),
                        f"kctx-{result.cycle_id}-{decision.ticker}",
                        json_column(list(decision.supporting_factors)),
                        json_column(list(decision.conflicting_factors)),
                        json_column([dict(item) for item in decision.ontology_relations]),
                        json_column(dict(decision.learned_relation_weights)),
                        decision.model_confidence,
                        decision.uncertainty,
                        json_column(decision.as_dict()),
                    ),
                )
                conn.execute(
                    "insert or replace into gate_decision"
                    " (gate_id, decision_id, evaluated_at, ticker, approved,"
                    "  hard_failures_json, soft_failures_json, reasons_json,"
                    "  position_multiplier, account_snapshot_id, data_health_id,"
                    "  detail_json) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        decision.gate_id or f"gate-{decision.decision_id}",
                        decision.decision_id,
                        iso_column(decision.timestamp),
                        decision.ticker,
                        1 if decision.gate_result else 0,
                        json_column(
                            [
                                reason.split(":", 1)[1]
                                for reason in decision.gate_reasons
                                if reason.startswith("HARD:")
                            ]
                        ),
                        json_column(
                            [
                                reason.split(":", 1)[1]
                                for reason in decision.gate_reasons
                                if reason.startswith("SOFT:")
                            ]
                        ),
                        json_column(list(decision.gate_reasons)),
                        decision.position_multiplier,
                        None,
                        None,
                        json_column(dict(decision.data_health)),
                    ),
                )


def _session_allows_entry(temporal: TemporalSnapshot) -> bool:
    """Is the session one in which a NEW position may be opened?

    Read from the capability service's own answer rather than from the phase label: the
    phase says where on the arc we are, the capability says whether the broker will
    accept the order, and only the second may authorise one.
    """
    try:
        from app.data.market_capabilities import default_service, normalize_market_group

        market = normalize_market_group(temporal.market_group)
        if market is None:
            return False
        return default_service().new_entry_allowed(market, temporal.as_of)
    except Exception:  # noqa: BLE001 - an unreadable capability is a closed session.
        return False

def _stale_reasons_for_ticker(
    reasons: Sequence[str], ticker: str
) -> tuple[str, ...]:
    """Return global reasons plus reasons scoped to ``ticker``.

    Freshness codes are ``STALE_DATA:source/type[:scope]``.  The registry keeps
    observations for rotating symbols, so passing its complete blocking list to
    every candidate made one inactive symbol veto the whole market.
    """
    wanted = str(ticker or "").strip().upper()
    selected: list[str] = []
    for raw in reasons:
        reason = str(raw)
        parts = reason.split(":", 2)
        if len(parts) < 3 or str(parts[2]).strip().upper() == wanted:
            selected.append(reason)
    return tuple(dict.fromkeys(selected))


def _global_stale_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    """Freshness failures without a symbol scope affect market-wide context."""
    return tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason).count(":") < 2))
