"""Read-mostly HTTP surface for the short-strategy deployment ladder.

Answers the operator questions that matter while a short strategy is being proven:

* what state is each arm in, and does that state submit orders?
* what is its confidence score, and which component is holding it back?
* what conditions remain before the next promotion?
* what did the automatic promotions and demotions actually decide, and on what
  numbers?
* is the borrow desk healthy, and what is it refusing?
* at any given moment, would a short have helped — LONG vs SHORT vs NO_TRADE?

One endpoint mutates: ``POST /api/short-strategies/{id}/suspend``. Suspension is the
only manual action exposed, deliberately. An operator can always make the system
SAFER without argument; making it less safe has to go through the evidence ladder.
There is no promote endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse

from app.strategy.catalog import SHORT_LONG_COUNTERPART, SHORT_STRATEGY_IDS
from app.trading.directional import (
    ENTRY_BLOCKADE_STAGES,
    DirectionalStrategyKey,
    StrategyDeploymentState,
)

logger = logging.getLogger(__name__)


def _json(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status_code)


def _error(exc: Exception, **extra: Any) -> JSONResponse:
    """Diagnostics must never 500 the dashboard.

    A short strategy is invisible until it is promoted, so the dashboard IS the only
    view into it. An endpoint that 500s while an arm is mid-ladder removes the only
    way to see why — so failures are reported as data.
    """
    logger.warning("short-strategy route failed: %s", exc, exc_info=True)
    return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}", **extra})


def create_short_strategy_router(
    *,
    controller_provider: Callable[[], Any] | None = None,
    shadow_store_provider: Callable[[], Any] | None = None,
    borrow_store_provider: Callable[[], Any] | None = None,
    session_snapshot_provider: Callable[[], dict[str, Any]] | None = None,
    markets: Sequence[str] = ("KR",),
) -> APIRouter:
    """Build the router. Providers are injected so tests need no live stores."""
    router = APIRouter(tags=["short-strategies"])

    def _controller() -> Any:
        if controller_provider is not None:
            return controller_provider()
        from app.trading.short_strategy_promotion import default_promotion_controller

        return default_promotion_controller()

    def _shadow_store() -> Any:
        if shadow_store_provider is not None:
            return shadow_store_provider()
        from app.trading.directional_shadow import default_shadow_store

        return default_shadow_store()

    def _borrow_store() -> Any:
        if borrow_store_provider is not None:
            return borrow_store_provider()
        from app.trading.borrow import default_borrow_store

        return default_borrow_store()

    def _resolve_key(strategy_id: str, market: str) -> DirectionalStrategyKey:
        return DirectionalStrategyKey.for_short(strategy_id, market)

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #
    @router.get("/api/short-strategies/status")
    def short_strategy_status() -> JSONResponse:
        """Every managed short arm: state, confidence, next rung, borrow health."""
        try:
            controller = _controller()
            payload = controller.status(markets=tuple(markets))
            payload["ok"] = True
            payload["catalogued_short_strategies"] = list(SHORT_STRATEGY_IDS)
            payload["long_counterparts"] = dict(SHORT_LONG_COUNTERPART)
            # Stated explicitly so an operator never has to infer it from the state
            # name: this is the set of arms that can currently place a real order.
            payload["order_authorized_arms"] = [
                arm.get("strategy_key")
                for arm in payload.get("arms", [])
                if arm.get("submits_orders")
            ]
            payload["shadow_summary"] = _shadow_store().summary(limit=50)
            # Where 대주 availability comes from, stated explicitly. With no source the
            # ladder is inert rather than broken, and an operator must be able to see
            # that difference — a status page showing only "SHADOW, 0 samples" looks
            # identical whether the system is patiently gathering evidence or has no
            # data path at all.
            payload["borrow_source"] = _borrow_source_status()
            # Which short indicators have no source, and what that costs. A silently
            # absent crowding metric looks identical to a crowding check that passed.
            payload["indicator_gaps"] = _indicator_gap_status()
            return _json(payload)
        except Exception as exc:  # noqa: BLE001
            return _error(exc, arms=[])

    @router.get("/api/short-strategies/{strategy_id}/validation")
    def short_strategy_validation(
        strategy_id: str, market: str = Query("KR")
    ) -> JSONResponse:
        """Full validation snapshot: every metric, every gate, what remains.

        This is the endpoint that answers "what does this strategy still need". It
        reports the FULL failing-gate list rather than the first failure, because an
        operator asking the question wants the whole list — one at a time turns a
        single diagnosis into as many evaluation cycles as there are gates.
        """
        try:
            controller = _controller()
            key = _resolve_key(strategy_id, market)
            record = controller.state_store.get(key)
            state = record.state if record else StrategyDeploymentState.SHADOW
            snapshot = controller.build_snapshot(key, state)
            thresholds = controller.config.thresholds_for(state)
            from app.trading.short_strategy_promotion import evaluate_hard_gates

            failing = (
                list(evaluate_hard_gates(snapshot, thresholds)) if thresholds else []
            )
            return _json(
                {
                    "ok": True,
                    **key.as_dict(),
                    "state": str(state),
                    "submits_orders": state.submits_orders,
                    "confidence_score": round(snapshot.confidence_score, 4),
                    "confidence_components": {
                        name: round(value, 4)
                        for name, value in snapshot.confidence_components.items()
                    },
                    "confidence_weights": _confidence_weights(),
                    "metrics": snapshot.as_dict(),
                    "thresholds": _threshold_dict(thresholds),
                    # The gate list IS the "remaining conditions" answer.
                    "failing_gates": failing,
                    "remaining_conditions": failing,
                    "consecutive_passes": record.consecutive_passes if record else 0,
                    "required_consecutive_cycles": (
                        thresholds.required_consecutive_evaluation_cycles
                        if thresholds
                        else None
                    ),
                    "next_state": _next_state_name(state),
                    # A high confidence score never overrides a failing hard gate.
                    # Surfaced so a dashboard cannot present the score as a countdown.
                    "hard_gate_precedence": (
                        "confidence_score does not override a failing hard gate"
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc, strategy_id=strategy_id)

    @router.get("/api/short-strategies/{strategy_id}/deployment-history")
    def short_strategy_deployment_history(
        strategy_id: str,
        market: str = Query("KR"),
        limit: int = Query(50, ge=1, le=500),
    ) -> JSONResponse:
        """Audit trail of every state transition, with the numbers that caused it.

        Written in the same transaction as the state change, so a promotion can always
        be re-argued from the metrics that actually caused it rather than from metrics
        recomputed against today's data.
        """
        try:
            controller = _controller()
            key = _resolve_key(strategy_id, market)
            events = controller.state_store.audit_history(key, limit=limit)
            return _json(
                {
                    "ok": True,
                    **key.as_dict(),
                    "event_count": len(events),
                    "events": list(events),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc, strategy_id=strategy_id, events=[])

    @router.get("/api/short-strategies/{strategy_id}/shadow-outcomes")
    def short_strategy_shadow_outcomes(
        strategy_id: str,
        market: str = Query("KR"),
        limit: int = Query(100, ge=1, le=1000),
        scored_only: bool = Query(False),
    ) -> JSONResponse:
        """Forward shadow outcomes for one arm.

        ``scored_only=false`` includes ``SIGNAL_VALID_BUT_UNEXECUTABLE`` rows — signals
        that fired with no locatable borrow. They are excluded from every promotion
        statistic (a strategy may not be promoted on trades it could not have taken) but
        they are the evidence behind ``borrow_availability_rate``, and hiding them would
        make an arm look inexplicably starved of samples.
        """
        try:
            key = _resolve_key(strategy_id, market)
            outcomes = _shadow_store().outcomes(
                key, scored_only=scored_only, limit=limit
            )
            scored = [item for item in outcomes if item.get("scored")]
            unexecutable = [item for item in outcomes if not item.get("executable")]
            return _json(
                {
                    "ok": True,
                    **key.as_dict(),
                    "outcome_count": len(outcomes),
                    "scored_count": len(scored),
                    "signal_valid_but_unexecutable_count": len(unexecutable),
                    "outcomes": list(outcomes),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc, strategy_id=strategy_id, outcomes=[])

    # ------------------------------------------------------------------ #
    # Borrow                                                              #
    # ------------------------------------------------------------------ #
    @router.get("/api/borrow/{symbol}/availability")
    def borrow_availability(
        symbol: str, history: int = Query(20, ge=0, le=200)
    ) -> JSONResponse:
        """Latest borrow locate for one symbol, plus recent history.

        ``available: null`` means NO OBSERVATION EXISTS, which is different from
        ``false`` (the broker said no). The first is an operational fault worth
        alerting on; the second is a normal market state. Collapsing them would hide a
        credentials or endpoint outage behind a normal-looking no-locate.
        """
        try:
            store = _borrow_store()
            latest = store.latest(symbol)
            payload: dict[str, Any] = {
                "ok": True,
                "symbol": str(symbol or "").upper(),
                "observed": latest is not None,
                "available": latest.available if latest else None,
                "snapshot": latest.as_dict() if latest else None,
            }
            if history:
                payload["history"] = [
                    item.as_dict() for item in store.history(symbol, limit=history)
                ]
            return _json(payload)
        except Exception as exc:  # noqa: BLE001
            return _error(exc, symbol=symbol, available=None)

    @router.get("/api/borrow/health")
    def borrow_health(window_seconds: float = Query(3600.0, gt=0)) -> JSONResponse:
        """Borrow-desk health: availability rate, rejection rate, refusal reasons.

        Rates are ``null`` when nothing was asked in the window. "We asked nothing"
        must not read as "nothing was available" — that would demote a strategy for an
        outage in the polling loop rather than for a market condition.
        """
        try:
            return _json({"ok": True, **_borrow_store().health(window_seconds=window_seconds)})
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ------------------------------------------------------------------ #
    # Directional comparison                                              #
    # ------------------------------------------------------------------ #
    @router.get("/api/directional-bandit/evaluations")
    def directional_bandit_evaluations() -> JSONResponse:
        """LONG vs SHORT vs NO_TRADE at the last election.

        ``short_rescued`` is the question the whole short programme has to justify
        itself against: did adding these arms convert a NO_TRADE into a real
        opportunity, or only add exposure alongside a better long? It is recorded on
        every cycle regardless of whether a short was selectable, and it feeds the
        ``minimum_short_rescue_rate`` promotion gate.
        """
        try:
            snapshot = (
                session_snapshot_provider() if session_snapshot_provider else {}
            ) or {}
            evaluations = list(snapshot.get("bandit_evaluations") or ())
            return _json(
                {
                    "ok": True,
                    "selected_arm": snapshot.get("bandit_selected_arm"),
                    "selected_direction": snapshot.get("selected_direction"),
                    "conservative_edge_bps": snapshot.get("bandit_conservative_edge_bps"),
                    "is_exploration": snapshot.get("bandit_is_exploration"),
                    "reason_codes": list(snapshot.get("bandit_reason_codes") or ()),
                    "shadow_arms": list(snapshot.get("bandit_shadow_arms") or ()),
                    "directional_comparison": dict(
                        snapshot.get("directional_comparison") or {}
                    ),
                    "evaluations": evaluations,
                    "long_evaluations": [
                        item for item in evaluations if item.get("direction") == "LONG"
                    ],
                    "short_evaluations": [
                        item for item in evaluations if item.get("direction") == "SHORT"
                    ],
                    "last_reason": snapshot.get("last_reason"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc, evaluations=[])

    @router.get("/api/short-strategies/entry-blockade")
    def short_entry_blockade(
        strategy_id: str = Query(""), market: str = Query("KR")
    ) -> JSONResponse:
        """The FIRST stage blocking a short entry, not the last complaint emitted.

        Ordered stages, so the answer is actionable: a deployment-authorization block is
        "still learning, nothing to fix", while a borrow-preflight block on an authorised
        arm is an operational problem. Reporting the last reason a candidate happened to
        emit conflates the two.
        """
        try:
            controller = _controller()
            targets = [strategy_id] if strategy_id else list(SHORT_STRATEGY_IDS)
            chain: list[dict[str, Any]] = []
            for target in targets:
                key = _resolve_key(target, market)
                record = controller.state_store.get(key)
                state = record.state if record else StrategyDeploymentState.SHADOW
                authorized, reasons = controller.may_submit_orders(key)
                chain.append(
                    {
                        **key.as_dict(),
                        "state": str(state),
                        "stage": (
                            "broker_execution"
                            if authorized
                            else "deployment_authorization"
                        ),
                        "ok": authorized,
                        "reason_codes": list(reasons),
                        "detail": (
                            ""
                            if authorized
                            else f"{target} is {state} and does not submit orders"
                        ),
                    }
                )
            blocker = next((item for item in chain if not item["ok"]), None)
            return _json(
                {
                    "ok": True,
                    "stages": list(ENTRY_BLOCKADE_STAGES),
                    "any_short_order_authorized": blocker is None and bool(chain),
                    "blocking_stage": blocker["stage"] if blocker else None,
                    "chain": chain,
                }
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc, chain=[])

    # ------------------------------------------------------------------ #
    # The one mutating endpoint                                           #
    # ------------------------------------------------------------------ #
    @router.post("/api/short-strategies/{strategy_id}/suspend")
    def suspend_short_strategy(
        strategy_id: str,
        market: str = Query("KR"),
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Manually SUSPEND an arm. The only manual action exposed.

        There is deliberately no promote endpoint. An operator can always make the
        system safer without argument; making it less safe has to earn it through the
        evidence ladder, and an HTTP call that skipped that would defeat the entire
        mechanism.

        Always writes an audit event tagged with the actor, because the spec requires
        that even a manual policy action be explicitly recorded — an undocumented
        manual state change is indistinguishable from a bug six months later.
        """
        try:
            controller = _controller()
            key = _resolve_key(strategy_id, market)
            actor = str(payload.get("actor") or "operator")
            reason = str(payload.get("reason") or "MANUAL_SUSPEND")
            before = controller.state_store.state_of(key)
            applied = controller.state_store.force_state(
                key,
                StrategyDeploymentState.SUSPENDED,
                actor=actor,
                reason=reason,
            )
            after = controller.state_store.state_of(key)
            return _json(
                {
                    "ok": bool(applied),
                    **key.as_dict(),
                    "from_state": str(before),
                    "to_state": str(after),
                    "submits_orders": after.submits_orders,
                    "actor": actor,
                    "reason": reason,
                    "recovery": (
                        "SUSPENDED recovers only to SHADOW, and only after the cause "
                        "clears and health checks pass. There is no direct path back "
                        "to a live state."
                    ),
                },
                status_code=200 if applied else 409,
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc, strategy_id=strategy_id)

    return router


def _borrow_source_status() -> dict[str, Any]:
    try:
        from app.trading.borrow_source import default_borrow_source

        return default_borrow_source().status()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _indicator_gap_status() -> dict[str, Any]:
    try:
        from app.features.short_indicators import (
            ShortIndicators,
            short_indicator_gaps,
        )

        # Reported for the CAPABILITY, not for one symbol: the question is "does this
        # deployment have a crowding source at all", which is answered the same way for
        # every name until a feed is added.
        return short_indicator_gaps(ShortIndicators())
    except Exception as exc:  # noqa: BLE001
        return {"unsourced": [], "detail": f"{type(exc).__name__}: {exc}"}


def _confidence_weights() -> dict[str, float]:
    from app.trading.short_strategy_promotion import CONFIDENCE_WEIGHTS

    return dict(CONFIDENCE_WEIGHTS)


def _threshold_dict(thresholds: Any) -> dict[str, Any]:
    if thresholds is None:
        return {}
    from dataclasses import asdict

    return asdict(thresholds)


def _next_state_name(state: StrategyDeploymentState) -> str | None:
    from app.trading.directional import next_promotion_state

    target = next_promotion_state(state)
    return str(target) if target else None
