"""Single-symbol, strategy-owned live trading session.

The session is deliberately narrower than the order engine.  Ontology/micro
reasoning (and an authorised GNN when available) elect one symbol and one
strategy.  The choice remains locked until the position is flat, preventing a
one-second trading loop from hopping between unrelated candidates.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from app.cost.cost_coverage import evaluate_cost_coverage
from app.routing.actions import is_actionable_strategy_route
from app.strategy.catalog import (
    STRATEGY_IDS,
    is_known_strategy,
    is_short_strategy,
    resolve_strategy_id,
)
from app.strategy.exit_geometry import FALLBACK_GEOMETRY_KEY
from app.strategy.exit_geometry import exit_bps as _strategy_exit_bps
from app.strategy.exit_geometry import exit_geometry as _exit_geometry
from app.strategy.exit_geometry import max_holding_seconds as _strategy_max_holding_seconds
from app.strategy.exit_geometry import resolve_exit_geometry as _resolve_exit_geometry
from app.trading.conservative_bandit import (
    BANDIT_EVIDENCE_WARMUP,
    BANDIT_MEASURED_EDGE_REJECTED,
    BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE,
    ArmCandidate,
    BanditContext,
    ConservativeStrategyBandit,
)
from app.trading.directional import (
    DirectionalStrategyKey,
    ExecutionProduct,
    PositionDirection,
    PositionEffect,
    DEPLOYMENT_SHADOW_ONLY,
    ShortReasonCodes,
    StrategyDeploymentState,
    default_product,
    favourable_watermark,
    parse_state,
    gross_return_bps as _directional_gross_bps,
    parse_direction,
    stop_breached,
    stop_price as _directional_stop_price,
    target_price as _directional_target_price,
    target_reached,
    trailing_breached,
    trailing_price as _directional_trailing_price,
)
from app.trading.strategy_performance_store import (
    EVALUATION_SOURCE_LIVE,
    EVALUATION_SOURCE_LIVE_PROBE,
    EVALUATION_SOURCE_SHADOW,
    StrategyPerformanceStore,
    default_store as _default_performance_store,
    market_for_symbol,
)


def _fallback_exit_geometry():
    """Geometry the executor applies to an unknown / not-yet-elected thesis."""
    return _exit_geometry(FALLBACK_GEOMETRY_KEY)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Marks a plan journaled for an arm that WAS order-authorised but lost the
# election. The signal is real and the measurement is real; the fill is simulated,
# which is why these land as ``evaluation_source=shadow`` and never as live
# evidence for the promotion ladder.
_COUNTERFACTUAL_REASON_CODE = "COUNTERFACTUAL_UNSELECTED_ARM"


def _cost_aware_profit_bps(
    model_evidence: Mapping[str, Any] | None,
    configured_profit_bps: float,
) -> float:
    """Never arm a target that cannot clear the modelled round-trip cost."""
    row = model_evidence if isinstance(model_evidence, Mapping) else {}
    try:
        expected_cost_bps = max(0.0, float(row.get("expected_cost_bps") or 0.0))
    except (TypeError, ValueError):
        expected_cost_bps = 0.0
    minimum_net_bps = max(
        0.0,
        _env_float("STRATEGY_SESSION_MIN_NET_TARGET_BPS", 25.0),
        expected_cost_bps,
    )
    return max(
        float(configured_profit_bps),
        expected_cost_bps + minimum_net_bps,
    )


def _measured_spread_bps(micro_result: Any, evidence_row: Mapping[str, Any] | None) -> float | None:
    """This symbol's top-of-book spread at decision time, if anything measured it.

    Returns ``None`` rather than a default. A stop is sized at three spreads, so
    substituting the KRX typical value for an unmeasured one would put the barrier
    inside the real spread on any wider name — the exact failure the geometry
    module was rewritten to eliminate.
    """
    diagnostics = getattr(micro_result, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        value = _optional_float(diagnostics.get("spread_bps"))
        if value is not None:
            return value
    value = _optional_float(getattr(micro_result, "spread_bps", None))
    if value is not None:
        return value
    if isinstance(evidence_row, Mapping):
        return _optional_float(evidence_row.get("spread_bps"))
    return None


def _market_round_trip_cost_bps(symbol: str, fallback_bps: float) -> float:
    """The venue fee policy's round trip for this symbol, in bps.

    Fees, tax, slippage and safety margin — everything the policy can state without
    knowing which symbol it is. It does NOT include the bid-ask spread, which is
    per-symbol and per-moment; :func:`_all_in_round_trip_cost_bps` adds that.

    Both venues resolve through the fee policy. Short-circuiting KR to the
    configured constant made every domestic proposal 28bps when the KRX policy says
    33.8, and the constant is retained only as a floor so a resolvable cost can
    never come out cheaper than the configured reference.
    """
    try:
        from app.technical.strategy_algorithms import round_trip_cost_bps

        measured = _optional_float(round_trip_cost_bps(symbol))
    except Exception:  # noqa: BLE001 - cost lookup must fail closed to config.
        measured = None
    if measured is None:
        return max(0.0, float(fallback_bps))
    # Floored, not replaced: the configured reference is an operator-set minimum and
    # a policy that resolves below it is a config gap, not a discount.
    return max(0.0, measured, float(fallback_bps))


def _all_in_round_trip_cost_bps(
    symbol: str,
    *,
    fallback_bps: float,
    model_estimate_bps: float | None = None,
    micro_result: Any = None,
    evidence_row: Mapping[str, Any] | None = None,
) -> float:
    """Everything a round trip actually costs, including the spread it crosses.

    Why the spread has to be in here
    --------------------------------
    ``config/profitability_policy.yaml`` sets ``spread_rate`` to 0 for both venues,
    because the spread is a property of the symbol and the moment rather than of the
    fee schedule. Nothing then put it back, so the cost-coverage gate divided the
    predicted edge by fees alone and called an edge smaller than one spread
    "SUFFICIENT".

    Measured on the 2026-08-21 KRX session, every arm the session took reported its
    coverage against 28bps. Against the real all-in number four of the five were
    below the 1.3 live threshold and three were below 1.0 — the cost was not even
    covered — and all four lost money::

        064260   gross 41bps / 52bps all-in = 0.79   traded, -73bps gross
        025980   gross 43bps / 52bps all-in = 0.83   traded, -73bps gross
        403870   gross 38bps / 44bps all-in = 0.86   ordered, unfilled
        010140   gross 61bps / 59bps all-in = 1.03   traded, exited flat at entry
        001510   gross 105bps / 51bps all-in = 2.08  the one that was worth taking

    A buy crosses the spread to get in and the sell crosses it to get out, which is
    one full spread over the round trip — charged once here, matching the
    "spread and impact are charged once each" contract in
    ``strategy_algorithms.round_trip_cost_bps``.

    A model-supplied estimate is FLOORED rather than replaced. Cost estimates are
    only dangerous when they are too low, and the trade-plan builder independently
    reaches the same conclusion through ``SPREAD_CONSUMES_ALPHA`` — this makes the
    election agree with the gate that already knew.
    """
    from app.cost.round_trip import all_in_round_trip_bps

    floor_bps = all_in_round_trip_bps(
        symbol,
        spread_bps=_measured_spread_bps(micro_result, evidence_row),
        fallback_bps=fallback_bps,
    )
    estimate = _optional_float(model_estimate_bps)
    return max(floor_bps, estimate if estimate is not None else 0.0)


def _cost_market_contract(symbol: str) -> tuple[str, str]:
    """Venue/product pair used by both the algorithm and the final cost gate."""

    if market_for_symbol(symbol) == "KR":
        return "KRX", "domestic_stock"
    return "NASD", "overseas_stock"


def _resolved_geometry(
    strategy_id: str,
    *,
    expected_cost_bps: float | None,
    micro_result: Any = None,
    evidence_row: Mapping[str, Any] | None = None,
):
    """Exit barriers sized against THIS trade's measured cost and spread.

    The table in :mod:`app.strategy.exit_geometry` is sized for a 28bps KRX round
    trip. Applying it unchanged to a venue that charges 63bps drops net reward:risk
    from the 1.5 it asserts to 0.83, which no win rate the strategies actually
    achieve can pay for. Passing the measurement in restores the invariant per
    venue; passing nothing returns the table exactly as before.
    """
    return _resolve_exit_geometry(
        strategy_id,
        round_trip_cost_bps=expected_cost_bps,
        spread_bps=_measured_spread_bps(micro_result, evidence_row),
    )


def _minimum_trailing_net_bps(strategy_id: str | None) -> float:
    continuation = {
        "intraday_momentum",
        "cross_sectional_relative_strength",
        "residual_relative_strength",
        "opening_range_breakout",
    }
    normalized = str(strategy_id or "")
    default = (
        30.0
        if normalized == "cross_sectional_relative_strength"
        else 15.0
        if normalized in continuation
        else 5.0
    )
    return max(
        0.0,
        _env_float("STRATEGY_SESSION_MIN_TRAILING_NET_BPS", default),
    )


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


_KST = timezone(timedelta(hours=9))


def _session_structure_context(now: datetime, symbol: str) -> dict[str, Any]:
    """Clock-derived session structure the session-boxed strategies need.

    Only the parts the clock can answer. ``in_last_continuous_half_hour`` and
    ``minutes_to_continuous_close`` are pure calendar facts, so withholding them
    would leave the strategies fail-closed for no reason; the price-derived fields
    (opening range, first-half-hour return) still require a producer and stay absent
    until one supplies them.

    The window is the symbol's OWN market. Reading the KRX close for every symbol
    put a US name's "last continuous half hour" at 14:50-15:20 Seoul — the middle
    of the New York night — so ``market_intraday_momentum`` and
    ``opening_range_breakout`` rejected every US tick with ``*_OUTSIDE_ENTRY_WINDOW``
    and could never fire on the market whose costs most need a once-a-day thesis.
    """
    from app.features.session_structure import regular_session

    session = regular_session(symbol)
    remaining = session.minutes_to_continuous_close(now)
    return {
        "in_last_continuous_half_hour": session.in_last_continuous_half_hour(now),
        # Negative after the continuous close, which reads as "no time left" to every
        # consumer rather than wrapping around to a large positive number.
        "minutes_to_continuous_close": round(remaining, 3),
    }


def _macro_permits(bundle: Any, strategy_id: str) -> bool | None:
    """Does the macro allow/block list still permit this strategy family?

    ``None`` means unanswerable (no bundle or no lists), which must not be read
    as a refusal.
    """
    macro = getattr(bundle, "macro_result", None)
    if macro is None:
        return None
    try:
        from app.technical.strategy_algorithms import macro_strategy_permitted

        return macro_strategy_permitted(
            strategy_id,
            tuple(getattr(macro, "allowed_micro_strategies", ()) or ()),
            tuple(getattr(macro, "blocked_micro_strategies", ()) or ()),
        )
    except Exception:  # noqa: BLE001 - a permission lookup must never crash election.
        return None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _market_group_for(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    return "KRX" if text.isdigit() and len(text) == 6 else "US"


def _new_entry_session_report(
    candidates: tuple[str, ...], now: datetime
) -> dict[str, Any]:
    """May any candidate's market accept a NEW entry right now?

    Answered per candidate market rather than globally: during the KR session the
    universe may be entirely US symbols (and vice versa), and "the market is open"
    is meaningless unless it is open for the symbols actually being scanned.
    """
    groups = tuple(dict.fromkeys(_market_group_for(symbol) for symbol in candidates))
    if not groups:
        return {
            "allows_new_entry": False,
            "reason": "NO_CANDIDATE_SYMBOLS",
            "phases": {},
            "open_groups": (),
            "eligible_symbols": (),
        }
    try:
        from app.data.market_session import allows_new_entry, market_phase
    except Exception:  # noqa: BLE001 - never let a session lookup break election.
        return {
            "allows_new_entry": True,
            "reason": "",
            "phases": {},
            "open_groups": groups,
            "eligible_symbols": tuple(candidates),
        }
    phases = {group: market_phase(group, now).value for group in groups}
    open_groups = tuple(group for group in groups if allows_new_entry(group, now))
    # 후보를 **자기 시장이 열린 것만** 남긴다.
    #
    # 이전에는 "그룹 중 하나라도 열려 있으면" 게이트를 통과시키고, 그 다음 두 제안
    # 경로가 후보 **전체** 를 순회했다. KR+US 혼합 유니버스에서 미국 정규장 시간이면
    # 마감된 국내 종목도 제안 대상이 됐다는 뜻이다. 세션 판정을 집합 단위로 하면서
    # 실행을 심볼 단위로 하는 불일치였다.
    eligible = tuple(
        symbol for symbol in candidates if _market_group_for(symbol) in set(open_groups)
    )
    if eligible:
        return {
            "allows_new_entry": True,
            "reason": "",
            "phases": phases,
            "open_groups": open_groups,
            "eligible_symbols": eligible,
        }
    detail = ",".join(f"{group}={phase}" for group, phase in sorted(phases.items()))
    return {
        "allows_new_entry": False,
        "reason": f"NEW_ENTRY_OUTSIDE_REGULAR_SESSION:{detail}",
        "phases": phases,
        "open_groups": (),
        "eligible_symbols": (),
    }


def _short_election_context(
    *,
    symbol: str,
    borrow_snapshot: Any,
    macro: Any,
    micro_diagnostics: Any,
    table: Any,
    now: datetime,
    orderbook: Any = None,
    average_daily_trading_value: Any = None,
    symbol_return: Any = None,
    market_return: Any = None,
) -> dict[str, Any]:
    """Point-in-time short facts, frozen at election.

    Every value comes from an observation that already existed at ``now``; nothing is
    re-derived later. Fields that cannot be resolved are OMITTED rather than defaulted,
    so the consuming algorithm fails closed — which for a borrow fact is the difference
    between a skipped trade and a rejected or force-closed position.
    """
    context: dict[str, Any] = {}

    # Borrow, straight off the frozen snapshot. ``short_sale_permitted`` is implied by
    # having a usable locate: ``_borrow_context`` only returns a snapshot when the
    # shared ``evaluate_borrow`` rule passed, and that rule already requires the broker
    # to have offered stock.
    if borrow_snapshot is not None:
        context["borrow_available"] = bool(getattr(borrow_snapshot, "available", False))
        context["short_sale_permitted"] = True
        quantity = getattr(borrow_snapshot, "available_quantity", None)
        if quantity is not None:
            context["borrow_available_quantity"] = int(quantity)
        fee = getattr(borrow_snapshot, "borrow_fee_bps_annualised", None)
        if fee is not None:
            context["borrow_fee_bps_annualised"] = float(fee)
        observed_at = getattr(borrow_snapshot, "observed_at", None)
        if observed_at is not None:
            context["borrow_observed_at"] = _iso(observed_at)
        deadline = getattr(borrow_snapshot, "return_deadline", None)
        if deadline is not None:
            context["return_deadline"] = _iso(deadline)

    # Market context. A directional short is fighting the tape when breadth is strong,
    # so the algorithms read it as an exclusion.
    if macro is not None:
        breadth = _optional_float(
            (getattr(macro, "diagnostics", None) or {}).get("market_breadth")
            if isinstance(getattr(macro, "diagnostics", None), Mapping)
            else None
        )
        if breadth is not None:
            context["market_breadth"] = breadth
        regime = getattr(getattr(macro, "market_regime", None), "value", None)
        if regime:
            context["market_trend"] = str(regime)

    # Weak-end sector rank. Derived from the SAME ranking the long side uses
    # (``size - rank + 1``) rather than a separately-built weak ranking, which could
    # disagree after a tie-break change and make a symbol simultaneously the strongest
    # and the weakest in its sector.
    if table is not None and symbol:
        try:
            ranked = table.weakness_rank_for(symbol)
        except Exception:  # noqa: BLE001 - a ranking lookup must never crash election.
            ranked = None
        if ranked is not None:
            context["sector_rank"] = int(ranked[0])
            context["sector_candidate_count"] = int(ranked[1])
        # The residual measurements themselves are shared with the long side — they are
        # the same market/sector-neutral quantity — but they are surfaced under the
        # SHORT field names so the weakness algorithm cannot accidentally consume the
        # strength thesis's field and inherit its sign convention.
        try:
            residual = table.residual_for(symbol)
            long_residual = table.long_residual_for(symbol)
        except Exception:  # noqa: BLE001
            residual = long_residual = None
        if residual is not None:
            context["residual_short_bps"] = float(residual) * 10_000.0
        if long_residual is not None:
            context["residual_long_bps"] = float(long_residual) * 10_000.0

    # Execution quality and crowding. The micro layer is consulted first; anything it
    # did not resolve is computed from the raw inputs by
    # ``app.features.short_indicators``.
    #
    # ``short_interest_ratio`` / ``days_to_cover`` have NO source in this repository —
    # KRX publishes 공매도 잔고 but nothing collects it, and the only in-repo
    # ``short_net_change`` comes from the synthetic demo pipeline. They are therefore
    # absent, the squeeze gates pass vacuously, and the fail-closed burden falls on the
    # borrow gates. That reduction in defence-in-depth is reported by
    # ``short_indicator_gaps`` rather than left to be inferred from a missing key.
    if isinstance(micro_diagnostics, Mapping):
        for key in (
            "liquidity_score",
            "spread_bps",
            "aggressor_imbalance",
            "short_interest_ratio",
            "days_to_cover",
            "breakdown_excess_bps",
            "market_alignment",
        ):
            value = micro_diagnostics.get(key)
            if value is not None:
                context[key] = value
    try:
        from app.features.short_indicators import compute_short_indicators

        computed = compute_short_indicators(
            orderbook=orderbook,
            average_daily_trading_value=average_daily_trading_value,
            symbol_return=symbol_return,
            market_return=market_return,
        ).as_context()
        # Micro-layer values win: they are closer to the decision and may incorporate
        # inputs this fallback cannot see.
        for key, value in computed.items():
            context.setdefault(key, value)
    except Exception:  # noqa: BLE001 - an indicator failure must not break election.
        pass
    return context


def _optional_float(value: Any) -> float | None:
    """``None``-preserving float. Unknown must never become 0.0."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN-safe


@dataclass
class _ElectionProposal:
    """One admissible (symbol, strategy) candidate awaiting a selection verdict.

    Extracted so both election paths produce the same shape and the conservative
    bandit can compare them side by side. Previously each path armed directly, so
    the two could never be weighed against each other or against NO_TRADE.
    """

    symbol: str
    strategy_id: str
    source: str
    entry_price: float | None
    target_return_rate: float
    stop_loss_rate: float
    trailing_stop_rate: float
    max_holding_seconds: int
    score: float
    confidence: float
    expected_net_return_bps: float | None
    expected_cost_bps: float | None
    gnn_actionable: bool
    gnn_action: str
    gnn_reason_codes: list[str]
    ontology_reason_codes: list[str]
    macro_regime: str
    micro_regime: str
    explanation_paths: list[dict[str, Any]]
    intent: Any
    candidate_count: int | None
    micro_result: Any
    evidence_row: Mapping[str, Any] | None
    last_reason: str
    conservative_edge_bps: float | None = None
    # --- Direction ---------------------------------------------------------- #
    # A proposal is now "open THIS direction", not implicitly "buy". ``effect`` is
    # always OPEN here (the session only elects entries; exits belong to the owned
    # position's own algorithm), but it is carried explicitly so the order contract
    # can be built without re-deriving it.
    direction: PositionDirection = PositionDirection.LONG
    position_effect: PositionEffect = PositionEffect.OPEN
    execution_product: ExecutionProduct = ExecutionProduct.CASH
    # Committed deployment state of this arm, read from the promotion controller at
    # proposal time. SHADOW means a shadow plan is journaled and NO entry intent is
    # produced.
    deployment_state: StrategyDeploymentState = StrategyDeploymentState.LIVE_FULL
    borrow_snapshot: Any = None
    borrow_reason_codes: tuple[str, ...] = ()
    # A deterministic proposal still receives a learned correction when one is
    # available, but absence of that correction is not absence of a signal.
    gnn_required_for_edge: bool = True
    # True only when the owned deterministic strategy algorithm fired in this
    # cycle. Model/ontology proposals remain observable, but algorithm-primary
    # live authority may never be inferred from their mere presence.
    algorithm_triggered: bool = False
    # The signal's own forward GROSS move, in bps, for a proposal that stated one but
    # no net figure. It is the last resort in ``predicted_gross_edge_bps``, and it
    # exists so that "no net estimate" cannot silently become "use the exit barrier".
    predicted_gross_move_bps: float | None = None

    @property
    def is_short(self) -> bool:
        return self.direction is PositionDirection.SHORT

    @property
    def submits_orders(self) -> bool:
        # Deployment authorisation is necessary but a live SHORT also needs a
        # point-in-time locate.  Keeping the proposal when the locate is absent is
        # intentional: SHADOW must record the unexecutable signal so borrow health
        # and signal quality can accumulate.  Only the broker-facing capability is
        # removed here.
        return self.deployment_state.submits_orders and (
            not self.is_short or self.borrow_snapshot is not None
        )

    def directional_key(self, market: str) -> DirectionalStrategyKey:
        return DirectionalStrategyKey(
            strategy_id=self.strategy_id,
            direction=self.direction,
            market=market,
            execution_product=self.execution_product,
        )

    def resolved_cost_bps(self, fallback_bps: float) -> float:
        cost = self.expected_cost_bps
        if cost is None or cost <= 0:
            return max(0.0, float(fallback_bps))
        return float(cost)

    def predicted_net_edge_bps(
        self, gnn_absence_penalty_bps: float, fallback_cost_bps: float
    ) -> float | None:
        """The caller's forward-looking net edge, docked when the GNN was absent.

        Returning ``None`` (rather than 0.0) when the edge cannot be formed at all
        is deliberate: the bandit then relies purely on realized history instead of
        blending in a fabricated zero.
        """
        edge = self.expected_net_return_bps
        if edge is None:
            gross = self.predicted_gross_edge_bps(fallback_cost_bps)
            if gross is None:
                return None
            edge = gross - self.resolved_cost_bps(fallback_cost_bps)
        if self.gnn_required_for_edge and not self.gnn_actionable:
            edge -= max(0.0, float(gnn_absence_penalty_bps))
        return edge

    def predicted_gross_edge_bps(self, fallback_cost_bps: float = 0.0) -> float | None:
        """How far the SIGNAL says price will move. Never a barrier.

        The last resort used to be ``target_return_rate * 10_000``, and that made
        every downstream cost test unfalsifiable. The target is defined by
        ``exit_geometry`` as ``cost + 1.5 x (stop + cost)``, so dividing it by the
        cost it was derived from asks whether 1.5 is greater than zero:

            ratio = (cost + 1.5(stop + cost)) / cost      # 5.71 at 28/60bps

        Raising the cost raised the numerator with it, which is why every arm on
        2026-08-21 reported THIN or SUFFICIENT no matter what the tape charged.

        A proposal with no forecast now returns ``None``, which
        ``evaluate_cost_coverage`` bands UNKNOWN and no gate treats as live
        eligible. Refusing to trade on an absent forecast is the whole point of
        asking for one.
        """
        if self.expected_net_return_bps is not None:
            return float(self.expected_net_return_bps) + self.resolved_cost_bps(
                fallback_cost_bps
            )
        return self.predicted_gross_move_bps


@dataclass(frozen=True)
class StrategySessionConfig:
    enabled: bool = field(
        default_factory=lambda: os.getenv("STRATEGY_SESSION_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    state_path: str = field(
        default_factory=lambda: os.getenv(
            "STRATEGY_SESSION_STATE_PATH", "data/store/strategy-session.json"
        )
    )
    # Floor applied to an elected proposal's target before the per-strategy
    # geometry is maxed in. Defaulted from the fallback geometry rather than a
    # literal so it cannot drift below the table it is meant to backstop.
    fallback_target_return_rate: float = field(
        default_factory=lambda: max(
            0.001,
            _env_float(
                "STRATEGY_SESSION_TARGET_RETURN_RATE",
                _fallback_exit_geometry().take_profit_bps / 10_000.0,
            ),
        )
    )
    cooldown_seconds: int = field(
        default_factory=lambda: max(5, _env_int("STRATEGY_SESSION_RESELECT_COOLDOWN_SEC", 30))
    )
    entry_timeout_seconds: int = field(
        default_factory=lambda: max(20, _env_int("STRATEGY_SESSION_ENTRY_TIMEOUT_SEC", 120))
    )
    armed_timeout_seconds: int = field(
        default_factory=lambda: max(
            30,
            _env_int("STRATEGY_SESSION_ARMED_TIMEOUT_SEC", 180),
        )
    )
    # A position is CLOSED only when something says so. The holdings inquiry is a
    # separate, lossy endpoint from the order endpoint: KIS intermittently returns a
    # balance that omits a lot it still holds. Believing one such response ended the
    # session mid-trade, wrote a phantom outcome, restarted the holding clock and
    # re-adopted the same lot under the unknown-thesis fallback geometry — the
    # 064260/010140 lifecycle on 2026-08-21, where one round trip produced six
    # "realized" rows and a 1200s time stop on a 5400s thesis. An unexplained
    # disappearance must therefore persist across several observations AND across
    # more wall clock than one account-cache refresh before it is believed.
    flat_confirm_observations: int = field(
        default_factory=lambda: max(
            1, _env_int("STRATEGY_SESSION_FLAT_CONFIRM_OBSERVATIONS", 3)
        )
    )
    flat_confirm_seconds: float = field(
        default_factory=lambda: max(
            0.0, _env_float("STRATEGY_SESSION_FLAT_CONFIRM_SEC", 90.0)
        )
    )
    # ``EXITING`` was the only phase with no way out. ``ENTERING`` has
    # ``entry_timeout_seconds``, ``ARMED`` has ``armed_timeout_seconds``, ``COOLDOWN``
    # has ``cooldown_until`` — but a session that reached EXITING depended entirely on
    # an external observation to leave it, and two reachable states supply none:
    #
    #   1. A broker-confirmed exit FILL together with a balance row that keeps
    #      reporting the lot. ``_reconcile_position`` then takes the "holding present"
    #      branch forever, ``exit_reason_for`` refuses to emit a second SELL because
    #      the fill is recorded, and no election can run because the phase is not
    #      SCANNING. Observed on DYN 2026-08-20: 8h27m of total paralysis across BOTH
    #      markets (``allowed_entry_candidates`` also returns nothing while any
    #      holding exists), ended only when the balance finally dropped the row.
    #   2. A supervisor HARD halt that moves ENTERING -> EXITING before the entry fill
    #      was ever observed. ``position_seen`` is False and ``exit_filled_at`` is
    #      unset, so the flat-reconciliation branch is skipped by its own guard and
    #      nothing else touches the phase.
    #
    # Both are released after this window. The release is safe in either world: if the
    # lot is genuinely still held, the very next SCANNING cycle re-adopts it through
    # ``_adopt_existing_position`` with ``owned_position_memo`` restoring the thesis,
    # so the exit rules re-arm instead of the position sitting unmanaged.
    exit_reconcile_timeout_seconds: float = field(
        default_factory=lambda: max(
            30.0, _env_float("STRATEGY_SESSION_EXIT_RECONCILE_TIMEOUT_SEC", 180.0)
        )
    )
    invalidation_confirm_cycles: int = field(
        default_factory=lambda: max(1, _env_int("STRATEGY_SESSION_INVALIDATION_CYCLES", 3))
    )
    # A market-wide regime change is only an exit input when it is both strong and
    # withdraws the strategy family that owns the position.  This keeps a routine
    # ontology rotation from liquidating a healthy holding while still allowing a
    # confirmed structural break to protect capital before the static stop is hit.
    invalidation_change_point_probability: float = field(
        default_factory=lambda: max(
            0.5,
            min(
                1.0,
                _env_float(
                    "STRATEGY_SESSION_INVALIDATION_CHANGE_POINT_PROBABILITY", 0.70
                ),
            ),
        )
    )
    require_live_gnn: bool = field(
        default_factory=lambda: os.getenv("STRATEGY_SESSION_REQUIRE_LIVE_GNN", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    # Ontology + deterministic strategies produce the complete candidate set;
    # learned estimates refine that set rather than determining whether it exists.
    algorithm_primary_election: bool = field(
        default_factory=lambda: os.getenv(
            "STRATEGY_SESSION_ALGORITHM_PRIMARY_ELECTION", "true"
        ).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    selection_evidence_max_age_seconds: int = field(
        default_factory=lambda: max(
            10,
            _env_int("STRATEGY_SESSION_EVIDENCE_MAX_AGE_SEC", 120),
        )
    )
    # Legacy conservative-bandit selector. The production default is OFF: the
    # owned deterministic algorithm already decides whether a setup fired and
    # supplies its point-in-time edge. Requiring a second historical-posterior
    # approval after that decision created a duplicate authority and cold-start
    # deadlock. It remains available only for explicit comparison/replay runs.
    bandit_enabled: bool = field(
        default_factory=lambda: os.getenv("STRATEGY_SESSION_BANDIT_ENABLED", "false").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # When the bandit is enabled, an unavailable / untrusted GNN downgrades a
    # candidate instead of vetoing election outright: the GNN is one estimator
    # among several, and treating its absence as a refusal made the whole session
    # dark whenever the checkpoint went stale (which adding a strategy does).
    gnn_absence_penalty_bps: float = field(
        default_factory=lambda: max(0.0, _env_float("STRATEGY_SESSION_GNN_ABSENCE_PENALTY_BPS", 15.0))
    )
    # Cold deterministic arms remain measurable in live execution, but never at
    # full size. This makes learning a bounded refinement instead of a deadlock.
    bandit_exploration_size_fraction: float = field(
        default_factory=lambda: max(
            0.0,
            min(
                1.0,
                _env_float("BANDIT_EXPLORATION_SIZE_FRACTION", 0.10),
            ),
        )
    )
    # Record realized outcomes so the bandit has something to learn from.
    record_outcomes: bool = field(
        default_factory=lambda: os.getenv("STRATEGY_SESSION_RECORD_OUTCOMES", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # Shadow observations must be independent enough to count as samples.  The
    # old loop wrote the same symbol/strategy every few seconds, creating many
    # overlapping plans that resolved on the same quote and falsely looked like
    # independent evidence.
    shadow_signal_cooldown_seconds: int = field(
        default_factory=lambda: max(
            60, _env_int("STRATEGY_SHADOW_SIGNAL_COOLDOWN_SECONDS", 300)
        )
    )
    # Round-trip cost assumed when the election evidence carried no estimate.
    # KRX round trip (sell tax + fees + spread) is ~25-30bps; 28 is the measured
    # per-strategy average in the R-GCN model card.
    fallback_round_trip_cost_bps: float = field(
        default_factory=lambda: max(
            0.0, _env_float("STRATEGY_SESSION_FALLBACK_COST_BPS", 28.0)
        )
    )
    # --- GNN-direct election (operator posture, 2026-08-08) ------------------ #
    # The GNN's own ranking becomes the selection, full stop: highest predicted
    # net edge is armed, with no pessimistic re-scoring and no NO_TRADE arm. Set
    # by an operator who holds that a model trained to pick the best strategy
    # should not then have its pick second-guessed by the layers below it.
    #
    # This overrides ``bandit_enabled``. What it gives up, stated plainly because
    # the flag cannot state it at runtime:
    #   * the pessimistic lower bound -- a cold arm is armed on its own optimism;
    #   * NO_TRADE as a selectable outcome -- if any proposal exists, one is armed;
    #   * the realized-history posterior and the BOCPD regime discount.
    #
    # Measured before this was switched on (2026-08-08, forward validation of GNN
    # elections on live ticks): 107 samples, positive_net_rate 0.0, mean realized
    # net -62.08bps, and the success head scored 61.8% on realized cells against
    # an 84.6% constant-predictor baseline. Those are the numbers this posture
    # accepts. Revert by unsetting the variable; no code path is deleted.
    gnn_direct_election: bool = field(
        default_factory=lambda: os.getenv(
            "STRATEGY_SESSION_GNN_DIRECT_ELECTION", "false"
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    )


@dataclass
class StrategySessionState:
    session_id: str | None = None
    phase: str = "SCANNING"
    selected_symbol: str | None = None
    selected_strategy: str | None = None
    # Which way the elected exposure points. Defaults to LONG so a session state file
    # written before shorts existed restores with its original meaning intact.
    selected_direction: str = "LONG"
    selected_execution_product: str = "CASH"
    selected_deployment_state: str = "LIVE_FULL"
    selection_source: str | None = None
    selection_score: float | None = None
    selection_confidence: float | None = None
    # Populated only when Selector V2 actually owns the election.  The context id
    # links the broker-realized outcome back to the exact shadow comparison group.
    selector_v2_context_id: str | None = None
    selector_v2_authority_state: str | None = None
    selector_v2_order_size_fraction: float = 1.0
    selected_at: str | None = None
    entry_submitted_at: str | None = None
    position_opened_at: str | None = None
    position_seen: bool = False
    # Last holdings snapshot that actually contained ``selected_symbol``, and how
    # many consecutive snapshots since then did not. Together they separate "the
    # broker's balance response dropped a lot it still holds" from "the position is
    # gone", which a single absent observation cannot distinguish.
    position_last_seen_at: str | None = None
    missing_holding_observations: int = 0
    # The live thesis this lot was entered under, carried across a session reset so
    # a re-adopted position is still managed with the geometry and the entry clock
    # it was armed with, instead of the unknown-thesis fallback.
    owned_position_memo: dict[str, Any] = field(default_factory=dict)
    entry_price: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    # Idle placeholders, shown on every dashboard while the session is SCANNING.
    # They were once hardcoded to 0.004 / 0.0022 / 0.0015 / 600s, which stopped
    # matching the exit-geometry table and then actively misled: an operator
    # reading the panel saw a 40bps target against a ~28bps round-trip cost and
    # concluded the target could not clear costs, when the real target binds from
    # ``exit_geometry`` at election time. A displayed number that no code path
    # ever uses is worse than no number, so these now derive from the same
    # fallback geometry the executor would actually apply to an unknown thesis.
    target_return_rate: float = field(
        default_factory=lambda: _fallback_exit_geometry().take_profit_bps / 10_000.0
    )
    target_profit_amount: float | None = None
    stop_loss_rate: float = field(
        default_factory=lambda: _fallback_exit_geometry().stop_loss_bps / 10_000.0
    )
    trailing_stop_rate: float = field(
        default_factory=lambda: _fallback_exit_geometry().trailing_bps / 10_000.0
    )
    high_watermark_price: float | None = None
    # Favourable extreme for a SHORT: the LOWEST price seen since entry. Tracked as a
    # separate field rather than overloading ``high_watermark_price`` so a persisted
    # long session restored after a restart cannot be reinterpreted as a short one,
    # and so a dashboard reading either field always knows which it has.
    low_watermark_price: float | None = None
    max_holding_seconds: int = field(
        default_factory=lambda: _fallback_exit_geometry().max_holding_seconds
    )
    exit_requested_at: str | None = None
    # Broker-confirmed terminal fill.  This is distinct from the requested time:
    # account balance endpoints may continue to show the sold lot for minutes.
    exit_filled_at: str | None = None
    # When the session ENTERED the EXITING phase, by any route: an exit rule, a
    # supervisor HARD halt, or a broker fill.  ``exit_requested_at`` cannot serve as
    # this clock because the halt path never submits an exit order and therefore never
    # sets it, which is exactly the case that used to strand the session with no
    # timestamp to measure a timeout against.
    exiting_since: str | None = None
    exit_reason: str | None = None
    cooldown_until: str | None = None
    invalidation_cycles: int = 0
    invalidation_reason_codes: list[str] = field(default_factory=list)
    # Macro/micro reasoning is throttled and the same bundle is returned to several
    # engine ticks.  Count each bundle once; otherwise one observation could satisfy
    # a three-cycle confirmation in three seconds without any new market evidence.
    last_invalidation_evidence_at: str | None = None
    last_evaluated_at: str | None = None
    last_reason: str = "WAITING_FOR_ONTOLOGY_SELECTION"
    macro_regime: str | None = None
    micro_regime: str | None = None
    # Catalogue size is not the same as the strategies reachable in this regime.
    # Publish both sets so "23 registered" cannot be mistaken for "23 evaluated".
    macro_permitted_strategy_ids: list[str] = field(default_factory=list)
    macro_filtered_strategy_ids: list[str] = field(default_factory=list)
    ontology_reason_codes: list[str] = field(default_factory=list)
    gnn_action: str | None = None
    gnn_reason_codes: list[str] = field(default_factory=list)
    explanation_paths: list[dict[str, Any]] = field(default_factory=list)
    candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    # Every actual strategy trigger checked this cycle, including rejected ones.
    # This is distinct from bandit_evaluations: the latter contains only pairs
    # that first passed their strategy's own mechanical entry algorithm.
    algorithm_evaluations: list[dict[str, Any]] = field(default_factory=list)
    # Slow context captured at election time and handed to the owning
    # algorithm. Fields the electing layer cannot supply stay absent, and the
    # algorithms that need them fail closed rather than assume a value.
    election_context: dict[str, Any] = field(default_factory=dict)
    halt_level: str = "NONE"
    halt_reason_codes: list[str] = field(default_factory=list)
    # --- Conservative-bandit election audit ------------------------------------
    # Why this arm and not NO_TRADE, in the numbers that decided it.
    bandit_selected_arm: str | None = None
    bandit_conservative_edge_bps: float | None = None
    # True when the arm was chosen to LEARN rather than because it has a
    # demonstrated edge. Without this an operator reading a NEGATIVE
    # conservative_edge_bps on an ARMED position has no way to tell a deliberate
    # minimum-size probe from a selection bug.
    bandit_is_exploration: bool = False
    bandit_reason_codes: list[str] = field(default_factory=list)
    bandit_evaluations: list[dict[str, Any]] = field(default_factory=list)
    bandit_shadow_arms: list[str] = field(default_factory=list)
    #: 이 bandit 진단값이 **언제** 계산됐는지. ``None`` 이면 이번 사이클에 bandit 이
    #: 돌지 않았다는 뜻이다.
    #:
    #: 이 필드가 없어서 실제로 오진이 발생했다. proposals 가 비면 ``_bandit_choice`` 가
    #: 호출되지 않는데 ``bandit_*`` 필드는 초기화되지 않아, 국내장 시간에 계산된
    #: "rvgi_box_breakout / 069500 / conservative_edge -109bps" 가 미국 정규장 진단으로
    #: 그대로 노출됐다. 운영자(그리고 사람)는 "미국 정규장인데 왜 마감된 국내 종목을
    #: 평가하나"라는 존재하지 않는 결함을 추적하게 된다.
    bandit_evaluated_at: str | None = None
    cost_coverage_ratio: float | None = None
    cost_coverage_band: str | None = None
    change_point_probability: float | None = None
    # Per-candidate-market session phase at the last evaluation. Makes
    # "nothing traded because no scanned market was in its regular session"
    # readable without cross-referencing clocks.
    session_phases: dict[str, str] = field(default_factory=dict)
    # --- Realized-outcome bookkeeping ------------------------------------------
    expected_cost_bps: float | None = None
    expected_net_return_bps: float | None = None
    exit_price: float | None = None
    outcome_recorded: bool = False
    # --- Short / borrow bookkeeping ------------------------------------------ #
    # ``loan_date`` is what a buy-to-cover needs to identify WHICH borrow lot it
    # repays. Its absence on an owned short is a fail-closed condition, not a
    # cosmetic gap, so it lives in persisted session state and survives a restart.
    loan_date: str | None = None
    borrow_fee_bps_annualised: float | None = None
    borrow_reference: str | None = None
    return_deadline: str | None = None
    # Shadow arms observed this cycle: arms that produced a valid signal but were not
    # order-authorised. Recorded so the dashboard can show "the short fired, and here
    # is why it did not trade" rather than showing nothing at all.
    shadow_plan_ids: list[str] = field(default_factory=list)
    directional_comparison: dict[str, Any] = field(default_factory=dict)
    # --- TradePlan link -------------------------------------------------------- #
    # The plan itself lives in ``trade_plan`` (durable, immutable); the session keeps
    # only the pointer, so a restart can reload the plan that owns the position rather
    # than reconstructing it from these display fields.
    trade_plan_id: str | None = None
    trade_plan_quantity: int | None = None
    trade_plan_expires_at: str | None = None


def _selector_v2_snapshot(runner: Any, symbol: str | None) -> dict[str, Any]:
    """Read the V2 runner's telemetry without letting it break ``snapshot()``."""
    try:
        return dict(runner.snapshot(symbol=symbol))
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "error": f"{type(exc).__name__}"}


def _build_selector_v2_runner() -> Any | None:
    """Construct the V2 observer/authority runner when enabled; ``None`` otherwise.

    Fails to ``None`` on any error, leaving legacy authority in place. Runtime promotion
    failures are handled by the controller and cannot grant authority without durable state.
    """
    try:
        from app.config.selector_v2_flags import SelectorV2Flags

        flags = SelectorV2Flags.from_env()
        if not flags.enabled:
            return None
        from app.routing.selector_v2_shadow import SelectorV2ShadowRunner

        return SelectorV2ShadowRunner(flags=flags)
    except Exception:  # noqa: BLE001
        return None


class StrategySessionManager:
    """Persistent closed-world ownership state machine for the live engine."""

    def __init__(
        self,
        *,
        config: StrategySessionConfig | None = None,
        selection_evidence_provider: Callable[[tuple[str, ...]], Mapping[str, Any]] | None = None,
        performance_store: StrategyPerformanceStore | None = None,
        bandit: ConservativeStrategyBandit | None = None,
        selector_v2_runner: Any | None = None,
        plan_builder: Any | None = None,
    ) -> None:
        self.config = config or StrategySessionConfig()
        self.selection_evidence_provider = selection_evidence_provider
        self.performance_store = (
            performance_store if performance_store is not None else _default_performance_store()
        )
        self.bandit = bandit or ConservativeStrategyBandit(store=self.performance_store)
        # Algorithms are immutable policy objects for the lifetime of a live
        # session. Building the registry per proposal reread and reparsed YAML
        # hundreds of times in one election cycle, delaying the first cycle by
        # minutes. Environment/config is already frozen when this manager is
        # constructed, so resolve it once and reuse it consistently.
        from app.technical.strategy_algorithms import build_algorithm_registry

        self._algorithm_registry = build_algorithm_registry()
        self._lock = threading.RLock()
        self._state = self._load()
        # Shadow plans journaled this cycle, awaiting adoption by
        # ``ShadowEvaluationService``. Drained rather than accumulated so a cycle whose
        # plans nobody collects cannot grow without bound.
        self._pending_shadow_plans: list[Any] = []
        # The frozen TradePlan for the current election. Built once in ``_arm`` and read
        # by every downstream stage; ``None`` while SCANNING or when the plan could not
        # be built (see ``_build_trade_plan``).
        self._trade_plan: Any | None = None
        # Live wiring supplies the same account-calibrated RiskRules used by the
        # decision engine. Falling back to a fresh default RiskManager here used
        # a hard-coded KRX 1bn liquidity threshold for US symbols and silently
        # disagreed with the preceding risk decision.
        self._plan_builder: Any | None = plan_builder
        # StrategySelectorV2, starting in SHADOW next to the live election. ``None`` unless
        # ``STRATEGY_SELECTOR_V2_ENABLED`` is set. Construct once rather than per cycle;
        # invalid configuration leaves the runner off and legacy authority intact.
        self._selector_v2: Any | None = selector_v2_runner
        if self._selector_v2 is None:
            self._selector_v2 = _build_selector_v2_runner()

    def evaluate(
        self,
        account: Any,
        candidates: tuple[str, ...],
        macro_micro_bundle: Any,
        now: datetime,
    ) -> dict[str, Any]:
        now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        with self._lock:
            self._state.last_evaluated_at = _iso(now)
            if not self.config.enabled:
                self._state.last_reason = "STRATEGY_SESSION_DISABLED"
                return self.snapshot()

            holdings = {
                str(getattr(item, "ticker", "") or "").upper(): item
                for item in tuple(getattr(account, "holdings", ()) or ())
                if int(getattr(item, "quantity", 0) or 0) > 0
            }
            self._reconcile_position(holdings, macro_micro_bundle, now)

            if self._state.phase == "COOLDOWN":
                until = _parse_time(self._state.cooldown_until)
                if until is not None and now >= until:
                    self._reset_to_scanning("RESELECTION_COOLDOWN_COMPLETE")

            if self._state.phase == "ENTERING" and not holdings:
                submitted = _parse_time(self._state.entry_submitted_at)
                if submitted is not None and (now - submitted).total_seconds() >= self.config.entry_timeout_seconds:
                    self._reset_to_scanning("ENTRY_NOT_FILLED_TIMEOUT")

            if self._state.phase == "ARMED" and not holdings:
                selected_at = _parse_time(self._state.selected_at)
                if (
                    selected_at is not None
                    and (now - selected_at).total_seconds()
                    >= self.config.armed_timeout_seconds
                ):
                    self._state.phase = "COOLDOWN"
                    self._state.cooldown_until = _iso(
                        now + timedelta(seconds=self.config.cooldown_seconds)
                    )
                    self._state.last_reason = "STRATEGY_ENTRY_WINDOW_EXPIRED"

            if self._state.phase == "SCANNING":
                if holdings:
                    self._adopt_existing_position(holdings, macro_micro_bundle, now)
                else:
                    self._select(candidates, macro_micro_bundle, now, account=account)

            self._persist()
            return self.snapshot()

    def allowed_entry_candidates(
        self, candidates: tuple[str, ...], account: Any
    ) -> tuple[str, ...]:
        """Symbols the session authorises a NEW ENTRY on, in either direction.

        Renamed from ``allowed_buy_candidates`` because "buy" is no longer synonymous
        with "enter": for a short thesis the entry order is a SELL. The old name is
        retained as an alias so existing callers keep working with identical
        behaviour.

        A SHADOW-state election never reaches here — it produces no ARMED phase — so
        an unvalidated short cannot appear in this list.
        """
        with self._lock:
            if not self.config.enabled:
                return candidates
            if tuple(getattr(account, "holdings", ()) or ()):
                return ()
            if self._state.phase != "ARMED" or not self._state.selected_symbol:
                return ()
            # Defence in depth. The election path already refuses to ARM a
            # non-order-authorised arm; asserting it again here means a future change
            # to that path cannot quietly make a SHADOW short enterable.
            if not parse_state(self._state.selected_deployment_state).submits_orders:
                return ()
            # Ownership is locked for the complete ARMED entry window.  A
            # discovery list changing on the next cycle must not silently drop
            # the elected symbol before its strategy can evaluate the trigger.
            return (self._state.selected_symbol,)

    # Backward-compatible alias: entries were BUY-only before short support.
    allowed_buy_candidates = allowed_entry_candidates

    def selected_direction_for(self, symbol: str) -> str | None:
        """Direction of the elected (or owned) position on ``symbol``."""
        with self._lock:
            if self._state.selected_symbol != str(symbol or "").upper():
                return None
            return self._state.selected_direction

    def entry_order_size_fraction(self, symbol: str) -> float:
        """Maximum entry size granted by the selector authority rung."""
        with self._lock:
            if self._state.selected_symbol != str(symbol or "").upper():
                return 0.0
            selector_cap = max(
                0.0,
                min(1.0, float(self._state.selector_v2_order_size_fraction or 0.0)),
            )
            if self._state.bandit_is_exploration:
                selector_cap = min(
                    selector_cap, self.config.bandit_exploration_size_fraction
                )
            # Automatically promoted long theses enter through a size-limited
            # probe.  The selector may impose an even smaller cap; neither layer
            # can enlarge the other layer's grant.
            try:
                from app.trading.long_strategy_promotion import deployment_size_cap

                deployment_cap = deployment_size_cap(
                    parse_state(self._state.selected_deployment_state)
                )
            except Exception:  # noqa: BLE001 - order sizing fails closed.
                return 0.0
            return min(selector_cap, deployment_cap)

    def order_contract_for(self, symbol: str, *, closing: bool = False) -> dict[str, Any] | None:
        """The (direction, effect, product, loan_date) contract for an order.

        The single place the execution layer should ask "what kind of order is this",
        so a caller cannot infer it from ``side`` and get the SELL ambiguity wrong.
        Returns ``None`` when this session does not own the symbol.
        """
        with self._lock:
            state = self._state
            if state.selected_symbol != str(symbol or "").upper():
                return None
            direction = parse_direction(state.selected_direction)
            effect = PositionEffect.CLOSE if closing else PositionEffect.OPEN
            return {
                "position_direction": str(direction),
                "position_effect": str(effect),
                "execution_product": state.selected_execution_product,
                "credit_type": "05" if direction is PositionDirection.SHORT else None,
                # Required on a SHORT CLOSE. Deliberately passed through as-is
                # (possibly None) so the execution layer's own check fails closed
                # rather than this method inventing a plausible date.
                "loan_date": state.loan_date if effect is PositionEffect.CLOSE else None,
                "deployment_state": state.selected_deployment_state,
            }

    def record_broker_loan_date(self, symbol: str, loan_date: str | None) -> None:
        """Store the broker-authoritative 대출일 for an owned short.

        Called after a short entry fills and after each reconciliation. Without it the
        buy-to-cover has no lot to repay, which the promotion controller treats as an
        immediate suspension condition rather than something to work around.
        """
        with self._lock:
            if self._state.selected_symbol != str(symbol or "").upper():
                return
            self._state.loan_date = str(loan_date).strip() if loan_date else None
            self._persist()

    def exit_reason_for(self, holding: Any) -> str | None:
        with self._lock:
            symbol = str(getattr(holding, "ticker", "") or "").upper()
            if self._state.phase != "EXITING" or symbol != self._state.selected_symbol:
                return None
            # The order-status endpoint is more authoritative about this order than
            # a lagging balance row.  Once the exit filled, never emit another SELL.
            if self._state.exit_filled_at:
                return None
            return self._state.exit_reason or "STRATEGY_THESIS_INVALIDATED"

    def owns_position(self, symbol: str) -> bool:
        with self._lock:
            return bool(
                self._state.selected_symbol == str(symbol or "").upper()
                and self._state.phase in {"ARMED", "ENTERING", "OWNED", "EXITING"}
            )

    def selected_strategy_for(self, symbol: str) -> str | None:
        with self._lock:
            if (
                self._state.phase != "ARMED"
                or self._state.selected_symbol != str(symbol or "").upper()
            ):
                return None
            return self._state.selected_strategy

    def trade_plan_for(self, symbol: str) -> Any | None:
        """The frozen plan that owns this symbol, or ``None``.

        This is the object every downstream stage reads instead of re-deriving cost,
        size or risk. ``None`` means no plan was built for this election, and the caller
        falls back to the legacy path — which still runs its own gates, so the absence is
        safe rather than permissive.
        """
        with self._lock:
            plan = self._trade_plan
            if plan is None:
                return None
            if str(getattr(plan, "symbol", "")).upper() != str(symbol or "").upper():
                return None
            return plan

    def note_plan_entry_fill(self, symbol: str, price: float, quantity: int) -> None:
        """Record the realised entry on the plan so its exit levels bind to the fill."""
        with self._lock:
            plan = self._trade_plan
            if plan is None or str(plan.symbol).upper() != str(symbol or "").upper():
                return
            try:
                self._trade_plan = plan.with_entry_fill(price, quantity)
            except Exception:  # noqa: BLE001 - a bad fill report must not lose the plan.
                return
            self._save_trade_plan(self._trade_plan)

    def _save_trade_plan(self, plan: Any) -> None:
        try:
            from app.trading.trade_plan import default_trade_plan_store

            default_trade_plan_store().save(plan)
        except Exception:  # noqa: BLE001 - persistence failure must not stop trading.
            return

    def _build_trade_plan(
        self, proposal: "_ElectionProposal", now: datetime, *, account: Any
    ) -> Any | None:
        """Run cost, sizing and risk ONCE, here, and freeze the result into a plan.

        Everything this computes used to be computed after election by
        ``SharedLiveDecisionEngine.evaluate_buy``, where each stage could veto or resize
        what had already been chosen. Same arithmetic, moved in front of the choice.

        Returns ``None`` when the plan cannot be built — a thin edge, a risk rejection, a
        missing account. The election still stands (the session owns the symbol) but no
        plan-driven fast path is available, so the legacy gated path handles it.
        """
        if account is None or proposal.entry_price is None or proposal.entry_price <= 0:
            return None
        try:
            from app.trading.trade_plan_builder import PlanRequest, TradePlanBuilder

            market = _market_group_for(proposal.symbol)
            snapshot = self._plan_market_snapshot(proposal, market, now)
            if snapshot is None:
                return None
            context = self._election_context(
                proposal.strategy_id,
                now,
                intent=proposal.intent,
                candidate_count=proposal.candidate_count,
                micro_result=proposal.micro_result,
                evidence_row=proposal.evidence_row,
                symbol=proposal.symbol,
            )
            builder = getattr(self, "_plan_builder", None)
            if builder is None:
                builder = TradePlanBuilder()
                self._plan_builder = builder
            venue, instrument_type = _cost_market_contract(proposal.symbol)
            outcome = builder.build(
                PlanRequest(
                    symbol=proposal.symbol,
                    strategy_id=proposal.strategy_id,
                    market=market,
                    account=account,
                    market_snapshot=snapshot,
                    reference_price=float(proposal.entry_price),
                    take_profit_rate=float(proposal.target_return_rate),
                    stop_loss_rate=float(proposal.stop_loss_rate),
                    trailing_rate=float(proposal.trailing_stop_rate) or None,
                    max_holding_seconds=int(proposal.max_holding_seconds),
                    gross_edge_bps=float(
                        proposal.predicted_gross_edge_bps(
                            self.config.fallback_round_trip_cost_bps
                        )
                        or 0.0
                    ),
                    direction=str(getattr(proposal.direction, "value", proposal.direction)),
                    confidence=float(proposal.confidence or 0.0) or None,
                    liquidity_score=self._plan_liquidity_score(proposal),
                    spread_bps=_measured_spread_bps(
                        proposal.micro_result, proposal.evidence_row
                    ),
                    authority_size_fraction=self._plan_authority_fraction(proposal),
                    entry_trigger=str(proposal.strategy_id),
                    strategy_exit_trigger=str(proposal.strategy_id),
                    weekday_time_context=self._weekday_time_context(proposal.symbol, now),
                    election_context=context,
                    order_contract={
                        "direction": str(
                            getattr(proposal.direction, "value", proposal.direction)
                        ),
                        "position_effect": "OPEN",
                        "execution_product": str(
                            getattr(
                                proposal.execution_product,
                                "value",
                                proposal.execution_product,
                            )
                        ),
                    },
                    venue=venue,
                    instrument_type=instrument_type,
                    source_ids=self._plan_source_ids(proposal, now),
                    session_id=self._state.session_id,
                    plan_ttl_seconds=self._plan_ttl_seconds(),
                ),
                now=now,
            )
        except Exception:  # noqa: BLE001 - a plan-build failure falls back, never crashes.
            return None
        if outcome.plan is None:
            self._state.last_reason = (
                f"PLAN_NOT_BUILT:{','.join((outcome.no_trade.reason_codes if outcome.no_trade else ()) or ('UNKNOWN',))}"
            )
            return None
        self._save_trade_plan(outcome.plan)
        return outcome.plan

    def _plan_ttl_seconds(self) -> float:
        """How long an elected plan stays executable.

        The configured value, floored by the session's own ARMED window: a plan that
        outlived the entry window it was elected for would authorise an order the session
        has already given up on.
        """
        try:
            from app.config.execution_authority import (
                default_execution_authority_config,
            )

            configured = float(
                default_execution_authority_config().trade_plan_ttl_seconds
            )
        except Exception:  # noqa: BLE001 - fall back to the session window.
            configured = float(self.config.armed_timeout_seconds)
        return max(30.0, min(configured, float(self.config.armed_timeout_seconds)))

    def _weekday_time_context(self, symbol: str, now: datetime) -> dict[str, Any]:
        """The weekday / session-phase context, taken from the temporal layer.

        This is the join between the calendar refactor and the election: the plan carries
        the same phase, day and seasonality bucket the context hierarchy resolved, so a
        stored plan can be replayed against the kind of time it was made in.
        """
        context: dict[str, Any] = dict(_session_structure_context(now, symbol))
        try:
            from app.context.temporal_context import build_temporal_snapshot

            snapshot = build_temporal_snapshot(_market_group_for(symbol), now)
            context.update(
                {
                    "market_group": snapshot.market_group,
                    "trading_day": str(snapshot.trading_day),
                    "day_of_week": snapshot.day_of_week_name,
                    "session_phase": snapshot.session_phase.value,
                    "session_progress": snapshot.session_progress,
                    "minutes_from_open": snapshot.minutes_from_open,
                    "minutes_to_close": snapshot.minutes_to_close,
                    "holiday_adjacent": snapshot.holiday_adjacent,
                    "month_end": snapshot.month_end,
                    "quarter_end": snapshot.quarter_end,
                    "expiry_context": snapshot.expiry_context.value,
                }
            )
        except Exception:  # noqa: BLE001 - the clock context is additive, never required.
            context.setdefault("temporal_context_unavailable", True)
        return context

    def _plan_market_snapshot(
        self, proposal: "_ElectionProposal", market: str, now: datetime
    ) -> Any | None:
        """A MarketSnapshot for the risk manager, from what the election measured."""
        try:
            from app.schemas.domain import MarketSnapshot, SourceMetadata
        except Exception:  # noqa: BLE001
            return None
        row = proposal.evidence_row if isinstance(proposal.evidence_row, Mapping) else {}
        return MarketSnapshot(
            ticker=str(proposal.symbol).upper(),
            market=market,
            company_name=str(row.get("company_name") or ""),
            sector=str(row.get("sector") or ""),
            last_price=float(proposal.entry_price or 0.0),
            average_daily_trading_value=float(
                row.get("average_daily_trading_value") or 0.0
            ),
            volatility_20d=float(row.get("volatility_20d") or 0.0),
            source=SourceMetadata(
                source_name="kis_realtime",
                retrieved_at=now,
                source_type="broker_api",
                trust_level=5,
                is_realtime=True,
                observed_at=now,
            ),
        )

    @staticmethod
    def _plan_liquidity_score(proposal: "_ElectionProposal") -> float:
        row = proposal.evidence_row if isinstance(proposal.evidence_row, Mapping) else {}
        value = row.get("liquidity_score")
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, score))

    def _plan_authority_fraction(self, proposal: "_ElectionProposal") -> float:
        """The deployment/selector size cap, applied at election rather than downstream."""
        try:
            from app.trading.long_strategy_promotion import deployment_size_cap

            cap = deployment_size_cap(parse_state(str(proposal.deployment_state)))
        except Exception:  # noqa: BLE001 - an unreadable cap sizes to zero, not to full.
            return 0.0
        selector_cap = max(
            0.0, min(1.0, float(self._state.selector_v2_order_size_fraction or 1.0))
        )
        if self._state.bandit_is_exploration:
            selector_cap = min(
                selector_cap, self.config.bandit_exploration_size_fraction
            )
        return max(0.0, min(1.0, min(cap, selector_cap)))

    @staticmethod
    def _plan_source_ids(proposal: "_ElectionProposal", now: datetime) -> tuple[str, ...]:
        row = proposal.evidence_row if isinstance(proposal.evidence_row, Mapping) else {}
        ids = tuple(
            str(item)
            for item in (row.get("source_record_ids") or ())
            if str(item or "").strip()
        )
        return ids or (
            f"election:{proposal.symbol}:{proposal.strategy_id}:{now.strftime('%Y%m%d%H%M%S')}",
        )

    def election_context_for(self, symbol: str) -> dict[str, Any] | None:
        """Slow context the ontology resolved when it elected this strategy."""
        with self._lock:
            if self._state.selected_symbol != str(symbol or "").upper():
                return None
            if not self._state.selected_strategy:
                return None
            return dict(self._state.election_context)

    def request_halt(self, symbol: str, level: str, reason_codes: tuple[str, ...]) -> bool:
        """Apply a supervisor verdict. Returns True when an exit was initiated.

        ``HARD`` hands the open position to the exit path immediately. ``SOFT``
        only records the violation and releases an ARMED (not yet filled)
        election; a position already owned keeps being managed by its own
        algorithm.
        """
        normalized = str(symbol or "").upper()
        graded = str(level or "NONE").upper()
        with self._lock:
            if self._state.selected_symbol != normalized:
                return False
            self._state.halt_level = graded
            self._state.halt_reason_codes = list(reason_codes)
            if graded == "HARD":
                if self._state.phase in {"OWNED", "ENTERING"}:
                    self._state.phase = "EXITING"
                    # No clock is passed to a supervisor verdict, and inventing one
                    # here would disagree with the engine's decision time. The
                    # reconciler stamps ``exiting_since`` on its first pass instead.
                    self._state.exit_reason = f"SUPERVISOR_HARD_HALT:{','.join(reason_codes) or 'UNSPECIFIED'}"
                    self._state.last_reason = self._state.exit_reason
                    self._persist()
                    return True
                if self._state.phase == "ARMED":
                    self._reset_to_scanning("SUPERVISOR_HARD_HALT_BEFORE_ENTRY")
                    self._persist()
                return False
            if graded == "SOFT" and self._state.phase == "ARMED":
                # Not filled yet, so releasing the election costs nothing.
                self._reset_to_scanning(
                    f"SUPERVISOR_SOFT_HALT:{','.join(reason_codes) or 'UNSPECIFIED'}"
                )
            self._persist()
            return False

    def mark_entry_submitted(self, symbol: str, now: datetime) -> None:
        with self._lock:
            if self._state.phase == "ARMED" and self._state.selected_symbol == str(symbol).upper():
                self._state.phase = "ENTERING"
                self._state.entry_submitted_at = _iso(now)
                self._state.last_reason = "ENTRY_ORDER_SUBMITTED"
                self._persist()

    def mark_exit_submitted(self, symbol: str, now: datetime) -> None:
        with self._lock:
            if self._state.selected_symbol == str(symbol).upper():
                self._state.phase = "EXITING"
                self._state.exiting_since = self._state.exiting_since or _iso(now)
                self._state.exit_requested_at = self._state.exit_requested_at or _iso(now)
                self._state.last_reason = "EXIT_ORDER_SUBMITTED_AWAITING_FLAT"
                self._persist()

    def mark_exit_filled(self, symbol: str, price: float, now: datetime) -> None:
        """Adopt the broker's terminal exit fill before balance reconciliation.

        This both freezes the actual execution price for learning and closes the
        duplicate-order window created when the holdings endpoint lags the order
        endpoint after a completed SELL.
        """
        normalized = str(symbol or "").upper()
        moment = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        with self._lock:
            if (
                self._state.selected_symbol != normalized
                or self._state.phase not in {"OWNED", "EXITING"}
            ):
                return
            if float(price or 0.0) > 0.0:
                self._state.exit_price = float(price)
            self._state.phase = "EXITING"
            self._state.exiting_since = self._state.exiting_since or _iso(moment)
            self._state.exit_filled_at = _iso(moment)
            self._state.last_reason = "EXIT_FILLED_AWAITING_ACCOUNT_FLAT"
            # Record now, while the true fill timestamp and price are available.
            # A later flat balance reconciliation is idempotent via outcome_recorded.
            self._record_outcome(moment)
            self._persist()

    def drain_shadow_plans(self) -> tuple[Any, ...]:
        """Take the shadow plans journaled since the last drain.

        Returns the plan OBJECTS, with their frozen borrow snapshots intact. Handing
        back ids instead would force the consumer to re-read the journal and
        re-resolve the locate, which is precisely where a fresher borrow observation
        could contaminate a point-in-time evaluation.
        """
        with self._lock:
            plans = tuple(self._pending_shadow_plans)
            self._pending_shadow_plans = []
            return plans

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._state)
            payload["enabled"] = self.config.enabled
            payload["single_position_enforced"] = self.config.enabled
            payload["require_live_gnn"] = self.config.require_live_gnn
            payload["bandit_enabled"] = self.config.bandit_enabled
            # Publish the posture, because ``bandit_enabled`` alone now lies. The
            # direct-election path overrides the bandit without switching that
            # flag off, so a reader of this snapshot would see bandit_enabled
            # true and conclude a pessimistic bound and a NO_TRADE option were in
            # play when neither is.
            payload["gnn_direct_election"] = self.config.gnn_direct_election
            # The plan is the real authority downstream, so it belongs in the snapshot
            # every dashboard reads. Without it the UI would still be describing the
            # pre-refactor path in which the gates decided after election.
            plan = self._trade_plan
            payload["trade_plan"] = plan.as_dict() if plan is not None else None
            payload["execution_authority"] = (
                "TRADE_PLAN" if plan is not None else "LEGACY_GATED_PATH"
            )
            payload["selection_authority"] = (
                "GNN_DIRECT" if self.config.gnn_direct_election
                else "DETERMINISTIC_ALGORITHM"
                if self.config.algorithm_primary_election
                else "CONSERVATIVE_BANDIT"
                if self.config.bandit_enabled
                else "FORWARD_EDGE_RANKING"
            )
            evaluated_pairs = {
                (
                    str(item.get("symbol") or ""),
                    str(item.get("strategy_id") or ""),
                )
                for item in self._state.algorithm_evaluations
                if isinstance(item, Mapping)
            }
            payload["strategy_catalogue_count"] = len(STRATEGY_IDS)
            payload["macro_permitted_strategy_count"] = len(
                self._state.macro_permitted_strategy_ids
            )
            payload["macro_filtered_strategy_count"] = len(
                self._state.macro_filtered_strategy_ids
            )
            payload["evaluated_strategy_ids"] = sorted(
                {strategy_id for _, strategy_id in evaluated_pairs if strategy_id}
            )
            payload["evaluated_strategy_count"] = len(payload["evaluated_strategy_ids"])
            payload["evaluated_symbol_count"] = len(
                {symbol for symbol, _ in evaluated_pairs if symbol}
            )
            runner = self._selector_v2
            # V2 telemetry includes configured posture and effective earned authority.
            # It is published even when disabled so the dashboard can distinguish OFF
            # from an unknown state.
            payload["selector_v2"] = (
                _selector_v2_snapshot(runner, self._state.selected_symbol)
                if runner is not None
                else {"enabled": False, "reason": "STRATEGY_SELECTOR_V2_DISABLED"}
            )
            return payload

    def _reconcile_position(
        self, holdings: Mapping[str, Any], bundle: Any, now: datetime
    ) -> None:
        state = self._state
        selected = state.selected_symbol
        holding = holdings.get(selected or "")
        if holding is not None:
            if state.phase == "EXITING" and self._exiting_window_expired(now):
                # The broker confirmed a terminal exit fill (or an exit was requested
                # and no rule can fire again) yet the balance keeps returning the lot.
                # The two endpoints contradict each other, and the previous behaviour
                # trusted BOTH in the direction that blocks: the fill suppressed any
                # further SELL, the balance row suppressed the flat transition, and the
                # session stopped electing on either market until the row happened to
                # disappear. Release it instead. If the lot really is still held the
                # next SCANNING cycle re-adopts it from this same balance response and
                # re-arms its exit rules, so neither reading is left unmanaged.
                self._release_stranded_exit(now, "EXIT_UNRECONCILED_BALANCE_ROW_STALE")
                return
            state.missing_holding_observations = 0
            state.position_last_seen_at = _iso(now)
            if state.phase in {"ARMED", "ENTERING"}:
                state.phase = "OWNED"
                state.position_seen = True
                state.position_opened_at = _iso(
                    getattr(holding, "opened_at", None)
                    or getattr(holding, "captured_at", None)
                    or now
                )
                state.entry_price = float(getattr(holding, "average_price", 0.0) or 0.0)
                # Adopt the BROKER's direction and loan date, not our own belief. The
                # broker is authoritative on what the position actually is, and a
                # disagreement here is precisely the condition the promotion
                # controller suspends on rather than trades through.
                broker_direction = str(getattr(holding, "direction", "") or "").upper()
                if broker_direction in {"LONG", "SHORT"}:
                    state.selected_direction = broker_direction
                broker_loan_date = getattr(holding, "loan_date", None)
                if broker_loan_date:
                    state.loan_date = str(broker_loan_date)
                direction = parse_direction(state.selected_direction)
                if state.entry_price:
                    self._apply_owned_exit_geometry(state.entry_price)
                    last_price = float(getattr(holding, "last_price", 0.0) or 0.0)
                    if direction is PositionDirection.LONG:
                        state.high_watermark_price = max(state.entry_price, last_price)
                    else:
                        # Favourable extreme for a short is the LOW. Seeding this from
                        # a high watermark would arm a trailing stop that is already
                        # triggered.
                        state.low_watermark_price = favourable_watermark(
                            state.entry_price, last_price or state.entry_price, direction
                        )
                state.last_reason = "POSITION_OWNED_MONITORING"
            self._evaluate_exit(holding, bundle, now)
            # After the exit evaluation, so the memo carries the watermark this tick
            # just advanced rather than the previous one.
            self._remember_owned_position()
            return

        if state.phase == "EXITING" and not (
            state.position_seen or state.exit_filled_at is not None
        ):
            # EXITING with no evidence that exposure ever existed. Reachable when a
            # supervisor HARD halt promotes ENTERING -> EXITING before the entry fill
            # is observed: the guard below excludes this state, no holding exists for
            # ``exit_reason_for`` to act on, and ``entry_timeout_seconds`` no longer
            # applies because the phase is no longer ENTERING. Nothing in the machine
            # could leave it. There is nothing to close here, so release the election.
            if self._exiting_window_expired(now):
                self._reset_to_scanning("EXIT_RELEASED_NO_CONFIRMED_EXPOSURE")
                return
            state.last_reason = "EXIT_AWAITING_EXPOSURE_CONFIRMATION"
            return

        if (
            state.phase in {"OWNED", "EXITING"}
            and (state.position_seen or state.exit_filled_at is not None)
        ):
            if not self._flat_confirmed(now):
                # The lot is missing from ONE balance response and nothing we did
                # explains it. Hold the whole session — phase, geometry, entry clock
                # and watermark — until the disappearance is corroborated. Closing
                # here is what produced phantom outcomes and downgraded a live thesis
                # to the fallback exit rule mid-position.
                state.last_reason = "POSITION_MISSING_FROM_SNAPSHOT_UNCONFIRMED"
                return
            # The position went flat, so this is the one moment the trade's
            # realized outcome is knowable. Record it before the state is reset:
            # without this the conservative bandit has no history to learn from and
            # every arm stays permanently cold.  A confirmed exit fill is also
            # authoritative evidence that exposure existed: fast overseas fills
            # can complete before the slower holdings snapshot ever observes the
            # position, leaving ``position_seen`` false indefinitely.
            if not state.exit_reason:
                # Nothing this session requested closed it. Say so in the record
                # rather than storing an outcome with a blank reason, which is
                # indistinguishable from a phantom row after the fact.
                state.exit_reason = "POSITION_CLOSED_EXTERNALLY"
            self._record_outcome(now)
            # The lot is gone for real, so the thesis frozen for it must not survive
            # to be inherited by the next position that happens to share its symbol
            # and average price.
            state.owned_position_memo = {}
            state.phase = "COOLDOWN"
            state.cooldown_until = _iso(now + timedelta(seconds=self.config.cooldown_seconds))
            state.exiting_since = None
            state.last_reason = "POSITION_FLAT_RESELECTION_COOLDOWN"
            state.exit_requested_at = state.exit_requested_at or _iso(now)

    def _exiting_window_expired(self, now: datetime) -> bool:
        """Has this session been EXITING longer than reconciliation can justify?

        Measured from ``exiting_since``, which every route into the phase stamps. A
        state file written before that field existed has none; treat the older
        ``exit_filled_at``/``exit_requested_at`` as the clock so a session restored
        mid-exit across the upgrade is still releasable rather than stranded forever.
        """
        state = self._state
        started = (
            _parse_time(state.exiting_since)
            or _parse_time(state.exit_filled_at)
            or _parse_time(state.exit_requested_at)
        )
        if started is None:
            # No timestamp at all. Stamp one now rather than releasing on a clock we
            # cannot read: the next pass measures a real window.
            state.exiting_since = _iso(now)
            return False
        return (
            now - started
        ).total_seconds() >= self.config.exit_reconcile_timeout_seconds

    def _release_stranded_exit(self, now: datetime, reason: str) -> None:
        """Leave an EXITING phase that no observation can resolve.

        Deliberately does NOT record an outcome. The fill that put the session here
        already recorded one (``mark_exit_filled`` -> ``_record_outcome``), and a
        second row for the same lot is the phantom-outcome defect that once turned one
        round trip into six "realized" records. Cooldown rather than an immediate
        reset, so a balance row that is merely lagging has one more window to clear
        before the lot can be re-adopted.
        """
        state = self._state
        state.exit_reason = state.exit_reason or reason
        state.owned_position_memo = dict(state.owned_position_memo or {})
        state.phase = "COOLDOWN"
        state.cooldown_until = _iso(now + timedelta(seconds=self.config.cooldown_seconds))
        state.exiting_since = None
        # The fill claim is what suppressed any further SELL. Retiring it with the
        # phase means a lot that turns out to still be held can be exited again.
        state.exit_filled_at = None
        state.last_reason = reason

    def _flat_confirmed(self, now: datetime) -> bool:
        """Is "the symbol is not in the holdings map" actually a closed position?

        Two sources answer this, and only one of them is the balance inquiry.

        A broker-confirmed exit FILL is authoritative and immediate. That is what
        keeps a fast round trip from sitting in EXITING while the slower holdings
        endpoint catches up.

        A *requested* exit is not. Deciding to sell, or even having an order accepted,
        says nothing about whether the lot is still held: 025980's stop-loss sell sat
        unfilled above the market for ten minutes while the balance response kept
        dropping and restoring the position, and treating "we asked to exit" as proof
        turned each drop into another phantom close of a lot we still owned.

        So absent a fill the balance inquiry is the only witness, and it is a lossy
        one — KIS returns a partial portfolio often enough that a single absent
        observation carries no information. It has to repeat, both across observations
        and across more wall clock than one account-cache refresh, before it is
        believed.
        """
        state = self._state
        if state.exit_filled_at is not None:
            return True
        state.missing_holding_observations += 1
        if state.missing_holding_observations < self.config.flat_confirm_observations:
            return False
        last_seen = _parse_time(state.position_last_seen_at)
        if last_seen is None:
            # Never observed in the balance at all. There is no "still held" claim to
            # protect, so the observation count alone decides.
            return True
        return (now - last_seen).total_seconds() >= self.config.flat_confirm_seconds

    def _remember_owned_position(self) -> None:
        """Freeze the live thesis for this lot so a reset cannot downgrade it.

        ``_adopt_existing_position`` rebuilds the session from the broker's balance
        row, which carries a quantity and an average price and nothing else. Without
        this memo the strategy id, the exit geometry it was armed with and — worst —
        the true entry time are all lost, and the lot is re-managed as an unknown
        thesis on a fresh 1200s clock.
        """
        state = self._state
        strategy = resolve_strategy_id(state.selected_strategy)
        if not strategy or not state.selected_symbol or not state.entry_price:
            return
        state.owned_position_memo = {
            "symbol": state.selected_symbol,
            "strategy_id": strategy,
            "entry_price": float(state.entry_price),
            "position_opened_at": state.position_opened_at,
            "selected_direction": state.selected_direction,
            "selected_execution_product": state.selected_execution_product,
            "selected_deployment_state": state.selected_deployment_state,
            "selection_source": state.selection_source,
            "session_id": state.session_id,
            "target_price": state.target_price,
            "stop_price": state.stop_price,
            "target_return_rate": state.target_return_rate,
            "stop_loss_rate": state.stop_loss_rate,
            "trailing_stop_rate": state.trailing_stop_rate,
            "max_holding_seconds": state.max_holding_seconds,
            "high_watermark_price": state.high_watermark_price,
            "low_watermark_price": state.low_watermark_price,
            "expected_cost_bps": state.expected_cost_bps,
            "expected_net_return_bps": state.expected_net_return_bps,
        }

    def _recall_imminent(self, now: datetime) -> bool:
        """Is the borrow's return deadline close enough to force a cover?

        Absent deadline == not imminent. A borrow with no stated deadline is the
        normal case (open-ended loan), and treating "no deadline" as "due now" would
        close every short immediately.
        """
        deadline = _parse_time(self._state.return_deadline)
        if deadline is None:
            return False
        return (deadline - now).total_seconds() <= _env_float(
            "SHORT_RECALL_EXIT_LEAD_SECONDS", 1800.0
        )

    def _record_outcome(self, now: datetime) -> None:
        """Persist the realized net outcome of the closed position.

        Uses the last observed mark as the exit reference and subtracts the same
        round-trip cost estimate the election used, so ``realized_net_bps`` is
        directly comparable with the ``expected_net_bps`` that armed the trade —
        which is what makes the stored prediction error meaningful.

        The gross return is direction-signed exactly once, in
        :func:`app.trading.directional.gross_return_bps`. A short that covered lower
        records a POSITIVE net.
        """
        state = self._state
        if not self.config.record_outcomes or state.outcome_recorded:
            return
        strategy = state.selected_strategy
        symbol = state.selected_symbol
        entry = state.entry_price
        direction = parse_direction(state.selected_direction)
        # Fall back to the FAVOURABLE extreme for this direction, so a short with no
        # frozen exit mark is not reconstructed from a high watermark that, for a
        # short, is its worst price.
        fallback_mark = (
            state.high_watermark_price
            if direction is PositionDirection.LONG
            else state.low_watermark_price
        )
        exit_price = state.exit_price or fallback_mark
        if not strategy or not symbol or not entry or entry <= 0 or not exit_price:
            return
        gross_bps = _directional_gross_bps(float(entry), float(exit_price), direction)
        cost_bps = (
            state.expected_cost_bps
            if state.expected_cost_bps is not None
            else self.config.fallback_round_trip_cost_bps
        )
        opened = _parse_time(state.position_opened_at)
        closed = _parse_time(state.exit_filled_at) or now
        holding_seconds = (
            max(0.0, (closed - opened).total_seconds()) if opened is not None else None
        )
        # Borrow accrues over the holding period and is NOT in ``expected_cost_bps``
        # (which is the round trip). Deducted here so the recorded net is the number
        # the promotion gates should actually judge.
        borrow_bps = 0.0
        if direction is PositionDirection.SHORT and holding_seconds is not None:
            from app.trading.borrow import borrow_cost_bps

            borrow_bps = borrow_cost_bps(
                state.borrow_fee_bps_annualised, holding_seconds
            ) or 0.0
        state_enum = StrategyDeploymentState(
            state.selected_deployment_state
            if state.selected_deployment_state in set(StrategyDeploymentState)
            else "LIVE_FULL"
        )
        recorded = self.performance_store.record(
            strategy_id=strategy,
            symbol=symbol,
            market=market_for_symbol(symbol),
            regime=state.macro_regime or "UNKNOWN",
            realized_net_bps=gross_bps - float(cost_bps) - borrow_bps,
            realized_gross_bps=gross_bps,
            expected_net_bps=state.expected_net_return_bps,
            holding_seconds=holding_seconds,
            exit_reason=state.exit_reason or "",
            recorded_at=closed,
            direction=str(direction),
            execution_product=state.selected_execution_product,
            deployment_state=str(state_enum),
            # A real fill under LIVE_PROBE is distinguished from a full-size live
            # fill, because the promotion gates weight them differently.
            evaluation_source=(
                EVALUATION_SOURCE_LIVE_PROBE
                if state_enum is StrategyDeploymentState.LIVE_PROBE
                else EVALUATION_SOURCE_LIVE
            ),
            borrow_available=(
                True if direction is PositionDirection.SHORT else None
            ),
            borrow_fee_bps=state.borrow_fee_bps_annualised,
            signal_executable=True,
        )
        state.outcome_recorded = bool(recorded)
        if recorded and state.selector_v2_context_id and self._selector_v2 is not None:
            try:
                self._selector_v2.record_live_outcome(
                    context_id=state.selector_v2_context_id,
                    strategy_id=strategy,
                    net_return_bps=gross_bps - float(cost_bps) - borrow_bps,
                    evidence_source=(
                        EVALUATION_SOURCE_LIVE_PROBE
                        if state.selector_v2_authority_state == "LIVE_PROBE"
                        else EVALUATION_SOURCE_LIVE
                    ),
                )
            except Exception:  # noqa: BLE001 - evidence linkage cannot break exits.
                pass

    def _continuation_invalidation_evidence(
        self, bundle: Any, symbol: str, strategy_id: str | None
    ) -> tuple[str | None, tuple[str, ...]]:
        """Return one fresh, forward-looking reason set for an owned position.

        Held symbols are deliberately included in every macro/micro reasoning run.
        The micro reasoner can therefore detect momentum loss, VWAP breakdown,
        volatility expansion, false breakouts and liquidity deterioration before a
        frozen price stop is reached.  This method converts that existing advisory
        output into strategy-lifecycle evidence; it does not invent a second signal.

        The bundle timestamp is returned as the observation identity so a throttled
        bundle can only advance confirmation once.
        """
        if bundle is None:
            return None, ()
        normalized = str(symbol or "").upper()
        bundle_at = _parse_time(getattr(bundle, "timestamp", None))
        evidence_id = _iso(bundle_at) if bundle_at is not None else None
        reasons: list[str] = []

        micro = next(
            (
                item
                for item in tuple(getattr(bundle, "micro_results", ()) or ())
                if str(getattr(item, "symbol", "") or "").upper() == normalized
            ),
            None,
        )
        if micro is not None:
            raw_exit = getattr(micro, "exit_signal", None)
            exit_signal = str(getattr(raw_exit, "value", raw_exit) or "").upper()
            if exit_signal in {
                "SELL_CANDIDATE",
                "RISK_REDUCE",
                "TAKE_PROFIT",
                "TRAILING_STOP",
            }:
                reasons.append(f"MICRO_{exit_signal}")
                micro_at = _parse_time(getattr(micro, "timestamp", None))
                if micro_at is not None:
                    evidence_id = _iso(micro_at)
                reasons.extend(
                    str(code)
                    for code in tuple(getattr(micro, "reason_codes", ()) or ())
                    if str(code)
                )

        # A high-probability structural break plus explicit withdrawal of the owning
        # strategy is independent confirmation that its expected edge has expired.
        macro = getattr(bundle, "macro_result", None)
        change_probability = _optional_float(
            getattr(macro, "change_point_probability", None)
        )
        if (
            strategy_id
            and _macro_permits(bundle, strategy_id) is False
            and change_probability is not None
            and change_probability
            >= self.config.invalidation_change_point_probability
        ):
            # When no micro exit exists, identify the observation by the macro
            # timestamp. A cached structural-break verdict must not be counted as
            # several independent confirmations by the faster micro loop.
            if not reasons:
                macro_at = _parse_time(getattr(macro, "timestamp", None))
                if macro_at is not None:
                    evidence_id = _iso(macro_at)
            reasons.extend(
                (
                    f"MACRO_WITHDREW_STRATEGY:{strategy_id}",
                    f"REGIME_CHANGE_PROBABILITY:{change_probability:.3f}",
                )
            )

        return evidence_id, tuple(dict.fromkeys(reasons))

    def _confirmed_continuation_exit_reason(
        self,
        holding: Any,
        bundle: Any,
        *,
        direction: PositionDirection,
    ) -> str | None:
        """Confirm thesis decay and classify the exit as profit protection/loss limit."""
        state = self._state
        evidence_id, reasons = self._continuation_invalidation_evidence(
            bundle, state.selected_symbol or "", state.selected_strategy
        )
        # No timestamp means the producer cannot prove that this is new evidence.
        # Fail closed for an automated early exit rather than counting loop ticks.
        if evidence_id is None:
            return None
        if evidence_id == state.last_invalidation_evidence_at:
            return None
        state.last_invalidation_evidence_at = evidence_id
        if reasons:
            state.invalidation_cycles += 1
            state.invalidation_reason_codes = list(reasons)
        else:
            # Continuation recovered on a genuinely newer observation. Requiring
            # consecutive evidence filters one-off indicator noise.
            state.invalidation_cycles = 0
            state.invalidation_reason_codes = []
            return None
        if state.invalidation_cycles < self.config.invalidation_confirm_cycles:
            return None

        entry = float(getattr(holding, "average_price", 0.0) or 0.0)
        mark = float(getattr(holding, "last_price", 0.0) or 0.0)
        if entry <= 0.0 or mark <= 0.0:
            return None
        gross_bps = _directional_gross_bps(entry, mark, direction)
        cost_bps = max(
            0.0,
            float(
                state.expected_cost_bps
                if state.expected_cost_bps is not None
                else self.config.fallback_round_trip_cost_bps
            ),
        )
        return (
            "STRATEGY_EDGE_DECAY_PROFIT_PROTECT"
            if gross_bps > cost_bps
            else "STRATEGY_EDGE_DECAY_LOSS_LIMIT"
        )

    def _evaluate_exit(self, holding: Any, bundle: Any, now: datetime) -> None:
        state = self._state
        if state.phase == "EXITING":
            return
        symbol = state.selected_symbol or ""
        direction = parse_direction(state.selected_direction)
        last_price = float(getattr(holding, "last_price", 0.0) or 0.0)
        average_price = float(getattr(holding, "average_price", 0.0) or 0.0)
        quantity = max(0, int(getattr(holding, "quantity", 0) or 0))
        # Direction-signed: a short's PnL is positive when the price has FALLEN below
        # the entry, so the unsigned (last - average) would report every winning short
        # as a loss and drive the wrong exit decisions downstream.
        pnl = direction.sign * quantity * (last_price - average_price)
        state.target_profit_amount = (
            quantity
            * max(
                0.0,
                direction.sign * (float(state.target_price or 0.0) - average_price),
            )
            if average_price > 0 and state.target_price
            else None
        )

        # Track the FAVOURABLE extreme for this direction: the high for a long, the
        # low for a short. Both fields are maintained so a restored session and the
        # dashboard always read the one that matches the position.
        if direction is PositionDirection.LONG:
            state.high_watermark_price = max(
                float(state.high_watermark_price or 0.0), last_price, average_price
            )
            watermark = state.high_watermark_price
        else:
            state.low_watermark_price = favourable_watermark(
                state.low_watermark_price
                if state.low_watermark_price
                else average_price or None,
                last_price,
                direction,
            )
            watermark = state.low_watermark_price

        resolved_stop = (
            float(state.stop_price)
            if state.stop_price
            else _directional_stop_price(average_price, state.stop_loss_rate, direction)
            if average_price > 0
            else 0.0
        )
        resolved_trailing = (
            _directional_trailing_price(watermark, state.trailing_stop_rate, direction)
            if watermark
            else 0.0
        )
        trailing_locked_gross_bps = _directional_gross_bps(
            average_price, resolved_trailing, direction
        )
        trailing_required_gross_bps = max(
            0.0,
            float(
                state.expected_cost_bps
                if state.expected_cost_bps is not None
                else self.config.fallback_round_trip_cost_bps
            ),
        ) + _minimum_trailing_net_bps(state.selected_strategy)

        reason: str | None = None
        if target_reached(last_price, state.target_price, direction):
            reason = "STRATEGY_PROFIT_TARGET"
        elif stop_breached(last_price, resolved_stop, direction):
            reason = "STRATEGY_STOP_LOSS"
        elif (
            trailing_locked_gross_bps >= trailing_required_gross_bps
            and trailing_breached(
                last_price, resolved_trailing, average_price, direction
            )
        ):
            reason = "STRATEGY_TRAILING_STOP"
        # A borrow recall is an exit reason with no long-side analogue: the position
        # must be covered whether or not the price thesis still holds, and waiting for
        # a price barrier would hand the timing to the lender.
        elif direction is PositionDirection.SHORT and self._recall_imminent(now):
            reason = "STRATEGY_SHORT_BORROW_RECALL"

        if reason is None:
            reason = self._confirmed_continuation_exit_reason(
                holding,
                bundle,
                direction=direction,
            )

        opened = _parse_time(state.position_opened_at) or getattr(holding, "opened_at", None)
        if (
            reason is None
            and isinstance(opened, datetime)
            and (now - (opened if opened.tzinfo else opened.replace(tzinfo=timezone.utc))).total_seconds()
            >= state.max_holding_seconds
        ):
            reason = "STRATEGY_MAX_HOLDING_TIME"

        if reason:
            state.phase = "EXITING"
            state.exiting_since = state.exiting_since or _iso(now)
            state.exit_reason = reason
            state.exit_requested_at = _iso(now)
            state.last_reason = reason
            # Freeze the exit reference now: once the broker reports flat the mark
            # is gone, and a realized outcome reconstructed from a stale watermark
            # would flatter every stop-out.
            if last_price > 0:
                state.exit_price = last_price
        elif pnl or state.position_seen:
            state.last_reason = "POSITION_OWNED_STRATEGY_MONITORING"
            if last_price > 0:
                state.exit_price = last_price

    def _select(
        self,
        candidates: tuple[str, ...],
        bundle: Any,
        now: datetime,
        *,
        account: Any = None,
    ) -> None:
        """Elect one (symbol, strategy) — or deliberately elect nothing.

        Two-phase by design:

        1. Collect every *admissible* proposal from both election paths (ranked
           macro/micro BUY intents, then fresh shadow-evidence rows). Admissibility
           is the same closed-world question it always was: known strategy, macro
           permits the family, deployment authorised, evidence fresh.
        2. Score all proposals together on a pessimistic net-edge lower bound and
           arm the winner, or arm nothing.

        Phase 2 is the change. Previously phase 1 armed the FIRST admissible
        proposal, which has no way to express "all of these are admissible and all
        of them lose money". With the conservative bandit, ``no_trade`` is a real
        arm and wins by default, so a tape with no positive-expectancy strategy
        produces no position instead of the least-bad negative one.
        """
        # 매 선택 사이클마다 bandit 진단값을 먼저 비운다.
        #
        # 이 초기화가 없으면 proposals 가 빈 사이클에서 ``_bandit_choice`` 가 호출되지
        # 않고, 몇 시간 전 다른 시장 세션에서 계산된 arm 평가가 현재 상태로 보고된다.
        # 실제로 그것 때문에 "미국 정규장에 마감된 국내 종목 arm 을 평가한다"는 오진이
        # 나왔다. 진단값은 계산된 사이클 안에서만 유효해야 한다.
        self._reset_bandit_diagnostics()

        if bundle is None:
            self._state.macro_permitted_strategy_ids = []
            self._state.macro_filtered_strategy_ids = []
            self._state.last_reason = "WAITING_FOR_MACRO_MICRO_BUNDLE"
            return
        macro = getattr(bundle, "macro_result", None)
        raw_macro_regime = getattr(macro, "market_regime", None)
        macro_regime = getattr(raw_macro_regime, "value", raw_macro_regime)
        self._state.macro_regime = str(macro_regime or "") or None
        permissions = {
            strategy_id: _macro_permits(bundle, strategy_id)
            for strategy_id in STRATEGY_IDS
        }
        self._state.macro_permitted_strategy_ids = [
            strategy_id
            for strategy_id, permitted in permissions.items()
            if permitted is not False
        ]
        self._state.macro_filtered_strategy_ids = [
            strategy_id
            for strategy_id, permitted in permissions.items()
            if permitted is False
        ]
        self._state.ontology_reason_codes = list(
            getattr(macro, "reason_codes", ()) or ()
        )
        self._state.explanation_paths = list(
            getattr(macro, "explanation_paths", ()) or ()
        )[:12]
        self._state.candidate_diagnostics = [
            {
                "symbol": str(getattr(result, "symbol", "") or "").upper(),
                "micro_regime": str(
                    getattr(getattr(result, "micro_regime", None), "value", "")
                    or ""
                ),
                "selected_strategy": str(
                    getattr(getattr(result, "selected_strategy", None), "value", "")
                    or ""
                ),
                "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
                "execution_quality": str(
                    getattr(getattr(result, "execution_quality", None), "value", "")
                    or ""
                ),
                "reason_codes": list(getattr(result, "reason_codes", ()) or ()),
                "diagnostics": dict(getattr(result, "diagnostics", {}) or {}),
            }
            for result in tuple(getattr(bundle, "micro_results", ()) or ())[:8]
        ]
        change_point_probability = _optional_float(
            getattr(macro, "change_point_probability", None)
        )
        self._state.change_point_probability = change_point_probability
        evidence: Mapping[str, Any] = {}
        if self.selection_evidence_provider is not None:
            try:
                evidence = self.selection_evidence_provider(candidates) or {}
            except Exception:
                evidence = {}

        intents = [
            item
            for item in tuple(getattr(bundle, "ranked_trade_intents", ()) or ())
            if str(getattr(item, "side", "") or "").upper() == "BUY"
            and str(getattr(item, "symbol", "") or "").upper() in set(candidates)
        ]

        # Session phase FIRST. Diagnosing "why has nothing traded" used to require
        # correlating timestamps by hand, because outside the regular session every
        # candidate failed individually on thin-book liquidity and the surviving
        # reason blamed the GNN. State the actual constraint instead.
        session_report = _new_entry_session_report(candidates, now)
        self._state.session_phases = dict(session_report.get("phases") or {})
        if not session_report["allows_new_entry"]:
            self._state.last_reason = session_report["reason"]
            return

        # 자기 시장이 열린 후보만 제안 대상이다. 세션 판정을 집합 단위로 하고 실행을
        # 심볼 단위로 하면, 미국 정규장 시간에 마감된 국내 종목이 제안될 수 있다.
        tradable_candidates = tuple(session_report.get("eligible_symbols") or candidates)
        tradable_set = set(tradable_candidates)
        intents = [
            item
            for item in intents
            if str(getattr(item, "symbol", "") or "").upper() in tradable_set
        ]

        proposals: list[_ElectionProposal] = []
        if self.config.algorithm_primary_election:
            proposals.extend(
                self._registry_algorithm_proposals(
                    tradable_candidates, evidence, bundle, now
                )
            )
        proposals.extend(self._intent_proposals(intents, evidence, bundle, now))
        proposals.extend(
            self._evidence_proposals(tradable_candidates, evidence, bundle, now)
        )
        proposals = self._deduplicate_joint_proposals(proposals)

        # Journal every SHADOW-state proposal before selection runs. This is how a
        # short accumulates the forward evidence it needs: the signal fired, the
        # borrow world at that instant is frozen into a plan, and the plan will be
        # scored from data that has not arrived yet. No order is involved.
        self._journal_shadow_proposals(proposals, now)

        # Only order-authorised proposals are selectable. Filtering here rather than
        # inside the bandit keeps SHADOW arms visible in the evaluation list (so the
        # dashboard can show what they WOULD have done) while making them structurally
        # unable to win.
        executable = [proposal for proposal in proposals if proposal.submits_orders]
        selected: "_ElectionProposal | None" = None
        decided = False
        if executable and self.config.gnn_direct_election:
            gnn_selected = self._gnn_direct_choice(executable)
            selected = gnn_selected
            decided = True
        elif (
            self.config.algorithm_primary_election
            and self._state.algorithm_evaluations
        ):
            # One authority only: a deterministic strategy must have fired, and
            # candidates are ranked on that same algorithm's forward net edge.
            # Historical bandit evidence and model availability cannot veto it.
            selected = self._algorithm_choice(executable, now)
            decided = True
        elif proposals and self.config.bandit_enabled:
            selected = self._bandit_choice(proposals, macro, now)
            decided = True
        elif executable:
            selected = self._forward_edge_choice(executable, now)
            decided = True

        # ...and journal the order-authorised arms that did NOT win. Without this the
        # evidence channel is wired backwards: a SHADOW arm accumulates a posterior
        # every cycle while a LIVE_FULL arm only ever produces a sample by winning an
        # election, and it can only win on a posterior it has no way to build. The
        # result was an absorbing state — after the first losing outcome the arm is
        # neither explorable (loss streak) nor exploitable (negative posterior), and
        # no further sample can arrive to change either. The whole funded KR side sat
        # in it: 1,650 shadow plans on record, every one of them US.
        #
        # These are counterfactuals, tagged as such and landing as
        # ``evaluation_source=shadow``, so promotion still weights them below a real
        # fill and cannot mistake a simulation for execution.
        self._journal_shadow_proposals(
            executable, now, counterfactual=True, exclude=selected
        )

        legacy_selected = selected
        v2_results = self._observe_selector_v2(
            tradable_candidates, evidence, bundle, now, legacy_selected
        )
        v2_context_id: str | None = None
        if self._selector_v2_live_authority():
            v2_selected, v2_context_id = self._selector_v2_live_choice(v2_results, proposals)
            selected = v2_selected
            if selected is None:
                v2_context_id = None
            decided = True

        if selected is not None:
            if not self._arm(selected, now, macro, account=account):
                return
            if v2_context_id:
                runner = self._selector_v2
                self._state.selector_v2_context_id = v2_context_id
                self._state.selector_v2_authority_state = str(
                    getattr(runner, "authority_state", "SHADOW")
                )
                self._state.selector_v2_order_size_fraction = max(
                    0.0, min(1.0, float(getattr(runner, "order_size_fraction", 0.0) or 0.0))
                )
                self._state.selection_source = "SELECTOR_V2"
            return
        if decided:
            # A selector ran and chose nothing; it owns the reason string.
            if self._selector_v2_live_authority():
                self._state.last_reason = "SELECTOR_V2_NO_TRADE"
            return
        if proposals:
            self._state.last_reason = (
                f"{DEPLOYMENT_SHADOW_ONLY}:"
                f"{','.join(sorted({p.strategy_id for p in proposals}))}"
            )
            return

        self._state.last_reason = self._no_election_reason(evidence, intents)

    # -- StrategySelectorV2 observation and authority ----------------------- #
    def _observe_selector_v2(
        self,
        candidates: tuple[str, ...],
        evidence: Mapping[str, Any],
        bundle: Any,
        now: datetime,
        selected: "_ElectionProposal | None",
    ) -> tuple[Any, ...]:
        """Run StrategySelectorV2 over the same point-in-time inputs.

        Legacy first constructs its candidate proposal set; V2 then sees the identical
        ``evidence`` mapping. In SHADOW the result is comparison-only. Once the persisted
        controller grants authority, the caller may resolve the result only to one of those
        independently order-authorised proposals. Three properties keep this safe:

        * it is skipped entirely unless ``STRATEGY_SELECTOR_V2_ENABLED`` is set (default off);
        * the runner catches its own exceptions, and this call catches anything that escapes.
          That double guard is not belt-and-braces: the engine wraps
          ``strategy_session_manager.evaluate`` in a handler that DISABLES BUYS on error, so
          an exception leaking from telemetry would stop trading;
        * the runner has no import path to the execution layer and cannot create an order;
          the session boundary can only select an existing authorised proposal.
        """
        runner = self._selector_v2
        if runner is None:
            return ()
        try:
            return tuple(runner.observe(
                candidates=candidates,
                evidence=evidence,
                bundle=bundle,
                now=now,
                legacy_strategy=selected.strategy_id if selected is not None else None,
                legacy_symbol=selected.symbol if selected is not None else None,
                legacy_reason=self._state.last_reason,
            ))
        except Exception:  # noqa: BLE001 - see the docstring: a raise here stops trading.
            return ()

    def _selector_v2_live_authority(self) -> bool:
        runner = self._selector_v2
        return bool(runner is not None and getattr(runner, "live_authority", False))

    @staticmethod
    def _selector_v2_live_choice(
        results: tuple[Any, ...], proposals: list["_ElectionProposal"]
    ) -> tuple["_ElectionProposal | None", str | None]:
        """Resolve V2's cross-symbol recommendation to an executable proposal.

        V2 is a selector, not an order builder.  It may only choose a proposal the live
        session independently constructed and marked order-authorised.  A missing match
        becomes NO_TRADE rather than inventing an execution contract.
        """
        selected_results = [
            item
            for item in results
            if str(getattr(item, "decision", "")).upper() == "SELECT"
            and getattr(item, "selected_strategy", None)
        ]
        if not selected_results:
            return None, None
        result = max(
            selected_results,
            key=lambda item: (
                float(getattr(item, "utility"))
                if getattr(item, "utility", None) is not None
                else -float("inf")
            ),
        )
        winner = next(
            (
                proposal
                for proposal in proposals
                if proposal.strategy_id == str(result.selected_strategy)
                and proposal.symbol == str(getattr(result, "symbol", "")).upper()
                and proposal.submits_orders
            ),
            None,
        )
        return winner, str(getattr(result, "context_id", "") or "") or None

    @property
    def selector_v2(self) -> Any | None:
        """The V2 observer/authority runner, or ``None`` when disabled."""
        return self._selector_v2

    # -- proposal construction --------------------------------------------- #
    def _registry_algorithm_proposals(
        self,
        candidates: tuple[str, ...],
        evidence: Mapping[str, Any],
        bundle: Any,
        now: datetime,
    ) -> list["_ElectionProposal"]:
        """Evaluate the complete deterministic strategy catalogue.

        This is the baseline decision system. Macro ontology permission is applied
        first, then every reachable strategy's own entry rule and cost-aware edge
        floor. GNN evidence is evaluated by the existing evidence path and wins the
        deduplication tie when it is trusted; a missing or stale checkpoint can no
        longer collapse the baseline candidate set to 0/0.
        """
        proposals: list[_ElectionProposal] = []
        micro_results = tuple(getattr(bundle, "micro_results", ()) or ())
        for raw_symbol in candidates:
            symbol = str(raw_symbol or "").upper()
            row = evidence.get(symbol) if isinstance(evidence, Mapping) else None
            if not isinstance(row, Mapping) or not self._fresh_evidence(row, now):
                continue
            raw_features = row.get("technical_features")
            if not isinstance(raw_features, Mapping):
                continue
            micro_result = next(
                (
                    item
                    for item in micro_results
                    if str(getattr(item, "symbol", "") or "").upper() == symbol
                ),
                None,
            )
            for strategy_id in STRATEGY_IDS:
                if _macro_permits(bundle, strategy_id) is False:
                    continue
                direction, product, deployment_state, borrow_snapshot, borrow_reasons = (
                    self._resolve_direction_context(strategy_id, symbol, now)
                )
                if deployment_state is StrategyDeploymentState.DISABLED:
                    continue
                decision = self._mechanical_entry_verdict(
                    symbol=symbol,
                    strategy_id=strategy_id,
                    evidence_row=row,
                    now=now,
                    macro=getattr(bundle, "macro_result", None),
                    intent=None,
                    micro_result=micro_result,
                    candidate_count=len(candidates),
                    borrow_snapshot=borrow_snapshot,
                )
                if not decision or not bool(decision.get("triggered")):
                    continue

                gross_edge = _optional_float(decision.get("expected_edge_bps"))
                if gross_edge is None:
                    continue
                expected_cost = _all_in_round_trip_cost_bps(
                    symbol,
                    fallback_bps=self.config.fallback_round_trip_cost_bps,
                    micro_result=micro_result,
                    evidence_row=row,
                )
                geometry = _resolved_geometry(
                    strategy_id,
                    expected_cost_bps=expected_cost,
                    micro_result=micro_result,
                    evidence_row=row,
                )
                entry_price = _optional_float(row.get("mark_price")) or _optional_float(
                    raw_features.get("price")
                )
                proposals.append(
                    _ElectionProposal(
                        symbol=symbol,
                        strategy_id=strategy_id,
                        source="ONTOLOGY_ALGORITHM_ELECTION",
                        entry_price=entry_price,
                        target_return_rate=max(
                            self.config.fallback_target_return_rate,
                            geometry.take_profit_bps / 10_000.0,
                        ),
                        stop_loss_rate=geometry.stop_loss_bps / 10_000.0,
                        trailing_stop_rate=geometry.trailing_bps / 10_000.0,
                        max_holding_seconds=geometry.max_holding_seconds,
                        score=float(decision.get("score") or 0.0),
                        confidence=float(decision.get("confidence") or 0.0),
                        expected_net_return_bps=gross_edge - expected_cost,
                        expected_cost_bps=expected_cost,
                        gnn_actionable=False,
                        gnn_action="AUXILIARY_PENDING",
                        gnn_reason_codes=["GNN_AUXILIARY_TO_ALGORITHM_ELECTION"],
                        ontology_reason_codes=["MACRO_ONTOLOGY_PERMITTED"],
                        macro_regime=str(self._state.macro_regime or ""),
                        micro_regime=str(
                            getattr(
                                getattr(micro_result, "micro_regime", None), "value", ""
                            )
                            or ""
                        ),
                        explanation_paths=[],
                        intent=None,
                        candidate_count=len(candidates),
                        micro_result=micro_result,
                        evidence_row=row,
                        last_reason="ONTOLOGY_ALGORITHM_STRATEGY_ARMED",
                        direction=direction,
                        position_effect=PositionEffect.OPEN,
                        execution_product=product,
                        deployment_state=deployment_state,
                        borrow_snapshot=borrow_snapshot,
                        borrow_reason_codes=borrow_reasons,
                        gnn_required_for_edge=False,
                        algorithm_triggered=True,
                    )
                )
        return proposals

    def _intent_proposals(
        self,
        intents: list[Any],
        evidence: Mapping[str, Any],
        bundle: Any,
        now: datetime,
    ) -> list["_ElectionProposal"]:
        proposals: list[_ElectionProposal] = []
        for intent in intents:
            symbol = str(getattr(intent, "symbol", "") or "").upper()
            ontology_strategy = str(getattr(intent, "selected_strategy", "") or "")
            if not symbol or not ontology_strategy:
                continue
            row = evidence.get(symbol) if isinstance(evidence, Mapping) else None
            decisions = list((row or {}).get("decisions") or ()) if isinstance(row, Mapping) else []
            gnn = next((item for item in decisions if item.get("path") == "cpu_gnn"), {})
            gnn_action = str(gnn.get("action") or "UNAVAILABLE").upper()
            gnn_strategy = str(gnn.get("strategy_id") or "")
            gnn_reason_codes = list(gnn.get("reason_codes") or ())
            gnn_actionable = (
                is_actionable_strategy_route(gnn_action)
                and is_known_strategy(gnn_strategy)
                and "GNN_REALTIME_TRUST_PASSED" in gnn_reason_codes
                and "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED" not in gnn_reason_codes
            )
            # An untrusted GNN is a MISSING estimator, not a refusal. With the
            # bandit on, the candidate survives and pays an explicit uncertainty
            # penalty; with the bandit off, the historical hard veto is kept.
            if not gnn_actionable and (
                self.config.require_live_gnn and not self.config.bandit_enabled
            ):
                continue
            entry_price = float(getattr(intent, "expected_entry_price", 0.0) or 0.0)
            expected_exit = float(getattr(intent, "expected_exit_price", 0.0) or 0.0)
            target_rate = self.config.fallback_target_return_rate
            rvgi_context = (
                row.get("rvgi_box_context")
                if isinstance(row, Mapping)
                else None
            )
            # Translate the micro layer's METHODOLOGY name into a catalogued
            # strategy id. Without this the ontology path elected names like
            # "momentum", which resolve to no algorithm and were rejected as
            # STRATEGY_NOT_LIVE_AUTHORIZED — 0 of 13 strategies were reachable.
            selected_strategy = (
                gnn_strategy
                if gnn_actionable
                else (resolve_strategy_id(ontology_strategy) or "")
            )
            if not selected_strategy:
                # A non-tradable verdict (hold/sell/reduce_risk) or an unmappable
                # name is not a candidate. Skipping is the honest outcome.
                continue
            if (
                not gnn_actionable
                and selected_strategy in {"breakout_volume", "rvgi_box_breakout"}
                and isinstance(rvgi_context, Mapping)
                and rvgi_context.get("ontology_eligible") is True
            ):
                selected_strategy = "rvgi_box_breakout"
            direction, product, deployment_state, borrow_snapshot, borrow_reasons = (
                self._resolve_direction_context(selected_strategy, symbol, now)
            )
            if deployment_state is StrategyDeploymentState.DISABLED:
                self._state.last_reason = (
                    f"{ShortReasonCodes.DEPLOYMENT_DISABLED}:{selected_strategy}"
                )
                continue
            # Do not discard SHADOW arms here.  Their proposals are the only
            # causal source of forward outcomes used to decide whether they can
            # ever be promoted.  A missing short locate is likewise journaled as
            # signal-valid-but-unexecutable; ``submits_orders`` still prevents an
            # order even if the deployment controller says LIVE.
            if _macro_permits(bundle, selected_strategy) is False:
                self._state.last_reason = (
                    f"MACRO_BLOCKS_ELECTED_STRATEGY:{selected_strategy}"
                )
                continue
            # Model-supplied target, folded in only once the direction is known.
            # ``target_rate`` is always a positive MAGNITUDE — direction is applied
            # when the price is computed — so the test is whether the expected exit
            # lies on this direction's FAVOURABLE side: above entry for a long, below
            # it for a short. A single ``expected_exit > entry_price`` test would
            # silently discard every short's model target and fall back to the
            # geometry floor.
            forecast_gross_bps: float | None = None
            if entry_price > 0 and expected_exit > 0:
                favourable = (
                    expected_exit > entry_price
                    if direction is PositionDirection.LONG
                    else expected_exit < entry_price
                )
                if favourable:
                    # The signal's OWN forward move. It was previously folded into
                    # the target and then thrown away, which left the cost gate with
                    # nothing to divide but the barrier.
                    forecast_gross_bps = abs(expected_exit / entry_price - 1.0) * 10_000.0
                    target_rate = max(target_rate, abs(expected_exit / entry_price - 1.0))
            micro_result = next(
                (
                    result
                    for result in tuple(getattr(bundle, "micro_results", ()) or ())
                    if str(getattr(result, "symbol", "") or "").upper() == symbol
                ),
                None,
            )
            expected_cost = _all_in_round_trip_cost_bps(
                symbol,
                fallback_bps=self.config.fallback_round_trip_cost_bps,
                model_estimate_bps=_optional_float(gnn.get("expected_cost_bps")),
                micro_result=micro_result,
                evidence_row=row if isinstance(row, Mapping) else None,
            )
            geometry = _resolved_geometry(
                selected_strategy,
                expected_cost_bps=expected_cost,
                micro_result=micro_result,
                evidence_row=row if isinstance(row, Mapping) else None,
            )
            stop_bps = geometry.stop_loss_bps
            profit_bps = geometry.take_profit_bps
            trailing_bps = geometry.trailing_bps
            profit_bps = _cost_aware_profit_bps(gnn, profit_bps)
            target_rate = max(target_rate, profit_bps / 10_000.0)
            mechanical = self._mechanical_entry_verdict(
                symbol=symbol,
                strategy_id=selected_strategy,
                evidence_row=row if isinstance(row, Mapping) else None,
                now=now,
                macro=getattr(bundle, "macro_result", None),
                intent=intent,
                micro_result=micro_result,
                candidate_count=len(intents),
                borrow_snapshot=borrow_snapshot,
            )
            if mechanical is not None and not mechanical.get("triggered", False):
                continue
            proposals.append(
                _ElectionProposal(
                    symbol=symbol,
                    strategy_id=selected_strategy,
                    source=(
                        "ONTOLOGY_GNN_AGREEMENT"
                        if gnn_actionable and gnn_strategy == ontology_strategy
                        else (
                            "GNN_STRATEGY_ELECTION"
                            if gnn_actionable
                            else "ONTOLOGY_WITH_GNN_GUARD"
                        )
                    ),
                    entry_price=entry_price or None,
                    target_return_rate=target_rate,
                    stop_loss_rate=stop_bps / 10_000.0,
                    trailing_stop_rate=trailing_bps / 10_000.0,
                    max_holding_seconds=geometry.max_holding_seconds,
                    score=float(getattr(intent, "score", 0.0) or 0.0),
                    confidence=float(getattr(intent, "confidence", 0.0) or 0.0),
                    expected_net_return_bps=_optional_float(
                        getattr(intent, "expected_net_return_bps", None)
                    ),
                    expected_cost_bps=expected_cost,
                    predicted_gross_move_bps=forecast_gross_bps,
                    gnn_actionable=gnn_actionable,
                    gnn_action=gnn_action,
                    gnn_reason_codes=gnn_reason_codes,
                    ontology_reason_codes=list(getattr(intent, "reason_codes", ()) or ()),
                    macro_regime=str(getattr(intent, "macro_regime", "") or ""),
                    micro_regime=str(getattr(intent, "micro_regime", "") or ""),
                    explanation_paths=list(getattr(intent, "explanation_paths", ()) or ()),
                    intent=intent,
                    candidate_count=len(intents),
                    micro_result=micro_result,
                    evidence_row=row if isinstance(row, Mapping) else None,
                    last_reason="SINGLE_SYMBOL_STRATEGY_ARMED",
                    direction=direction,
                    position_effect=PositionEffect.OPEN,
                    execution_product=product,
                    deployment_state=deployment_state,
                    borrow_snapshot=borrow_snapshot,
                    borrow_reason_codes=borrow_reasons,
                    algorithm_triggered=bool(
                        mechanical is not None and mechanical.get("triggered", False)
                    ),
                )
            )
        return proposals

    def _evidence_proposals(
        self,
        candidates: tuple[str, ...],
        evidence: Mapping[str, Any],
        bundle: Any,
        now: datetime,
    ) -> list["_ElectionProposal"]:
        """Build every admissible ``(symbol, strategy, direction)`` proposal.

        The GNN already emits a vector for every strategy.  Older code discarded
        that vector after choosing one strategy inside each symbol, then compared
        only those per-symbol winners.  That nested ranking used a different
        objective from the final bandit and could therefore discard the global
        winner.  Validation rows carry the full vector without order authority;
        this method converts them into proposals and leaves the one final choice to
        the global bandit.
        """
        proposals: list[_ElectionProposal] = []
        ranked_intents = tuple(getattr(bundle, "ranked_trade_intents", ()) or ())
        micro_results = tuple(getattr(bundle, "micro_results", ()) or ())
        for symbol in candidates:
            normalized = str(symbol or "").upper()
            row = evidence.get(normalized) if isinstance(evidence, Mapping) else None
            if not isinstance(row, Mapping) or not self._fresh_evidence(row, now):
                continue
            decisions = [
                item
                for item in list(row.get("decisions") or ())
                if isinstance(item, Mapping)
            ]
            ontology = next(
                (item for item in decisions if item.get("path") == "ontology"),
                {},
            )
            gnn = next(
                (item for item in decisions if item.get("path") == "cpu_gnn"),
                {},
            )
            ontology_action = str(ontology.get("action") or "").upper()
            ontology_strategy = str(ontology.get("strategy_id") or "")
            # A generic ontology allow/admissible result is only a gate.  It
            # carries no strategy-specific evidence and must not elect the
            # first catalog item as an executable strategy.  Explicit
            # ACTIVATE_STRATEGY decisions are produced by strategy-specific
            # ontology rules (for example, RVGI box breakout).
            ontology_actionable = (
                ontology_action == "ACTIVATE_STRATEGY"
                and is_known_strategy(ontology_strategy)
            )
            gnn_action = str(gnn.get("action") or "UNAVAILABLE").upper()
            gnn_strategy = str(gnn.get("strategy_id") or "")
            gnn_reason_codes = list(gnn.get("reason_codes") or ())
            gnn_actionable = (
                is_actionable_strategy_route(gnn_action)
                and is_known_strategy(gnn_strategy)
                and "GNN_REALTIME_TRUST_PASSED" in gnn_reason_codes
                and "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED" not in gnn_reason_codes
            )
            strategy_rows = [
                item
                for item in tuple(row.get("validation_candidates") or ())
                if isinstance(item, Mapping)
                and str(item.get("path") or "") == "cpu_gnn_validation"
                and is_known_strategy(str(item.get("strategy_id") or ""))
            ]
            # Backward compatibility for logs written before full-vector rows were
            # persisted.  New rows normally take the branch above.
            if not strategy_rows and gnn_strategy:
                strategy_rows = [gnn]
            if ontology_actionable and not any(
                str(item.get("strategy_id") or "") == ontology_strategy
                for item in strategy_rows
            ):
                strategy_rows.append(ontology)

            intent = next(
                (
                    item
                    for item in ranked_intents
                    if str(getattr(item, "symbol", "") or "").upper() == normalized
                ),
                None,
            )
            micro_result = next(
                (
                    item
                    for item in micro_results
                    if str(getattr(item, "symbol", "") or "").upper() == normalized
                ),
                None,
            )
            for strategy_row in strategy_rows:
                selected_strategy = str(strategy_row.get("strategy_id") or "")
                if not selected_strategy:
                    continue
                row_reasons = list(strategy_row.get("reason_codes") or ())
                row_trusted = "GNN_REALTIME_TRUST_PASSED" in row_reasons
                checkpoint_authorized = (
                    "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED" not in row_reasons
                )
                row_path = str(strategy_row.get("path") or "")
                row_is_gnn = row_path.startswith("cpu_gnn")
                row_is_validation = row_path == "cpu_gnn_validation"
                row_actionable = row_is_gnn and row_trusted and checkpoint_authorized
                # Full-vector rows are evidence, not permission.  Keep untrusted
                # rows as SHADOW proposals so they are compared and journalled,
                # but force their deployment state below to SHADOW.  The previous
                # ``continue`` made the trust gate self-locking: no proposal meant
                # no forward outcome, hence no strategy could ever earn trust.
                # A legacy untrusted winner row is not a full-vector validation
                # sample and retains the historical hard skip.
                if row_is_gnn and not row_actionable and not row_is_validation:
                    continue
                direction, product, deployment_state, borrow_snapshot, borrow_reasons = (
                    self._resolve_direction_context(selected_strategy, normalized, now)
                )
                # An untrusted validation vector has no MODEL order authority,
                # but it must not revoke an algorithm's independent deployment
                # authorization.  The mechanical trigger supplies the forward
                # edge below and the bandit's cold-start exploration policy keeps
                # sizing conservative. Deployment-gated strategies remain SHADOW
                # because _resolve_direction_context already returned SHADOW.
                if deployment_state is StrategyDeploymentState.DISABLED:
                    self._state.last_reason = (
                        f"{ShortReasonCodes.DEPLOYMENT_DISABLED}:{selected_strategy}"
                    )
                    continue
                if _macro_permits(bundle, selected_strategy) is False:
                    self._state.last_reason = (
                        f"MACRO_BLOCKS_ELECTED_STRATEGY:{selected_strategy}"
                    )
                    continue
                expected_cost = _all_in_round_trip_cost_bps(
                    normalized,
                    fallback_bps=self.config.fallback_round_trip_cost_bps,
                    model_estimate_bps=_optional_float(
                        strategy_row.get("expected_cost_bps")
                    ),
                    micro_result=micro_result,
                    evidence_row=row if isinstance(row, Mapping) else None,
                )
                geometry = _resolved_geometry(
                    selected_strategy,
                    expected_cost_bps=expected_cost,
                    micro_result=micro_result,
                    evidence_row=row if isinstance(row, Mapping) else None,
                )
                stop_bps = geometry.stop_loss_bps
                profit_bps = geometry.take_profit_bps
                trailing_bps = geometry.trailing_bps
                profit_bps = _cost_aware_profit_bps(strategy_row, profit_bps)
                entry_price = _optional_float(
                    getattr(intent, "expected_entry_price", None)
                )
                if entry_price is None:
                    entry_price = _optional_float(row.get("mark_price"))
                expected_exit = _optional_float(
                    getattr(intent, "expected_exit_price", None)
                )
                target_rate = max(
                    self.config.fallback_target_return_rate,
                    profit_bps / 10_000.0,
                )
                forecast_gross_bps: float | None = None
                if entry_price and expected_exit:
                    favourable = (
                        expected_exit > entry_price
                        if direction is PositionDirection.LONG
                        else expected_exit < entry_price
                    )
                    if favourable:
                        forecast_gross_bps = (
                            abs(expected_exit / entry_price - 1.0) * 10_000.0
                        )
                        target_rate = max(
                            target_rate,
                            abs(expected_exit / entry_price - 1.0),
                        )
                mechanical = self._mechanical_entry_verdict(
                    symbol=normalized,
                    strategy_id=selected_strategy,
                    evidence_row=row,
                    now=now,
                    macro=getattr(bundle, "macro_result", None),
                    intent=intent,
                    micro_result=micro_result,
                    candidate_count=len(candidates),
                    borrow_snapshot=borrow_snapshot,
                    predicted_net_edge_bps=_optional_float(
                        strategy_row.get("expected_net_return_bps")
                    ),
                )
                if mechanical is not None and not mechanical.get("triggered", False):
                    continue
                # Historical/replay rows can lack the feature vector required to
                # prove an algorithm trigger.  Keep those untrusted validation
                # rows for posterior measurement, but never let the absence of a
                # verdict become live order authority.  A cold-start live probe is
                # possible only when the owned algorithm actually fired above.
                if row_is_gnn and not row_actionable and mechanical is None:
                    deployment_state = StrategyDeploymentState.SHADOW
                model_net = _optional_float(strategy_row.get("expected_net_return_bps"))
                # An untrusted GNN vector proposes a pair but cannot define its
                # forward edge.  For a real mechanical trigger, use the strategy's
                # own measured gross move less point-in-time cost. Trusted rows keep
                # the trained model estimate.
                if mechanical is not None and not row_actionable:
                    mechanical_gross = _optional_float(mechanical.get("expected_edge_bps"))
                    if mechanical_gross is not None:
                        model_net = mechanical_gross - (
                            expected_cost
                            if expected_cost is not None
                            else self.config.fallback_round_trip_cost_bps
                        )
                proposals.append(
                    _ElectionProposal(
                        symbol=normalized,
                        strategy_id=selected_strategy,
                        source=(
                            "GNN_JOINT_SYMBOL_STRATEGY_ELECTION"
                            if row_actionable
                            else "ALGORITHM_MECHANICAL_ELECTION"
                            if row_is_gnn and mechanical is not None
                            else "ONTOLOGY_STRATEGY_ELECTION"
                        ),
                        entry_price=entry_price,
                        target_return_rate=target_rate,
                        stop_loss_rate=stop_bps / 10_000.0,
                        trailing_stop_rate=trailing_bps / 10_000.0,
                        max_holding_seconds=geometry.max_holding_seconds,
                        score=_optional_float(strategy_row.get("utility")) or 0.0,
                        confidence=(
                            _optional_float(strategy_row.get("probability_success"))
                            or 0.0
                        ),
                        expected_net_return_bps=model_net,
                        expected_cost_bps=expected_cost,
                        predicted_gross_move_bps=forecast_gross_bps,
                        gnn_actionable=row_actionable,
                        gnn_action=(
                            "ACTIVATE_STRATEGY" if row_actionable else gnn_action
                        ),
                        gnn_reason_codes=row_reasons,
                        ontology_reason_codes=list(
                            ontology.get("reason_codes") or ()
                        ),
                        macro_regime=self._state.macro_regime or "",
                        micro_regime=str(
                            getattr(
                                getattr(micro_result, "micro_regime", None),
                                "value",
                                getattr(micro_result, "micro_regime", ""),
                            )
                            or ""
                        ),
                        explanation_paths=[],
                        intent=intent,
                        candidate_count=len(candidates),
                        micro_result=micro_result,
                        evidence_row=row,
                        last_reason="JOINT_SYMBOL_STRATEGY_ARMED",
                        direction=direction,
                        position_effect=PositionEffect.OPEN,
                        execution_product=product,
                        deployment_state=deployment_state,
                        borrow_snapshot=borrow_snapshot,
                        borrow_reason_codes=borrow_reasons,
                        # This proposal's edge came from the deterministic
                        # algorithm verdict above.  The validation GNN row is
                        # auxiliary context only; charging the model-absence
                        # penalty here turned the SAME algorithm signal from
                        # 19.4bp net into 4.4bp net in production.
                        gnn_required_for_edge=not (
                            row_is_gnn and mechanical is not None
                        ),
                        algorithm_triggered=bool(
                            mechanical is not None and mechanical.get("triggered", False)
                        ),
                    )
                )
        return proposals

    def _mechanical_entry_verdict(
        self,
        *,
        symbol: str,
        strategy_id: str,
        evidence_row: Mapping[str, Any] | None,
        now: datetime,
        macro: Any,
        intent: Any,
        micro_result: Any,
        candidate_count: int,
        borrow_snapshot: Any,
        predicted_net_edge_bps: float | None = None,
    ) -> dict[str, Any] | None:
        """Run the strategy's own entry trigger before scoring or journalling.

        ``None`` preserves old/replay fixtures that have no point-in-time feature
        snapshot. Production evidence always has one. A concrete HOLD is retained
        for diagnostics but cannot become a proposal, shadow outcome, posterior
        sample, or live order.
        """
        raw_features = (
            evidence_row.get("technical_features")
            if isinstance(evidence_row, Mapping)
            else None
        )
        if not isinstance(raw_features, Mapping):
            return None
        try:
            from app.technical.signals import TechnicalFeatureSet
            from app.technical.strategy_algorithms import ElectionContext, get_algorithm

            feature_names = TechnicalFeatureSet.__dataclass_fields__.keys()
            features = TechnicalFeatureSet(
                **{key: value for key, value in raw_features.items() if key in feature_names}
            )
            algorithm = get_algorithm(
                strategy_id, registry=self._algorithm_registry
            )
            if algorithm is None:
                decision = {
                    "strategy_id": strategy_id,
                    "triggered": False,
                    "score": 0.0,
                    "confidence": 0.0,
                    "expected_edge_bps": 0.0,
                    "horizon_seconds": 0,
                    "reason_codes": ["STRATEGY_IMPLEMENTATION_MISSING"],
                    "diagnostics": {},
                }
            else:
                context = self._election_context(
                    strategy_id,
                    now,
                    intent=intent,
                    candidate_count=candidate_count,
                    micro_result=micro_result,
                    evidence_row=evidence_row,
                    macro=macro,
                    symbol=symbol,
                    change_point_probability=self._state.change_point_probability,
                    borrow_snapshot=borrow_snapshot,
                )
                if predicted_net_edge_bps is not None:
                    context["expected_net_return_bps"] = float(predicted_net_edge_bps)
                allowed = ElectionContext.__dataclass_fields__.keys()
                payload = {
                    key: value
                    for key, value in context.items()
                    if key in allowed and key != "elected_at"
                }
                payload["strategy_id"] = strategy_id
                payload["elected_at"] = now
                decision = algorithm.entry(
                    features,
                    ElectionContext(**payload),
                ).as_dict()
        except Exception as exc:  # noqa: BLE001 - malformed live evidence fails closed.
            decision = {
                "strategy_id": strategy_id,
                "triggered": False,
                "score": 0.0,
                "confidence": 0.0,
                "expected_edge_bps": 0.0,
                "horizon_seconds": 0,
                "reason_codes": [f"STRATEGY_ENTRY_EVALUATION_ERROR:{type(exc).__name__}"],
                "diagnostics": {},
            }
        # The algorithm-primary path and a trusted GNN path may inspect the same
        # pair in one cycle. It is one evaluation, not two denominator samples.
        key = (str(symbol).upper(), str(strategy_id))
        if not any(
            (
                str(item.get("symbol") or "").upper(),
                str(item.get("strategy_id") or ""),
            )
            == key
            for item in self._state.algorithm_evaluations
        ):
            self._state.algorithm_evaluations.append({"symbol": symbol, **decision})
        return decision

    @staticmethod
    def _deduplicate_joint_proposals(
        proposals: list["_ElectionProposal"],
    ) -> list["_ElectionProposal"]:
        """Keep one rich proposal for each global election arm."""
        selected: dict[tuple[str, str, PositionDirection], _ElectionProposal] = {}
        for proposal in proposals:
            key = (proposal.symbol, proposal.strategy_id, proposal.direction)
            incumbent = selected.get(key)
            if incumbent is None:
                selected[key] = proposal
                continue
            def quality(item: "_ElectionProposal") -> tuple[int, bool, bool, float]:
                # The owned algorithm is the authority. A learned proposal may
                # be measured alongside it, but may not replace its edge or
                # confidence for the same symbol-strategy-direction arm.
                authority = (
                    3
                    if item.source == "ONTOLOGY_ALGORITHM_ELECTION"
                    else 2
                    if item.algorithm_triggered
                    else 1
                    if item.gnn_actionable
                    else 0
                )
                return (
                    authority,
                    item.expected_net_return_bps is not None,
                    item.intent is not None,
                    float(item.confidence or 0.0),
                )

            incumbent_quality = quality(incumbent)
            proposal_quality = quality(proposal)
            if proposal_quality > incumbent_quality:
                selected[key] = proposal
        return list(selected.values())

    def _deployment_authorized(self, strategy_id: str) -> tuple[bool, str]:
        """Is this strategy authorised for LIVE deployment (not just enabled)?

        Deployment-gated strategies (RVGI box breakout and the three added for the
        current tape) ship shadow-only until they have per-regime samples.

        This is the LONG-side gate and remains a plain per-strategy flag. Short arms
        go through :meth:`_directional_deployment_state` instead, which reads the
        committed per-arm state from the promotion controller — a strategy-level
        boolean is not enough for a short, because the same strategy id can be at
        different rungs in different markets.
        """
        try:
            from app.technical.strategy_algorithms import strategy_live_authorized

            if strategy_live_authorized(
                strategy_id, registry=self._algorithm_registry
            ):
                return True, ""
        except Exception:  # noqa: BLE001 - a lookup failure must fail closed.
            return False, f"STRATEGY_AUTHORIZATION_UNAVAILABLE:{strategy_id}"
        return False, f"STRATEGY_NOT_LIVE_AUTHORIZED:{strategy_id}"

    def _directional_deployment_state(
        self, strategy_id: str, direction: PositionDirection, market: str
    ) -> StrategyDeploymentState:
        """Committed deployment state for one arm.

        The per-strategy ``live_authorized`` flag grants LONG arms LIVE_FULL, and it
        still does — but it may no longer outrank a measured NEGATIVE edge.

        The flag used to short-circuit before the evidence-based controller ran, so
        that controller was consulted only for arms already switched off: the one
        case where its answer could not matter. Measured 2026-08-21 over the stored
        outcomes, ``evaluate_long_promotion`` returned SHADOW for every LONG strategy
        in the catalogue, with conservative edges from -85 to -281bps, while all of
        them were arming at LIVE_FULL. Against the tape those same signals produced a
        forward return 21bps BELOW an unconditional entry on the same symbols
        (``vwap_mean_reversion`` 93bps below, ``breakout_volume`` 118bps below). Full
        size on a measured negative edge is not a risk posture, it is an arithmetic
        mistake.

        The demotion is deliberately narrow. "No evidence yet" is NOT "bad evidence":
        a cold arm keeps whatever the operator's flag grants it, because the flag is
        how a new thesis gets its first fills and the promotion ladder is how it
        earns the rest. Only an arm the store can actually speak about — at least
        ``minimum_shadow_samples`` outcomes — and whose conservative edge is
        non-positive is pulled down to the rung it earned.

        SHORT arms were already evidence-gated per-arm in the promotion store and are
        unchanged.

        Any failure resolves to SHADOW. An unreadable deployment state must never
        authorise an order, and for a short that is the difference between a journal
        entry and a borrowed position.
        """
        if direction is PositionDirection.LONG:
            authorized, _ = self._deployment_authorized(strategy_id)
            try:
                from app.technical.strategy_algorithms import strategy_shadow_authorized
                from app.trading.long_strategy_promotion import (
                    LongPromotionConfig,
                    evaluate_long_promotion,
                )

                if not authorized and not strategy_shadow_authorized(
                    strategy_id, registry=self._algorithm_registry
                ):
                    return StrategyDeploymentState.DISABLED
                decision = evaluate_long_promotion(
                    strategy_id, market, self.performance_store
                )
            except Exception:  # noqa: BLE001 - fail closed to SHADOW.
                return StrategyDeploymentState.SHADOW
            if not authorized:
                return decision.state
            config = LongPromotionConfig.from_env()
            measured_negative = (
                decision.sample_count >= config.minimum_shadow_samples
                and decision.conservative_edge_bps
                <= config.minimum_shadow_conservative_edge_bps
            )
            if not measured_negative:
                return StrategyDeploymentState.LIVE_FULL
            # Demotion only: the flag is a ceiling, never a floor under evidence.
            return min(
                StrategyDeploymentState.LIVE_FULL,
                decision.state,
                key=lambda item: item.rank,
            )
        try:
            from app.trading.short_strategy_promotion import default_promotion_controller

            key = DirectionalStrategyKey.for_short(strategy_id, market)
            return default_promotion_controller().authorized_state(key)
        except Exception:  # noqa: BLE001 - fail closed to SHADOW.
            return StrategyDeploymentState.SHADOW

    def _resolve_direction_context(
        self, strategy_id: str, symbol: str, now: datetime
    ) -> tuple[
        PositionDirection, ExecutionProduct, StrategyDeploymentState, Any, tuple[str, ...]
    ]:
        """Direction, product, deployment state and borrow locate for one candidate.

        Collected in one place so both election paths agree. Direction comes from the
        STRATEGY (each thesis is one-directional); the product follows from the
        direction; the deployment state is per-arm; the borrow locate is only fetched
        for shorts, because a long has no borrow leg to fail on.
        """
        from app.technical.strategy_algorithms import strategy_direction

        direction = parse_direction(strategy_direction(strategy_id))
        product = default_product(direction)
        market = market_for_symbol(symbol)
        deployment_state = self._directional_deployment_state(strategy_id, direction, market)
        if direction is PositionDirection.LONG:
            return direction, product, deployment_state, None, ()
        snapshot, reasons = self._borrow_context(symbol, now)
        return direction, product, deployment_state, snapshot, reasons

    def _borrow_context(
        self, symbol: str, now: datetime
    ) -> tuple[Any, tuple[str, ...]]:
        """Latest borrow observation for ``symbol``, plus any blocking reasons.

        Returns ``(None, reasons)`` whenever a locate cannot be established. The
        caller must treat that as "not a short candidate" — never as "probably fine".
        """
        try:
            from app.trading.borrow import default_borrow_store, evaluate_borrow

            snapshot = default_borrow_store().latest(symbol, as_of=now)
            verdict = evaluate_borrow(snapshot, quantity=1, now=now)
            return (snapshot if verdict.allowed else None), verdict.reason_codes
        except Exception:  # noqa: BLE001 - a lookup failure is a no-locate.
            return None, (ShortReasonCodes.BORROW_LOOKUP_FAILED,)

    # -- shadow journaling -------------------------------------------------- #
    def _journal_shadow_proposals(
        self,
        proposals: list["_ElectionProposal"],
        now: datetime,
        *,
        counterfactual: bool = False,
        exclude: "_ElectionProposal | None" = None,
    ) -> None:
        """Write a :class:`ShadowTradePlan` for proposals that produced no order.

        Two callers, one mechanism:

        * ``counterfactual=False`` (before selection) journals the non-order-authorised
          proposals. This is how a SHADOW short earns its way to LIVE_PROBE.
        * ``counterfactual=True`` (after selection) journals the order-authorised
          proposals that lost the election, ``exclude`` being the winner. Its outcome
          is recorded from the real fill instead, so journaling it here would score
          the same trade twice against the same arm.

        The second case exists because without it the evidence channel only feeds arms
        that are forbidden from trading. An arm allowed to trade produced a sample only
        by winning, and could only win on a posterior it had no way to accumulate.

        Three properties are load-bearing:

        * The plan is written NOW, with the entry reference, barriers and borrow
          observation frozen. Scoring happens later, from later data only.
        * The borrow snapshot is embedded by value, so a scoring pass physically
          cannot consult a fresher locate — which would be the leak that makes shadow
          results unachievable live.
        * No order is created and none can be: this method has no path to the
          execution layer.

        Failures are swallowed. A journal write is bookkeeping; losing one costs a
        sample, while raising here would break the live LONG election path.
        """
        if not proposals:
            return
        try:
            from app.trading.directional_shadow import ShadowTradePlan, default_shadow_store

            store = default_shadow_store()
        except Exception:  # noqa: BLE001 - journaling must never break election.
            return
        recorded: list[str] = []
        pending: list[Any] = []
        for proposal in proposals:
            if counterfactual:
                # The winner's outcome comes from its real fill; journaling it here
                # would enter the same trade twice into the same arm's posterior.
                if not proposal.submits_orders or proposal is exclude:
                    continue
            elif proposal.submits_orders:
                continue
            entry_reference = proposal.entry_price
            if not entry_reference or entry_reference <= 0:
                # Without a point-in-time entry reference there is nothing to measure
                # a return against, and inventing one from a later quote is the leak
                # this whole module exists to prevent.
                continue
            try:
                key = proposal.directional_key(market_for_symbol(proposal.symbol))
                spacing_seconds = max(
                    self.config.shadow_signal_cooldown_seconds,
                    min(int(proposal.max_holding_seconds), 900),
                )
                recent_reader = getattr(store, "has_recent_plan", None)
                if callable(recent_reader) and recent_reader(
                    key, proposal.symbol,
                    since=now - timedelta(seconds=spacing_seconds),
                ):
                    continue
                plan = ShadowTradePlan(
                    plan_id="",
                    key=key,
                    symbol=proposal.symbol,
                    signal_at=now,
                    entry_reference_price=float(entry_reference),
                    target_rate=proposal.target_return_rate,
                    stop_rate=proposal.stop_loss_rate,
                    max_holding_seconds=proposal.max_holding_seconds,
                    expected_trading_cost_bps=proposal.resolved_cost_bps(
                        self.config.fallback_round_trip_cost_bps
                    ),
                    predicted_gross_edge_bps=proposal.predicted_gross_edge_bps(
                        self.config.fallback_round_trip_cost_bps
                    ),
                    predicted_net_edge_bps=proposal.predicted_net_edge_bps(
                        self.config.gnn_absence_penalty_bps,
                        self.config.fallback_round_trip_cost_bps,
                    ),
                    predicted_success_probability=proposal.confidence or None,
                    regime=self._state.macro_regime or "UNKNOWN",
                    signal_reason_codes=(
                        (*proposal.ontology_reason_codes, _COUNTERFACTUAL_REASON_CODE)
                        if counterfactual
                        else tuple(proposal.ontology_reason_codes)
                    ),
                    borrow_snapshot=proposal.borrow_snapshot,
                    borrow_reason_codes=proposal.borrow_reason_codes,
                    deployment_state=str(proposal.deployment_state),
                )
            except Exception:  # noqa: BLE001
                continue
            if store.record_plan(plan):
                recorded.append(plan.plan_id)
                pending.append(plan)
        if recorded:
            # Extend rather than replace: the counterfactual pass runs after the
            # SHADOW pass in the same cycle, and overwriting here would drop the
            # shorts' plan ids from the state the dashboard reads.
            self._state.shadow_plan_ids = (
                [*self._state.shadow_plan_ids, *recorded][:16]
                if counterfactual
                else recorded[:16]
            )
        # Hand the plan objects to the evaluation service so its barrier walk starts on
        # the NEXT quote. Passing objects rather than ids keeps the frozen borrow
        # snapshot intact — a re-read from the journal would have to re-resolve it, and
        # that is the one place a fresher locate could sneak in.
        if pending:
            self._pending_shadow_plans = (
                [*(self._pending_shadow_plans or []), *pending]
                if counterfactual
                else pending
            )

    # -- single-pass deterministic selection ------------------------------ #
    def _forward_edge_choice(
        self,
        proposals: list["_ElectionProposal"],
        now: datetime,
    ) -> "_ElectionProposal | None":
        """Rank already-admissible proposals without creating another authority."""
        del now  # Kept in the signature for selector-call symmetry and future audit use.
        positive: list[tuple[float, _ElectionProposal]] = []
        for proposal in proposals:
            edge = proposal.predicted_net_edge_bps(
                0.0, self.config.fallback_round_trip_cost_bps
            )
            if edge is not None and edge > 0.0:
                positive.append((float(edge), proposal))
        if not positive:
            self._state.last_reason = "NO_POSITIVE_ALGORITHM_NET_EDGE"
            return None
        edge, chosen = max(
            positive,
            key=lambda item: (
                item[0],
                float(item[1].score or 0.0),
                float(item[1].confidence or 0.0),
                item[1].strategy_id,
                item[1].symbol,
            ),
        )
        chosen.conservative_edge_bps = edge
        return chosen

    def _algorithm_choice(
        self,
        proposals: list["_ElectionProposal"],
        now: datetime,
    ) -> "_ElectionProposal | None":
        """Select only arms whose owned deterministic algorithm fired this cycle."""
        algorithm_arms = [item for item in proposals if item.algorithm_triggered]
        if not algorithm_arms:
            self._state.last_reason = "NO_MECHANICAL_STRATEGY_TRIGGER"
            return None
        return self._forward_edge_choice(algorithm_arms, now)

    # -- legacy bandit scoring (explicit non-primary comparison runs only) -- #
    def _reset_bandit_diagnostics(self) -> None:
        """bandit 진단값을 "이번 사이클엔 아직 안 돌았음" 상태로 되돌린다."""
        self._state.bandit_selected_arm = None
        self._state.bandit_conservative_edge_bps = None
        self._state.bandit_is_exploration = False
        self._state.bandit_reason_codes = []
        self._state.bandit_evaluations = []
        self._state.bandit_shadow_arms = []
        self._state.bandit_evaluated_at = None
        self._state.directional_comparison = {}
        self._state.algorithm_evaluations = []

    def _gnn_direct_choice(
        self,
        executable: list["_ElectionProposal"],
        _now: datetime | None = None,
    ) -> "_ElectionProposal":
        """Arm the GNN's own top pick. Always returns a proposal -- never NO_TRADE.

        Ranking mirrors ``StrategyRouter``: highest forward net edge first, then
        the model's score and confidence, then the id so the choice is stable
        across cycles. ``gnn_actionable`` proposals outrank ones the model could
        not speak to at all -- honouring the model's pick presupposes it made one.

        No pessimistic bound, no realized-history posterior, no regime discount:
        the edge is taken at face value, which is the whole point of the posture.
        """
        self._reset_bandit_diagnostics()

        def key(proposal: "_ElectionProposal") -> tuple[int, float, float, float, str]:
            edge = proposal.predicted_net_edge_bps(
                # No absence penalty: this posture does not dock a candidate for
                # the model's silence, it simply ranks silent ones last.
                0.0,
                self.config.fallback_round_trip_cost_bps,
            )
            return (
                1 if proposal.gnn_actionable else 0,
                float(edge) if edge is not None else float("-inf"),
                float(proposal.score or 0.0),
                float(proposal.confidence or 0.0),
                proposal.strategy_id,
            )

        chosen = max(executable, key=key)
        edge = chosen.predicted_net_edge_bps(
            0.0, self.config.fallback_round_trip_cost_bps
        )
        self._state.bandit_evaluated_at = _iso(_now or datetime.now(timezone.utc))
        self._state.bandit_selected_arm = chosen.strategy_id
        self._state.bandit_conservative_edge_bps = (
            float(edge) if edge is not None else None
        )
        # Not exploration: this is a conviction pick, and labelling it otherwise
        # would let an operator read a loss as a deliberate probe.
        self._state.bandit_is_exploration = False
        self._state.bandit_reason_codes = [
            "GNN_DIRECT_ELECTION",
            (
                "GNN_ESTIMATE_PRESENT"
                if chosen.gnn_actionable
                else "GNN_ESTIMATE_UNAVAILABLE_RANKED_LAST"
            ),
            f"CANDIDATES:{len(executable)}",
        ]
        return chosen

    def _bandit_choice(
        self,
        proposals: list["_ElectionProposal"],
        macro: Any,
        now: datetime,
    ) -> "_ElectionProposal | None":
        """Score every proposal on a pessimistic lower bound; may choose nothing."""
        # 제안 집합이 여러 시장에 걸쳐 있으면 첫 제안의 시장으로 전체를 라벨링할 수 없다.
        # (KR 비용 ~28bps vs US 60-87bps 처럼 시장별 비용·사후분포가 다르므로 잘못된
        # 라벨은 잘못된 posterior 를 고른다.) 단일 시장이면 그 시장, 혼합이면 MIXED 로
        # 명시해 사후분포가 시장별 이력과 섞이지 않게 한다.
        proposal_markets = {market_for_symbol(item.symbol) for item in proposals}
        context = BanditContext(
            market=(
                next(iter(proposal_markets))
                if len(proposal_markets) == 1
                else "MIXED"
            ),
            macro_regime=self._state.macro_regime or "UNKNOWN",
            change_point_probability=self._state.change_point_probability or 0.0,
            regime_stability=_optional_float(getattr(macro, "regime_stability", None)),
            volatility_percentile=_optional_float(
                getattr(macro, "volatility_percentile", None)
            ),
            market_breadth=_optional_float(
                (getattr(macro, "diagnostics", None) or {}).get("market_breadth")
                if isinstance(getattr(macro, "diagnostics", None), Mapping)
                else None
            ),
            foreign_flow_zscore=_optional_float(getattr(macro, "foreign_flow_zscore", None)),
            spread_percentile=_optional_float(getattr(macro, "spread_percentile", None)),
            time_of_day_bucket=f"h{now.astimezone(timezone.utc).hour:02d}",
        )
        arms = [
            ArmCandidate(
                arm=proposal.strategy_id,
                symbol=proposal.symbol,
                predicted_net_edge_bps=proposal.predicted_net_edge_bps(
                    self.config.gnn_absence_penalty_bps,
                    self.config.fallback_round_trip_cost_bps,
                ),
                predicted_gross_edge_bps=proposal.predicted_gross_edge_bps(
                    self.config.fallback_round_trip_cost_bps
                ),
                expected_cost_bps=proposal.resolved_cost_bps(
                    self.config.fallback_round_trip_cost_bps
                ),
                confidence=proposal.confidence,
                live_authorized=True,
                direction=proposal.direction,
                execution_product=proposal.execution_product,
                # The bandit re-applies the deployment gate itself. Passing the state
                # rather than a pre-computed boolean lets it distinguish SHADOW (rank
                # and report) from SUSPENDED (rank and flag the fault), which the
                # dashboard needs to tell "still learning" from "something broke".
                deployment_state=proposal.deployment_state,
                # A snapshot only survives ``_borrow_context`` when the shared
                # ``evaluate_borrow`` rule passed, so its presence IS the locate.
                # ``None`` (not False) when absent, so the bandit records "not
                # established" rather than "broker refused".
                borrow_available=True if proposal.borrow_snapshot is not None else None,
                borrow_fee_bps_annualised=getattr(
                    proposal.borrow_snapshot, "borrow_fee_bps_annualised", None
                ),
                reason_codes=(
                    ()
                    if proposal.gnn_actionable
                    else ("BANDIT_GNN_ESTIMATE_UNAVAILABLE",)
                ),
            )
            for proposal in proposals
        ]
        selection = self.bandit.select(arms, context, now=now)
        self._state.bandit_evaluated_at = _iso(now)
        self._state.bandit_selected_arm = selection.selected_arm
        self._state.bandit_conservative_edge_bps = selection.conservative_edge_bps
        self._state.bandit_is_exploration = selection.is_exploration
        self._state.bandit_reason_codes = list(selection.reason_codes)
        self._state.bandit_evaluations = [item.as_dict() for item in selection.evaluations][:12]
        self._state.bandit_shadow_arms = list(selection.shadow_arms)
        # LONG vs SHORT vs NO_TRADE, recorded on every cycle whether or not a short
        # was selectable. This is the evidence ``short_rescue_rate`` is built from, and
        # the only way to answer after the fact whether short support bought anything.
        self._state.directional_comparison = selection.as_dict().get(
            "directional_comparison", {}
        )
        if selection.is_no_trade:
            # This is a successful outcome, not a failure: no candidate cleared a
            # positive lower bound after costs and uncertainty.
            if BANDIT_MEASURED_EDGE_REJECTED in selection.reason_codes:
                self._state.last_reason = "BANDIT_NO_TRADE_MEASURED_EDGE_REJECTED"
            elif BANDIT_EVIDENCE_WARMUP in selection.reason_codes:
                self._state.last_reason = "BANDIT_NO_TRADE_EVIDENCE_WARMUP"
            else:
                self._state.last_reason = (
                    "BANDIT_NO_TRADE_NO_POSITIVE_CONSERVATIVE_EDGE"
                    if BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE in selection.reason_codes
                    else f"BANDIT_NO_TRADE:{','.join(selection.reason_codes) or 'UNSPECIFIED'}"
                )
            return None
        # Matched on (strategy, symbol, DIRECTION). The bandit labels a short arm
        # ``strategy:SHORT``, and matching on strategy id alone would resolve a
        # selected short to the long proposal of the same strategy on the same symbol
        # — arming the opposite position to the one that was chosen.
        selected_direction = parse_direction(selection.selected_direction)
        winner = next(
            (
                proposal
                for proposal in proposals
                if proposal.strategy_id == selection.selected_arm.split(":", 1)[0]
                and proposal.symbol == selection.selected_symbol
                and proposal.direction is selected_direction
            ),
            None,
        )
        if winner is None:
            self._state.last_reason = "BANDIT_SELECTION_UNRESOLVABLE"
            return None
        # Last line of defence before ARMED. The bandit already refuses to select a
        # non-order-authorised arm, so reaching here means the two disagree, which is
        # a bug — refuse the trade and say so rather than resolving it optimistically.
        if not winner.submits_orders:
            self._state.last_reason = (
                f"{DEPLOYMENT_SHADOW_ONLY}:{winner.strategy_id}"
            )
            return None
        winner.conservative_edge_bps = selection.conservative_edge_bps
        return winner

    def _arm(
        self,
        proposal: "_ElectionProposal",
        now: datetime,
        macro: Any = None,
        *,
        account: Any = None,
    ) -> bool:
        """Commit one proposal to ARMED, preserving the audit trail."""
        coverage = evaluate_cost_coverage(
            proposal.predicted_gross_edge_bps(self.config.fallback_round_trip_cost_bps),
            proposal.resolved_cost_bps(self.config.fallback_round_trip_cost_bps),
        )
        # Selection may rank a proposal, but it may not waive the minimum
        # economics required to survive round-trip costs.  Previously this
        # assessment was written to the UI and then ignored, which allowed an
        # INSUFFICIENT 1.269x DYN proposal to reach the live order path.
        if not coverage.live_eligible:
            ratio = "UNKNOWN" if coverage.ratio is None else f"{coverage.ratio:.3f}"
            self._state.last_reason = (
                f"ENTRY_COST_COVERAGE_REJECTED:{coverage.band.value}:{ratio}"
            )
            self._state.cost_coverage_ratio = coverage.ratio
            self._state.cost_coverage_band = coverage.band.value
            return False
        entry_price = proposal.entry_price
        # Direction-aware: a short's target sits BELOW the entry. The previous
        # unconditional ``* (1 + rate)`` would have armed a short whose "target" was
        # above its entry — i.e. a position that takes profit only when losing.
        target_price = (
            _directional_target_price(
                entry_price, proposal.target_return_rate, proposal.direction
            )
            if entry_price and entry_price > 0
            else None
        )
        self._state = StrategySessionState(
            session_id=f"session-{uuid4().hex}",
            phase="ARMED",
            selected_symbol=proposal.symbol,
            selected_strategy=proposal.strategy_id,
            selected_direction=str(proposal.direction),
            selected_execution_product=str(proposal.execution_product),
            selected_deployment_state=str(proposal.deployment_state),
            loan_date=None,
            borrow_fee_bps_annualised=(
                getattr(proposal.borrow_snapshot, "borrow_fee_bps_annualised", None)
                if proposal.borrow_snapshot is not None
                else None
            ),
            borrow_reference=(
                getattr(proposal.borrow_snapshot, "snapshot_id", None)
                if proposal.borrow_snapshot is not None
                else None
            ),
            return_deadline=_iso(
                getattr(proposal.borrow_snapshot, "return_deadline", None)
                if proposal.borrow_snapshot is not None
                else None
            ),
            directional_comparison=dict(self._state.directional_comparison),
            selection_source=proposal.source,
            selection_score=proposal.score,
            selection_confidence=proposal.confidence,
            selected_at=_iso(now),
            entry_price=entry_price,
            target_price=target_price,
            target_return_rate=proposal.target_return_rate,
            stop_loss_rate=proposal.stop_loss_rate,
            trailing_stop_rate=proposal.trailing_stop_rate,
            max_holding_seconds=proposal.max_holding_seconds,
            last_evaluated_at=_iso(now),
            last_reason=proposal.last_reason,
            macro_regime=proposal.macro_regime or self._state.macro_regime,
            micro_regime=proposal.micro_regime or None,
            ontology_reason_codes=list(proposal.ontology_reason_codes),
            gnn_action=proposal.gnn_action,
            gnn_reason_codes=list(proposal.gnn_reason_codes),
            explanation_paths=list(proposal.explanation_paths),
            candidate_diagnostics=list(self._state.candidate_diagnostics),
            algorithm_evaluations=list(self._state.algorithm_evaluations),
            bandit_selected_arm=self._state.bandit_selected_arm,
            bandit_conservative_edge_bps=proposal.conservative_edge_bps,
            bandit_is_exploration=self._state.bandit_is_exploration,
            bandit_reason_codes=list(self._state.bandit_reason_codes),
            bandit_evaluations=list(self._state.bandit_evaluations),
            bandit_shadow_arms=list(self._state.bandit_shadow_arms),
            bandit_evaluated_at=self._state.bandit_evaluated_at,
            cost_coverage_ratio=coverage.ratio,
            cost_coverage_band=coverage.band.value,
            change_point_probability=self._state.change_point_probability,
            session_phases=dict(self._state.session_phases),
            expected_cost_bps=proposal.resolved_cost_bps(
                self.config.fallback_round_trip_cost_bps
            ),
            expected_net_return_bps=proposal.expected_net_return_bps,
            election_context=self._election_context(
                proposal.strategy_id,
                now,
                intent=proposal.intent,
                candidate_count=proposal.candidate_count,
                micro_result=proposal.micro_result,
                evidence_row=proposal.evidence_row,
                macro=macro,
                symbol=proposal.symbol,
                change_point_probability=self._state.change_point_probability,
                borrow_snapshot=proposal.borrow_snapshot,
            ),
        )
        # The plan is the durable, immutable output of this election. Building it here —
        # inside _arm, with the account in hand — is what moves cost, sizing and risk in
        # front of the selection instead of behind it. A failure to build one is not
        # fatal to the election: the session still owns the symbol, and the legacy
        # evaluate_buy path (which still runs its own gates) handles it.
        self._trade_plan = self._build_trade_plan(proposal, now, account=account)
        if self._trade_plan is not None:
            self._state.trade_plan_id = self._trade_plan.plan_id
            self._state.trade_plan_quantity = self._trade_plan.quantity
            self._state.trade_plan_expires_at = _iso(self._trade_plan.expires_at)
        return True

    def _no_election_reason(
        self, evidence: Mapping[str, Any], intents: list[Any]
    ) -> str:
        if self.config.algorithm_primary_election and self._state.algorithm_evaluations:
            if not any(
                bool(item.get("triggered"))
                for item in self._state.algorithm_evaluations
            ):
                return "NO_MECHANICAL_STRATEGY_TRIGGER"
            # A trigger existed but could not become an admissible proposal (for
            # example deployment or borrow authority). Do not blame GNN state for
            # a deterministic gate's decision.
            return "NO_ADMISSIBLE_TRIGGERED_STRATEGY"
        model_trust_ready = any(
            "GNN_REALTIME_MODEL_TRUST_PASSED"
            in tuple(decision.get("reason_codes") or ())
            for row in (
                evidence.values()
                if isinstance(evidence, Mapping)
                else ()
            )
            if isinstance(row, Mapping)
            for decision in tuple(row.get("decisions") or ())
            if isinstance(decision, Mapping)
            and decision.get("path") == "cpu_gnn"
        )
        positive_edge_awaiting_validation = any(
            is_actionable_strategy_route(decision.get("action"))
            and "GNN_REALTIME_MODEL_TRUST_PASSED"
            in tuple(decision.get("reason_codes") or ())
            and "GNN_REALTIME_TRUST_PASSED"
            not in tuple(decision.get("reason_codes") or ())
            for row in (
                evidence.values()
                if isinstance(evidence, Mapping)
                else ()
            )
            if isinstance(row, Mapping)
            for decision in tuple(row.get("decisions") or ())
            if isinstance(decision, Mapping)
            and decision.get("path") == "cpu_gnn"
        )
        # When the operator asked for a live GNN, its state is the most informative
        # thing to report — that stays true with the bandit on, where an untrusted
        # GNN is a penalty rather than a veto but is still why nothing was elected.
        if self.config.require_live_gnn:
            if positive_edge_awaiting_validation:
                return "GNN_POSITIVE_EDGE_AWAITING_ENTRY_VALIDATION"
            return (
                "NO_POSITIVE_NET_GNN_EDGE" if model_trust_ready else "GNN_NOT_LIVE_AUTHORIZED"
            )
        return (
            "NO_FRESH_STRATEGY_ELECTION"
            if not intents
            else "NO_ADMISSIBLE_STRATEGY_PROPOSAL"
        )

    @staticmethod
    def _election_context(
        strategy_id: str,
        now: datetime,
        *,
        intent: Any = None,
        candidate_count: int | None = None,
        micro_result: Any = None,
        evidence_row: Mapping[str, Any] | None = None,
        macro: Any = None,
        symbol: str = "",
        change_point_probability: float | None = None,
        borrow_snapshot: Any = None,
    ) -> dict[str, Any]:
        """Freeze the slow context the algorithm will consume.

        Only facts the electing layer actually resolved are written. Event
        freshness comes from classified issuer evidence and opening-gap
        references come from completed regular-session bars. Either remains
        absent when its point-in-time producer has no measurement, so the
        consuming strategy still fails closed rather than assuming a value.
        """
        context: dict[str, Any] = {
            "strategy_id": strategy_id,
            "elected_at": _iso(now),
        }
        if change_point_probability is not None:
            context["change_point_probability"] = float(change_point_probability)
        if intent is not None:
            entry = float(getattr(intent, "expected_entry_price", 0.0) or 0.0)
            if entry > 0:
                context["reference_price"] = entry
            net_bps = getattr(intent, "expected_net_return_bps", None)
            if net_bps is not None:
                context["expected_net_return_bps"] = float(net_bps)
        # Real WITHIN-SECTOR rank on residual (market/sector-neutral) return.
        #
        # This used to be the arbiter's global BUY rank ordered by expected net
        # return, which is not a sector rank at all: a name could be "rank 1" while
        # being the weakest stock in its own sector, so the relative-strength
        # thesis was never actually tested. When the macro layer cannot resolve a
        # sector (or the sector has one tracked name) the rank stays ABSENT and the
        # consuming algorithm fails closed, which is the honest outcome.
        table = getattr(macro, "sector_rank_table", None)
        normalized_symbol = str(symbol or "").upper()
        if table is not None and normalized_symbol:
            try:
                ranked = table.rank_for(normalized_symbol)
            except Exception:  # noqa: BLE001 - a ranking lookup must never crash election.
                ranked = None
            if ranked is not None:
                context["sector_rank"] = int(ranked[0])
                context["sector_candidate_count"] = int(ranked[1])
                sector = dict(getattr(table, "sector_of", {}) or {}).get(normalized_symbol)
                if sector:
                    context["sector"] = str(sector)
            residual = None
            long_residual = None
            beta = None
            try:
                residual = table.residual_for(normalized_symbol)
                long_residual = table.long_residual_for(normalized_symbol)
                beta = table.beta_for(normalized_symbol)
            except Exception:  # noqa: BLE001
                residual = long_residual = beta = None
            # Residuals are rates over the macro trend windows; algorithms consume
            # bps. Each is written only when actually measured, so a strategy that
            # needs both horizons fails closed on a short data window instead of
            # seeing one window twice.
            if residual is not None:
                context["residual_return_short_bps"] = float(residual) * 10_000.0
            if long_residual is not None:
                context["residual_return_long_bps"] = float(long_residual) * 10_000.0
            if beta is not None:
                context["market_beta"] = float(beta)
        if macro is not None:
            spread_percentile = getattr(macro, "spread_percentile", None)
            if spread_percentile is not None:
                context["spread_percentile"] = float(spread_percentile)
        diagnostics = getattr(micro_result, "diagnostics", None)
        if isinstance(diagnostics, Mapping):
            age = diagnostics.get("event_age_seconds")
            ttl = diagnostics.get("event_ttl_seconds")
            if age is not None and ttl is not None:
                context["event_fresh"] = True
                context["event_age_seconds"] = float(age)
                context["event_ttl_seconds"] = float(ttl)
            for key in ("gap_rate", "gap_submode", "session_open_price", "previous_close_price"):
                value = diagnostics.get(key)
                if value is not None:
                    context[key] = value
        rvgi_context = (
            evidence_row.get("rvgi_box_context")
            if isinstance(evidence_row, Mapping)
            else None
        )
        if strategy_id == "rvgi_box_breakout" and isinstance(rvgi_context, Mapping):
            for key in (
                "rvgi",
                "rvgi_signal",
                "rvgi_diff",
                "rvgi_bullish_cross",
                "box_high",
                "box_low",
                "box_mid",
                "box_width_pct",
                "box_position",
                "box_context_timestamp",
                "box_previous_close",
                "volume_confirmed",
            ):
                value = rvgi_context.get(key)
                if value is not None:
                    context[key] = value
        # Session structure for the two session-boxed strategies. Without this they
        # declared their context fields and nothing ever populated them, so both
        # rejected every tick with *_CONTEXT_ABSENT — implemented, registered, and
        # permanently inert. The clock is authoritative here; only the price-derived
        # parts need a producer, and those stay absent when unavailable.
        if strategy_id in {
            "market_intraday_momentum",
            "opening_range_breakout",
            # Both legs of each session-boxed thesis read the same clock facts.
            "market_intraday_momentum_short",
            "opening_range_breakdown",
            # The carry is entered from the same clock facts, one session later.
            "overnight_gap_carry",
        }:
            context.update(_session_structure_context(now, symbol))
            if isinstance(diagnostics, Mapping):
                for key in (
                    "opening_range_high",
                    "opening_range_low",
                    "opening_range_minutes",
                    "relative_volume",
                    "first_half_hour_return_bps",
                    "first_half_hour_volatility_percentile",
                ):
                    value = diagnostics.get(key)
                    if value is not None:
                        context[key] = value

        # --- SHORT context ---------------------------------------------------- #
        # Without this block every short algorithm fails closed on its borrow
        # preconditions and can never fire — the strategies would be registered,
        # evaluated, and permanently inert, which is the exact defect the long
        # session-boxed strategies already hit once (declared context fields that
        # nothing populated).
        if is_short_strategy(strategy_id):
            context.update(
                _short_election_context(
                    symbol=normalized_symbol,
                    borrow_snapshot=borrow_snapshot,
                    macro=macro,
                    micro_diagnostics=diagnostics,
                    table=table,
                    now=now,
                )
            )
        return context

    def _apply_owned_exit_geometry(self, entry_price: float) -> None:
        """Resolve the elected algorithm's structural exit rule at fill time.

        Applies to EVERY elected strategy. It used to special-case
        ``rvgi_box_breakout`` and give everything else a flat percentage target,
        which silently discarded each algorithm's own stop/target structure — the
        docstring already claimed the general behaviour the code did not implement.

        The concrete hazard that exposed it: ``market_intraday_momentum`` shrinks its
        holding horizon toward the KRX continuous close so the position is flat
        before the 15:20 single-price auction. With the rule uncalled, that horizon
        never reached the session state and the position could be carried into the
        auction — the exact outcome its unit tests forbid.
        """
        state = self._state
        direction = parse_direction(state.selected_direction)
        if not state.selected_strategy or entry_price <= 0:
            state.target_price = _directional_target_price(
                entry_price, state.target_return_rate, direction
            )
            return
        try:
            from app.technical.signals import TechnicalFeatureSet
            from app.technical.strategy_algorithms import ElectionContext, get_algorithm

            algorithm = get_algorithm(
                state.selected_strategy, registry=self._algorithm_registry
            )
            if algorithm is None:
                return
            allowed = ElectionContext.__dataclass_fields__.keys()
            payload = {
                key: value
                for key, value in state.election_context.items()
                if key in allowed and key != "elected_at"
            }
            payload["strategy_id"] = state.selected_strategy
            rule = algorithm.exit_rule(
                entry_price,
                TechnicalFeatureSet(symbol=state.selected_symbol or "", price=entry_price),
                ElectionContext(**payload),
            )
            # Only ADOPT values the rule actually resolved. The rule is built from a
            # bare feature set (no realized volatility at fill time), so a
            # volatility-derived stop comes back as None for most algorithms —
            # assigning it blindly would DELETE the stop and leave the position
            # unprotected. The exit-geometry table stays authoritative unless the
            # algorithm produced something better.
            if rule.stop_price and rule.stop_price > 0:
                state.stop_price = rule.stop_price
            # The algorithm's target is on its own direction's favourable side, and
            # ``target_return_rate`` is a positive magnitude. Comparing
            # ``rule.target_price > entry_price`` for a short would read its correct
            # (lower) target as "no target" and fall back to the geometry floor.
            algorithm_target_rate = 0.0
            if rule.target_price and rule.target_price > 0:
                favourable = (
                    rule.target_price > entry_price
                    if direction is PositionDirection.LONG
                    else rule.target_price < entry_price
                )
                if favourable:
                    algorithm_target_rate = abs(rule.target_price / entry_price - 1.0)
            state.target_return_rate = max(
                state.target_return_rate,
                algorithm_target_rate,
            )
            state.target_price = _directional_target_price(
                entry_price, state.target_return_rate, direction
            )
            if rule.trailing_bps and float(rule.trailing_bps) > 0:
                state.trailing_stop_rate = float(rule.trailing_bps) / 10_000.0
            # A horizon may only be SHORTENED here. Lengthening it would let an
            # algorithm quietly overrule the table's holding limit; shortening is how
            # a session-boxed thesis (flat before the 15:20 auction) is enforced.
            rule_holding = int(rule.max_holding_seconds or 0)
            if 0 < rule_holding < int(state.max_holding_seconds or rule_holding):
                state.max_holding_seconds = rule_holding
        except Exception:  # noqa: BLE001 - persisted fallback exits remain authoritative.
            state.target_price = _directional_target_price(
                entry_price, state.target_return_rate, direction
            )

    def _fresh_evidence(self, row: Mapping[str, Any], now: datetime) -> bool:
        observed = _parse_time(row.get("as_of"))
        if observed is None:
            return False
        return abs((now - observed).total_seconds()) <= self.config.selection_evidence_max_age_seconds

    def _adopt_existing_position(
        self, holdings: Mapping[str, Any], bundle: Any, now: datetime
    ) -> None:
        symbol, holding = next(iter(holdings.items()))
        average = float(getattr(holding, "average_price", 0.0) or 0.0)
        memo = self._owned_position_memo_for(symbol, average)
        strategy = "risk_managed_existing_position"
        for result in tuple(getattr(bundle, "micro_results", ()) or ()):
            if str(getattr(result, "symbol", "") or "").upper() == symbol:
                # The micro reasoner emits an ACTION, and ``hold``/``sell``/
                # ``reduce_risk`` are not theses. Taking them verbatim named the
                # session's strategy ``hold``, which resolves to no geometry row and
                # silently re-armed the lot on the unknown-thesis fallback — a 60bps
                # stop and a 1200s clock in place of the rule it was entered under.
                resolved = resolve_strategy_id(
                    getattr(getattr(result, "selected_strategy", None), "value", None)
                    or getattr(result, "selected_strategy", None)
                )
                if resolved:
                    strategy = resolved
                break
        if memo:
            # This is the same lot we were already managing. Its own thesis outranks
            # anything re-derived from a bare balance row.
            strategy = str(memo.get("strategy_id") or strategy)
        opened = memo.get("position_opened_at") if memo else None
        opened = _parse_time(opened) or getattr(holding, "opened_at", None) or now
        stop_bps, profit_bps, trailing_bps = _strategy_exit_bps(strategy)
        target_rate = max(
            self.config.fallback_target_return_rate,
            profit_bps / 10_000.0,
        )
        # Direction comes from the BROKER. An adopted short misread as a long would be
        # managed with an inverted stop and target — exiting on the winning side and
        # running on the losing one, with an unbounded downside.
        direction = parse_direction(getattr(holding, "direction", None))
        last_price = float(getattr(holding, "last_price", 0.0) or 0.0)
        loan_date = getattr(holding, "loan_date", None)
        self._state = StrategySessionState(
            session_id=f"session-{uuid4().hex}",
            phase="OWNED",
            selected_symbol=symbol,
            selected_strategy=strategy,
            selected_direction=str(direction),
            selected_execution_product=str(
                getattr(holding, "execution_product", None) or default_product(direction)
            ),
            selection_source="BROKER_POSITION_RECONCILIATION",
            selected_at=_iso(now),
            position_opened_at=_iso(opened),
            position_seen=True,
            entry_price=average or None,
            target_price=(
                _directional_target_price(average, target_rate, direction)
                if average
                else None
            ),
            target_return_rate=target_rate,
            stop_loss_rate=stop_bps / 10_000.0,
            trailing_stop_rate=trailing_bps / 10_000.0,
            high_watermark_price=(
                (max(average, last_price) or None)
                if direction is PositionDirection.LONG
                else None
            ),
            low_watermark_price=(
                None
                if direction is PositionDirection.LONG
                else (favourable_watermark(average or None, last_price or average, direction) or None)
            ),
            loan_date=str(loan_date) if loan_date else None,
            borrow_fee_bps_annualised=_optional_float(
                getattr(holding, "borrow_fee_rate", None)
            ),
            return_deadline=_iso(getattr(holding, "return_deadline", None)),
            max_holding_seconds=_strategy_max_holding_seconds(strategy),
            last_evaluated_at=_iso(now),
            last_reason=(
                "EXISTING_MULTIPLE_HOLDINGS_BUYS_BLOCKED"
                if len(holdings) > 1
                # An adopted SHORT with no loan date cannot be covered through the
                # 매수상환 contract. Flagged in the reason so the operator and the
                # promotion controller both see it, rather than discovering it when
                # the exit order is rejected.
                else (
                    "EXISTING_SHORT_ADOPTED_WITHOUT_LOAN_DATE"
                    if direction is PositionDirection.SHORT and not loan_date
                    else "EXISTING_POSITION_ADOPTED"
                )
            ),
        )
        self._state.position_last_seen_at = _iso(now)
        self._state.owned_position_memo = dict(memo) if memo else {}
        if memo:
            self._restore_owned_position(memo, last_price=last_price, direction=direction)

    def _owned_position_memo_for(
        self, symbol: str, average_price: float
    ) -> dict[str, Any]:
        """The frozen thesis for this exact lot, or ``{}``.

        Identity is (symbol, average price). A different average price is a
        different lot — a partial fill, an add, or a position opened outside this
        system — and inheriting the previous trade's barriers and entry clock would
        manage it against a thesis it was never entered under.
        """
        memo = self._state.owned_position_memo or {}
        if not isinstance(memo, Mapping) or not memo:
            return {}
        if str(memo.get("symbol") or "").upper() != str(symbol or "").upper():
            return {}
        remembered = _optional_float(memo.get("entry_price")) or 0.0
        if remembered <= 0.0 or average_price <= 0.0:
            return {}
        if abs(average_price - remembered) > max(1e-9, remembered * 1e-4):
            return {}
        if not resolve_strategy_id(memo.get("strategy_id")):
            return {}
        return dict(memo)

    def _restore_owned_position(
        self,
        memo: Mapping[str, Any],
        *,
        last_price: float,
        direction: PositionDirection,
    ) -> None:
        """Put the armed exit contract back, exactly as the entry set it.

        Re-deriving it from the table would already be closer than the fallback, but
        it still would not be the same contract: the armed geometry was sized against
        the cost and spread measured for THIS trade, and the watermark and the entry
        clock have no table to be re-derived from at all.
        """
        state = self._state
        state.session_id = str(memo.get("session_id") or state.session_id)
        state.selection_source = str(
            memo.get("selection_source") or "BROKER_POSITION_RECONCILIATION"
        )
        state.selected_deployment_state = str(
            memo.get("selected_deployment_state") or state.selected_deployment_state
        )
        state.selected_execution_product = str(
            memo.get("selected_execution_product") or state.selected_execution_product
        )
        for key in (
            "target_price",
            "stop_price",
            "expected_cost_bps",
            "expected_net_return_bps",
        ):
            value = _optional_float(memo.get(key))
            if value is not None:
                setattr(state, key, value)
        for key in ("target_return_rate", "stop_loss_rate", "trailing_stop_rate"):
            value = _optional_float(memo.get(key))
            if value is not None and value > 0.0:
                setattr(state, key, value)
        holding_seconds = _optional_float(memo.get("max_holding_seconds"))
        if holding_seconds is not None and holding_seconds > 0:
            state.max_holding_seconds = int(holding_seconds)
        # The favourable extreme survives too. Reseeding it from the entry would
        # rearm a trailing stop the position had already moved past.
        if direction is PositionDirection.LONG:
            remembered = _optional_float(memo.get("high_watermark_price"))
            state.high_watermark_price = max(
                remembered or 0.0, state.high_watermark_price or 0.0, last_price
            ) or None
        else:
            state.low_watermark_price = favourable_watermark(
                _optional_float(memo.get("low_watermark_price"))
                or state.low_watermark_price,
                last_price or (state.entry_price or 0.0),
                direction,
            ) or None
        # Keep a warning reason (multiple holdings, short without a loan date); only
        # the plain adoption line is replaced, because it is no longer the truth.
        if state.last_reason == "EXISTING_POSITION_ADOPTED":
            state.last_reason = "EXISTING_POSITION_READOPTED_WITH_ARMED_THESIS"

    def _reset_to_scanning(self, reason: str) -> None:
        # The memo outlives the reset on purpose. A reset that happens while the lot
        # is still held (cooldown after an unconfirmed flat, a supervisor halt, a
        # rejected exit) is exactly when re-adoption needs the thesis it is about to
        # forget. ``_adopt_existing_position`` discards it unless the same lot at the
        # same average price comes back.
        self._state = StrategySessionState(
            target_return_rate=self.config.fallback_target_return_rate,
            last_reason=reason,
            last_evaluated_at=self._state.last_evaluated_at,
            owned_position_memo=dict(self._state.owned_position_memo or {}),
        )

    def _load(self) -> StrategySessionState:
        path = Path(self.config.state_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            allowed = StrategySessionState.__dataclass_fields__.keys()
            restored = StrategySessionState(
                **{key: value for key, value in raw.items() if key in allowed}
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return StrategySessionState(target_return_rate=self.config.fallback_target_return_rate)
        return self._refresh_idle_exit_geometry(restored)

    @staticmethod
    def _refresh_idle_exit_geometry(state: StrategySessionState) -> StrategySessionState:
        """Re-derive the exit geometry when no thesis is active.

        With a position open these four fields are real state -- the geometry the
        live trade was actually armed with -- and must survive a restart exactly
        as persisted. With nothing selected they are only placeholders, and
        restoring them from disk made the state file outrank the geometry table:
        after the table moved to 60/160, a months-old file kept every dashboard
        reporting 22/40 across restarts, which is precisely the stale number that
        caused a target to be misread as unable to clear costs.
        """
        if state.selected_strategy:
            return state
        geometry = _fallback_exit_geometry()
        state.target_return_rate = geometry.take_profit_bps / 10_000.0
        state.stop_loss_rate = geometry.stop_loss_bps / 10_000.0
        state.trailing_stop_rate = geometry.trailing_bps / 10_000.0
        state.max_holding_seconds = geometry.max_holding_seconds
        return state

    def _persist(self) -> None:
        path = Path(self.config.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self._state), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
