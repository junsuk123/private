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
from app.strategy.catalog import is_known_strategy, is_short_strategy, resolve_strategy_id
from app.strategy.exit_geometry import FALLBACK_GEOMETRY_KEY
from app.strategy.exit_geometry import exit_bps as _strategy_exit_bps
from app.strategy.exit_geometry import exit_geometry as _exit_geometry
from app.strategy.exit_geometry import max_holding_seconds as _strategy_max_holding_seconds
from app.trading.conservative_bandit import (
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


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


_KST = timezone(timedelta(hours=9))
# KRX continuous trading ends at 15:20; 15:20-15:30 is a closing single-price
# auction with different matching, which this system does not model.
_KRX_CONTINUOUS_CLOSE_MINUTE = 15 * 60 + 20
_KRX_LAST_CONTINUOUS_HALF_HOUR_START = _KRX_CONTINUOUS_CLOSE_MINUTE - 30


def _session_structure_context(now: datetime) -> dict[str, Any]:
    """Clock-derived KRX session structure the session-boxed strategies need.

    Only the parts the clock can answer. ``in_last_continuous_half_hour`` and
    ``minutes_to_continuous_close`` are pure calendar facts, so withholding them
    would leave the strategies fail-closed for no reason; the price-derived fields
    (opening range, first-half-hour return) still require a producer and stay absent
    until one supplies them.
    """
    local = now.astimezone(_KST)
    minute_of_day = local.hour * 60 + local.minute
    remaining = (_KRX_CONTINUOUS_CLOSE_MINUTE - minute_of_day) - local.second / 60.0
    return {
        "in_last_continuous_half_hour": bool(
            _KRX_LAST_CONTINUOUS_HALF_HOUR_START <= minute_of_day < _KRX_CONTINUOUS_CLOSE_MINUTE
        ),
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
            target_bps = self.target_return_rate * 10_000.0
            if target_bps <= 0:
                return None
            edge = target_bps - self.resolved_cost_bps(fallback_cost_bps)
        if not self.gnn_actionable:
            edge -= max(0.0, float(gnn_absence_penalty_bps))
        return edge

    def predicted_gross_edge_bps(self, fallback_cost_bps: float = 0.0) -> float | None:
        if self.expected_net_return_bps is not None:
            return float(self.expected_net_return_bps) + self.resolved_cost_bps(
                fallback_cost_bps
            )
        if self.target_return_rate > 0:
            return self.target_return_rate * 10_000.0
        return None


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
    invalidation_confirm_cycles: int = field(
        default_factory=lambda: max(1, _env_int("STRATEGY_SESSION_INVALIDATION_CYCLES", 3))
    )
    require_live_gnn: bool = field(
        default_factory=lambda: os.getenv("STRATEGY_SESSION_REQUIRE_LIVE_GNN", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    selection_evidence_max_age_seconds: int = field(
        default_factory=lambda: max(
            10,
            _env_int("STRATEGY_SESSION_EVIDENCE_MAX_AGE_SEC", 120),
        )
    )
    # Conservative bandit. When enabled, election no longer means "arm the first
    # admissible candidate": every candidate is scored on a pessimistic net-edge
    # lower bound and NO_TRADE is a real, selectable outcome. This is what stops a
    # tape in which every strategy has negative expectancy from being traded with
    # the least-bad negative expectancy.
    bandit_enabled: bool = field(
        default_factory=lambda: os.getenv("STRATEGY_SESSION_BANDIT_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # When the bandit is enabled, an unavailable / untrusted GNN downgrades a
    # candidate instead of vetoing election outright: the GNN is one estimator
    # among several, and treating its absence as a refusal made the whole session
    # dark whenever the checkpoint went stale (which adding a strategy does).
    gnn_absence_penalty_bps: float = field(
        default_factory=lambda: max(0.0, _env_float("STRATEGY_SESSION_GNN_ABSENCE_PENALTY_BPS", 15.0))
    )
    # Record realized outcomes so the bandit has something to learn from.
    record_outcomes: bool = field(
        default_factory=lambda: os.getenv("STRATEGY_SESSION_RECORD_OUTCOMES", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # Round-trip cost assumed when the election evidence carried no estimate.
    # KRX round trip (sell tax + fees + spread) is ~25-30bps; 28 is the measured
    # per-strategy average in the R-GCN model card.
    fallback_round_trip_cost_bps: float = field(
        default_factory=lambda: max(
            0.0, _env_float("STRATEGY_SESSION_FALLBACK_COST_BPS", 28.0)
        )
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
    selected_at: str | None = None
    entry_submitted_at: str | None = None
    position_opened_at: str | None = None
    position_seen: bool = False
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
    exit_reason: str | None = None
    cooldown_until: str | None = None
    invalidation_cycles: int = 0
    last_evaluated_at: str | None = None
    last_reason: str = "WAITING_FOR_ONTOLOGY_SELECTION"
    macro_regime: str | None = None
    micro_regime: str | None = None
    ontology_reason_codes: list[str] = field(default_factory=list)
    gnn_action: str | None = None
    gnn_reason_codes: list[str] = field(default_factory=list)
    explanation_paths: list[dict[str, Any]] = field(default_factory=list)
    candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)
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


class StrategySessionManager:
    """Persistent closed-world ownership state machine for the live engine."""

    def __init__(
        self,
        *,
        config: StrategySessionConfig | None = None,
        selection_evidence_provider: Callable[[tuple[str, ...]], Mapping[str, Any]] | None = None,
        performance_store: StrategyPerformanceStore | None = None,
        bandit: ConservativeStrategyBandit | None = None,
    ) -> None:
        self.config = config or StrategySessionConfig()
        self.selection_evidence_provider = selection_evidence_provider
        self.performance_store = (
            performance_store if performance_store is not None else _default_performance_store()
        )
        self.bandit = bandit or ConservativeStrategyBandit(store=self.performance_store)
        self._lock = threading.RLock()
        self._state = self._load()
        # Shadow plans journaled this cycle, awaiting adoption by
        # ``ShadowEvaluationService``. Drained rather than accumulated so a cycle whose
        # plans nobody collects cannot grow without bound.
        self._pending_shadow_plans: list[Any] = []

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
                    self._select(candidates, macro_micro_bundle, now)

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
                self._state.exit_requested_at = self._state.exit_requested_at or _iso(now)
                self._state.last_reason = "EXIT_ORDER_SUBMITTED_AWAITING_FLAT"
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
            return payload

    def _reconcile_position(
        self, holdings: Mapping[str, Any], bundle: Any, now: datetime
    ) -> None:
        state = self._state
        selected = state.selected_symbol
        holding = holdings.get(selected or "")
        if holding is not None:
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
            return

        if state.position_seen and state.phase in {"OWNED", "EXITING"}:
            # The position just went flat, so this is the one moment the trade's
            # realized outcome is knowable. Record it before the state is reset:
            # without this the conservative bandit has no history to learn from and
            # every arm stays permanently cold.
            self._record_outcome(now)
            state.phase = "COOLDOWN"
            state.cooldown_until = _iso(now + timedelta(seconds=self.config.cooldown_seconds))
            state.last_reason = "POSITION_FLAT_RESELECTION_COOLDOWN"
            state.exit_requested_at = state.exit_requested_at or _iso(now)

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
        holding_seconds = (
            max(0.0, (now - opened).total_seconds()) if opened is not None else None
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
            recorded_at=now,
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

        reason: str | None = None
        if target_reached(last_price, state.target_price, direction):
            reason = "STRATEGY_PROFIT_TARGET"
        elif stop_breached(last_price, resolved_stop, direction):
            reason = "STRATEGY_STOP_LOSS"
        elif trailing_breached(last_price, resolved_trailing, average_price, direction):
            reason = "STRATEGY_TRAILING_STOP"
        # A borrow recall is an exit reason with no long-side analogue: the position
        # must be covered whether or not the price thesis still holds, and waiting for
        # a price barrier would hand the timing to the lender.
        elif direction is PositionDirection.SHORT and self._recall_imminent(now):
            reason = "STRATEGY_SHORT_BORROW_RECALL"

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

    def _select(self, candidates: tuple[str, ...], bundle: Any, now: datetime) -> None:
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
            self._state.last_reason = "WAITING_FOR_MACRO_MICRO_BUNDLE"
            return
        macro = getattr(bundle, "macro_result", None)
        macro_regime = getattr(getattr(macro, "market_regime", None), "value", None)
        self._state.macro_regime = str(macro_regime or "") or None
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
        proposals.extend(self._intent_proposals(intents, evidence, bundle, now))
        proposals.extend(
            self._evidence_proposals(tradable_candidates, evidence, bundle, now)
        )

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
        if proposals and self.config.bandit_enabled:
            selected = self._bandit_choice(proposals, macro, now)
            if selected is not None:
                self._arm(selected, now, macro)
            return
        if executable:
            # Bandit disabled: preserve the historical first-admissible behaviour,
            # over the order-authorised subset only.
            self._arm(executable[0], now, macro)
            return
        if proposals:
            self._state.last_reason = (
                f"{ShortReasonCodes.SHADOW_ONLY}:"
                f"{','.join(sorted({p.strategy_id for p in proposals}))}"
            )
            return

        self._state.last_reason = self._no_election_reason(evidence, intents)

    # -- proposal construction --------------------------------------------- #
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
            if entry_price > 0 and expected_exit > 0:
                favourable = (
                    expected_exit > entry_price
                    if direction is PositionDirection.LONG
                    else expected_exit < entry_price
                )
                if favourable:
                    target_rate = max(target_rate, abs(expected_exit / entry_price - 1.0))
            stop_bps, profit_bps, trailing_bps = _strategy_exit_bps(selected_strategy)
            profit_bps = _cost_aware_profit_bps(gnn, profit_bps)
            target_rate = max(target_rate, profit_bps / 10_000.0)
            micro_result = next(
                (
                    result
                    for result in tuple(getattr(bundle, "micro_results", ()) or ())
                    if str(getattr(result, "symbol", "") or "").upper() == symbol
                ),
                None,
            )
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
                    max_holding_seconds=_strategy_max_holding_seconds(selected_strategy),
                    score=float(getattr(intent, "score", 0.0) or 0.0),
                    confidence=float(getattr(intent, "confidence", 0.0) or 0.0),
                    expected_net_return_bps=_optional_float(
                        getattr(intent, "expected_net_return_bps", None)
                    ),
                    expected_cost_bps=_optional_float(gnn.get("expected_cost_bps")),
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
        """Shadow-evidence path: an explicit ACTIVATE_STRATEGY or a trusted GNN.

        Election and entry timing stay separate responsibilities. A fresh
        admissibility decision may arm a strategy whose tick trigger is not ready;
        the owned strategy executor then waits in ARMED without placing an order.
        """
        proposals: list[_ElectionProposal] = []
        claimed = set()
        for symbol in candidates:
            normalized = str(symbol or "").upper()
            if normalized in claimed:
                continue
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
            )
            if not gnn_actionable and (
                self.config.require_live_gnn and not self.config.bandit_enabled
            ):
                continue
            if not gnn_actionable and not ontology_actionable:
                continue
            selected_strategy = (
                gnn_strategy if gnn_actionable else ontology_strategy
            )
            if not selected_strategy:
                continue
            direction, product, deployment_state, borrow_snapshot, borrow_reasons = (
                self._resolve_direction_context(selected_strategy, normalized, now)
            )
            if deployment_state is StrategyDeploymentState.DISABLED:
                self._state.last_reason = (
                    f"{ShortReasonCodes.DEPLOYMENT_DISABLED}:{selected_strategy}"
                )
                continue
            # Preserve non-authorised and no-locate arms as SHADOW evidence.
            # Order capability is derived separately by ``submits_orders``.
            # The shadow evidence path does not know the macro regime, so it
            # would happily arm a strategy family the macro layer has blocked.
            # The supervisor would then flag it every cycle; refuse it here.
            if _macro_permits(bundle, selected_strategy) is False:
                self._state.last_reason = (
                    f"MACRO_BLOCKS_ELECTED_STRATEGY:{selected_strategy}"
                )
                continue
            stop_bps, profit_bps, trailing_bps = _strategy_exit_bps(selected_strategy)
            profit_bps = _cost_aware_profit_bps(gnn, profit_bps)
            claimed.add(normalized)
            proposals.append(
                _ElectionProposal(
                    symbol=normalized,
                    strategy_id=selected_strategy,
                    source=(
                        "GNN_STRATEGY_ELECTION"
                        if gnn_actionable
                        else "ONTOLOGY_STRATEGY_ELECTION"
                    ),
                    entry_price=None,
                    target_return_rate=max(
                        self.config.fallback_target_return_rate,
                        profit_bps / 10_000.0,
                    ),
                    stop_loss_rate=stop_bps / 10_000.0,
                    trailing_stop_rate=trailing_bps / 10_000.0,
                    max_holding_seconds=_strategy_max_holding_seconds(selected_strategy),
                    score=0.0,
                    confidence=0.0,
                    expected_net_return_bps=_optional_float(gnn.get("expected_net_return_bps")),
                    expected_cost_bps=_optional_float(gnn.get("expected_cost_bps")),
                    gnn_actionable=gnn_actionable,
                    gnn_action=gnn_action,
                    gnn_reason_codes=gnn_reason_codes,
                    ontology_reason_codes=list(ontology.get("reason_codes") or ()),
                    macro_regime=self._state.macro_regime or "",
                    micro_regime="",
                    explanation_paths=[],
                    intent=None,
                    candidate_count=None,
                    micro_result=None,
                    evidence_row=row,
                    last_reason="STRATEGY_ELECTED_WAITING_FOR_ENTRY_TRIGGER",
                    direction=direction,
                    position_effect=PositionEffect.OPEN,
                    execution_product=product,
                    deployment_state=deployment_state,
                    borrow_snapshot=borrow_snapshot,
                    borrow_reason_codes=borrow_reasons,
                )
            )
        return proposals

    @staticmethod
    def _deployment_authorized(strategy_id: str) -> tuple[bool, str]:
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

            if strategy_live_authorized(strategy_id):
                return True, ""
        except Exception:  # noqa: BLE001 - a lookup failure must fail closed.
            return False, f"STRATEGY_AUTHORIZATION_UNAVAILABLE:{strategy_id}"
        return False, f"STRATEGY_NOT_LIVE_AUTHORIZED:{strategy_id}"

    def _directional_deployment_state(
        self, strategy_id: str, direction: PositionDirection, market: str
    ) -> StrategyDeploymentState:
        """Committed deployment state for one arm.

        LONG arms keep the pre-existing behaviour exactly: the per-strategy
        ``live_authorized`` flag maps to LIVE_FULL or SHADOW, so nothing about the
        long path changes. SHORT arms are looked up per-arm in the promotion store.

        Any failure resolves to SHADOW. An unreadable deployment state must never
        authorise an order, and for a short that is the difference between a journal
        entry and a borrowed position.
        """
        if direction is PositionDirection.LONG:
            authorized, _ = self._deployment_authorized(strategy_id)
            return (
                StrategyDeploymentState.LIVE_FULL
                if authorized
                else StrategyDeploymentState.SHADOW
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
        self, proposals: list["_ElectionProposal"], now: datetime
    ) -> None:
        """Write a :class:`ShadowTradePlan` for every non-order-authorised proposal.

        This is the mechanism by which a SHADOW short earns its way to LIVE_PROBE, and
        the reason a SHADOW arm is evaluated at all rather than skipped. Three
        properties are load-bearing:

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
            if proposal.submits_orders:
                continue
            entry_reference = proposal.entry_price
            if not entry_reference or entry_reference <= 0:
                # Without a point-in-time entry reference there is nothing to measure
                # a return against, and inventing one from a later quote is the leak
                # this whole module exists to prevent.
                continue
            try:
                plan = ShadowTradePlan(
                    plan_id="",
                    key=proposal.directional_key(market_for_symbol(proposal.symbol)),
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
                    signal_reason_codes=tuple(proposal.ontology_reason_codes),
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
            self._state.shadow_plan_ids = recorded[:16]
        # Hand the plan objects to the evaluation service so its barrier walk starts on
        # the NEXT quote. Passing objects rather than ids keeps the frozen borrow
        # snapshot intact — a re-read from the journal would have to re-resolve it, and
        # that is the one place a fresher locate could sneak in.
        if pending:
            self._pending_shadow_plans = pending

    # -- bandit scoring ---------------------------------------------------- #
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
                f"{ShortReasonCodes.SHADOW_ONLY}:{winner.strategy_id}"
            )
            return None
        winner.conservative_edge_bps = selection.conservative_edge_bps
        return winner

    def _arm(self, proposal: "_ElectionProposal", now: datetime, macro: Any = None) -> None:
        """Commit one proposal to ARMED, preserving the audit trail."""
        coverage = evaluate_cost_coverage(
            proposal.predicted_gross_edge_bps(self.config.fallback_round_trip_cost_bps),
            proposal.resolved_cost_bps(self.config.fallback_round_trip_cost_bps),
        )
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

    def _no_election_reason(
        self, evidence: Mapping[str, Any], intents: list[Any]
    ) -> str:
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
        freshness and opening-gap references have no point-in-time source in
        this repository yet, so they stay absent and ``event_momentum`` /
        ``gap_context`` fail closed instead of trading on an assumed value.
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
        }:
            context.update(_session_structure_context(now))
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

            algorithm = get_algorithm(state.selected_strategy)
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
        strategy = "risk_managed_existing_position"
        for result in tuple(getattr(bundle, "micro_results", ()) or ()):
            if str(getattr(result, "symbol", "") or "").upper() == symbol:
                strategy = str(getattr(getattr(result, "selected_strategy", None), "value", "") or strategy)
                break
        average = float(getattr(holding, "average_price", 0.0) or 0.0)
        opened = getattr(holding, "opened_at", None) or now
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

    def _reset_to_scanning(self, reason: str) -> None:
        self._state = StrategySessionState(
            target_return_rate=self.config.fallback_target_return_rate,
            last_reason=reason,
            last_evaluated_at=self._state.last_evaluated_at,
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
