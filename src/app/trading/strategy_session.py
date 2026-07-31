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
from app.strategy.catalog import is_known_strategy
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
from app.trading.strategy_performance_store import (
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
        }
    try:
        from app.data.market_session import allows_new_entry, market_phase
    except Exception:  # noqa: BLE001 - never let a session lookup break election.
        return {"allows_new_entry": True, "reason": "", "phases": {}}
    phases = {group: market_phase(group, now).value for group in groups}
    open_groups = tuple(group for group in groups if allows_new_entry(group, now))
    if open_groups:
        return {"allows_new_entry": True, "reason": "", "phases": phases}
    detail = ",".join(f"{group}={phase}" for group, phase in sorted(phases.items()))
    return {
        "allows_new_entry": False,
        "reason": f"NEW_ENTRY_OUTSIDE_REGULAR_SESSION:{detail}",
        "phases": phases,
    }


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

    def allowed_buy_candidates(self, candidates: tuple[str, ...], account: Any) -> tuple[str, ...]:
        with self._lock:
            if not self.config.enabled:
                return candidates
            if tuple(getattr(account, "holdings", ()) or ()):
                return ()
            if self._state.phase != "ARMED" or not self._state.selected_symbol:
                return ()
            # Ownership is locked for the complete ARMED entry window.  A
            # discovery list changing on the next cycle must not silently drop
            # the elected symbol before its strategy can evaluate the trigger.
            return (self._state.selected_symbol,)

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
                if state.entry_price:
                    self._apply_owned_exit_geometry(state.entry_price)
                    state.high_watermark_price = max(
                        state.entry_price,
                        float(getattr(holding, "last_price", 0.0) or 0.0),
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

    def _record_outcome(self, now: datetime) -> None:
        """Persist the realized net outcome of the closed position.

        Uses the last observed mark as the exit reference and subtracts the same
        round-trip cost estimate the election used, so ``realized_net_bps`` is
        directly comparable with the ``expected_net_bps`` that armed the trade —
        which is what makes the stored prediction error meaningful.
        """
        state = self._state
        if not self.config.record_outcomes or state.outcome_recorded:
            return
        strategy = state.selected_strategy
        symbol = state.selected_symbol
        entry = state.entry_price
        exit_price = state.exit_price or state.high_watermark_price
        if not strategy or not symbol or not entry or entry <= 0 or not exit_price:
            return
        gross_bps = (float(exit_price) / float(entry) - 1.0) * 10_000.0
        cost_bps = (
            state.expected_cost_bps
            if state.expected_cost_bps is not None
            else self.config.fallback_round_trip_cost_bps
        )
        opened = _parse_time(state.position_opened_at)
        holding_seconds = (
            max(0.0, (now - opened).total_seconds()) if opened is not None else None
        )
        recorded = self.performance_store.record(
            strategy_id=strategy,
            symbol=symbol,
            market=market_for_symbol(symbol),
            regime=state.macro_regime or "UNKNOWN",
            realized_net_bps=gross_bps - float(cost_bps),
            realized_gross_bps=gross_bps,
            expected_net_bps=state.expected_net_return_bps,
            holding_seconds=holding_seconds,
            exit_reason=state.exit_reason or "",
            recorded_at=now,
        )
        state.outcome_recorded = bool(recorded)

    def _evaluate_exit(self, holding: Any, bundle: Any, now: datetime) -> None:
        state = self._state
        if state.phase == "EXITING":
            return
        symbol = state.selected_symbol or ""
        last_price = float(getattr(holding, "last_price", 0.0) or 0.0)
        average_price = float(getattr(holding, "average_price", 0.0) or 0.0)
        quantity = max(0, int(getattr(holding, "quantity", 0) or 0))
        pnl = quantity * (last_price - average_price)
        state.target_profit_amount = (
            quantity * max(0.0, float(state.target_price or 0.0) - average_price)
            if average_price > 0
            else None
        )

        state.high_watermark_price = max(
            float(state.high_watermark_price or 0.0),
            last_price,
            average_price,
        )
        stop_price = (
            float(state.stop_price)
            if state.stop_price
            else average_price * (1.0 - state.stop_loss_rate)
            if average_price > 0
            else 0.0
        )
        trailing_price = (
            state.high_watermark_price * (1.0 - state.trailing_stop_rate)
            if state.high_watermark_price
            else 0.0
        )

        reason: str | None = None
        if state.target_price and last_price >= state.target_price:
            reason = "STRATEGY_PROFIT_TARGET"
        elif stop_price > 0 and last_price <= stop_price:
            reason = "STRATEGY_STOP_LOSS"
        elif (
            trailing_price > average_price
            and last_price <= trailing_price
        ):
            reason = "STRATEGY_TRAILING_STOP"

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

        proposals: list[_ElectionProposal] = []
        proposals.extend(self._intent_proposals(intents, evidence, bundle, now))
        proposals.extend(self._evidence_proposals(candidates, evidence, bundle, now))

        if proposals and self.config.bandit_enabled:
            selected = self._bandit_choice(proposals, macro, now)
            if selected is not None:
                self._arm(selected, now, macro)
            return
        if proposals:
            # Bandit disabled: preserve the historical first-admissible behaviour.
            self._arm(proposals[0], now, macro)
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
            if entry_price > 0 and expected_exit > entry_price:
                target_rate = max(target_rate, expected_exit / entry_price - 1.0)
            rvgi_context = (
                row.get("rvgi_box_context")
                if isinstance(row, Mapping)
                else None
            )
            selected_strategy = gnn_strategy if gnn_actionable else ontology_strategy
            if (
                not gnn_actionable
                and ontology_strategy in {"breakout", "rvgi_box_breakout"}
                and isinstance(rvgi_context, Mapping)
                and rvgi_context.get("ontology_eligible") is True
            ):
                selected_strategy = "rvgi_box_breakout"
            authorized, authorization_reason = self._deployment_authorized(selected_strategy)
            if not authorized:
                self._state.last_reason = authorization_reason
                continue
            if _macro_permits(bundle, selected_strategy) is False:
                self._state.last_reason = (
                    f"MACRO_BLOCKS_ELECTED_STRATEGY:{selected_strategy}"
                )
                continue
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
            authorized, authorization_reason = self._deployment_authorized(selected_strategy)
            if not authorized:
                self._state.last_reason = authorization_reason
                continue
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
                )
            )
        return proposals

    @staticmethod
    def _deployment_authorized(strategy_id: str) -> tuple[bool, str]:
        """Is this strategy authorised for LIVE deployment (not just enabled)?

        Deployment-gated strategies (RVGI box breakout and the three added for the
        current tape) ship shadow-only until they have per-regime samples.
        """
        try:
            from app.technical.strategy_algorithms import strategy_live_authorized

            if strategy_live_authorized(strategy_id):
                return True, ""
        except Exception:  # noqa: BLE001 - a lookup failure must fail closed.
            return False, f"STRATEGY_AUTHORIZATION_UNAVAILABLE:{strategy_id}"
        return False, f"STRATEGY_NOT_LIVE_AUTHORIZED:{strategy_id}"

    # -- bandit scoring ---------------------------------------------------- #
    def _bandit_choice(
        self,
        proposals: list["_ElectionProposal"],
        macro: Any,
        now: datetime,
    ) -> "_ElectionProposal | None":
        """Score every proposal on a pessimistic lower bound; may choose nothing."""
        context = BanditContext(
            market=market_for_symbol(proposals[0].symbol),
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
                reason_codes=(
                    ()
                    if proposal.gnn_actionable
                    else ("BANDIT_GNN_ESTIMATE_UNAVAILABLE",)
                ),
            )
            for proposal in proposals
        ]
        selection = self.bandit.select(arms, context, now=now)
        self._state.bandit_selected_arm = selection.selected_arm
        self._state.bandit_conservative_edge_bps = selection.conservative_edge_bps
        self._state.bandit_is_exploration = selection.is_exploration
        self._state.bandit_reason_codes = list(selection.reason_codes)
        self._state.bandit_evaluations = [item.as_dict() for item in selection.evaluations][:12]
        self._state.bandit_shadow_arms = list(selection.shadow_arms)
        if selection.is_no_trade:
            # This is a successful outcome, not a failure: no candidate cleared a
            # positive lower bound after costs and uncertainty.
            self._state.last_reason = (
                "BANDIT_NO_TRADE_NO_POSITIVE_CONSERVATIVE_EDGE"
                if BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE in selection.reason_codes
                else f"BANDIT_NO_TRADE:{','.join(selection.reason_codes) or 'UNSPECIFIED'}"
            )
            return None
        winner = next(
            (
                proposal
                for proposal in proposals
                if proposal.strategy_id == selection.selected_arm
                and proposal.symbol == selection.selected_symbol
            ),
            None,
        )
        if winner is None:
            self._state.last_reason = "BANDIT_SELECTION_UNRESOLVABLE"
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
        target_price = (
            entry_price * (1.0 + proposal.target_return_rate)
            if entry_price and entry_price > 0
            else None
        )
        self._state = StrategySessionState(
            session_id=f"session-{uuid4().hex}",
            phase="ARMED",
            selected_symbol=proposal.symbol,
            selected_strategy=proposal.strategy_id,
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
        return context

    def _apply_owned_exit_geometry(self, entry_price: float) -> None:
        """Resolve the elected algorithm's structural exit rule at fill time."""
        state = self._state
        if state.selected_strategy != "rvgi_box_breakout" or entry_price <= 0:
            state.target_price = entry_price * (1.0 + state.target_return_rate)
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
            state.stop_price = rule.stop_price
            algorithm_target_rate = (
                rule.target_price / entry_price - 1.0
                if rule.target_price and rule.target_price > entry_price
                else 0.0
            )
            state.target_return_rate = max(
                state.target_return_rate,
                algorithm_target_rate,
            )
            state.target_price = entry_price * (1.0 + state.target_return_rate)
            state.trailing_stop_rate = float(rule.trailing_bps or 0.0) / 10_000.0
            state.max_holding_seconds = int(rule.max_holding_seconds)
        except Exception:  # noqa: BLE001 - persisted fallback exits remain authoritative.
            state.target_price = entry_price * (1.0 + state.target_return_rate)

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
        self._state = StrategySessionState(
            session_id=f"session-{uuid4().hex}",
            phase="OWNED",
            selected_symbol=symbol,
            selected_strategy=strategy,
            selection_source="BROKER_POSITION_RECONCILIATION",
            selected_at=_iso(now),
            position_opened_at=_iso(opened),
            position_seen=True,
            entry_price=average or None,
            target_price=average * (1.0 + target_rate) if average else None,
            target_return_rate=target_rate,
            stop_loss_rate=stop_bps / 10_000.0,
            trailing_stop_rate=trailing_bps / 10_000.0,
            high_watermark_price=max(
                average,
                float(getattr(holding, "last_price", 0.0) or 0.0),
            ) or None,
            max_holding_seconds=_strategy_max_holding_seconds(strategy),
            last_evaluated_at=_iso(now),
            last_reason=(
                "EXISTING_MULTIPLE_HOLDINGS_BUYS_BLOCKED"
                if len(holdings) > 1
                else "EXISTING_POSITION_ADOPTED"
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
