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

from app.routing.actions import is_actionable_strategy_route
from app.strategy.catalog import is_known_strategy

_MAX_HOLDING_SECONDS = {
    "intraday_momentum": 1800,
    "breakout_volume": 2700,
    "vwap_mean_reversion": 1800,
    "liquidity_shock_reversal": 1200,
    "event_momentum": 3600,
    "cross_sectional_relative_strength": 3600,
    "gap_context": 2700,
    "rvgi_box_breakout": 1800,
}
_STRATEGY_EXIT_BPS = {
    "intraday_momentum": (22.0, 100.0, 15.0),
    "breakout_volume": (25.0, 120.0, 18.0),
    "vwap_mean_reversion": (18.0, 100.0, 12.0),
    "liquidity_shock_reversal": (30.0, 100.0, 18.0),
    "event_momentum": (35.0, 140.0, 24.0),
    "cross_sectional_relative_strength": (28.0, 120.0, 20.0),
    "gap_context": (32.0, 130.0, 22.0),
    "rvgi_box_breakout": (20.0, 120.0, 15.0),
}


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
    fallback_target_return_rate: float = field(
        default_factory=lambda: max(0.001, _env_float("STRATEGY_SESSION_TARGET_RETURN_RATE", 0.004))
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
    target_return_rate: float = 0.004
    target_profit_amount: float | None = None
    stop_loss_rate: float = 0.0022
    trailing_stop_rate: float = 0.0015
    high_watermark_price: float | None = None
    max_holding_seconds: int = 600
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


class StrategySessionManager:
    """Persistent closed-world ownership state machine for the live engine."""

    def __init__(
        self,
        *,
        config: StrategySessionConfig | None = None,
        selection_evidence_provider: Callable[[tuple[str, ...]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config or StrategySessionConfig()
        self.selection_evidence_provider = selection_evidence_provider
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
            state.phase = "COOLDOWN"
            state.cooldown_until = _iso(now + timedelta(seconds=self.config.cooldown_seconds))
            state.last_reason = "POSITION_FLAT_RESELECTION_COOLDOWN"
            state.exit_requested_at = state.exit_requested_at or _iso(now)

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
        elif pnl or state.position_seen:
            state.last_reason = "POSITION_OWNED_STRATEGY_MONITORING"

    def _select(self, candidates: tuple[str, ...], bundle: Any, now: datetime) -> None:
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
            if self.config.require_live_gnn and not gnn_actionable:
                continue
            entry_price = float(getattr(intent, "expected_entry_price", 0.0) or 0.0)
            expected_exit = float(getattr(intent, "expected_exit_price", 0.0) or 0.0)
            target_rate = self.config.fallback_target_return_rate
            if entry_price > 0 and expected_exit > entry_price:
                target_rate = max(target_rate, expected_exit / entry_price - 1.0)
            target_price = (
                max(expected_exit, entry_price * (1.0 + target_rate)) if entry_price > 0 else None
            )
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
            if selected_strategy == "rvgi_box_breakout":
                from app.technical.strategy_algorithms import strategy_live_authorized

                if not strategy_live_authorized(selected_strategy):
                    self._state.last_reason = "RVGI_BOX_NOT_LIVE_AUTHORIZED"
                    continue
            stop_bps, profit_bps, trailing_bps = _STRATEGY_EXIT_BPS.get(
                selected_strategy,
                (25.0, 40.0, 15.0),
            )
            profit_bps = _cost_aware_profit_bps(gnn, profit_bps)
            target_rate = max(target_rate, profit_bps / 10_000.0)
            target_price = (
                entry_price * (1.0 + target_rate) if entry_price > 0 else None
            )
            self._state = StrategySessionState(
                session_id=f"session-{uuid4().hex}",
                phase="ARMED",
                selected_symbol=symbol,
                selected_strategy=selected_strategy,
                selection_source=(
                    "ONTOLOGY_GNN_AGREEMENT"
                    if gnn_actionable and gnn_strategy == ontology_strategy
                    else (
                        "GNN_STRATEGY_ELECTION"
                        if gnn_actionable
                        else "ONTOLOGY_WITH_GNN_GUARD"
                    )
                ),
                selection_score=float(getattr(intent, "score", 0.0) or 0.0),
                selection_confidence=float(getattr(intent, "confidence", 0.0) or 0.0),
                selected_at=_iso(now),
                entry_price=entry_price or None,
                target_price=target_price,
                target_return_rate=target_rate,
                stop_loss_rate=stop_bps / 10_000.0,
                trailing_stop_rate=trailing_bps / 10_000.0,
                max_holding_seconds=_MAX_HOLDING_SECONDS.get(selected_strategy, 600),
                last_evaluated_at=_iso(now),
                last_reason="SINGLE_SYMBOL_STRATEGY_ARMED",
                macro_regime=str(getattr(intent, "macro_regime", "") or ""),
                micro_regime=str(getattr(intent, "micro_regime", "") or ""),
                ontology_reason_codes=list(getattr(intent, "reason_codes", ()) or ()),
                gnn_action=gnn_action,
                gnn_reason_codes=gnn_reason_codes,
                explanation_paths=list(getattr(intent, "explanation_paths", ()) or ()),
                candidate_diagnostics=list(self._state.candidate_diagnostics),
                election_context=self._election_context(
                    selected_strategy,
                    now,
                    intent=intent,
                    candidate_count=len(intents),
                    micro_result=next(
                        (
                            result
                            for result in tuple(getattr(bundle, "micro_results", ()) or ())
                            if str(getattr(result, "symbol", "") or "").upper() == symbol
                        ),
                        None,
                    ),
                    evidence_row=row,
                ),
            )
            return
        # Election and entry timing are separate responsibilities.  A fresh
        # ontology/GNN admissibility decision may arm a strategy even when its
        # tick-level entry trigger is not ready yet; the owned strategy executor
        # will wait in ARMED without placing an order.
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
            )
            if self.config.require_live_gnn and not gnn_actionable:
                continue
            if not gnn_actionable and not ontology_actionable:
                continue
            selected_strategy = (
                gnn_strategy if gnn_actionable else ontology_strategy
            )
            if not selected_strategy:
                continue
            if selected_strategy == "rvgi_box_breakout":
                from app.technical.strategy_algorithms import strategy_live_authorized

                if not strategy_live_authorized(selected_strategy):
                    self._state.last_reason = "RVGI_BOX_NOT_LIVE_AUTHORIZED"
                    continue
            # The shadow evidence path does not know the macro regime, so it
            # would happily arm a strategy family the macro layer has blocked.
            # The supervisor would then flag it every cycle; refuse it here.
            if _macro_permits(bundle, selected_strategy) is False:
                self._state.last_reason = (
                    f"MACRO_BLOCKS_ELECTED_STRATEGY:{selected_strategy}"
                )
                continue
            stop_bps, profit_bps, trailing_bps = _STRATEGY_EXIT_BPS.get(
                selected_strategy,
                (25.0, 40.0, 15.0),
            )
            profit_bps = _cost_aware_profit_bps(gnn, profit_bps)
            self._state = StrategySessionState(
                session_id=f"session-{uuid4().hex}",
                phase="ARMED",
                selected_symbol=normalized,
                selected_strategy=selected_strategy,
                selection_source=(
                    "GNN_STRATEGY_ELECTION"
                    if gnn_actionable
                    else "ONTOLOGY_STRATEGY_ELECTION"
                ),
                selected_at=_iso(now),
                target_return_rate=max(
                    self.config.fallback_target_return_rate,
                    profit_bps / 10_000.0,
                ),
                stop_loss_rate=stop_bps / 10_000.0,
                trailing_stop_rate=trailing_bps / 10_000.0,
                max_holding_seconds=_MAX_HOLDING_SECONDS.get(
                    selected_strategy,
                    600,
                ),
                last_evaluated_at=_iso(now),
                last_reason="STRATEGY_ELECTED_WAITING_FOR_ENTRY_TRIGGER",
                macro_regime=self._state.macro_regime,
                ontology_reason_codes=list(
                    ontology.get("reason_codes") or ()
                ),
                gnn_action=gnn_action,
                gnn_reason_codes=gnn_reason_codes,
                candidate_diagnostics=list(self._state.candidate_diagnostics),
                election_context=self._election_context(
                    selected_strategy,
                    now,
                    evidence_row=row,
                ),
            )
            return
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
        self._state.last_reason = (
            (
                (
                    "GNN_POSITIVE_EDGE_AWAITING_ENTRY_VALIDATION"
                    if positive_edge_awaiting_validation
                    else (
                        "NO_POSITIVE_NET_GNN_EDGE"
                        if model_trust_ready
                        else "GNN_NOT_LIVE_AUTHORIZED"
                    )
                )
            )
            if self.config.require_live_gnn
            else (
                "NO_FRESH_STRATEGY_ELECTION"
                if not intents
                else "ONTOLOGY_GNN_STRATEGY_DISAGREEMENT"
            )
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
        if intent is not None:
            entry = float(getattr(intent, "expected_entry_price", 0.0) or 0.0)
            if entry > 0:
                context["reference_price"] = entry
            net_bps = getattr(intent, "expected_net_return_bps", None)
            if net_bps is not None:
                context["expected_net_return_bps"] = float(net_bps)
            # The arbiter orders BUY candidates by expected net return, which is
            # the cross-sectional ranking this system actually has.
            rank = getattr(intent, "rank", None)
            if rank is not None and candidate_count:
                context["sector_rank"] = int(rank)
                context["sector_candidate_count"] = int(candidate_count)
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
        stop_bps, profit_bps, trailing_bps = _STRATEGY_EXIT_BPS.get(
            strategy,
            (25.0, 40.0, 15.0),
        )
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
            max_holding_seconds=_MAX_HOLDING_SECONDS.get(strategy, 600),
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
            return StrategySessionState(**{key: value for key, value in raw.items() if key in allowed})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return StrategySessionState(target_return_rate=self.config.fallback_target_return_rate)

    def _persist(self) -> None:
        path = Path(self.config.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self._state), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
