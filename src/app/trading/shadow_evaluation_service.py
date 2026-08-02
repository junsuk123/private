"""Drives shadow plans to resolution and turns them into promotion evidence.

The missing link
----------------
``StrategySessionManager`` journals a :class:`ShadowTradePlan` whenever a
non-order-authorised arm produces a signal, and ``ShortStrategyPromotionController``
reads realized outcomes to decide promotions. Without this service in between,
plans would be written and never scored: no forward evidence would accumulate, and
no arm could ever leave ``SHADOW``.

That failure mode is *safe* (nothing is promoted) but it is not *correct* — a ladder
whose bottom rung has no exit is a permanent block dressed up as a validation
process. This service is what makes the ladder able to progress.

What it does per tick
---------------------
1. Adopts newly journaled plans into the in-memory :class:`ShadowFillSimulator`.
2. Feeds the current bid/ask to every in-flight plan.
3. Persists resolved outcomes to BOTH stores:
   * the shadow journal (full detail, for audit and holdout windows), and
   * the performance store (so the bandit's posterior sees them).
4. Expires plans whose horizon elapsed without further quotes.

It never places an order and has no path to the execution layer. The
:class:`ShadowFillSimulator` it drives is the same one the tests exercise, so its
three leak defences (temporal, borrow, pricing) apply unchanged.

Restart behaviour
-----------------
In-flight walks live in memory and are lost on restart; their plans stay in the
journal unresolved. That is deliberate: reconstructing a barrier walk after a
restart would mean replaying quotes we did not observe at the time, which is the
leak this whole subsystem exists to prevent. A lost plan costs one sample.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from app.trading.directional import DirectionalStrategyKey, PositionDirection
from app.trading.directional_shadow import (
    QuoteObservation,
    ShadowFillSimulator,
    ShadowOutcome,
    ShadowPlanStore,
    ShadowTradePlan,
    default_shadow_store,
)
from app.trading.strategy_performance_store import (
    EVALUATION_SOURCE_SHADOW,
    StrategyPerformanceStore,
    default_store as default_performance_store,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowEvaluationStats:
    """What one tick of evaluation did."""

    adopted: int = 0
    resolved: int = 0
    scored: int = 0
    unexecutable: int = 0
    expired: int = 0
    open_plans: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "adopted": self.adopted,
            "resolved": self.resolved,
            "scored": self.scored,
            "unexecutable": self.unexecutable,
            "expired": self.expired,
            "open_plans": self.open_plans,
        }


class ShadowEvaluationService:
    """Scores journaled shadow plans from post-signal quotes only."""

    def __init__(
        self,
        *,
        simulator: ShadowFillSimulator | None = None,
        shadow_store: ShadowPlanStore | None = None,
        performance_store: StrategyPerformanceStore | None = None,
        max_open_plans: int = 512,
    ) -> None:
        self.simulator = simulator or ShadowFillSimulator()
        self.shadow_store = shadow_store or default_shadow_store()
        self.performance_store = performance_store or default_performance_store()
        # Bounded so a stuck feed cannot grow the walk set without limit. Plans past
        # the cap are simply not adopted; they stay in the journal unresolved, which
        # loses a sample rather than leaking memory.
        self.max_open_plans = max(1, int(max_open_plans))
        self._lock = threading.RLock()
        self._adopted: set[str] = set()
        # Rolling short-rescue tally. The promotion gate reads a RATE, and the only
        # place that can observe the numerator is the election loop, so it is
        # accumulated here rather than recomputed from storage.
        self._rescue_observations = 0
        self._rescue_hits = 0
        self._last_stats = ShadowEvaluationStats()

    # -- plan intake --------------------------------------------------------- #
    def adopt(
        self, plans: Iterable[ShadowTradePlan]
    ) -> tuple[int, tuple[ShadowOutcome, ...]]:
        """Register plans for barrier walking. Idempotent by ``plan_id``.

        Returns ``(adopted_count, immediately_resolved)``. Unexecutable plans (no
        locate) resolve on submit without entering the walk set, and they are returned
        so the caller can COUNT them — reporting them as merely "adopted" would make a
        run where every short lacked a locate look identical to one where every short
        was still in flight, and those need different operator responses.
        """
        adopted = 0
        resolved: list[ShadowOutcome] = []
        with self._lock:
            for plan in plans:
                if plan.plan_id in self._adopted:
                    continue
                if self.simulator.open_plan_count >= self.max_open_plans:
                    logger.warning(
                        "shadow evaluation at capacity (%d); plan %s left unresolved",
                        self.max_open_plans,
                        plan.plan_id,
                    )
                    break
                self._adopted.add(plan.plan_id)
                outcome = self.simulator.submit(plan)
                adopted += 1
                if outcome is not None:
                    self._persist(outcome)
                    resolved.append(outcome)
        return adopted, tuple(resolved)

    # -- per-tick evaluation ------------------------------------------------- #
    def observe(
        self,
        symbol: str,
        *,
        bid_price: float,
        ask_price: float,
        observed_at: datetime,
        last_price: float | None = None,
    ) -> tuple[ShadowOutcome, ...]:
        """Feed one quote. Returns outcomes that resolved on it.

        The simulator itself refuses quotes at or before each plan's ``signal_at``, so
        this method does not need to filter by symbol correctness or timing — but it
        DOES route by symbol, because feeding one symbol's book to another symbol's
        plan would fabricate the entire price path.
        """
        quote = QuoteObservation(
            observed_at=observed_at,
            bid_price=float(bid_price or 0.0),
            ask_price=float(ask_price or 0.0),
            last_price=last_price,
        )
        if not quote.usable:
            return ()
        normalized = str(symbol or "").strip().upper()
        with self._lock:
            resolved = self.simulator.observe_symbol(normalized, quote)
            for outcome in resolved:
                self._persist(outcome)
        return resolved

    def evaluate_tick(
        self,
        quotes: Mapping[str, Mapping[str, Any]],
        *,
        now: datetime | None = None,
        new_plans: Sequence[ShadowTradePlan] = (),
    ) -> ShadowEvaluationStats:
        """One full evaluation step: adopt, walk every quote, then expire.

        ``quotes`` maps symbol -> ``{"bid_price", "ask_price", "observed_at"}``.
        Designed to be called from the engine loop with whatever book it already has,
        so shadow evaluation adds no extra market-data fetch.
        """
        moment = _aware(now or datetime.now(timezone.utc))
        adopted, resolved_on_adopt = self.adopt(new_plans)
        resolved: list[ShadowOutcome] = list(resolved_on_adopt)
        for symbol, book in (quotes or {}).items():
            observed_at = book.get("observed_at")
            if not isinstance(observed_at, datetime):
                # Without a real observation time the anti-leak check cannot be
                # applied, so the quote is skipped rather than stamped with `now`.
                continue
            resolved.extend(
                self.observe(
                    symbol,
                    bid_price=book.get("bid_price") or 0.0,
                    ask_price=book.get("ask_price") or 0.0,
                    observed_at=observed_at,
                    last_price=book.get("last_price"),
                )
            )
        with self._lock:
            expired = self.simulator.expire(moment)
            for outcome in expired:
                self._persist(outcome)
            stats = ShadowEvaluationStats(
                adopted=adopted,
                resolved=len(resolved) + len(expired),
                scored=sum(1 for item in (*resolved, *expired) if item.scored),
                unexecutable=sum(
                    1 for item in (*resolved, *expired) if not item.executable
                ),
                expired=len(expired),
                open_plans=self.simulator.open_plan_count,
            )
            self._last_stats = stats
        return stats

    # -- short rescue rate --------------------------------------------------- #
    def record_directional_comparison(self, comparison: Mapping[str, Any] | None) -> None:
        """Accumulate the ``short_rescue_rate`` numerator/denominator.

        Called once per election with ``BanditSelection.as_dict()["directional_comparison"]``.
        Every election counts toward the denominator — including ones where no short was
        selectable — because the question is "how often would a short have helped",
        not "how often did a short win".
        """
        if not isinstance(comparison, Mapping) or not comparison:
            return
        with self._lock:
            self._rescue_observations += 1
            if bool(comparison.get("short_rescued")):
                self._rescue_hits += 1

    @property
    def short_rescue_rate(self) -> float | None:
        """``None`` until at least one election has been observed.

        ``None`` rather than 0.0, because an unmeasured rate must FAIL the promotion
        gate rather than pass it as "no rescues observed". Zero observations means we
        have not asked the question.
        """
        with self._lock:
            if self._rescue_observations <= 0:
                return None
            return self._rescue_hits / self._rescue_observations

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "open_plans": self.simulator.open_plan_count,
                "adopted_total": len(self._adopted),
                "short_rescue_rate": self.short_rescue_rate,
                "short_rescue_observations": self._rescue_observations,
                "short_rescue_hits": self._rescue_hits,
                "last_tick": self._last_stats.as_dict(),
            }

    # -- persistence --------------------------------------------------------- #
    def _persist(self, outcome: ShadowOutcome) -> None:
        """Write one resolved outcome to both stores.

        The shadow journal always gets it (including unexecutable signals, which the
        holdout windows and borrow-availability rate both need). The performance store
        gets it too, tagged ``evaluation_source=shadow`` and with
        ``signal_executable`` reflecting whether a locate existed — that flag is what
        keeps unexecutable signals out of every promotion statistic while still
        counting them in the borrow denominator.
        """
        try:
            self.shadow_store.record_outcome(outcome)
        except Exception:  # noqa: BLE001 - journaling must not break the engine loop.
            logger.exception("failed to journal shadow outcome %s", outcome.plan_id)
        if outcome.net_return_bps is None and outcome.executable:
            # Unfilled entries produced no round trip, so there is no return to
            # record. They stay in the journal only.
            return
        try:
            self.performance_store.record_directional(
                outcome.key,
                symbol=outcome.symbol,
                # Already direction-signed by ``gross_return_bps``; a short that
                # covered lower records a POSITIVE net.
                realized_net_bps=float(outcome.net_return_bps or 0.0),
                realized_gross_bps=outcome.gross_return_bps,
                holding_seconds=outcome.holding_seconds,
                slippage_error_bps=outcome.slippage_error_bps,
                max_adverse_excursion_bps=outcome.max_adverse_excursion_bps,
                exit_reason=outcome.outcome,
                recorded_at=outcome.resolved_at,
                evaluation_source=EVALUATION_SOURCE_SHADOW,
                deployment_state="SHADOW",
                borrow_available=(
                    outcome.executable
                    if outcome.key.direction is PositionDirection.SHORT
                    else None
                ),
                borrow_fee_bps=None,
                signal_executable=outcome.executable,
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed to record shadow outcome %s", outcome.plan_id)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


_DEFAULT_SERVICE: ShadowEvaluationService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def default_shadow_evaluation_service() -> ShadowEvaluationService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = ShadowEvaluationService()
    return _DEFAULT_SERVICE


def reset_default_shadow_evaluation_service() -> None:
    global _DEFAULT_SERVICE
    with _DEFAULT_SERVICE_LOCK:
        _DEFAULT_SERVICE = None
