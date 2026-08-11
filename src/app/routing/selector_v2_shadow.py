"""Runs StrategySelectorV2 beside the legacy selector and publishes earned authority.

This runner owns no broker or execution capability. It observes and scores contexts,
maintains counterfactual evidence, and exposes the persisted promotion controller's
effective authority. ``StrategySessionManager`` is the downstream boundary that may use a
promoted result, and only by matching it to an independently order-authorised proposal:

* **No execution imports.** This module imports nothing from ``app.execution``,
  ``app.risk``, ``app.cost.profitability_gate`` or any broker client. There is no code path
  from here to an order; a result cannot construct or submit an order.
* **Never raises.** ``observe`` catches everything. A shadow comparison that could break the
  live election would be worse than no comparison, and the live path's own
  ``try/except`` around ``strategy_session_manager.evaluate`` already fails closed by
  disabling buys — so an exception here would DISABLE TRADING, which is exactly the
  outcome a telemetry feature must not be able to cause.
* **Bounded work.** ``max_symbols_per_cycle`` caps how many candidates are evaluated, and the
  cap is reported in the telemetry rather than silently truncating.

What it produces
----------------
``snapshot()`` is the dashboard/API view described in the runtime-diagnostics requirement:
the context id, the ontology hard blocks, every eligible proposal, every predicted utility
with its full term decomposition, the NO_TRADE threshold, the final selection, and the
legacy-vs-V2 comparison.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Mapping, Sequence

from app.config.selector_v2_flags import SelectorV2Flags
from app.context.context_builder import MarketContextBuilder, SymbolContextInputs
from app.context.context_store import MarketContextStore
from app.context.market_context import MarketContext
from app.evaluation.counterfactual_engine import CounterfactualEngine
from app.ontology.strategy_eligibility import StrategyEligibilityEngine
from app.routing.selector_v2_promotion import (
    SelectorPromotionConfig,
    SelectorPromotionController,
)
from app.routing.ontology_strategy_mask import OntologyStrategyMask
from app.routing.strategy_selector import (
    StrategySelectionResult,
    StrategySelectorV2,
    UtilityWeights,
)
from app.strategy.coverage import StrategyCoverageAnalyzer
from app.strategy.registry import default_strategy_registry

__all__ = ["SelectorV2ShadowRunner", "ShadowComparison", "load_utility_weights"]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def load_utility_weights(
    path: str = "config/strategy_selector_v2.yaml",
) -> UtilityWeights:
    """Read the lambdas from config, falling back to the dataclass defaults."""
    try:
        from pathlib import Path

        import yaml

        target = Path(path)
        if not target.exists():
            return UtilityWeights()
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - an unreadable config uses defaults.
        return UtilityWeights()
    values = payload.get("weights") if isinstance(payload, Mapping) else None
    return UtilityWeights.from_mapping(values if isinstance(values, Mapping) else None)


@dataclass(frozen=True)
class ShadowComparison:
    """One cycle's legacy-vs-V2 verdict."""

    at: datetime
    symbol: str
    context_id: str
    legacy_strategy: str | None
    legacy_reason: str
    v2_strategy: str | None
    v2_decision: str
    v2_utility_bps: float | None
    agreement: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "symbol": self.symbol,
            "context_id": self.context_id,
            "legacy_strategy": self.legacy_strategy,
            "legacy_reason": self.legacy_reason,
            "v2_strategy": self.v2_strategy,
            "v2_decision": self.v2_decision,
            "v2_utility_bps": (
                round(self.v2_utility_bps, 3) if self.v2_utility_bps is not None else None
            ),
            "agreement": self.agreement,
        }


def _classify_agreement(legacy: str | None, v2: str | None) -> str:
    if legacy is None and v2 is None:
        return "BOTH_NO_TRADE"
    if legacy is not None and v2 is None:
        return "V2_DECLINED"
    if legacy is None and v2 is not None:
        return "V2_TRADED"
    return "SAME_STRATEGY" if legacy == v2 else "DIFFERENT_STRATEGY"


class SelectorV2ShadowRunner:
    """Build contexts and evidence, and publish effective selector authority.

    The runner owns no broker capability; the session boundary consumes its authority
    state and can only match a result to an independently executable proposal.
    """

    def __init__(
        self,
        *,
        flags: SelectorV2Flags | None = None,
        selector: StrategySelectorV2 | None = None,
        context_builder: MarketContextBuilder | None = None,
        context_store: MarketContextStore | None = None,
        coverage: StrategyCoverageAnalyzer | None = None,
        counterfactual: CounterfactualEngine | None = None,
        promotion: SelectorPromotionController | None = None,
        max_symbols_per_cycle: int = 8,
        history_size: int = 200,
    ) -> None:
        self._flags = flags or SelectorV2Flags.from_env()
        registry = default_strategy_registry()
        self._selector = selector or StrategySelectorV2(
            registry=registry,
            mask=OntologyStrategyMask(
                engine=StrategyEligibilityEngine(registry=registry),
                enabled=self._flags.ontology_mask_v2_enabled,
            ),
            weights=load_utility_weights(),
            bandit_enabled=self._flags.bandit_adapter_enabled,
        )
        self._contexts = context_builder or MarketContextBuilder()
        self._store = context_store or MarketContextStore()
        self._coverage = coverage or StrategyCoverageAnalyzer()
        self._counterfactual = (
            counterfactual
            if counterfactual is not None
            else (CounterfactualEngine() if self._flags.counterfactual_enabled else None)
        )
        self._promotion = (
            promotion
            if promotion is not None
            else (
                SelectorPromotionController(SelectorPromotionConfig.load())
                if self._flags.auto_promote
                else None
            )
        )
        self._max_symbols = max(1, int(max_symbols_per_cycle))
        self._lock = threading.RLock()
        self._comparisons: Deque[ShadowComparison] = deque(maxlen=max(1, int(history_size)))
        self._last_results: dict[str, StrategySelectionResult] = {}
        self._last_error: str | None = None
        self._cycles = 0
        self._skipped_symbols = 0
        self._evaluated_symbols = 0

    # -- properties --------------------------------------------------------- #
    @property
    def enabled(self) -> bool:
        return self._flags.enabled

    @property
    def flags(self) -> SelectorV2Flags:
        return self._flags

    @property
    def live_authority(self) -> bool:
        """Effective authority, whether operator-forced or earned automatically."""
        return bool(
            self._flags.live_authority
            or (self._promotion is not None and self._promotion.live_authority)
        )

    @property
    def order_size_fraction(self) -> float:
        if self._flags.live_authority:
            return 1.0
        return self._promotion.order_size_fraction if self._promotion is not None else 0.0

    @property
    def authority_state(self) -> str:
        if self._flags.live_authority:
            return "LIVE"
        return str(self._promotion.state) if self._promotion is not None else "SHADOW"

    @property
    def open_symbols(self) -> tuple[str, ...]:
        if self._counterfactual is None:
            return ()
        return self._counterfactual.open_symbols

    # -- the one entry point ------------------------------------------------ #
    def observe(
        self,
        *,
        candidates: Sequence[str],
        evidence: Mapping[str, Any],
        bundle: Any,
        now: datetime,
        legacy_strategy: str | None,
        legacy_symbol: str | None,
        legacy_reason: str,
    ) -> tuple[StrategySelectionResult, ...]:
        """Run V2 over the SAME inputs the legacy selector just used.

        Using the same ``evidence`` mapping is what makes the comparison meaningful: a
        difference in outcome is then a difference in the selector, not in what it saw.
        """
        if not self._flags.enabled:
            return ()
        try:
            return self._observe(
                candidates=candidates,
                evidence=evidence,
                bundle=bundle,
                now=_aware(now),
                legacy_strategy=legacy_strategy,
                legacy_symbol=legacy_symbol,
                legacy_reason=legacy_reason,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must never break the cycle.
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:240]
            return ()

    def _observe(
        self,
        *,
        candidates: Sequence[str],
        evidence: Mapping[str, Any],
        bundle: Any,
        now: datetime,
        legacy_strategy: str | None,
        legacy_symbol: str | None,
        legacy_reason: str,
    ) -> tuple[StrategySelectionResult, ...]:
        macro = getattr(bundle, "macro_result", None)
        allowed, blocked = _macro_lists(macro)
        wanted = [str(symbol or "").strip().upper() for symbol in candidates if symbol]
        if len(wanted) > self._max_symbols:
            # Keep the legacy pick in the set whatever the cap: dropping it would make the
            # comparison meaningless for the one symbol that matters most.
            head = wanted[: self._max_symbols]
            legacy_upper = str(legacy_symbol or "").strip().upper()
            if legacy_upper and legacy_upper in wanted and legacy_upper not in head:
                head[-1] = legacy_upper
            self._skipped_symbols += len(wanted) - len(head)
            wanted = head

        inputs: list[SymbolContextInputs] = []
        election_inputs_by_symbol: dict[str, Mapping[str, Any]] = {}
        for symbol in wanted:
            row = evidence.get(symbol) if isinstance(evidence, Mapping) else None
            features = _features_from_row(row)
            if features is None:
                self._skipped_symbols += 1
                continue
            micro = _micro_result_for(bundle, symbol)
            election_inputs = _election_inputs_from(row, micro)
            election_inputs_by_symbol[symbol] = election_inputs
            inputs.append(
                SymbolContextInputs(
                    symbol=symbol,
                    features=features,
                    election_inputs=election_inputs,
                    micro_result=micro,
                    evidence_row=row if isinstance(row, Mapping) else None,
                    tick_freshness_sec=_row_age_seconds(row, now),
                    orderbook_freshness_sec=_row_age_seconds(row, now),
                    history_bar_count=_history_bars(row),
                )
            )
        if not inputs:
            return ()

        contexts = self._contexts.build_cycle(
            inputs,
            captured_at=now,
            macro=macro,
            macro_age_seconds=0.0,
        )
        self._store.put_all(contexts)

        results: list[StrategySelectionResult] = []
        for context in contexts:
            election_inputs = election_inputs_by_symbol.get(context.symbol_id, {})
            gnn_rows = (
                _gnn_rows_for(evidence.get(context.symbol_id))
                if self._flags.utility_gnn_enabled
                else ()
            )
            result = self._selector.select(
                context,
                election_inputs=election_inputs,
                gnn_rows=gnn_rows,
                macro_allowed=allowed,
                macro_blocked=blocked,
                now=now,
            )
            results.append(result)
            self._record(context, result, legacy_strategy, legacy_symbol, legacy_reason, now)

        self._evaluated_symbols += len(contexts)
        self._cycles += 1
        return tuple(results)

    def _record(
        self,
        context: MarketContext,
        result: StrategySelectionResult,
        legacy_strategy: str | None,
        legacy_symbol: str | None,
        legacy_reason: str,
        now: datetime,
    ) -> None:
        self._coverage.record_selection(context, result)
        if self._counterfactual is not None:
            # Alternatives get virtual positions. This is the counterfactual dataset the
            # selector-regret metric is computed from, and it cannot reach a broker.
            self._counterfactual.open_from_selection(
                context=context,
                selection=result,
                proposals=result.proposals,
                trailing_bps_by_strategy=_trailing_bps_for(result),
            )
        legacy_here = (
            legacy_strategy
            if str(legacy_symbol or "").strip().upper() == context.symbol_id
            else None
        )
        with self._lock:
            self._last_results[context.symbol_id] = result
            self._comparisons.append(
                ShadowComparison(
                    at=now,
                    symbol=context.symbol_id,
                    context_id=context.context_id,
                    legacy_strategy=legacy_here,
                    legacy_reason=str(legacy_reason or ""),
                    v2_strategy=result.selected_strategy,
                    v2_decision=result.decision,
                    v2_utility_bps=result.utility,
                    agreement=_classify_agreement(legacy_here, result.selected_strategy),
                )
            )

    # -- quote feed for the counterfactual engine ---------------------------- #
    def observe_quote(self, symbol: str, price: float, at: datetime) -> int:
        """Walk open shadow positions forward. Returns how many resolved.

        Separate from ``observe`` because quotes arrive far more often than election cycles,
        and the barrier walk must see each one to resolve honestly.
        """
        if self._counterfactual is None:
            return 0
        try:
            return len(self._counterfactual.observe_quote(symbol, price, _aware(at)))
        except Exception:  # noqa: BLE001
            return 0

    def record_live_outcome(
        self,
        *,
        context_id: str,
        strategy_id: str,
        net_return_bps: float,
        evidence_source: str,
    ) -> bool:
        if self._counterfactual is None or not context_id:
            return False
        try:
            return self._counterfactual.record_live_outcome(
                context_id=context_id,
                strategy_id=strategy_id,
                net_return_bps=net_return_bps,
                evidence_source=evidence_source,
            ) is not None
        except Exception:  # noqa: BLE001
            return False

    def evaluate_authority(self, now: datetime) -> dict[str, Any] | None:
        """Refresh the automatic ladder from resolved forward contexts."""
        if self._promotion is None or self._counterfactual is None:
            return None
        try:
            decision = self._promotion.evaluate(
                self._counterfactual.resolved_groups(limit=4000), now=_aware(now)
            )
            return decision.as_dict()
        except Exception as exc:  # noqa: BLE001
            # A broken evaluator must remove, never grant, selector authority.
            return self._promotion.suspend(
                f"{type(exc).__name__}:{exc}", now=_aware(now)
            ).as_dict()

    def expire_stale(self, now: datetime) -> int:
        if self._counterfactual is None:
            return 0
        try:
            return len(self._counterfactual.expire_stale(_aware(now)))
        except Exception:  # noqa: BLE001
            return 0

    # -- telemetry ---------------------------------------------------------- #
    def snapshot(self, *, symbol: str | None = None, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            comparisons = list(self._comparisons)[-max(1, int(limit)) :]
            results = dict(self._last_results)
            error = self._last_error
            cycles = self._cycles
            evaluated = self._evaluated_symbols
            skipped = self._skipped_symbols

        selected_result = None
        if symbol:
            selected_result = results.get(str(symbol).strip().upper())
        elif results:
            selected_result = next(iter(results.values()))

        agreement_counts: dict[str, int] = {}
        for item in comparisons:
            agreement_counts[item.agreement] = agreement_counts.get(item.agreement, 0) + 1

        promotion = self._promotion.snapshot() if self._promotion is not None else None
        effective_live = self.live_authority
        return {
            "enabled": self._flags.enabled,
            "shadow_only": not effective_live,
            "configured_shadow_only": self._flags.shadow_only,
            "live_authority": effective_live,
            "order_size_fraction": self.order_size_fraction,
            "auto_promotion": promotion,
            "flags": self._flags.as_dict(),
            "cycles": cycles,
            "symbols_evaluated": evaluated,
            "symbols_skipped": skipped,
            "last_error": error,
            "latest_selection": selected_result.as_dict() if selected_result else None,
            "selections_by_symbol": {
                key: value.as_dict() for key, value in results.items()
            },
            "comparisons": [item.as_dict() for item in comparisons],
            "agreement_counts": agreement_counts,
            "coverage": self._coverage.summary(),
            "coverage_gaps": self._coverage.research_candidates(
                minimum_observations=5, limit=10
            ),
            "counterfactual": (
                self._counterfactual.stats().as_dict()
                if self._counterfactual is not None
                else None
            ),
        }

    def regret_summary(self) -> dict[str, Any] | None:
        """Selector regret over resolved counterfactual groups, or ``None``."""
        if self._counterfactual is None:
            return None
        try:
            from app.evaluation.selector_evaluator import SelectorEvaluator

            groups = self._counterfactual.resolved_groups()
            if not groups:
                return None
            return SelectorEvaluator().evaluate(groups).as_dict()
        except Exception:  # noqa: BLE001
            return None

    def flush(self) -> None:
        """Persist the coverage tally. Called on a schedule, never mid-decision."""
        try:
            self._coverage.flush()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Adapters over the live evidence row                                          #
# --------------------------------------------------------------------------- #
def _features_from_row(row: Any) -> Any | None:
    """Rebuild a ``TechnicalFeatureSet`` from the live evidence row.

    The live path already puts ``asdict(TechnicalFeatureSet)`` into
    ``row["technical_features"]`` (``web._strategy_session_selection_evidence``), so this
    reads the SAME numbers the legacy mechanical trigger fired on rather than recomputing
    them from a second store read.
    """
    if not isinstance(row, Mapping):
        return None
    raw = row.get("technical_features")
    if not isinstance(raw, Mapping):
        return None
    try:
        from dataclasses import fields

        from app.technical.signals import TechnicalFeatureSet

        names = {member.name for member in fields(TechnicalFeatureSet)}
        return TechnicalFeatureSet(
            **{key: value for key, value in raw.items() if key in names}
        )
    except Exception:  # noqa: BLE001
        return None


def _election_inputs_from(row: Any, micro_result: Any) -> dict[str, Any]:
    """Slow context the electing layer resolved, as ``ElectionContext`` field names.

    Read from the evidence row's ``rvgi_box_context`` and the micro result's diagnostics —
    the two places the live path actually publishes these — rather than recomputed. A
    recomputation here would be a second estimator for the same quantity.
    """
    inputs: dict[str, Any] = {}
    if isinstance(row, Mapping):
        mark = row.get("mark_price")
        if mark:
            inputs["reference_price"] = mark
        box = row.get("rvgi_box_context")
        if isinstance(box, Mapping):
            for name in (
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
                if box.get(name) is not None:
                    inputs[name] = box[name]
    diagnostics = getattr(micro_result, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        for name, value in diagnostics.items():
            if value is not None and name not in inputs:
                inputs[str(name)] = value
    return inputs


def _gnn_rows_for(row: Any) -> tuple[Mapping[str, Any], ...]:
    """Full-vector GNN rows from the evidence row, if present."""
    if not isinstance(row, Mapping):
        return ()
    return tuple(
        item
        for item in tuple(row.get("validation_candidates") or ())
        if isinstance(item, Mapping)
        and str(item.get("path") or "").startswith("cpu_gnn")
    )


def _row_age_seconds(row: Any, now: datetime) -> float | None:
    if not isinstance(row, Mapping):
        return None
    for name in ("mark_price_as_of", "as_of", "rvgi_box_as_of"):
        raw = row.get(name)
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(str(raw))
        except ValueError:
            continue
        return max(0.0, (now - _aware(parsed)).total_seconds())
    return None


def _history_bars(row: Any) -> int | None:
    if not isinstance(row, Mapping):
        return None
    for name in ("history_bar_count", "bar_count"):
        value = row.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    features = row.get("technical_features")
    if isinstance(features, Mapping):
        # The box/Donchian columns only exist once their lookback is satisfied, so their
        # presence is a floor on the history that was available. Reporting the floor is
        # honest; reporting zero would hard-block every strategy with a history requirement.
        if features.get("box_high") is not None or features.get("donchian_high") is not None:
            return 20
    return None


def _micro_result_for(bundle: Any, symbol: str) -> Any:
    for item in tuple(getattr(bundle, "micro_results", ()) or ()):
        if str(getattr(item, "symbol", "") or "").strip().upper() == symbol:
            return item
    return None


def _macro_lists(macro: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Macro allow/block family lists, as the existing permission check expects them."""
    allowed = tuple(
        str(item) for item in (getattr(macro, "allowed_strategies", ()) or ())
    )
    blocked = tuple(
        str(item) for item in (getattr(macro, "blocked_strategies", ()) or ())
    )
    return allowed, blocked


def _trailing_bps_for(result: StrategySelectionResult) -> dict[str, float]:
    """Trailing-stop distance per strategy, from the single exit-geometry authority.

    Read from ``app.strategy.exit_geometry`` rather than carried on the proposal, so a shadow
    position trails exactly as the live executor would. The proposal carries the target and
    stop because those are priced against its own point-in-time reference; the trailing
    distance is a property of the strategy, not of the observation.
    """
    try:
        from app.strategy.exit_geometry import exit_geometry
    except Exception:  # noqa: BLE001
        return {}
    trailing: dict[str, float] = {}
    for proposal in result.proposals:
        try:
            trailing[proposal.strategy_id] = float(
                exit_geometry(proposal.strategy_id).trailing_bps
            )
        except Exception:  # noqa: BLE001
            continue
    return trailing
