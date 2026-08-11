"""Boolean eligibility mask ``M_s(x)`` plus a soft compatibility score ``O_s(x)``.

This is the ontology's ONLY output in the V2 pipeline. It does not pick a strategy, does
not rank, and cannot authorise anything — those were the overlapping responsibilities the
refactor exists to separate.

Two independent numbers come out per strategy:

``eligible``            hard mask. ``False`` removes the strategy from the utility
                        ranking entirely. Only the hard relation types can produce it.
``compatibility_score`` soft evidence in ``[-1, 1]``, the weighted mean of the soft
                        relations that fired. It becomes the utility's ``O_s`` term.
                        A strategy with no matching soft relation scores 0.0 — neutral,
                        not penalised, because absence of evidence is not evidence.

Fail-closed, with one deliberate exception
------------------------------------------
A missing *requirement* blocks (fail-closed). A missing *market-state label* does NOT
block: ``forbiddenUnder`` and the market-wide no-entry set can only fire on a label that
is actually present, because an unresolved regime is an unanswered question and the
existing code already treats an unanswerable permission check as "not a withdrawal"
(see ``strategy_algorithms.macro_strategy_permitted``). The data-quality relations are
what catch a context too empty to trust, and they block explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.context.market_context import MarketContext
from app.ontology.strategy_ontology import StrategyOntology, default_strategy_ontology
from app.strategy.registry import StrategyRegistry, default_strategy_registry
from app.strategy.spec import StrategyLifecycleState, StrategySpec

__all__ = [
    "ELIGIBILITY_REASONS",
    "StrategyEligibility",
    "StrategyEligibilityEngine",
    "StrategyEligibilityResult",
]


class ELIGIBILITY_REASONS:
    """Hard-block reason codes. Prefixed so a dashboard can group them."""

    MARKET_NOT_ALLOWED = "ONTO_ELIG_MARKET_NOT_ALLOWED"
    SESSION_NOT_ALLOWED = "ONTO_ELIG_SESSION_NOT_ALLOWED"
    MISSING_CONTEXT = "ONTO_ELIG_MISSING_CONTEXT"
    MISSING_FEATURE = "ONTO_ELIG_MISSING_FEATURE"
    MISSING_ELECTION_INPUT = "ONTO_ELIG_MISSING_ELECTION_INPUT"
    LIQUIDITY_BELOW_FLOOR = "ONTO_ELIG_LIQUIDITY_BELOW_FLOOR"
    LIQUIDITY_UNKNOWN = "ONTO_ELIG_LIQUIDITY_UNKNOWN"
    SPREAD_ABOVE_CEILING = "ONTO_ELIG_SPREAD_ABOVE_CEILING"
    SPREAD_UNKNOWN = "ONTO_ELIG_SPREAD_UNKNOWN"
    HISTORY_INSUFFICIENT = "ONTO_ELIG_HISTORY_INSUFFICIENT"
    TICK_WINDOW_NOT_READY = "ONTO_ELIG_TICK_WINDOW_NOT_READY"
    ORDERBOOK_SAMPLE_MISSING = "ONTO_ELIG_ORDERBOOK_SAMPLE_MISSING"
    COMPLETENESS_BELOW_FLOOR = "ONTO_ELIG_COMPLETENESS_BELOW_FLOOR"
    STALE_DATA = "ONTO_ELIG_STALE_DATA"
    FORBIDDEN_MARKET_STATE = "ONTO_ELIG_FORBIDDEN_MARKET_STATE"
    NO_NEW_ENTRY_MARKET_STATE = "ONTO_ELIG_NO_NEW_ENTRY_MARKET_STATE"
    LIFECYCLE_RETIRED = "ONTO_ELIG_LIFECYCLE_RETIRED"
    DIRECTION_NOT_PERMITTED = "ONTO_ELIG_DIRECTION_NOT_PERMITTED"
    MACRO_FAMILY_BLOCKED = "ONTO_ELIG_MACRO_FAMILY_BLOCKED"
    NO_REFERENCE_PRICE = "ONTO_ELIG_NO_REFERENCE_PRICE"


@dataclass(frozen=True)
class StrategyEligibility:
    """One strategy's hard verdict and soft evidence for one context."""

    strategy_id: str
    eligible: bool
    compatibility_score: float
    hard_block_reasons: tuple[str, ...] = ()
    supporting_relations: tuple[str, ...] = ()
    context_id: str = ""
    #: 1.0 / 0.0, matching the ``M_s`` of the utility formula. Exposed so the selector
    #: multiplies rather than branches.
    @property
    def mask(self) -> float:
        return 1.0 if self.eligible else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "eligible": self.eligible,
            "mask": self.mask,
            "compatibility_score": round(self.compatibility_score, 4),
            "hard_block_reasons": list(self.hard_block_reasons),
            "supporting_relations": list(self.supporting_relations),
            "context_id": self.context_id,
        }


@dataclass(frozen=True)
class StrategyEligibilityResult:
    context_id: str
    symbol: str
    eligibilities: tuple[StrategyEligibility, ...]

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return tuple(item.strategy_id for item in self.eligibilities if item.eligible)

    @property
    def blocked(self) -> tuple[StrategyEligibility, ...]:
        return tuple(item for item in self.eligibilities if not item.eligible)

    def by_id(self) -> dict[str, StrategyEligibility]:
        return {item.strategy_id: item for item in self.eligibilities}

    def mask(self) -> dict[str, float]:
        """``{strategy_id: 0.0 | 1.0}`` — the ``M_s`` vector."""
        return {item.strategy_id: item.mask for item in self.eligibilities}

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "symbol": self.symbol,
            "eligible": list(self.eligible_ids),
            "eligibilities": [item.as_dict() for item in self.eligibilities],
        }


@dataclass(frozen=True)
class EligibilityConfig:
    """Thresholds for the data-quality relations.

    ``minimum_feature_completeness`` is deliberately low. It is a floor against a context
    that is essentially empty (a symbol with no tape at all), not a quality target — the
    per-strategy ``requiresFeature`` relations are what enforce quality, and they do it
    field by field instead of against an aggregate.
    """

    minimum_feature_completeness: float = 0.25
    max_tick_age_seconds: float | None = None
    max_orderbook_age_seconds: float | None = None
    #: Features whose presence implies the sub-second window must be ready. Derived from
    #: the naming convention of ``TechnicalFeatureSet``'s tick block, so a new tick
    #: feature is covered without editing this list.
    tick_feature_prefixes: tuple[str, ...] = (
        "return_1s", "return_5s", "return_10s", "return_30s",
        "tick_count_", "volume_1s", "volume_5s", "aggressor_imbalance_",
        "realized_volatility_10s", "spread_change_", "orderbook_imbalance_change_",
    )
    #: Features that only exist when a real orderbook sample was taken.
    orderbook_features: frozenset[str] = frozenset(
        {
            "spread_bps",
            "orderbook_imbalance",
            "bid_depth",
            "ask_depth",
            "depth_ratio",
            "microprice_edge_bps",
            "expected_slippage_bps",
            "liquidity_score",
        }
    )


class StrategyEligibilityEngine:
    """Evaluates the hard mask and the soft score for every catalogued strategy."""

    def __init__(
        self,
        *,
        registry: StrategyRegistry | None = None,
        ontology: StrategyOntology | None = None,
        config: EligibilityConfig | None = None,
        long_only: bool = True,
    ) -> None:
        self._registry = registry or default_strategy_registry()
        self._ontology = ontology or default_strategy_ontology()
        self._config = config or EligibilityConfig()
        # The account cannot trade 대주/공매도 (config/short_strategy_deployment.yaml
        # enabled=false). Blocking here is belt-and-braces: the promotion controller and
        # the borrow locate already gate it, and this must not be the only lock.
        self._long_only = bool(long_only)

    # -- public API --------------------------------------------------------- #
    def evaluate(
        self,
        context: MarketContext,
        *,
        election_inputs: Mapping[str, Any] | None = None,
        strategy_ids: Sequence[str] | None = None,
        macro_allowed: Iterable[str] = (),
        macro_blocked: Iterable[str] = (),
    ) -> StrategyEligibilityResult:
        specs = (
            tuple(spec for spec in self._registry.all_specs() if spec.strategy_id in set(strategy_ids))
            if strategy_ids is not None
            else self._registry.all_specs()
        )
        inputs: Mapping[str, Any] = election_inputs or {}
        allowed = tuple(macro_allowed or ())
        blocked = tuple(macro_blocked or ())
        market_states = _market_state_labels(context)
        return StrategyEligibilityResult(
            context_id=context.context_id,
            symbol=context.symbol_id,
            eligibilities=tuple(
                self._evaluate_one(
                    spec,
                    context,
                    election_inputs=inputs,
                    macro_allowed=allowed,
                    macro_blocked=blocked,
                    market_states=market_states,
                )
                for spec in specs
            ),
        )

    # -- internals ---------------------------------------------------------- #
    def _evaluate_one(
        self,
        spec: StrategySpec,
        context: MarketContext,
        *,
        election_inputs: Mapping[str, Any],
        macro_allowed: tuple[str, ...],
        macro_blocked: tuple[str, ...],
        market_states: frozenset[str],
    ) -> StrategyEligibility:
        blocks: list[str] = []

        # --- lifecycle / direction -------------------------------------- #
        if spec.lifecycle_state is StrategyLifecycleState.RETIRED:
            blocks.append(ELIGIBILITY_REASONS.LIFECYCLE_RETIRED)
        if self._long_only and spec.is_short:
            blocks.append(f"{ELIGIBILITY_REASONS.DIRECTION_NOT_PERMITTED}:SHORT")

        # --- allowedMarket / requiresSession ----------------------------- #
        if not spec.permits_market(context.market):
            blocks.append(f"{ELIGIBILITY_REASONS.MARKET_NOT_ALLOWED}:{context.market}")
        if not spec.permits_session(context.temporal.session_phase):
            blocks.append(
                f"{ELIGIBILITY_REASONS.SESSION_NOT_ALLOWED}:{context.temporal.session_phase}"
            )

        # --- forbiddenUnder --------------------------------------------- #
        # Only fires on a label that is actually present; see the module docstring.
        forbidden = set(self._ontology.forbidden_states(spec.strategy_id))
        for state in sorted(market_states & forbidden):
            blocks.append(f"{ELIGIBILITY_REASONS.FORBIDDEN_MARKET_STATE}:{state}")
        for state in sorted(market_states & set(self._ontology.no_new_entry_states)):
            blocks.append(f"{ELIGIBILITY_REASONS.NO_NEW_ENTRY_MARKET_STATE}:{state}")

        # --- macro family permission ------------------------------------ #
        if macro_allowed or macro_blocked:
            permitted = _macro_permits(spec.strategy_id, macro_allowed, macro_blocked)
            if permitted is False:
                blocks.append(f"{ELIGIBILITY_REASONS.MACRO_FAMILY_BLOCKED}:{spec.strategy_id}")

        # --- requires (context fields) ----------------------------------- #
        for name in context.missing(spec.required_context):
            blocks.append(f"{ELIGIBILITY_REASONS.MISSING_CONTEXT}:{name}")

        # --- requiresFeature -------------------------------------------- #
        snapshot = context.feature_snapshot
        for name in spec.required_features:
            if not _feature_present(snapshot, name):
                blocks.append(f"{ELIGIBILITY_REASONS.MISSING_FEATURE}:{name}")

        # --- required election inputs ------------------------------------ #
        for name in spec.required_election_inputs:
            if not _present(election_inputs.get(name)):
                blocks.append(f"{ELIGIBILITY_REASONS.MISSING_ELECTION_INPUT}:{name}")

        # --- requiresLiquidity ------------------------------------------- #
        liquidity = context.microstructure.liquidity_score
        if spec.min_liquidity_score is not None:
            if liquidity is None:
                blocks.append(ELIGIBILITY_REASONS.LIQUIDITY_UNKNOWN)
            elif liquidity < spec.min_liquidity_score:
                blocks.append(
                    f"{ELIGIBILITY_REASONS.LIQUIDITY_BELOW_FLOOR}:"
                    f"{liquidity:.3f}<{spec.min_liquidity_score:.3f}"
                )
        spread = context.microstructure.spread_bps
        if spec.max_spread_bps is not None:
            if spread is None:
                blocks.append(ELIGIBILITY_REASONS.SPREAD_UNKNOWN)
            elif spread > spec.max_spread_bps:
                blocks.append(
                    f"{ELIGIBILITY_REASONS.SPREAD_ABOVE_CEILING}:"
                    f"{spread:.1f}>{spec.max_spread_bps:.1f}"
                )

        # --- requiresHistory -------------------------------------------- #
        if spec.minimum_history_bars > 0:
            bars = int(context.data_quality.history_bar_count or 0)
            if bars < spec.minimum_history_bars:
                blocks.append(
                    f"{ELIGIBILITY_REASONS.HISTORY_INSUFFICIENT}:"
                    f"{bars}<{spec.minimum_history_bars}"
                )

        # --- requiresDataQuality ---------------------------------------- #
        blocks.extend(self._data_quality_blocks(spec, context))

        # --- price reference -------------------------------------------- #
        # Not a preference: without a point-in-time reference there is nothing to
        # measure a return against, and inventing one from a later quote is the leak the
        # shadow journal exists to prevent.
        if context.symbol.reference_price is None:
            blocks.append(ELIGIBILITY_REASONS.NO_REFERENCE_PRICE)

        score, supporting = self._compatibility(spec, context)
        return StrategyEligibility(
            strategy_id=spec.strategy_id,
            eligible=not blocks,
            compatibility_score=score,
            hard_block_reasons=tuple(dict.fromkeys(blocks)),
            supporting_relations=supporting,
            context_id=context.context_id,
        )

    def _data_quality_blocks(
        self, spec: StrategySpec, context: MarketContext
    ) -> list[str]:
        blocks: list[str] = []
        config = self._config
        quality = context.data_quality

        if quality.feature_completeness < config.minimum_feature_completeness:
            blocks.append(
                f"{ELIGIBILITY_REASONS.COMPLETENESS_BELOW_FLOOR}:"
                f"{quality.feature_completeness:.3f}<{config.minimum_feature_completeness:.3f}"
            )

        needs_tick = any(
            name.startswith(config.tick_feature_prefixes) or name in config.tick_feature_prefixes
            for name in spec.required_features
        )
        if needs_tick and not quality.second_level_data_ready:
            blocks.append(ELIGIBILITY_REASONS.TICK_WINDOW_NOT_READY)

        needs_book = bool(set(spec.required_features) & config.orderbook_features)
        if needs_book and context.microstructure.spread_bps is None:
            # A zero spread is an absent sample, not a free round trip — the context
            # builder already collapses that to ``None``.
            blocks.append(ELIGIBILITY_REASONS.ORDERBOOK_SAMPLE_MISSING)

        if config.max_tick_age_seconds is not None:
            age = quality.tick_freshness_sec
            if age is None or age > config.max_tick_age_seconds:
                blocks.append(f"{ELIGIBILITY_REASONS.STALE_DATA}:tick")
        if config.max_orderbook_age_seconds is not None and needs_book:
            age = quality.orderbook_freshness_sec
            if age is None or age > config.max_orderbook_age_seconds:
                blocks.append(f"{ELIGIBILITY_REASONS.STALE_DATA}:orderbook")
        return blocks

    def _compatibility(
        self, spec: StrategySpec, context: MarketContext
    ) -> tuple[float, tuple[str, ...]]:
        """Weighted mean of the soft relations that fired, in ``[-1, 1]``.

        Weighted by |weight| so a strong relation dominates a weak one, and normalised so
        a strategy with many relations is not automatically scored higher than one with
        few. Relations that did not fire contribute nothing to either the numerator or
        the denominator — an unmatched relation is silence, not a zero vote.
        """
        relations = self._ontology.soft_relations(spec.strategy_id)
        if not relations:
            return 0.0, ()
        flat = context.flat()
        numerator = 0.0
        denominator = 0.0
        supporting: list[str] = []
        for relation in relations:
            if relation.weight == 0.0:
                continue
            value = flat.get(relation.field)
            if value is None:
                continue
            # A boolean field is matched by the label form ("TRUE"), which is how the
            # session-window relations are declared.
            probe = "TRUE" if value is True else "FALSE" if value is False else value
            if not relation.matches(probe):
                continue
            weight = abs(float(relation.weight))
            numerator += float(relation.weight)
            denominator += weight
            supporting.append(
                f"{relation.relation}:{relation.field}"
                f"{'+' if relation.weight > 0 else '-'}"
            )
        if denominator <= 0.0:
            return 0.0, ()
        score = numerator / denominator
        if not math.isfinite(score):
            return 0.0, ()
        return max(-1.0, min(1.0, score)), tuple(supporting)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return bool(value.strip())
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True  # a non-numeric object that is not None counts as supplied
    return math.isfinite(number)


def _feature_present(snapshot: Mapping[str, Any], name: str) -> bool:
    """Is this ``TechnicalFeatureSet`` field usable in the snapshot?

    ``microprice_edge_bps`` and the other derived properties are not dataclass fields, so
    they never appear in the snapshot dict. They are identities over fields that DO
    appear, so presence is checked against their inputs.
    """
    if name == "microprice_edge_bps":
        return _present(snapshot.get("spread_bps")) and _present(
            snapshot.get("orderbook_imbalance")
        )
    if name in {"vwap_zscore", "residual_volatility_bps"}:
        return _present(snapshot.get("realized_volatility")) or _present(
            snapshot.get("realized_volatility_10s")
        )
    if name == "tick_data_ready":
        return _present(snapshot.get("second_data_ready"))
    return _present(snapshot.get(name))


def _market_state_labels(context: MarketContext) -> frozenset[str]:
    """Upper-cased market/risk state labels present on the context."""
    labels: set[str] = set()
    for value in (context.macro.market_regime, context.macro.risk_regime):
        text = str(value or "").strip().upper()
        if text:
            labels.add(text)
    return frozenset(labels)


def _macro_permits(
    strategy_id: str, allowed: tuple[str, ...], blocked: tuple[str, ...]
) -> bool | None:
    """Delegate to the existing macro family permission check.

    Reused rather than reimplemented: ``MACRO_FAMILY_BY_STRATEGY`` is where the coarse
    methodology vocabulary legitimately still lives, and a second copy of the mapping is
    how the two would drift.
    """
    try:
        from app.technical.strategy_algorithms import macro_strategy_permitted

        return macro_strategy_permitted(strategy_id, allowed, blocked)
    except Exception:  # noqa: BLE001 - an unanswerable check is not a withdrawal.
        return None
