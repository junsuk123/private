"""The selector's view of the ontology: a mask, a score, and nothing else.

Deliberately thin. Its whole job is to keep ``StrategySelectorV2`` from importing the
ontology package directly, so that:

* the selector cannot reach any ontology API other than eligibility — there is no import
  path from the selector to a reasoner that could return an intent or a strategy pick;
* the mask can be swapped for a stored one during replay without the selector knowing.

``STRATEGY_ONTOLOGY_MASK_V2_ENABLED`` off yields an all-pass mask with an explicit reason
code, so "the mask is disabled" is visible in the diagnostics rather than looking like
"every strategy was eligible".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.context.market_context import MarketContext
from app.ontology.strategy_eligibility import (
    StrategyEligibility,
    StrategyEligibilityEngine,
    StrategyEligibilityResult,
)

__all__ = ["MASK_DISABLED", "OntologyStrategyMask"]

MASK_DISABLED = "ONTOLOGY_MASK_V2_DISABLED"


@dataclass(frozen=True)
class OntologyStrategyMask:
    """Wraps :class:`StrategyEligibilityEngine` for the selector."""

    engine: StrategyEligibilityEngine
    enabled: bool = True

    def evaluate(
        self,
        context: MarketContext,
        *,
        election_inputs: Mapping[str, Any] | None = None,
        strategy_ids: Sequence[str] | None = None,
        macro_allowed: Iterable[str] = (),
        macro_blocked: Iterable[str] = (),
    ) -> StrategyEligibilityResult:
        if self.enabled:
            return self.engine.evaluate(
                context,
                election_inputs=election_inputs,
                strategy_ids=strategy_ids,
                macro_allowed=macro_allowed,
                macro_blocked=macro_blocked,
            )
        # All-pass, but the soft score is still computed: disabling the MASK must not
        # also silence the compatibility evidence, which is a separate term of the
        # utility and has no power to block anything.
        measured = self.engine.evaluate(
            context,
            election_inputs=election_inputs,
            strategy_ids=strategy_ids,
            macro_allowed=macro_allowed,
            macro_blocked=macro_blocked,
        )
        return StrategyEligibilityResult(
            context_id=measured.context_id,
            symbol=measured.symbol,
            eligibilities=tuple(
                StrategyEligibility(
                    strategy_id=item.strategy_id,
                    eligible=True,
                    compatibility_score=item.compatibility_score,
                    hard_block_reasons=(MASK_DISABLED, *item.hard_block_reasons),
                    supporting_relations=item.supporting_relations,
                    context_id=item.context_id,
                )
                for item in measured.eligibilities
            ),
        )
