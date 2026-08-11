"""Unified market context: one snapshot per (symbol, cycle), with an id.

Import from here rather than from the submodules, so the group layout stays an internal
detail of the package.
"""

from app.context.context_builder import (
    CONTEXT_NO_REFERENCE_PRICE,
    CONTEXT_NO_TICK_WINDOW,
    CONTEXT_SESSION_UNRESOLVED,
    MarketContextBuilder,
    SymbolContextInputs,
)
from app.context.context_store import (
    MarketContextStore,
    default_context_store,
    reset_default_context_store,
)
from app.context.market_context import (
    ContextIdentity,
    CrossSectionalContext,
    DataQualityContext,
    EventContext,
    FeatureSource,
    MacroContext,
    MarketContext,
    MicrostructureContext,
    PriceGeometryContext,
    SymbolContext,
    TemporalContext,
    declared_context_fields,
    new_context_id,
)
from app.context.microstructure_context import CONTEXT_NO_ORDERBOOK_SAMPLE

__all__ = [
    "CONTEXT_NO_ORDERBOOK_SAMPLE",
    "CONTEXT_NO_REFERENCE_PRICE",
    "CONTEXT_NO_TICK_WINDOW",
    "CONTEXT_SESSION_UNRESOLVED",
    "ContextIdentity",
    "CrossSectionalContext",
    "DataQualityContext",
    "EventContext",
    "FeatureSource",
    "MacroContext",
    "MarketContext",
    "MarketContextBuilder",
    "MarketContextStore",
    "MicrostructureContext",
    "PriceGeometryContext",
    "SymbolContext",
    "SymbolContextInputs",
    "TemporalContext",
    "declared_context_fields",
    "default_context_store",
    "new_context_id",
    "reset_default_context_store",
]
