"""Cross-sectional slice: where this symbol sits inside the scanned universe.

These are the only context fields a single symbol's own tape cannot produce, which is
why they arrive from the electing layer (sector ranks, breadth) rather than from the
feature frame. ``app.technical.strategy_algorithms.ElectionContext`` already documents
this asymmetry for ``residual_*``; the same rule applies here.

Dispersion is computed from the peer returns actually supplied. Passing one peer yields
``None`` rather than 0.0: a dispersion of zero is a market where every name moved
identically, and that is a different claim from "we only looked at one name".
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from app.context.market_context import CrossSectionalContext, FeatureSource

__all__ = ["build_cross_sectional_context"]

_SOURCE = "cross_sectional"


def _number(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int(raw: Any) -> int | None:
    number = _number(raw)
    return int(number) if number is not None else None


def _dispersion(peer_returns: Sequence[float] | None) -> float | None:
    if not peer_returns:
        return None
    finite = [value for raw in peer_returns if (value := _number(raw)) is not None]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(max(0.0, variance))


def _leadership(rank: int | None, candidate_count: int | None) -> float | None:
    """1.0 for the strongest name in the peer set, 0.0 for the weakest.

    Rank is 1-based (that is what ``ElectionContext.sector_rank`` carries). A rank with
    no candidate count is unscoreable — "3rd" means nothing without "of how many".
    """
    if rank is None or candidate_count is None or candidate_count <= 1:
        return None
    clamped = max(1, min(int(candidate_count), int(rank)))
    return 1.0 - (clamped - 1) / (int(candidate_count) - 1)


def build_cross_sectional_context(
    *,
    election_inputs: Mapping[str, Any] | None = None,
    peer_returns: Sequence[float] | None = None,
    market_breadth: Any = None,
    age_seconds: float | None = None,
) -> tuple[CrossSectionalContext, dict[str, FeatureSource]]:
    inputs: Mapping[str, Any] = election_inputs or {}
    rank = _int(inputs.get("sector_rank"))
    candidate_count = _int(inputs.get("sector_candidate_count"))
    context = CrossSectionalContext(
        market_breadth=_number(market_breadth),
        dispersion=_dispersion(peer_returns),
        sector_strength=_number(inputs.get("sector_strength")),
        relative_strength_rank=rank,
        market_leadership_score=_leadership(rank, candidate_count),
        sector_candidate_count=candidate_count,
    )
    sources = {
        name: FeatureSource(source=_SOURCE, age_seconds=age_seconds)
        for name, value in (
            ("market_breadth", context.market_breadth),
            ("dispersion", context.dispersion),
            ("sector_strength", context.sector_strength),
            ("relative_strength_rank", context.relative_strength_rank),
            ("market_leadership_score", context.market_leadership_score),
            ("sector_candidate_count", context.sector_candidate_count),
        )
        if value is not None
    }
    return context, sources
