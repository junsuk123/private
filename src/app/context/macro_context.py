"""Macro slice of a :class:`~app.context.market_context.MarketContext`.

Reads the macro reasoner's result object by attribute rather than by type, because the
live bundle is assembled in ``app.graph.macro_micro_feed`` and the replay bundle in
``app.graph.macro_micro_replay``; both expose the same names and neither shares a base
class. Duck typing here is deliberate — importing either would make the context layer
depend on the reasoning layer it is supposed to feed.

Nothing is defaulted. A macro result that cannot answer a question leaves the field
``None``, and the ontology's ``requiresFeature`` relations are what decide whether that
absence blocks a strategy.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.context.market_context import FeatureSource, MacroContext

__all__ = ["build_macro_context"]

_SOURCE = "macro_reasoner"


def _enum_value(raw: Any) -> str | None:
    """``StrEnum``/``Enum``/plain-string label, or ``None`` for an unanswered one."""
    if raw is None:
        return None
    value = getattr(raw, "value", raw)
    text = str(value or "").strip()
    return text or None


def _number(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _diagnostic(macro: Any, name: str) -> Any:
    diagnostics = getattr(macro, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        return diagnostics.get(name)
    return None


def build_macro_context(
    macro: Any,
    *,
    age_seconds: float | None = None,
) -> tuple[MacroContext, dict[str, FeatureSource]]:
    """Macro slice plus its provenance.

    ``macro`` is the ``macro_result`` of the live macro/micro bundle. ``None`` yields an
    empty context with no sources, which is the correct representation of "the macro
    layer did not run this cycle" — as opposed to "the macro layer said neutral".
    """
    if macro is None:
        return MacroContext(), {}

    context = MacroContext(
        market_regime=_enum_value(getattr(macro, "market_regime", None)),
        risk_regime=_enum_value(getattr(macro, "risk_regime", None))
        or _enum_value(_diagnostic(macro, "risk_regime")),
        index_return=_number(getattr(macro, "index_return", None))
        or _number(_diagnostic(macro, "index_return")),
        market_volatility=_number(getattr(macro, "market_volatility", None))
        or _number(_diagnostic(macro, "market_volatility")),
        fx_regime=_enum_value(getattr(macro, "fx_regime", None))
        or _enum_value(_diagnostic(macro, "fx_regime")),
        rate_regime=_enum_value(getattr(macro, "rate_regime", None))
        or _enum_value(_diagnostic(macro, "rate_regime")),
        risk_on_off_score=_number(getattr(macro, "risk_on_off_score", None))
        or _number(_diagnostic(macro, "risk_on_off_score")),
        change_point_probability=_number(
            getattr(macro, "change_point_probability", None)
        ),
        regime_stability=_number(getattr(macro, "regime_stability", None)),
        volatility_percentile=_number(getattr(macro, "volatility_percentile", None)),
    )
    sources = {
        name: FeatureSource(source=_SOURCE, age_seconds=age_seconds)
        for name, value in (
            ("market_regime", context.market_regime),
            ("risk_regime", context.risk_regime),
            ("index_return", context.index_return),
            ("market_volatility", context.market_volatility),
            ("fx_regime", context.fx_regime),
            ("rate_regime", context.rate_regime),
            ("risk_on_off_score", context.risk_on_off_score),
            ("change_point_probability", context.change_point_probability),
            ("regime_stability", context.regime_stability),
            ("volatility_percentile", context.volatility_percentile),
        )
        if value is not None
    }
    return context, sources
