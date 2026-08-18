"""Domestic (Korean) market context, conditioned on the global layer.

``Calendar/Session -> Global -> DOMESTIC -> Sector -> Stock``.

The rule this layer enforces
----------------------------
A weak global tape is not a domestic sell signal. The requirement is stated once, here,
and carried in the output rather than left to each caller:

* :attr:`DomesticContext.direction` is measured from **domestic** prices, breadth and
  flow only.
* :attr:`DomesticContext.global_confirmation` says whether the domestic tape agrees with
  the global one, and :attr:`DomesticContext.global_conflict` whether it contradicts it.
* :meth:`DomesticContext.confirms_global_weakness` is the *only* sanctioned way to turn
  global weakness into a domestic bearish conclusion, and it requires domestic direction,
  breadth and flow to agree — three independent domestic witnesses, not one foreign one.

Venue divergence
----------------
KRX and NXT quote the same securities on different books. A persistent spread between
them is not an arbitrage the system trades; it is a *data quality and liquidity* signal —
when the two venues disagree the consolidated price is less trustworthy and any decision
resting on it should be smaller. That is what ``venue_divergence`` feeds.

Purity
------
The builder performs no IO. Everything arrives in :class:`DomesticContextInputs`, so the
same inputs always produce the same context and a stored context can be replayed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.context.global_context import GlobalContext

__all__ = [
    "DOMESTIC_INSUFFICIENT_BREADTH",
    "DOMESTIC_NO_FLOW_DATA",
    "DOMESTIC_NO_INDEX_DATA",
    "DOMESTIC_VENUE_DIVERGENCE",
    "DomesticContext",
    "DomesticContextBuilder",
    "DomesticContextInputs",
    "VenueQuote",
]

DOMESTIC_NO_INDEX_DATA = "DOMESTIC_NO_INDEX_DATA"
DOMESTIC_INSUFFICIENT_BREADTH = "DOMESTIC_INSUFFICIENT_BREADTH"
DOMESTIC_NO_FLOW_DATA = "DOMESTIC_NO_FLOW_DATA"
DOMESTIC_VENUE_DIVERGENCE = "DOMESTIC_VENUE_DIVERGENCE"
DOMESTIC_SINGLE_VENUE = "DOMESTIC_SINGLE_VENUE"

#: Spread between the KRX and NXT consolidated mid, in bps, at which
#: ``venue_divergence`` reads 1.0. 25bps is several times a normal KRX top-of-book spread
#: on a liquid name, so an ordinary quote difference stays near zero and a genuine
#: dislocation saturates.
VENUE_DIVERGENCE_REFERENCE_BPS = 25.0

#: Fractional index move that maps to |direction| = tanh(1) on the domestic scale. KOSPI
#: sessions of +-1% are ordinary; +-3% are not.
DOMESTIC_DIRECTION_SCALE = 0.010

#: Net flow, as a fraction of session trading value, that maps to |flow| = tanh(1).
DOMESTIC_FLOW_SCALE = 0.05

#: Minimum advancing+declining count before breadth is reported at all. Below it the
#: ratio is dominated by whichever handful of names happened to be sampled.
MINIMUM_BREADTH_SAMPLE = 20


@dataclass(frozen=True)
class VenueQuote:
    """One venue's view of the domestic market."""

    venue: str
    #: Consolidated index level or volume-weighted mid across the sampled universe.
    mid: float | None = None
    trading_value: float | None = None
    spread_bps: float | None = None
    depth: float | None = None
    observed_at: datetime | None = None


@dataclass(frozen=True)
class DomesticContextInputs:
    """Everything the domestic layer needs, supplied by the caller.

    Every field is optional. A field the caller cannot answer stays ``None`` and shows up
    as reduced confidence and a reason code, never as a zero.
    """

    kospi_return: float | None = None
    kosdaq_return: float | None = None
    kospi_volatility: float | None = None
    kosdaq_volatility: float | None = None
    advancing_count: int | None = None
    declining_count: int | None = None
    unchanged_count: int | None = None
    breadth_momentum: float | None = None
    total_trading_value: float | None = None
    average_trading_value: float | None = None
    realized_volatility: float | None = None
    #: Net buy value by investor class, in the same currency as ``total_trading_value``.
    foreign_flow: float | None = None
    institution_flow: float | None = None
    retail_flow: float | None = None
    program_flow: float | None = None
    average_spread_bps: float | None = None
    average_depth: float | None = None
    sector_dispersion: float | None = None
    #: Returns of the strongest sectors, used for the leadership reading.
    sector_returns: Mapping[str, float] = field(default_factory=dict)
    venues: Sequence[VenueQuote] = ()
    symbol_count: int = 0


@dataclass(frozen=True)
class DomesticContext:
    """Domestic market state, with its relationship to the global layer attached."""

    captured_at: datetime
    context_id: str
    global_context_id: str | None = None
    direction: float | None = None
    breadth: float | None = None
    liquidity: float | None = None
    volatility: float | None = None
    flow: float | None = None
    leadership: float | None = None
    venue_divergence: float | None = None
    confidence: float = 0.0
    #: Agreement with the global layer in [-1, 1]: +1 both risk-on, -1 opposed.
    global_agreement: float | None = None
    components: Mapping[str, float] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    # -- global relationship ------------------------------------------------ #
    @property
    def global_confirmation(self) -> bool:
        return self.global_agreement is not None and self.global_agreement > 0.25

    @property
    def global_conflict(self) -> bool:
        return self.global_agreement is not None and self.global_agreement < -0.25

    def confirms_global_weakness(self, *, threshold: float = 0.0) -> bool:
        """Is domestic evidence sufficient to act on a weak global tape?

        Requires three independent domestic witnesses — price direction, market breadth
        and investor flow — to be negative at once. One of them alone is noise; the
        global tape does not count as a witness to a domestic condition.
        """
        witnesses = (self.direction, self.breadth, self.flow)
        if any(value is None for value in witnesses):
            return False
        return all(float(value) < threshold for value in witnesses)  # type: ignore[arg-type]

    def numeric_features(self) -> dict[str, float]:
        values: dict[str, float] = {"domestic_confidence": self.confidence}
        for name, value in (
            ("domestic_direction", self.direction),
            ("domestic_breadth", self.breadth),
            ("domestic_liquidity", self.liquidity),
            ("domestic_volatility", self.volatility),
            ("domestic_flow", self.flow),
            ("domestic_leadership", self.leadership),
            ("domestic_venue_divergence", self.venue_divergence),
            ("domestic_global_agreement", self.global_agreement),
        ):
            if value is not None:
                values[name] = float(value)
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "captured_at": _aware(self.captured_at).isoformat(),
            "global_context_id": self.global_context_id,
            "direction": self.direction,
            "breadth": self.breadth,
            "liquidity": self.liquidity,
            "volatility": self.volatility,
            "flow": self.flow,
            "leadership": self.leadership,
            "venue_divergence": self.venue_divergence,
            "confidence": self.confidence,
            "global_agreement": self.global_agreement,
            "global_confirmation": self.global_confirmation,
            "global_conflict": self.global_conflict,
            "components": dict(self.components),
            "reason_codes": list(self.reason_codes),
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _squash(value: float, scale: float) -> float:
    return math.tanh(value / scale) if scale > 0 else 0.0


def _mean(values: Sequence[float]) -> float | None:
    usable = [value for raw in values if (value := _finite(raw)) is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


class DomesticContextBuilder:
    """Builds one :class:`DomesticContext` per cycle. Pure; no IO."""

    def build(
        self,
        inputs: DomesticContextInputs,
        *,
        captured_at: datetime,
        global_context: GlobalContext | None = None,
        context_id: str | None = None,
    ) -> DomesticContext:
        now = _aware(captured_at)
        reasons: list[str] = []
        components: dict[str, float] = {}

        direction = self._direction(inputs, reasons, components)
        breadth = self._breadth(inputs, reasons, components)
        liquidity = self._liquidity(inputs, components)
        volatility = self._volatility(inputs, components)
        flow = self._flow(inputs, reasons, components)
        leadership = self._leadership(inputs, components)
        divergence = self._venue_divergence(inputs, reasons, components)

        answered = [
            value
            for value in (direction, breadth, liquidity, volatility, flow, leadership)
            if value is not None
        ]
        confidence = round(len(answered) / 6.0, 6)
        # A dislocated pair of venues means the price everything else rests on is less
        # trustworthy, so it lowers confidence rather than merely being reported.
        if divergence is not None:
            confidence = round(confidence * (1.0 - min(0.5, divergence * 0.5)), 6)

        agreement = self._global_agreement(direction, global_context)

        return DomesticContext(
            captured_at=now,
            context_id=context_id or _context_id(now),
            global_context_id=global_context.context_id if global_context else None,
            direction=direction,
            breadth=breadth,
            liquidity=liquidity,
            volatility=volatility,
            flow=flow,
            leadership=leadership,
            venue_divergence=divergence,
            confidence=confidence,
            global_agreement=agreement,
            components=components,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    # -- components --------------------------------------------------------- #
    def _direction(
        self,
        inputs: DomesticContextInputs,
        reasons: list[str],
        components: dict[str, float],
    ) -> float | None:
        returns = [
            value
            for value in (
                _finite(inputs.kospi_return),
                _finite(inputs.kosdaq_return),
            )
            if value is not None
        ]
        if not returns:
            reasons.append(DOMESTIC_NO_INDEX_DATA)
            return None
        raw = sum(returns) / len(returns)
        components["index_return"] = round(raw, 8)
        return round(_squash(raw, DOMESTIC_DIRECTION_SCALE), 6)

    def _breadth(
        self,
        inputs: DomesticContextInputs,
        reasons: list[str],
        components: dict[str, float],
    ) -> float | None:
        advancing = inputs.advancing_count
        declining = inputs.declining_count
        if advancing is None or declining is None:
            reasons.append(DOMESTIC_INSUFFICIENT_BREADTH)
            return None
        total = int(advancing) + int(declining)
        if total < MINIMUM_BREADTH_SAMPLE:
            reasons.append(DOMESTIC_INSUFFICIENT_BREADTH)
            return None
        components["advancing"] = float(advancing)
        components["declining"] = float(declining)
        # [-1, 1]: +1 every name up, -1 every name down.
        return round((int(advancing) - int(declining)) / total, 6)

    def _liquidity(
        self, inputs: DomesticContextInputs, components: dict[str, float]
    ) -> float | None:
        """[0, 1]. Turnover relative to its own average, discounted by the spread."""
        total = _finite(inputs.total_trading_value)
        average = _finite(inputs.average_trading_value)
        ratio: float | None = None
        if total is not None and average and average > 0.0:
            ratio = total / average
            components["turnover_ratio"] = round(ratio, 6)
        spread = _finite(inputs.average_spread_bps)
        if ratio is None and spread is None:
            return None
        turnover_term = min(1.0, ratio) if ratio is not None else 0.5
        # A wide market-wide spread is a liquidity failure regardless of turnover.
        spread_term = (
            max(0.0, 1.0 - spread / (VENUE_DIVERGENCE_REFERENCE_BPS * 4.0))
            if spread is not None
            else 0.5
        )
        return round(max(0.0, min(1.0, 0.5 * turnover_term + 0.5 * spread_term)), 6)

    def _volatility(
        self, inputs: DomesticContextInputs, components: dict[str, float]
    ) -> float | None:
        candidates = [
            _finite(inputs.realized_volatility),
            _finite(inputs.kospi_volatility),
            _finite(inputs.kosdaq_volatility),
        ]
        value = _mean([item for item in candidates if item is not None])
        if value is None:
            return None
        components["realized_volatility"] = round(value, 8)
        return round(value, 8)

    def _flow(
        self,
        inputs: DomesticContextInputs,
        reasons: list[str],
        components: dict[str, float],
    ) -> float | None:
        """[-1, 1] net institutional-plus-foreign flow as a share of turnover.

        Retail flow is recorded but excluded from the score: it is the residual of the
        other two by construction in the KRX投資者 breakdown, so including it would
        double-count the same money with the sign flipped.
        """
        foreign = _finite(inputs.foreign_flow)
        institution = _finite(inputs.institution_flow)
        if foreign is None and institution is None:
            reasons.append(DOMESTIC_NO_FLOW_DATA)
            return None
        net = (foreign or 0.0) + (institution or 0.0)
        components["net_smart_flow"] = round(net, 4)
        retail = _finite(inputs.retail_flow)
        if retail is not None:
            components["retail_flow"] = round(retail, 4)
        program = _finite(inputs.program_flow)
        if program is not None:
            components["program_flow"] = round(program, 4)
        turnover = _finite(inputs.total_trading_value)
        if turnover and turnover > 0.0:
            return round(_squash(net / turnover, DOMESTIC_FLOW_SCALE), 6)
        # Without a turnover denominator the flow cannot be scaled; report its sign only,
        # which is honest about what is known rather than inventing a magnitude.
        return 1.0 if net > 0 else (-1.0 if net < 0 else 0.0)

    def _leadership(
        self, inputs: DomesticContextInputs, components: dict[str, float]
    ) -> float | None:
        """[0, 1] concentration of the advance in the leading sectors.

        High leadership means a narrow market: a few sectors carrying the index. That is
        not automatically bad, but it changes which strategies work, so it is measured
        rather than folded into breadth.
        """
        returns = [
            value
            for raw in inputs.sector_returns.values()
            if (value := _finite(raw)) is not None
        ]
        if len(returns) < 2:
            return None
        positive = sorted((value for value in returns if value > 0.0), reverse=True)
        if not positive:
            components["leading_sector_count"] = 0.0
            return 0.0
        total = sum(positive)
        top = sum(positive[: max(1, len(returns) // 4)])
        components["leading_sector_count"] = float(len(positive))
        return round(top / total, 6) if total > 0.0 else None

    def _venue_divergence(
        self,
        inputs: DomesticContextInputs,
        reasons: list[str],
        components: dict[str, float],
    ) -> float | None:
        mids = [
            (quote.venue, value)
            for quote in inputs.venues
            if (value := _finite(quote.mid)) is not None and value > 0.0
        ]
        if len(mids) < 2:
            if inputs.venues:
                reasons.append(DOMESTIC_SINGLE_VENUE)
            return None
        levels = [value for _, value in mids]
        reference = sum(levels) / len(levels)
        spread_bps = (max(levels) - min(levels)) / reference * 10_000.0
        components["venue_spread_bps"] = round(spread_bps, 4)
        divergence = min(1.0, spread_bps / VENUE_DIVERGENCE_REFERENCE_BPS)
        if divergence >= 1.0:
            reasons.append(DOMESTIC_VENUE_DIVERGENCE)
        return round(divergence, 6)

    def _global_agreement(
        self, direction: float | None, global_context: GlobalContext | None
    ) -> float | None:
        if direction is None or global_context is None:
            return None
        global_direction = global_context.direction
        if global_direction is None:
            return None
        # Product of the two directions, normalised by the larger magnitude: +1 when both
        # lean the same way with comparable conviction, -1 when they oppose. Using the
        # product alone would report near-zero agreement whenever either side is calm,
        # which is not the same claim as disagreement.
        magnitude = max(abs(direction), abs(global_direction))
        if magnitude <= 0.0:
            return 0.0
        return round(
            max(-1.0, min(1.0, (direction * global_direction) / (magnitude**2))), 6
        )


def _context_id(moment: datetime) -> str:
    from uuid import uuid4

    return f"dctx-{moment.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
