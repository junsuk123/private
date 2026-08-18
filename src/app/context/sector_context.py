"""Sector context — the layer between the domestic market and a single name.

``Calendar/Session -> Global -> Domestic -> SECTOR -> Stock``.

Why a sector layer exists at all
--------------------------------
Without it, a stock's relative strength is measured against the whole market, and a semi
name in a semi rally scores as "strong" when it is merely present. The sector layer makes
two distinctions the stock layer cannot:

* **Sector versus market.** ``relative_strength`` here is the sector's move net of its
  beta to the domestic market, so a sector that simply tracks the index does not read as
  leadership.
* **Sector versus its own members.** ``leader_concentration`` says whether the sector's
  move is broad or is two names dragging the average. A breakout in a sector whose
  advance is one name is a different trade from the same breakout in a sector where four
  of five members are up, and ``breadth`` plus ``leader_concentration`` separate them.

``global_alignment`` carries the cross-market link: the semiconductor sector's alignment
with SOX is a genuine relationship, and the ontology declares it as an ``INFLUENCES``
edge. It is an input to the sector's confidence, never an instruction.

Purity
------
No IO. Member observations arrive from the caller, which is what lets a sector context be
replayed exactly as it was built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.context.cross_market import BetaEstimate, RelativeStrength, relative_strength
from app.context.domestic_context import DomesticContext
from app.context.global_context import GlobalContext

__all__ = [
    "SECTOR_NO_MEMBERS",
    "SECTOR_THIN_MEMBERSHIP",
    "SectorContext",
    "SectorContextBuilder",
    "SectorMemberObservation",
]

SECTOR_NO_MEMBERS = "SECTOR_NO_MEMBERS"
SECTOR_THIN_MEMBERSHIP = "SECTOR_THIN_MEMBERSHIP"
SECTOR_NO_VOLUME_BASELINE = "SECTOR_NO_VOLUME_BASELINE"

#: Below this member count the cross-sectional statistics (breadth, concentration,
#: dispersion) describe the sample rather than the sector, and are suppressed.
MINIMUM_SECTOR_MEMBERS = 3


@dataclass(frozen=True)
class SectorMemberObservation:
    """One constituent's contribution to its sector this cycle."""

    ticker: str
    session_return: float | None = None
    volume: float | None = None
    average_volume: float | None = None
    realized_volatility: float | None = None
    trading_value: float | None = None
    foreign_flow: float | None = None
    #: Return history aligned with the market history, for the sector beta estimate.
    return_history: Sequence[float] = ()


@dataclass(frozen=True)
class SectorContext:
    """One sector's state at an instant."""

    captured_at: datetime
    context_id: str
    sector: str
    market_group: str
    domestic_context_id: str | None = None
    sector_return: float | None = None
    breadth: float | None = None
    volume_z: float | None = None
    volatility: float | None = None
    relative_strength: float | None = None
    foreign_flow: float | None = None
    leader_strength: float | None = None
    leader_concentration: float | None = None
    global_alignment: float | None = None
    confidence: float = 0.0
    member_count: int = 0
    beta: BetaEstimate | None = None
    reason_codes: tuple[str, ...] = ()
    components: Mapping[str, float] = field(default_factory=dict)

    def numeric_features(self) -> dict[str, float]:
        values: dict[str, float] = {"sector_confidence": self.confidence}
        for name, value in (
            ("sector_return", self.sector_return),
            ("sector_breadth", self.breadth),
            ("sector_volume_z", self.volume_z),
            ("sector_volatility", self.volatility),
            ("sector_relative_strength", self.relative_strength),
            ("sector_foreign_flow", self.foreign_flow),
            ("sector_leader_strength", self.leader_strength),
            ("sector_leader_concentration", self.leader_concentration),
            ("sector_global_alignment", self.global_alignment),
        ):
            if value is not None:
                values[name] = float(value)
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "captured_at": _aware(self.captured_at).isoformat(),
            "sector": self.sector,
            "market_group": self.market_group,
            "domestic_context_id": self.domestic_context_id,
            "return": self.sector_return,
            "breadth": self.breadth,
            "volume_z": self.volume_z,
            "volatility": self.volatility,
            "relative_strength": self.relative_strength,
            "foreign_flow": self.foreign_flow,
            "leader_strength": self.leader_strength,
            "leader_concentration": self.leader_concentration,
            "global_alignment": self.global_alignment,
            "confidence": self.confidence,
            "member_count": self.member_count,
            "beta": self.beta.as_dict() if self.beta else None,
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


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


class SectorContextBuilder:
    """Builds one :class:`SectorContext` per sector per cycle. Pure; no IO."""

    def build(
        self,
        sector: str,
        members: Sequence[SectorMemberObservation],
        *,
        captured_at: datetime,
        market_group: str = "KR",
        market_return: float | None = None,
        market_return_history: Sequence[float] = (),
        domestic_context: DomesticContext | None = None,
        global_context: GlobalContext | None = None,
        global_group: str | None = None,
        context_id: str | None = None,
    ) -> SectorContext:
        now = _aware(captured_at)
        reasons: list[str] = []
        components: dict[str, float] = {}
        name = str(sector or "").strip() or "UNKNOWN"

        returns = [
            value
            for member in members
            if (value := _finite(member.session_return)) is not None
        ]
        if not returns:
            reasons.append(SECTOR_NO_MEMBERS)
            return SectorContext(
                captured_at=now,
                context_id=context_id or _context_id(now, name),
                sector=name,
                market_group=str(market_group).upper(),
                domestic_context_id=(
                    domestic_context.context_id if domestic_context else None
                ),
                member_count=len(members),
                reason_codes=tuple(reasons),
            )
        thin = len(returns) < MINIMUM_SECTOR_MEMBERS
        if thin:
            reasons.append(SECTOR_THIN_MEMBERSHIP)

        sector_return = _mean(returns)
        components["member_return_dispersion"] = round(_stdev(returns) or 0.0, 8)

        breadth = None if thin else self._breadth(returns)
        volume_z = self._volume_z(members, reasons, components)
        volatility = _mean(
            [
                value
                for member in members
                if (value := _finite(member.realized_volatility)) is not None
            ]
        )
        strength = self._relative_strength(
            sector_return, market_return, members, market_return_history
        )
        foreign_flow = self._foreign_flow(members, components)
        leader_strength, concentration = self._leadership(returns, thin)
        alignment = self._global_alignment(sector_return, global_context, global_group)

        answered = [
            value
            for value in (
                sector_return,
                breadth,
                volume_z,
                volatility,
                strength.value if strength else None,
                leader_strength,
            )
            if value is not None
        ]
        confidence = round(
            (len(answered) / 6.0) * (0.6 if thin else 1.0),
            6,
        )

        return SectorContext(
            captured_at=now,
            context_id=context_id or _context_id(now, name),
            sector=name,
            market_group=str(market_group).upper(),
            domestic_context_id=(
                domestic_context.context_id if domestic_context else None
            ),
            sector_return=None if sector_return is None else round(sector_return, 8),
            breadth=breadth,
            volume_z=volume_z,
            volatility=None if volatility is None else round(volatility, 8),
            relative_strength=strength.value if strength else None,
            foreign_flow=foreign_flow,
            leader_strength=leader_strength,
            leader_concentration=concentration,
            global_alignment=alignment,
            confidence=confidence,
            member_count=len(members),
            beta=strength.beta if strength else None,
            reason_codes=tuple(dict.fromkeys(reasons)),
            components=components,
        )

    # -- components --------------------------------------------------------- #
    @staticmethod
    def _breadth(returns: Sequence[float]) -> float:
        advancing = sum(1 for value in returns if value > 0.0)
        declining = sum(1 for value in returns if value < 0.0)
        total = advancing + declining
        return round((advancing - declining) / total, 6) if total else 0.0

    @staticmethod
    def _volume_z(
        members: Sequence[SectorMemberObservation],
        reasons: list[str],
        components: dict[str, float],
    ) -> float | None:
        """Cross-member mean of each name's volume relative to its own average.

        Ratios are averaged in log space so a single name printing 20x its average does
        not carry the sector on its own — a linear mean of ratios is dominated by the
        largest term, which is the failure mode this statistic exists to avoid.
        """
        ratios: list[float] = []
        for member in members:
            volume = _finite(member.volume)
            average = _finite(member.average_volume)
            if volume is None or not average or average <= 0.0 or volume <= 0.0:
                continue
            ratios.append(math.log(volume / average))
        if not ratios:
            reasons.append(SECTOR_NO_VOLUME_BASELINE)
            return None
        components["volume_log_ratio_members"] = float(len(ratios))
        return round(sum(ratios) / len(ratios), 6)

    @staticmethod
    def _relative_strength(
        sector_return: float | None,
        market_return: float | None,
        members: Sequence[SectorMemberObservation],
        market_history: Sequence[float],
    ) -> RelativeStrength | None:
        if sector_return is None or market_return is None:
            return None
        # The sector's own return history is the member average, aligned bar-for-bar.
        histories = [
            list(member.return_history) for member in members if member.return_history
        ]
        sector_history: list[float] = []
        if histories:
            length = min(len(history) for history in histories)
            sector_history = [
                sum(history[-length:][index] for history in histories) / len(histories)
                for index in range(length)
            ]
        return relative_strength(
            sector_return,
            market_return,
            local_history=sector_history,
            reference_history=list(market_history),
            reference="DOMESTIC_MARKET",
        )

    @staticmethod
    def _foreign_flow(
        members: Sequence[SectorMemberObservation], components: dict[str, float]
    ) -> float | None:
        flows = [
            value
            for member in members
            if (value := _finite(member.foreign_flow)) is not None
        ]
        if not flows:
            return None
        total_value = sum(
            value
            for member in members
            if (value := _finite(member.trading_value)) is not None
        )
        net = sum(flows)
        components["sector_net_foreign_flow"] = round(net, 4)
        if total_value > 0.0:
            return round(math.tanh(net / total_value / 0.05), 6)
        return 1.0 if net > 0 else (-1.0 if net < 0 else 0.0)

    @staticmethod
    def _leadership(
        returns: Sequence[float], thin: bool
    ) -> tuple[float | None, float | None]:
        """Strength of the best member, and how concentrated the advance is."""
        if not returns:
            return None, None
        leader_strength = round(max(returns), 8)
        if thin:
            return leader_strength, None
        positive = sorted((value for value in returns if value > 0.0), reverse=True)
        if not positive:
            return leader_strength, 0.0
        total = sum(positive)
        top = sum(positive[: max(1, len(returns) // 4)])
        return leader_strength, (round(top / total, 6) if total > 0.0 else None)

    @staticmethod
    def _global_alignment(
        sector_return: float | None,
        global_context: GlobalContext | None,
        global_group: str | None,
    ) -> float | None:
        """Agreement between this sector and the global group it is linked to.

        Falls back to the headline global direction when no group is named, so a sector
        with no declared cross-market counterpart still reports its relationship with the
        world rather than nothing.
        """
        if sector_return is None or global_context is None:
            return None
        reference: float | None = None
        if global_group:
            score = global_context.groups.get(str(global_group))
            reference = score.score if score else None
        if reference is None:
            reference = global_context.direction
        if reference is None:
            return None
        magnitude = max(abs(math.tanh(sector_return / 0.01)), abs(reference))
        if magnitude <= 0.0:
            return 0.0
        return round(
            max(
                -1.0,
                min(1.0, (math.tanh(sector_return / 0.01) * reference) / (magnitude**2)),
            ),
            6,
        )


def _context_id(moment: datetime, sector: str) -> str:
    from uuid import uuid4

    slug = "".join(ch for ch in sector.upper() if ch.isalnum())[:12] or "SECTOR"
    return f"sctx-{slug}-{moment.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:6]}"
