"""Keeps LIVE, LIVE_PROBE, SHADOW and BACKTEST evidence apart, and weights them.

Why this is a module and not a constant
--------------------------------------
A shadow outcome and a broker fill are not the same claim. The shadow one assumes a fill
at a modelled price; the live one *is* the price. Averaging them produces a number that
describes neither, and — worse — lets a strategy be promoted on trades it never took. The
existing performance store already carries ``evaluation_source`` per row; this module is
where the *policy* for using those sources lives, so the weighting is one auditable
decision rather than a scatter of comparisons.

The weights are initial design values, NOT measurements. They are read from
``config/strategy_validation.yaml`` when present so recalibration does not need a code
change, and the resolver reports which source each aggregate came from so a mostly-shadow
result cannot be mistaken for a mostly-live one.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.evaluation.shadow_position import (
    EVIDENCE_BACKTEST,
    EVIDENCE_LIVE,
    EVIDENCE_LIVE_PROBE,
    EVIDENCE_SHADOW,
)

__all__ = [
    "EvidenceWeights",
    "ResolvedOutcome",
    "OutcomeResolver",
    "SourceMix",
]

#: Sources that may support a promotion decision at all. ``BACKTEST`` is excluded on
#: purpose: it is a prior, not evidence about the live tape.
PROMOTABLE_SOURCES: frozenset[str] = frozenset({EVIDENCE_LIVE, EVIDENCE_LIVE_PROBE})


@dataclass(frozen=True)
class EvidenceWeights:
    """How much each evidence source counts. Config, never hardcoded at a call site."""

    live: float = 1.0
    live_probe: float = 0.7
    shadow: float = 0.3
    #: ``0.0`` means "prior only": a backtest contributes to no aggregate here. Kept as a
    #: number rather than an exclusion so a caller that genuinely wants a prior-weighted
    #: view can set it, and so the choice is visible in the config.
    backtest: float = 0.0

    def weight_for(self, source: str) -> float:
        return {
            EVIDENCE_LIVE: self.live,
            EVIDENCE_LIVE_PROBE: self.live_probe,
            EVIDENCE_SHADOW: self.shadow,
            EVIDENCE_BACKTEST: self.backtest,
        }.get(str(source or "").strip().upper(), 0.0)

    @classmethod
    def from_config(
        cls, path: str | Path = "config/strategy_validation.yaml"
    ) -> "EvidenceWeights":
        values = _read_yaml(path).get("evidence_weights") or {}
        if not isinstance(values, Mapping):
            return cls()

        def read(name: str, default: float) -> float:
            try:
                number = float(values.get(name, default))
            except (TypeError, ValueError):
                return default
            return number if math.isfinite(number) and number >= 0.0 else default

        return cls(
            live=read("live", 1.0),
            live_probe=read("live_probe", 0.7),
            shadow=read("shadow", 0.3),
            backtest=read("backtest", 0.0),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "live": self.live,
            "live_probe": self.live_probe,
            "shadow": self.shadow,
            "backtest": self.backtest,
        }


@dataclass(frozen=True)
class SourceMix:
    """Where an aggregate's evidence came from, so it cannot be over-read."""

    counts: Mapping[str, int]
    weight_total: float

    @property
    def live_share(self) -> float:
        total = sum(self.counts.values())
        if total <= 0:
            return 0.0
        live = sum(
            count
            for source, count in self.counts.items()
            if str(source).upper() in PROMOTABLE_SOURCES
        )
        return live / total

    @property
    def promotable(self) -> bool:
        """Does any real-money evidence exist at all?"""
        return any(
            str(source).upper() in PROMOTABLE_SOURCES and count > 0
            for source, count in self.counts.items()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "weight_total": round(self.weight_total, 3),
            "live_share": round(self.live_share, 4),
            "promotable": self.promotable,
        }


@dataclass(frozen=True)
class ResolvedOutcome:
    """One outcome normalised into the shape every consumer here expects."""

    strategy_id: str
    symbol: str
    market: str
    context_id: str
    net_return_bps: float
    gross_return_bps: float | None
    evidence_source: str
    regime: str = "UNKNOWN"
    holding_seconds: float | None = None
    max_adverse_excursion_bps: float | None = None
    weight: float = 0.0

    @property
    def is_live(self) -> bool:
        return self.evidence_source.upper() in PROMOTABLE_SOURCES

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "market": self.market,
            "context_id": self.context_id,
            "net_return_bps": round(self.net_return_bps, 3),
            "gross_return_bps": (
                round(self.gross_return_bps, 3) if self.gross_return_bps is not None else None
            ),
            "evidence_source": self.evidence_source,
            "regime": self.regime,
            "holding_seconds": self.holding_seconds,
            "max_adverse_excursion_bps": self.max_adverse_excursion_bps,
            "weight": round(self.weight, 4),
        }


@dataclass(frozen=True)
class WeightedAggregate:
    strategy_id: str
    sample_count: int
    weighted_net_bps: float | None
    live_only_net_bps: float | None
    mix: SourceMix

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "sample_count": self.sample_count,
            "weighted_net_bps": (
                round(self.weighted_net_bps, 3) if self.weighted_net_bps is not None else None
            ),
            "live_only_net_bps": (
                round(self.live_only_net_bps, 3)
                if self.live_only_net_bps is not None
                else None
            ),
            "mix": self.mix.as_dict(),
        }


class OutcomeResolver:
    """Normalises heterogeneous outcome records and aggregates them by source weight."""

    def __init__(self, *, weights: EvidenceWeights | None = None) -> None:
        self._weights = weights or EvidenceWeights.from_config()

    @property
    def weights(self) -> EvidenceWeights:
        return self._weights

    def resolve(self, outcome: Any) -> ResolvedOutcome | None:
        """Accepts a ``ShadowOutcome``, a ``StrategyOutcome`` or a plain mapping.

        Returns ``None`` for a record with no usable net return or no strategy id — a row
        that cannot be attributed is not evidence.
        """
        read = _reader(outcome)
        strategy_id = str(read("strategy_id") or "").strip().lower()
        if not strategy_id:
            return None
        net = _finite(read("net_return_bps"))
        if net is None:
            net = _finite(read("realized_net_bps"))
        if net is None:
            return None
        source = str(
            read("evidence_source") or read("source") or EVIDENCE_SHADOW
        ).strip().upper()
        quotes = read("quotes_observed")
        if quotes is not None and int(quotes or 0) <= 0 and source == EVIDENCE_SHADOW:
            # A shadow position that never received a quote reports a zero return as a
            # marker. Counting it would add fabricated break-even trades.
            return None
        return ResolvedOutcome(
            strategy_id=strategy_id,
            symbol=str(read("symbol") or "").strip().upper(),
            market=str(read("market") or "").strip().upper() or "UNKNOWN",
            context_id=str(read("context_id") or ""),
            net_return_bps=net,
            gross_return_bps=_finite(read("gross_return_bps"))
            if read("gross_return_bps") is not None
            else _finite(read("realized_gross_bps")),
            evidence_source=source,
            regime=str(read("regime") or "UNKNOWN").strip().upper(),
            holding_seconds=_finite(read("holding_seconds")),
            max_adverse_excursion_bps=_finite(read("max_adverse_excursion_bps")),
            weight=self._weights.weight_for(source),
        )

    def resolve_all(self, outcomes: Iterable[Any]) -> tuple[ResolvedOutcome, ...]:
        resolved = (self.resolve(item) for item in outcomes or ())
        return tuple(item for item in resolved if item is not None)

    def aggregate_by_strategy(
        self, outcomes: Iterable[Any]
    ) -> dict[str, WeightedAggregate]:
        """Weighted and live-only means per strategy, side by side.

        Both are reported because they answer different questions: the weighted mean is
        the best available estimate, the live-only mean is the only one a promotion may
        rest on. A strategy whose weighted mean is positive and whose live-only mean is
        absent has not earned anything.
        """
        by_strategy: dict[str, list[ResolvedOutcome]] = defaultdict(list)
        for item in self.resolve_all(outcomes):
            by_strategy[item.strategy_id].append(item)

        aggregates: dict[str, WeightedAggregate] = {}
        for strategy_id, rows in by_strategy.items():
            counts: dict[str, int] = defaultdict(int)
            for row in rows:
                counts[row.evidence_source] += 1
            weight_total = sum(row.weight for row in rows)
            weighted = (
                sum(row.net_return_bps * row.weight for row in rows) / weight_total
                if weight_total > 0
                else None
            )
            live_rows = [row for row in rows if row.is_live]
            live_mean = (
                sum(row.net_return_bps for row in live_rows) / len(live_rows)
                if live_rows
                else None
            )
            aggregates[strategy_id] = WeightedAggregate(
                strategy_id=strategy_id,
                sample_count=len(rows),
                weighted_net_bps=weighted,
                live_only_net_bps=live_mean,
                mix=SourceMix(counts=dict(counts), weight_total=weight_total),
            )
        return aggregates

    @staticmethod
    def cluster_by_context(
        outcomes: Sequence[ResolvedOutcome],
    ) -> dict[str, tuple[ResolvedOutcome, ...]]:
        """Group outcomes by ``context_id``.

        Shadow outcomes from one context are the same price path cut by different
        barriers, so they are not independent samples. Any statistic over them has to
        cluster here first or its sample count is inflated by roughly the group size.
        """
        grouped: dict[str, list[ResolvedOutcome]] = defaultdict(list)
        for item in outcomes:
            grouped[item.context_id].append(item)
        return {key: tuple(value) for key, value in grouped.items()}


def _reader(outcome: Any):
    if isinstance(outcome, Mapping):
        return lambda name: outcome.get(name)
    return lambda name: getattr(outcome, name, None)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - an unreadable config uses defaults.
        return {}
    return loaded if isinstance(loaded, dict) else {}
