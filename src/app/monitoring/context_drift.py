"""Has the market moved away from the distribution the models were fitted on?

A strategy can be unchanged and still stop working because the tape it was measured on no
longer exists. This monitor watches the CONTEXT distribution rather than any outcome, so it
can flag that before the P&L does.

Method: population stability index (PSI) over fixed bins.

    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

PSI is used rather than a KS test because the bins can be fixed once and shared between the
reference and the live window, which makes the number comparable across runs and cheap to
compute incrementally. The conventional reading — <0.1 stable, 0.1-0.25 moderate, >0.25
significant — is applied as configured thresholds, not hardcoded.

Bins are fixed, not quantile-derived from the live data. Quantile bins would move with the
distribution they are meant to be measuring, which is how a drift monitor reports stability
through a regime change.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.context.market_context import MarketContext

__all__ = [
    "ContextDriftConfig",
    "ContextDriftMonitor",
    "ContextDriftReport",
    "FeatureDrift",
]

#: Bin edges per monitored context field, in that field's own units. Chosen to bracket the
#: values the code already treats as meaningful thresholds so a shift across one of these
#: edges is a shift the strategies will actually feel.
_BIN_EDGES: Mapping[str, tuple[float, ...]] = {
    # bps of price; +-10bps is about one typical KRX top-of-book spread.
    "trend_strength": (-40.0, -10.0, 10.0, 40.0),
    # per-observation realised volatility as a fraction.
    "realized_volatility": (0.001, 0.0025, 0.005),
    # 0.35 / 0.65 are the liquidity floors declared by strategies in the catalogue.
    "liquidity_score": (0.35, 0.65),
    # 25 / 40bps are the ``max_spread_bps`` values declared in the catalogue.
    "spread_bps": (10.0, 25.0, 40.0),
    "orderflow_imbalance": (-0.15, 0.0, 0.15),
    "vwap_distance_bps": (-50.0, -15.0, 15.0, 50.0),
    "short_term_price_impact": (1.0, 2.0, 3.0),
    "change_point_probability": (0.2, 0.4, 0.6),
}

#: Label fields whose *frequency* is monitored rather than a numeric distribution.
_LABEL_FIELDS: tuple[str, ...] = ("market_regime", "session_phase")


@dataclass(frozen=True)
class ContextDriftConfig:
    minimum_samples: int = 200
    moderate_psi: float = 0.10
    significant_psi: float = 0.25
    #: Laplace smoothing so an empty bin does not make PSI infinite. Small relative to any
    #: realistic bin share, so it shifts the number without hiding a real shift.
    epsilon: float = 1e-4


@dataclass(frozen=True)
class FeatureDrift:
    field_name: str
    psi: float | None
    reference_shares: Mapping[str, float]
    live_shares: Mapping[str, float]
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "psi": round(self.psi, 4) if self.psi is not None else None,
            "verdict": self.verdict,
            "reference_shares": {k: round(v, 4) for k, v in self.reference_shares.items()},
            "live_shares": {k: round(v, 4) for k, v in self.live_shares.items()},
        }


@dataclass(frozen=True)
class ContextDriftReport:
    reference_count: int
    live_count: int
    features: tuple[FeatureDrift, ...]
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def worst(self) -> FeatureDrift | None:
        scored = [item for item in self.features if item.psi is not None]
        return max(scored, key=lambda item: item.psi) if scored else None

    @property
    def significant_fields(self) -> tuple[str, ...]:
        return tuple(
            item.field_name for item in self.features if item.verdict == "SIGNIFICANT"
        )

    def as_dict(self) -> dict[str, Any]:
        worst = self.worst
        return {
            "reference_count": self.reference_count,
            "live_count": self.live_count,
            "worst_field": worst.field_name if worst else None,
            "worst_psi": round(worst.psi, 4) if worst and worst.psi is not None else None,
            "significant_fields": list(self.significant_fields),
            "features": [item.as_dict() for item in self.features],
            "reason_codes": list(self.reason_codes),
        }


class ContextDriftMonitor:
    """Accumulates binned context histograms and compares live against reference."""

    def __init__(self, *, config: ContextDriftConfig | None = None) -> None:
        self._config = config or ContextDriftConfig()
        self._reference: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._live: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._reference_count = 0
        self._live_count = 0

    # -- ingestion ---------------------------------------------------------- #
    def add_reference(self, context: MarketContext) -> None:
        self._add(context, self._reference)
        self._reference_count += 1

    def add_live(self, context: MarketContext) -> None:
        self._add(context, self._live)
        self._live_count += 1

    def promote_live_to_reference(self) -> None:
        """Adopt the live window as the new reference.

        Explicit rather than automatic: a rolling reference cannot detect slow drift,
        because the baseline follows the data. Promotion is what a human does after
        deciding the new distribution is the one to measure against.
        """
        self._reference = {
            field_name: dict(bins) for field_name, bins in self._live.items()
        }
        self._reference_count = self._live_count
        self._live = defaultdict(lambda: defaultdict(int))
        self._live_count = 0

    def _add(self, context: MarketContext, target: dict[str, dict[str, int]]) -> None:
        flat = context.flat()
        for field_name, edges in _BIN_EDGES.items():
            target[field_name][_bin_label(flat.get(field_name), edges)] += 1
        for field_name in _LABEL_FIELDS:
            value = flat.get(field_name)
            label = str(getattr(value, "value", value) or "UNKNOWN").strip().upper()
            target[field_name][label or "UNKNOWN"] += 1

    # -- reporting ---------------------------------------------------------- #
    def report(self) -> ContextDriftReport:
        config = self._config
        reasons: list[str] = []
        if self._reference_count < config.minimum_samples:
            reasons.append(f"CONTEXT_DRIFT_REFERENCE_BELOW_{config.minimum_samples}")
        if self._live_count < config.minimum_samples:
            reasons.append(f"CONTEXT_DRIFT_LIVE_BELOW_{config.minimum_samples}")

        features: list[FeatureDrift] = []
        for field_name in (*_BIN_EDGES, *_LABEL_FIELDS):
            reference = dict(self._reference.get(field_name, {}))
            live = dict(self._live.get(field_name, {}))
            reference_shares = _shares(reference)
            live_shares = _shares(live)
            psi = (
                _psi(reference_shares, live_shares, config.epsilon)
                if reference and live
                else None
            )
            features.append(
                FeatureDrift(
                    field_name=field_name,
                    psi=psi,
                    reference_shares=reference_shares,
                    live_shares=live_shares,
                    verdict=_verdict(psi, config),
                )
            )
        return ContextDriftReport(
            reference_count=self._reference_count,
            live_count=self._live_count,
            features=tuple(features),
            reason_codes=tuple(reasons),
        )

    def summary(self) -> dict[str, Any]:
        report = self.report()
        return {
            "reference_count": report.reference_count,
            "live_count": report.live_count,
            "significant_fields": list(report.significant_fields),
            "worst_psi": (
                round(report.worst.psi, 4)
                if report.worst and report.worst.psi is not None
                else None
            ),
            "reason_codes": list(report.reason_codes),
        }


def _bin_label(value: Any, edges: Sequence[float]) -> str:
    if value is None or isinstance(value, bool):
        # Missing is its own bin. Dropping it would hide the most important drift of all:
        # a feature that stopped being produced.
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(number):
        return "missing"
    for index, edge in enumerate(edges):
        if number < edge:
            return f"b{index}"
    return f"b{len(edges)}"


def _shares(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _psi(
    reference: Mapping[str, float], live: Mapping[str, float], epsilon: float
) -> float:
    keys = set(reference) | set(live)
    total = 0.0
    for key in keys:
        expected = max(epsilon, reference.get(key, 0.0))
        actual = max(epsilon, live.get(key, 0.0))
        total += (actual - expected) * math.log(actual / expected)
    return total


def _verdict(psi: float | None, config: ContextDriftConfig) -> str:
    if psi is None:
        return "UNKNOWN"
    if psi >= config.significant_psi:
        return "SIGNIFICANT"
    if psi >= config.moderate_psi:
        return "MODERATE"
    return "STABLE"
