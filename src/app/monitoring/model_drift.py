"""Per-(strategy x market x regime) model trust, not one global number.

The failure this prevents
------------------------
A single global "GNN trust passed" flag authorises every strategy at once. That is how a
model calibrated on the two strategies with data ends up speaking for the seventeen without
any. The existing ``app.routing.gnn_realtime_trust`` already computes forward-only Brier,
net sign accuracy and net MAE, and already distinguishes model-wide trust from
strategy-market trust; this module extends the same idea to the cell the V2 selector
actually acts on — ``(strategy, market, regime_cluster)`` — and adds calibration of the
*utility* prediction, which the classifier-era metrics did not cover.

Shrinkage, not a cliff
----------------------
A cell with three samples gets a trust score shrunk toward the model-wide score rather than
a confident verdict of its own, and cells below ``minimum_samples`` are reported as
``INSUFFICIENT`` so a caller can fail closed. The alternative — a hard sample threshold —
makes every newly added strategy permanently untrusted, which is the self-locking trust
gate this codebase has already hit once.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

__all__ = [
    "ModelDriftMonitor",
    "ModelTrustCell",
    "ModelTrustConfig",
    "TrustVerdict",
]


class TrustVerdict(StrEnum):
    TRUSTED = "TRUSTED"
    #: Enough evidence, and it says the estimate is not usable for this cell.
    UNTRUSTED = "UNTRUSTED"
    #: Not enough evidence to say. Callers must fail closed on this, never treat it as
    #: trusted-by-default.
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class ModelTrustConfig:
    minimum_samples: int = 12
    #: Brier above this is worse than a constant predictor on a balanced outcome, so the
    #: probability head carries no information.
    max_brier: float = 0.25
    #: Net sign accuracy at or below this is a coin flip on direction.
    min_sign_accuracy: float = 0.55
    #: Mean absolute error ceiling on the net-return forecast, in bps.
    max_net_mae_bps: float = 80.0
    #: Predicted-band coverage that counts as calibrated. A one-sigma band should contain
    #: ~68% of outcomes; the window is generous because the sample is small.
    calibration_low: float = 0.45
    calibration_high: float = 0.90
    #: Weight given to a cell's own evidence at exactly ``minimum_samples``. Below that the
    #: score is blended toward the model-wide score in proportion to the sample count.
    shrinkage_pivot: int = 12


@dataclass
class _Cell:
    samples: list[tuple[float, float, float | None, float | None]] = field(
        default_factory=list
    )
    #: ``(predicted_net_bps, actual_net_bps, predicted_probability, predicted_uncertainty)``

    def add(
        self,
        predicted_net: float,
        actual_net: float,
        probability: float | None,
        uncertainty: float | None,
    ) -> None:
        self.samples.append((predicted_net, actual_net, probability, uncertainty))


@dataclass(frozen=True)
class ModelTrustCell:
    strategy_id: str
    market: str
    regime_cluster: str
    sample_count: int
    mean_actual_net_bps: float | None
    mean_predicted_net_bps: float | None
    net_mae_bps: float | None
    brier_score: float | None
    sign_accuracy: float | None
    uncertainty_calibration: float | None
    verdict: TrustVerdict
    #: 0.0-1.0 composite, shrunk toward the model-wide score when the cell is thin.
    trust_score: float
    reason_codes: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.strategy_id}|{self.market}|{self.regime_cluster}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "market": self.market,
            "regime_cluster": self.regime_cluster,
            "sample_count": self.sample_count,
            "mean_actual_net_bps": _round(self.mean_actual_net_bps),
            "mean_predicted_net_bps": _round(self.mean_predicted_net_bps),
            "net_mae_bps": _round(self.net_mae_bps),
            "brier_score": _round(self.brier_score, 4),
            "sign_accuracy": _round(self.sign_accuracy, 4),
            "uncertainty_calibration": _round(self.uncertainty_calibration, 4),
            "verdict": str(self.verdict),
            "trust_score": round(self.trust_score, 4),
            "reason_codes": list(self.reason_codes),
        }


class ModelDriftMonitor:
    """Accumulates prediction/outcome pairs and scores trust per cell."""

    def __init__(self, *, config: ModelTrustConfig | None = None) -> None:
        self._config = config or ModelTrustConfig()
        self._cells: dict[tuple[str, str, str], _Cell] = defaultdict(_Cell)

    def record(
        self,
        *,
        strategy_id: str,
        market: str,
        regime_cluster: str,
        predicted_net_bps: float,
        actual_net_bps: float,
        predicted_probability: float | None = None,
        predicted_uncertainty_bps: float | None = None,
    ) -> None:
        key = (
            str(strategy_id or "").strip().lower(),
            str(market or "UNKNOWN").strip().upper(),
            str(regime_cluster or "UNKNOWN").strip().upper(),
        )
        predicted = _finite(predicted_net_bps)
        actual = _finite(actual_net_bps)
        if not key[0] or predicted is None or actual is None:
            return
        self._cells[key].add(
            predicted,
            actual,
            _finite(predicted_probability),
            _finite(predicted_uncertainty_bps),
        )

    def record_pair(self, prediction: Any, outcome: Any) -> None:
        """Convenience over ``record`` for a prediction object and an outcome object."""
        read = (
            (lambda name: outcome.get(name))
            if isinstance(outcome, Mapping)
            else (lambda name: getattr(outcome, name, None))
        )
        actual = _finite(read("net_return_bps")) or _finite(read("realized_net_bps"))
        if actual is None:
            return
        self.record(
            strategy_id=str(getattr(prediction, "strategy_id", "") or read("strategy_id") or ""),
            market=str(read("market") or "UNKNOWN"),
            regime_cluster=str(read("regime") or "UNKNOWN"),
            predicted_net_bps=_finite(getattr(prediction, "expected_net_return_bps", None)) or 0.0,
            actual_net_bps=actual,
            predicted_probability=_finite(getattr(prediction, "probability_profit", None)),
            predicted_uncertainty_bps=_finite(getattr(prediction, "uncertainty_bps", None)),
        )

    # -- scoring ------------------------------------------------------------ #
    def model_wide_score(self) -> float:
        """One composite over every cell, used only as the shrinkage target."""
        all_samples = [sample for cell in self._cells.values() for sample in cell.samples]
        if not all_samples:
            return 0.0
        return _composite(_metrics(all_samples), self._config)

    def cells(self) -> tuple[ModelTrustCell, ...]:
        model_wide = self.model_wide_score()
        return tuple(
            self._score_cell(key, cell, model_wide)
            for key, cell in sorted(self._cells.items())
        )

    def trust_for(
        self, strategy_id: str, *, market: str, regime_cluster: str = "UNKNOWN"
    ) -> ModelTrustCell:
        key = (
            str(strategy_id or "").strip().lower(),
            str(market or "UNKNOWN").strip().upper(),
            str(regime_cluster or "UNKNOWN").strip().upper(),
        )
        cell = self._cells.get(key)
        if cell is None:
            # Widen to the market before giving up: a regime with no history is a thinner
            # claim than a strategy with none, and the wider cell is still about the same
            # strategy in the same market.
            widened = self._widen(key)
            if widened is None:
                return ModelTrustCell(
                    strategy_id=key[0],
                    market=key[1],
                    regime_cluster=key[2],
                    sample_count=0,
                    mean_actual_net_bps=None,
                    mean_predicted_net_bps=None,
                    net_mae_bps=None,
                    brier_score=None,
                    sign_accuracy=None,
                    uncertainty_calibration=None,
                    verdict=TrustVerdict.INSUFFICIENT,
                    trust_score=0.0,
                    reason_codes=("MODEL_TRUST_NO_SAMPLES",),
                )
            cell = widened
        return self._score_cell(key, cell, self.model_wide_score())

    def _widen(self, key: tuple[str, str, str]) -> _Cell | None:
        strategy_id, market, _ = key
        merged = _Cell()
        for (cell_strategy, cell_market, _regime), cell in self._cells.items():
            if cell_strategy == strategy_id and cell_market == market:
                merged.samples.extend(cell.samples)
        return merged if merged.samples else None

    def _score_cell(
        self, key: tuple[str, str, str], cell: _Cell, model_wide: float
    ) -> ModelTrustCell:
        config = self._config
        metrics = _metrics(cell.samples)
        own = _composite(metrics, config)
        count = len(cell.samples)
        pivot = max(1, int(config.shrinkage_pivot))
        weight = min(1.0, count / pivot)
        # Shrunk toward the model-wide score, not toward 1.0 or 0.0: the honest prior for a
        # thin cell is "however well this model does in general".
        score = weight * own + (1.0 - weight) * model_wide

        reasons: list[str] = []
        if count < config.minimum_samples:
            reasons.append(f"MODEL_TRUST_SAMPLE_BELOW_{config.minimum_samples}")
            verdict = TrustVerdict.INSUFFICIENT
        else:
            failures = _failures(metrics, config)
            reasons.extend(failures)
            verdict = TrustVerdict.UNTRUSTED if failures else TrustVerdict.TRUSTED

        return ModelTrustCell(
            strategy_id=key[0],
            market=key[1],
            regime_cluster=key[2],
            sample_count=count,
            mean_actual_net_bps=metrics.get("mean_actual"),
            mean_predicted_net_bps=metrics.get("mean_predicted"),
            net_mae_bps=metrics.get("mae"),
            brier_score=metrics.get("brier"),
            sign_accuracy=metrics.get("sign_accuracy"),
            uncertainty_calibration=metrics.get("calibration"),
            verdict=verdict,
            trust_score=max(0.0, min(1.0, score)),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    def summary(self) -> dict[str, Any]:
        cells = self.cells()
        return {
            "model_wide_trust_score": round(self.model_wide_score(), 4),
            "cell_count": len(cells),
            "trusted": [item.key for item in cells if item.verdict is TrustVerdict.TRUSTED],
            "untrusted": [
                item.key for item in cells if item.verdict is TrustVerdict.UNTRUSTED
            ],
            "insufficient": [
                item.key for item in cells if item.verdict is TrustVerdict.INSUFFICIENT
            ],
        }


def _metrics(
    samples: Sequence[tuple[float, float, float | None, float | None]]
) -> dict[str, float | None]:
    if not samples:
        return {}
    predicted = [item[0] for item in samples]
    actual = [item[1] for item in samples]
    probabilities = [(item[2], item[1]) for item in samples if item[2] is not None]
    banded = [item for item in samples if item[3] is not None]
    return {
        "mean_predicted": sum(predicted) / len(predicted),
        "mean_actual": sum(actual) / len(actual),
        "mae": sum(abs(p - a) for p, a in zip(predicted, actual, strict=True)) / len(samples),
        "brier": (
            sum((p - (1.0 if a > 0 else 0.0)) ** 2 for p, a in probabilities)
            / len(probabilities)
            if probabilities
            else None
        ),
        "sign_accuracy": (
            sum(1 for p, a in zip(predicted, actual, strict=True) if (p > 0) == (a > 0))
            / len(samples)
        ),
        "calibration": (
            sum(1 for item in banded if abs(item[1] - item[0]) <= (item[3] or 0.0))
            / len(banded)
            if banded
            else None
        ),
    }


def _failures(
    metrics: Mapping[str, float | None], config: ModelTrustConfig
) -> list[str]:
    failures: list[str] = []
    brier = metrics.get("brier")
    if brier is not None and brier > config.max_brier:
        failures.append("MODEL_TRUST_BRIER_ABOVE_CEILING")
    accuracy = metrics.get("sign_accuracy")
    if accuracy is not None and accuracy < config.min_sign_accuracy:
        failures.append("MODEL_TRUST_SIGN_ACCURACY_BELOW_FLOOR")
    mae = metrics.get("mae")
    if mae is not None and mae > config.max_net_mae_bps:
        failures.append("MODEL_TRUST_NET_MAE_ABOVE_CEILING")
    calibration = metrics.get("calibration")
    if calibration is not None and not (
        config.calibration_low <= calibration <= config.calibration_high
    ):
        failures.append("MODEL_TRUST_UNCERTAINTY_MISCALIBRATED")
    return failures


def _composite(
    metrics: Mapping[str, float | None], config: ModelTrustConfig
) -> float:
    """0.0-1.0 from whichever metrics exist. Missing metrics are skipped, not zeroed."""
    parts: list[float] = []
    brier = metrics.get("brier")
    if brier is not None:
        parts.append(max(0.0, 1.0 - brier / max(1e-9, config.max_brier)))
    accuracy = metrics.get("sign_accuracy")
    if accuracy is not None:
        # Rescaled so the floor maps to 0 and perfect maps to 1.
        span = max(1e-9, 1.0 - config.min_sign_accuracy)
        parts.append(max(0.0, min(1.0, (accuracy - config.min_sign_accuracy) / span)))
    mae = metrics.get("mae")
    if mae is not None:
        parts.append(max(0.0, 1.0 - mae / max(1e-9, config.max_net_mae_bps)))
    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None
