"""Serving wrapper for the temporal hetero GNN, with an explicit health state.

Three states, three different permissions
-----------------------------------------

============  =========================================================
``HEALTHY``   Checkpoint loaded, inference succeeding, predictions fresh.
              Full sizing.
``DEGRADED``  The model is answering but something is wrong with it —
              stale predictions, elevated uncertainty, a recent failure
              that has not yet recurred. Rule and context evidence only;
              model terms are dropped and position size is reduced.
``OFFLINE``   No usable model. **New entries are blocked.** Existing
              positions are still managed, because refusing to exit is a
              worse failure than refusing to enter.
============  =========================================================

The asymmetry is deliberate and is the whole reason this class exists. A missing model is
a reason not to open risk; it is never a reason to be unable to close it.

Failures are counted, not swallowed
-----------------------------------
An inference exception is recorded with its reason and re-surfaced through
:meth:`GnnRuntime.health`; it never propagates into the trading loop as a crash and never
disappears into a log line. Consecutive failures past ``failure_threshold`` take the
runtime OFFLINE, and it stays there until a checkpoint reloads cleanly — there is no
automatic recovery from "it worked that time", because an intermittently-failing model is
one whose outputs cannot be trusted individually.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from app.models.graph_snapshot import GraphSnapshot
from app.models.temporal_hetero_gnn import (
    TemporalHeteroGnn,
    TemporalHeteroGnnConfig,
    TemporalHeteroGnnOutput,
)

__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "GnnHealth",
    "GnnHealthState",
    "GnnRuntime",
    "GnnPrediction",
]

DEFAULT_CHECKPOINT_PATH = Path("data/models/temporal_hetero_gnn/latest.npz")

GNN_NO_CHECKPOINT = "GNN_NO_CHECKPOINT"
GNN_CHECKPOINT_INVALID = "GNN_CHECKPOINT_INVALID"
GNN_INFERENCE_FAILED = "GNN_INFERENCE_FAILED"
GNN_PREDICTION_STALE = "GNN_PREDICTION_STALE"
GNN_HIGH_UNCERTAINTY = "GNN_HIGH_UNCERTAINTY"
GNN_SHAPE_MISMATCH = "GNN_SHAPE_MISMATCH"
GNN_UNTRAINED_WEIGHTS = "GNN_UNTRAINED_WEIGHTS"


class GnnHealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass(frozen=True)
class GnnHealth:
    state: GnnHealthState
    observed_at: datetime
    reason_codes: tuple[str, ...] = ()
    checkpoint_path: str | None = None
    checkpoint_loaded_at: datetime | None = None
    consecutive_failures: int = 0
    last_inference_at: datetime | None = None
    last_error: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def allows_new_entry(self) -> bool:
        """OFFLINE blocks new entries. DEGRADED permits them at reduced size."""
        return self.state is not GnnHealthState.OFFLINE

    @property
    def allows_model_evidence(self) -> bool:
        """Only a HEALTHY model's outputs may enter a decision as model evidence."""
        return self.state is GnnHealthState.HEALTHY

    @property
    def size_multiplier(self) -> float:
        """Sizing factor contributed by model health alone."""
        if self.state is GnnHealthState.HEALTHY:
            return 1.0
        if self.state is GnnHealthState.DEGRADED:
            return 0.5
        return 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "observed_at": self.observed_at.isoformat(),
            "reason_codes": list(self.reason_codes),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_loaded_at": (
                self.checkpoint_loaded_at.isoformat()
                if self.checkpoint_loaded_at
                else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "last_inference_at": (
                self.last_inference_at.isoformat() if self.last_inference_at else None
            ),
            "last_error": self.last_error,
            "allows_new_entry": self.allows_new_entry,
            "allows_model_evidence": self.allows_model_evidence,
            "size_multiplier": self.size_multiplier,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class GnnPrediction:
    """One inference result, tagged with the health it was produced under."""

    output: TemporalHeteroGnnOutput
    snapshot: GraphSnapshot
    health: GnnHealth
    predicted_at: datetime
    model_version: str

    def for_ticker(self, ticker: str) -> dict[str, Any] | None:
        index = self.snapshot.index_of(f"STOCK::{str(ticker).strip().upper()}")
        if index is None:
            return None
        return self.output.for_node(index)

    def for_market(self, market_group: str = "KR") -> dict[str, Any] | None:
        node = "KR_MARKET" if str(market_group).upper() == "KR" else "US_MARKET"
        index = self.snapshot.index_of(node)
        if index is None:
            return None
        return self.output.for_node(index)

    def as_trace(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "predicted_at": self.predicted_at.isoformat(),
            "health": self.health.as_dict(),
            "graph": {
                "node_count": len(self.snapshot.node_ids),
                "active_node_count": self.snapshot.active_node_count,
                "reason_codes": list(self.snapshot.reason_codes),
            },
            "attention": self.output.trace.as_dict(),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GnnRuntime:
    """Loads, serves and health-tracks the temporal hetero GNN.

    Thread-safe. The realtime loop calls :meth:`predict` and reads :meth:`health`; nothing
    else touches the model object.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        config: TemporalHeteroGnnConfig | None = None,
        failure_threshold: int = 3,
        max_prediction_age_seconds: float = 300.0,
        high_uncertainty_threshold: float = 1.5,
        require_checkpoint: bool = True,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._config = config
        self._failure_threshold = max(1, int(failure_threshold))
        self._max_prediction_age = float(max_prediction_age_seconds)
        self._high_uncertainty = float(high_uncertainty_threshold)
        #: When False an untrained (randomly initialised) model may serve, in DEGRADED.
        #: Live trading leaves this True: random weights are not evidence.
        self._require_checkpoint = bool(require_checkpoint)

        self._lock = threading.RLock()
        self._model: TemporalHeteroGnn | None = None
        self._checkpoint_loaded_at: datetime | None = None
        self._model_version = "none"
        self._consecutive_failures = 0
        self._last_inference_at: datetime | None = None
        self._last_error: str | None = None
        self._load_reasons: tuple[str, ...] = (GNN_NO_CHECKPOINT,)
        self._offline_latched = False
        self.reload()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def reload(self) -> GnnHealth:
        """(Re)load the checkpoint. The only way out of a latched OFFLINE."""
        with self._lock:
            reasons: list[str] = []
            model: TemporalHeteroGnn | None = None
            version = "none"
            loaded_at: datetime | None = None
            if self._checkpoint_path.exists():
                try:
                    model = TemporalHeteroGnn.load_checkpoint(self._checkpoint_path)
                    loaded_at = _utcnow()
                    stat = self._checkpoint_path.stat()
                    version = f"{self._checkpoint_path.stem}:{int(stat.st_mtime)}"
                except Exception as exc:  # noqa: BLE001 - reported as health, not raised.
                    reasons.append(GNN_CHECKPOINT_INVALID)
                    self._last_error = f"{type(exc).__name__}: {exc}"
            else:
                reasons.append(GNN_NO_CHECKPOINT)

            if model is None and not self._require_checkpoint and self._config is not None:
                # Explicitly opted in: an untrained model that can be exercised end to end
                # without pretending its numbers mean anything.
                model = TemporalHeteroGnn(self._config)
                version = "untrained"
                loaded_at = _utcnow()
                reasons.append(GNN_UNTRAINED_WEIGHTS)

            if model is not None and self._config is not None:
                if model.config != self._config:
                    reasons.append(GNN_SHAPE_MISMATCH)
                    model = None
                    version = "none"
                    loaded_at = None

            self._model = model
            self._model_version = version
            self._checkpoint_loaded_at = loaded_at
            self._load_reasons = tuple(dict.fromkeys(reasons))
            self._consecutive_failures = 0
            self._offline_latched = False
            return self._health_locked()

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def config(self) -> TemporalHeteroGnnConfig | None:
        with self._lock:
            return self._model.config if self._model is not None else self._config

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def predict(
        self, snapshot: GraphSnapshot, *, now: datetime | None = None
    ) -> GnnPrediction | None:
        """Run inference, or return ``None`` and record why.

        Never raises. A model failure must become a *health state* the gate can read, not
        an exception that takes down the trading loop mid-cycle — but it must also never
        become a silent absence, which is why the reason is latched into the health
        record and surfaced on the dashboard.
        """
        moment = now or _utcnow()
        with self._lock:
            model = self._model
            if model is None:
                return None
            try:
                output = model.infer(
                    snapshot.features,
                    snapshot.adjacency,
                    snapshot.prior_bias,
                    snapshot.node_type_index,
                    snapshot.node_mask,
                    node_ids=snapshot.node_ids,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced through health.
                self._consecutive_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                if self._consecutive_failures >= self._failure_threshold:
                    self._offline_latched = True
                return None
            self._consecutive_failures = 0
            self._last_inference_at = moment
            health = self._health_locked(
                now=moment, uncertainty=self._mean_uncertainty(output)
            )
            return GnnPrediction(
                output=output,
                snapshot=snapshot,
                health=health,
                predicted_at=moment,
                model_version=self._model_version,
            )

    @staticmethod
    def _mean_uncertainty(output: TemporalHeteroGnnOutput) -> float | None:
        mask = output.node_mask > 0
        if not mask.any():
            return None
        return float(output.uncertainty[mask].mean())

    # ------------------------------------------------------------------ #
    # health
    # ------------------------------------------------------------------ #
    def health(self, *, now: datetime | None = None) -> GnnHealth:
        with self._lock:
            return self._health_locked(now=now)

    def _health_locked(
        self, *, now: datetime | None = None, uncertainty: float | None = None
    ) -> GnnHealth:
        moment = now or _utcnow()
        reasons: list[str] = list(self._load_reasons)
        state = GnnHealthState.HEALTHY

        if self._model is None or self._offline_latched:
            state = GnnHealthState.OFFLINE
            if self._offline_latched:
                reasons.append(GNN_INFERENCE_FAILED)
        else:
            if self._consecutive_failures > 0:
                reasons.append(GNN_INFERENCE_FAILED)
                state = GnnHealthState.DEGRADED
            if GNN_UNTRAINED_WEIGHTS in reasons:
                state = GnnHealthState.DEGRADED
            if self._last_inference_at is not None and self._max_prediction_age > 0:
                age = (moment - self._last_inference_at).total_seconds()
                if age > self._max_prediction_age:
                    reasons.append(GNN_PREDICTION_STALE)
                    state = GnnHealthState.DEGRADED
            if uncertainty is not None and uncertainty > self._high_uncertainty:
                reasons.append(GNN_HIGH_UNCERTAINTY)
                state = GnnHealthState.DEGRADED

        return GnnHealth(
            state=state,
            observed_at=moment,
            reason_codes=tuple(dict.fromkeys(reasons)),
            checkpoint_path=str(self._checkpoint_path),
            checkpoint_loaded_at=self._checkpoint_loaded_at,
            consecutive_failures=self._consecutive_failures,
            last_inference_at=self._last_inference_at,
            last_error=self._last_error,
            detail={
                "model_version": self._model_version,
                "mean_uncertainty": uncertainty,
                "failure_threshold": self._failure_threshold,
                "max_prediction_age_seconds": self._max_prediction_age,
            },
        )

    def mark_failure(self, reason: str) -> GnnHealth:
        """Record an externally-detected failure (a caller's shape mismatch, say)."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = str(reason)
            if self._consecutive_failures >= self._failure_threshold:
                self._offline_latched = True
            return self._health_locked()

    def is_prediction_stale(self, *, now: datetime | None = None) -> bool:
        with self._lock:
            if self._last_inference_at is None:
                return True
            age = ((now or _utcnow()) - self._last_inference_at).total_seconds()
            return age > self._max_prediction_age

    def next_expected_by(self) -> datetime | None:
        with self._lock:
            if self._last_inference_at is None:
                return None
            return self._last_inference_at + timedelta(seconds=self._max_prediction_age)
