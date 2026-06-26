from __future__ import annotations

import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from app.schemas.domain import IndicatorSnapshot, MarketSnapshot


@dataclass(frozen=True)
class OntologyNpuStatus:
    backend: str
    uses_npu: bool
    batch_size: int
    feature_dim: int
    score_dim: int
    last_latency_ms: float | None = None
    last_items: int = 0
    last_batches: int = 0
    last_items_per_second: float | None = None
    fallback_reason: str | None = None


class OntologyNpuClassifier:
    """OpenVINO NPU batch scorer for indicator-to-ontology relation features."""

    def __init__(self, batch_size: int = 2048, feature_dim: int = 8, score_dim: int = 6) -> None:
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.score_dim = score_dim
        self._compiled: Any | None = None
        self._backend = "uninitialized"
        self._fallback_reason: str | None = None
        self._last_latency_ms: float | None = None
        self._last_items = 0
        self._last_batches = 0
        self._last_items_per_second: float | None = None
        self._batch_buffer: np.ndarray | None = None
        self._lock = Lock()

    def classify(
        self,
        markets: tuple[MarketSnapshot, ...],
        indicators: dict[str, IndicatorSnapshot],
    ) -> dict[str, tuple[float, ...]]:
        tickers: list[str] = []
        feature_rows: list[tuple[float, ...]] = []
        for market in markets:
            indicator = indicators.get(market.ticker)
            if indicator is None:
                continue
            tickers.append(market.ticker)
            feature_rows.append(_features(market, indicator))
        if not tickers:
            return {}

        compiled = self._compiled_model()
        started = time.perf_counter()
        scores: dict[str, tuple[float, ...]] = {}
        with self._lock:
            batch = self._input_buffer()
            batches = 0
            for offset in range(0, len(tickers), self.batch_size):
                chunk_size = min(self.batch_size, len(tickers) - offset)
                batch[:chunk_size, :] = feature_rows[offset : offset + chunk_size]
                if chunk_size < self.batch_size:
                    batch[chunk_size:, :] = 0.0
                output = compiled([batch])[0]
                batches += 1
                for index, ticker in enumerate(tickers[offset : offset + chunk_size]):
                    scores[ticker] = tuple(float(value) for value in output[index, : self.score_dim])
        elapsed_seconds = time.perf_counter() - started
        self._last_latency_ms = round(elapsed_seconds * 1000.0, 3)
        self._last_items = len(tickers)
        self._last_batches = batches
        self._last_items_per_second = round(len(tickers) / elapsed_seconds, 2) if elapsed_seconds > 0 else None
        return scores

    def status(self) -> OntologyNpuStatus:
        self._compiled_model()
        return OntologyNpuStatus(
            backend=self._backend,
            uses_npu=self._backend.upper() == "NPU",
            batch_size=self.batch_size,
            feature_dim=self.feature_dim,
            score_dim=self.score_dim,
            last_latency_ms=self._last_latency_ms,
            last_items=self._last_items,
            last_batches=self._last_batches,
            last_items_per_second=self._last_items_per_second,
            fallback_reason=self._fallback_reason,
        )

    def _input_buffer(self) -> np.ndarray:
        if self._batch_buffer is None:
            self._batch_buffer = np.zeros((self.batch_size, self.feature_dim), dtype=np.float32)
        return self._batch_buffer

    def _compiled_model(self) -> Any:
        if self._compiled is not None:
            return self._compiled
        with self._lock:
            if self._compiled is not None:
                return self._compiled
            weights = _linear_weights()
            try:
                import openvino as ov
            except ModuleNotFoundError as exc:
                self._compiled = _NumpyLinearModel(weights)
                self._backend = "CPU_NUMPY"
                self._fallback_reason = f"OpenVINO unavailable: {exc}"
                return self._compiled

            ops = ov.opset8
            x = ops.parameter([self.batch_size, self.feature_dim], ov.Type.f32, name="ontology_features")
            model = ov.Model([ops.matmul(x, ops.constant(weights), False, False)], [x], "ontology_npu_classifier")
            core = ov.Core()
            requested = os.getenv("ONTOLOGY_CLASSIFIER_DEVICE", os.getenv("OPENVINO_DEVICE", "NPU"))
            try:
                self._compiled = core.compile_model(model, requested)
                self._backend = requested
                self._fallback_reason = None
            except Exception as exc:
                self._compiled = core.compile_model(model, "CPU")
                self._backend = "CPU"
                self._fallback_reason = f"{requested} compile failed: {exc}"
            return self._compiled


class _NumpyLinearModel:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0] @ self.weights]


def _linear_weights() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, -0.25, -0.25, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.2, 0.0, 0.0, 0.0, 0.6],
        ],
        dtype=np.float32,
    )


def _features(market: MarketSnapshot, indicator: IndicatorSnapshot) -> tuple[float, ...]:
    per_score = (indicator.per or 0.0) / 100.0
    rsi_score = (indicator.rsi_14d or 50.0) / 100.0
    volume_score = indicator.volume_ratio or 1.0
    return (
        float(indicator.operating_margin or 0.0),
        float(indicator.operating_income_growth or 0.0),
        float(per_score),
        float(indicator.macro_risk_score),
        float(market.volatility_20d),
        float(indicator.revenue_growth or 0.0),
        float(rsi_score),
        float(volume_score),
    )


_CLASSIFIER: OntologyNpuClassifier | None = None
_LOCK = Lock()


def get_ontology_npu_classifier() -> OntologyNpuClassifier:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        with _LOCK:
            if _CLASSIFIER is None:
                batch_size = max(128, int(os.getenv("ONTOLOGY_NPU_BATCH_SIZE", "4096")))
                _CLASSIFIER = OntologyNpuClassifier(batch_size=batch_size)
    return _CLASSIFIER
