from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np

from app.schemas.domain import IndicatorSnapshot, MarketSnapshot

SCORE_SCHEMA = (
    "support_score",
    "risk_score",
    "momentum_score",
    "value_score",
    "liquidity_score",
    "confidence_score",
)
FEATURE_SCHEMA = (
    "operating_margin",
    "operating_income_growth",
    "per_score",
    "macro_risk_score",
    "volatility_20d",
    "revenue_growth",
    "rsi_score",
    "volume_ratio",
)
BATCH_BUCKETS = (512, 1024, 2048, 4096)


@dataclass(frozen=True)
class OntologyNpuStatus:
    backend: str
    uses_npu: bool
    batch_size: int
    feature_dim: int
    score_dim: int
    model_name: str = "ontology_candidate_scorer"
    requested_device: str = "NPU"
    selected_device: str = "uninitialized"
    model_kind: str = "heuristic_linear_scorer"
    last_latency_ms: float | None = None
    last_items: int = 0
    last_batches: int = 0
    last_items_per_second: float | None = None
    fallback_reason: str | None = None
    last_profile: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScoreBatch:
    tickers: tuple[str, ...]
    scores: np.ndarray
    score_schema: tuple[str, ...]
    profile: dict[str, float | int | str]

    def score_dict(self) -> dict[str, tuple[float, ...]]:
        return {
            ticker: tuple(float(value) for value in self.scores[index])
            for index, ticker in enumerate(self.tickers)
        }


class OntologyNpuLinearScorer:
    """Vectorized ontology evidence scorer with OpenVINO/NPU acceleration when available.

    The output is evidence only. It is not an order decision and must continue
    through graph reasoning, strategy policy, RiskManager, and manual approval.
    """

    def __init__(
        self,
        batch_size: int | str = "auto",
        feature_dim: int = len(FEATURE_SCHEMA),
        score_dim: int = len(SCORE_SCHEMA),
    ) -> None:
        self.batch_size = _bucket_from_value(batch_size)
        self.feature_dim = feature_dim
        self.score_dim = score_dim
        self._compiled: dict[int, Any] = {}
        self._backend = "uninitialized"
        self._fallback_reason: str | None = None
        self._last_latency_ms: float | None = None
        self._last_items = 0
        self._last_batches = 0
        self._last_items_per_second: float | None = None
        self._last_profile: dict[str, float | int | str] = {}
        self._batch_buffers: dict[int, np.ndarray] = {}
        self._lock = Lock()

    def score_candidates(
        self,
        tickers: Sequence[str],
        feature_matrix: np.ndarray | Sequence[Sequence[float]],
        *,
        top_k: int = 50,
    ) -> CandidateScoreBatch:
        started_total = time.perf_counter()
        started = time.perf_counter()
        ticker_tuple = tuple(str(ticker) for ticker in tickers)
        features = _normalize_feature_matrix(feature_matrix, self.feature_dim)
        if len(ticker_tuple) != features.shape[0]:
            raise ValueError("tickers and feature_matrix row count must match")
        input_count = len(ticker_tuple)
        if input_count == 0:
            profile = _profile(started_total, started, 0.0, 0.0, 0, self.batch_size, self._backend, top_k)
            return CandidateScoreBatch((), np.zeros((0, self.score_dim), dtype=np.float32), SCORE_SCHEMA, profile)

        preprocess_ms = (time.perf_counter() - started) * 1000.0
        bucket = _bucket_for_count(input_count, self.batch_size)
        compiled = self._compiled_model(bucket)

        started_inference = time.perf_counter()
        output_chunks: list[np.ndarray] = []
        batches = 0
        with self._lock:
            for offset in range(0, input_count, bucket):
                chunk_size = min(bucket, input_count - offset)
                batch = self._input_buffer(bucket)
                batch[:chunk_size, :] = features[offset : offset + chunk_size, :]
                if chunk_size < bucket:
                    batch[chunk_size:, :] = 0.0
                output = np.asarray(compiled([batch])[0], dtype=np.float32)[:chunk_size, : self.score_dim]
                output_chunks.append(output.copy())
                batches += 1
        inference_ms = (time.perf_counter() - started_inference) * 1000.0

        started_post = time.perf_counter()
        scores = np.vstack(output_chunks) if output_chunks else np.zeros((0, self.score_dim), dtype=np.float32)
        ranking_score = _ranking_score(scores)
        started_topk = time.perf_counter()
        count = min(max(0, int(top_k)), input_count)
        if count < input_count:
            top_indices = np.argpartition(-ranking_score, count - 1)[:count] if count else np.array([], dtype=np.int64)
            top_indices = top_indices[np.argsort(-ranking_score[top_indices])]
        else:
            top_indices = np.argsort(-ranking_score)
        topk_ms = (time.perf_counter() - started_topk) * 1000.0
        top_scores = scores[top_indices].astype(np.float32, copy=True)
        top_tickers = tuple(ticker_tuple[int(index)] for index in top_indices)
        postprocess_ms = (time.perf_counter() - started_post) * 1000.0

        total_ms = (time.perf_counter() - started_total) * 1000.0
        self._last_latency_ms = round(total_ms, 3)
        self._last_items = input_count
        self._last_batches = batches
        self._last_items_per_second = round(input_count / (total_ms / 1000.0), 2) if total_ms > 0 else None
        self._last_profile = {
            "feature_build_ms": round(preprocess_ms, 3),
            "preprocess_ms": round(preprocess_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "topk_ms": round(topk_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "total_ms": round(total_ms, 3),
            "input_count": input_count,
            "batch_bucket": bucket,
            "device": self._backend,
            "top_k": count,
        }
        return CandidateScoreBatch(top_tickers, top_scores, SCORE_SCHEMA, dict(self._last_profile))

    def score_full_debug(
        self,
        tickers: Sequence[str],
        feature_matrix: np.ndarray | Sequence[Sequence[float]],
    ) -> dict[str, tuple[float, ...]]:
        return self.score_candidates(tickers, feature_matrix, top_k=len(tickers)).score_dict()

    def classify(
        self,
        markets: tuple[MarketSnapshot, ...],
        indicators: Mapping[str, IndicatorSnapshot],
    ) -> dict[str, tuple[float, ...]]:
        tickers, features = feature_matrix_from_markets(markets, indicators)
        return self.score_full_debug(tickers, features)

    def status(self) -> OntologyNpuStatus:
        self._compiled_model(self.batch_size)
        return OntologyNpuStatus(
            backend=self._backend,
            uses_npu=self._backend.upper() == "NPU",
            batch_size=self.batch_size,
            feature_dim=self.feature_dim,
            score_dim=self.score_dim,
            model_name="ontology_candidate_scorer",
            requested_device=os.getenv("OPENVINO_DEVICE", os.getenv("ONTOLOGY_CLASSIFIER_DEVICE", "NPU")),
            selected_device=self._backend,
            model_kind="heuristic_linear_scorer",
            last_latency_ms=self._last_latency_ms,
            last_items=self._last_items,
            last_batches=self._last_batches,
            last_items_per_second=self._last_items_per_second,
            fallback_reason=self._fallback_reason,
            last_profile=dict(self._last_profile),
        )

    def _input_buffer(self, bucket: int) -> np.ndarray:
        buffer = self._batch_buffers.get(bucket)
        if buffer is None:
            buffer = np.zeros((bucket, self.feature_dim), dtype=np.float32)
            self._batch_buffers[bucket] = buffer
        return buffer

    def _compiled_model(self, bucket: int) -> Any:
        if bucket in self._compiled:
            return self._compiled[bucket]
        with self._lock:
            if bucket in self._compiled:
                return self._compiled[bucket]
            weights = _linear_weights()
            try:
                import openvino as ov
            except Exception as exc:
                self._compiled[bucket] = _NumpyLinearModel(weights)
                self._backend = "CPU_NUMPY"
                self._fallback_reason = f"OpenVINO unavailable: {exc}"
                return self._compiled[bucket]

            ops = ov.opset8
            x = ops.parameter([bucket, self.feature_dim], ov.Type.f32, name="ontology_features")
            model = ov.Model(
                [ops.matmul(x, ops.constant(weights), False, False)],
                [x],
                "ontology_npu_linear_scorer",
            )
            core = ov.Core()
            requested = os.getenv("OPENVINO_DEVICE", os.getenv("ONTOLOGY_CLASSIFIER_DEVICE", "NPU"))
            try:
                self._compiled[bucket] = core.compile_model(model, requested)
                self._backend = requested
                self._fallback_reason = None
            except Exception as exc:
                self._compiled[bucket] = core.compile_model(model, "CPU")
                self._backend = "CPU"
                self._fallback_reason = f"{requested} compile failed: {exc}"
            return self._compiled[bucket]


class _NumpyLinearModel:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.nan_to_num(inputs[0], nan=0.0, posinf=0.0, neginf=0.0) @ self.weights]


def feature_matrix_from_markets(
    markets: Sequence[MarketSnapshot],
    indicators: Mapping[str, IndicatorSnapshot],
) -> tuple[tuple[str, ...], np.ndarray]:
    tickers: list[str] = []
    matrix = np.zeros((len(markets), len(FEATURE_SCHEMA)), dtype=np.float32)
    row = 0
    for market in markets:
        indicator = indicators.get(market.ticker)
        if indicator is None:
            continue
        tickers.append(market.ticker)
        matrix[row, :] = _features(market, indicator)
        row += 1
    return tuple(tickers), matrix[:row, :]


def _normalize_feature_matrix(
    feature_matrix: np.ndarray | Sequence[Sequence[float]],
    feature_dim: int,
) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError(f"feature_matrix must have shape [N, {feature_dim}]")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _linear_weights() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.15, 0.0, 0.0, 0.0, 0.0],
            [0.22, 0.0, 0.18, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.30, 0.0, -0.08],
            [0.0, -0.45, 0.0, 0.0, 0.0, -0.10],
            [0.0, -0.35, -0.20, 0.0, -0.15, -0.08],
            [0.18, 0.0, 0.16, 0.0, 0.0, 0.08],
            [0.05, 0.0, 0.14, 0.0, 0.0, 0.12],
            [0.08, -0.05, 0.16, 0.0, 0.24, 0.14],
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


def _ranking_score(scores: np.ndarray) -> np.ndarray:
    return (
        scores[:, 0] * 0.40
        - scores[:, 1] * 0.35
        + scores[:, 2] * 0.18
        + scores[:, 3] * 0.12
        + scores[:, 4] * 0.10
        + scores[:, 5] * 0.22
    )


def _bucket_from_value(value: int | str) -> int:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return BATCH_BUCKETS[-1]
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = BATCH_BUCKETS[-1]
    return _bucket_for_count(max(1, numeric), BATCH_BUCKETS[-1])


def _bucket_for_count(count: int, max_bucket: int = BATCH_BUCKETS[-1]) -> int:
    allowed = tuple(bucket for bucket in BATCH_BUCKETS if bucket <= max_bucket) or (max_bucket,)
    for bucket in allowed:
        if count <= bucket:
            return bucket
    return allowed[-1]


def _profile(
    started_total: float,
    started_preprocess: float,
    inference_ms: float,
    postprocess_ms: float,
    input_count: int,
    batch_bucket: int,
    device: str,
    top_k: int,
) -> dict[str, float | int | str]:
    total_ms = (time.perf_counter() - started_total) * 1000.0
    return {
        "preprocess_ms": round((time.perf_counter() - started_preprocess) * 1000.0, 3),
        "inference_ms": round(inference_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
        "total_ms": round(total_ms, 3),
        "input_count": input_count,
        "batch_bucket": batch_bucket,
        "device": device,
        "top_k": top_k,
    }


OntologyNpuClassifier = OntologyNpuLinearScorer


_CLASSIFIER: OntologyNpuLinearScorer | None = None
_LOCK = Lock()


def get_ontology_npu_classifier() -> OntologyNpuLinearScorer:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        with _LOCK:
            if _CLASSIFIER is None:
                _CLASSIFIER = OntologyNpuClassifier(
                    batch_size=os.getenv("ONTOLOGY_NPU_BATCH_SIZE", "auto")
                )
    return _CLASSIFIER
