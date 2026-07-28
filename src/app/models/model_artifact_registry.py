from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    path: Path
    feature_schema_hash: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    expected_return_weights: tuple[float, ...]
    expected_return_bias: float
    thresholds: dict[str, float]
    metrics: dict[str, float]
    live_eligible: bool


class ModelArtifactRegistry:
    def __init__(self, root: str | Path = "data/models/live_short_horizon") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # In-memory cache of the parsed latest artifact, keyed by file mtime. The live
        # predictor calls load_latest_live_eligible() once per candidate AND per holding
        # every ~1s sweep; latest.json only changes on retrain (~300s), so re-reading and
        # JSON-parsing the weights every call was pure per-cycle waste. Invalidated
        # automatically when the file's mtime changes (training writes via os.replace).
        self._cache_key: int | None = None
        self._cache_artifact: ModelArtifact | None = None

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    def save(self, artifact: dict[str, Any]) -> Path:
        artifact_id = str(artifact["artifact_id"])
        path = self.root / f"{artifact_id}.json"
        incumbent = self._read_latest_payload()
        promoted, reason = _promotion_decision(artifact, incumbent)
        artifact["deployment"] = {
            "promoted": promoted,
            "reason": reason,
            "incumbent_artifact_id": (
                str(incumbent.get("artifact_id") or "") if incumbent else None
            ),
        }
        payload = json.dumps(artifact, indent=2, sort_keys=True)
        _atomic_write_text(path, payload)
        if promoted:
            # Only a live-eligible challenger that improves on the incumbent
            # becomes the active model. Threshold eligibility alone is not
            # sufficient to replace a stronger production model.
            _atomic_write_text(self.latest_path, payload)
        return path

    def _read_latest_payload(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        try:
            payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def load_latest_live_eligible(self) -> ModelArtifact:
        if not self.latest_path.exists():
            raise RuntimeError("NO_LIVE_ELIGIBLE_MODEL_ARTIFACT")
        artifact = self._load_latest_cached()
        if not artifact.live_eligible:
            raise RuntimeError("LATEST_MODEL_NOT_LIVE_ELIGIBLE")
        return artifact

    def _load_latest_cached(self) -> ModelArtifact:
        try:
            mtime = self.latest_path.stat().st_mtime_ns
        except OSError:
            mtime = None
        cached = self._cache_artifact
        if mtime is not None and self._cache_key == mtime and cached is not None:
            return cached
        payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        artifact = _artifact_from_payload(payload, self.latest_path)
        # Publish the artifact before the key so a concurrent reader that sees the new
        # key also sees the matching artifact (write is under the GIL; worst case is a
        # harmless redundant reload).
        self._cache_artifact = artifact
        self._cache_key = mtime
        return artifact


def _atomic_write_text(path: Path, text: str) -> None:
    # 같은 디렉터리에 임시 파일로 쓴 뒤 원자적으로 교체한다. 라이브 예측기는 매 예측마다
    # latest.json을 다시 읽으므로, 재학습 중 부분 기록된 파일을 읽어 파싱 오류가 나는 것을 막는다.
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _promotion_decision(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> tuple[bool, str]:
    if candidate.get("live_eligible") is not True:
        return False, "NOT_LIVE_ELIGIBLE"
    if not incumbent or incumbent.get("live_eligible") is not True:
        return True, "FIRST_LIVE_ELIGIBLE_MODEL"

    candidate_metrics = candidate.get("metrics") or {}
    incumbent_metrics = incumbent.get("metrics") or {}
    candidate_auc = _finite_metric(candidate_metrics, "auc")
    candidate_precision = _finite_metric(candidate_metrics, "precision_at_k")
    candidate_return = _finite_metric(
        candidate_metrics,
        "avg_forward_net_return_bps_top_k",
    )
    incumbent_auc = _finite_metric(incumbent_metrics, "auc")
    incumbent_precision = _finite_metric(incumbent_metrics, "precision_at_k")
    incumbent_return = _finite_metric(
        incumbent_metrics,
        "avg_forward_net_return_bps_top_k",
    )

    max_auc_drop = _env_float("LIVE_MODEL_PROMOTION_MAX_AUC_DROP", 0.01)
    max_precision_drop = _env_float("LIVE_MODEL_PROMOTION_MAX_PRECISION_DROP", 0.02)
    max_return_drop = _env_float("LIVE_MODEL_PROMOTION_MAX_RETURN_BPS_DROP", 2.0)
    no_material_regression = (
        candidate_auc >= incumbent_auc - max_auc_drop
        and candidate_precision >= incumbent_precision - max_precision_drop
        and candidate_return >= incumbent_return - max_return_drop
    )
    improves = (
        candidate_auc >= incumbent_auc + _env_float("LIVE_MODEL_PROMOTION_MIN_AUC_GAIN", 0.005)
        or candidate_precision
        >= incumbent_precision
        + _env_float("LIVE_MODEL_PROMOTION_MIN_PRECISION_GAIN", 0.01)
        or candidate_return
        >= incumbent_return
        + _env_float("LIVE_MODEL_PROMOTION_MIN_RETURN_BPS_GAIN", 1.0)
    )
    if not no_material_regression:
        return False, "CHALLENGER_REGRESSES_ACTIVE_MODEL"
    if not improves:
        return False, "NO_MEASURABLE_IMPROVEMENT"
    return True, "CHALLENGER_IMPROVED_ACTIVE_MODEL"


def _finite_metric(metrics: dict[str, Any], key: str) -> float:
    try:
        value = float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _artifact_from_payload(payload: dict[str, Any], path: Path) -> ModelArtifact:
    return ModelArtifact(
        artifact_id=str(payload["artifact_id"]),
        path=path,
        feature_schema_hash=str(payload["feature_schema_hash"]),
        feature_names=tuple(payload["feature_names"]),
        weights=tuple(float(value) for value in payload["classification"]["weights"]),
        bias=float(payload["classification"]["bias"]),
        expected_return_weights=tuple(float(value) for value in payload["regression"]["weights"]),
        expected_return_bias=float(payload["regression"]["bias"]),
        thresholds={str(key): float(value) for key, value in payload["thresholds"].items()},
        metrics={str(key): float(value) for key, value in payload["metrics"].items()},
        live_eligible=bool(payload["live_eligible"]),
    )
