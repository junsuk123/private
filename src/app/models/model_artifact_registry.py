from __future__ import annotations

import json
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

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    def save(self, artifact: dict[str, Any]) -> Path:
        artifact_id = str(artifact["artifact_id"])
        path = self.root / f"{artifact_id}.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        if artifact.get("live_eligible") is True:
            self.latest_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_latest_live_eligible(self) -> ModelArtifact:
        if not self.latest_path.exists():
            raise RuntimeError("NO_LIVE_ELIGIBLE_MODEL_ARTIFACT")
        payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        artifact = _artifact_from_payload(payload, self.latest_path)
        if not artifact.live_eligible:
            raise RuntimeError("LATEST_MODEL_NOT_LIVE_ELIGIBLE")
        return artifact


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
