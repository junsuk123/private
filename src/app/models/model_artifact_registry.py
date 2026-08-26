from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.model_staleness import (
    ModelTrustLevel,
    StalenessConfig,
    StalenessVerdict,
    evaluate_model_staleness,
)


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
    created_at: str = ""
    trust_level: str = ModelTrustLevel.LIVE.value

    @property
    def shadow_only(self) -> bool:
        return self.trust_level == ModelTrustLevel.SHADOW_ONLY.value


class ModelArtifactRegistry:
    def __init__(self, root: str | Path = "data/models/live_short_horizon") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # In-memory cache of the parsed latest artifact, keyed by an identity triple
        # for latest.json. The live predictor calls load_latest_live_eligible() once
        # per candidate AND per holding every ~1s sweep; latest.json only changes on
        # retrain (~300s), so re-reading and JSON-parsing the weights every call was
        # pure per-cycle waste.
        #
        # The key is (mtime_ns, size, inode), not mtime_ns alone. A file timestamp is
        # only as fine as the filesystem and the kernel's cached clock make it - on a
        # 250Hz kernel that is 4ms even in the nanosecond field - so two writes inside
        # one tick produce an IDENTICAL mtime_ns and the cache would keep serving the
        # superseded model. Inode covers the production path exactly, because training
        # publishes with os.replace and that always swaps in a new inode; size covers
        # an in-place rewrite that changes length. Nothing here weakens the guarantee,
        # it only closes the window where mtime alone could not see a change.
        self._cache_key: tuple[int, int, int] | None = None
        self._cache_artifact: ModelArtifact | None = None

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    @property
    def deployment_state_path(self) -> Path:
        """Sidecar tracking challenger history across retrains.

        Kept beside ``latest.json`` rather than inside it because the incumbent
        payload is only rewritten on promotion — which is exactly the case where
        the interesting history (repeated FAILED challengers) is being recorded.
        """
        return self.root / "deployment_state.json"

    def save(self, artifact: dict[str, Any]) -> Path:
        artifact_id = str(artifact["artifact_id"])
        path = self.root / f"{artifact_id}.json"
        incumbent = self._read_latest_payload()
        candidate_stale = evaluate_model_staleness(artifact).stale
        incumbent_stale = bool(incumbent) and self.staleness().stale
        promoted, reason = _promotion_decision(
            artifact,
            incumbent,
            candidate_stale=candidate_stale,
            incumbent_stale=incumbent_stale,
        )
        deployment_state = self._record_challenger(artifact, promoted)
        artifact["deployment"] = {
            "promoted": promoted,
            "reason": reason,
            "incumbent_artifact_id": (
                str(incumbent.get("artifact_id") or "") if incumbent else None
            ),
            "consecutive_negative_challengers": int(
                deployment_state.get("consecutive_negative_challengers") or 0
            ),
        }
        payload = json.dumps(artifact, indent=2, sort_keys=True)
        _atomic_write_text(path, payload)
        if promoted:
            # Only a live-eligible challenger that improves on the incumbent
            # becomes the active model. Threshold eligibility alone is not
            # sufficient to replace a stronger production model.
            _atomic_write_text(self.latest_path, payload)
        self._prune_saved_artifacts(
            protected_artifact_ids={
                artifact_id,
                str((incumbent or {}).get("artifact_id") or ""),
            }
        )
        return path

    def _prune_saved_artifacts(self, *, protected_artifact_ids: set[str]) -> int:
        """Bound challenger history without ever removing the active model.

        Training creates a challenger every cycle. On a continuously running
        workstation that otherwise leaves thousands of JSON files in a synced
        directory, making status checks and backup/sync progressively slower.
        ``latest.json`` and deployment state are sidecars and are never matched.
        """
        try:
            limit = max(
                2,
                int(float(os.getenv("LIVE_MODEL_ARTIFACT_RETENTION_COUNT", "512"))),
            )
        except (TypeError, ValueError):
            limit = 512
        try:
            candidates = sorted(
                (
                    candidate
                    for candidate in self.root.glob("live_short_horizon.*.json")
                    if candidate.is_file()
                ),
                key=lambda candidate: candidate.name,
                reverse=True,
            )
        except OSError:
            return 0
        keep = set(candidates[:limit])
        keep.update(
            self.root / f"{artifact_id}.json"
            for artifact_id in protected_artifact_ids
            if artifact_id
        )
        removed = 0
        for candidate in candidates:
            if candidate in keep:
                continue
            try:
                candidate.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    def _read_latest_payload(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        try:
            payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _record_challenger(self, artifact: dict[str, Any], promoted: bool) -> dict[str, Any]:
        """Update the consecutive-negative-challenger counter.

        A challenger measuring a non-positive top-K net return is a vote that the
        current market has no edge for this model family. Three such votes in a row
        expire the incumbent, however good its own historical metrics look — which
        is the situation this whole mechanism exists for.
        """
        state = self._read_deployment_state()
        metrics = artifact.get("metrics") or {}
        top_k_return = _finite_metric(metrics, "avg_forward_net_return_bps_top_k")
        eligible = artifact.get("live_eligible") is True
        streak = int(state.get("consecutive_negative_challengers") or 0)
        if eligible and top_k_return > 0.0:
            streak = 0
        else:
            streak += 1
        if promoted:
            # A promoted challenger becomes the incumbent, so the history it was
            # measured against no longer applies to the model now serving.
            streak = 0
        updated = {
            "consecutive_negative_challengers": streak,
            "last_challenger_artifact_id": str(artifact.get("artifact_id") or ""),
            "last_challenger_at": datetime.now(timezone.utc).isoformat(),
            "last_challenger_top_k_net_bps": top_k_return,
            "last_challenger_live_eligible": eligible,
            "last_challenger_promoted": bool(promoted),
            "recent_challenger_top_k_net_bps": (
                [top_k_return, *list(state.get("recent_challenger_top_k_net_bps") or ())][:10]
            ),
        }
        try:
            _atomic_write_text(
                self.deployment_state_path,
                json.dumps(updated, indent=2, sort_keys=True),
            )
        except OSError:
            pass
        return updated

    def _read_deployment_state(self) -> dict[str, Any]:
        if not self.deployment_state_path.exists():
            return {}
        try:
            payload = json.loads(self.deployment_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def staleness(
        self,
        *,
        now: datetime | None = None,
        config: StalenessConfig | None = None,
        current_regime: str | None = None,
        feature_drift_score: float | None = None,
    ) -> StalenessVerdict:
        """Expiry verdict for the currently deployed artifact."""
        state = self._read_deployment_state()
        return evaluate_model_staleness(
            self._read_latest_payload(),
            now=now,
            config=config,
            feature_drift_score=feature_drift_score,
            consecutive_negative_challengers=int(
                state.get("consecutive_negative_challengers") or 0
            ),
            current_regime=current_regime,
        )

    def load_latest_live_eligible(
        self,
        *,
        current_regime: str | None = None,
        enforce_staleness: bool | None = None,
    ) -> ModelArtifact:
        if not self.latest_path.exists():
            raise RuntimeError("NO_LIVE_ELIGIBLE_MODEL_ARTIFACT")
        artifact = self._load_latest_cached()
        if not artifact.live_eligible:
            raise RuntimeError("LATEST_MODEL_NOT_LIVE_ELIGIBLE")
        enforce = (
            _env_bool("LIVE_MODEL_ENFORCE_STALENESS", True)
            if enforce_staleness is None
            else bool(enforce_staleness)
        )
        if not enforce:
            return artifact
        verdict = self.staleness(current_regime=current_regime)
        if verdict.stale:
            # The artifact stays on disk for audit; callers that treat a raise as
            # "no trained model" then fall back to ontology / bandit / no_trade,
            # which is the intended graded demotion.
            raise RuntimeError(
                "LATEST_MODEL_STALE:" + ",".join(verdict.reason_codes)
            )
        return artifact

    def _load_latest_cached(self) -> ModelArtifact:
        try:
            status = self.latest_path.stat()
            key: tuple[int, int, int] | None = (
                status.st_mtime_ns,
                status.st_size,
                status.st_ino,
            )
        except OSError:
            key = None
        cached = self._cache_artifact
        if key is not None and self._cache_key == key and cached is not None:
            return cached
        payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        artifact = _artifact_from_payload(payload, self.latest_path)
        # Publish the artifact before the key so a concurrent reader that sees the new
        # key also sees the matching artifact (write is under the GIL; worst case is a
        # harmless redundant reload).
        self._cache_artifact = artifact
        self._cache_key = key
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


def _meets_absolute_economics(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Does this artifact pay for a round trip on its OWN measurements?

    Relative promotion is necessary but not sufficient. The floors below are the
    same quantities the runtime already believes in — the artifact's own
    ``minimum_expected_net_return_bps`` threshold and its measured top-k net
    return — so this adds no new tunable that could be relaxed to manufacture a
    promotion. Sample-size floors exist because a top-k of 3 rows can show any
    number at all.

    Env overrides raise, never lower, the bar unless explicitly configured.
    """
    metrics = candidate.get("metrics") or {}
    thresholds = candidate.get("thresholds") or {}

    # Holdout evidence is a REQUIREMENT, not a tunable.
    #
    # The trainer falls back to evaluating on the training set itself when the
    # sample is too small to split. Those metrics are a fit, not a measurement:
    # observed on 2026-08-06 a freshly-seeded artifact reported auc 0.9865,
    # precision@k 1.000 and +43.8bps top-k with validation_example_count == 0, and
    # reached live order authority because the sample-size floors below default to
    # off. A configurable floor was the wrong instrument -- the question is not
    # "how many rows" but "was this ever measured out of sample at all".
    #
    # Enforced only when the field is PRESENT: absence means an artifact written
    # before the field existed, and a floor cannot be applied to something that was
    # never recorded. The live trainer always records both.
    if "holdout_evaluated" in metrics and _finite_metric(metrics, "holdout_evaluated") <= 0.0:
        return False, "IN_SAMPLE_METRICS_ONLY"
    if (
        "validation_example_count" in metrics
        and _finite_metric(metrics, "validation_example_count") <= 0.0
    ):
        return False, "NO_HOLDOUT_VALIDATION"

    minimum_net = _env_float(
        "LIVE_MODEL_PROMOTION_MIN_TOP_K_NET_BPS",
        _finite_metric(thresholds, "minimum_expected_net_return_bps"),
    )
    top_k_net = _finite_metric(metrics, "avg_forward_net_return_bps_top_k")
    if top_k_net <= 0.0:
        return False, "TOP_K_NET_RETURN_NON_POSITIVE"
    if minimum_net > 0.0 and top_k_net < minimum_net:
        return False, "TOP_K_NET_RETURN_BELOW_RUNTIME_MINIMUM"
    if _finite_metric(metrics, "runtime_policy_aligned_evaluation") >= 1.0:
        minimum_deployable = max(
            2,
            int(_env_float("LIVE_MODEL_MIN_DEPLOYABLE_HOLDOUT_ROWS", 10.0)),
        )
        if _finite_metric(metrics, "top_k_count") < minimum_deployable:
            return False, "DEPLOYABLE_HOLDOUT_SAMPLE_TOO_SMALL"

    min_precision = _env_float("LIVE_MODEL_PROMOTION_MIN_PRECISION_AT_K", 0.0)
    if min_precision > 0.0 and _finite_metric(metrics, "precision_at_k") < min_precision:
        return False, "PRECISION_AT_K_BELOW_MINIMUM"

    # Sample-size floors. A holdout of 16 rows across 3 symbols is a holdout in
    # name only: on 2026-08-06 exactly that produced auc 0.983 / +35bps and took
    # live authority, which is noise wearing a measurement's clothes. Defaulting
    # these OFF (an earlier choice made to keep small-fixture tests green) is what
    # allowed it, so they now default ON at production-scale values.
    #
    # Still enforced only when the artifact RECORDED the quantity: an absent count
    # is unknown, not zero, and rejecting on absence would block every artifact
    # written before the field existed. The live trainer always writes both.
    for metric_name, env_name, code in (
        (
            "validation_example_count",
            "LIVE_MODEL_PROMOTION_MIN_VALIDATION_ROWS",
            "VALIDATION_SAMPLE_TOO_SMALL",
        ),
        (
            "validation_symbol_count",
            "LIVE_MODEL_PROMOTION_MIN_VALIDATION_SYMBOLS",
            "VALIDATION_SYMBOL_COUNT_TOO_SMALL",
        ),
    ):
        default = 200.0 if metric_name == "validation_example_count" else 5.0
        minimum = _env_float(env_name, default)
        if minimum <= 0.0 or metric_name not in metrics:
            continue
        if _finite_metric(metrics, metric_name) < minimum:
            return False, code

    # Lower confidence bound on the top-k mean. Without a dispersion measure in the
    # artifact this uses top_k_count as the effective n and treats the mean itself
    # as the scale, which is deliberately crude but strictly conservative: it can
    # only reject, never admit, relative to the point estimate.
    top_k_count = _finite_metric(metrics, "top_k_count")
    if top_k_count >= 2.0:
        standard_error = abs(top_k_net) / math.sqrt(top_k_count)
        if top_k_net - standard_error <= 0.0:
            return False, "TOP_K_NET_RETURN_LOWER_BOUND_NON_POSITIVE"
    return True, "ABSOLUTE_ECONOMICS_OK"


def _schema_is_current(artifact: dict[str, Any] | None) -> bool:
    """Was this artifact trained on the feature schema the process runs today?

    Imported lazily: ``feature_schema`` is a leaf module, but the registry is
    imported by the training pipeline that also builds frames, and a module-level
    import here would tie artifact bookkeeping to feature construction.
    """
    from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA

    if not artifact:
        return False
    recorded = str(artifact.get("feature_schema_hash") or "")
    # An artifact that never recorded a hash predates the field; treat it as
    # unknown-and-therefore-not-current rather than assuming a match.
    return recorded == LIVE_SHORT_HORIZON_SCHEMA.schema_hash


def _promotion_decision(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    candidate_stale: bool = False,
    incumbent_stale: bool = False,
) -> tuple[bool, str]:
    if candidate.get("live_eligible") is not True:
        return False, "NOT_LIVE_ELIGIBLE"
    # Absolute economic floor, checked before any RELATIVE comparison.
    #
    # Every other branch here is relative: better than the incumbent, or the
    # incumbent is stale/obsolete. That let a model reach live authority on
    # relative merit alone -- "first eligible" and "stale incumbent replaced" both
    # promote without ever asking whether the candidate's own top-k net return is
    # large enough to pay for a round trip. A model whose best decile loses money
    # is not a usable model just because the alternative is worse.
    economic_ok, economic_reason = _meets_absolute_economics(candidate)
    if not economic_ok:
        return False, economic_reason
    if not incumbent or incumbent.get("live_eligible") is not True:
        return True, "FIRST_LIVE_ELIGIBLE_MODEL"
    if candidate_stale:
        return False, "CHALLENGER_STALE"
    if incumbent_stale:
        return True, "STALE_INCUMBENT_REPLACED"
    # An incumbent trained on a retired feature schema cannot serve a prediction at
    # all — live_signal_predictor raises MODEL_FEATURE_SCHEMA_MISMATCH — so it must
    # not be allowed to veto its own replacement. The metric comparison below is
    # meaningless across schemas: precision_at_k computed over a different feature
    # set is a different measurement, and treating it as a regression is what let a
    # dead incumbent block every successor. Observed on the 2026-08-05 v4->v5 change,
    # where a v5 candidate (net +19.75bps, eligible) lost to a v4 incumbent's higher
    # precision and the schema migration could never land.
    if _schema_is_current(candidate) and not _schema_is_current(incumbent):
        return True, "OBSOLETE_SCHEMA_INCUMBENT_REPLACED"

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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
        created_at=str(payload.get("created_at") or ""),
    )
