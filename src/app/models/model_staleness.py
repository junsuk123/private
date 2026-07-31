"""Model expiry: when may ``latest.json`` no longer be believed?

The gap this closes
-------------------
The registry preserved the last live-eligible artifact whenever a challenger
failed. That is right for file safety and wrong for regime change: an incumbent
trained on 722 examples from four symbols kept serving live predictions while
every subsequent challenger — 8,126 examples, 23 symbols — measured an AUC of
0.49 and a top-25 net return of -54bp. The most likely reading is not "the
incumbent generalises better"; it is "the incumbent is overfitted to a market
that no longer exists", and nothing in the system could say so.

So an artifact now expires. It is demoted when ANY of these holds:

    model_age            > maximum allowed age
    feature_drift_score  > threshold
    the last N challengers all measured a negative top-K net return
    the current market regime differs from the one it was trained in

Demotion is graded, not binary — the caller decides how far to fall back:

    trained_model -> shadow_only -> ontology / bandit / no_trade

This module only produces the verdict. It never deletes an artifact and never
rewrites ``latest.json``; the file stays on disk for audit and can be reinstated
by relaxing the policy.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

MODEL_AGE_EXCEEDED = "MODEL_AGE_EXCEEDED"
MODEL_FEATURE_DRIFT = "MODEL_FEATURE_DRIFT_EXCEEDED"
MODEL_CONSECUTIVE_NEGATIVE_CHALLENGERS = "MODEL_CONSECUTIVE_NEGATIVE_CHALLENGERS"
MODEL_REGIME_MISMATCH = "MODEL_REGIME_MISMATCH"
MODEL_CREATED_AT_UNKNOWN = "MODEL_CREATED_AT_UNKNOWN"
MODEL_FRESH = "MODEL_FRESH"


class ModelTrustLevel(str, Enum):
    """How far a caller may trust the artifact."""

    LIVE = "LIVE"
    SHADOW_ONLY = "SHADOW_ONLY"
    UNUSABLE = "UNUSABLE"


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StalenessConfig:
    # 6 hours: longer than a single session, short enough that an artifact cannot
    # survive a weekend of repricing and still be called live.
    max_age_seconds: float = 21_600.0
    max_feature_drift_score: float = 0.35
    consecutive_negative_challengers: int = 3
    enforce_regime_match: bool = True
    # An artifact whose age is unknown is treated as stale: an unparseable
    # timestamp is not evidence of freshness.
    treat_unknown_age_as_stale: bool = True
    # Demoted artifacts are still useful for shadow scoring and diagnostics.
    demoted_trust_level: ModelTrustLevel = ModelTrustLevel.SHADOW_ONLY

    @classmethod
    def from_env(cls) -> "StalenessConfig":
        return cls(
            max_age_seconds=max(
                60.0, _env_float("LIVE_MODEL_MAX_AGE_SECONDS", cls.max_age_seconds)
            ),
            max_feature_drift_score=_env_float(
                "LIVE_MODEL_MAX_FEATURE_DRIFT", cls.max_feature_drift_score
            ),
            consecutive_negative_challengers=max(
                1,
                _env_int(
                    "LIVE_MODEL_MAX_CONSECUTIVE_NEGATIVE_CHALLENGERS",
                    cls.consecutive_negative_challengers,
                ),
            ),
            enforce_regime_match=_env_bool(
                "LIVE_MODEL_ENFORCE_REGIME_MATCH", cls.enforce_regime_match
            ),
            treat_unknown_age_as_stale=_env_bool(
                "LIVE_MODEL_UNKNOWN_AGE_IS_STALE", cls.treat_unknown_age_as_stale
            ),
        )


@dataclass(frozen=True)
class StalenessVerdict:
    stale: bool
    trust_level: ModelTrustLevel
    reason_codes: tuple[str, ...]
    age_seconds: float | None
    feature_drift_score: float | None
    consecutive_negative_challengers: int
    training_regime: str | None
    current_regime: str | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stale": self.stale,
            "trust_level": self.trust_level.value,
            "reason_codes": list(self.reason_codes),
            "age_seconds": self.age_seconds,
            "feature_drift_score": self.feature_drift_score,
            "consecutive_negative_challengers": self.consecutive_negative_challengers,
            "training_regime": self.training_regime,
            "current_regime": self.current_regime,
            "diagnostics": dict(self.diagnostics),
        }


def evaluate_model_staleness(
    payload: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
    config: StalenessConfig | None = None,
    feature_drift_score: float | None = None,
    recent_challenger_return_bps: Sequence[float] = (),
    consecutive_negative_challengers: int | None = None,
    current_regime: str | None = None,
) -> StalenessVerdict:
    """Decide whether this artifact may still serve live predictions."""
    cfg = config or StalenessConfig.from_env()
    moment = now or datetime.now(timezone.utc)
    moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    reasons: list[str] = []

    if not payload:
        return StalenessVerdict(
            stale=True,
            trust_level=ModelTrustLevel.UNUSABLE,
            reason_codes=(MODEL_CREATED_AT_UNKNOWN,),
            age_seconds=None,
            feature_drift_score=feature_drift_score,
            consecutive_negative_challengers=0,
            training_regime=None,
            current_regime=current_regime,
        )

    created = _parse_iso(payload.get("created_at"))
    age_seconds = max(0.0, (moment - created).total_seconds()) if created else None
    if age_seconds is None:
        if cfg.treat_unknown_age_as_stale:
            reasons.append(MODEL_CREATED_AT_UNKNOWN)
    elif age_seconds > cfg.max_age_seconds:
        reasons.append(MODEL_AGE_EXCEEDED)

    drift = _finite(feature_drift_score)
    if drift is None:
        drift = _finite((payload.get("metrics") or {}).get("feature_drift_score"))
    if drift is not None and drift > cfg.max_feature_drift_score:
        reasons.append(MODEL_FEATURE_DRIFT)

    negative_streak = consecutive_negative_challengers
    if negative_streak is None:
        negative_streak = 0
        for value in recent_challenger_return_bps:
            number = _finite(value)
            if number is None or number > 0.0:
                break
            negative_streak += 1
    if negative_streak >= cfg.consecutive_negative_challengers:
        # Every recent independent re-measurement says there is no edge. The
        # incumbent's own metrics are the outlier, not the evidence.
        reasons.append(MODEL_CONSECUTIVE_NEGATIVE_CHALLENGERS)

    training_regime = _training_regime(payload)
    normalized_current = _normalize_regime(current_regime)
    if (
        cfg.enforce_regime_match
        and training_regime
        and normalized_current
        and _normalize_regime(training_regime) != normalized_current
    ):
        reasons.append(MODEL_REGIME_MISMATCH)

    stale = bool(reasons)
    if not stale:
        reasons.append(MODEL_FRESH)
    return StalenessVerdict(
        stale=stale,
        trust_level=cfg.demoted_trust_level if stale else ModelTrustLevel.LIVE,
        reason_codes=tuple(dict.fromkeys(reasons)),
        age_seconds=age_seconds,
        feature_drift_score=drift,
        consecutive_negative_challengers=int(negative_streak),
        training_regime=training_regime,
        current_regime=normalized_current,
        diagnostics={
            "artifact_id": str(payload.get("artifact_id") or ""),
            "max_age_seconds": cfg.max_age_seconds,
            "max_feature_drift_score": cfg.max_feature_drift_score,
            "consecutive_negative_challenger_limit": cfg.consecutive_negative_challengers,
        },
    )


def _training_regime(payload: Mapping[str, Any]) -> str | None:
    for source in (payload.get("training_state"), payload.get("training_data"), payload):
        if isinstance(source, Mapping):
            value = source.get("macro_regime") or source.get("training_regime")
            if value:
                return str(value)
    return None


def _normalize_regime(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
