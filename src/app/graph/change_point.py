"""Bayesian online change-point detection over market-level channels.

Why this exists
---------------
Every learned model and every accumulated per-strategy performance statistic in
this system is conditioned on "the market behaves like it did while I was
fitted". When the market re-prices regime (2026-07-28: KOSPI -10.84% with a
market-wide circuit breaker, then +4% two days later), that conditioning silently
breaks: the model keeps producing confident scores and the bandit keeps trusting
a history that no longer describes the present.

This module answers ONE question, before any strategy is scored:

    can the current model / performance history still be believed?

It is deliberately NOT a buy strategy. Its outputs are consumed by
:mod:`app.graph.macro_reasoner` (sub-regime classification), by
:mod:`app.trading.conservative_bandit` (history discounting) and by
:mod:`app.models.model_staleness` (model demotion).

Method
------
Adams & MacKay (2007) Bayesian online change-point detection with a constant
hazard and a Normal–Gamma conjugate prior per channel, so the predictive is a
Student-t and the run-length posterior is updated exactly in closed form. Each
channel is tracked independently (a diagonal approximation — we do not estimate
the cross-channel covariance, which would need far more data than a trading
session provides).

Combining channels is where a naive implementation goes wrong. Taking the max
per-channel probability makes one noisy channel able to declare a regime break
and freeze all trading; taking the mean makes a genuine single-driver break
invisible. This module therefore uses the same corroboration rule the repo
already applies to macro news shocks: the combined probability is the
``corroborating_channels``-th largest per-channel probability (2nd largest by
default), so two independent channels must agree before a change point is
declared.

Everything is NaN-safe and pure arithmetic: a missing channel is skipped, never
imputed. With no history at all the detector reports probability 0.0 and the
reason code ``MACRO_CHANGE_POINT_INSUFFICIENT_HISTORY`` — "I cannot tell" is
never reported as "nothing changed".
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# --- Reason codes (advisory diagnostics, never authorization outcomes) ------- #
CHANGE_POINT_DETECTED = "MACRO_CHANGE_POINT_DETECTED"
CHANGE_POINT_SUSPECTED = "MACRO_CHANGE_POINT_SUSPECTED"
CHANGE_POINT_REGIME_YOUNG = "MACRO_REGIME_TOO_YOUNG_FOR_TRUST"
CHANGE_POINT_INSUFFICIENT_HISTORY = "MACRO_CHANGE_POINT_INSUFFICIENT_HISTORY"
CHANGE_POINT_STABLE = "MACRO_REGIME_STABLE"

DEFAULT_STATE_PATH = "data/store/change-point-state.json"

# The channels the market feed can supply. A deployment that cannot compute one
# simply omits it; the detector degrades to the channels it actually received.
CHANGE_POINT_CHANNELS: tuple[str, ...] = (
    "index_return",
    "market_volatility",
    "market_breadth",
    "cross_sectional_dispersion",
    "average_correlation",
    "foreign_flow_zscore",
    "spread_percentile",
    "strategy_prediction_error_bps",
)


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


@dataclass(frozen=True)
class ChangePointConfig:
    """Detector knobs. Defaults are deliberately conservative, not tuned-in."""

    # Expected run length in observations. The macro loop runs once a minute, so
    # 120 == "a regime typically lasts about two hours of session time".
    expected_run_length: float = 120.0
    max_run_length: int = 240
    # Normal-Gamma prior. kappa0/alpha0 are pseudo-counts: small enough that a
    # real shift is picked up within a handful of observations, large enough that
    # the very first observations cannot manufacture a change point.
    prior_kappa: float = 1.0
    prior_alpha: float = 1.5
    prior_beta: float = 1.0
    # Causal window used to standardise each channel before the run-length filter.
    # Long enough that a level shift is not absorbed into the baseline within a few
    # observations, short enough to forget a regime that ended hours ago.
    standardization_window: int = 240
    # The detection statistic is P(current run length <= this), NOT P(run length = 0).
    # Those are different quantities and only the first one detects anything: at the
    # first post-shift observation the old long run's predictive collapses, but the
    # surviving mass moves to run length 1, not 0, so P(r = 0) sits at the hazard
    # rate forever. "The current regime is only a few observations old" is the
    # statement that actually means a break just happened.
    short_run_length: int = 3
    # A channel needs at least this many observations before its probability is
    # allowed to contribute to the combined verdict.
    min_channel_observations: int = 8
    # How many channels must independently agree before a change point is called.
    corroborating_channels: int = 2
    detection_threshold: float = 0.5
    suspicion_threshold: float = 0.25
    # A regime younger than this is not yet trustworthy for live sizing even when
    # the change-point probability has fallen back to zero.
    regime_trust_age_seconds: float = 900.0

    @classmethod
    def from_env(cls) -> "ChangePointConfig":
        return cls(
            expected_run_length=max(
                5.0, _env_float("CHANGE_POINT_EXPECTED_RUN_LENGTH", cls.expected_run_length)
            ),
            max_run_length=max(
                10, _env_int("CHANGE_POINT_MAX_RUN_LENGTH", cls.max_run_length)
            ),
            standardization_window=max(
                8, _env_int("CHANGE_POINT_STANDARDIZATION_WINDOW", cls.standardization_window)
            ),
            short_run_length=max(
                0, _env_int("CHANGE_POINT_SHORT_RUN_LENGTH", cls.short_run_length)
            ),
            min_channel_observations=max(
                2, _env_int("CHANGE_POINT_MIN_CHANNEL_OBSERVATIONS", cls.min_channel_observations)
            ),
            corroborating_channels=max(
                1, _env_int("CHANGE_POINT_CORROBORATING_CHANNELS", cls.corroborating_channels)
            ),
            detection_threshold=_env_float(
                "CHANGE_POINT_DETECTION_THRESHOLD", cls.detection_threshold
            ),
            suspicion_threshold=_env_float(
                "CHANGE_POINT_SUSPICION_THRESHOLD", cls.suspicion_threshold
            ),
            regime_trust_age_seconds=max(
                0.0,
                _env_float("CHANGE_POINT_REGIME_TRUST_AGE_SECONDS", cls.regime_trust_age_seconds),
            ),
        )

    @property
    def hazard(self) -> float:
        return 1.0 / max(1.0, float(self.expected_run_length))


@dataclass(frozen=True)
class ChangePointResult:
    """What the rest of the system consumes."""

    timestamp: datetime
    change_point_probability: float
    current_regime_age_seconds: float
    regime_stability: float
    observation_count: int
    channel_probabilities: Mapping[str, float]
    contributing_channels: tuple[str, ...]
    reason_codes: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def change_point_detected(self) -> bool:
        return CHANGE_POINT_DETECTED in self.reason_codes

    @property
    def history_trustworthy(self) -> bool:
        """May accumulated per-strategy history be used for live sizing?"""
        return not (
            CHANGE_POINT_DETECTED in self.reason_codes
            or CHANGE_POINT_INSUFFICIENT_HISTORY in self.reason_codes
            or CHANGE_POINT_REGIME_YOUNG in self.reason_codes
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "change_point_probability": round(self.change_point_probability, 6),
            "current_regime_age_seconds": round(self.current_regime_age_seconds, 3),
            "regime_stability": round(self.regime_stability, 6),
            "observation_count": self.observation_count,
            "channel_probabilities": {
                str(name): round(float(value), 6)
                for name, value in sorted(self.channel_probabilities.items())
            },
            "contributing_channels": list(self.contributing_channels),
            "reason_codes": list(self.reason_codes),
            "change_point_detected": self.change_point_detected,
            "history_trustworthy": self.history_trustworthy,
            "diagnostics": dict(self.diagnostics),
        }


def _student_t_log_pdf(x: float, mu: float, precision_scale: float, df: float) -> float:
    """log Student-t density; ``precision_scale`` is the scale (not variance)."""
    scale = max(1e-12, float(precision_scale))
    dof = max(1e-6, float(df))
    z = (float(x) - float(mu)) / scale
    return (
        math.lgamma((dof + 1.0) / 2.0)
        - math.lgamma(dof / 2.0)
        - 0.5 * math.log(math.pi * dof)
        - math.log(scale)
        - ((dof + 1.0) / 2.0) * math.log1p(z * z / dof)
    )


class _ChannelDetector:
    """Exact BOCPD run-length posterior for one scalar channel.

    Observations are standardised against the channel's own causal history before
    they reach the run-length filter. This is not cosmetic: the Normal-Gamma prior
    has unit-ish scale, and the raw channels do not (``index_return`` lives near
    1e-3, ``market_breadth`` near 0.5). Fed raw, the prior predictive is orders of
    magnitude flatter than the fitted run's, the growth branch wins every time,
    and P(run length = 0) collapses to the hazard rate — a detector that can never
    detect anything. Standardising puts both branches on the same footing, and the
    Student-t's heavy tails then make a genuine level shift stand out.
    """

    __slots__ = (
        "config",
        "run_length",
        "mu",
        "kappa",
        "alpha",
        "beta",
        "observations",
        "history",
    )

    def __init__(self, config: ChangePointConfig, *, seed_mean: float = 0.0) -> None:
        self.config = config
        self.run_length: list[float] = [1.0]
        self.mu: list[float] = [0.0]
        self.kappa: list[float] = [float(config.prior_kappa)]
        self.alpha: list[float] = [float(config.prior_alpha)]
        self.beta: list[float] = [float(config.prior_beta)]
        self.observations = 0
        self.history: list[float] = []
        del seed_mean  # standardisation replaces the raw-scale seed

    def _standardize(self, value: float) -> float:
        """Causal z-score: only observations strictly before this one are used."""
        window = self.config.standardization_window
        history = self.history[-window:]
        if len(history) < 4:
            return 0.0
        mean = sum(history) / len(history)
        variance = sum((item - mean) ** 2 for item in history) / len(history)
        scale = math.sqrt(max(0.0, variance))
        if scale <= 0.0:
            # A perfectly constant channel: any departure is unbounded, so express
            # it as a large but finite z rather than a division by zero.
            return 0.0 if value == mean else math.copysign(10.0, value - mean)
        return (value - mean) / scale

    def update(self, raw_value: float) -> float:
        """Absorb one observation; return P(current run length <= short_run_length)."""
        cfg = self.config
        value = self._standardize(float(raw_value))
        self.history.append(float(raw_value))
        if len(self.history) > max(8, cfg.standardization_window) * 2:
            self.history = self.history[-cfg.standardization_window :]
        hazard = cfg.hazard
        # Predictives are computed in LOG space and shifted by their maximum before
        # exponentiating. A genuine regime break produces a |z| of tens, where every
        # raw density underflows to 0.0 — and a zero total used to trigger the
        # "degenerate observation" restart below, wiping out the run-length belief at
        # exactly the moment it had detected something. Log-sum-exp keeps the ratios
        # exact no matter how extreme the observation is.
        log_predictive: list[float] = []
        for index in range(len(self.run_length)):
            alpha = self.alpha[index]
            beta = self.beta[index]
            kappa = self.kappa[index]
            # Student-t predictive of the Normal-Gamma posterior.
            df = 2.0 * alpha
            scale = math.sqrt(max(1e-18, beta * (kappa + 1.0) / (alpha * max(1e-12, kappa))))
            log_predictive.append(_student_t_log_pdf(value, self.mu[index], scale, df))
        shift = max(log_predictive) if log_predictive else 0.0
        predictive = [
            math.exp(max(-700.0, item - shift)) if math.isfinite(item) else 0.0
            for item in log_predictive
        ]

        growth = [
            self.run_length[index] * predictive[index] * (1.0 - hazard)
            for index in range(len(self.run_length))
        ]
        change = sum(
            self.run_length[index] * predictive[index] * hazard
            for index in range(len(self.run_length))
        )
        new_run_length = [change, *growth]
        total = sum(new_run_length)
        if not math.isfinite(total) or total <= 0.0:
            # Numerically degenerate observation (NaN/inf leaking through despite
            # the log-space shift): restart the run-length belief rather than
            # propagate NaNs into the trading decision. A restart IS a change point.
            history = self.history[-self.config.standardization_window :]
            self.__init__(self.config)  # type: ignore[misc]
            self.history = history
            self.observations = 1
            return 1.0

        new_run_length = [item / total for item in new_run_length]

        new_mu = [self.mu[0]]
        new_kappa = [self.kappa[0]]
        new_alpha = [self.alpha[0]]
        new_beta = [self.beta[0]]
        for index in range(len(self.run_length)):
            kappa = self.kappa[index]
            mu = self.mu[index]
            new_mu.append((kappa * mu + value) / (kappa + 1.0))
            new_kappa.append(kappa + 1.0)
            new_alpha.append(self.alpha[index] + 0.5)
            new_beta.append(
                self.beta[index] + (kappa * (value - mu) ** 2) / (2.0 * (kappa + 1.0))
            )

        limit = max(2, int(cfg.max_run_length))
        self.run_length = new_run_length[:limit]
        self.mu = new_mu[:limit]
        self.kappa = new_kappa[:limit]
        self.alpha = new_alpha[:limit]
        self.beta = new_beta[:limit]
        remaining = sum(self.run_length)
        if remaining > 0:
            self.run_length = [item / remaining for item in self.run_length]
        self.observations += 1
        return self.short_run_probability()

    def short_run_probability(self) -> float:
        limit = max(1, int(self.config.short_run_length) + 1)
        return float(sum(self.run_length[:limit]))

    def as_state(self) -> dict[str, Any]:
        return {
            "run_length": list(self.run_length),
            "mu": list(self.mu),
            "kappa": list(self.kappa),
            "alpha": list(self.alpha),
            "beta": list(self.beta),
            "observations": int(self.observations),
            "history": [float(item) for item in self.history[-self.config.standardization_window :]],
        }

    @classmethod
    def from_state(cls, config: ChangePointConfig, state: Mapping[str, Any]) -> "_ChannelDetector":
        detector = cls(config)
        try:
            run_length = [float(value) for value in state.get("run_length") or ()]
            mu = [float(value) for value in state.get("mu") or ()]
            kappa = [float(value) for value in state.get("kappa") or ()]
            alpha = [float(value) for value in state.get("alpha") or ()]
            beta = [float(value) for value in state.get("beta") or ()]
        except (TypeError, ValueError):
            return detector
        sizes = {len(run_length), len(mu), len(kappa), len(alpha), len(beta)}
        if len(sizes) != 1 or not run_length:
            return detector
        if not all(math.isfinite(value) for value in (*run_length, *mu, *kappa, *alpha, *beta)):
            return detector
        detector.run_length = run_length
        detector.mu = mu
        detector.kappa = kappa
        detector.alpha = alpha
        detector.beta = beta
        try:
            detector.observations = max(0, int(state.get("observations") or 0))
        except (TypeError, ValueError):
            detector.observations = 0
        try:
            history = [float(item) for item in state.get("history") or ()]
        except (TypeError, ValueError):
            history = []
        detector.history = [item for item in history if math.isfinite(item)]
        return detector


class BayesianOnlineChangePointDetector:
    """Multi-channel BOCPD with corroboration and durable state.

    Not thread-safe by itself; the macro loop is the single writer. Callers that
    share an instance across threads must serialize :meth:`update`.
    """

    def __init__(
        self,
        config: ChangePointConfig | None = None,
        *,
        state_path: str | Path | None = DEFAULT_STATE_PATH,
    ) -> None:
        self.config = config or ChangePointConfig.from_env()
        self.state_path = Path(state_path) if state_path else None
        self._channels: dict[str, _ChannelDetector] = {}
        self._observation_count = 0
        self._last_change_at: datetime | None = None
        self._first_observed_at: datetime | None = None
        self._load()

    # -- API ---------------------------------------------------------------- #
    def update(
        self,
        channels: Mapping[str, Any],
        *,
        timestamp: datetime | None = None,
        persist: bool = True,
    ) -> ChangePointResult:
        moment = _aware(timestamp or datetime.now(timezone.utc))
        cfg = self.config
        observed: dict[str, float] = {}
        for name, raw in (channels or {}).items():
            value = _finite(raw)
            if value is None:
                continue
            observed[str(name)] = value
        if self._first_observed_at is None and observed:
            self._first_observed_at = moment

        probabilities: dict[str, float] = {}
        for name, value in sorted(observed.items()):
            detector = self._channels.get(name)
            if detector is None:
                detector = _ChannelDetector(cfg)
                self._channels[name] = detector
            probabilities[name] = detector.update(value)

        if observed:
            self._observation_count += 1

        mature = {
            name: probability
            for name, probability in probabilities.items()
            if self._channels[name].observations >= cfg.min_channel_observations
        }
        reasons: list[str] = []
        if not mature:
            combined = 0.0
            contributing: tuple[str, ...] = ()
            reasons.append(CHANGE_POINT_INSUFFICIENT_HISTORY)
        else:
            ordered = sorted(mature.items(), key=lambda item: item[1], reverse=True)
            index = min(max(1, cfg.corroborating_channels) - 1, len(ordered) - 1)
            combined = float(ordered[index][1])
            contributing = tuple(name for name, _ in ordered[: index + 1])
            if combined >= cfg.detection_threshold:
                reasons.append(CHANGE_POINT_DETECTED)
                self._last_change_at = moment
            elif combined >= cfg.suspicion_threshold:
                reasons.append(CHANGE_POINT_SUSPECTED)

        anchor = self._last_change_at or self._first_observed_at
        age_seconds = max(0.0, (moment - anchor).total_seconds()) if anchor else 0.0
        if CHANGE_POINT_DETECTED not in reasons and mature:
            if age_seconds < cfg.regime_trust_age_seconds:
                reasons.append(CHANGE_POINT_REGIME_YOUNG)
            elif not reasons:
                reasons.append(CHANGE_POINT_STABLE)
        maturity = (
            min(1.0, age_seconds / cfg.regime_trust_age_seconds)
            if cfg.regime_trust_age_seconds > 0
            else 1.0
        )
        stability = max(0.0, min(1.0, (1.0 - combined) * maturity))

        result = ChangePointResult(
            timestamp=moment,
            change_point_probability=combined,
            current_regime_age_seconds=age_seconds,
            regime_stability=stability,
            observation_count=self._observation_count,
            channel_probabilities=probabilities,
            contributing_channels=contributing,
            reason_codes=tuple(dict.fromkeys(reasons)),
            diagnostics={
                "channel_count": len(self._channels),
                "mature_channel_count": len(mature),
                "corroborating_channels": cfg.corroborating_channels,
                "detection_threshold": cfg.detection_threshold,
                "hazard": cfg.hazard,
                "last_change_at": _iso(self._last_change_at),
                "channel_observations": {
                    name: detector.observations
                    for name, detector in sorted(self._channels.items())
                },
            },
        )
        if persist:
            self._save()
        return result

    def snapshot(self) -> dict[str, Any]:
        """Read-only view without absorbing an observation."""
        return {
            "observation_count": self._observation_count,
            "last_change_at": _iso(self._last_change_at),
            "first_observed_at": _iso(self._first_observed_at),
            "channels": {
                name: detector.observations for name, detector in sorted(self._channels.items())
            },
        }

    def reset(self) -> None:
        self._channels.clear()
        self._observation_count = 0
        self._last_change_at = None
        self._first_observed_at = None
        self._save()

    # -- persistence -------------------------------------------------------- #
    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        channels = payload.get("channels")
        if isinstance(channels, dict):
            for name, state in channels.items():
                if isinstance(state, dict):
                    self._channels[str(name)] = _ChannelDetector.from_state(self.config, state)
        try:
            self._observation_count = max(0, int(payload.get("observation_count") or 0))
        except (TypeError, ValueError):
            self._observation_count = 0
        self._last_change_at = _parse_iso(payload.get("last_change_at"))
        self._first_observed_at = _parse_iso(payload.get("first_observed_at"))

    def _save(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "observation_count": self._observation_count,
            "last_change_at": _iso(self._last_change_at),
            "first_observed_at": _iso(self._first_observed_at),
            "channels": {
                name: detector.as_state() for name, detector in self._channels.items()
            },
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, self.state_path)
        except OSError:
            # Persistence is an optimisation; an unwritable disk must not stop
            # the detector from working in memory for this session.
            pass


_DEFAULT_DETECTOR: BayesianOnlineChangePointDetector | None = None


def default_detector() -> BayesianOnlineChangePointDetector:
    """Process-wide detector used by the macro loop."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = BayesianOnlineChangePointDetector(
            state_path=os.getenv("CHANGE_POINT_STATE_PATH", DEFAULT_STATE_PATH)
        )
    return _DEFAULT_DETECTOR


def reset_default_detector() -> None:
    global _DEFAULT_DETECTOR
    _DEFAULT_DETECTOR = None


def unavailable_result(timestamp: datetime | None = None) -> ChangePointResult:
    """The honest 'no detector output' value: unknown, not 'stable'."""
    return ChangePointResult(
        timestamp=_aware(timestamp or datetime.now(timezone.utc)),
        change_point_probability=0.0,
        current_regime_age_seconds=0.0,
        regime_stability=0.0,
        observation_count=0,
        channel_probabilities={},
        contributing_channels=(),
        reason_codes=(CHANGE_POINT_INSUFFICIENT_HISTORY,),
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _iso(moment: datetime | None) -> str | None:
    return _aware(moment).isoformat() if moment is not None else None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None
