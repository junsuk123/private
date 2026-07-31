"""Durable per-strategy realized-outcome store and its posterior statistics.

Why this exists
---------------
Before this module, ``SharedLiveDecisionEngine`` passed literal constants into
the sizing / tuning layers::

    recent_performance=0.0
    recent_same_strategy_loss=False

so a strategy that had just lost five times in a row received no penalty on the
sixth selection, and a strategy that was working in the current regime received
no credit. In a fast-changing tape that is the single most expensive omission in
the selection path.

This store is the online feedback channel. It records every *closed* strategy
outcome with the regime it was taken in, and answers the three questions the
selection path actually needs:

* ``recent_net_bps`` — the realized net (after cost) series,
* ``loss_streak`` — how many consecutive losses this strategy just took,
* ``posterior`` — a shrunk mean with an explicit LOWER confidence bound, which is
  what :mod:`app.trading.conservative_bandit` selects on. Selecting on the mean
  is how a three-sample fluke becomes a live position.

Design notes
------------
* SQLite, append-only, one row per closed outcome. Small (KB/day) and durable
  across restarts, unlike an in-memory deque.
* A short in-process TTL cache, because the trading loop asks these questions
  every tick and the answers only change when a position closes.
* Posterior uses a Normal prior centred on ``prior_mean_net_bps`` (0.0 — "no edge
  until demonstrated") with ``prior_weight`` pseudo-observations, then a
  t-style lower bound. Sample counts are additionally *discounted* by the
  change-point probability, so a regime break mechanically widens the
  uncertainty penalty instead of requiring a manual history reset.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

DEFAULT_STORE_PATH = "data/store/strategy-performance.sqlite3"
NO_TRADE_ARM = "no_trade"

_SCHEMA_VERSION = 1


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


def normalize_market(market: str | None) -> str:
    name = str(market or "").strip().upper()
    if name in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        return "KR"
    if name in {"US", "USA", "NASD", "NASDAQ", "NYSE", "AMEX"}:
        return "US"
    return name or "UNKNOWN"


def market_for_symbol(symbol: str) -> str:
    """6-digit numeric -> KR, everything else -> US (matches the routing rule)."""
    text = str(symbol or "").strip().upper()
    return "KR" if text.isdigit() and len(text) == 6 else "US"


def normalize_regime(regime: str | None) -> str:
    return str(regime or "").strip().upper() or "UNKNOWN"


@dataclass(frozen=True)
class StrategyOutcome:
    """One closed strategy trade, as realized (never as expected)."""

    strategy_id: str
    symbol: str
    market: str
    regime: str
    realized_net_bps: float
    recorded_at: datetime
    realized_gross_bps: float | None = None
    expected_net_bps: float | None = None
    holding_seconds: float | None = None
    slippage_error_bps: float | None = None
    max_adverse_excursion_bps: float | None = None
    exit_reason: str = ""
    source: str = "live"

    @property
    def is_loss(self) -> bool:
        return self.realized_net_bps < 0.0


@dataclass(frozen=True)
class StrategyPosterior:
    """Shrunk posterior over one strategy's net edge, with a lower bound.

    ``conservative_edge_bps`` is the only field a selection rule should compare
    against zero: it is ``posterior_mean_net_bps - uncertainty_penalty_bps``.
    """

    strategy_id: str
    market: str
    regime: str
    sample_count: int
    effective_sample_count: float
    observed_mean_net_bps: float
    posterior_mean_net_bps: float
    standard_error_bps: float
    uncertainty_penalty_bps: float
    conservative_edge_bps: float
    win_rate: float
    loss_streak: int
    mean_slippage_error_bps: float
    mean_prediction_error_bps: float
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "market": self.market,
            "regime": self.regime,
            "sample_count": self.sample_count,
            "effective_sample_count": round(self.effective_sample_count, 3),
            "observed_mean_net_bps": round(self.observed_mean_net_bps, 3),
            "posterior_mean_net_bps": round(self.posterior_mean_net_bps, 3),
            "standard_error_bps": round(self.standard_error_bps, 3),
            "uncertainty_penalty_bps": round(self.uncertainty_penalty_bps, 3),
            "conservative_edge_bps": round(self.conservative_edge_bps, 3),
            "win_rate": round(self.win_rate, 4),
            "loss_streak": self.loss_streak,
            "mean_slippage_error_bps": round(self.mean_slippage_error_bps, 3),
            "mean_prediction_error_bps": round(self.mean_prediction_error_bps, 3),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PosteriorConfig:
    """Shrinkage / pessimism knobs for :meth:`StrategyPerformanceStore.posterior`."""

    prior_mean_net_bps: float = 0.0
    # Pseudo-observations of the prior. 8 means "eight break-even trades" must be
    # out-voted before the observed mean dominates.
    prior_weight: float = 8.0
    # Prior dispersion used when the sample is too small to estimate one.
    prior_stdev_bps: float = 60.0
    # Multiplier on the standard error. 1.64 == a one-sided 95% lower bound.
    pessimism_z: float = 1.64
    # Extra penalty while the sample is below this count, decaying linearly.
    minimum_samples: int = 12
    cold_start_penalty_bps: float = 40.0
    # Consecutive losses each add this much penalty (capped at 5 losses).
    loss_streak_penalty_bps: float = 8.0
    window: int = 60

    @classmethod
    def from_env(cls) -> "PosteriorConfig":
        return cls(
            prior_mean_net_bps=_env_float("STRATEGY_POSTERIOR_PRIOR_MEAN_BPS", cls.prior_mean_net_bps),
            prior_weight=max(0.0, _env_float("STRATEGY_POSTERIOR_PRIOR_WEIGHT", cls.prior_weight)),
            prior_stdev_bps=max(1.0, _env_float("STRATEGY_POSTERIOR_PRIOR_STDEV_BPS", cls.prior_stdev_bps)),
            pessimism_z=max(0.0, _env_float("STRATEGY_POSTERIOR_PESSIMISM_Z", cls.pessimism_z)),
            minimum_samples=max(1, _env_int("STRATEGY_POSTERIOR_MIN_SAMPLES", cls.minimum_samples)),
            cold_start_penalty_bps=max(
                0.0, _env_float("STRATEGY_POSTERIOR_COLD_START_PENALTY_BPS", cls.cold_start_penalty_bps)
            ),
            loss_streak_penalty_bps=max(
                0.0, _env_float("STRATEGY_POSTERIOR_LOSS_STREAK_PENALTY_BPS", cls.loss_streak_penalty_bps)
            ),
            window=max(5, _env_int("STRATEGY_POSTERIOR_WINDOW", cls.window)),
        )


POSTERIOR_NO_SAMPLES = "STRATEGY_POSTERIOR_NO_SAMPLES"
POSTERIOR_BELOW_MIN_SAMPLES = "STRATEGY_POSTERIOR_BELOW_MIN_SAMPLES"
POSTERIOR_LOSS_STREAK = "STRATEGY_POSTERIOR_LOSS_STREAK"
POSTERIOR_REGIME_HISTORY_DISCOUNTED = "STRATEGY_POSTERIOR_REGIME_HISTORY_DISCOUNTED"
POSTERIOR_REGIME_FALLBACK = "STRATEGY_POSTERIOR_REGIME_HISTORY_UNAVAILABLE"


class StrategyPerformanceStore:
    """Append-only realized-outcome store with cached posterior queries."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        posterior_config: PosteriorConfig | None = None,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        self.path = Path(path or os.getenv("STRATEGY_PERFORMANCE_STORE_PATH", DEFAULT_STORE_PATH))
        self.posterior_config = posterior_config or PosteriorConfig.from_env()
        self.cache_ttl_seconds = (
            float(cache_ttl_seconds)
            if cache_ttl_seconds is not None
            else max(0.0, _env_float("STRATEGY_PERFORMANCE_CACHE_TTL_SECONDS", 5.0))
        )
        self._lock = threading.RLock()
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._available = True
        self._migrate()

    # -- writes ------------------------------------------------------------- #
    def record(
        self,
        *,
        strategy_id: str,
        symbol: str,
        realized_net_bps: float,
        market: str | None = None,
        regime: str | None = None,
        realized_gross_bps: float | None = None,
        expected_net_bps: float | None = None,
        holding_seconds: float | None = None,
        slippage_error_bps: float | None = None,
        max_adverse_excursion_bps: float | None = None,
        exit_reason: str = "",
        source: str = "live",
        recorded_at: datetime | None = None,
    ) -> bool:
        """Persist one closed outcome. Returns False when the store is unusable."""
        if not self._available:
            return False
        strategy = str(strategy_id or "").strip()
        if not strategy:
            return False
        net_bps = _finite(realized_net_bps)
        if net_bps is None:
            return False
        moment = _aware(recorded_at or datetime.now(timezone.utc))
        resolved_market = normalize_market(market or market_for_symbol(symbol))
        row = (
            f"outcome-{uuid4().hex}",
            moment.isoformat(),
            strategy,
            resolved_market,
            normalize_regime(regime),
            str(symbol or "").upper(),
            float(net_bps),
            _finite(realized_gross_bps),
            _finite(expected_net_bps),
            _finite(holding_seconds),
            _finite(slippage_error_bps),
            _finite(max_adverse_excursion_bps),
            str(exit_reason or ""),
            str(source or "live"),
        )
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    """
                    insert into strategy_outcomes(
                        outcome_id, recorded_at, strategy_id, market, regime, symbol,
                        realized_net_bps, realized_gross_bps, expected_net_bps,
                        holding_seconds, slippage_error_bps, max_adverse_excursion_bps,
                        exit_reason, source
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                conn.commit()
                self._cache.clear()
        except sqlite3.Error:
            return False
        return True

    # -- reads -------------------------------------------------------------- #
    def recent_outcomes(
        self,
        strategy_id: str,
        *,
        market: str | None = None,
        regime: str | None = None,
        limit: int | None = None,
    ) -> tuple[StrategyOutcome, ...]:
        window = max(1, int(limit or self.posterior_config.window))
        key = ("outcomes", str(strategy_id), normalize_market(market) if market else None,
               normalize_regime(regime) if regime else None, window)
        cached = self._cached(key)
        if cached is not None:
            return cached
        clauses = ["strategy_id = ?"]
        params: list[Any] = [str(strategy_id)]
        if market:
            clauses.append("market = ?")
            params.append(normalize_market(market))
        if regime:
            clauses.append("regime = ?")
            params.append(normalize_regime(regime))
        params.append(window)
        sql = (
            "select recorded_at, strategy_id, market, regime, symbol, realized_net_bps, "
            "realized_gross_bps, expected_net_bps, holding_seconds, slippage_error_bps, "
            "max_adverse_excursion_bps, exit_reason, source from strategy_outcomes "
            f"where {' and '.join(clauses)} order by recorded_at desc, rowid desc limit ?"
        )
        rows: Sequence[Any] = ()
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            rows = ()
        outcomes = tuple(
            StrategyOutcome(
                recorded_at=_parse_iso(row[0]) or datetime.now(timezone.utc),
                strategy_id=str(row[1]),
                market=str(row[2]),
                regime=str(row[3]),
                symbol=str(row[4]),
                realized_net_bps=float(row[5]),
                realized_gross_bps=_finite(row[6]),
                expected_net_bps=_finite(row[7]),
                holding_seconds=_finite(row[8]),
                slippage_error_bps=_finite(row[9]),
                max_adverse_excursion_bps=_finite(row[10]),
                exit_reason=str(row[11] or ""),
                source=str(row[12] or ""),
            )
            for row in rows
        )
        self._store(key, outcomes)
        return outcomes

    def recent_net_bps(
        self,
        strategy_id: str,
        *,
        market: str | None = None,
        regime: str | None = None,
        limit: int | None = None,
    ) -> tuple[float, ...]:
        """Most recent first."""
        return tuple(
            outcome.realized_net_bps
            for outcome in self.recent_outcomes(
                strategy_id, market=market, regime=regime, limit=limit
            )
        )

    def loss_streak(
        self,
        strategy_id: str,
        *,
        market: str | None = None,
        regime: str | None = None,
    ) -> int:
        streak = 0
        for value in self.recent_net_bps(strategy_id, market=market, regime=regime):
            if value >= 0.0:
                break
            streak += 1
        return streak

    def had_recent_loss(
        self,
        strategy_id: str,
        *,
        market: str | None = None,
        regime: str | None = None,
    ) -> bool:
        """Did the last closed trade of this strategy lose money?"""
        recent = self.recent_net_bps(strategy_id, market=market, regime=regime, limit=1)
        return bool(recent) and recent[0] < 0.0

    def recent_performance_rate(
        self,
        strategy_id: str | None = None,
        *,
        market: str | None = None,
        regime: str | None = None,
        limit: int | None = None,
    ) -> float:
        """Mean recent net return as a RATE (not bps), clamped to [-1, 1].

        This is the unit ``AutoTuningEngine`` / ``MarketRegimeEstimate`` expect.
        With no history it returns 0.0 — a neutral value, which is correct here
        because "unknown" must not masquerade as either good or bad performance.
        """
        if strategy_id:
            samples = self.recent_net_bps(strategy_id, market=market, regime=regime, limit=limit)
        else:
            samples = self.recent_net_bps_all(market=market, regime=regime, limit=limit)
        if not samples:
            return 0.0
        mean_bps = sum(samples) / len(samples)
        return max(-1.0, min(1.0, mean_bps / 10_000.0))

    def recent_net_bps_all(
        self,
        *,
        market: str | None = None,
        regime: str | None = None,
        limit: int | None = None,
    ) -> tuple[float, ...]:
        window = max(1, int(limit or self.posterior_config.window))
        key = ("all", normalize_market(market) if market else None,
               normalize_regime(regime) if regime else None, window)
        cached = self._cached(key)
        if cached is not None:
            return cached
        clauses: list[str] = []
        params: list[Any] = []
        if market:
            clauses.append("market = ?")
            params.append(normalize_market(market))
        if regime:
            clauses.append("regime = ?")
            params.append(normalize_regime(regime))
        params.append(window)
        where = f"where {' and '.join(clauses)} " if clauses else ""
        sql = (
            "select realized_net_bps from strategy_outcomes "
            f"{where}order by recorded_at desc, rowid desc limit ?"
        )
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            rows = []
        values = tuple(float(row[0]) for row in rows)
        self._store(key, values)
        return values

    def posterior(
        self,
        strategy_id: str,
        *,
        market: str | None = None,
        regime: str | None = None,
        change_point_probability: float = 0.0,
        limit: int | None = None,
    ) -> StrategyPosterior:
        """Shrunk posterior with an explicit pessimistic lower bound.

        ``change_point_probability`` discounts the effective sample count: a
        regime break does not make the history wrong, it makes it *less
        relevant*, and that is exactly an uncertainty statement.
        """
        cfg = self.posterior_config
        resolved_market = normalize_market(market) if market else None
        resolved_regime = normalize_regime(regime) if regime else None
        reasons: list[str] = []

        outcomes = self.recent_outcomes(
            strategy_id, market=resolved_market, regime=resolved_regime, limit=limit
        )
        if not outcomes and resolved_regime:
            # Regime-conditioned history is the ideal; market-wide history is the
            # honest fallback, flagged so the caller can see it was widened.
            outcomes = self.recent_outcomes(
                strategy_id, market=resolved_market, regime=None, limit=limit
            )
            if outcomes:
                reasons.append(POSTERIOR_REGIME_FALLBACK)

        samples = [outcome.realized_net_bps for outcome in outcomes]
        count = len(samples)
        observed_mean = sum(samples) / count if count else 0.0
        if count >= 2:
            variance = sum((value - observed_mean) ** 2 for value in samples) / (count - 1)
            stdev = math.sqrt(max(0.0, variance))
        else:
            stdev = cfg.prior_stdev_bps
        stdev = max(1.0, stdev)

        discount = max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, float(change_point_probability)))))
        effective_count = count * discount
        if discount < 1.0 and count:
            reasons.append(POSTERIOR_REGIME_HISTORY_DISCOUNTED)

        denominator = cfg.prior_weight + effective_count
        posterior_mean = (
            (cfg.prior_weight * cfg.prior_mean_net_bps + effective_count * observed_mean)
            / denominator
            if denominator > 0
            else cfg.prior_mean_net_bps
        )
        standard_error = stdev / math.sqrt(max(1.0, effective_count))
        penalty = cfg.pessimism_z * standard_error
        if effective_count < cfg.minimum_samples:
            shortfall = (cfg.minimum_samples - effective_count) / cfg.minimum_samples
            penalty += cfg.cold_start_penalty_bps * max(0.0, min(1.0, shortfall))
            reasons.append(POSTERIOR_NO_SAMPLES if count == 0 else POSTERIOR_BELOW_MIN_SAMPLES)

        streak = 0
        for value in samples:
            if value >= 0.0:
                break
            streak += 1
        if streak:
            penalty += cfg.loss_streak_penalty_bps * min(5, streak)
            reasons.append(POSTERIOR_LOSS_STREAK)

        wins = sum(1 for value in samples if value > 0.0)
        slippage_errors = [
            outcome.slippage_error_bps
            for outcome in outcomes
            if outcome.slippage_error_bps is not None
        ]
        prediction_errors = [
            abs(outcome.realized_net_bps - outcome.expected_net_bps)
            for outcome in outcomes
            if outcome.expected_net_bps is not None
        ]
        return StrategyPosterior(
            strategy_id=str(strategy_id),
            market=resolved_market or "ALL",
            regime=resolved_regime or "ALL",
            sample_count=count,
            effective_sample_count=effective_count,
            observed_mean_net_bps=observed_mean,
            posterior_mean_net_bps=posterior_mean,
            standard_error_bps=standard_error,
            uncertainty_penalty_bps=penalty,
            conservative_edge_bps=posterior_mean - penalty,
            win_rate=wins / count if count else 0.0,
            loss_streak=streak,
            mean_slippage_error_bps=(
                sum(slippage_errors) / len(slippage_errors) if slippage_errors else 0.0
            ),
            mean_prediction_error_bps=(
                sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
            ),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    def summary(self, *, limit: int = 200) -> dict[str, Any]:
        """Dashboard view: per-strategy realized net edge, newest first."""
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    select strategy_id, market, count(*), avg(realized_net_bps),
                           sum(case when realized_net_bps > 0 then 1 else 0 end),
                           max(recorded_at)
                    from strategy_outcomes
                    group by strategy_id, market
                    order by max(recorded_at) desc
                    limit ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
        except sqlite3.Error:
            rows = []
        return {
            "store_path": str(self.path),
            "available": self._available,
            "strategies": [
                {
                    "strategy_id": str(row[0]),
                    "market": str(row[1]),
                    "sample_count": int(row[2] or 0),
                    "mean_net_bps": round(float(row[3] or 0.0), 3),
                    "win_rate": round(float(row[4] or 0) / max(1, int(row[2] or 1)), 4),
                    "last_recorded_at": row[5],
                }
                for row in rows
            ],
        }

    def prune(self, *, keep_rows: int = 20_000) -> int:
        """Bound the store. Returns the number of deleted rows."""
        try:
            with self._lock, closing(self._connect()) as conn:
                cursor = conn.execute(
                    """
                    delete from strategy_outcomes where rowid not in (
                        select rowid from strategy_outcomes
                        order by recorded_at desc, rowid desc limit ?
                    )
                    """,
                    (max(100, int(keep_rows)),),
                )
                conn.commit()
                self._cache.clear()
                return int(cursor.rowcount or 0)
        except sqlite3.Error:
            return 0

    # -- internals ---------------------------------------------------------- #
    def _cached(self, key: tuple[Any, ...]) -> Any:
        if self.cache_ttl_seconds <= 0:
            return None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            return value

    def _store(self, key: tuple[Any, ...], value: Any) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        with self._lock:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = (time.monotonic(), value)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("pragma journal_mode = wal")
        return conn

    def _migrate(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, closing(self._connect()) as conn:
                conn.executescript(
                    """
                    create table if not exists strategy_outcomes (
                        outcome_id text primary key,
                        recorded_at text not null,
                        strategy_id text not null,
                        market text not null,
                        regime text not null,
                        symbol text not null,
                        realized_net_bps real not null,
                        realized_gross_bps real,
                        expected_net_bps real,
                        holding_seconds real,
                        slippage_error_bps real,
                        max_adverse_excursion_bps real,
                        exit_reason text,
                        source text
                    );
                    create index if not exists idx_outcomes_strategy
                        on strategy_outcomes(strategy_id, market, regime, recorded_at desc);
                    create index if not exists idx_outcomes_recorded
                        on strategy_outcomes(recorded_at desc);
                    create table if not exists schema_version (
                        version integer primary key
                    );
                    """
                )
                conn.execute(
                    "insert or ignore into schema_version(version) values (?)", (_SCHEMA_VERSION,)
                )
                conn.commit()
        except (OSError, sqlite3.Error):
            # A read-only or missing volume must not stop trading; every read
            # then returns empty and every posterior falls back to its prior,
            # which is the conservative direction.
            self._available = False


_DEFAULT_STORE: StrategyPerformanceStore | None = None
_DEFAULT_STORE_LOCK = threading.Lock()


def default_store() -> StrategyPerformanceStore:
    """Process-wide store shared by the decision engine and the session."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _DEFAULT_STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = StrategyPerformanceStore()
    return _DEFAULT_STORE


def reset_default_store() -> None:
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None
