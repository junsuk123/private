"""Market-adaptive, outcome-learning entry thresholds.

The defect this addresses
-------------------------
Every trigger constant in ``strategy_algorithms`` is an absolute number in bps.
``liquidity_shock_reversal`` fires on ``return_10s <= -40bps`` and predicts an
edge of ``0.4 * |shock|``, so at the threshold it claims 16bps against a round
trip that the tape measures at 58-74bps on US and 48-53bps on KR. It cannot pay
for itself at the point it chooses to fire, and the measured consequence is a
shadow population averaging -110bps net over 1,383 resolved round trips.

Two things are wrong with a constant, and they need different fixes.

1. A CONSTANT IS THE WRONG UNIT (fixed here, no labels required)

   -40bps is a different event in a name whose 10-second volatility is 5bps than
   in one where it is 50bps: the first is a dislocation, the second is noise. The
   measured reversion is monotone in shock size *relative to the name's own
   scale*, so the threshold belongs in units of that scale, not in absolute bps.
   ``MarketScale`` restates each constant as "the static value, rescaled by how
   the current scale compares to its own running reference". At typical
   conditions the multiplier is 1.0 and behaviour is exactly what it is today,
   which is the property that makes this a change of units rather than a new
   fitted parameter.

2. THE RIGHT LEVEL IS NOT KNOWN A PRIORI (learned here, from realized outcomes)

   Searching a grid of (shock x spread x horizon) against this tape produced
   +84bps in sample and -36bps out of sample: fitting three thresholds to eleven
   days of 186 symbols recovers noise. So this module does NOT fit a threshold.
   It adapts ONE bounded scalar per arm -- how many multiples of the round-trip
   cost an edge must clear before the arm may fire -- by stochastic approximation
   against the arm's own realized net outcomes. One monotone control variable
   with a hard floor cannot reproduce that overfit: there is no combination to
   select, only a level to walk, and every level it can reach is one the cost
   arithmetic already permits.

Learning signal
---------------
Self-supervised in the strict sense: the system acts, observes the realized net
return of what it did, and uses that as the label. No annotation exists or is
needed. ``ShadowEvaluationService`` already resolves every plan into a signed
net return, including for plans the entry gate refused, so both the accepted and
the rejected region are measurable -- which is what makes "should this threshold
move" answerable rather than a guess.

Safety invariants, enforced in code rather than described in a comment
----------------------------------------------------------------------
* **Never below the cost floor.** ``required_edge >= round_trip_cost * 1.0``
  always. The controller may only raise the multiple above 1.0, never below it.
  This is arithmetic, not policy: an edge under its own round trip is a loss with
  extra steps.
* **Never looser than the static configuration.** The adapted threshold is
  clamped to be at least as strict as the value in ``AlgorithmConfig``. The worst
  case this module can produce is today's behaviour; it cannot invent risk that
  the static configuration did not already permit.
* **Evidence before movement.** Below ``minimum_samples`` resolved outcomes the
  multiplier does not move at all, so a cold arm behaves exactly as configured.
* **Bounded.** The multiple is clamped to ``[1.0, maximum_multiple]`` so a run of
  losses cannot drive an arm to a threshold it can never satisfy, which would be
  an absorbing state of the kind ``strategy_performance_store`` documents.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

#: Where learned policy lives. Overridable because the process-wide instance is
#: reached through a module-level singleton, so anything importing the algorithms
#: -- a test run, a backtest, a second tool -- would otherwise read AND write the
#: live trading policy. A test suite silently mutating the thresholds a funded
#: account trades on is not a hypothetical: ``observe_scale`` writes, and reading
#: alone already changed a signal-engine test's outcome, because the calibrator
#: had learned enough from the live tape to refuse the very trade the test
#: asserted. Point this at a scratch path for anything that is not the live run.
_FALLBACK_STORE_PATH = Path("data/store/adaptive-thresholds.sqlite3")


def default_store_path() -> Path:
    """Resolved per call, not per import.

    A module-level constant would freeze the path at import time, and the
    processes that most need to redirect it -- a test session, a backtest -- set
    the variable after importing the algorithms, not before.
    """
    return Path(os.getenv("ADAPTIVE_THRESHOLDS_STORE_PATH") or _FALLBACK_STORE_PATH)

#: Below this many resolved outcomes an arm keeps the configured threshold.
#: Twelve matches ``StrategyPosteriorConfig.minimum_samples`` on purpose: the two
#: layers should not disagree about when an arm has said anything.
DEFAULT_MINIMUM_SAMPLES = 12

#: Hard ceiling on the cost multiple. At 4x a 74bps US round trip the arm needs a
#: ~300bps edge, which the excursion measurements put at the top of what the tape
#: offers at any horizon -- past this the arm is switched off in all but name, and
#: switching an arm off is a decision that belongs to an operator.
DEFAULT_MAXIMUM_MULTIPLE = 4.0

#: Stochastic-approximation step. Deliberately small: each resolved outcome is one
#: noisy sample of a distribution whose standard deviation the tape puts near
#: 150bps, so a step large enough to react to one trade is large enough to chase
#: noise. At 0.05 an arm needs a sustained run in one direction to move a full
#: multiple, which is the intended time constant.
DEFAULT_STEP = 0.05

#: Half-life, in resolved outcomes, of the running scale reference. Long enough to
#: span a session so an opening burst does not redefine "typical", short enough to
#: follow a genuine volatility regime change within a day or two.
DEFAULT_SCALE_HALF_LIFE = 200.0

#: Scale observations to coalesce before touching disk. The EWMA is cheap to
#: re-warm and expensive to persist at feature-evaluation rate.
_SCALE_PERSIST_EVERY = 250


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AdaptiveConfig:
    enabled: bool = True
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES
    maximum_multiple: float = DEFAULT_MAXIMUM_MULTIPLE
    step: float = DEFAULT_STEP
    scale_half_life: float = DEFAULT_SCALE_HALF_LIFE
    #: Clamp on the market-scale multiplier. A name that momentarily prints a
    #: volatility ten times its reference is far more likely to be a bad tick than
    #: a tradeable regime, and letting that set the threshold would hand the
    #: trigger to the worst data in the feed.
    minimum_scale_multiplier: float = 0.5
    maximum_scale_multiplier: float = 3.0

    @classmethod
    def from_env(cls) -> "AdaptiveConfig":
        return cls(
            enabled=_env_bool("ADAPTIVE_THRESHOLDS_ENABLED", True),
            minimum_samples=int(_env_float("ADAPTIVE_MINIMUM_SAMPLES", DEFAULT_MINIMUM_SAMPLES)),
            maximum_multiple=_env_float("ADAPTIVE_MAXIMUM_MULTIPLE", DEFAULT_MAXIMUM_MULTIPLE),
            step=_env_float("ADAPTIVE_STEP", DEFAULT_STEP),
            scale_half_life=_env_float("ADAPTIVE_SCALE_HALF_LIFE", DEFAULT_SCALE_HALF_LIFE),
            minimum_scale_multiplier=_env_float("ADAPTIVE_MIN_SCALE_MULTIPLIER", 0.5),
            maximum_scale_multiplier=_env_float("ADAPTIVE_MAX_SCALE_MULTIPLIER", 3.0),
        )


@dataclass
class ArmState:
    """What has been learned about one (strategy, market) arm."""

    strategy_id: str
    market: str
    #: Multiples of round-trip cost an edge must clear. 1.0 == the cost floor
    #: itself, which is the configured default and the permanent lower bound.
    cost_multiple: float = 1.0
    #: EWMA of the scale variable, in bps. ``None`` until the first observation.
    scale_reference_bps: float | None = None
    sample_count: int = 0
    positive_count: int = 0
    mean_net_bps: float = 0.0
    updated_at: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "market": self.market,
            "cost_multiple": round(self.cost_multiple, 4),
            "scale_reference_bps": (
                round(self.scale_reference_bps, 4)
                if self.scale_reference_bps is not None
                else None
            ),
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "mean_net_bps": round(self.mean_net_bps, 3),
            "win_rate": (
                round(self.positive_count / self.sample_count, 4)
                if self.sample_count
                else None
            ),
            "updated_at": self.updated_at,
        }


class AdaptiveThresholdStore:
    """Persistence for :class:`ArmState`.

    Separate from ``strategy_performance_store`` on purpose. That store answers
    "how did this arm do", which the bandit reads to decide whether to trade at
    all. This one answers "how strict should this arm be to fire", which the
    algorithm layer reads before an edge even exists. Sharing a table would tie a
    schema change in one decision to the other.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15.0)
        conn.execute("pragma busy_timeout = 15000")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                create table if not exists adaptive_arm_state(
                  strategy_id text not null,
                  market text not null,
                  cost_multiple real not null default 1.0,
                  scale_reference_bps real,
                  sample_count integer not null default 0,
                  positive_count integer not null default 0,
                  mean_net_bps real not null default 0.0,
                  updated_at text not null default '',
                  primary key (strategy_id, market)
                )
                """
            )
            conn.commit()

    def load(self, strategy_id: str, market: str) -> ArmState:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "select cost_multiple, scale_reference_bps, sample_count, "
                "positive_count, mean_net_bps, updated_at from adaptive_arm_state "
                "where strategy_id = ? and market = ?",
                (strategy_id, market),
            ).fetchone()
        if not row:
            return ArmState(strategy_id=strategy_id, market=market)
        return ArmState(
            strategy_id=strategy_id,
            market=market,
            cost_multiple=float(row[0]),
            scale_reference_bps=float(row[1]) if row[1] is not None else None,
            sample_count=int(row[2]),
            positive_count=int(row[3]),
            mean_net_bps=float(row[4]),
            updated_at=str(row[5] or ""),
        )

    def save(self, state: ArmState) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                insert into adaptive_arm_state(
                  strategy_id, market, cost_multiple, scale_reference_bps,
                  sample_count, positive_count, mean_net_bps, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(strategy_id, market) do update set
                  cost_multiple=excluded.cost_multiple,
                  scale_reference_bps=excluded.scale_reference_bps,
                  sample_count=excluded.sample_count,
                  positive_count=excluded.positive_count,
                  mean_net_bps=excluded.mean_net_bps,
                  updated_at=excluded.updated_at
                """,
                (
                    state.strategy_id,
                    state.market,
                    float(state.cost_multiple),
                    state.scale_reference_bps,
                    int(state.sample_count),
                    int(state.positive_count),
                    float(state.mean_net_bps),
                    state.updated_at,
                ),
            )
            conn.commit()

    # -- calibration ---------------------------------------------------------- #
    def ensure_calibration_schema(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                create table if not exists edge_calibration(
                  strategy_id text not null,
                  market text not null,
                  bucket text not null,
                  sample_count integer not null default 0,
                  mean_gross_bps real not null default 0.0,
                  m2 real not null default 0.0,
                  updated_at text not null default '',
                  primary key (strategy_id, market, bucket)
                )
                """
            )
            conn.commit()

    def load_calibration(self, strategy_id: str, market: str, bucket: str):
        from app.technical.adaptive_thresholds import CalibrationCell

        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "select sample_count, mean_gross_bps, m2 from edge_calibration "
                "where strategy_id = ? and market = ? and bucket = ?",
                (strategy_id, market, bucket),
            ).fetchone()
        if not row:
            return CalibrationCell(strategy_id=strategy_id, market=market, bucket=bucket)
        return CalibrationCell(
            strategy_id=strategy_id, market=market, bucket=bucket,
            sample_count=int(row[0]), mean_gross_bps=float(row[1]), m2=float(row[2]),
        )

    def save_calibration(self, cell) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                insert into edge_calibration(
                  strategy_id, market, bucket, sample_count, mean_gross_bps, m2, updated_at)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(strategy_id, market, bucket) do update set
                  sample_count=excluded.sample_count,
                  mean_gross_bps=excluded.mean_gross_bps,
                  m2=excluded.m2,
                  updated_at=excluded.updated_at
                """,
                (
                    cell.strategy_id, cell.market, cell.bucket,
                    int(cell.sample_count), float(cell.mean_gross_bps), float(cell.m2),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def all_calibration(self) -> list:
        from app.technical.adaptive_thresholds import CalibrationCell

        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "select strategy_id, market, bucket, sample_count, mean_gross_bps, m2 "
                "from edge_calibration order by strategy_id, market, bucket"
            ).fetchall()
        return [
            CalibrationCell(
                strategy_id=str(r[0]), market=str(r[1]), bucket=str(r[2]),
                sample_count=int(r[3]), mean_gross_bps=float(r[4]), m2=float(r[5]),
            )
            for r in rows
        ]

    def all_states(self) -> list[ArmState]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "select strategy_id, market, cost_multiple, scale_reference_bps, "
                "sample_count, positive_count, mean_net_bps, updated_at "
                "from adaptive_arm_state order by strategy_id, market"
            ).fetchall()
        return [
            ArmState(
                strategy_id=str(r[0]),
                market=str(r[1]),
                cost_multiple=float(r[2]),
                scale_reference_bps=float(r[3]) if r[3] is not None else None,
                sample_count=int(r[4]),
                positive_count=int(r[5]),
                mean_net_bps=float(r[6]),
                updated_at=str(r[7] or ""),
            )
            for r in rows
        ]


class AdaptiveThresholds:
    """The layer the algorithms consult instead of a bare constant.

    Stateless with respect to any one decision: everything it remembers lives in
    :class:`AdaptiveThresholdStore`, so two processes reading the same store see
    the same policy and a restart does not discard what was learned.
    """

    def __init__(
        self,
        store: AdaptiveThresholdStore | None = None,
        config: AdaptiveConfig | None = None,
    ) -> None:
        self.config = config or AdaptiveConfig.from_env()
        self.store = store or AdaptiveThresholdStore()
        self._cache: dict[tuple[str, str], ArmState] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._scale_writes_pending = 0
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------- #
    def state(self, strategy_id: str, market: str) -> ArmState:
        key = (strategy_id, market)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        loaded = self.store.load(strategy_id, market)
        with self._lock:
            self._cache[key] = loaded
        return loaded

    def _persist(self, state: ArmState, *, durable: bool = True) -> None:
        """Update the in-memory policy, and write it when the write is worth it.

        ``durable=False`` is for the scale EWMA, which is folded on every feature
        evaluation -- every symbol, every strategy, every few seconds. Writing
        SQLite on each of those put hundreds of transactions per second on the
        same disk the live tick loop and the training row store are using, to
        persist a running average whose loss on restart costs one half-life of
        re-warming and nothing else. The controller's own state, which encodes
        realized outcomes that cannot be recovered by observing, is always
        durable.
        """
        state.updated_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._cache[(state.strategy_id, state.market)] = state
            if not durable:
                self._dirty.add((state.strategy_id, state.market))
                due = self._scale_writes_pending >= _SCALE_PERSIST_EVERY
                if due:
                    self._scale_writes_pending = 0
                else:
                    self._scale_writes_pending += 1
                if not due:
                    return
                pending = [self._cache[key] for key in self._dirty if key in self._cache]
                self._dirty.clear()
            else:
                self._dirty.discard((state.strategy_id, state.market))
                pending = [state]
        for item in pending:
            self.store.save(item)

    # -- layer 1: market-scale normalization (unsupervised) ------------------ #
    def observe_scale(self, strategy_id: str, market: str, scale_bps: float | None) -> None:
        """Fold one observation of the current market scale into the reference.

        EWMA rather than a window, because a window needs a buffer per arm and the
        reference only has to answer "is now unusual relative to lately".
        """
        if scale_bps is None or not math.isfinite(scale_bps) or scale_bps <= 0:
            return
        state = self.state(strategy_id, market)
        if state.scale_reference_bps is None:
            state.scale_reference_bps = float(scale_bps)
        else:
            alpha = 1.0 - math.exp(-math.log(2.0) / max(1.0, self.config.scale_half_life))
            state.scale_reference_bps = (
                (1.0 - alpha) * state.scale_reference_bps + alpha * float(scale_bps)
            )
        self._persist(state, durable=False)

    def scale_multiplier(
        self, strategy_id: str, market: str, scale_bps: float | None
    ) -> tuple[float, dict[str, object]]:
        """How much wider than usual the market currently is, clamped.

        1.0 means "typical for this arm", which is the value that leaves every
        configured constant exactly where it is.
        """
        state = self.state(strategy_id, market)
        reference = state.scale_reference_bps
        if (
            not self.config.enabled
            or scale_bps is None
            or reference is None
            or reference <= 0
            or not math.isfinite(scale_bps)
            or scale_bps <= 0
        ):
            return 1.0, {"scale_basis": "unavailable"}
        raw = float(scale_bps) / float(reference)
        clamped = max(
            self.config.minimum_scale_multiplier,
            min(self.config.maximum_scale_multiplier, raw),
        )
        return clamped, {
            "scale_basis": "realized_volatility_ewma",
            "scale_bps": round(float(scale_bps), 3),
            "scale_reference_bps": round(float(reference), 3),
            "scale_multiplier_raw": round(raw, 4),
            "scale_multiplier": round(clamped, 4),
        }

    def adapt_threshold(
        self,
        strategy_id: str,
        market: str,
        *,
        static_value: float,
        scale_bps: float | None,
        stricter_is_larger: bool,
    ) -> tuple[float, dict[str, object]]:
        """Restate one configured constant in units of the current market scale.

        ``stricter_is_larger`` says which direction is conservative for this
        particular constant, because the sign convention is not uniform: a shock
        threshold of ``-40`` gets STRICTER as it goes more negative, while a
        minimum-confidence threshold gets stricter as it goes up. Clamping "toward
        strict" without being told which way that is would loosen half the
        constants in the table.
        """
        multiplier, diagnostics = self.scale_multiplier(strategy_id, market, scale_bps)
        scaled = float(static_value) * multiplier
        # Never looser than configured. With the market quieter than its reference
        # the multiplier is below 1.0, which would pull a threshold toward zero and
        # admit trades the static policy refuses; that is the one direction this
        # layer is not allowed to move.
        if stricter_is_larger:
            adapted = max(float(static_value), scaled)
        else:
            adapted = min(float(static_value), scaled)
        diagnostics.update(
            {
                "static_value": float(static_value),
                "adapted_value": round(adapted, 4),
                "clamped_to_static": adapted == float(static_value) and multiplier != 1.0,
            }
        )
        return adapted, diagnostics

    # -- layer 2: outcome-driven strictness (self-supervised) ---------------- #
    def cost_multiple(self, strategy_id: str, market: str) -> tuple[float, dict[str, object]]:
        """Multiples of round-trip cost this arm must currently clear."""
        state = self.state(strategy_id, market)
        if not self.config.enabled or state.sample_count < self.config.minimum_samples:
            return 1.0, {
                "cost_multiple": 1.0,
                "cost_multiple_basis": "insufficient_evidence",
                "sample_count": state.sample_count,
                "minimum_samples": self.config.minimum_samples,
            }
        return state.cost_multiple, {
            "cost_multiple": round(state.cost_multiple, 4),
            "cost_multiple_basis": "learned_from_realized_outcomes",
            "sample_count": state.sample_count,
            "mean_net_bps": round(state.mean_net_bps, 3),
            "win_rate": round(state.positive_count / max(1, state.sample_count), 4),
        }

    def record_outcome(
        self,
        strategy_id: str,
        market: str,
        *,
        realized_net_bps: float,
        admissible: bool = True,
    ) -> ArmState:
        """Fold one resolved round trip into the arm's strictness.

        Only admissible outcomes move the control variable. An inadmissible plan
        is one the entry gate already refused, so its result measures the rejected
        region -- informative about where the boundary should be, but not evidence
        about trades this arm would place, and treating it as such is the mistake
        that let 1,315 refused plans set the bandit's posterior.

        The update is stochastic approximation on the sign of the realized net:
        losses push the required multiple up, wins let it come back down, and the
        floor at 1.0 means "down" can never reach a level the cost arithmetic
        forbids. Sign rather than magnitude because the magnitude distribution has
        a standard deviation near 150bps -- scaling the step by it would let one
        tail event move the threshold further than fifty ordinary trades.
        """
        state = self.state(strategy_id, market)
        if not math.isfinite(realized_net_bps):
            return state
        state.sample_count += 1
        if realized_net_bps > 0:
            state.positive_count += 1
        # Running mean, so the reported figure does not need the whole history.
        state.mean_net_bps += (realized_net_bps - state.mean_net_bps) / state.sample_count

        if admissible and state.sample_count >= self.config.minimum_samples:
            direction = -1.0 if realized_net_bps > 0 else 1.0
            state.cost_multiple = max(
                1.0,
                min(
                    self.config.maximum_multiple,
                    state.cost_multiple + direction * self.config.step,
                ),
            )
        self._persist(state)
        return state

    # -- reporting ----------------------------------------------------------- #
    def snapshot(self) -> dict[str, object]:
        """Everything learned so far, for the dashboard and for an audit."""
        states = self.store.all_states()
        return {
            "enabled": self.config.enabled,
            "minimum_samples": self.config.minimum_samples,
            "maximum_multiple": self.config.maximum_multiple,
            "step": self.config.step,
            "arms": [state.as_dict() for state in states],
        }


_DEFAULT_INSTANCE: AdaptiveThresholds | None = None
# ``default_edge_calibrator`` is built from ``default_adaptive_thresholds().store``.
# Both factories use this guard, so a plain Lock deadlocks on the first calibrator
# access in a fresh process.  The live server happened to avoid it only when another
# worker had already initialised the threshold singleton.  Factory order must not
# determine whether the strategy loop can run.
_DEFAULT_LOCK = threading.RLock()


def default_adaptive_thresholds() -> AdaptiveThresholds:
    """Process-wide instance, so every algorithm reads one policy."""
    global _DEFAULT_INSTANCE
    if _DEFAULT_INSTANCE is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_INSTANCE is None:
                _DEFAULT_INSTANCE = AdaptiveThresholds()
    return _DEFAULT_INSTANCE


def reset_default_adaptive_thresholds() -> None:
    """Drop the singletons so the next call re-reads the configured path.

    Needed because the instance caches both the store handle and the learned
    state in memory: redirecting the path without this would leave a process
    reading the policy it loaded from wherever it looked first.
    """
    global _DEFAULT_INSTANCE, _DEFAULT_CALIBRATOR
    with _DEFAULT_LOCK:
        _DEFAULT_INSTANCE = None
        _DEFAULT_CALIBRATOR = None


def resolve_market(symbol: str) -> str:
    """Same 6-digit rule the cost floor and the counterfactual evaluator use."""
    normalized = str(symbol or "").upper().strip()
    return "KR" if normalized.isdigit() and len(normalized) == 6 else "US"


# --------------------------------------------------------------------------- #
# Edge calibration                                                             #
# --------------------------------------------------------------------------- #
#: Buckets over the PREDICTED edge, in bps. Calibration is per bucket because the
#: error is not a constant offset: the rule is roughly right about small moves and
#: badly wrong about large ones, so a single scalar correction would overcorrect
#: one end while undercorrecting the other.
_EDGE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 25.0, "0-25"),
    (25.0, 50.0, "25-50"),
    (50.0, 100.0, "50-100"),
    (100.0, 200.0, "100-200"),
    (200.0, float("inf"), "200+"),
)


def edge_bucket(predicted_edge_bps: float) -> str:
    value = abs(float(predicted_edge_bps))
    for low, high, name in _EDGE_BUCKETS:
        if low <= value < high:
            return name
    return _EDGE_BUCKETS[-1][2]


@dataclass
class CalibrationCell:
    strategy_id: str
    market: str
    bucket: str
    sample_count: int = 0
    mean_gross_bps: float = 0.0
    m2: float = 0.0  # Welford, for a variance without keeping the samples.

    @property
    def stdev_bps(self) -> float:
        if self.sample_count < 2:
            return 0.0
        return math.sqrt(max(0.0, self.m2 / (self.sample_count - 1)))

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "market": self.market,
            "bucket": self.bucket,
            "sample_count": self.sample_count,
            "mean_gross_bps": round(self.mean_gross_bps, 3),
            "stdev_bps": round(self.stdev_bps, 3),
        }


class EdgeCalibrator:
    """Corrects a rule's predicted edge against what that prediction actually paid.

    What is being fixed
    -------------------
    Every algorithm predicts its edge the same way: ``|displacement| * fraction``,
    with the fraction a hand-set constant (``retrace_fraction`` 0.4,
    ``target_capture_fraction``, ...). It is an assumption about how much of an
    observed move is subsequently captured, and on this tape it is wrong in the
    direction that costs money -- ``liquidity_shock_reversal`` predicted +16bps
    gross at its own trigger and realized -32bps gross, and across the arms the
    gap between the predicted net and the realized net measured about 160bps.

    Rather than re-fitting each constant, this learns the mapping
    ``predicted edge -> realized gross`` per (strategy, market, bucket) from the
    shadow tape and applies it before the cost floor sees the number. Every
    algorithm is corrected by the same mechanism, including ones added later.

    Why an empirical-Bayes mean and not a model
    -------------------------------------------
    Eleven days over 186 symbols. A three-parameter grid search on this same tape
    returned +84bps in sample and -36bps out of sample, so capacity is the enemy
    here. A shrunk bucket mean has one effective parameter per bucket and reverts
    to the rule's own prediction exactly when the evidence is thin, which is the
    behaviour that makes it safe to switch on before it has learned anything.

    Why it can only ever LOWER the edge
    -----------------------------------
    The correction is ``min(rule_prediction, lower_confidence_bound)``. Allowing
    it to raise an edge would let a thin, lucky bucket manufacture optimism and
    push a trade through the cost floor -- the failure mode this system already
    demonstrated once, when refused plans set the bandit's posterior. A genuinely
    good arm does not need the help: its raw prediction already clears the floor.
    """

    def __init__(
        self,
        store: "AdaptiveThresholdStore",
        *,
        prior_weight: float = 12.0,
        pessimism_z: float = 1.0,
        minimum_samples: int = 8,
    ) -> None:
        self.store = store
        self.prior_weight = max(0.0, prior_weight)
        self.pessimism_z = max(0.0, pessimism_z)
        self.minimum_samples = max(1, minimum_samples)
        self._cache: dict[tuple[str, str, str], CalibrationCell] = {}
        self._lock = threading.Lock()
        self.store.ensure_calibration_schema()

    def _cell(self, strategy_id: str, market: str, bucket: str) -> CalibrationCell:
        key = (strategy_id, market, bucket)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        loaded = self.store.load_calibration(strategy_id, market, bucket)
        with self._lock:
            self._cache[key] = loaded
        return loaded

    def calibrate(
        self, strategy_id: str, market: str, predicted_edge_bps: float
    ) -> tuple[float, dict[str, object]]:
        """The edge this prediction has historically been worth, never more."""
        raw = float(predicted_edge_bps)
        bucket = edge_bucket(raw)
        cell = self._cell(strategy_id, market, bucket)
        if cell.sample_count < self.minimum_samples:
            return raw, {
                "edge_calibration": "insufficient_evidence",
                "edge_bucket": bucket,
                "calibration_samples": cell.sample_count,
            }
        # Shrink the measured mean toward the rule's own claim, then take a lower
        # confidence bound on the result. Thin evidence therefore moves the number
        # very little, and noisy evidence moves it less than clean evidence does.
        n = float(cell.sample_count)
        shrunk = (self.prior_weight * raw + n * cell.mean_gross_bps) / (self.prior_weight + n)
        stderr = cell.stdev_bps / math.sqrt(n) if n > 0 else 0.0
        lower_bound = shrunk - self.pessimism_z * stderr
        calibrated = min(raw, lower_bound)
        return calibrated, {
            "edge_calibration": "measured",
            "edge_bucket": bucket,
            "calibration_samples": cell.sample_count,
            "raw_edge_bps": round(raw, 3),
            "measured_mean_gross_bps": round(cell.mean_gross_bps, 3),
            "shrunk_edge_bps": round(shrunk, 3),
            "calibrated_edge_bps": round(calibrated, 3),
        }

    def record(
        self,
        strategy_id: str,
        market: str,
        *,
        predicted_edge_bps: float,
        realized_gross_bps: float,
    ) -> None:
        """One resolved round trip, filed under the edge that was claimed for it."""
        if not math.isfinite(predicted_edge_bps) or not math.isfinite(realized_gross_bps):
            return
        bucket = edge_bucket(predicted_edge_bps)
        cell = self._cell(strategy_id, market, bucket)
        cell.sample_count += 1
        delta = realized_gross_bps - cell.mean_gross_bps
        cell.mean_gross_bps += delta / cell.sample_count
        cell.m2 += delta * (realized_gross_bps - cell.mean_gross_bps)
        with self._lock:
            self._cache[(strategy_id, market, bucket)] = cell
        self.store.save_calibration(cell)

    def snapshot(self) -> list[dict[str, object]]:
        return [cell.as_dict() for cell in self.store.all_calibration()]


_DEFAULT_CALIBRATOR: EdgeCalibrator | None = None


def default_edge_calibrator() -> EdgeCalibrator:
    """Process-wide calibrator sharing the adaptive store."""
    global _DEFAULT_CALIBRATOR
    if _DEFAULT_CALIBRATOR is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_CALIBRATOR is None:
                _DEFAULT_CALIBRATOR = EdgeCalibrator(default_adaptive_thresholds().store)
    return _DEFAULT_CALIBRATOR
