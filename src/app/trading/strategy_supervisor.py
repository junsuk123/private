"""Background supervisor for a running strategy.

The ontology and GNN resolve market admissibility ONCE, at election time. After
that the elected algorithm owns the trade and contains no regime, liquidity or
volatility analysis of its own. This module is what keeps watching, and it is
the only component allowed to stop a running algorithm.

Two violation tiers, deliberately separated:

``HARD``
    The premise of trading at all is gone — stale data, an untradable session,
    an unreadable broker, the daily loss budget breached, or the ontology
    withdrawing the elected strategy. New entries stop AND the open position is
    handed to the exit path immediately.

``SOFT``
    Conditions have deteriorated but the position is still governed by a valid
    thesis — macro BLOCK_BUY, thin liquidity, a widened spread, elevated
    volatility. New entries stop; the owning algorithm keeps managing its own
    target/stop/trailing/time.

The supervisor never opens a position, never picks a strategy and never prices
an order. It only reports a verdict; the caller applies it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class HaltLevel(str, Enum):
    NONE = "NONE"
    SOFT = "SOFT"
    HARD = "HARD"

    @property
    def blocks_new_entries(self) -> bool:
        return self is not HaltLevel.NONE

    @property
    def forces_exit(self) -> bool:
        return self is HaltLevel.HARD


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SupervisorConfig:
    enabled: bool = field(
        default_factory=lambda: _env_bool("STRATEGY_SUPERVISOR_ENABLED", True)
    )
    max_data_age_seconds: float = field(
        default_factory=lambda: max(1.0, _env_float("STRATEGY_SUPERVISOR_MAX_DATA_AGE_SEC", 15.0))
    )
    min_liquidity_score: float = field(
        default_factory=lambda: _env_float("STRATEGY_SUPERVISOR_MIN_LIQUIDITY", 0.25)
    )
    max_spread_bps: float = field(
        default_factory=lambda: _env_float("STRATEGY_SUPERVISOR_MAX_SPREAD_BPS", 60.0)
    )
    max_realized_volatility: float = field(
        default_factory=lambda: _env_float("STRATEGY_SUPERVISOR_MAX_VOLATILITY", 0.02)
    )
    # Fraction of the daily loss budget that still counts as SOFT before the
    # HARD breach at 1.0.
    daily_loss_soft_fraction: float = field(
        default_factory=lambda: max(
            0.0, min(1.0, _env_float("STRATEGY_SUPERVISOR_DAILY_LOSS_SOFT_FRACTION", 0.8))
        )
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_data_age_seconds": self.max_data_age_seconds,
            "min_liquidity_score": self.min_liquidity_score,
            "max_spread_bps": self.max_spread_bps,
            "max_realized_volatility": self.max_realized_volatility,
            "daily_loss_soft_fraction": self.daily_loss_soft_fraction,
        }


@dataclass(frozen=True)
class SupervisorObservation:
    """Everything the supervisor watches. ``None`` means "not observed"."""

    symbol: str
    as_of: datetime
    strategy_id: str | None = None
    position_open: bool = False
    # Data / venue health
    data_age_seconds: float | None = None
    session_tradable: bool | None = None
    broker_healthy: bool | None = None
    # Ontology withdrawal
    ontology_allows_strategy: bool | None = None
    macro_blocks_buy: bool = False
    macro_risk_level: str | None = None
    # Market condition (moved out of the algorithms)
    liquidity_score: float | None = None
    spread_bps: float | None = None
    realized_volatility: float | None = None
    # Account risk
    daily_realized_loss: float | None = None
    daily_loss_limit: float | None = None


@dataclass(frozen=True)
class SupervisorVerdict:
    level: HaltLevel
    symbol: str
    as_of: datetime
    reason_codes: tuple[str, ...] = ()
    hard_reason_codes: tuple[str, ...] = ()
    soft_reason_codes: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def blocks_new_entries(self) -> bool:
        return self.level.blocks_new_entries

    @property
    def forces_exit(self) -> bool:
        return self.level.forces_exit

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "reason_codes": list(self.reason_codes),
            "hard_reason_codes": list(self.hard_reason_codes),
            "soft_reason_codes": list(self.soft_reason_codes),
            "blocks_new_entries": self.blocks_new_entries,
            "forces_exit": self.forces_exit,
            "diagnostics": dict(self.diagnostics),
        }


class StrategySupervisor:
    """Stateless-per-call observer that grades violations into SOFT/HARD."""

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        self.config = config or SupervisorConfig()
        self._last: dict[str, SupervisorVerdict] = {}

    def evaluate(self, observation: SupervisorObservation) -> SupervisorVerdict:
        config = self.config
        hard: list[str] = []
        soft: list[str] = []

        if not config.enabled:
            verdict = SupervisorVerdict(
                level=HaltLevel.NONE,
                symbol=observation.symbol,
                as_of=observation.as_of,
                reason_codes=("SUPERVISOR_DISABLED",),
                diagnostics={"config": config.as_dict()},
            )
            self._last[observation.symbol] = verdict
            return verdict

        # ---- HARD: the premise of trading is gone ------------------------ #
        age = observation.data_age_seconds
        session_closed = observation.session_tradable is False
        if session_closed:
            # A closed session is a normal overnight state, not a violation. It
            # must not queue an exit: nothing can be sold while closed, so a
            # HARD verdict here would just dump the position at the next open.
            soft.append("SESSION_NOT_TRADABLE")
        elif age is not None and age > config.max_data_age_seconds:
            # Staleness only means something while the venue is actually
            # trading; when it is closed, stale data is expected.
            hard.append(f"DATA_STALE:{age:.1f}s>{config.max_data_age_seconds:.0f}s")
        if observation.broker_healthy is False:
            hard.append("BROKER_UNHEALTHY")

        loss, limit = observation.daily_realized_loss, observation.daily_loss_limit
        if loss is not None and limit is not None and limit > 0:
            consumed = abs(loss) / limit if loss < 0 else 0.0
            if consumed >= 1.0:
                hard.append("DAILY_LOSS_LIMIT_BREACHED")
            elif consumed >= config.daily_loss_soft_fraction:
                soft.append(f"DAILY_LOSS_BUDGET_NEAR_BREACH:{consumed:.2f}")

        # ---- SOFT: conditions deteriorated, thesis still valid ----------- #
        # A macro regime rotation withdrawing the elected strategy family is a
        # regime event, not a broken premise: it stops new entries, but forcing
        # an immediate liquidation would dump an open position on a rotation.
        if observation.strategy_id and observation.ontology_allows_strategy is False:
            soft.append(f"ONTOLOGY_WITHDREW_STRATEGY:{observation.strategy_id}")
        if observation.macro_blocks_buy:
            soft.append("MACRO_BLOCK_BUY")
        if observation.macro_risk_level and str(observation.macro_risk_level).upper() in {
            "HIGH",
            "CRITICAL",
            "EXTREME",
        }:
            soft.append(f"MACRO_RISK_{str(observation.macro_risk_level).upper()}")
        liquidity = observation.liquidity_score
        if liquidity is not None and liquidity < config.min_liquidity_score:
            soft.append(f"LIQUIDITY_DEGRADED:{liquidity:.2f}")
        spread = observation.spread_bps
        if spread is not None and spread > config.max_spread_bps:
            soft.append(f"SPREAD_WIDENED:{spread:.1f}bps")
        volatility = observation.realized_volatility
        if volatility is not None and volatility > config.max_realized_volatility:
            soft.append(f"VOLATILITY_ELEVATED:{volatility:.4f}")

        if hard:
            level = HaltLevel.HARD
        elif soft:
            level = HaltLevel.SOFT
        else:
            level = HaltLevel.NONE

        verdict = SupervisorVerdict(
            level=level,
            symbol=observation.symbol,
            as_of=observation.as_of,
            reason_codes=tuple(dict.fromkeys((*hard, *soft))),
            hard_reason_codes=tuple(dict.fromkeys(hard)),
            soft_reason_codes=tuple(dict.fromkeys(soft)),
            diagnostics={
                "position_open": observation.position_open,
                "strategy_id": observation.strategy_id,
                "data_age_seconds": age,
                "liquidity_score": liquidity,
                "spread_bps": spread,
                "realized_volatility": volatility,
            },
        )
        self._last[observation.symbol] = verdict
        return verdict

    def last_verdict(self, symbol: str) -> SupervisorVerdict | None:
        return self._last.get(symbol)

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "last_verdicts": {
                symbol: verdict.as_dict() for symbol, verdict in self._last.items()
            },
        }
