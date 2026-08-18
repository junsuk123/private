"""Typed access to ``config/execution_authority.yaml``, with the invariants enforced.

Why the invariants are validated rather than applied
-----------------------------------------------------
``post_selection_profitability_veto: false`` is not a setting the code reads and obeys —
there is no post-selection ``ProfitabilityGate`` call left to switch on. Writing it in a
config file is useful (an operator can see the contract in one place) and dangerous in
exactly one way: someone flips it to ``true``, nothing happens, and the file now describes
a system that does not exist.

So the loader **refuses** a file whose invariants disagree with the architecture. The
config can document the contract; it cannot quietly contradict it.

The operational block is ordinary configuration and is applied normally.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ExecutionAuthorityConfig",
    "ExecutionAuthorityConfigError",
    "REQUIRED_INVARIANTS",
    "default_execution_authority_config",
    "load_execution_authority_config",
    "reset_execution_authority_config_cache",
]

DEFAULT_CONFIG_PATH = Path("config/execution_authority.yaml")

#: The architecture, as booleans. A config file that disagrees is rejected at load.
REQUIRED_INVARIANTS: Mapping[str, bool] = {
    "strategy_owned_execution": True,
    "post_selection_profitability_veto": False,
    "post_selection_position_resizing": False,
    "post_selection_risk_veto": False,
    "post_selection_ontology_reapproval": False,
    "execution_guard": True,
    "synthetic_live_data": False,
    "unknown_source_live": False,
    "duplicate_protection": True,
    "idempotency": True,
}


class ExecutionAuthorityConfigError(RuntimeError):
    """The file exists and contradicts the architecture, or does not parse."""


@dataclass(frozen=True)
class ExecutionAuthorityConfig:
    order_type: str = "LIMIT"
    max_buy_orders_per_cycle: int = 1
    manual_strategy_confirmation: bool = False
    trade_plan_ttl_seconds: float = 300.0
    strict_affordability: bool = True
    invariants: Mapping[str, bool] = field(
        default_factory=lambda: dict(REQUIRED_INVARIANTS)
    )
    source_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_type": self.order_type,
            "max_buy_orders_per_cycle": self.max_buy_orders_per_cycle,
            "manual_strategy_confirmation": self.manual_strategy_confirmation,
            "trade_plan_ttl_seconds": self.trade_plan_ttl_seconds,
            "strict_affordability": self.strict_affordability,
            "invariants": dict(self.invariants),
            "source_path": self.source_path,
        }


def load_execution_authority_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ExecutionAuthorityConfig:
    target = Path(path)
    if not target.exists():
        return ExecutionAuthorityConfig()
    try:
        import yaml

        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - a malformed authority file is fatal.
        raise ExecutionAuthorityConfigError(f"cannot parse {target}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ExecutionAuthorityConfigError(f"{target} must be a mapping")

    declared = raw.get("invariants") or {}
    if not isinstance(declared, Mapping):
        raise ExecutionAuthorityConfigError(f"{target}: invariants must be a mapping")
    for name, required in REQUIRED_INVARIANTS.items():
        if name not in declared:
            continue
        if bool(declared[name]) is not required:
            raise ExecutionAuthorityConfigError(
                f"{target}: {name} is an architectural invariant fixed at {required}; "
                f"the code has no path that honours {bool(declared[name])}. "
                "Change the code and this constant together, or leave the entry out."
            )
    unknown = set(declared) - set(REQUIRED_INVARIANTS)
    if unknown:
        raise ExecutionAuthorityConfigError(
            f"{target}: unknown invariants {sorted(unknown)}"
        )

    operational = raw.get("operational") or {}
    if not isinstance(operational, Mapping):
        raise ExecutionAuthorityConfigError(f"{target}: operational must be a mapping")
    order_type = str(operational.get("order_type", "LIMIT")).upper()
    if order_type != "LIMIT":
        raise ExecutionAuthorityConfigError(
            f"{target}: order_type must be LIMIT; market orders are not authorised on "
            "this account and no code path constructs one."
        )
    return ExecutionAuthorityConfig(
        order_type=order_type,
        max_buy_orders_per_cycle=max(
            1, int(operational.get("max_buy_orders_per_cycle", 1))
        ),
        manual_strategy_confirmation=bool(
            operational.get("manual_strategy_confirmation", False)
        ),
        trade_plan_ttl_seconds=float(
            operational.get("trade_plan_ttl_seconds", 300.0)
        ),
        strict_affordability=bool(operational.get("strict_affordability", True)),
        invariants=dict(REQUIRED_INVARIANTS),
        source_path=str(target),
    )


_cache: ExecutionAuthorityConfig | None = None
_lock = threading.Lock()


def default_execution_authority_config() -> ExecutionAuthorityConfig:
    global _cache
    with _lock:
        if _cache is None:
            _cache = load_execution_authority_config()
        return _cache


def reset_execution_authority_config_cache() -> None:
    """Test hook. Never called from the trading path."""
    global _cache
    with _lock:
        _cache = None
