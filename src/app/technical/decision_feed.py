"""Bounded in-memory feed of the latest per-symbol technical decision context.

The realtime trading engine records each evaluated decision here; the account
dashboard's technical panel reads a snapshot. This is a read-only diagnostic
surface — nothing here influences a trading decision. Thread-safe and bounded
so it never grows without limit.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

_LOCK = Lock()
_FEED: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_IN_PROGRESS: "OrderedDict[str, dict[str, Any]] | None" = None
_MAX_SYMBOLS = 60


def _norm_action(value: str) -> str:
    return str(value or "").split(".")[-1].upper()


def record_decision(
    symbol: str,
    action: str,
    approved: bool,
    reason_codes,
    diagnostics: dict[str, Any] | None,
) -> None:
    diagnostics = diagnostics or {}
    exit_action = diagnostics.get("exit_action")
    resolved_action = _norm_action(exit_action) if exit_action else _norm_action(action)
    payload = {
        "symbol": symbol,
        "action": resolved_action or _norm_action(action),
        "approved": bool(approved),
        "reason_codes": [str(c) for c in (reason_codes or ())],
        "technical": diagnostics.get("technical_prediction"),
        "technical_methodology": diagnostics.get("technical_methodology"),
        "technical_regime": diagnostics.get("technical_regime"),
        "technical_exit_deterioration": diagnostics.get("technical_exit_deterioration"),
        "profitability": diagnostics.get("profitability_decision"),
    }
    with _LOCK:
        target = _IN_PROGRESS if _IN_PROGRESS is not None else _FEED
        target.pop(symbol, None)
        target[symbol] = payload
        while len(target) > _MAX_SYMBOLS:
            target.popitem(last=False)


def begin_cycle() -> None:
    """Collect decisions for a new engine cycle without hiding the last complete one."""
    global _IN_PROGRESS
    with _LOCK:
        _IN_PROGRESS = OrderedDict()


def commit_cycle() -> None:
    """Publish the current engine cycle atomically to dashboard readers."""
    global _IN_PROGRESS
    with _LOCK:
        if _IN_PROGRESS is not None:
            _FEED.clear()
            _FEED.update(_IN_PROGRESS)
            _IN_PROGRESS = None


def snapshot() -> list[dict[str, Any]]:
    with _LOCK:
        return list(_FEED.values())


def clear() -> None:
    global _IN_PROGRESS
    with _LOCK:
        _FEED.clear()
        _IN_PROGRESS = None
