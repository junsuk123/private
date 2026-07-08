"""Latest macro–micro reasoning bundle for the advisory GUI panel.

The coordinator/runtime records the most recent MacroMicroReasoningBundle (as a
plain dict) here; the account dashboard reads a snapshot. Read-only diagnostic
surface — nothing here influences a trading decision. Thread-safe.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

_LOCK = Lock()
_LATEST: dict[str, Any] | None = None


def record_bundle(bundle_dict: dict[str, Any] | None) -> None:
    global _LATEST
    with _LOCK:
        _LATEST = dict(bundle_dict) if bundle_dict else None


def snapshot() -> dict[str, Any] | None:
    with _LOCK:
        return dict(_LATEST) if _LATEST else None


def clear() -> None:
    global _LATEST
    with _LOCK:
        _LATEST = None
