"""Capture actual live graph inputs before forward policy outcomes exist.

The canonical raw vector has 49 fields. The model's 72 columns include 23
strategy identity columns; those identities must never be stored as raw context.
This module has no store, model, broker, filesystem or historical reconstruction
dependency. The journal writer receives a detached snapshot, not a mutable cache.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Callable, Mapping

from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_DIM, STRATEGY_GRAPH_CONTEXT_SCHEMA, context_index,
)

SOURCE = "live_strategy_graph_context"
MAX_CONTEXT_AGE_SECONDS = 5.0
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_FINGERPRINT_FIELDS = (
    "schema", "feature_schema_name", "symbol", "market", "as_of",
    "feature_snapshot_as_of", "valid_until", "recorded_at", "features", "source",
    "source_snapshot_id", "source_feature_snapshot_id", "source_provenance",
    "data_fresh", "tradable",
)


def _get(value: Any, field: str, default: Any = None) -> Any:
    return value.get(field, default) if isinstance(value, Mapping) else getattr(value, field, default)


def _aware(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _market_for_symbol(symbol: str) -> str:
    # Keep the exact market classifier used by graph training/serving. Newly
    # assigned alphanumeric KRX codes are unsupported here until that entire
    # pipeline changes together; _supported_symbol rejects them, never US-tags.
    return "KR" if len(symbol) == 6 and symbol.isdigit() else "US"


def _supported_symbol(symbol: str) -> bool:
    return bool(_SYMBOL.fullmatch(symbol)) and not (
        len(symbol) == 6 and symbol[0].isdigit() and not symbol.isdigit()
    )


def _vector(values: Any) -> tuple[float, ...] | None:
    if not isinstance(values, (tuple, list)) or len(values) != STRATEGY_GRAPH_CONTEXT_DIM:
        return None
    result = []
    for value in values:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        result.append(number)
    return tuple(result)


def graph_training_snapshot_id(context: Mapping[str, Any]) -> str:
    """Fingerprint exact captured context; identifiers themselves are excluded.

    This detects accidental mutation, not malicious forgery: it is a content
    checksum, not a signature or a substitute for the live producer boundary.
    """
    canonical = {field: context.get(field) for field in _FINGERPRINT_FIELDS}
    encoded = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_graph_training_context(
    context: Any, *, symbol: str, market: str, as_of: datetime,
) -> bool:
    """Strict causal/provenance validator shared by capture and label ingestion."""
    if not isinstance(context, Mapping):
        return False
    wanted = str(symbol or "").strip().upper()
    decision = _aware(as_of)
    if decision is None or not _supported_symbol(wanted):
        return False
    if market not in {"KR", "US"} or _market_for_symbol(wanted) != market:
        return False
    if context.get("symbol") != wanted or context.get("market") != market:
        return False
    if context.get("schema") != STRATEGY_GRAPH_CONTEXT_SCHEMA or context.get("feature_schema_name") != STRATEGY_GRAPH_CONTEXT_SCHEMA:
        return False
    if context.get("source") != SOURCE or context.get("data_fresh") is not True or context.get("tradable") is not True:
        return False
    features = _vector(context.get("features"))
    if features is None or features[context_index("is_krx")] != float(market == "KR"):
        return False
    observed, recorded, valid = (_aware(context.get(name)) for name in ("as_of", "recorded_at", "valid_until"))
    if observed is None or recorded is None or valid is None:
        return False
    if not observed <= recorded <= decision <= valid or decision - observed > timedelta(seconds=MAX_CONTEXT_AGE_SECONDS):
        return False
    if valid - observed > timedelta(seconds=MAX_CONTEXT_AGE_SECONDS) or _aware(context.get("feature_snapshot_as_of")) != observed:
        return False
    source_snapshot = context.get("source_snapshot_id")
    source_feature = context.get("source_feature_snapshot_id")
    references = context.get("source_provenance")
    if not isinstance(source_snapshot, str) or not source_snapshot or not isinstance(source_feature, str) or not source_feature:
        return False
    if not isinstance(references, (tuple, list)) or not references or any(not isinstance(item, str) or not item for item in references):
        return False
    if source_snapshot not in references or source_feature not in references:
        return False
    try:
        expected = graph_training_snapshot_id(context)
    except (TypeError, ValueError, OverflowError):
        return False
    return context.get("snapshot_id") == expected and context.get("feature_snapshot_id") == expected


class GraphPolicyContextCache:
    """At most 256 recent actual live inputs, copied on record and retrieval."""

    def __init__(
        self, max_entries: int = 256, *, max_age_seconds: float = MAX_CONTEXT_AGE_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._maximum = max(1, min(256, int(max_entries)))
        age = float(max_age_seconds)
        if not math.isfinite(age) or age <= 0:
            raise ValueError("max_age_seconds must be positive and finite")
        self._max_age = min(MAX_CONTEXT_AGE_SECONDS, age)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Mapping[str, Any]] = OrderedDict()

    def record(self, snapshot: Any) -> bool:
        symbol = str(_get(snapshot, "symbol", "") or "").strip().upper()
        current = _aware(self._clock())
        observed, expiry = _aware(_get(snapshot, "as_of")), _aware(_get(snapshot, "valid_until"))
        features = _vector(_get(snapshot, "features"))
        market = _market_for_symbol(symbol) if symbol else ""
        declared = getattr(_get(snapshot, "market"), "value", _get(snapshot, "market"))
        valid = (
            current is not None and observed is not None and expiry is not None
            and _supported_symbol(symbol) and features is not None
            and _get(snapshot, "feature_schema_name") == STRATEGY_GRAPH_CONTEXT_SCHEMA
            and _get(snapshot, "data_fresh") is True and _get(snapshot, "tradable") is True
            and features[context_index("is_krx")] == float(market == "KR")
            and (declared is None or declared == market)
            and observed <= current <= expiry
            and current - observed <= timedelta(seconds=self._max_age)
            and isinstance(_get(snapshot, "snapshot_id"), str) and bool(_get(snapshot, "snapshot_id"))
            and isinstance(_get(snapshot, "feature_snapshot_id"), str) and bool(_get(snapshot, "feature_snapshot_id"))
        )
        if not valid:
            # A newly known invalid/low-trust state must not expose the older
            # healthy snapshot as though the newer observation never arrived.
            with self._lock:
                self._cache.pop(symbol, None)
            return False
        effective_expiry = min(expiry, observed + timedelta(seconds=self._max_age))
        source_id, source_feature = _get(snapshot, "snapshot_id"), _get(snapshot, "feature_snapshot_id")
        values = {
            "schema": STRATEGY_GRAPH_CONTEXT_SCHEMA,
            "feature_schema_name": STRATEGY_GRAPH_CONTEXT_SCHEMA,
            "features": features, "symbol": symbol, "market": market,
            "as_of": observed.isoformat(), "feature_snapshot_as_of": observed.isoformat(),
            "valid_until": effective_expiry.isoformat(), "recorded_at": current.isoformat(),
            "source_snapshot_id": source_id, "source_feature_snapshot_id": source_feature,
            "source_provenance": tuple(dict.fromkeys((source_id, source_feature))),
            "source": SOURCE, "data_fresh": True, "tradable": True,
        }
        with self._lock:
            previous = self._cache.get(symbol)
            if previous is not None:
                previous_time = _aware(previous["as_of"])
                if previous_time > observed:
                    return False
                if previous_time == observed:
                    comparable = ("features", "source_snapshot_id", "source_feature_snapshot_id", "valid_until")
                    if any(previous.get(key) != values[key] for key in comparable):
                        self._cache.pop(symbol, None)
                        return False
                    return True
            fingerprint = graph_training_snapshot_id(values)
            values["snapshot_id"] = values["feature_snapshot_id"] = fingerprint
            self._cache[symbol] = MappingProxyType(values)
            self._cache.move_to_end(symbol)
            while len(self._cache) > self._maximum:
                self._cache.popitem(last=False)
        return True

    def latest(self, symbol: str, as_of: datetime) -> dict[str, Any] | None:
        symbol = str(symbol or "").strip().upper()
        with self._lock:
            found = self._cache.get(symbol)
            if found is None:
                return None
            context = dict(found)
        if not validate_graph_training_context(context, symbol=symbol, market=_market_for_symbol(symbol), as_of=as_of):
            return None
        context["features"] = list(context["features"])
        context["source_provenance"] = list(context["source_provenance"])
        return context

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
