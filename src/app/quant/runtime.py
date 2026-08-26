from __future__ import annotations

import threading
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any

from app.quant.config import QuantConfig, default_quant_config
from app.quant.contracts import QuantBar, QuantEvidence
from app.quant.engine import IncrementalQuantEngine
from app.quant.store import QuantEvidenceStore


class CompletedBarQuantSink:
    """Background-worker adapter for genuinely completed minute bars.

    It never raises into market-data persistence. Errors are explicit in ``health``;
    affected evidence stays absent instead of being replaced with a fake value.
    """

    def __init__(
        self,
        engine: IncrementalQuantEngine | None = None,
        store: QuantEvidenceStore | None = None,
        config: QuantConfig | None = None,
        activation_conditions: tuple[str, ...] = (),
    ) -> None:
        self.config = config or default_quant_config()
        self.engine = engine or IncrementalQuantEngine(self.config)
        self.store = store
        self.activation_conditions = activation_conditions
        self._processed = 0
        self._errors = 0
        self._consecutive_errors = 0
        self._alternate_streams_skipped = 0
        self._selected_streams: dict[tuple[str, str], str] = {}
        self._last_error: str | None = None
        self._circuit_open_until = 0.0
        self._lock = threading.RLock()

    def __call__(self, bar: Any, event: Any) -> tuple[QuantEvidence, ...]:
        with self._lock:
            if self._circuit_open_until > time.monotonic():
                return ()
            if self._circuit_open_until:
                passed, reason = local_quant_self_test(self.config)
                if not passed:
                    self._circuit_open_until = (
                        time.monotonic() + self.config.auto_retry_cooldown_seconds
                    )
                    self._last_error = reason
                    return ()
                self._circuit_open_until = 0.0
                self._consecutive_errors = 0
        try:
            group = getattr(getattr(bar, "meta", None), "market_group", None)
            market = str(getattr(group, "value", group) or "").strip()
            if not market:
                raise ValueError("completed bar market metadata is unavailable")
            symbol = str(getattr(bar, "symbol"))
            stream_id = str(getattr(bar, "stream_id", "") or "").strip()
            if not stream_id:
                raise ValueError("completed bar stream identity is unavailable")
            selection_key = (market, symbol)
            with self._lock:
                selected = self._selected_streams.setdefault(selection_key, stream_id)
                if selected != stream_id:
                    # Never mix KRX/NXT/unified feeds in one rolling state. The first
                    # eligible stream observed for this runtime remains stable.
                    self._alternate_streams_skipped += 1
                    return ()
            received_at = getattr(event, "received_at")
            start = getattr(bar, "minute_start")
            quant_bar = QuantBar(
                symbol=symbol, market=market, interval="1m",
                start_time=start, end_time=start + timedelta(minutes=1),
                received_at=received_at, open=float(getattr(bar, "open")),
                high=float(getattr(bar, "high")), low=float(getattr(bar, "low")),
                close=float(getattr(bar, "close")), volume=float(getattr(bar, "volume")),
            )
            evidence = self.engine.update(quant_bar, as_of=received_at)
            if self.store is not None:
                self.store.append(evidence)
            with self._lock:
                self._processed += 1
                self._consecutive_errors = 0
                self._last_error = None
            return evidence
        except Exception as exc:  # noqa: BLE001 - evidence cannot break KIS persistence.
            with self._lock:
                self._errors += 1
                self._consecutive_errors += 1
                self._last_error = f"{type(exc).__name__}:{exc}"
                if self._consecutive_errors >= self.config.auto_disable_consecutive_errors:
                    self._circuit_open_until = (
                        time.monotonic() + self.config.auto_retry_cooldown_seconds
                    )
            return ()

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.engine.health(),
                "completed_bars_processed": self._processed,
                "completed_bar_errors": self._errors,
                "consecutive_errors": self._consecutive_errors,
                "alternate_streams_skipped": self._alternate_streams_skipped,
                "selected_stream_count": len(self._selected_streams),
                "last_completed_bar_error": self._last_error,
                "activation_mode": self.config.activation_mode,
                "activation_conditions": list(self.activation_conditions),
                "enabled": self._circuit_open_until <= time.monotonic(),
                "circuit_open": self._circuit_open_until > time.monotonic(),
            }


@dataclass(frozen=True)
class QuantActivationDecision:
    enabled: bool
    mode: str
    conditions: tuple[str, ...]
    unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "conditions": list(self.conditions),
            "unavailable_reason": self.unavailable_reason,
        }


def build_quant_sink(
    *,
    config: QuantConfig | None = None,
    store: QuantEvidenceStore | None = None,
    market_store: Any | None = None,
) -> tuple[CompletedBarQuantSink | None, QuantActivationDecision]:
    """Safely auto-activate the non-authoritative local layer.

    Explicit ``off`` always wins. Explicit ``on`` still cannot bypass compatibility,
    deterministic self-test, or persistence initialization.
    """
    cfg = config or default_quant_config()
    mode = str(os.getenv("QUANT_REFERENCE_ACTIVATION_MODE", cfg.activation_mode)).lower()
    legacy = os.getenv("QUANT_REFERENCE_LIVE_ENABLED")
    if legacy is not None:
        normalized = legacy.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            mode = "off"
        elif normalized in {"1", "true", "yes", "on"}:
            mode = "on"
        else:
            return None, QuantActivationDecision(False, mode, (), "invalid_legacy_enable_value")
    if mode not in {"auto", "off", "on"}:
        return None, QuantActivationDecision(False, mode, (), "invalid_activation_mode")
    if mode == "off":
        return None, QuantActivationDecision(False, mode, (), "disabled_by_policy")

    conditions: list[str] = []
    required = (cfg.minimum_python_major, cfg.minimum_python_minor)
    if sys.version_info[:2] < required:
        return None, QuantActivationDecision(
            False, mode, (), f"python_too_old:requires_{required[0]}.{required[1]}"
        )
    conditions.append(f"python>={required[0]}.{required[1]}")
    passed, reason = local_quant_self_test(cfg)
    if not passed:
        return None, QuantActivationDecision(False, mode, tuple(conditions), reason)
    conditions.append("deterministic_self_test_passed")
    try:
        if store is not None:
            evidence_store = store
        else:
            market_path = getattr(market_store, "db_path", None)
            if market_path is None:
                return None, QuantActivationDecision(
                    False, mode, tuple(conditions), "market_store_path_unavailable"
                )
            evidence_store = QuantEvidenceStore(Path(market_path).with_name("quant_reference.sqlite3"))
    except Exception as exc:  # noqa: BLE001
        return None, QuantActivationDecision(
            False, mode, tuple(conditions), f"evidence_store_unavailable:{type(exc).__name__}:{exc}"
        )
    conditions.append("evidence_store_ready")
    conditions.extend(("gs_quant_not_required", "no_order_authority"))
    sink = CompletedBarQuantSink(
        store=evidence_store, config=cfg, activation_conditions=tuple(conditions)
    )
    return sink, QuantActivationDecision(True, mode, tuple(conditions), None)


def local_quant_self_test(config: QuantConfig) -> tuple[bool, str | None]:
    """Small deterministic math check; values are never stored as market evidence."""
    try:
        engine = IncrementalQuantEngine(config)
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        count = max(config.price_window + 1, config.ema_slow + 1, config.rsi_window + 2)
        rows = ()
        for index in range(count):
            price = 100.0 + index
            bar_start = start + timedelta(minutes=index)
            bar_end = bar_start + timedelta(minutes=1)
            rows = engine.update(QuantBar(
                symbol="__SELF_TEST__", market="SELF_TEST", interval="1m",
                start_time=bar_start, end_time=bar_end, received_at=bar_end,
                open=price, high=price + 0.5, low=price - 0.5,
                close=price, volume=1.0,
            ))
        metrics = {row.metric: row for row in rows}
        mean = metrics["rolling_mean"].value
        expected = sum(100.0 + index for index in range(count - config.price_window, count)) / config.price_window
        if mean is None or abs(mean - expected) > 1e-12:
            return False, "deterministic_self_test_failed:rolling_mean"
        if metrics["rsi"].value != 100.0:
            return False, "deterministic_self_test_failed:rsi"
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"deterministic_self_test_error:{type(exc).__name__}:{exc}"
