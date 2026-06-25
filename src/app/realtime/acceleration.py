from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from app.graph import get_ontology_runtime


@dataclass(frozen=True)
class RealtimeRuntimeStatus:
    requested_backend: str
    active_backend: str
    provider: str
    available_devices: tuple[str, ...]
    uses_npu: bool
    latency_profile: str
    prediction_horizons_seconds: tuple[int, ...]
    fallback_reason: str | None
    runtime_notes: tuple[str, ...]


class RealtimeAccelerationPolicy:
    """Central low-latency policy for short-horizon prediction paths."""

    def __init__(
        self,
        latency_profile: str | None = None,
        prediction_horizons_seconds: tuple[int, ...] = (5, 15, 30, 60, 300, 3600),
    ) -> None:
        self.latency_profile = (latency_profile or os.getenv("REALTIME_LATENCY_PROFILE", "low_latency")).strip()
        self.prediction_horizons_seconds = prediction_horizons_seconds

    def apply_process_hints(self) -> None:
        os.environ.setdefault("ONTOLOGY_ACCELERATOR", "NPU")
        os.environ.setdefault("OPENVINO_DEVICE", "NPU")
        os.environ.setdefault("OPENVINO_HINT_PERFORMANCE_MODE", "LATENCY")
        os.environ.setdefault("OPENVINO_ENABLE_CPU_PINNING", "YES")
        os.environ.setdefault("OPENVINO_CACHE_DIR", "data/runtime/openvino_cache")
        if self._openvino_npu_available():
            os.environ.setdefault("LLM_EVENT_INFERENCE_BACKEND", "openvino")
            os.environ.setdefault("LLM_EVENT_DEVICE", "NPU")
            os.environ.setdefault("LLM_EVENT_PROVIDER", "openvino-llm")

    def status(self) -> RealtimeRuntimeStatus:
        self.apply_process_hints()
        runtime = get_ontology_runtime()
        notes = [
            "OpenVINO NPU is preferred for compatible model inference, including local event classification.",
            "Pure Python trading logic, graph rules, and risk checks stay deterministic and are distributed on CPU workers.",
            "CPU deterministic fallback remains enabled so trading logic never depends on unavailable acceleration.",
            "Short-horizon predictions are configured for seconds-to-one-hour horizons.",
        ]
        if not runtime.uses_npu:
            notes.append("Install/configure OpenVINO NPU runtime to move compatible inference graphs to NPU.")
        return RealtimeRuntimeStatus(
            requested_backend=runtime.requested_backend,
            active_backend=runtime.active_backend,
            provider=runtime.provider,
            available_devices=runtime.available_devices,
            uses_npu=runtime.uses_npu,
            latency_profile=self.latency_profile,
            prediction_horizons_seconds=self.prediction_horizons_seconds,
            fallback_reason=runtime.fallback_reason,
            runtime_notes=tuple(notes),
        )

    @staticmethod
    def _openvino_npu_available() -> bool:
        if importlib.util.find_spec("openvino") is None:
            return False
        try:
            from openvino import Core  # type: ignore
        except Exception:
            try:
                from openvino.runtime import Core  # type: ignore
            except Exception:
                return False
        try:
            return "NPU" in {str(device).upper() for device in Core().available_devices}
        except Exception:
            return False
