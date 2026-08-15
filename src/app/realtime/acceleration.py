from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from app.graph import get_ontology_runtime
from app.realtime.device_plan import (
    CPU,
    Placement,
    plan_as_dict,
    plan_devices,
    probe_devices,
)


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
    device_plan: tuple[dict, ...] = ()
    detected_devices: tuple[str, ...] = ()
    device_probe_error: str | None = None


class RealtimeAccelerationPolicy:
    """Central low-latency policy for short-horizon prediction paths."""

    def __init__(
        self,
        latency_profile: str | None = None,
        prediction_horizons_seconds: tuple[int, ...] = (5, 15, 30, 60, 300, 3600),
    ) -> None:
        self.latency_profile = (latency_profile or os.getenv("REALTIME_LATENCY_PROFILE", "low_latency")).strip()
        self.prediction_horizons_seconds = prediction_horizons_seconds
        self._placements: tuple[Placement, ...] | None = None
        self._inventory = None

    def placements(self) -> tuple[Placement, ...]:
        """Per-workload device assignment for THIS machine."""
        if self._placements is None:
            self._inventory = probe_devices()
            self._placements = plan_devices(self._inventory)
        return self._placements

    def apply_process_hints(self) -> None:
        # Device hints follow the plan instead of a hardcoded "NPU".
        #
        # These were pinned to NPU unconditionally, which did two things wrong: on a
        # machine with no NPU every consumer took a failed-compile fallback path
        # rather than simply being told to use what exists, and on a machine WITH an
        # iGPU (this one reports Intel Arc alongside AI Boost) the GPU was never
        # eligible for anything. The plan picks per workload and always terminates
        # at CPU, so a machine with neither accelerator gets a valid assignment
        # rather than a degraded one.
        by_key = {item.workload: item.device for item in self.placements()}
        os.environ.setdefault(
            "ONTOLOGY_ACCELERATOR", by_key.get("ontology_candidate_scorer", CPU)
        )
        os.environ.setdefault(
            "OPENVINO_DEVICE", by_key.get("strategy_utility_rgcn_shadow", CPU)
        )
        os.environ.setdefault("OPENVINO_HINT_PERFORMANCE_MODE", "LATENCY")
        os.environ.setdefault("OPENVINO_ENABLE_CPU_PINNING", "YES")
        os.environ.setdefault("OPENVINO_CACHE_DIR", "data/runtime/openvino_cache")
        # Event classification is a separate model lifecycle. NPU availability alone
        # must not turn an Ollama model id into a nonexistent embedded model path.
        # Explicit LLM_EVENT_* configuration may still select OpenVINO/NPU.

    def status(self) -> RealtimeRuntimeStatus:
        self.apply_process_hints()
        runtime = get_ontology_runtime()
        notes = [
            "OpenVINO NPU is preferred for compatible model inference, including local event classification.",
            "Pure Python trading logic, graph rules, and risk checks stay deterministic and are distributed on CPU workers.",
            "CPU deterministic fallback remains enabled so trading logic never depends on unavailable acceleration.",
            "Short-horizon predictions are configured for seconds-to-one-hour horizons.",
        ]
        placements = self.placements()
        inventory = self._inventory
        # Advise on the NPU only when reasoning actually landed on CPU. A machine
        # running the scorer on an Intel iGPU is accelerated; telling it to
        # "install OpenVINO NPU runtime" reads as a defect report on a healthy box.
        if not runtime.uses_accelerator:
            notes.append("Install/configure OpenVINO NPU runtime to move compatible inference graphs to NPU.")
        elif not runtime.uses_npu:
            notes.append(
                f"Ontology reasoning is accelerated on {runtime.active_backend} via OpenVINO; "
                "an NPU would be preferred where one exists."
            )
        accelerated = [p for p in placements if p.device != CPU]
        notes.append(
            f"Device plan: {len(accelerated)}/{len(placements)} workloads accelerated; "
            "every workload falls back to CPU and decision paths are pinned to CPU."
        )
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
            device_plan=tuple(plan_as_dict(placements)),
            detected_devices=tuple(inventory.available) if inventory else (CPU,),
            device_probe_error=inventory.probe_error if inventory else None,
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
