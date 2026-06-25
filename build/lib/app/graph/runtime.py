from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class OntologyRuntime:
    requested_backend: str
    active_backend: str
    provider: str
    available_devices: tuple[str, ...]
    fallback_reason: str | None = None

    @property
    def uses_npu(self) -> bool:
        return self.active_backend == "NPU"

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "provider": self.provider,
            "available_devices": self.available_devices,
            "uses_npu": self.uses_npu,
            "fallback_reason": self.fallback_reason,
        }


@lru_cache(maxsize=1)
def get_ontology_runtime() -> OntologyRuntime:
    requested = os.getenv("ONTOLOGY_ACCELERATOR", "NPU").strip().upper() or "NPU"
    if requested in {"AUTO", "NPU"}:
        runtime = _openvino_npu_runtime(requested)
        if runtime is not None:
            return runtime
        return OntologyRuntime(
            requested_backend=requested,
            active_backend="CPU",
            provider="python-rules",
            available_devices=(),
            fallback_reason=(
                "OpenVINO NPU runtime was not detected. "
                "Ontology reasoning is running on deterministic CPU rules."
            ),
        )
    return OntologyRuntime(
        requested_backend=requested,
        active_backend="CPU",
        provider="python-rules",
        available_devices=(),
        fallback_reason=None if requested == "CPU" else f"Unsupported ontology backend: {requested}",
    )


def reset_ontology_runtime_cache() -> None:
    get_ontology_runtime.cache_clear()


def _openvino_npu_runtime(requested: str) -> OntologyRuntime | None:
    if importlib.util.find_spec("openvino") is None:
        return None

    try:
        from openvino.runtime import Core  # type: ignore
    except Exception:
        try:
            from openvino import Core  # type: ignore
        except Exception:
            return None

    try:
        devices = tuple(str(device).upper() for device in Core().available_devices)
    except Exception:
        return None

    if "NPU" not in devices:
        return OntologyRuntime(
            requested_backend=requested,
            active_backend="CPU",
            provider="openvino",
            available_devices=devices,
            fallback_reason="OpenVINO is installed, but no NPU device is available.",
        )

    return OntologyRuntime(
        requested_backend=requested,
        active_backend="NPU",
        provider="openvino",
        available_devices=devices,
    )
