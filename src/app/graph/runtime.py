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

    @property
    def uses_accelerator(self) -> bool:
        """True when reasoning runs anywhere other than the CPU fallback.

        ``uses_npu`` answers a narrower question and is kept because the dashboard
        and the NPU diagnostics page ask exactly that one. This is the question the
        acceleration summary actually wants: an Intel iGPU carrying the ontology
        scorer is acceleration, and reporting it as "CPU fallback" because it is not
        an NPU understates what the machine is doing.
        """
        return self.active_backend != "CPU"

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "provider": self.provider,
            "available_devices": self.available_devices,
            "uses_npu": self.uses_npu,
            "uses_accelerator": self.uses_accelerator,
            "fallback_reason": self.fallback_reason,
        }


#: OpenVINO devices this runtime knows how to place ontology scoring on, in
#: preference order. CPU is deliberately absent: it is not a device that is chosen
#: here, it is the fallback every path below terminates at.
OPENVINO_DEVICES: tuple[str, ...] = ("NPU", "GPU")


@lru_cache(maxsize=1)
def get_ontology_runtime() -> OntologyRuntime:
    requested = os.getenv("ONTOLOGY_ACCELERATOR", "NPU").strip().upper() or "NPU"
    if requested in {"AUTO", *OPENVINO_DEVICES}:
        runtime = _openvino_runtime(requested)
        if runtime is not None:
            return runtime
        return OntologyRuntime(
            requested_backend=requested,
            active_backend="CPU",
            provider="python-rules",
            available_devices=(),
            fallback_reason=(
                "OpenVINO runtime was not detected. "
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


def _openvino_runtime(requested: str) -> OntologyRuntime | None:
    """Resolve ``requested`` against the devices OpenVINO reports on this machine.

    GPU is accepted here, not only NPU. ``ONTOLOGY_ACCELERATOR`` is written by
    ``RealtimeAccelerationPolicy.apply_process_hints`` from the per-workload plan,
    and ``ontology_candidate_scorer``'s ladder is ``(NPU, GPU, CPU)`` — so on a
    machine with an Intel iGPU and no NPU the plan legitimately asks for GPU. This
    function used to recognise only NPU, so that request fell through to
    "Unsupported ontology backend: GPU" and the scorer dropped to CPU rules while
    the plan reported it as accelerated. The two disagreed, and the plan was right.
    """
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

    # AUTO walks the whole preference order; an explicit device asks for that one
    # and is not silently upgraded to a different accelerator.
    wanted = OPENVINO_DEVICES if requested == "AUTO" else (requested,)
    for device in wanted:
        if device in devices:
            return OntologyRuntime(
                requested_backend=requested,
                active_backend=device,
                provider="openvino",
                available_devices=devices,
            )

    return OntologyRuntime(
        requested_backend=requested,
        active_backend="CPU",
        provider="openvino",
        available_devices=devices,
        fallback_reason=(
            f"OpenVINO is installed, but no {' or '.join(wanted)} device is available."
        ),
    )
