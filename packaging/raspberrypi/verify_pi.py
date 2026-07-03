"""Verify the NPU-free (CPU-only) runtime on Raspberry Pi.

Run standalone:
    PYTHONPATH=src ONTOLOGY_ACCELERATOR=CPU .venv-pi/bin/python \
        packaging/raspberrypi/verify_pi.py

Exits non-zero on the first failed check so `bootstrap.sh` fails loudly.
"""

from __future__ import annotations

import importlib.util
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Force the CPU-only profile regardless of the caller's environment.
os.environ.setdefault("ONTOLOGY_ACCELERATOR", "CPU")
os.environ.setdefault("LLM_EVENT_CLASSIFIER_ENABLED", "false")
for npu_var in ("OPENVINO_DEVICE", "LLM_EVENT_DEVICE", "LLM_EVENT_INFERENCE_BACKEND"):
    os.environ.pop(npu_var, None)


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    print("Raspberry Pi CPU-only runtime verification")

    # 1. NPU/OpenVINO must NOT be a hard requirement.
    if importlib.util.find_spec("openvino") is not None:
        print("  [note] openvino is importable but will not be used (ONTOLOGY_ACCELERATOR=CPU).")
    else:
        _ok("openvino absent — pure CPU install, as intended.")

    # 2. Ontology runtime falls back to deterministic CPU rules.
    from app.graph.runtime import get_ontology_runtime

    rt = get_ontology_runtime()
    if rt.uses_npu:
        _fail(f"ontology runtime unexpectedly using NPU: {rt.as_dict()}")
    _ok(f"ontology runtime backend={rt.active_backend} provider={rt.provider}")

    # 3. CPU inference backend produces correct output.
    import numpy as np

    from app.models.inference_backend import CpuSignalModel

    weights = np.ones((3, 2), dtype=np.float32)
    out = CpuSignalModel(weights).infer(np.ones((1, 3), dtype=np.float32))
    if out.tolist() != [[3.0, 3.0]]:
        _fail(f"CPU inference produced wrong result: {out.tolist()}")
    _ok("CPU signal inference correct.")

    # 4. Native screening is optional; Python fallback must be usable.
    from app.native.screening import native_screening_available

    if native_screening_available():
        _ok("native Rust screening core present (optional accelerator active).")
    else:
        _ok("native screening core absent — pure-Python fallback in use.")

    # 5. The realtime acceleration policy reports a non-NPU CPU status.
    from app.realtime.acceleration import RealtimeAccelerationPolicy

    status = RealtimeAccelerationPolicy().status()
    if status.uses_npu:
        _fail("realtime acceleration policy unexpectedly reports NPU in use.")
    _ok(f"realtime policy active_backend={status.active_backend} latency={status.latency_profile}")

    # 6. The FastAPI app imports and constructs.
    import app.web as web

    if type(web.app).__name__ != "FastAPI":
        _fail(f"app.web.app is not a FastAPI instance: {type(web.app)}")
    _ok("FastAPI application imported and constructed.")

    print("\nVERIFICATION_OK — the system runs fully on CPU without NPU.")


if __name__ == "__main__":
    main()
