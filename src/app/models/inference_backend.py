from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BackendStatus:
    requested_backend: str
    active_backend: str
    uses_npu: bool
    fallback_reason: str | None = None


class InferenceBackend(Protocol):
    def infer(self, features: np.ndarray) -> np.ndarray:
        ...

    def status(self) -> BackendStatus:
        ...


class CpuSignalModel:
    def __init__(self, weights: np.ndarray, requested_backend: str = "CPU") -> None:
        self.weights = weights.astype(np.float32)
        self.requested_backend = requested_backend

    def infer(self, features: np.ndarray) -> np.ndarray:
        return features.astype(np.float32) @ self.weights

    def status(self) -> BackendStatus:
        return BackendStatus(
            requested_backend=self.requested_backend,
            active_backend="CPU_NUMPY",
            uses_npu=False,
            fallback_reason=None if self.requested_backend.upper() in {"CPU", "CPU_NUMPY"} else "Using CPU fallback.",
        )


class OpenVinoNpuSignalModel:
    def __init__(self, weights: np.ndarray, requested_device: str = "NPU") -> None:
        self.weights = weights.astype(np.float32)
        self.requested_device = requested_device
        self._compiled = None
        self._fallback_reason: str | None = None
        self._active_backend = "uninitialized"

    def infer(self, features: np.ndarray) -> np.ndarray:
        compiled = self._compiled_model(features.shape)
        return compiled([features.astype(np.float32)])[0]

    def status(self) -> BackendStatus:
        if self._compiled is None:
            try:
                self._compiled_model((1, self.weights.shape[0]))
            except Exception:
                pass
        return BackendStatus(
            requested_backend=self.requested_device,
            active_backend=self._active_backend,
            uses_npu=self._active_backend.upper() == "NPU",
            fallback_reason=self._fallback_reason,
        )

    def _compiled_model(self, input_shape: tuple[int, ...]):
        if self._compiled is not None:
            return self._compiled
        try:
            import openvino as ov
        except ModuleNotFoundError as exc:
            self._compiled = _NumpyCompiled(self.weights)
            self._active_backend = "CPU_NUMPY"
            self._fallback_reason = f"OpenVINO unavailable: {exc}"
            return self._compiled
        ops = ov.opset8
        x = ops.parameter(list(input_shape), ov.Type.f32, name="signal_features")
        model = ov.Model([ops.matmul(x, ops.constant(self.weights), False, False)], [x], "signal_model")
        core = ov.Core()
        try:
            self._compiled = core.compile_model(model, self.requested_device)
            self._active_backend = self.requested_device
        except Exception as exc:
            self._compiled = core.compile_model(model, "CPU")
            self._active_backend = "CPU"
            self._fallback_reason = f"{self.requested_device} compile failed: {exc}"
        return self._compiled


class _NumpyCompiled:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0] @ self.weights]
