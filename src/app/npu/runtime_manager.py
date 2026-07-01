from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Any, Callable

import numpy as np


DEFAULT_BATCH_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class NpuModuleStatus:
    enabled: bool
    backend: str
    uses_npu: bool
    requested_device: str
    selected_device: str
    fallback_reason: str | None = None
    batch_size: int | None = None
    feature_dim: int | None = None
    last_latency_ms: float | None = None
    last_items: int = 0
    items_per_second: float | None = None
    compile_latency_ms: float | None = None
    last_profile: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "uses_npu": self.uses_npu,
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "fallback_reason": self.fallback_reason,
            "batch_size": self.batch_size,
            "feature_dim": self.feature_dim,
            "last_latency_ms": self.last_latency_ms,
            "last_items": self.last_items,
            "items_per_second": self.items_per_second,
            "compile_latency_ms": self.compile_latency_ms,
            "last_profile": dict(self.last_profile),
        }


class _NumpyLinearModel:
    def __init__(self, weights: np.ndarray, bias: np.ndarray | None = None, activation: str = "linear") -> None:
        self.weights = weights.astype(np.float32, copy=False)
        self.bias = None if bias is None else bias.astype(np.float32, copy=False)
        self.activation = activation

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        output = np.nan_to_num(inputs[0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False) @ self.weights
        if self.bias is not None:
            output = output + self.bias
        if self.activation == "relu":
            output = np.maximum(output, 0.0)
        if self.activation == "sigmoid":
            output = 1.0 / (1.0 + np.exp(-output))
        return [output.astype(np.float32, copy=False)]


class NpuRuntimeManager:
    def __init__(
        self,
        *,
        device_preference: str | None = None,
        fallback_device: str = "CPU",
        batch_buckets: tuple[int, ...] = DEFAULT_BATCH_BUCKETS,
        min_batch_for_npu: int | None = None,
    ) -> None:
        self.device_preference = (device_preference or os.getenv("NPU_DEVICE_PREFERENCE", "NPU")).strip() or "NPU"
        self.fallback_device = fallback_device
        self.batch_buckets = tuple(sorted(int(bucket) for bucket in batch_buckets))
        self.min_batch_for_npu = int(os.getenv("NPU_MIN_BATCH_FOR_NPU", str(min_batch_for_npu or 128)))
        self._core: Any | None = None
        self._available_devices: tuple[str, ...] | None = None
        self._compiled: dict[tuple[str, str, int, int, int, str], Any] = {}
        self._statuses: dict[str, NpuModuleStatus] = {}
        self._lock = Lock()

    @property
    def available_devices(self) -> tuple[str, ...]:
        if self._available_devices is not None:
            return self._available_devices
        try:
            import openvino as ov

            self._core = ov.Core()
            self._available_devices = tuple(str(device).upper() for device in self._core.available_devices)
        except Exception:
            self._available_devices = ()
        return self._available_devices

    def batch_bucket(self, count: int) -> int:
        for bucket in self.batch_buckets:
            if count <= bucket:
                return bucket
        return self.batch_buckets[-1]

    def run_linear(
        self,
        *,
        module_name: str,
        features: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray | None = None,
        activation: str = "linear",
        enabled: bool = True,
        cpu_func: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, NpuModuleStatus]:
        matrix = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if matrix.ndim != 2:
            raise ValueError("features must be a 2D matrix")
        count, feature_dim = matrix.shape
        bucket = self.batch_bucket(max(1, count))
        requested = self._requested_device(count, enabled)
        selected = requested
        fallback_reason = None
        started = time.perf_counter()
        compile_latency_ms = None

        if requested == "CPU_NUMPY":
            model = _NumpyLinearModel(weights, bias, activation)
            selected = "CPU_NUMPY"
            fallback_reason = "NPU disabled or batch below min_batch_for_npu"
        else:
            key = (module_name, requested, bucket, feature_dim, weights.shape[1], activation)
            with self._lock:
                model = self._compiled.get(key)
                if model is None:
                    compile_started = time.perf_counter()
                    model, selected, fallback_reason = self._compile_linear_model(
                        module_name, requested, bucket, feature_dim, weights, bias, activation
                    )
                    compile_latency_ms = round((time.perf_counter() - compile_started) * 1000.0, 3)
                    self._compiled[key] = model

        padded = np.zeros((bucket, feature_dim), dtype=np.float32)
        padded[:count, :] = matrix
        try:
            output = np.asarray(model([padded])[0], dtype=np.float32)[:count]
        except Exception as exc:
            fallback_reason = f"{selected} inference failed: {exc}"
            selected = "CPU_NUMPY"
            output = cpu_func(matrix) if cpu_func is not None else _NumpyLinearModel(weights, bias, activation)([matrix])[0]
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        status = NpuModuleStatus(
            enabled=enabled,
            backend=selected,
            uses_npu=selected.upper().startswith("NPU"),
            requested_device=requested,
            selected_device=selected,
            fallback_reason=fallback_reason,
            batch_size=bucket,
            feature_dim=feature_dim,
            last_latency_ms=latency_ms,
            last_items=count,
            items_per_second=round(count / (latency_ms / 1000.0), 2) if latency_ms > 0 else None,
            compile_latency_ms=compile_latency_ms,
            last_profile={"module": module_name, "batch_bucket": bucket, "items": count},
        )
        self._statuses[module_name] = status
        return output.astype(np.float32, copy=False), status

    def status(self, module_name: str | None = None) -> dict[str, Any]:
        if module_name is not None:
            status = self._statuses.get(module_name)
            return status.as_dict() if status is not None else self._default_status(module_name).as_dict()
        return {
            "available_devices": self.available_devices,
            "selected_device": self.device_preference if "NPU" in self.available_devices else "CPU",
            "modules": {name: status.as_dict() for name, status in sorted(self._statuses.items())},
        }

    def _requested_device(self, count: int, enabled: bool) -> str:
        if not enabled or count < self.min_batch_for_npu:
            return "CPU_NUMPY"
        if "NPU" not in self.available_devices:
            return "CPU_NUMPY"
        return self.device_preference

    def _compile_linear_model(
        self,
        module_name: str,
        requested: str,
        bucket: int,
        feature_dim: int,
        weights: np.ndarray,
        bias: np.ndarray | None,
        activation: str,
    ) -> tuple[Any, str, str | None]:
        try:
            import openvino as ov

            core = self._core or ov.Core()
            ops = ov.opset8
            x = ops.parameter([bucket, feature_dim], ov.Type.f32, name=f"{module_name}_features")
            y = ops.matmul(x, ops.constant(weights.astype(np.float32, copy=False)), False, False)
            if bias is not None:
                y = ops.add(y, ops.constant(bias.astype(np.float32, copy=False)))
            if activation == "relu":
                y = ops.relu(y)
            if activation == "sigmoid":
                y = ops.sigmoid(y)
            model = ov.Model([y], [x], module_name)
            try:
                return core.compile_model(model, requested), requested, None
            except Exception as exc:
                return core.compile_model(model, self.fallback_device), self.fallback_device, f"{requested} compile failed: {exc}"
        except Exception as exc:
            return _NumpyLinearModel(weights, bias, activation), "CPU_NUMPY", f"OpenVINO unavailable: {exc}"

    def _default_status(self, module_name: str) -> NpuModuleStatus:
        return NpuModuleStatus(
            enabled=True,
            backend="uninitialized",
            uses_npu=False,
            requested_device=self.device_preference,
            selected_device="uninitialized",
            fallback_reason=f"{module_name} has not run yet",
        )


@lru_cache(maxsize=1)
def get_npu_runtime_manager() -> NpuRuntimeManager:
    return NpuRuntimeManager()


def reset_npu_runtime_manager() -> None:
    get_npu_runtime_manager.cache_clear()
