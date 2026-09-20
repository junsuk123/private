from __future__ import annotations

import os
import time
import hashlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Any, Callable

import numpy as np

from app.paths import runtime_database_path


DEFAULT_BATCH_BUCKETS = (1, 8, 32, 128, 256, 512, 1024, 2048, 4096)


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
            output = 1.0 / (1.0 + np.exp(-np.clip(output, -60.0, 60.0)))
        return [output.astype(np.float32, copy=False)]


class _TorchLinearModel:
    """CUDA is a distinct provider; OpenVINO GPU means Intel graphics."""

    def __init__(self, weights: np.ndarray, bias: np.ndarray | None, activation: str) -> None:
        import torch

        self.torch = torch
        self.weights = torch.as_tensor(weights, device="cuda")
        self.bias = None if bias is None else torch.as_tensor(bias, device="cuda")
        self.activation = activation

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            output = torch.as_tensor(inputs[0], device="cuda") @ self.weights
            if self.bias is not None:
                output = output + self.bias
            if self.activation == "relu":
                output = torch.relu(output)
            elif self.activation == "sigmoid":
                output = torch.sigmoid(output)
            return [output.cpu().numpy()]


class _NumpyNetwork:
    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        self.parameters = parameters

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        p = self.parameters
        x = np.clip((inputs[0] - p["mean"]) / p["scale"], -6.0, 6.0)
        hidden = np.maximum(x @ p["hidden_weights"] + p["hidden_bias"], 0.0)
        return [hidden @ p["output_weights"] + p["output_bias"]]


class _TorchNetwork:
    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        import torch
        self.torch = torch
        self.parameters = {key: torch.as_tensor(value, device="cuda") for key, value in parameters.items()}

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        t, p = self.torch, self.parameters
        with t.inference_mode():
            x = t.clamp((t.as_tensor(inputs[0], device="cuda") - p["mean"]) / p["scale"], -6.0, 6.0)
            hidden = t.relu(x @ p["hidden_weights"] + p["hidden_bias"])
            return [(hidden @ p["output_weights"] + p["output_bias"]).cpu().numpy()]


class NpuRuntimeManager:
    def __init__(
        self,
        *,
        device_preference: str | None = None,
        fallback_device: str = "CPU",
        batch_buckets: tuple[int, ...] = DEFAULT_BATCH_BUCKETS,
        min_batch_for_npu: int | None = None,
    ) -> None:
        self.device_preference = (
            device_preference
            or os.getenv("NPU_DEVICE_PREFERENCE")
            or os.getenv("OPENVINO_DEVICE")
            or "AUTO"
        ).strip().upper() or "AUTO"
        self.fallback_device = fallback_device
        self.batch_buckets = tuple(sorted(set(int(bucket) for bucket in batch_buckets)))
        if not self.batch_buckets or self.batch_buckets[0] < 1:
            raise ValueError("batch_buckets must contain positive sizes")
        self.min_batch_for_npu = int(os.getenv("NPU_MIN_BATCH_FOR_NPU", str(min_batch_for_npu or 128)))
        self._core: Any | None = None
        self._available_devices: tuple[str, ...] | None = None
        self._compiled: OrderedDict[tuple, tuple[Any, str, str | None]] = OrderedDict()
        self._max_compiled = 32
        self._cuda_available: bool | None = None
        self._pending: dict[tuple, Future] = {}
        self._compiler: ThreadPoolExecutor | None = None
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
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        if weights.ndim != 2 or weights.shape[0] != matrix.shape[1]:
            raise ValueError("weights do not match feature dimension")
        if not np.isfinite(weights).all() or (bias is not None and not np.isfinite(bias).all()):
            raise ValueError("model parameters must be finite")
        if bias is not None and bias.shape != (weights.shape[1],):
            raise ValueError("bias does not match output dimension")
        if activation not in {"linear", "relu", "sigmoid"}:
            raise ValueError("unsupported activation")
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
            # Model identity is content based. A shape-only cache silently kept the
            # old weights after every online update.
            digest = hashlib.sha256(weights.tobytes() + (b"" if bias is None else bias.tobytes())).hexdigest()
            key = (module_name, requested, bucket, feature_dim, weights.shape[1], activation, digest)
            with self._lock:
                entry = self._compiled.get(key)
                if entry is not None:
                    model, selected, fallback_reason = entry
                    self._compiled.move_to_end(key)
            if entry is None:
                # This manager also serves the nonblocking live network. A slow
                # legacy scorer compilation must not hold its cache lock hostage.
                compile_started = time.perf_counter()
                model, selected, fallback_reason = self._compile_linear_model(
                    module_name, requested, bucket, feature_dim, weights, bias, activation
                )
                compile_latency_ms = round((time.perf_counter() - compile_started) * 1000.0, 3)
                with self._lock:
                    self._cache(key, (model, selected, fallback_reason))

        try:
            chunks = []
            for start in range(0, count, bucket):
                chunk = matrix[start:start + bucket]
                padded = np.zeros((bucket, feature_dim), dtype=np.float32)
                padded[:len(chunk)] = chunk
                chunks.append(np.asarray(model([padded])[0], dtype=np.float32)[:len(chunk)])
            output = np.concatenate(chunks) if chunks else np.empty((0, weights.shape[1]), dtype=np.float32)
            if not np.isfinite(output).all():
                raise FloatingPointError("non-finite accelerator output")
        except Exception as exc:
            fallback_reason = f"{selected} inference failed: {exc}"
            selected = "CPU_NUMPY"
            output = cpu_func(matrix) if cpu_func is not None else _NumpyLinearModel(weights, bias, activation)([matrix])[0]
            if requested != "CPU_NUMPY":
                with self._lock:
                    self._cache(key, (_NumpyLinearModel(weights, bias, activation), selected, fallback_reason))
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
            "selected_device": self._best_openvino_device(),
            "modules": {name: status.as_dict() for name, status in sorted(self._statuses.items())},
        }

    def snapshot_status(self) -> dict[str, Any]:
        """Read-only diagnostics: never import an accelerator or compile a graph."""
        return {
            "available_devices": self._available_devices or (),
            "requested_device": self.device_preference,
            "modules": {name: status.as_dict() for name, status in tuple(self._statuses.items())},
            "cached_models": len(self._compiled),
            "pending_compilations": len(self._pending),
        }

    def run_network(
        self,
        *,
        module_name: str,
        features: np.ndarray,
        parameters: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, NpuModuleStatus]:
        """Serve a tiny nonlinear model; device discovery/compilation stays off the tick path.

        NumPy serves the exact same fitted parameters while one worker prepares a
        fixed-shape graph. A failed compile is cached with its real fallback device.
        """
        started = time.perf_counter()
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("network features must be a finite 2D matrix")
        p = {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in parameters.items()}
        feature_dim = matrix.shape[1]
        if (p["mean"].shape != (feature_dim,) or p["scale"].shape != (feature_dim,)
                or p["hidden_weights"].ndim != 2 or p["hidden_weights"].shape[0] != feature_dim
                or p["hidden_bias"].shape != (p["hidden_weights"].shape[1],)
                or p["output_weights"].shape != (p["hidden_weights"].shape[1], 2)
                or p["output_bias"].shape != (2,) or np.any(p["scale"] <= 0)
                or any(not np.isfinite(value).all() for value in p.values())):
            raise ValueError("invalid nonlinear model parameters")
        bucket = self.batch_bucket(max(1, len(matrix)))
        digest = hashlib.sha256(b"".join(p[key].tobytes() for key in sorted(p))).hexdigest()
        key = ("network", module_name, self.device_preference, bucket, digest)
        reference = _NumpyNetwork(p)
        model, selected, reason = reference, "CPU_NUMPY", None
        if self.device_preference not in {"CPU", "CPU_NUMPY"}:
            with self._lock:
                # Drain old versions too. Otherwise two completed compiles for
                # retired artifacts could occupy the queue forever.
                for pending_key, future in tuple(self._pending.items()):
                    if not future.done():
                        continue
                    try:
                        self._cache(pending_key, future.result())
                    except Exception:
                        pass
                    del self._pending[pending_key]
                if key in self._compiled:
                    model, selected, reason = self._compiled[key]
                    self._compiled.move_to_end(key)
                else:
                    reason = "Accelerator warming up; serving fitted model on CPU"
                    if key not in self._pending and len(self._pending) < 2:
                        if self._compiler is None:
                            self._compiler = ThreadPoolExecutor(max_workers=1, thread_name_prefix="obaits-model-compile")
                        self._pending[key] = self._compiler.submit(self._compile_network, module_name, bucket, p)
        try:
            outputs = []
            for start in range(0, len(matrix), bucket):
                chunk = matrix[start:start + bucket]
                padded = np.zeros((bucket, feature_dim), dtype=np.float32)
                padded[:len(chunk)] = chunk
                outputs.append(np.asarray(model([padded])[0], dtype=np.float32)[:len(chunk)])
            output = np.concatenate(outputs) if outputs else np.empty((0, 2), dtype=np.float32)
            if not np.isfinite(output).all():
                raise FloatingPointError("non-finite accelerator output")
        except Exception as exc:
            selected, reason = "CPU_NUMPY", f"accelerator inference failed: {exc}"
            output = reference([matrix])[0]
            with self._lock:
                self._cache(key, (reference, selected, reason))
        elapsed = (time.perf_counter() - started) * 1000.0
        status = NpuModuleStatus(
            enabled=True, backend=selected, uses_npu=selected.startswith("NPU"),
            requested_device=self.device_preference, selected_device=selected,
            fallback_reason=reason, batch_size=bucket, feature_dim=feature_dim,
            last_latency_ms=round(elapsed, 3), last_items=len(matrix),
            last_profile={"module": module_name, "model_digest": digest[:16]},
        )
        self._statuses[module_name] = status
        return output, status

    def _compile_network(self, module_name: str, bucket: int, p: dict[str, np.ndarray]) -> tuple[Any, str, str | None]:
        selected = self._best_openvino_device()
        reference = _NumpyNetwork(p)
        if selected == "CPU_NUMPY":
            return reference, selected, "No compatible accelerator available"
        try:
            if selected == "CUDA":
                compiled = _TorchNetwork(p)
            else:
                import openvino as ov
                ops = ov.opset8
                x = ops.parameter([bucket, len(p["mean"])], ov.Type.f32, name="features")
                scaled = ops.clamp(ops.divide(ops.subtract(x, ops.constant(p["mean"])), ops.constant(p["scale"])), -6.0, 6.0)
                hidden = ops.relu(ops.add(ops.matmul(scaled, ops.constant(p["hidden_weights"]), False, False), ops.constant(p["hidden_bias"])))
                output = ops.add(ops.matmul(hidden, ops.constant(p["output_weights"]), False, False), ops.constant(p["output_bias"]))
                cache_path = runtime_database_path("openvino_cache")
                cache_path.mkdir(parents=True, exist_ok=True)
                compiled = (self._core or ov.Core()).compile_model(
                    ov.Model([output], [x], module_name), selected,
                    {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(cache_path)},
                )
            # Reject materially different numerics before exposing the backend.
            probe = np.tile(p["mean"], (bucket, 1)).astype(np.float32)
            probe += np.linspace(-2.0, 2.0, bucket, dtype=np.float32)[:, None] * p["scale"]
            if not np.allclose(compiled([probe])[0], reference([probe])[0], rtol=0.005, atol=0.01):
                raise ValueError("accelerator parity check failed")
            return compiled, selected, None
        except Exception as exc:
            return reference, "CPU_NUMPY", f"{selected} compile/parity failed: {exc}"

    def _requested_device(self, count: int, enabled: bool) -> str:
        if not enabled or count < self.min_batch_for_npu:
            return "CPU_NUMPY"
        return self._best_openvino_device()

    def _best_openvino_device(self) -> str:
        available = {device.split(".")[0] for device in self.available_devices}
        preference = self.device_preference
        if preference in {"CPU", "CPU_NUMPY"}:
            return "CPU_NUMPY"
        if self._cuda_available is None:
            try:
                import torch
                self._cuda_available = bool(torch.cuda.is_available())
            except Exception:
                self._cuda_available = False
        if self._cuda_available:
            available.add("CUDA")
        ladder = (
            ("NPU", "CUDA", "GPU", "CPU")
            if preference == "AUTO"
            else tuple(dict.fromkeys((preference, "NPU", "CUDA", "GPU", "CPU")))
        )
        return next((device for device in ladder if device in available), "CPU_NUMPY")

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
        if requested == "CUDA":
            try:
                return _TorchLinearModel(weights, bias, activation), "CUDA", None
            except Exception as exc:
                return _NumpyLinearModel(weights, bias, activation), "CPU_NUMPY", f"CUDA unavailable: {exc}"
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
                return core.compile_model(model, requested, {"PERFORMANCE_HINT": "LATENCY"}), requested, None
            except Exception as exc:
                return core.compile_model(model, self.fallback_device), self.fallback_device, f"{requested} compile failed: {exc}"
        except Exception as exc:
            return _NumpyLinearModel(weights, bias, activation), "CPU_NUMPY", f"OpenVINO unavailable: {exc}"

    def _cache(self, key: tuple, entry: tuple[Any, str, str | None]) -> None:
        self._compiled[key] = entry
        self._compiled.move_to_end(key)
        while len(self._compiled) > self._max_compiled:
            self._compiled.popitem(last=False)

    def close(self) -> None:
        if self._compiler is not None:
            self._compiler.shutdown(wait=False, cancel_futures=True)

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
    if get_npu_runtime_manager.cache_info().currsize:
        get_npu_runtime_manager().close()
    get_npu_runtime_manager.cache_clear()
