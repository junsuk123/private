from __future__ import annotations

import hashlib
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import numpy as np

from app.models.strategy_utility.rgcn import (
    FixedShapeStrategyUtilityModel,
    StrategyUtilityOutput,
    output_from_raw,
)


@dataclass(frozen=True)
class OpenVinoUtilityStatus:
    requested_device: str
    compiled_devices: tuple[str, ...]
    fallback_reason: str | None
    compile_ms: float
    model_hash: str
    precision: str


class OpenVinoStrategyUtilityRuntime:
    def __init__(
        self,
        reference: FixedShapeStrategyUtilityModel,
        *,
        requested_device: str = "CPU",
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.reference = reference
        try:
            import openvino as ov
        except ImportError as exc:
            # OpenVINO is an optional ``npu`` extra, while the order-free shadow
            # evaluator is part of the base runtime.  A base install must retain
            # the deterministic NumPy reference path instead of preventing the
            # entire event collector from starting.
            self.core = None
            self.compiled = None
            self.status = OpenVinoUtilityStatus(
                requested_device=requested_device,
                compiled_devices=("CPU",),
                fallback_reason=f"OpenVINO unavailable; using NumPy CPU: {exc}",
                compile_ms=0.0,
                model_hash=_model_hash(reference),
                precision="FP32",
            )
            return

        self.core = ov.Core()
        graph = _build_graph(reference)
        started = time.perf_counter()
        fallback = None
        try:
            self.compiled = self.core.compile_model(graph, requested_device)
        except Exception as exc:
            if not allow_cpu_fallback or requested_device.upper() == "CPU":
                raise
            fallback = f"{requested_device} compile failed: {exc}"
            self.compiled = self.core.compile_model(graph, "CPU")
        compile_ms = (time.perf_counter() - started) * 1000
        try:
            execution_devices = self.compiled.get_property("EXECUTION_DEVICES")
            devices = (
                (execution_devices,)
                if isinstance(execution_devices, str)
                else tuple(execution_devices)
            )
        except Exception:
            devices = (requested_device if fallback is None else "CPU",)
        self.status = OpenVinoUtilityStatus(
            requested_device=requested_device,
            compiled_devices=tuple(str(value) for value in devices),
            fallback_reason=fallback,
            compile_ms=compile_ms,
            model_hash=_model_hash(reference),
            precision="FP32",
        )

    def infer(
        self,
        x: np.ndarray,
        adjacency: np.ndarray,
        node_mask: np.ndarray,
        strategy_mask: np.ndarray,
    ) -> StrategyUtilityOutput:
        self.reference._validate(x, adjacency, node_mask, strategy_mask)
        if self.compiled is None:
            return self.reference.infer(x, adjacency, node_mask, strategy_mask)
        result = self.compiled(
            {
                "features": x.astype(np.float32, copy=False),
                "adjacency": adjacency.astype(np.float32, copy=False),
                "node_mask": node_mask.astype(np.float32, copy=False),
            }
        )
        raw = np.asarray(result[self.compiled.output(0)])
        no_trade_raw = np.asarray(result[self.compiled.output(1)])
        if not np.isfinite(raw).all() or not np.isfinite(no_trade_raw).all():
            raise FloatingPointError("non-finite graph accelerator output")
        return output_from_raw(raw, no_trade_raw, node_mask, strategy_mask)


_COMPILER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="obaits-temporal-rgcn")


class AdaptiveStrategyUtilityRuntime:
    """Serve exact fitted NumPy weights while NPU compilation runs off the tick path."""

    def __init__(self, reference: FixedShapeStrategyUtilityModel, *, requested_device: str | None = None) -> None:
        self.reference = reference
        self.requested_device = (requested_device or os.getenv("NPU_DEVICE_PREFERENCE", "AUTO")).upper()
        self._runtime = None
        self._error = "Accelerator warming up; using fitted NumPy R-GCN"
        self._pending = (_COMPILER.submit(self._compile) if self.requested_device != "CPU_NUMPY" else None)

    @property
    def status(self) -> OpenVinoUtilityStatus:
        if self._runtime is not None:
            return self._runtime.status
        return OpenVinoUtilityStatus(self.requested_device, ("CPU_NUMPY",), self._error, 0.0,
                                     _model_hash(self.reference), "FP32")

    def _compile(self):
        import openvino as ov

        devices = {str(value).split(".")[0] for value in ov.Core().available_devices}
        requested = self.requested_device
        if requested == "AUTO":
            requested = next((value for value in ("NPU", "GPU", "CPU") if value in devices), "CPU")
        runtime = OpenVinoStrategyUtilityRuntime(self.reference, requested_device=requested)
        # Compare ALL raw channels, not only a postprocessed utility score.
        c = self.reference.config
        rng = np.random.default_rng(223)
        x = rng.normal(0, 0.15, (c.batch_size, c.time_steps, c.max_nodes, c.feature_dim)).astype(np.float32)
        adjacency = np.full((c.batch_size, c.time_steps, c.relation_count, c.max_nodes, c.max_nodes),
                            1.0 / max(1, c.max_nodes), dtype=np.float32)
        masks = np.ones((c.batch_size, c.time_steps, c.max_nodes), dtype=np.float32) / c.time_steps
        strategy_mask = np.ones((c.batch_size, c.max_nodes, c.strategy_count), dtype=np.float32)
        expected = self.reference.infer_raw(x, adjacency, masks, strategy_mask)
        if runtime.compiled is not None:
            result = runtime.compiled({"features": x, "adjacency": adjacency, "node_mask": masks})
            for index in (0, 1):
                if not np.allclose(np.asarray(result[runtime.compiled.output(index)]), expected[index], rtol=.01, atol=.03):
                    raise ValueError("temporal R-GCN accelerator raw-output parity failed")
        return runtime

    def infer(self, x, adjacency, node_mask, strategy_mask):
        if self._pending is not None and self._pending.done():
            try:
                self._runtime = self._pending.result()
            except Exception as exc:
                self._error = f"R-GCN accelerator unavailable: {type(exc).__name__}: {exc}"
            self._pending = None
        if self._runtime is not None:
            try:
                return self._runtime.infer(x, adjacency, node_mask, strategy_mask)
            except Exception as exc:
                self._error = f"R-GCN inference failed: {type(exc).__name__}: {exc}"
                self._runtime = None
        return self.reference.infer(x, adjacency, node_mask, strategy_mask)


def benchmark_runtime(
    runtime: OpenVinoStrategyUtilityRuntime,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    warmup: int = 3,
    iterations: int = 20,
) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        runtime.infer(*inputs)
    samples: list[float] = []
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        runtime.infer(*inputs)
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    return {
        **asdict(runtime.status),
        "iterations": len(samples),
        "p50_ms": median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "p99_ms": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
        "throughput_per_second": 1000 / (sum(samples) / len(samples)),
    }


def _build_graph(reference: FixedShapeStrategyUtilityModel):
    import openvino as ov

    c = reference.config
    ops = ov.opset13
    x = ops.parameter(
        [c.batch_size, c.time_steps, c.max_nodes, c.feature_dim],
        ov.Type.f32,
        name="features",
    )
    adjacency = ops.parameter(
        [
            c.batch_size,
            c.time_steps,
            c.relation_count,
            c.max_nodes,
            c.max_nodes,
        ],
        ov.Type.f32,
        name="adjacency",
    )
    node_mask = ops.parameter(
        [c.batch_size, c.time_steps, c.max_nodes],
        ov.Type.f32,
        name="node_mask",
    )
    axis_one = ops.constant(np.array(1, dtype=np.int64))
    temporal = None
    for time_index in range(c.time_steps):
        index = ops.constant(np.array(time_index, dtype=np.int64))
        xt = ops.gather(x, index, axis_one)
        hidden = ops.matmul(xt, ops.constant(reference.self_weight), False, False)
        adjacency_t = ops.gather(adjacency, index, axis_one)
        for relation_index in range(c.relation_count):
            relation = ops.constant(np.array(relation_index, dtype=np.int64))
            adjacency_r = ops.gather(adjacency_t, relation, axis_one)
            message = ops.matmul(adjacency_r, xt, False, False)
            projected = ops.matmul(
                message,
                ops.constant(reference.relation_weights[relation_index]),
                False,
                False,
            )
            hidden = ops.add(hidden, projected)
        hidden = ops.relu(hidden)
        mask_t = ops.gather(node_mask, index, axis_one)
        mask_t = ops.unsqueeze(
            mask_t, ops.constant(np.array([-1], dtype=np.int64))
        )
        hidden = ops.multiply(hidden, mask_t)
        weighted = ops.multiply(
            hidden, ops.constant(np.array(reference.temporal_weights[time_index], np.float32))
        )
        temporal = weighted if temporal is None else ops.add(temporal, weighted)
    heads = []
    for strategy_index in range(c.strategy_count):
        head = ops.matmul(
            temporal,
            ops.constant(reference.strategy_heads[strategy_index]),
            False,
            False,
        )
        heads.append(
            ops.unsqueeze(head, ops.constant(np.array([2], dtype=np.int64)))
        )
    raw = ops.concat(heads, 2)
    no_trade = ops.matmul(
        temporal,
        ops.constant(reference.no_trade_head[:, None]),
        False,
        False,
    )
    no_trade = ops.squeeze(
        no_trade, ops.constant(np.array([-1], dtype=np.int64))
    )
    raw.set_friendly_name("strategy_raw")
    no_trade.set_friendly_name("no_trade_raw")
    return ov.Model([raw, no_trade], [x, adjacency, node_mask], "strategy_utility_rgcn")


def _model_hash(reference: FixedShapeStrategyUtilityModel) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(reference.config), sort_keys=True).encode())
    for value in (
        reference.relation_weights,
        reference.self_weight,
        reference.strategy_heads,
        reference.no_trade_head,
        reference.temporal_weights,
    ):
        digest.update(value.tobytes())
    return digest.hexdigest()
