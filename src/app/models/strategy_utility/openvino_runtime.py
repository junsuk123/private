from __future__ import annotations

import hashlib
import json
import time
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
        import openvino as ov

        self.reference = reference
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
        result = self.compiled(
            {
                "features": x.astype(np.float32, copy=False),
                "adjacency": adjacency.astype(np.float32, copy=False),
                "node_mask": node_mask.astype(np.float32, copy=False),
            }
        )
        raw = np.asarray(result[self.compiled.output(0)])
        no_trade_raw = np.asarray(result[self.compiled.output(1)])
        return output_from_raw(raw, no_trade_raw, node_mask, strategy_mask)


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
    ):
        digest.update(value.tobytes())
    return digest.hexdigest()
