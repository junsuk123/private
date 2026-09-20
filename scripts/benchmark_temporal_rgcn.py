"""Order-free, synthetic-data full trained R-GCN parity/latency benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.features.strategy_graph_context import STRATEGY_GRAPH_CONTEXT_DIM, STRATEGY_GRAPH_CONTEXT_SCHEMA, context_index
from app.models.strategy_utility.rgcn import FixedShapeStrategyUtilityModel
from app.models.strategy_utility.openvino_runtime import OpenVinoStrategyUtilityRuntime, benchmark_runtime
from app.models.strategy_utility.strategy_graph import diagonal_strategy_mask, strategy_ids_for_market
from app.models.strategy_utility.temporal_graph import ARCHITECTURE, CausalGraphHistory, GraphObservation
from app.models.strategy_utility.training import train_counterfactual_checkpoint
from app.strategy.catalog import STRATEGY_IDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import openvino as ov

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(883)
    rows, observations = [], []
    for i in range(64):
        values = rng.normal(0, .15, STRATEGY_GRAPH_CONTEXT_DIM).astype(np.float32)
        values[context_index("is_krx")] = 1
        as_of = start + timedelta(minutes=i * 3)
        observation = GraphObservation("900101", as_of, tuple(float(v) for v in values))
        observations.append(observation)
        for strategy in STRATEGY_IDS:
            active = strategy in {"intraday_momentum", "breakout_volume"}
            net = 30.0 if values[4] > 0 else -25.0
            rows.append(CounterfactualLabel(as_of=as_of, label_end=as_of + timedelta(seconds=60),
                symbol="900101", strategy_id=strategy, triggered=active, filled=active,
                net_return_bps=net if active else 0, cost_bps=12,
                exit_reason="TIME", features=observation.features))
    os.environ["GNN_TRAINING_MAX_STEPS"] = "24"
    with tempfile.TemporaryDirectory(prefix="obaits-rgcn-benchmark-") as root:
        path = Path(root) / "synthetic.npz"
        begin = time.perf_counter()
        training = train_counterfactual_checkpoint(rows, path, input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA)
        training_ms = (time.perf_counter() - begin) * 1000
        model = FixedShapeStrategyUtilityModel.load_checkpoint(path)
    history = CausalGraphHistory()
    for observation in observations[-3:]:
        x, adjacency, weights = history.inputs(observation)
    inputs = (x[None], adjacency[None], weights[None],
              diagonal_strategy_mask(strategy_ids_for_market("900101"))[None])
    reference = model.infer(*inputs)
    raw_reference = model.infer_raw(*inputs)
    runtime = OpenVinoStrategyUtilityRuntime(model, requested_device=args.device, allow_cpu_fallback=False)
    actual = runtime.infer(*inputs)
    raw = runtime.compiled({"features": inputs[0], "adjacency": inputs[1], "node_mask": inputs[2]})
    errors = [float(np.max(np.abs(np.asarray(raw[runtime.compiled.output(i)]) - raw_reference[i]))) for i in (0, 1)]
    net_reference = reference.gross_return_bps - reference.cost_bps
    net_actual = actual.gross_return_bps - actual.cost_bps
    eligible = inputs[3] > 0
    core = ov.Core()
    report = {
        "architecture": ARCHITECTURE,
        "data": "synthetic; no market performance or trading latency claim",
        "live_authorized": training["live_authorized"],
        "openvino_version": ov.__version__,
        "precision_scope": "FP32 inputs and constants; device internal precision is compiler-selected",
        "device_names": {device: core.get_property(device, "FULL_DEVICE_NAME") for device in core.available_devices},
        "trained_parameters": sum(value.size for value in (model.relation_weights, model.self_weight, model.strategy_heads, model.no_trade_head)),
        "shapes": {name: list(value.shape) for name, value in zip(("features", "adjacency", "time_masks", "strategy_masks"), inputs)},
        "training_ms": round(training_ms, 3),
        "training_steps": training["validation_metrics"]["gradient_steps"],
        "relation_weight_update_l2": training["validation_metrics"]["relation_weight_update_l2"],
        "raw_max_abs_error": max(errors),
        "eligible_expected_net_bps_max_abs_error": float(np.max(np.abs(net_reference[eligible] - net_actual[eligible]))),
        **benchmark_runtime(runtime, inputs, warmup=5, iterations=args.iterations),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
