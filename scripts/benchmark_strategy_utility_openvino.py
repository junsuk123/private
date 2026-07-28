from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.models.strategy_utility import (  # noqa: E402
    FixedShapeStrategyUtilityModel,
    StrategyUtilityModelConfig,
)
from app.models.strategy_utility.openvino_runtime import (  # noqa: E402
    OpenVinoStrategyUtilityRuntime,
    benchmark_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/reports/strategy_utility_openvino.json")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    config = StrategyUtilityModelConfig(1, 4, 16, 12, 4, 7, hidden_dim=16, seed=17)
    model = FixedShapeStrategyUtilityModel(config)
    inputs = _inputs(config)
    cpu = OpenVinoStrategyUtilityRuntime(model, requested_device="CPU")
    npu = OpenVinoStrategyUtilityRuntime(model, requested_device="NPU")
    cpu_output = cpu.infer(*inputs)
    npu_output = npu.infer(*inputs)
    finite = np.isfinite(cpu_output.utility) & np.isfinite(npu_output.utility)
    max_error = float(
        np.max(np.abs(cpu_output.utility[finite] - npu_output.utility[finite]))
    )
    cpu_rank = np.argmax(cpu_output.utility, axis=-1)
    npu_rank = np.argmax(npu_output.utility, axis=-1)
    payload = {
        "cpu": benchmark_runtime(cpu, inputs, iterations=args.iterations),
        "npu": benchmark_runtime(npu, inputs, iterations=args.iterations),
        "parity": {
            "max_abs_utility_error": max_error,
            "top1_agreement": float(np.mean(cpu_rank == npu_rank)),
            "no_trade_max_abs_error": float(
                np.max(
                    np.abs(
                        cpu_output.no_trade_probability
                        - npu_output.no_trade_probability
                    )
                )
            ),
        },
        "promotion_eligible": (
            "NPU" in npu.status.compiled_devices
            and max_error <= 1e-3
            and float(np.mean(cpu_rank == npu_rank)) == 1.0
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _inputs(config: StrategyUtilityModelConfig):
    rng = np.random.default_rng(29)
    x = rng.normal(
        size=(
            config.batch_size,
            config.time_steps,
            config.max_nodes,
            config.feature_dim,
        )
    ).astype(np.float32)
    adjacency = np.zeros(
        (
            config.batch_size,
            config.time_steps,
            config.relation_count,
            config.max_nodes,
            config.max_nodes,
        ),
        dtype=np.float32,
    )
    for relation in range(config.relation_count):
        adjacency[:, :, relation, np.arange(config.max_nodes), np.arange(config.max_nodes)] = 1
        adjacency[:, :, relation, np.arange(config.max_nodes - 1), np.arange(1, config.max_nodes)] = 0.25
    node_mask = np.ones(
        (config.batch_size, config.time_steps, config.max_nodes), dtype=np.float32
    )
    node_mask[:, :, -2:] = 0
    strategy_mask = np.ones(
        (config.batch_size, config.max_nodes, config.strategy_count), dtype=np.float32
    )
    strategy_mask[:, 0, -1] = 0
    return x, adjacency, node_mask, strategy_mask


if __name__ == "__main__":
    raise SystemExit(main())
