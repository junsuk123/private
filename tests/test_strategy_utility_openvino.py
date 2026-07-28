from __future__ import annotations

import numpy as np

from app.models.strategy_utility import (
    FixedShapeStrategyUtilityModel,
    StrategyUtilityModelConfig,
)
from app.models.strategy_utility.openvino_runtime import (
    OpenVinoStrategyUtilityRuntime,
    benchmark_runtime,
)


def test_openvino_cpu_matches_numpy_golden_vectors() -> None:
    model = FixedShapeStrategyUtilityModel(
        StrategyUtilityModelConfig(1, 2, 3, 4, 2, 7, hidden_dim=6, seed=4)
    )
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    adjacency = np.zeros((1, 2, 2, 3, 3), dtype=np.float32)
    adjacency[:, :, :, np.arange(3), np.arange(3)] = 1
    node_mask = np.ones((1, 2, 3), dtype=np.float32)
    strategy_mask = np.ones((1, 3, 7), dtype=np.float32)
    strategy_mask[0, 2, 6] = 0
    numpy_output = model.infer(x, adjacency, node_mask, strategy_mask)
    runtime = OpenVinoStrategyUtilityRuntime(model, requested_device="CPU")
    openvino_output = runtime.infer(x, adjacency, node_mask, strategy_mask)
    np.testing.assert_allclose(openvino_output.utility, numpy_output.utility, rtol=1e-5, atol=1e-5)
    assert runtime.status.compiled_devices == ("CPU",)
    metrics = benchmark_runtime(
        runtime, (x, adjacency, node_mask, strategy_mask), warmup=1, iterations=3
    )
    assert metrics["p95_ms"] > 0
    assert metrics["model_hash"]
