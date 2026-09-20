"""Synthetic device microbenchmark. No broker, registry, market feed, or model promotion."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.npu.runtime_manager import NpuRuntimeManager


def benchmark(*, device: str = "AUTO", iterations: int = 100) -> dict:
    iterations = max(1, min(10000, int(iterations)))
    devices = {}
    openvino_version = None
    try:
        import openvino as ov
        openvino_version = ov.__version__
        core = ov.Core()
        devices = {name: str(core.get_property(name, "FULL_DEVICE_NAME")) for name in core.available_devices}
    except ImportError:
        pass
    nvidia = []
    utility = shutil.which("nvidia-smi")
    if utility:
        result = subprocess.run([utility, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            nvidia = result.stdout.strip().splitlines()
    feature_dim = len(LIVE_SHORT_HORIZON_SCHEMA.feature_names)
    rng = np.random.default_rng(11)
    parameters = {
        "mean": np.zeros(feature_dim, dtype=np.float32), "scale": np.ones(feature_dim, dtype=np.float32),
        "hidden_weights": rng.normal(0.0, 0.1, (feature_dim, 64)).astype(np.float32),
        "hidden_bias": np.zeros(64, dtype=np.float32),
        "output_weights": rng.normal(0.0, 0.1, (64, 2)).astype(np.float32),
        "output_bias": np.zeros(2, dtype=np.float32),
    }
    previous_root = os.environ.get("REALTIME_STORE_ROOT")
    try:
        with tempfile.TemporaryDirectory(prefix="obaits-synthetic-benchmark-") as cache_root:
            os.environ["REALTIME_STORE_ROOT"] = cache_root
            manager = NpuRuntimeManager(device_preference=device)
            started = time.perf_counter()
            compiled, backend, reason = manager._compile_network("synthetic_benchmark", 1, parameters)
            compile_ms = (time.perf_counter() - started) * 1000.0
            features = rng.normal(size=(1, feature_dim)).astype(np.float32)
            compiled([features])
            latencies = []
            for _ in range(iterations):
                started = time.perf_counter()
                output = np.asarray(compiled([features])[0], dtype=np.float32)
                latencies.append((time.perf_counter() - started) * 1000.0)
            expected = np.maximum(np.clip(features, -6, 6) @ parameters["hidden_weights"], 0) @ parameters["output_weights"]
            error = float(np.max(np.abs(output - expected)))
            result = {
                "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "synthetic model microbenchmark; excludes market feed, strategy, risk and broker latency",
                "openvino_version": openvino_version, "openvino_devices": devices,
                "nvidia_smi_devices": nvidia,
                "torch_installed": importlib.util.find_spec("torch") is not None,
                "requested_device": device, "actual_model_backend": backend, "fallback_reason": reason,
                "feature_dim": feature_dim, "hidden_units": 64, "output_dim": 2, "batch_size": 1,
                "parameter_count": sum(value.size for value in parameters.values()), "iterations": iterations,
                "cold_compile_ms": round(compile_ms, 3),
                "warm_inference_median_ms": round(float(np.median(latencies)), 3),
                "warm_inference_p95_ms": round(float(np.percentile(latencies, 95)), 3),
                "maximum_absolute_difference_vs_numpy": error,
            }
            manager.close()
            del compiled
            return result
    finally:
        if previous_root is None:
            os.environ.pop("REALTIME_STORE_ROOT", None)
        else:
            os.environ["REALTIME_STORE_ROOT"] = previous_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="AUTO", choices=("AUTO", "NPU", "CUDA", "GPU", "CPU"))
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(benchmark(device=args.device, iterations=args.iterations), ensure_ascii=False, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
