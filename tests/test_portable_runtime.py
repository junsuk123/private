from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.npu.runtime_manager import NpuRuntimeManager
from app.paths import (
    PROJECT_ROOT,
    project_path,
    realtime_market_database_path,
    runtime_store_root,
)
from app.realtime.device_plan import CPU, GPU, DeviceInventory, plan_devices
from app.realtime.acceleration import RealtimeAccelerationPolicy


def test_project_path_is_rooted_in_codespace() -> None:
    assert project_path("config/example.json") == PROJECT_ROOT / "config" / "example.json"
    absolute = PROJECT_ROOT / "data"
    assert project_path(absolute) == absolute


def test_synced_workspace_uses_a_machine_local_store(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("REALTIME_STORE_ROOT", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setattr(
        "app.paths.PROJECT_ROOT", tmp_path / "SynologyDrive" / "project"
    )

    root = runtime_store_root()
    assert root.parent.parent == tmp_path / "local-app-data" / "OBAITS"
    assert root.name == "store"
    assert root.parent.name.startswith("project-")


def test_explicit_runtime_store_root_wins(monkeypatch) -> None:
    monkeypatch.setenv("REALTIME_STORE_ROOT", "data/custom-store")
    assert runtime_store_root() == Path("data/custom-store")


def test_realtime_database_is_outside_a_synced_workspace(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("REALTIME_MARKET_DATA_DB", raising=False)
    monkeypatch.delenv("REALTIME_STORE_ROOT", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setattr(
        "app.paths.PROJECT_ROOT", tmp_path / "SynologyDrive" / "project"
    )

    database = realtime_market_database_path()

    assert database.parent.parent.parent == tmp_path / "local-app-data" / "OBAITS"
    assert database.name == "realtime_market_data.sqlite3"
    assert "SynologyDrive" not in database.parts


def test_relative_realtime_database_override_is_project_relative(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("app.paths.PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setenv("REALTIME_MARKET_DATA_DB", "data/custom/realtime.sqlite3")

    assert realtime_market_database_path() == (
        tmp_path / "project/data/custom/realtime.sqlite3"
    ).resolve()


def test_cuda_only_gpu_is_used_only_by_a_compatible_workload() -> None:
    inventory = DeviceInventory(
        (CPU, GPU),
        {CPU: "CPU", GPU: "NVIDIA"},
        providers={CPU: ("python",), GPU: ("torch-cuda",)},
    )
    plan = {item.workload: item.device for item in plan_devices(inventory)}

    assert plan["event_classification"] == GPU
    assert plan["ontology_candidate_scorer"] == CPU
    assert plan["strategy_utility_rgcn_shadow"] == CPU


def test_openvino_npu_preference_falls_through_to_available_gpu() -> None:
    manager = NpuRuntimeManager(device_preference="NPU", min_batch_for_npu=1)
    manager._available_devices = ("CPU", "GPU")

    assert manager._requested_device(10, True) == "GPU"


def test_explicit_device_configuration_wins_over_hardware_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_EVENT_DEVICE", "cpu")
    monkeypatch.setenv("LLM_EVENT_INFERENCE_BACKEND", "transformers")

    RealtimeAccelerationPolicy().apply_process_hints()

    assert os.environ["LLM_EVENT_DEVICE"] == "cpu"
    assert os.environ["LLM_EVENT_INFERENCE_BACKEND"] == "transformers"


def test_root_entrypoint_works_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run.py"), "--help"],
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run the complete local investment system" in completed.stdout
