from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import rdflib
import yaml

from app import paths


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launcher_preflight", ROOT / "scripts/launcher_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
PATH_KEYS = (
    "REALTIME_STORE_ROOT", "REALTIME_MARKET_DATA_DB", "DIRECTIONAL_SHADOW_STORE_PATH",
    "REFACTOR_GNN_CHECKPOINT", "LIVE_MODEL_ARTIFACT_ROOT", "OPENVINO_CACHE_DIR",
    "OBAITS_LAUNCHER_RUNTIME_DIR",
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "SynologyDrive" / "연구 프로젝트"
    (root / "src/app/ontology").mkdir(parents=True)
    (root / "config").mkdir()
    shutil.copyfile(ROOT / "pyproject.toml", root / "pyproject.toml")
    for filename in preflight.ONTOLOGY_FILES:
        shutil.copyfile(ROOT / "src/app/ontology" / filename, root / "src/app/ontology" / filename)
    shutil.copyfile(ROOT / "config/ontology_risk_policy.yaml", root / "config/ontology_risk_policy.yaml")
    monkeypatch.setattr(paths, "PROJECT_ROOT", root)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "machine-local"))
    for key in PATH_KEYS:
        monkeypatch.delenv(key, raising=False)
    return root


def importer(*, missing=(), devices=("CPU", "NPU"), cuda=True, noisy=False, broken_probe=None):
    seen = []

    def load(name):
        seen.append(name)
        if noisy:
            print("library startup must not be stdout")
            os.write(1, b"native library startup must not be stdout\n")
        if name in missing:
            raise ImportError("credential-value-must-not-be-printed")
        if name == "app.paths":
            return paths
        if name == "tomllib":
            return importlib.import_module("tomllib")
        if name == "yaml":
            return yaml
        if name == "rdflib":
            return rdflib
        if name == "openvino":
            def core():
                if broken_probe == "openvino":
                    raise RuntimeError("credential-value-must-not-be-printed")
                return SimpleNamespace(available_devices=devices)
            return SimpleNamespace(Core=core)
        if name == "torch":
            def available():
                if broken_probe == "torch":
                    raise RuntimeError("credential-value-must-not-be-printed")
                return cuda
            return SimpleNamespace(cuda=SimpleNamespace(is_available=available, device_count=lambda: int(cuda)))
        assert not name.startswith("app."), "No workers, stores, web or model registry may be imported"
        return SimpleNamespace()

    return load, seen


def test_offline_readonly_success_resolves_machine_local_paths(project, monkeypatch):
    monkeypatch.setenv("KIS_APP_SECRET", "credential-value-must-not-be-printed")
    load, seen = importer()
    before_files = set(project.parent.parent.rglob("*"))
    before_environment = dict(os.environ)
    report = preflight.build_report(project_root=project, importer=load)
    assert report["ok"] and report["errors"] == report["dependency_errors"] == []
    assert report["devices"]["openvino"]["available_devices"] == ["CPU", "NPU"]
    assert report["devices"]["torch"]["cuda_available"] is True
    defaults = report["environment_defaults"]
    assert set(defaults) == set(PATH_KEYS)
    assert all(Path(value).is_absolute() for value in defaults.values())
    assert not any(Path(value).is_relative_to(project) for value in defaults.values())
    assert defaults["LIVE_MODEL_ARTIFACT_ROOT"] == str(paths.runtime_database_path("models/live_short_horizon"))
    assert defaults["REALTIME_MARKET_DATA_DB"] == str(paths.realtime_market_database_path())
    assert set(project.parent.parent.rglob("*")) == before_files
    assert dict(os.environ) == before_environment
    assert "yaml" in seen and "exchange_calendars" in seen and "pyshacl" in seen
    assert {name for name in seen if name.startswith("app.")} == {"app.paths"}
    assert "credential-value-must-not-be-printed" not in json.dumps(report)


def test_explicit_relative_paths_are_canonical_and_not_created(project, monkeypatch):
    for index, key in enumerate(PATH_KEYS):
        monkeypatch.setenv(key, f"custom/선택-{index}")
    load, _ = importer()
    report = preflight.build_report(project_root=project, importer=load)
    assert report["ok"]
    for index, key in enumerate(PATH_KEYS):
        assert report["environment_defaults"][key] == str((project / f"custom/선택-{index}").resolve())
    assert not (project / "custom").exists()


def test_missing_optional_accelerators_are_warnings_with_cpu_fallback(project):
    load, _ = importer(missing=("openvino", "torch"))
    report = preflight.build_report(project_root=project, importer=load)
    assert report["ok"] and not report["dependency_errors"]
    assert report["devices"]["cpu"] is True
    assert len(report["warnings"]) == 2
    assert all("CPU_FALLBACK" in warning for warning in report["warnings"])


@pytest.mark.parametrize("probe", ["openvino", "torch"])
def test_broken_accelerator_probe_is_sanitized_warning(project, probe):
    load, _ = importer(broken_probe=probe)
    report = preflight.build_report(project_root=project, importer=load)
    assert report["ok"] and any("PROBE_FAILED_CPU_FALLBACK" in warning for warning in report["warnings"])
    assert "credential-value-must-not-be-printed" not in json.dumps(report)


def test_missing_required_dependencies_are_distinguished_for_setup(project):
    load, _ = importer(missing=("yaml", "owlrl"))
    report = preflight.build_report(project_root=project, importer=load)
    assert not report["ok"] and len(report["dependency_errors"]) == 2
    assert set(report["dependency_errors"]) <= set(report["errors"])
    assert any("PyYAML" in error for error in report["dependency_errors"])


@pytest.mark.parametrize("relative,value", [
    ("src/app/ontology/policy_shapes.ttl", "this is invalid Turtle"),
    ("src/app/ontology/trading_core.ttl", ""),
    ("config/ontology_risk_policy.yaml", "[credential-value-must-not-be-printed"),
    ("config/ontology_risk_policy.yaml", "schema_version: 1\nmaximum_trade_loss_rate: .nan"),
])
def test_bad_resources_are_configuration_errors_not_install_requests(project, relative, value):
    (project / relative).write_text(value, encoding="utf-8")
    load, _ = importer()
    report = preflight.build_report(project_root=project, importer=load)
    assert not report["ok"] and not report["dependency_errors"]
    assert "credential-value-must-not-be-printed" not in json.dumps(report)


def test_missing_ontology_and_unsupported_python_fail_without_install_request(project):
    (project / "src/app/ontology/policy_ontology.ttl").unlink()
    load, _ = importer()
    report = preflight.build_report(project_root=project, importer=load, version_info=(3, 10, 14))
    assert not report["ok"] and report["python"]["supported"] is False
    assert any("PYTHON_VERSION_UNSUPPORTED" in error for error in report["errors"])
    assert any("policy_ontology.ttl:FileNotFoundError" in error for error in report["errors"])
    assert report["dependency_errors"] == []


def test_probe_suppresses_python_and_native_output_and_restores_bytecode_flag(project, capfd):
    load, _ = importer(noisy=True)
    previous = sys.dont_write_bytecode
    assert preflight.build_report(project_root=project, importer=load)["ok"]
    assert sys.dont_write_bytecode is previous
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("okay,expected_exit", [(True, 0), (False, 1)])
def test_cli_emits_exactly_one_json_report_and_expected_exit(monkeypatch, capsys, okay, expected_exit):
    report = {"ok": okay, "errors": [] if okay else ["TEST_FAILURE"], "environment_defaults": {"path": "한글 경로"}}
    monkeypatch.setattr(preflight, "build_report", lambda: report)
    assert preflight.main() == expected_exit
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1 and json.loads(captured.out) == report
    assert captured.err == ""
