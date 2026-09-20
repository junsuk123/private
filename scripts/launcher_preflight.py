#!/usr/bin/env python3
"""Offline, read-only launcher checks. Stdout contains exactly one JSON report.

This module imports no application module except app.paths. It never constructs
stores, loads credentials, starts workers, downloads models or creates folders.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_FILES = (
    "trading_core.ttl", "trading_rules.ttl", "trading_shapes.ttl",
    "macro_market_ontology.ttl", "micro_symbol_ontology.ttl",
    "policy_ontology.ttl", "policy_shapes.ttl",
)
IMPORT_NAMES = {"exchange-calendars": "exchange_calendars", "pyyaml": "yaml"}


@contextmanager
def _quiet_probes():
    """Suppress both Python and native optional-library startup output."""
    saved = []
    with open(os.devnull, "w", encoding="utf-8") as sink:
        try:
            for descriptor in (1, 2):
                try:
                    duplicate = os.dup(descriptor)
                    os.dup2(sink.fileno(), descriptor)
                    saved.append((descriptor, duplicate))
                except OSError:
                    # Embedded hosts may not expose the ordinary console fds.
                    pass
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            for descriptor, duplicate in reversed(saved):
                os.dup2(duplicate, descriptor)
                os.close(duplicate)


def _environment_defaults(paths: Any) -> dict[str, str]:
    # Path resolution only: do not instantiate the artifact registry, whose
    # constructor creates its directory. The filename matches its actual default.
    store = paths.runtime_database_path("realtime_market_data.sqlite3").parent
    runtime = store.parent / "runtime"

    def explicit_or(name: str, default: Path) -> str:
        value = os.getenv(name, "").strip()
        return str((paths.project_path(value) if value else default).resolve())

    return {
        "REALTIME_STORE_ROOT": str(store),
        "REALTIME_MARKET_DATA_DB": str(paths.realtime_market_database_path()),
        "DIRECTIONAL_SHADOW_STORE_PATH": str(paths.runtime_database_path(
            "directional-shadow.sqlite3", env_var="DIRECTIONAL_SHADOW_STORE_PATH")),
        "REFACTOR_GNN_CHECKPOINT": str(paths.runtime_database_path(
            "models/strategy_utility/temporal_rgcn.npz", env_var="REFACTOR_GNN_CHECKPOINT")),
        "LIVE_MODEL_ARTIFACT_ROOT": str(paths.runtime_database_path(
            "models/live_short_horizon", env_var="LIVE_MODEL_ARTIFACT_ROOT")),
        "OPENVINO_CACHE_DIR": explicit_or("OPENVINO_CACHE_DIR", runtime / "openvino_cache"),
        "OBAITS_LAUNCHER_RUNTIME_DIR": explicit_or("OBAITS_LAUNCHER_RUNTIME_DIR", runtime / "browser"),
    }


def _dependency_modules(project_root: Path, importer) -> list[tuple[str, str]]:
    tomllib = importer("tomllib")
    raw = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8-sig"))
    dependencies = raw["project"]["dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("base dependency list is absent")
    result = []
    for requirement in dependencies:
        matched = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", str(requirement))
        if matched is None:
            raise ValueError("base dependency name is invalid")
        package = matched.group()
        result.append((package, IMPORT_NAMES.get(package.lower(), package.replace("-", "_"))))
    return result


def _check_resources(project_root: Path, modules: dict, errors: list[str]) -> None:
    for filename in ONTOLOGY_FILES:
        try:
            source = (project_root / "src/app/ontology" / filename).read_text(encoding="utf-8-sig")
            if "rdflib" in modules:
                # Explicit local text + Turtle never resolves OWL imports or URLs.
                graph = modules["rdflib"].Graph().parse(data=source, format="turtle")
                if not len(graph):
                    raise ValueError("empty ontology graph")
        except Exception as exc:
            errors.append(f"ONTOLOGY_RESOURCE_INVALID:{filename}:{type(exc).__name__}")
    try:
        source = (project_root / "config/ontology_risk_policy.yaml").read_text(encoding="utf-8-sig")
        if "yaml" in modules:
            policy = modules["yaml"].safe_load(source)
            if not isinstance(policy, dict) or not policy or policy.get("schema_version") != 1:
                raise ValueError("invalid risk policy schema")
            for name, value in policy.items():
                if str(name).startswith("maximum_"):
                    numeric = float(value)
                    if not math.isfinite(numeric) or numeric <= 0:
                        raise ValueError("invalid risk policy ceiling")
    except Exception as exc:
        # Do not include parser messages: those can contain configuration values.
        errors.append(f"ONTOLOGY_RISK_POLICY_INVALID:{type(exc).__name__}")


def _probe_devices(importer, warnings: list[str]) -> dict:
    devices = {
        "cpu": True,
        "openvino": {"installed": False, "available_devices": [], "probe_ok": False},
        "torch": {"installed": False, "cuda_available": False, "cuda_device_count": 0},
    }
    try:
        module = importer("openvino")
        devices["openvino"]["installed"] = True
        available = tuple(str(item) for item in module.Core().available_devices)
        devices["openvino"].update(available_devices=list(available), probe_ok=True)
        if not any(item.startswith(("NPU", "GPU")) for item in available):
            warnings.append("OPENVINO_ACCELERATOR_UNAVAILABLE_CPU_FALLBACK")
    except ImportError:
        warnings.append("OPENVINO_UNAVAILABLE_CPU_FALLBACK")
    except Exception as exc:
        warnings.append(f"OPENVINO_PROBE_FAILED_CPU_FALLBACK:{type(exc).__name__}")
    try:
        module = importer("torch")
        devices["torch"]["installed"] = True
        available = bool(module.cuda.is_available())
        count = max(0, int(module.cuda.device_count())) if available else 0
        devices["torch"].update(cuda_available=available and count > 0, cuda_device_count=count)
        if not available or count == 0:
            warnings.append("TORCH_CUDA_UNAVAILABLE_CPU_FALLBACK")
    except ImportError:
        warnings.append("TORCH_UNAVAILABLE_CPU_FALLBACK")
    except Exception as exc:
        warnings.append(f"TORCH_CUDA_PROBE_FAILED_CPU_FALLBACK:{type(exc).__name__}")
    return devices


def build_report(*, project_root: Path | None = None, importer=None, version_info=None) -> dict:
    """Return the report without changing the environment or runtime files."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    importer = importer or importlib.import_module
    version = tuple(version_info or sys.version_info[:3])
    errors: list[str] = []
    dependency_errors: list[str] = []
    warnings: list[str] = []
    report = {
        "ok": False, "errors": errors, "dependency_errors": dependency_errors,
        "warnings": warnings,
        "python": {"version": ".".join(map(str, version)), "executable": sys.executable,
                   "required": ">=3.11", "supported": version[:2] >= (3, 11)},
        "devices": {}, "environment_defaults": {},
    }
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with _quiet_probes():
            if not report["python"]["supported"]:
                errors.append("PYTHON_VERSION_UNSUPPORTED:requires>=3.11")
            try:
                report["environment_defaults"] = _environment_defaults(importer("app.paths"))
            except Exception as exc:
                errors.append(f"APPLICATION_PATHS_UNAVAILABLE:{type(exc).__name__}")
            try:
                dependencies = _dependency_modules(root, importer)
            except Exception as exc:
                errors.append(f"PYPROJECT_DEPENDENCIES_INVALID:{type(exc).__name__}")
                dependencies = []
            modules = {}
            for package, name in dependencies:
                try:
                    modules[name] = importer(name)
                except Exception as exc:
                    issue = f"DEPENDENCY_IMPORT_FAILED:{package}:{type(exc).__name__}"
                    dependency_errors.append(issue)
                    errors.append(issue)
            _check_resources(root, modules, errors)
            report["devices"] = _probe_devices(importer, warnings)
    finally:
        sys.dont_write_bytecode = old_bytecode
    report["ok"] = not errors
    return report


def main() -> int:
    # A direct script invocation starts with scripts/ on sys.path, not src/.
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        report = build_report()
    except Exception as exc:
        report = {"ok": False, "errors": [f"PREFLIGHT_FAILED:{type(exc).__name__}"],
                  "dependency_errors": [], "warnings": [], "python": {"executable": sys.executable},
                  "devices": {}, "environment_defaults": {}}
    # ASCII escaping makes the single JSON line safe under legacy Windows pipes.
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":")))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
