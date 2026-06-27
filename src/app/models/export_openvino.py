from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpenVinoExportResult:
    exported: bool
    output_path: str | None
    reason: str | None = None


def export_openvino_if_available(model: Any, output_dir: Path, model_name: str = "signal_model") -> OpenVinoExportResult:
    try:
        import openvino as ov  # noqa: F401
    except ModuleNotFoundError as exc:
        return OpenVinoExportResult(False, None, f"OpenVINO unavailable: {exc}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return OpenVinoExportResult(
        False,
        str(output_dir / f"{model_name}.xml"),
        "Export hook available, but concrete model conversion must be supplied by the trained model adapter.",
    )
