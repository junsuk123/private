from __future__ import annotations

from pathlib import Path


def short_horizon_model_export_status(model_path: str = "models/short_horizon/openvino_model.xml") -> dict[str, str | bool]:
    path = Path(model_path)
    return {
        "model_path": str(path),
        "exists": path.exists(),
        "format": "openvino_ir" if path.suffix == ".xml" else "unknown",
    }
