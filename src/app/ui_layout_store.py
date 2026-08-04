"""Persistent frame layout for the strategy terminal.

The terminal's frames are drag-resizable AND drag-movable, and an operator who
spends a session arranging them should not lose that arrangement to a cleared
browser cache or a second machine, so the saved layout lives in ``data/ui/``
instead of only in localStorage. The browser is the only writer, which makes
every payload untrusted input: nothing survives normalisation except the shape
below, so a corrupted or hostile POST can neither grow the file without bound
nor feed arbitrary strings back into the page that renders it.

The shape is three levels deep - layers stack down the screen, columns sit
side by side inside a layer, frames stack down a column - which is the smallest
model that expresses "move this panel next to that one" and "stack these two".
v1 layouts (a flat ``panels`` list per layer) are still accepted on read and
upconverted, because an operator's saved arrangement must survive the upgrade
that added columns.

Sizes are stored as relative weights rather than pixels. A layout arranged on a
1080p laptop then has to mean the same thing on a 1440p desk monitor, which a
pixel geometry cannot promise.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "strategy_terminal_layout_v2"
LEGACY_SCHEMA = "strategy_terminal_layout_v1"
DEFAULT_LAYOUT_PATH = Path("data/ui/strategy_terminal_layout.json")

# Bounds exist to cap the file, not to express taste: a frame thinner than the
# minimum weight is invisible anyway, and a terminal with more than a dozen
# layers is a bug in the client rather than an operator preference.
MAX_LAYERS = 16
MAX_COLUMNS_PER_LAYER = 12
MAX_FRAMES_PER_COLUMN = 12
MIN_WEIGHT = 0.05
MAX_WEIGHT = 24.0
_FRAME_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class TerminalLayoutStore:
    """Reads and writes the one saved terminal layout for this deployment."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_LAYOUT_PATH

    def load(self) -> dict[str, Any]:
        """Return the stored layout, or an empty one when nothing is saved.

        A missing or unreadable file is not an error: the client falls back to
        its built-in default layout, which is exactly what a first-run browser
        should see.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return empty_layout()
        try:
            saved_at = raw.get("saved_at") if isinstance(raw, dict) else None
            return normalise_layout(raw, saved_at=saved_at)
        except ValueError:
            # A hand-edited or truncated file must not brick the terminal.
            return empty_layout()

    def save(self, payload: Any) -> dict[str, Any]:
        """Validate and persist ``payload``; raises ValueError when malformed."""
        document = normalise_layout(payload, saved_at=_utc_now())
        _atomic_write_text(self.path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        return document

    def clear(self) -> dict[str, Any]:
        """Forget the saved layout so every browser returns to the default."""
        try:
            self.path.unlink()
        except OSError:
            pass
        return empty_layout()


def empty_layout() -> dict[str, Any]:
    return {"schema": SCHEMA, "saved_at": None, "layers": []}


def normalise_layout(payload: Any, *, saved_at: str | None = None) -> dict[str, Any]:
    """Coerce an untrusted payload into the stored shape or raise ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("layout payload must be an object")
    layers_raw = payload.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ValueError("layout requires a non-empty 'layers' list")
    if len(layers_raw) > MAX_LAYERS:
        raise ValueError(f"layout accepts at most {MAX_LAYERS} layers")

    seen_keys: set[str] = set()
    layers: list[dict[str, Any]] = []
    for layer_raw in layers_raw:
        if not isinstance(layer_raw, dict):
            raise ValueError("each layer must be an object")
        columns_raw = _columns_of(layer_raw)
        if len(columns_raw) > MAX_COLUMNS_PER_LAYER:
            raise ValueError(f"a layer accepts at most {MAX_COLUMNS_PER_LAYER} columns")
        columns: list[dict[str, Any]] = []
        for column_raw in columns_raw:
            if not isinstance(column_raw, dict):
                raise ValueError("each column must be an object")
            frames_raw = column_raw.get("frames")
            if not isinstance(frames_raw, list) or not frames_raw:
                raise ValueError("each column requires a non-empty 'frames' list")
            if len(frames_raw) > MAX_FRAMES_PER_COLUMN:
                raise ValueError(f"a column accepts at most {MAX_FRAMES_PER_COLUMN} frames")
            frames: list[dict[str, Any]] = []
            for frame_raw in frames_raw:
                if not isinstance(frame_raw, dict):
                    raise ValueError("each frame must be an object")
                key = frame_raw.get("key")
                if not isinstance(key, str) or not _FRAME_KEY.match(key):
                    raise ValueError(f"invalid frame key: {key!r}")
                if key in seen_keys:
                    # A duplicated key would make the client place one panel twice.
                    raise ValueError(f"frame {key!r} appears more than once")
                seen_keys.add(key)
                frames.append({"key": key, "height": _weight(frame_raw.get("height"))})
            columns.append({"width": _weight(column_raw.get("width")), "frames": frames})
        layers.append({"height": _weight(layer_raw.get("height")), "columns": columns})

    return {"schema": SCHEMA, "saved_at": saved_at, "layers": layers}


def _columns_of(layer_raw: dict[str, Any]) -> list[Any]:
    """Return the layer's columns, upconverting a v1 flat panel list."""
    columns_raw = layer_raw.get("columns")
    if isinstance(columns_raw, list) and columns_raw:
        return columns_raw
    panels_raw = layer_raw.get("panels")
    if isinstance(panels_raw, list) and panels_raw:
        # v1: every panel was its own full-height column.
        return [
            {
                "width": panel.get("width") if isinstance(panel, dict) else None,
                "frames": [{"key": panel.get("key") if isinstance(panel, dict) else panel, "height": 1}],
            }
            for panel in panels_raw
        ]
    raise ValueError("each layer requires a non-empty 'columns' list")


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"size weight must be a number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("size weight must be finite")
    return round(min(max(number, MIN_WEIGHT), MAX_WEIGHT), 4)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    # Same pattern as the model registry: a half-written layout file would be
    # read back as corrupt and silently reset the operator's arrangement.
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
