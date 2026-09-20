"""Stable project-relative paths shared by every runtime entry point."""

from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SYNC_DIRECTORY_NAMES = {
    "dropbox",
    "google drive",
    "googledrive",
    "onedrive",
    "synologydrive",
}


def project_path(value: str | os.PathLike[str] | Path) -> Path:
    """Return an absolute path, resolving relative values under the repository."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def runtime_store_root() -> Path:
    """Return a safe local store, outside cloud sync when the project is synced."""

    configured = os.getenv("REALTIME_STORE_ROOT", "").strip()
    if configured:
        return Path(configured)
    if not any(part.casefold() in _SYNC_DIRECTORY_NAMES for part in PROJECT_ROOT.parts):
        return Path("data/store")
    local_state = os.getenv("LOCALAPPDATA", "").strip() or os.getenv(
        "XDG_STATE_HOME", ""
    ).strip()
    state_root = Path(local_state) if local_state else Path.home() / ".local" / "state"
    project_name = re.sub(
        r"[^a-zA-Z0-9._-]+", "-", PROJECT_ROOT.name
    ).strip("-.") or "project"
    project_key = hashlib.sha256(
        str(PROJECT_ROOT.resolve()).casefold().encode("utf-8")
    ).hexdigest()[:12]
    return state_root / "OBAITS" / f"{project_name}-{project_key}" / "store"


def runtime_database_path(
    filename: str,
    *,
    env_var: str | None = None,
) -> Path:
    """Resolve one mutable SQLite database to this machine's runtime store.

    Source code and configuration may still name project-relative paths, but a
    cloud-synchronised checkout must never host a live SQLite WAL.  An explicit
    environment override wins and is resolved relative to the project; otherwise
    the per-machine runtime root is used.
    """

    configured = os.getenv(env_var, "").strip() if env_var else ""
    if configured:
        return project_path(configured).resolve()
    root = runtime_store_root()
    if not root.is_absolute():
        root = project_path(root)
    return (root / filename).resolve()


def realtime_market_database_path() -> Path:
    """Canonical database used by every realtime producer and consumer."""

    return runtime_database_path(
        "realtime_market_data.sqlite3",
        env_var="REALTIME_MARKET_DATA_DB",
    )


def activate_project_root() -> Path:
    """Make legacy relative defaults independent of the caller's working directory."""

    os.environ.setdefault("OBAITS_PROJECT_ROOT", str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("REALTIME_STORE_ROOT", str(runtime_store_root()))
    return PROJECT_ROOT
