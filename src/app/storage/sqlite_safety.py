from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


RECOVERY_LOCK = threading.Lock()
SQLITE_JOURNAL_MODES = {"delete", "persist", "truncate", "wal"}
SYNC_DIRECTORY_NAMES = {
    "dropbox",
    "google drive",
    "googledrive",
    "onedrive",
    "synologydrive",
}


def is_sqlite_corruption(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "database disk image is malformed",
            "database corruption",
            "file is not a database",
        )
    )


def journal_mode_for_path(path: Path, *, override_env: str | None = None) -> str:
    """Choose a cloud-safe journal while keeping WAL for normal local disks."""

    configured = ""
    if override_env:
        configured = os.getenv(override_env, "").strip().lower()
    if not configured:
        configured = os.getenv("OBAITS_SQLITE_JOURNAL_MODE", "").strip().lower()
    if configured:
        if configured not in SQLITE_JOURNAL_MODES:
            choices = ", ".join(sorted(SQLITE_JOURNAL_MODES))
            variable = override_env or "OBAITS_SQLITE_JOURNAL_MODE"
            raise ValueError(
                f"Unsupported {variable}={configured!r}; expected one of: {choices}"
            )
        return configured

    # WAL is a coordinated database/WAL/SHM set. Cloud-drive clients upload
    # those files independently, so a rollback journal is the safer default in
    # a synced code space. On ordinary local disks WAL remains the default.
    resolved_parts = path.resolve().parts
    if any(part.casefold() in SYNC_DIRECTORY_NAMES for part in resolved_parts):
        return "delete"
    return "wal"


def quarantine_sqlite_files(path: Path) -> tuple[Path, ...]:
    """Move a corrupt database and its sidecars aside without deleting evidence."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantined: list[Path] = []
    for source in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not source.exists():
            continue
        suffix = source.name.removeprefix(path.name)
        target = path.parent / f"{path.name}.corrupt.{stamp}{suffix}"
        os.replace(source, target)
        quarantined.append(target)
    return tuple(quarantined)
