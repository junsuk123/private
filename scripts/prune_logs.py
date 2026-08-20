#!/usr/bin/env python3
"""Bound the size of ``logs/`` without touching anything a writer still owns.

WHY THIS EXISTS, AND WHAT IT DELIBERATELY DOES NOT DO

Most journals in this tree already cap themselves: AuditLogger and
DecisionLogger keep ``.1``-``.N`` backups, ShadowComparisonRecorder keeps the
newest N immutable ``.r<ns>-<pid>-<tid>`` segments, and the feature journal
discards itself when oversized. Those bounds are correct and this script does
not second-guess them. What no writer owns is the residue:

  * segments orphaned when a process died between rotating and pruning --
    ShadowComparisonRecorder's own cleanup is explicitly best effort, so a
    crash leaves 256 MB segments behind that nothing will ever revisit;
  * one-off run logs (``server-20260728-005420.stdout.log`` and 178 siblings)
    written once by a launch that has long since exited;
  * ``logs/archive/`` compressed exports, which no code in this repo writes or
    expires;
  * ``.tmp`` scratch files left by interrupted syncs.

It also does NOT prune data/store. Those SQLite files look like the obvious
target at 11 GB, but their freelists measure ~0%: the rows are live, retained
on purpose, and already pruned hourly by RealtimeStore. Deleting there would
destroy real data to reclaim nothing.

THE SAFETY RULE THAT MATTERS

An active journal is never a candidate. Two independent guards enforce that,
because getting this wrong silently truncates a live trading audit trail:

  1. Anything modified more recently than --min-age-days is skipped outright.
     Every active journal is written continuously, so this alone excludes them.
  2. Files held open by a running process are skipped. Unlinking one would not
     free the space until the writer exits, and any output it appends after the
     unlink is written into a file with no name -- lost without an error.

Retention of rotated segments is by count, matching the writers' own scheme, so
this script and the writers agree on what "keep the newest N" means.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"

#: A rotated, immutable segment: ``name.jsonl.1`` / ``.r<ns>-<pid>-<tid>`` /
#: ``.rotated`` / ``.oversized``. The base name is what the writer still owns.
ROTATED = re.compile(r"^(?P<base>.+?)\.(?:\d+|r\d+-\d+-\d+|rotated|oversized|bak)$")

#: Logs written once by a launch that has exited. Matched by name, then still
#: subject to the age and open-file guards below.
ONE_OFF = (
    re.compile(r"^server-.*\.(?:out|err|stdout|stderr)\.log$"),
    re.compile(r"^server-\d{8}-\d{6}\..*\.log$"),
    re.compile(r"^server-restart-.*\.log$"),
    re.compile(r"^uvicorn-.*\.log$"),
    re.compile(r"^.*\.tmp$"),
)


def open_files() -> set[Path]:
    """Absolute paths currently held open by any process we can inspect.

    Best effort by necessity: /proc entries for other users' processes are
    unreadable, and a process can open a file a microsecond after we look. It
    is a guard against the common case (the live server writing its journal),
    layered on top of the age rule -- not a lock.
    """
    held: set[Path] = set()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            for fd in fd_dir.iterdir():
                try:
                    held.add(fd.resolve(strict=True))
                except (OSError, RuntimeError):
                    continue
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
    return held


def collect(min_age_days: float, keep_segments: int, archive_days: float) -> list[tuple[Path, str]]:
    """Every deletion candidate, paired with the rule that selected it."""
    if not LOGS.is_dir():
        return []
    now = time.time()
    min_age = min_age_days * 86400
    held = open_files()
    candidates: list[tuple[Path, str]] = []

    def eligible(path: Path) -> bool:
        try:
            if path.resolve() in held:
                return False
            return (now - path.stat().st_mtime) >= min_age
        except OSError:
            return False

    # Rotated segments: keep the newest N per base name, drop the rest. Counted
    # per base so a busy journal cannot evict another journal's history.
    segments: dict[str, list[Path]] = defaultdict(list)
    for path in LOGS.iterdir():
        if not path.is_file():
            continue
        match = ROTATED.match(path.name)
        if match:
            segments[match.group("base")].append(path)
    for base, paths in segments.items():
        paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for stale in paths[keep_segments:]:
            if eligible(stale):
                candidates.append((stale, f"rotated segment beyond newest {keep_segments} of {base}"))

    # One-off run logs from launches that have exited.
    for path in LOGS.iterdir():
        if not path.is_file() or ROTATED.match(path.name):
            continue
        if any(pattern.match(path.name) for pattern in ONE_OFF) and eligible(path):
            candidates.append((path, "one-off run log"))

    # logs/archive: nothing in this repo writes or expires it.
    archive = LOGS / "archive"
    if archive.is_dir():
        cutoff = archive_days * 86400
        for path in archive.rglob("*"):
            if not path.is_file():
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age >= cutoff and path.resolve() not in held:
                candidates.append((path, f"archive older than {archive_days:g}d"))

    return candidates


def enforce_budget(budget_bytes: int, already: set[Path], min_age_days: float) -> list[tuple[Path, str]]:
    """Oldest-first deletions until logs/ fits the budget.

    A backstop, not the main mechanism: if the per-rule retention above already
    keeps the tree small this selects nothing. Active journals are excluded by
    the same two guards, which means the budget can be genuinely unreachable --
    if live journals alone exceed it we report the shortfall rather than
    deleting something we promised not to touch.
    """
    if budget_bytes <= 0:
        return []
    now = time.time()
    min_age = min_age_days * 86400
    held = open_files()
    total = 0
    pool: list[tuple[float, int, Path]] = []
    for path in LOGS.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        total += stat.st_size
        if path in already:
            continue
        try:
            if path.resolve() in held or (now - stat.st_mtime) < min_age:
                continue
        except OSError:
            continue
        pool.append((stat.st_mtime, stat.st_size, path))

    freed = sum(p.stat().st_size for p in already if p.exists())
    if total - freed <= budget_bytes:
        return []
    pool.sort()
    picked: list[tuple[Path, str]] = []
    for _mtime, size, path in pool:
        if total - freed <= budget_bytes:
            break
        freed += size
        picked.append((path, "over total budget"))
    return picked


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G"):
        if abs(n) < 1024 or unit == "G":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-age-days", type=float, default=float(os.getenv("LOG_PRUNE_MIN_AGE_DAYS", "14")),
                        help="Never touch anything modified more recently than this (default: 14)")
    parser.add_argument("--keep-segments", type=int, default=int(os.getenv("LOG_PRUNE_KEEP_SEGMENTS", "3")),
                        help="Rotated segments to keep per journal (default: 3, matching the writers)")
    parser.add_argument("--archive-days", type=float, default=float(os.getenv("LOG_PRUNE_ARCHIVE_DAYS", "90")),
                        help="Expire logs/archive entries older than this (default: 90)")
    parser.add_argument("--budget-gb", type=float, default=float(os.getenv("LOG_PRUNE_BUDGET_GB", "3")),
                        help="Total ceiling for logs/ in GiB; 0 disables (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted and exit")
    args = parser.parse_args()

    if not LOGS.is_dir():
        print(f"no logs directory at {LOGS}")
        return 0

    before = sum(f.stat().st_size for f in LOGS.rglob("*") if f.is_file())

    candidates = collect(args.min_age_days, args.keep_segments, args.archive_days)
    chosen = {path for path, _ in candidates}
    candidates += enforce_budget(int(args.budget_gb * 1024**3), chosen, args.min_age_days)

    if not candidates:
        print(f"logs/ {human(before)} - nothing to prune "
              f"(budget {args.budget_gb:g}G, keep {args.keep_segments} segments, min age {args.min_age_days:g}d)")
        return 0

    reclaimed = 0
    for path, reason in sorted(candidates, key=lambda item: -item[0].stat().st_size if item[0].exists() else 0):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        verb = "would delete" if args.dry_run else "deleted"
        if not args.dry_run:
            try:
                path.unlink()
            except OSError as exc:
                print(f"  skipped {path.name}: {exc}", file=sys.stderr)
                continue
        reclaimed += size
        print(f"  {verb} {human(size):>7}  {path.relative_to(ROOT)}  [{reason}]")

    print(f"logs/ {human(before)} -> {human(before - reclaimed)} "
          f"({human(reclaimed)} {'reclaimable' if args.dry_run else 'reclaimed'}, {len(candidates)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
