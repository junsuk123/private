"""``_live_lock`` is a plain threading.Lock. Re-acquiring it deadlocks forever.

`app.web` has ~74 functions that take ``_live_lock``. Calling any of them from
inside an existing ``with _live_lock:`` block deadlocks that thread permanently,
and because the ASGI event loop and every background worker also need the lock,
the entire process stops serving while burning 0% CPU. That failure mode is
invisible in unit tests, silent in the log, and looks exactly like a hang.

It happened: a telemetry field added inside an existing locked logging block
called a helper that took the lock again, wedging the server under a live KR
session. This test is a static scan rather than a runtime check because the code
path only executes on a real collector loop.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "src" / "app" / "web.py"


def _functions_that_take_the_lock(lines: list[str]) -> set[str]:
    taking: set[str] = set()
    current: str | None = None
    for line in lines:
        match = re.match(r"\s*def (\w+)", line)
        if match:
            current = match.group(1)
        if current and "with _live_lock" in line:
            taking.add(current)
    return taking


def _nested_acquisitions(lines: list[str], lock_takers: set[str]):
    problems = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)with _live_lock", lines[index])
        if not match:
            index += 1
            continue
        indent = len(match.group(1))
        cursor = index + 1
        while cursor < len(lines):
            body = lines[cursor]
            if body.strip() and (len(body) - len(body.lstrip())) <= indent:
                break
            for name in lock_takers:
                if re.search(rf"\b{name}\s*\(", body):
                    problems.append((cursor + 1, name, body.strip()))
            cursor += 1
        index = cursor
    return problems


def test_live_lock_is_never_acquired_reentrantly():
    lines = WEB.read_text(encoding="utf-8").splitlines()
    lock_takers = _functions_that_take_the_lock(lines)
    assert lock_takers, "scan found no _live_lock users; the pattern must have changed"

    problems = _nested_acquisitions(lines, lock_takers)
    detail = "\n".join(
        f"  web.py:{line} calls {name}() inside a `with _live_lock:` block\n"
        f"      {text}"
        for line, name, text in problems
    )
    assert not problems, (
        "nested _live_lock acquisition self-deadlocks the process:\n"
        f"{detail}\n"
        "Compute the value BEFORE entering the locked block."
    )


def test_scanner_detects_a_planted_violation():
    """The guard above is only worth having if it can actually fail."""
    planted = [
        "def _helper():",
        "    with _live_lock:",
        "        return 1",
        "",
        "def _caller():",
        "    with _live_lock:",
        "        value = _helper()",
        "        return value",
    ]
    lock_takers = _functions_that_take_the_lock(planted)
    assert "_helper" in lock_takers
    assert _nested_acquisitions(planted, lock_takers)
