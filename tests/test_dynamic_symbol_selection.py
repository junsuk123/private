from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

from app.refactor_dashboard import _default_symbol


def test_dashboard_default_symbol_comes_from_latest_observed_market(tmp_path: Path) -> None:
    database = tmp_path / "market.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            create table realtime_minute_bars (symbol text, minute_start text);
            insert into realtime_minute_bars values ('111111', '2026-08-02T00:01:00+00:00');
            insert into realtime_minute_bars values ('222222', '2026-08-02T00:02:00+00:00');
            """
        )

    assert _default_symbol({}, {}, database) == "222222"


PREFERRED_ISSUER = "005930"


def _code_mentions(source: str, needle: str) -> bool:
    """Does the EXECUTABLE code contain ``needle``?

    Comments and docstrings are excluded. The point of this rule is that the
    runtime must not prefer a particular issuer; prose explaining WHY the
    selection is dynamic is the opposite of a violation, and the measurement
    that motivated the change names the symbol it measured. A raw substring scan
    cannot tell ``symbol = "005930"`` from "we measured that 005930 was the only
    name recurring often enough to build history", and scrubbing the second to
    satisfy the first would delete the evidence for the design.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None) or []
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) not in docstrings:
            if needle in str(node.value):
                return True
        elif isinstance(node, ast.Name) and needle in node.id:
            return True
        elif isinstance(node, ast.Attribute) and needle in node.attr:
            return True
    return False


def _script_mentions(source: str, needle: str) -> bool:
    """Same rule for the PowerShell launcher, whose comments start with ``#``."""
    for line in source.splitlines():
        code = line.split("#", 1)[0]
        if needle in code:
            return True
    return False


def test_runtime_does_not_embed_a_preferred_issuer_symbol() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(path.relative_to(root))
        for path in (
            *sorted((root / "src" / "app").rglob("*.py")),
            *sorted((root / "scripts").glob("*.py")),
        )
        if _code_mentions(path.read_text(encoding="utf-8-sig"), PREFERRED_ISSUER)
    ]
    launcher = root / "run.ps1"
    # ``utf-8-sig`` throughout: several sources carry a BOM, which ``ast.parse``
    # rejects as a non-printable character on the first line.
    if launcher.exists() and _script_mentions(
        launcher.read_text(encoding="utf-8-sig"), PREFERRED_ISSUER
    ):
        offenders.append("run.ps1")
    assert offenders == []


def test_the_issuer_detector_still_catches_a_real_embedding() -> None:
    """Guards the guard: relaxing the scan must not disarm it."""
    assert _code_mentions('SYMBOL = "005930"\n', PREFERRED_ISSUER)
    assert _code_mentions('universe = ["005930", "000660"]\n', PREFERRED_ISSUER)
    assert _code_mentions('default = {"ticker": "005930"}\n', PREFERRED_ISSUER)
    # ...while prose about the measurement stays allowed.
    assert not _code_mentions('"""Only 005930 recurred."""\n', PREFERRED_ISSUER)
    assert not _code_mentions("x = 1  # 005930 averaged 2,097 bars\n", PREFERRED_ISSUER)
    assert _script_mentions('$s = "005930"\n', PREFERRED_ISSUER)
    assert not _script_mentions("# only 005930 recurred\n", PREFERRED_ISSUER)
