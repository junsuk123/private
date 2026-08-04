from __future__ import annotations

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


def test_runtime_does_not_embed_a_preferred_issuer_symbol() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        *sorted((root / "src" / "app").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        root / "run.ps1",
    ]
    offenders = [
        str(path.relative_to(root))
        for path in targets
        if "005930" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
