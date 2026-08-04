"""The terminal's movable frames are only as trustworthy as what persists.

The browser is the sole writer of this file, so these tests pin three things:
a dragged arrangement survives a round trip unchanged, a layout saved before
columns existed still loads, and a malformed or hostile payload is rejected
outright instead of being written back into the page that renders it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ui_layout_store import SCHEMA, TerminalLayoutStore, normalise_layout
from app.web_account_routes import create_account_router


def _layout() -> dict[str, object]:
    """A layer holding a wide frame beside a column of two stacked frames."""
    return {
        "schema": SCHEMA,
        "layers": [
            {
                "height": 1.1,
                "columns": [
                    {"width": 6.25, "frames": [{"key": "ops", "height": 1}]},
                    {"width": 2.75, "frames": [{"key": "own", "height": 1.4}, {"key": "ast", "height": 0.6}]},
                ],
            },
            {"height": 1.35, "columns": [{"width": 12, "frames": [{"key": "dia", "height": 1}]}]},
        ],
    }


def _client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_account_router(layout_store=TerminalLayoutStore(tmp_path / "layout.json"))
    )
    return TestClient(app)


def test_missing_layout_reads_as_empty(tmp_path: Path) -> None:
    store = TerminalLayoutStore(tmp_path / "absent.json")

    assert store.load() == {"schema": SCHEMA, "saved_at": None, "layers": []}


def test_saved_layout_round_trips(tmp_path: Path) -> None:
    store = TerminalLayoutStore(tmp_path / "layout.json")

    saved = store.save(_layout())
    reloaded = store.load()

    assert saved["layers"] == reloaded["layers"]
    stacked = reloaded["layers"][0]["columns"][1]
    assert [frame["key"] for frame in stacked["frames"]] == ["own", "ast"]
    assert stacked["frames"][0]["height"] == 1.4
    assert saved["saved_at"] and saved["saved_at"].endswith("Z")


def test_v1_layout_upconverts_to_columns(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    # Written by the release before frames could stack: one flat panel list.
    path.write_text(
        json.dumps(
            {
                "schema": "strategy_terminal_layout_v1",
                "saved_at": "2026-08-02T13:49:59Z",
                "layers": [{"height": 1.1, "panels": [{"key": "ops", "width": 7.0652}, {"key": "own", "width": 1.9348}]}],
            }
        ),
        encoding="utf-8",
    )

    loaded = TerminalLayoutStore(path).load()

    assert loaded["schema"] == SCHEMA
    columns = loaded["layers"][0]["columns"]
    assert [column["width"] for column in columns] == [7.0652, 1.9348]
    assert [column["frames"][0]["key"] for column in columns] == ["ops", "own"]
    assert all(len(column["frames"]) == 1 for column in columns)


def test_weights_are_clamped_not_rejected(tmp_path: Path) -> None:
    store = TerminalLayoutStore(tmp_path / "layout.json")

    saved = store.save(
        {"layers": [{"height": 900, "columns": [{"width": -4, "frames": [{"key": "ops", "height": 0}]}]}]}
    )

    layer = saved["layers"][0]
    assert layer["height"] == 24.0
    assert layer["columns"][0]["width"] == 0.05
    assert layer["columns"][0]["frames"][0]["height"] == 0.05


@pytest.mark.parametrize(
    "payload",
    [
        "not-an-object",
        {},
        {"layers": []},
        {"layers": [{"height": 1, "columns": []}]},
        {"layers": [{"height": 1, "columns": [{"width": 1, "frames": []}]}]},
        {"layers": [{"height": 1, "columns": [{"width": 1, "frames": [{"key": "Ops Overview", "height": 1}]}]}]},
        {"layers": [{"height": 1, "columns": [{"width": 1, "frames": [{"key": "<script>", "height": 1}]}]}]},
        {"layers": [{"height": 1, "columns": [{"width": 1, "frames": [{"key": "ops", "height": "tall"}]}]}]},
        {"layers": [{"height": float("nan"), "columns": [{"width": 1, "frames": [{"key": "ops", "height": 1}]}]}]},
        {
            "layers": [
                {
                    "height": 1,
                    "columns": [
                        {"width": 1, "frames": [{"key": "ops", "height": 1}]},
                        {"width": 1, "frames": [{"key": "ops", "height": 1}]},
                    ],
                }
            ]
        },
        {"layers": [{"height": 1, "columns": [{"width": 1, "frames": [{"key": "ops", "height": 1}]}]}] * 17},
    ],
)
def test_malformed_payloads_are_rejected(payload: object) -> None:
    with pytest.raises(ValueError):
        normalise_layout(payload)


def test_frame_and_column_counts_are_capped() -> None:
    frames = [{"key": f"panel-{index}", "height": 1} for index in range(13)]
    with pytest.raises(ValueError):
        normalise_layout({"layers": [{"height": 1, "columns": [{"width": 1, "frames": frames}]}]})

    columns = [{"width": 1, "frames": [{"key": f"panel-{index}", "height": 1}]} for index in range(13)]
    with pytest.raises(ValueError):
        normalise_layout({"layers": [{"height": 1, "columns": columns}]})


def test_corrupt_file_falls_back_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    path.write_text("{ this is not json", encoding="utf-8")

    # A hand-edited or truncated file must send the browser back to its default
    # layout, not break the terminal it is supposed to describe.
    assert TerminalLayoutStore(path).load()["layers"] == []


def test_routes_save_read_and_clear(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/api/account/layout").json()["layers"] == []

    saved = client.post("/api/account/layout", json=_layout())
    assert saved.status_code == 200
    assert client.get("/api/account/layout").json()["layers"] == saved.json()["layers"]

    cleared = client.delete("/api/account/layout")
    assert cleared.status_code == 200
    assert client.get("/api/account/layout").json()["layers"] == []


def test_route_rejects_malformed_layout(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/api/account/layout", json={"layers": [{"columns": "everything"}]})

    assert response.status_code == 400
    assert "error" in response.json()
    # A rejected payload must not have touched the stored layout.
    assert client.get("/api/account/layout").json()["layers"] == []


def test_stored_file_is_json_with_the_declared_schema(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    TerminalLayoutStore(path).save(_layout())

    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema"] == SCHEMA
    assert [frame["key"] for frame in document["layers"][0]["columns"][0]["frames"]] == ["ops"]


def test_terminal_page_ships_the_layout_controls(tmp_path: Path) -> None:
    page = _client(tmp_path).get("/account").text

    assert 'id="layout-save"' in page
    assert 'id="layout-reset"' in page
    assert "terminal_layout.js" in page
    assert "terminal_layout.css" in page
