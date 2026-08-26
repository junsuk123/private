"""Training labels must describe the trade the executor would actually hold.

The defect: the shared label was ``tp=25bps / sl=100bps / 600s`` while the live
session exited at ``-22bps / +100bps / 1800s``. A pattern that rose 25bps and then
rolled over was a training SUCCESS and a live STOP-OUT, so the model was rewarded
for finding exactly the trades the executor loses on.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.live_training_pipeline import (
    _label_barriers,
    _load_trade_plan_label_contracts,
    _observed_cost_bps,
    _row_market,
    _trade_plan_contract_for_frame,
    build_live_training_rows_from_feature_journal,
)
from app.strategy.exit_geometry import exit_geometry


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)


def _frame(symbol: str, index: int, price: float) -> dict:
    values = {name: 0.0 for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names}
    values.update(
        {
            "second_data_ready": 1.0,
            "tick_count_5s": 8.0,
            "spread_bps": 5.0,
            "return_1m": 0.001,
            "liquidity_score": 0.8,
            "orderbook_imbalance": 0.2,
            "realized_volatility_3m": 0.003,
            # A frame with no book depth is dropped by the training-quality gate.
            "bid_depth": 5_000.0,
            "ask_depth": 4_000.0,
            "depth_ratio": 1.25,
        }
    )
    return {
        "symbol": symbol,
        "decision_time": (NOW + timedelta(seconds=30 * index)).isoformat(),
        "mark_price": price,
        "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
        "values": values,
    }


def _journal(tmp_path, frames) -> str:
    path = tmp_path / "frames.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame) + "\n")
    return str(path)


def test_default_label_barriers_come_from_a_real_exit_geometry(monkeypatch):
    for name in (
        "LIVE_LABEL_STRATEGY",
        "LIVE_LABEL_TAKE_PROFIT_BPS",
        "LIVE_LABEL_STOP_LOSS_BPS",
        "LIVE_LABEL_HORIZON_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    take_profit, stop_loss, horizon, basis = _label_barriers()
    geometry = exit_geometry("intraday_momentum")
    assert take_profit == geometry.take_profit_bps
    assert stop_loss == geometry.stop_loss_bps
    assert horizon == float(geometry.max_holding_seconds)
    assert basis == "strategy_exit_geometry:intraday_momentum"
    # The executor risks less than it targets; the old label had it backwards.
    assert stop_loss < take_profit


def test_legacy_label_barriers_remain_available(monkeypatch):
    monkeypatch.setenv("LIVE_LABEL_STRATEGY", "legacy")
    for name in (
        "LIVE_LABEL_TAKE_PROFIT_BPS",
        "LIVE_LABEL_STOP_LOSS_BPS",
        "LIVE_LABEL_HORIZON_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    take_profit, stop_loss, horizon, basis = _label_barriers()
    assert (take_profit, stop_loss, horizon) == (25.0, 100.0, 600.0)
    assert basis == "legacy_constants"


def test_explicit_environment_overrides_still_win(monkeypatch):
    monkeypatch.setenv("LIVE_LABEL_STRATEGY", "intraday_momentum")
    monkeypatch.setenv("LIVE_LABEL_STOP_LOSS_BPS", "40")
    monkeypatch.delenv("LIVE_LABEL_TAKE_PROFIT_BPS", raising=False)
    monkeypatch.delenv("LIVE_LABEL_HORIZON_SECONDS", raising=False)
    _take_profit, stop_loss, _horizon, _basis = _label_barriers()
    assert stop_loss == 40.0


def test_market_is_recorded_on_every_row(tmp_path, monkeypatch):
    monkeypatch.delenv("LIVE_LABEL_MARKET_ADJUST", raising=False)
    monkeypatch.setenv("LIVE_LABEL_MARKET_ADJUST", "false")
    frames = [_frame("005930", index, 70_000.0 * (1.0 + 0.002 * index)) for index in range(8)]
    frames += [_frame("AAPL", index, 200.0 * (1.0 + 0.002 * index)) for index in range(8)]
    rows = build_live_training_rows_from_feature_journal(
        _journal(tmp_path, frames), db_path=tmp_path / "missing.sqlite3"
    )
    assert rows
    markets = {row["ticker"]: row["market"] for row in rows}
    assert markets["005930"] == "KR"
    assert markets["AAPL"] == "US"
    assert _row_market("000660") == "KR"
    assert _row_market("MSFT") == "US"


def test_label_cost_uses_shared_all_in_round_trip(monkeypatch):
    seen = {}

    def fake_all_in(symbol, *, spread_bps=None, fallback_bps=0.0):
        seen.update(symbol=symbol, spread_bps=spread_bps)
        return 73.7

    monkeypatch.setattr("app.cost.round_trip.all_in_round_trip_bps", fake_all_in)
    assert _observed_cost_bps(_frame("AAPL", 0, 200.0)) == 73.7
    assert seen == {"symbol": "AAPL", "spread_bps": 5.0}


def test_per_strategy_labels_use_each_strategys_own_barriers(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_LABEL_PER_STRATEGY", "true")
    monkeypatch.setenv("LIVE_LABEL_MARKET_ADJUST", "false")
    monkeypatch.delenv("LIVE_LABEL_TAKE_PROFIT_BPS", raising=False)
    monkeypatch.delenv("LIVE_LABEL_STOP_LOSS_BPS", raising=False)
    monkeypatch.delenv("LIVE_LABEL_HORIZON_SECONDS", raising=False)
    # A path that rises ~30bps and then falls back: a success under the OLD
    # +25bps label and a failure under the executor's real +100bps target.
    prices = [70_000.0, 70_150.0, 70_210.0, 70_100.0, 69_900.0, 69_800.0, 69_750.0]
    frames = [_frame("005930", index, price) for index, price in enumerate(prices)]
    rows = build_live_training_rows_from_feature_journal(
        _journal(tmp_path, frames), db_path=tmp_path / "missing.sqlite3"
    )
    assert rows
    labels = rows[0]["strategy_labels"]
    assert "intraday_momentum" in labels
    assert "ofi_microprice_exhaustion_reversal" in labels
    momentum = labels["intraday_momentum"]
    assert momentum["take_profit_bps"] == exit_geometry("intraday_momentum").take_profit_bps
    assert momentum["stop_loss_bps"] == exit_geometry("intraday_momentum").stop_loss_bps
    # Each strategy is labelled against its own barriers, so the parameters differ.
    exhaustion = labels["ofi_microprice_exhaustion_reversal"]
    assert exhaustion["horizon_seconds"] != momentum["horizon_seconds"]


def test_per_strategy_labels_are_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LIVE_LABEL_PER_STRATEGY", raising=False)
    monkeypatch.setenv("LIVE_LABEL_MARKET_ADJUST", "false")
    frames = [_frame("005930", index, 70_000.0 + 50.0 * index) for index in range(6)]
    rows = build_live_training_rows_from_feature_journal(
        _journal(tmp_path, frames), db_path=tmp_path / "missing.sqlite3"
    )
    assert rows
    assert "strategy_labels" not in rows[0]
    assert rows[0]["label_basis"].startswith("strategy_exit_geometry:")


def test_trade_plan_contract_is_the_label_authority_for_its_signal_frame(
    tmp_path, monkeypatch
):
    database = tmp_path / "trading_state.sqlite3"
    payload = {
        "take_profit_rule": {"rate": 0.008},
        "stop_loss_rule": {"rate": 0.0035},
        "time_exit": {"max_holding_seconds": 240},
    }
    with sqlite3.connect(database) as conn:
        conn.execute(
            "create table trade_plan (plan_id text, created_at text, symbol text, "
            "strategy_id text, payload_json text)"
        )
        conn.execute(
            "insert into trade_plan values (?, ?, ?, ?, ?)",
            (
                "plan-exact",
                (NOW + timedelta(seconds=4)).isoformat(),
                "005930",
                "adaptive_anchored_vwap_reversion",
                json.dumps(payload),
            ),
        )
    monkeypatch.setenv("TRADE_PLAN_LABEL_STORE_PATH", str(database))

    contracts = _load_trade_plan_label_contracts()
    contract = _trade_plan_contract_for_frame("005930", NOW, contracts)

    assert contract is not None
    assert contract["plan_id"] == "plan-exact"
    assert contract["take_profit_bps"] == 80.0
    assert contract["stop_loss_bps"] == 35.0
    assert contract["horizon_seconds"] == 240.0
