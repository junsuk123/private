from datetime import datetime, timedelta, timezone
import json

import pytest

from app.execution.entry_activity import BotConfirmedEntryActivity


NOW = datetime(2026, 9, 21, 16, tzinfo=timezone.utc)  # KR Sep22; US Sep21


def submission(order_id, when, *, market="KR", ticker="005930", **contract):
    return {"event_type": "live_order_submitted", "recorded_at": when.isoformat(),
        "payload": {"broker_order_id": order_id, "order": {"ticker": ticker, "market": market,
            "side": "BUY", "position_direction": "LONG", "position_effect": "OPEN",
            "execution_product": "CASH", **contract}}}


def status(order_id, when, *, ticker="005930", quantity=1, side="BUY", state="FILLED"):
    return {"event_type": "live_order_status", "recorded_at": when.isoformat(),
        "payload": {"order_id": order_id, "observed_at": when.isoformat(), "status": state,
            "raw": {"order_id": order_id, "ticker": ticker, "side": side, "quantity": quantity}}}


def write(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_missing_fresh_journal_is_zero_without_creating_files(tmp_path):
    path = tmp_path / "missing" / "live-orders.jsonl"
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) == 0
    assert provider.snapshot()["reason"] == "NO_BOT_JOURNAL_YET"
    assert not path.parent.exists()


def test_previously_observed_journal_disappearance_is_not_a_fresh_install(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("1", NOW-timedelta(minutes=2)), status("1", NOW-timedelta(minutes=1))])
    provider = BotConfirmedEntryActivity(path, cache_seconds=60)
    assert provider("KR", NOW) == 1
    # Only the explicitly created temporary test fixture is removed.
    path.unlink()
    assert provider("KR", NOW) is None
    assert provider.snapshot()["reason"] == "ENTRY_ACTIVITY_HISTORY_MISSING"


def test_submissions_and_rejections_without_positive_fill_are_not_entries(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("1", NOW-timedelta(minutes=2)),
                 status("1", NOW-timedelta(minutes=1), quantity=0, state="REJECTED")])
    assert BotConfirmedEntryActivity(path)("KR", NOW) == 0


def test_partial_and_repeated_fill_statuses_count_once_in_market_local_day(tmp_path):
    path = tmp_path / "orders.jsonl"
    yesterday_kr = NOW - timedelta(hours=2)
    events = [submission("kr-old", yesterday_kr-timedelta(minutes=1)),
              status("kr-old", yesterday_kr, quantity=1, state="PARTIALLY_FILLED"),
              submission("us", NOW-timedelta(minutes=20), market="NASD", ticker="AAPL"),
              status("us", NOW-timedelta(minutes=19), ticker="AAPL"),
              submission("kr-new", NOW-timedelta(minutes=10)),
              status("kr-new", NOW-timedelta(minutes=9), quantity=1, state="PARTIALLY_FILLED"),
              status("kr-old", NOW-timedelta(minutes=8), quantity=2),
              status("kr-new", NOW-timedelta(minutes=7), quantity=2),
              status("kr-new", NOW-timedelta(minutes=6), quantity=2)]
    write(path, events)
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) == 1
    assert provider("US", NOW) == 1
    assert provider.snapshot()["scope"] == "bot_confirmed_entries"
    assert provider.snapshot()["time_basis"] == "first_bot_observed_fill"


def test_close_buys_and_sell_exits_do_not_count(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("cover", NOW-timedelta(minutes=4), position_direction="SHORT",
                            position_effect="CLOSE", execution_product="CREDIT_BORROW"),
                 status("cover", NOW-timedelta(minutes=3)),
                 status("sell", NOW-timedelta(minutes=2), side="SELL")])
    assert BotConfirmedEntryActivity(path)("KR", NOW) == 0


def test_rotated_history_joins_submission_and_first_fill_without_double_count(tmp_path):
    path = tmp_path / "orders.jsonl"
    origin = NOW-timedelta(hours=2)
    write(path.with_name(path.name + ".1"), [submission("1", origin-timedelta(minutes=1)),
         status("1", origin, state="PARTIALLY_FILLED")])
    write(path, [status("1", NOW-timedelta(minutes=1), quantity=2)])
    assert BotConfirmedEntryActivity(path)("KR", NOW) == 0


@pytest.mark.parametrize("contents", ["{broken}\n", "{}\n", '{"event_type":"status"}', "not-json"])
def test_existing_malformed_or_truncated_journal_is_unavailable(tmp_path, contents):
    path = tmp_path / "orders.jsonl"
    path.write_text(contents, encoding="utf-8")
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) is None
    assert not provider.snapshot()["available"]


def test_missing_origin_or_history_rotation_gap_is_unavailable(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [status("unknown", NOW-timedelta(minutes=1))])
    assert BotConfirmedEntryActivity(path)("KR", NOW) is None
    write(path, [])
    write(path.with_name(path.name + ".2"), [])
    assert BotConfirmedEntryActivity(path)("KR", NOW) is None


def test_cache_invalidates_when_new_confirmed_entry_is_appended(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [])
    provider = BotConfirmedEntryActivity(path, cache_seconds=60)
    assert provider("KR", NOW) == 0
    write(path, [submission("1", NOW-timedelta(minutes=2)), status("1", NOW-timedelta(minutes=1))])
    assert provider("KR", NOW) == 1
    assert provider("KR", NOW-timedelta(minutes=3)) == 0


@pytest.mark.parametrize("limits", [{"maximum_bytes": 1}, {"maximum_records": 1}])
def test_exhausted_read_budget_is_unavailable_not_partial_count(tmp_path, limits):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("1", NOW-timedelta(minutes=2)), status("1", NOW-timedelta(minutes=1))])
    assert BotConfirmedEntryActivity(path, **limits)("KR", NOW) is None


def test_small_record_budget_resumes_without_assuming_midnight_order(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("old", NOW-timedelta(hours=3)),
        status("old", NOW-timedelta(hours=2)), submission("today", NOW-timedelta(minutes=2)),
        status("today", NOW-timedelta(minutes=1))])
    provider = BotConfirmedEntryActivity(path, maximum_records=3)
    assert provider("KR", NOW) is None
    assert provider.snapshot()["reason"] == "ENTRY_ACTIVITY_HISTORY_LOADING"
    assert provider("KR", NOW) == 1


def test_future_records_do_not_enter_historical_count_and_clock_must_be_aware(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("1", NOW+timedelta(seconds=1)), status("1", NOW+timedelta(seconds=2))])
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) == 0
    assert provider("KR", NOW.replace(tzinfo=None)) is None


def test_unreadable_existing_journal_does_not_become_zero(tmp_path, monkeypatch):
    path = tmp_path / "orders.jsonl"
    write(path, [])
    provider = BotConfirmedEntryActivity(path)
    monkeypatch.setattr(provider, "_read", lambda *a: (_ for _ in ()).throw(PermissionError()))
    assert provider("KR", NOW) is None


def test_actual_live_journal_dataclass_format_is_read_without_broker(tmp_path):
    from app.execution.live_order_journal import LiveOrderJournal
    from app.execution.order_status_tracker import OrderStatusSnapshot
    from app.execution.kis_mock import MockKisExecution
    from app.schemas.domain import FinalOrder, OrderSide, OrderType

    path = tmp_path / "orders.jsonl"
    journal = LiveOrderJournal(path)
    order = FinalOrder(ticker="005930", market="KR", order_type=OrderType.LIMIT,
        side=OrderSide.BUY, quantity=1, limit_price=1000, manual_approval_required=False,
        position_direction="LONG", position_effect="OPEN", execution_product="CASH")
    journal.record("live_order_submitted", {"broker_order_id": "actual-schema", "order": order})
    observed = datetime.now(timezone.utc)
    raw = MockKisExecution(order_id="actual-schema", ticker="005930", side=OrderSide.BUY,
        quantity=1, price=1000, executed_value=1000, status="FILLED", message="fixture", executed_at=observed)
    journal.record("live_order_status", OrderStatusSnapshot("actual-schema", "FILLED", observed, raw))
    assert BotConfirmedEntryActivity(path)("KR", datetime.now(timezone.utc)) == 1


def test_us_day_uses_new_york_midnight_on_dst_transition(tmp_path):
    path = tmp_path / "orders.jsonl"
    now = datetime(2026, 11, 1, 7, tzinfo=timezone.utc)
    prior = datetime(2026, 11, 1, 3, 59, tzinfo=timezone.utc)
    today = datetime(2026, 11, 1, 4, 1, tzinfo=timezone.utc)
    write(path, [submission("old", prior-timedelta(minutes=1), market="US", ticker="AAPL"),
        status("old", prior, ticker="AAPL"),
        submission("new", today-timedelta(seconds=30), market="US", ticker="AAPL"),
        status("new", today, ticker="AAPL")])
    assert BotConfirmedEntryActivity(path)("US", now) == 1


def amendment(order_id, previous_id, when, **kwargs):
    event = submission(order_id, when, **kwargs)
    event["event_type"] = "live_order_amended"
    event["payload"]["previous_broker_order_id"] = previous_id
    return event


def test_changed_broker_id_amendments_count_one_economic_entry(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("original", NOW-timedelta(minutes=6)),
        status("original", NOW-timedelta(minutes=5), state="PARTIALLY_FILLED"),
        amendment("replacement", "original", NOW-timedelta(minutes=4)),
        status("replacement", NOW-timedelta(minutes=3), state="PARTIALLY_FILLED"),
        amendment("last", "replacement", NOW-timedelta(minutes=2)),
        status("last", NOW-timedelta(minutes=1))])
    assert BotConfirmedEntryActivity(path)("KR", NOW) == 1


@pytest.mark.parametrize("replacement_id", ["original", "replacement"])
def test_amendment_preserves_first_fill_day_across_midnight(tmp_path, replacement_id):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("original", NOW-timedelta(hours=2, minutes=1)),
        status("original", NOW-timedelta(hours=2), state="PARTIALLY_FILLED"),
        amendment(replacement_id, "original", NOW-timedelta(minutes=2)),
        status(replacement_id, NOW-timedelta(minutes=1))])
    assert BotConfirmedEntryActivity(path)("KR", NOW) == 0


def test_reused_broker_id_without_amendment_is_a_new_entry(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("reused", NOW-timedelta(hours=2, minutes=1)),
        status("reused", NOW-timedelta(hours=2)),
        submission("reused", NOW-timedelta(minutes=2)),
        status("reused", NOW-timedelta(minutes=1))])
    assert BotConfirmedEntryActivity(path)("KR", NOW) == 1


def test_missing_amendment_origin_does_not_invent_entry_history(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [amendment("new", "missing", NOW-timedelta(minutes=2)),
        status("new", NOW-timedelta(minutes=1))])
    assert BotConfirmedEntryActivity(path)("KR", NOW) is None


def test_old_timestamp_does_not_hide_later_current_day_fill(tmp_path):
    path = tmp_path / "orders.jsonl"
    old = {"event_type": "delayed", "recorded_at": (NOW-timedelta(hours=3)).isoformat(), "payload": {}}
    # Actual append order can differ from the writer's timestamp order.
    write(path, [submission("today", NOW-timedelta(minutes=2)),
                 status("today", NOW-timedelta(minutes=1)), old])
    assert BotConfirmedEntryActivity(path)("KR", NOW) == 1


def test_actual_concurrent_logger_timestamp_inversion_is_valid_history(tmp_path, monkeypatch):
    import threading
    from app.audit import logger
    from app.execution.live_order_journal import LiveOrderJournal

    path = tmp_path / "concurrent.jsonl"
    journal = LiveOrderJournal(path)
    entered, release = threading.Event(), threading.Event()
    original = logger._to_jsonable

    def delayed(value):
        if isinstance(value, dict) and value.get("delay"):
            entered.set()
            assert release.wait(5)
        return original(value)

    monkeypatch.setattr(logger, "_to_jsonable", delayed)
    worker = threading.Thread(target=lambda: journal.record("slow", {"delay": True}))
    worker.start()
    try:
        assert entered.wait(5)
        journal.record("fast", {"delay": False})
    finally:
        release.set()
        worker.join(5)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["recorded_at"] > rows[1]["recorded_at"]
    assert BotConfirmedEntryActivity(path)("KR", datetime.now(timezone.utc)) == 0


def test_unchanged_successful_history_is_cached_past_old_ttl(tmp_path, monkeypatch):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("1", NOW-timedelta(minutes=2)), status("1", NOW-timedelta(minutes=1))])
    provider = BotConfirmedEntryActivity(path, cache_seconds=0)
    assert provider("KR", NOW) == 1
    monkeypatch.setattr(provider, "_read", lambda *args: pytest.fail("unchanged history was reread"))
    assert provider("KR", NOW+timedelta(hours=1)) == 1


def test_future_event_releases_unchanged_cache_when_it_becomes_observable(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("future", NOW+timedelta(seconds=1)), status("future", NOW+timedelta(seconds=2))])
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) == 0
    assert provider("KR", NOW+timedelta(seconds=3)) == 1


def test_tiny_byte_budget_catches_up_despite_append_then_reads_only_tail(tmp_path, monkeypatch):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("first", NOW-timedelta(minutes=5)), status("first", NOW-timedelta(minutes=4))])
    provider = BotConfirmedEntryActivity(path, maximum_bytes=128, maximum_records=1)
    assert provider("KR", NOW) is None
    with path.open("a", encoding="utf-8") as handle:
        for row in [submission("second", NOW-timedelta(minutes=3)), status("second", NOW-timedelta(minutes=2))]:
            handle.write(json.dumps(row)+"\n")
    for _ in range(50):
        result = provider("KR", NOW)
        if result is not None:
            break
    assert result == 2
    assert provider._projection.sequence == 4
    ingested = []
    original = provider._projection._ingest
    monkeypatch.setattr(provider._projection, "_ingest", lambda row: (ingested.append(row), original(row))[-1])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status("second", NOW-timedelta(minutes=1)))+"\n")
    for _ in range(20):
        result = provider("KR", NOW)
        if result is not None:
            break
    assert result == 2
    assert len(ingested) == 1


def test_projection_tracks_rotation_and_is_shared_between_markets(tmp_path, monkeypatch):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("kr", NOW-timedelta(minutes=5)), status("kr", NOW-timedelta(minutes=4))])
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) == 1
    # Rename only this explicitly created temporary fixture to simulate rotation.
    path.replace(path.with_name(path.name+".1"))
    write(path, [submission("us", NOW-timedelta(minutes=3), ticker="AAPL", market="NASD"),
                 status("us", NOW-timedelta(minutes=2), ticker="AAPL")])
    ingested = []
    original = provider._projection._ingest
    monkeypatch.setattr(provider._projection, "_ingest", lambda row: (ingested.append(row), original(row))[-1])
    # The process already observed the complete unrotated origin, so rotation
    # does not imply an unknown earlier portion of this same trading day.
    assert provider("US", NOW) == 1
    assert provider("KR", NOW) == 1
    assert len(ingested) == 2


def test_same_inode_same_size_rewrite_cannot_reuse_old_count(tmp_path):
    import os
    path = tmp_path / "orders.jsonl"
    events = [submission("1", NOW-timedelta(minutes=2)), status("1", NOW-timedelta(minutes=1))]
    write(path, events)
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) == 1
    previous = path.stat()
    events[-1]["payload"]["raw"]["quantity"] = 0
    write(path, events)
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns+1_000_000))
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_ino == previous.st_ino
    assert provider("KR", NOW) is None
    assert provider.snapshot()["reason"] == "ENTRY_ACTIVITY_HISTORY_TRUNCATED"


def test_same_broker_id_in_two_markets_has_separate_lineage(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("shared", NOW-timedelta(minutes=6)),
        submission("shared", NOW-timedelta(minutes=5), ticker="AAPL", market="NASD"),
        status("shared", NOW-timedelta(minutes=4)),
        status("shared", NOW-timedelta(minutes=3), ticker="AAPL"),
        amendment("new", "shared", NOW-timedelta(minutes=2)),
        status("new", NOW-timedelta(minutes=1))])
    provider = BotConfirmedEntryActivity(path)
    assert provider("KR", NOW) == 1
    assert provider("US", NOW) == 1


def test_rotation_during_budget_limited_bootstrap_does_not_restart(tmp_path):
    path = tmp_path / "orders.jsonl"
    write(path, [submission("old", NOW-timedelta(minutes=6)), status("old", NOW-timedelta(minutes=5))])
    provider = BotConfirmedEntryActivity(path, maximum_bytes=128, maximum_records=1)
    assert provider("KR", NOW) is None
    path.replace(path.with_name(path.name+".1"))
    write(path, [submission("new", NOW-timedelta(minutes=3)), status("new", NOW-timedelta(minutes=2))])
    for _ in range(50):
        result = provider("KR", NOW)
        if result is not None:
            break
    assert result == 2
    assert provider._projection.sequence == 4
