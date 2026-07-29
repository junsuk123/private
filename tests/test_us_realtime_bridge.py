from __future__ import annotations

from app.trading import us_realtime_bridge


def _quote(*, price: float = 10.0, volume: float = 1_000.0) -> dict[str, float]:
    return {
        "last": price,
        "bid": price - 0.01,
        "ask": price + 0.01,
        "bid_size": 100.0,
        "ask_size": 100.0,
        "volume": volume,
    }


def test_us_rest_poll_does_not_turn_cumulative_volume_into_repeated_trades() -> None:
    with us_realtime_bridge._US_POLL_STATE_LOCK:
        us_realtime_bridge._US_POLL_STATE.clear()

    first_tick, first_book = us_realtime_bridge._make_records(
        "TEST",
        "NAS",
        _quote(volume=10_000),
    )
    repeated_tick, repeated_book = us_realtime_bridge._make_records(
        "TEST",
        "NAS",
        _quote(volume=10_000),
    )
    changed_tick, _ = us_realtime_bridge._make_records(
        "TEST",
        "NAS",
        _quote(price=10.02, volume=10_025),
    )

    assert first_tick is None
    assert repeated_tick is None
    assert first_book is not None
    assert repeated_book is not None
    assert changed_tick is not None
    assert changed_tick.volume == 25
