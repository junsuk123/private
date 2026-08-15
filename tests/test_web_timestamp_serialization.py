from datetime import datetime, timezone

from app import web as web_module


def test_iso_or_none_attaches_seoul_offset_to_naive_live_timestamp() -> None:
    serialized = web_module._iso_or_none(datetime(2026, 8, 14, 14, 29, 32))

    assert serialized == "2026-08-14T14:29:32+09:00"


def test_iso_or_none_preserves_aware_utc_timestamp() -> None:
    serialized = web_module._iso_or_none(
        datetime(2026, 8, 14, 5, 29, 32, tzinfo=timezone.utc)
    )

    assert serialized == "2026-08-14T05:29:32+00:00"
