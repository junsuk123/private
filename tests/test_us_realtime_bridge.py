from app.trading.us_realtime_bridge import _extract_price_book, _fetch_overseas_quote
from app.data.market_data_health import evaluate_market_data_health
from app.data.realtime_store import RealtimeMarketDataStore
from app.trading.us_realtime_bridge import _make_records


def test_extract_price_book_uses_kis_price_fields_not_change_fields() -> None:
    payload = {
        "output1": {
            "last": "12.42",
            "bvol": "999",
            "avol": "888",
        },
        "output2": {
            "pbid1": "12.41",
            "pask1": "12.43",
            "vbid1": "120",
            "vask1": "150",
            "dbid1": "-630",
            "dask1": "700",
        },
    }

    book = _extract_price_book(payload)

    assert book["bid"] == 12.41
    assert book["ask"] == 12.43
    assert book["bid_size"] == 120.0
    assert book["ask_size"] == 150.0


def test_extract_price_book_rejects_implausibly_wide_source_book() -> None:
    payload = {
        "output1": {"last": "16.00"},
        "output2": {
            "pbid1": "5.00",
            "pask1": "706.00",
            "vbid1": "10",
            "vask1": "20",
        },
    }

    try:
        _extract_price_book(payload)
        raise AssertionError("implausible source book must be rejected")
    except RuntimeError as exc:
        assert "IMPLAUSIBLE_BID_ASK_SPREAD" in str(exc)


def test_fetch_overseas_quote_uses_daytime_exchange_family(monkeypatch) -> None:
    """A base NYS request returns a successful but empty book during daytime."""
    from app.data import kis_realtime
    from app.trading import us_realtime_bridge

    requests: list[str] = []

    def fake_get(_path, _tr_id, params):
        requests.append(params["EXCD"])
        return {"rt_cd": "0"}

    monkeypatch.setattr(us_realtime_bridge, "_exchange_code", lambda *_: "NYS")
    monkeypatch.setattr(us_realtime_bridge, "_kis_get", fake_get)
    monkeypatch.setattr(kis_realtime, "is_us_daytime_quote_session", lambda: True)

    result = _fetch_overseas_quote("BAC")

    assert requests == ["BAY", "BAY"]
    assert result["exchange"] == "BAY"


def test_rest_snapshot_book_is_not_live_entry_evidence(tmp_path) -> None:
    _tick, book = _make_records(
        "BAC",
        "BAY",
        {"last": 62.4, "bid": 62.38, "ask": 62.4, "bid_size": 6, "ask_size": 4, "volume": 1},
    )
    store = RealtimeMarketDataStore(tmp_path / "market.sqlite3")
    store.save_orderbooks((book,))

    health = evaluate_market_data_health(store, "BAC", now=book.received_at)

    assert health.ok_for_live_buy is False
    assert "REST_SNAPSHOT_ONLY" in health.reason_codes
    assert health.market_group == "US"
    # _make_records resolves the authoritative session at the record timestamp;
    # this test runs in every US phase, so a fixed daytime assertion is clock flaky.
    assert health.session == book.meta.session.value
