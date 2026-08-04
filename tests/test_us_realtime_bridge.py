from app.trading.us_realtime_bridge import _extract_price_book


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
