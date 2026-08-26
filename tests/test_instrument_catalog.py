from __future__ import annotations

from app.data.instrument_catalog import load_us_instrument_catalog


def test_official_catalog_etf_flag_is_loaded_without_network(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text(
        "symbol,security_name,exchange,is_etf,source,collected_at\n"
        "SPY,SPDR S&P 500 ETF,P,Y,official,2026-08-25T00:00:00+00:00\n"
        "AAPL,Apple Inc.,Q,N,official,2026-08-25T00:00:00+00:00\n",
        encoding="utf-8",
    )

    rows = load_us_instrument_catalog(str(path))

    assert rows["SPY"].is_etf is True
    assert rows["AAPL"].is_etf is False
