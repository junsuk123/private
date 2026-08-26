"""Local, point-in-time instrument-type catalogue used by hot-path gates.

The live path never performs network IO.  ``scripts/collect_us_instrument_catalog.py``
refreshes this cache from Nasdaq Trader's official symbol-directory files, whose
``ETF`` column covers Nasdaq and other U.S. listed venues.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_US_CATALOG_PATH = Path("data/universe/us_instrument_types.csv")


@dataclass(frozen=True)
class UsInstrumentRecord:
    symbol: str
    security_name: str
    exchange: str
    is_etf: bool


def _catalog_path() -> Path:
    return Path(os.getenv("US_INSTRUMENT_CATALOG_PATH", str(DEFAULT_US_CATALOG_PATH)))


@lru_cache(maxsize=4)
def load_us_instrument_catalog(path_text: str = "") -> dict[str, UsInstrumentRecord]:
    path = Path(path_text) if path_text else _catalog_path()
    if not path.exists():
        return {}
    rows: dict[str, UsInstrumentRecord] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                rows[symbol] = UsInstrumentRecord(
                    symbol=symbol,
                    security_name=str(row.get("security_name") or "").strip(),
                    exchange=str(row.get("exchange") or "").strip().upper(),
                    is_etf=str(row.get("is_etf") or "").strip().upper() in {"1", "Y", "TRUE"},
                )
    except (OSError, csv.Error):
        return {}
    return rows


def us_instrument(symbol: str) -> UsInstrumentRecord | None:
    return load_us_instrument_catalog().get(str(symbol or "").strip().upper())


def is_us_etf(symbol: str) -> bool:
    record = us_instrument(symbol)
    return bool(record and record.is_etf)

