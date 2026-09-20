"""Offline reader for the official U.S. instrument catalogue."""

from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_CATALOG_PATH = Path("data/universe/us_instrument_types.csv")


@dataclass(frozen=True)
class UsInstrument:
    symbol: str
    security_name: str
    exchange: str
    is_etf: bool
    source: str = ""
    collected_at: datetime | None = None


def _flag(value: object) -> bool:
    return str(value or "").strip().upper() in {"1", "TRUE", "T", "YES", "Y"}


def _timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_us_instrument_catalog(
    path: str | os.PathLike[str] = DEFAULT_CATALOG_PATH,
) -> dict[str, UsInstrument]:
    catalog_path = Path(path)
    if not catalog_path.is_file():
        return {}
    rows: dict[str, UsInstrument] = {}
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            symbol = str(raw.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            rows[symbol] = UsInstrument(
                symbol=symbol,
                security_name=str(raw.get("security_name") or symbol).strip(),
                exchange=str(raw.get("exchange") or "").strip(),
                is_etf=_flag(raw.get("is_etf")),
                source=str(raw.get("source") or "").strip(),
                collected_at=_timestamp(raw.get("collected_at")),
            )
    return rows


_CACHE_LOCK = threading.Lock()
_CACHE_PATH: Path | None = None
_CACHE_MTIME_NS: int | None = None
_CACHE: dict[str, UsInstrument] = {}


def us_instrument(
    symbol: str,
    path: str | os.PathLike[str] | None = None,
) -> UsInstrument | None:
    global _CACHE_PATH, _CACHE_MTIME_NS, _CACHE
    catalog_path = Path(path or os.getenv("US_INSTRUMENT_CATALOG", DEFAULT_CATALOG_PATH))
    try:
        mtime_ns = catalog_path.stat().st_mtime_ns
    except OSError:
        return None
    resolved = catalog_path.resolve()
    with _CACHE_LOCK:
        if resolved != _CACHE_PATH or mtime_ns != _CACHE_MTIME_NS:
            _CACHE = load_us_instrument_catalog(resolved)
            _CACHE_PATH = resolved
            _CACHE_MTIME_NS = mtime_ns
        return _CACHE.get(str(symbol or "").strip().upper())

