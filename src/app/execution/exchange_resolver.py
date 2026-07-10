"""Routing-exchange resolution and validation for live orders.

A bare US ticker only tells us "US" — but KIS rejects an order whose
``OVRS_EXCG_CD`` does not match the security's real listing exchange, and silently
defaulting an unknown US BUY to NASD sends a live order to the wrong venue (or gets
it rejected). This resolver determines the routing exchange from authoritative
sources and, in live strict mode, BLOCKS a US BUY whose exchange cannot be resolved
rather than guessing.

Resolution priority (BUY):
  1. domestic 6-digit numeric ticker            -> KR
  2. the broker-reported exchange of a position we already hold (same ticker)
  3. operator override map                       (env ``KIS_US_EXCHANGE_MAP`` JSON)
  4. built ticker->exchange listing map          (``data/universe/us_exchange_map.csv``)
  5. quote-confirmed exchange (if supplied)
  6. configured default                          (env ``KIS_DEFAULT_US_EXCHANGE``)

SELL prefers the holding's broker exchange and is never hard-blocked (exiting a
position must not be prevented by a routing ambiguity).

This module is intentionally self-contained (no imports from ``app.trading``) so it
can be used from the execution layer without an import cycle.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

ALLOWED_US_EXCHANGES = ("NASD", "NYSE", "AMEX")

# Reason codes (spec).
US_EXCHANGE_UNKNOWN = "US_EXCHANGE_UNKNOWN"
US_EXCHANGE_UNSUPPORTED = "US_EXCHANGE_UNSUPPORTED"
US_EXCHANGE_DEFAULTED_PAPER_ONLY = "US_EXCHANGE_DEFAULTED_PAPER_ONLY"
US_EXCHANGE_DEFAULTED = "US_EXCHANGE_DEFAULTED"

_DEFAULT_CSV_PATH = "data/universe/us_exchange_map.csv"
_CSV_CACHE: dict[str, Any] = {"mtime": None, "map": {}, "path": None}


@dataclass(frozen=True)
class ExchangeResolution:
    symbol: str
    side: str
    exchange: str               # KR / NASD / NYSE / AMEX (best-effort even when not allowed)
    source: str                 # domestic | holding | env_map | csv_map | quote | default | unknown
    confidence: float           # 1.0 = authoritative, 0.0 = defaulted/unknown
    allowed: bool
    reason_code: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ExchangeResolver:
    def __init__(
        self,
        *,
        strict: bool | None = None,
        allow_default_in_live: bool | None = None,
        default_exchange: str | None = None,
        csv_path: str = _DEFAULT_CSV_PATH,
    ) -> None:
        self._strict_override = strict
        self._allow_default_in_live_override = allow_default_in_live
        self._default_exchange_override = default_exchange
        self.csv_path = csv_path

    # -- config (env-overridable at call time) --------------------------
    @property
    def strict(self) -> bool:
        if self._strict_override is not None:
            return self._strict_override
        return _env_bool("KIS_US_EXCHANGE_STRICT", True)

    @property
    def allow_default_in_live(self) -> bool:
        if self._allow_default_in_live_override is not None:
            return self._allow_default_in_live_override
        return _env_bool("KIS_ALLOW_DEFAULT_US_EXCHANGE_IN_LIVE", False)

    @property
    def default_exchange(self) -> str:
        if self._default_exchange_override:
            return self._default_exchange_override
        return (os.getenv("KIS_DEFAULT_US_EXCHANGE", "NASD").upper() or "NASD")

    # -- resolution -----------------------------------------------------
    def resolve(
        self,
        symbol: str,
        side: str,
        *,
        account: Any | None = None,
        live: bool = True,
        quote_exchange: str | None = None,
    ) -> ExchangeResolution:
        s = str(symbol or "").strip().upper()
        side_u = str(side or "BUY").strip().upper()

        if s.isdigit() and len(s) == 6:
            return ExchangeResolution(s, side_u, "KR", "domestic", 1.0, True, None, {"live": live})

        # Authoritative sources, in priority order.
        held = self._exchange_from_holdings(s, account)
        if held:
            return self._decide(s, side_u, held, "holding", 1.0, live)

        # For SELL, prefer the holding exchange; if we don't hold it (unusual for a
        # sell) fall through to the maps but never hard-block an exit.
        mapped = _load_env_map().get(s)
        if mapped:
            return self._decide(s, side_u, mapped, "env_map", 1.0, live)

        listed = self._load_csv_map().get(s)
        if listed:
            return self._decide(s, side_u, listed, "csv_map", 1.0, live)

        if quote_exchange:
            q = str(quote_exchange).strip().upper()
            if q in ALLOWED_US_EXCHANGES:
                return self._decide(s, side_u, q, "quote", 0.9, live)

        # Nothing authoritative — only the configured default remains.
        return self._decide(s, side_u, self.default_exchange, "default", 0.0, live)

    def _decide(
        self, symbol: str, side: str, exchange: str, source: str, confidence: float, live: bool
    ) -> ExchangeResolution:
        exch = str(exchange or "").strip().upper()
        diag = {"live": live, "strict": self.strict, "allow_default_in_live": self.allow_default_in_live}
        if exch not in ALLOWED_US_EXCHANGES:
            return ExchangeResolution(
                symbol, side, exch, source, confidence, False, US_EXCHANGE_UNSUPPORTED, diag
            )
        if source != "default":
            return ExchangeResolution(symbol, side, exch, source, confidence, True, None, diag)

        # Defaulted (no authoritative source).
        if side == "SELL":
            # Never block an exit on routing ambiguity; route to the default and warn.
            return ExchangeResolution(symbol, side, exch, source, 0.0, True, US_EXCHANGE_DEFAULTED, diag)
        if not live:
            # Paper: allow the default but mark it so it can never be mistaken for live-safe.
            return ExchangeResolution(
                symbol, side, exch, source, 0.0, True, US_EXCHANGE_DEFAULTED_PAPER_ONLY, diag
            )
        if self.allow_default_in_live and not self.strict:
            return ExchangeResolution(symbol, side, exch, source, 0.0, True, US_EXCHANGE_DEFAULTED, diag)
        # Live + strict + unknown BUY: block rather than guess a venue.
        return ExchangeResolution(symbol, side, exch, source, 0.0, False, US_EXCHANGE_UNKNOWN, diag)

    def _exchange_from_holdings(self, symbol: str, account: Any | None) -> str | None:
        if account is None:
            return None
        for holding in getattr(account, "holdings", ()) or ():
            if str(getattr(holding, "ticker", "") or "").strip().upper() != symbol:
                continue
            held = str(getattr(holding, "market", "") or "").strip().upper()
            for code in ("NYSE", "AMEX", "NASD"):
                if code in held:
                    return code
        return None

    def _load_csv_map(self) -> dict[str, str]:
        return _load_csv_map(self.csv_path)


def _load_env_map() -> dict[str, str]:
    raw = os.getenv("KIS_US_EXCHANGE_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k).upper().strip(): str(v).upper().strip()
        for k, v in data.items()
        if str(k).strip() and str(v).strip()
    }


def _load_csv_map(path: str = _DEFAULT_CSV_PATH) -> dict[str, str]:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    if _CSV_CACHE.get("mtime") != mtime or _CSV_CACHE.get("path") != path:
        mapping: dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    sym = str(row.get("symbol") or "").strip().upper()
                    exch = str(row.get("exchange") or "").strip().upper()
                    if sym and exch in ALLOWED_US_EXCHANGES:
                        mapping[sym] = exch
        except OSError:
            return dict(_CSV_CACHE.get("map") or {})
        _CSV_CACHE["map"] = mapping
        _CSV_CACHE["mtime"] = mtime
        _CSV_CACHE["path"] = path
    return _CSV_CACHE.get("map") or {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
