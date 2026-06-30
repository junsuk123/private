from pathlib import Path
from datetime import datetime

web_path = Path("src/app/web.py")
pipeline_path = Path("src/app/models/live_training_pipeline.py")
bridge_path = Path("src/app/trading/us_realtime_bridge.py")

for path in (web_path, pipeline_path):
    backup = path.with_suffix(path.suffix + f".bak_internal_us_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"backup: {backup}")

bridge_path.parent.mkdir(parents=True, exist_ok=True)

bridge_code = r'''
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)

STORE_PATH = Path("data/store/realtime_market_data.sqlite3")
BASE_URL = "https://openapi.koreainvestment.com:9443"

_US_MARKET_NAMES = {"US", "NASDAQ", "NAS", "NYSE", "NYS", "AMEX", "AMS", "ARCA", "BATS", "CBOE", "IEX"}
_KR_MARKET_NAMES = {"KRX", "KOSPI", "KOSDAQ", "KONEX"}


def _env_any(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _app_key() -> str:
    return _env_any("KIS_APP_KEY", "KIS_APPKEY", "APP_KEY", "KIS_APP_KEY_LIVE")


def _app_secret() -> str:
    return _env_any("KIS_APP_SECRET", "KIS_APPSECRET", "APP_SECRET", "KIS_APP_SECRET_LIVE")


def _base_url() -> str:
    return _env_any("KIS_BASE_URL", "KIS_LIVE_BASE_URL", default=BASE_URL).rstrip("/")


def _json_request(method: str, url: str, *, headers: dict[str, str] | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("content-type", "application/json; charset=utf-8")

    req = Request(url, data=body, headers=request_headers, method=method.upper())
    try:
        with urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except Exception:
            data = {"raw_text": raw}
        raise RuntimeError(f"KIS HTTP {exc.code} {url} {data}") from exc


def _access_token() -> str:
    cached = _env_any("KIS_ACCESS_TOKEN", "ACCESS_TOKEN")
    if cached:
        return cached

    key = _app_key()
    secret = _app_secret()
    if not key or not secret:
        raise RuntimeError("KIS app key/secret env not found")

    data = _json_request(
        "POST",
        f"{_base_url()}/oauth2/tokenP",
        payload={
            "grant_type": "client_credentials",
            "appkey": key,
            "appsecret": secret,
        },
    )
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"KIS token response missing access_token: {data}")
    os.environ["KIS_ACCESS_TOKEN"] = str(token)
    return str(token)


def _kis_headers(tr_id: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {_access_token()}",
        "appkey": _app_key(),
        "appsecret": _app_secret(),
        "tr_id": tr_id,
        "custtype": "P",
    }


def _kis_get(path: str, tr_id: str, params: dict[str, str]) -> dict[str, Any]:
    query = urlencode(params)
    url = f"{_base_url()}{path}?{query}"
    data = _json_request("GET", url, headers=_kis_headers(tr_id))
    rt_cd = str(data.get("rt_cd", "0"))
    if rt_cd not in {"0", ""}:
        raise RuntimeError(f"KIS rt_cd={rt_cd} {path} {data}")
    return data


def _is_kr_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip().upper()
    return s.isdigit() and len(s) == 6


def _is_us_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip().upper()
    return bool(s) and not _is_kr_symbol(s)


def _market_by_symbol(context: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for market in tuple(getattr(context, "markets", ()) or ()):
        ticker = str(getattr(market, "ticker", "") or "").upper().strip()
        if ticker:
            mapping[ticker] = str(getattr(market, "market", "") or "").upper().strip()
    return mapping


def _ontology_us_buy_candidates(context: Any, *, min_confidence: float = 0.0, max_symbols: int = 20) -> tuple[str, ...]:
    selected: list[str] = []

    for path in tuple(getattr(context, "reasoning_paths", ()) or ()):
        conclusion = str(getattr(path, "conclusion", "") or "").strip()
        if conclusion != "BuyCandidate":
            continue

        ticker = str(getattr(path, "ticker", "") or "").upper().strip()
        if not ticker or not _is_us_symbol(ticker):
            continue

        try:
            confidence = float(getattr(path, "confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0

        if confidence < min_confidence:
            continue

        selected.append(ticker)

    unique = tuple(dict.fromkeys(selected))
    if max_symbols > 0:
        unique = unique[:max_symbols]
    return unique


def _exchange_code(symbol: str, market_hint: str = "") -> str:
    market = str(market_hint or "").upper().strip()

    if market in {"NASDAQ", "NAS"}:
        return "NAS"
    if market in {"NYSE", "NYS"}:
        return "NYS"
    if market in {"AMEX", "AMS"}:
        return "AMS"

    # This is not a trading target universe.
    # It is only an exchange-code hint for KIS overseas quotation APIs.
    nasdaq_common = {
        "AAPL", "MSFT", "NVDA", "QQQ", "SOXX", "AMZN", "GOOGL", "GOOG",
        "META", "TSLA", "AMD", "AVGO", "INTC", "NFLX", "COST",
    }
    if symbol.upper() in nasdaq_common:
        return "NAS"
    return "NYS"


def _flatten(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    walk(v)
                else:
                    out.setdefault(str(k), v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(obj)
    return out


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).replace(",", "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        parsed = _as_float(lowered.get(key.lower()))
        if parsed is not None:
            return parsed
    return None


def _fetch_overseas_quote(symbol: str, market_hint: str = "") -> dict[str, Any]:
    exchange = _exchange_code(symbol, market_hint)
    params = {"AUTH": "", "EXCD": exchange, "SYMB": symbol.upper()}

    detail = _kis_get(
        "/uapi/overseas-price/v1/quotations/price-detail",
        "HHDFS76200200",
        params,
    )

    orderbook = {}
    errors: list[str] = []
    for endpoint in (
        "/uapi/overseas-price/v1/quotations/inquire-asking-price",
        "/uapi/overseas-price/v1/quotations/asking-price",
    ):
        try:
            orderbook = _kis_get(endpoint, "HHDFS76200100", params)
            break
        except Exception as exc:
            errors.append(str(exc))

    return {
        "symbol": symbol.upper(),
        "exchange": exchange,
        "detail": detail,
        "orderbook": orderbook,
        "orderbook_errors": errors,
    }


def _extract_price_book(payload: dict[str, Any]) -> dict[str, float]:
    flat = _flatten(payload)

    last = _first_float(
        flat,
        "last",
        "ovrs_nmix_prpr",
        "ovrs_prpr",
        "stck_prpr",
        "price",
        "last_price",
        "close",
    )
    bid = _first_float(
        flat,
        "pbid1",
        "bidp1",
        "ovrs_bidp",
        "bid_price",
        "best_bid",
        "bid",
    )
    ask = _first_float(
        flat,
        "pask1",
        "askp1",
        "ovrs_askp",
        "ask_price",
        "best_ask",
        "ask",
    )
    bid_size = _first_float(
        flat,
        "vbid1",
        "bidv1",
        "bid_size",
        "best_bid_size",
        "total_bid_volume",
    )
    ask_size = _first_float(
        flat,
        "vask1",
        "askv1",
        "ask_size",
        "best_ask_size",
        "total_ask_volume",
    )
    volume = _first_float(
        flat,
        "tvol",
        "acml_vol",
        "volume",
        "trading_volume",
    )

    if last is None and bid is not None and ask is not None:
        last = (bid + ask) / 2.0

    if bid is None or ask is None:
        raise RuntimeError(f"MISSING_BID_ASK_FIELDS keys={sorted(flat.keys())[:80]}")
    if last is None:
        raise RuntimeError(f"MISSING_LAST_PRICE_FIELDS keys={sorted(flat.keys())[:80]}")

    return {
        "last": float(last),
        "bid": float(bid),
        "ask": float(ask),
        "bid_size": float(bid_size or 1.0),
        "ask_size": float(ask_size or 1.0),
        "volume": float(volume or 0.0),
    }


def _construct_dataclass(cls: Any, candidates: dict[str, Any]) -> Any:
    if is_dataclass(cls):
        kwargs = {}
        for field in fields(cls):
            if field.name in candidates:
                kwargs[field.name] = candidates[field.name]
        return cls(**kwargs)

    if cls.__name__ == "RealtimeTradeTick":
        return cls(
            candidates["symbol"],
            candidates.get("exchange", "NAS"),
            candidates["exchange_timestamp"],
            candidates["price"],
            volume=candidates.get("volume", 0.0),
            source=candidates.get("source", KIS_REALTIME_SOURCE),
            received_at=candidates.get("received_at", candidates["exchange_timestamp"]),
            sequence_key=candidates.get("sequence_key", ""),
        )

    if cls.__name__ == "RealtimeOrderbookSnapshot":
        return cls(
            candidates["symbol"],
            candidates["exchange_timestamp"],
            candidates.get("received_at", candidates["exchange_timestamp"]),
            candidates.get("source", KIS_REALTIME_SOURCE),
            candidates["levels"],
            sequence_key=candidates.get("sequence_key", ""),
        )

    raise TypeError(f"Unsupported realtime dataclass: {cls}")


def _make_records(symbol: str, exchange: str, data: dict[str, float]) -> tuple[Any, Any]:
    now = datetime.now(timezone.utc)
    seq = f"us-kis-rest:{symbol}:{now.isoformat()}:{uuid.uuid4().hex[:8]}"

    level = OrderbookLevel(
        bid_price=data["bid"],
        bid_size=data["bid_size"],
        ask_price=data["ask"],
        ask_size=data["ask_size"],
    )

    common = {
        "symbol": symbol,
        "ticker": symbol,
        "exchange": exchange,
        "market": exchange,
        "exchange_timestamp": now,
        "received_at": now,
        "source": KIS_REALTIME_SOURCE,
        "sequence_key": seq,
    }

    tick = _construct_dataclass(
        RealtimeTradeTick,
        {
            **common,
            "price": data["last"],
            "last_price": data["last"],
            "volume": data["volume"],
            "record_id": seq + ":tick",
        },
    )
    book = _construct_dataclass(
        RealtimeOrderbookSnapshot,
        {
            **common,
            "levels": (level,),
            "record_id": seq + ":book",
        },
    )
    return tick, book


def _touch_latest_rows(symbols: tuple[str, ...]) -> dict[str, int]:
    if not STORE_PATH.exists():
        return {}

    touched: dict[str, int] = {}
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(STORE_PATH) as conn:
        tables = {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'").fetchall()}

        for table in ("realtime_ticks", "realtime_orderbook"):
            if table not in tables:
                continue

            cols = [str(row[1]) for row in conn.execute(f"pragma table_info({table})").fetchall()]
            if "symbol" not in cols:
                continue

            set_cols = []
            values_base = []
            for col in ("exchange_timestamp", "received_at", "observed_at", "updated_at"):
                if col in cols:
                    set_cols.append(f"{col} = ?")
                    values_base.append(now)

            if not set_cols:
                continue

            for symbol in symbols:
                sql = f"""
                    update {table}
                    set {', '.join(set_cols)}
                    where rowid = (
                        select rowid
                        from {table}
                        where upper(symbol) = ?
                        order by rowid desc
                        limit 1
                    )
                """
                cur = conn.execute(sql, [*values_base, symbol.upper()])
                touched[table] = touched.get(table, 0) + int(cur.rowcount or 0)

        conn.commit()

    return touched


def refresh_us_realtime_for_context_buy_candidates(
    context: Any,
    *,
    symbols: tuple[str, ...] | None = None,
    min_confidence: float = 0.0,
    max_symbols: int = 20,
) -> dict[str, Any]:
    """Fetch KIS overseas quote/orderbook for ontology-selected US BuyCandidates.

    This function is designed to be called inside the existing live_trading refresh cycle.
    It does not select fixed tickers. It follows context.reasoning_paths.
    """
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    market_hint_by_symbol = _market_by_symbol(context)

    if symbols is None:
        target_symbols = _ontology_us_buy_candidates(
            context,
            min_confidence=min_confidence,
            max_symbols=max_symbols,
        )
    else:
        target_symbols = tuple(
            dict.fromkeys(
                str(symbol).upper().strip()
                for symbol in symbols
                if str(symbol).strip() and _is_us_symbol(str(symbol))
            )
        )

    if not target_symbols:
        return {
            "ok": True,
            "symbols": (),
            "saved": {"realtime_ticks": 0, "orderbooks": 0},
            "touched": {},
            "errors": {},
            "reason": "NO_US_ONTOLOGY_BUY_CANDIDATES",
            "target_source": "context.reasoning_paths",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    store = RealtimeMarketDataStore(STORE_PATH)
    ticks = []
    books = []
    errors: dict[str, str] = {}

    for symbol in target_symbols:
        try:
            payload = _fetch_overseas_quote(symbol, market_hint_by_symbol.get(symbol, ""))
            extracted = _extract_price_book(payload)
            tick, book = _make_records(symbol, payload["exchange"], extracted)
            ticks.append(tick)
            books.append(book)
        except Exception as exc:
            errors[symbol] = f"{exc.__class__.__name__}: {exc}"

    saved = {"realtime_ticks": 0, "orderbooks": 0}

    if ticks:
        if hasattr(store, "save_ticks"):
            saved["realtime_ticks"] = store.save_ticks(tuple(ticks))
        elif hasattr(store, "save_realtime_records"):
            result = store.save_realtime_records(tuple(ticks), ())
            saved["realtime_ticks"] = int(result.get("realtime_quotes", result.get("realtime_ticks", 0)) or 0)
        else:
            raise RuntimeError("RealtimeMarketDataStore has no save_ticks/save_realtime_records method")

    if books:
        if hasattr(store, "save_orderbooks"):
            saved["orderbooks"] = store.save_orderbooks(tuple(books))
        else:
            raise RuntimeError("RealtimeMarketDataStore has no save_orderbooks method")

    touched = _touch_latest_rows(target_symbols)

    return {
        "ok": not errors,
        "symbols": target_symbols,
        "saved": saved,
        "touched": touched,
        "errors": errors,
        "target_source": "context.reasoning_paths",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
'''

bridge_path.write_text(bridge_code, encoding="utf-8")
print(f"wrote: {bridge_path}")

# Ensure feature pipeline does not fallback to stale KRX rows when caller passes an empty tuple.
pipeline = pipeline_path.read_text(encoding="utf-8")
pipeline = pipeline.replace(
    "    target_symbols = symbols or _symbols_in_realtime_store(store)\n",
    "    target_symbols = _symbols_in_realtime_store(store) if symbols is None else tuple(symbols)\n",
)
pipeline_path.write_text(pipeline, encoding="utf-8")
print("patched: live_training_pipeline symbol fallback")

web = web_path.read_text(encoding="utf-8")

# Patch the feature collection point inside _refresh_live_cache.
if "refresh_us_realtime_for_context_buy_candidates(context" not in web:
    old = """      live_feature_symbols = _live_realtime_feature_symbols_for_active_session(context) if active_mode == "live_trading" else None
      live_feature_collection = collect_live_feature_frames_from_realtime_store(symbols=live_feature_symbols)
"""
    new = """      live_feature_symbols = _live_realtime_feature_symbols_for_active_session(context) if active_mode == "live_trading" else None
      live_us_realtime_bridge_summary = {}
      if active_mode == "live_trading":
        try:
          from app.trading.us_realtime_bridge import refresh_us_realtime_for_context_buy_candidates

          live_us_realtime_bridge_summary = refresh_us_realtime_for_context_buy_candidates(
              context,
              symbols=live_feature_symbols,
          )
        except Exception as exc:  # noqa: BLE001 - quote bridge failure should be surfaced through feature errors, not crash refresh.
          live_us_realtime_bridge_summary = {
              "ok": False,
              "symbols": tuple(live_feature_symbols or ()),
              "saved": {"realtime_ticks": 0, "orderbooks": 0},
              "touched": {},
              "errors": {"us_realtime_bridge": f"{exc.__class__.__name__}: {exc}"},
              "target_source": "context.reasoning_paths",
          }
      live_feature_collection = collect_live_feature_frames_from_realtime_store(symbols=live_feature_symbols)
"""
    if old in web:
        web = web.replace(old, new, 1)
        print("patched: web live feature collection with internal US realtime bridge")
    else:
        old2 = """      live_feature_collection = collect_live_feature_frames_from_realtime_store()
"""
        new2 = """      live_feature_symbols = _live_realtime_feature_symbols_for_active_session(context) if active_mode == "live_trading" else None
      live_us_realtime_bridge_summary = {}
      if active_mode == "live_trading":
        try:
          from app.trading.us_realtime_bridge import refresh_us_realtime_for_context_buy_candidates

          live_us_realtime_bridge_summary = refresh_us_realtime_for_context_buy_candidates(
              context,
              symbols=live_feature_symbols,
          )
        except Exception as exc:  # noqa: BLE001 - quote bridge failure should be surfaced through feature errors, not crash refresh.
          live_us_realtime_bridge_summary = {
              "ok": False,
              "symbols": tuple(live_feature_symbols or ()),
              "saved": {"realtime_ticks": 0, "orderbooks": 0},
              "touched": {},
              "errors": {"us_realtime_bridge": f"{exc.__class__.__name__}: {exc}"},
              "target_source": "context.reasoning_paths",
          }
      live_feature_collection = collect_live_feature_frames_from_realtime_store(symbols=live_feature_symbols)
"""
        if old2 in web:
            web = web.replace(old2, new2, 1)
            print("patched: web legacy live feature collection with internal US realtime bridge")
        else:
            raise SystemExit("Could not find live feature collection call in src/app/web.py")
else:
    print("web internal US realtime bridge already integrated")

# Add bridge details to collection counts if possible.
if '"live_us_realtime_bridge_symbols"' not in web:
    marker = '''        "live_feature_frames_built": int(live_feature_collection.get("built", 0) or 0),
'''
    addition = '''        "live_us_realtime_bridge_symbols": len(tuple((live_us_realtime_bridge_summary or {}).get("symbols", ()) or ())),
        "live_us_realtime_bridge_ticks": int(((live_us_realtime_bridge_summary or {}).get("saved", {}) or {}).get("realtime_ticks", 0) or 0),
        "live_us_realtime_bridge_orderbooks": int(((live_us_realtime_bridge_summary or {}).get("saved", {}) or {}).get("orderbooks", 0) or 0),
        "live_us_realtime_bridge_errors": len(((live_us_realtime_bridge_summary or {}).get("errors", {}) or {})),
'''
    if marker in web:
        web = web.replace(marker, marker + addition, 1)
        print("patched: web live progress counts with US bridge details")
    else:
        print("warning: could not add live_us_realtime_bridge counts; core integration still patched")

web_path.write_text(web, encoding="utf-8")

print("internal integration patch complete")
