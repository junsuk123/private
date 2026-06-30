from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from dataclasses import fields, is_dataclass
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import requests

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)

ENV_PATH = Path("config/secrets/kis_api_keys.env")
STORE_PATH = Path("data/store/realtime_market_data.sqlite3")
BASE_URL = "https://openapi.koreainvestment.com:9443"


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_any(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def app_key() -> str:
    return env_any("KIS_APP_KEY", "KIS_APPKEY", "APP_KEY", "KIS_APP_KEY_LIVE")


def app_secret() -> str:
    return env_any("KIS_APP_SECRET", "KIS_APPSECRET", "APP_SECRET", "KIS_APP_SECRET_LIVE")


def base_url() -> str:
    return env_any("KIS_BASE_URL", "KIS_LIVE_BASE_URL", default=BASE_URL).rstrip("/")


def get_access_token() -> str:
    cached = env_any("KIS_ACCESS_TOKEN", "ACCESS_TOKEN")
    if cached:
        return cached

    key = app_key()
    secret = app_secret()
    if not key or not secret:
        raise RuntimeError("KIS app key/secret env not found. Check config/secrets/kis_api_keys.env")

    url = f"{base_url()}/oauth2/tokenP"
    payload = {
        "grant_type": "client_credentials",
        "appkey": key,
        "appsecret": secret,
    }
    res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(payload), timeout=10)
    res.raise_for_status()
    data = res.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"KIS token response missing access_token: {data}")
    os.environ["KIS_ACCESS_TOKEN"] = token
    return token


def headers(tr_id: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {get_access_token()}",
        "appkey": app_key(),
        "appsecret": app_secret(),
        "tr_id": tr_id,
        "custtype": "P",
    }


def kis_get(path: str, tr_id: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{base_url()}{path}"
    res = requests.get(url, headers=headers(tr_id), params=params, timeout=10)
    try:
        data = res.json()
    except Exception:
        data = {"raw_text": res.text}
    if res.status_code >= 400:
        raise RuntimeError(f"KIS HTTP {res.status_code} {path} {data}")
    rt_cd = str(data.get("rt_cd", "0"))
    if rt_cd not in {"0", ""}:
        raise RuntimeError(f"KIS rt_cd={rt_cd} {path} {data}")
    return data


def is_kr_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip().upper()
    return s.isdigit() and len(s) == 6


def is_us_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip().upper()
    return bool(s) and not is_kr_symbol(s)


def us_market_open_now() -> bool:
    from zoneinfo import ZoneInfo

    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    return now_et.weekday() < 5 and dt_time(9, 30) <= now_et.time() <= dt_time(16, 0)


def kr_market_open_now() -> bool:
    from zoneinfo import ZoneInfo

    now_kst = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))
    return now_kst.weekday() < 5 and dt_time(9, 0) <= now_kst.time() <= dt_time(15, 30)


def fetch_json_url(url: str, timeout: float = 10.0) -> Any:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ontology_buy_candidates(
    *,
    api_base: str,
    min_confidence: float,
    market: str,
    max_symbols: int,
) -> tuple[str, ...]:
    """Read current ontology BuyCandidate tickers from the running FastAPI server.

    This is the important part:
    - No fixed trading universe is used.
    - The KIS poller follows ontology reasoning output.
    - Market-session filtering only prevents collecting stale KRX data during US hours.
    """
    api_base = api_base.rstrip("/")
    diagnostics = fetch_json_url(f"{api_base}/api/research/diagnostics")

    paths = diagnostics.get("reasoning_paths") or []
    selected: list[str] = []

    for path in paths:
        conclusion = str(path.get("conclusion") or "").strip()
        if conclusion != "BuyCandidate":
            continue

        ticker = str(path.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        try:
            confidence = float(path.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0

        if confidence < min_confidence:
            continue

        if market == "US" and not is_us_symbol(ticker):
            continue
        if market == "KRX" and not is_kr_symbol(ticker):
            continue

        selected.append(ticker)

    unique = tuple(dict.fromkeys(selected))
    if max_symbols > 0:
        unique = unique[:max_symbols]
    return unique


def resolve_target_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    # Explicit --symbols remains available only for manual debugging.
    if args.symbols:
        return tuple(dict.fromkeys(str(s).upper().strip() for s in args.symbols if str(s).strip()))

    market = args.market.upper()

    if args.from_ontology:
        if market == "AUTO":
            if us_market_open_now():
                market = "US"
            elif kr_market_open_now():
                market = "KRX"
            else:
                market = "US"

        if market != "US":
            raise RuntimeError(
                f"This poller currently supports KIS overseas quote/orderbook bridge only. "
                f"Resolved market={market}. Use a KRX realtime collector for Korean equities."
            )

        symbols = ontology_buy_candidates(
            api_base=args.api_base,
            min_confidence=args.min_confidence,
            market="US",
            max_symbols=args.max_symbols,
        )

        if not symbols:
            raise RuntimeError(
                "NO_ONTOLOGY_US_BUY_CANDIDATES: ontology diagnostics returned no US BuyCandidate symbols. "
                "Do not fallback to hard-coded tickers in live trading."
            )

        return symbols

    raise RuntimeError("No symbols provided and --from-ontology is false. Refusing to use hard-coded live targets.")


def exchange_code(symbol: str) -> str:
    symbol = symbol.upper()

    nasdaq_like = {
        "AAPL", "MSFT", "NVDA", "QQQ", "SOXX", "AMZN", "GOOGL", "GOOG",
        "META", "TSLA", "AMD", "AVGO", "INTC", "NFLX", "COST",
    }
    if symbol in nasdaq_like:
        return "NAS"
    return "NYS"


def flatten(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(str(k), v)
        elif isinstance(value, list):
            for item in value:
                walk(prefix, item)
        else:
            out.setdefault(prefix, value)

    walk("", obj)
    return out


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).replace(",", "").strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        value = lowered.get(key.lower())
        parsed = as_float(value)
        if parsed is not None:
            return parsed
    return None


def fetch_overseas_quote(symbol: str) -> dict[str, Any]:
    params = {"AUTH": "", "EXCD": exchange_code(symbol), "SYMB": symbol.upper()}

    detail = kis_get(
        "/uapi/overseas-price/v1/quotations/price-detail",
        "HHDFS76200200",
        params,
    )

    orderbook = {}
    orderbook_errors: list[str] = []
    for endpoint, tr_id in (
        ("/uapi/overseas-price/v1/quotations/inquire-asking-price", "HHDFS76200100"),
        ("/uapi/overseas-price/v1/quotations/asking-price", "HHDFS76200100"),
    ):
        try:
            orderbook = kis_get(endpoint, tr_id, params)
            break
        except Exception as exc:
            orderbook_errors.append(str(exc))

    return {
        "symbol": symbol.upper(),
        "exchange": exchange_code(symbol),
        "detail": detail,
        "orderbook": orderbook,
        "orderbook_errors": orderbook_errors,
    }


def extract_price_book(payload: dict[str, Any]) -> dict[str, float]:
    flat = flatten(payload)

    last = first_float(
        flat,
        "last",
        "ovrs_nmix_prpr",
        "ovrs_prpr",
        "stck_prpr",
        "price",
        "last_price",
        "close",
    )

    bid = first_float(
        flat,
        "pbid1",
        "bidp1",
        "ovrs_bidp",
        "bid_price",
        "best_bid",
        "bid",
    )

    ask = first_float(
        flat,
        "pask1",
        "askp1",
        "ovrs_askp",
        "ask_price",
        "best_ask",
        "ask",
    )

    bid_size = first_float(
        flat,
        "vbid1",
        "bidv1",
        "bid_size",
        "best_bid_size",
        "total_bid_volume",
    )

    ask_size = first_float(
        flat,
        "vask1",
        "askv1",
        "ask_size",
        "best_ask_size",
        "total_ask_volume",
    )

    volume = first_float(
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

    if bid_size is None or bid_size <= 0:
        bid_size = 1.0
    if ask_size is None or ask_size <= 0:
        ask_size = 1.0
    if volume is None or volume < 0:
        volume = 0.0

    return {
        "last": float(last),
        "bid": float(bid),
        "ask": float(ask),
        "bid_size": float(bid_size),
        "ask_size": float(ask_size),
        "volume": float(volume),
    }


def construct_dataclass(cls: Any, candidates: dict[str, Any]) -> Any:
    if is_dataclass(cls):
        kwargs = {}
        for f in fields(cls):
            if f.name in candidates:
                kwargs[f.name] = candidates[f.name]
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

    raise TypeError(f"Unsupported constructor fallback: {cls}")


def make_records(symbol: str, exchange: str, data: dict[str, float]) -> tuple[Any, Any]:
    now = datetime.now(timezone.utc)
    seq = f"us-kis-rest:{symbol}:{now.isoformat()}:{uuid.uuid4().hex[:8]}"

    level = OrderbookLevel(
        bid_price=data["bid"],
        bid_size=data["bid_size"],
        ask_price=data["ask"],
        ask_size=data["ask_size"],
    )

    tick_candidates = {
        "symbol": symbol,
        "ticker": symbol,
        "exchange": exchange,
        "market": exchange,
        "exchange_timestamp": now,
        "received_at": now,
        "price": data["last"],
        "last_price": data["last"],
        "volume": data["volume"],
        "source": KIS_REALTIME_SOURCE,
        "sequence_key": seq,
        "record_id": seq + ":tick",
    }

    book_candidates = {
        "symbol": symbol,
        "ticker": symbol,
        "exchange": exchange,
        "market": exchange,
        "exchange_timestamp": now,
        "received_at": now,
        "source": KIS_REALTIME_SOURCE,
        "levels": (level,),
        "sequence_key": seq,
        "record_id": seq + ":book",
    }

    tick = construct_dataclass(RealtimeTradeTick, tick_candidates)
    book = construct_dataclass(RealtimeOrderbookSnapshot, book_candidates)
    return tick, book


def touch_latest_rows_for_symbols(symbols: tuple[str, ...]) -> dict[str, int]:
    if not STORE_PATH.exists():
        return {}

    touched: dict[str, int] = {}
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(STORE_PATH) as conn:
        tables = {str(row[0]) for row in conn.execute(
            "select name from sqlite_master where type='table'"
        ).fetchall()}

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
                symbol = str(symbol).upper().strip()
                if not symbol:
                    continue

                values = [*values_base, symbol]
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
                cur = conn.execute(sql, values)
                touched[table] = touched.get(table, 0) + int(cur.rowcount or 0)

        conn.commit()

    return touched


def save_batch(symbols: tuple[str, ...]) -> dict[str, Any]:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = RealtimeMarketDataStore(STORE_PATH)

    ticks = []
    books = []
    errors: dict[str, str] = {}

    for symbol in symbols:
        symbol = symbol.upper().strip()
        if not symbol:
            continue
        try:
            payload = fetch_overseas_quote(symbol)
            extracted = extract_price_book(payload)
            tick, book = make_records(symbol, payload["exchange"], extracted)
            ticks.append(tick)
            books.append(book)
        except Exception as exc:
            errors[symbol] = f"{exc.__class__.__name__}: {exc}"

    saved: dict[str, Any] = {"realtime_ticks": 0, "orderbooks": 0}
    if ticks:
        if hasattr(store, "save_ticks"):
            saved["realtime_ticks"] = store.save_ticks(tuple(ticks))
        elif hasattr(store, "save_realtime_records"):
            result = store.save_realtime_records(tuple(ticks), ())
            saved["realtime_ticks"] = result.get("realtime_quotes", result.get("realtime_ticks", 0))
        else:
            raise RuntimeError("RealtimeMarketDataStore has no save_ticks/save_realtime_records method")

    if books:
        if hasattr(store, "save_orderbooks"):
            saved["orderbooks"] = store.save_orderbooks(tuple(books))
        else:
            raise RuntimeError("RealtimeMarketDataStore has no save_orderbooks method")

    touched = touch_latest_rows_for_symbols(symbols)

    return {
        "ok": not errors,
        "symbols": symbols,
        "saved": saved,
        "touched": touched,
        "errors": errors,
        "store": str(STORE_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Poll KIS overseas stock quote/orderbook for ontology-selected BuyCandidates."
    )
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--from-ontology", action="store_true", default=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--market", choices=["AUTO", "US", "KRX"], default="AUTO")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    while True:
        try:
            symbols = resolve_target_symbols(args)
            result = save_batch(symbols)
            result["target_source"] = "ontology_buy_candidates" if not args.symbols else "explicit_debug_symbols"
        except Exception as exc:
            result = {
                "ok": False,
                "symbols": [],
                "saved": {"realtime_ticks": 0, "orderbooks": 0},
                "touched": {},
                "errors": {"target_resolution": f"{exc.__class__.__name__}: {exc}"},
                "target_source": "ontology_buy_candidates",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        print(json.dumps(result, ensure_ascii=False, indent=2))

        if not args.loop:
            break
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    main()
