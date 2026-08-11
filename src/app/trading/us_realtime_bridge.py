
from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3
import threading
import time
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
from app.execution.kis_auth import build_kis_client
from app.execution.kis_real import load_kis_env_file

STORE_PATH = Path("data/store/realtime_market_data.sqlite3")
BASE_URL = "https://openapi.koreainvestment.com:9443"

_US_MARKET_NAMES = {"US", "NASDAQ", "NAS", "NYSE", "NYS", "AMEX", "AMS", "ARCA", "BATS", "CBOE", "IEX"}
_KR_MARKET_NAMES = {"KRX", "KOSPI", "KOSDAQ", "KONEX"}
_US_POLL_STATE_LOCK = threading.Lock()
_US_POLL_STATE: dict[str, dict[str, float]] = {}


def _env_any(*names: str, default: str = "") -> str:
    load_kis_env_file()
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
    client = build_kis_client(enabled=True)
    data = client._get(path, tr_id, params)
    rt_cd = str(data.get("rt_cd", "0"))
    if rt_cd not in {"0", ""}:
        raise RuntimeError(f"KIS rt_cd={rt_cd} {path} {data}")
    return data


def _clean_overseas_symbol(raw: Any) -> str:
    s = str(raw or "").upper().strip()
    # KIS rsym 형태(예: DNASAAPL = D + NAS + AAPL)에서 거래소 접두사를 제거한다.
    for prefix in ("DNAS", "DNYS", "DAMS", "RNAS", "RNYS", "RAMS", "NAS", "NYS", "AMS"):
        if s.startswith(prefix) and len(s) > len(prefix) and s[len(prefix):].isalpha():
            return s[len(prefix):]
    return s


def fetch_overseas_volume_surge_symbols(
    *,
    exchanges: tuple[str, ...] = ("NAS", "NYS", "AMS"),
    max_symbols: int = 20,
) -> dict[str, Any]:
    """KIS 해외주식 거래량급증(HHDFS76270000)으로 거래량 급증 종목을 받아온다.

    GET /uapi/overseas-stock/v1/ranking/volume-surge. 응답 필드/파라미터는 소스별
    편차가 있어 방어적으로 파싱한다(실패 시 빈 결과). 단타 매수 후보 발굴에 사용.
    """
    selected: list[str] = []
    errors: dict[str, str] = {}
    for exchange in exchanges:
        params = {
            "AUTH": "",
            "EXCD": exchange,
            "MINX": "0",
            "VOL_RANG": "0",
            "KEYB": "",
        }
        try:
            data = _kis_get("/uapi/overseas-stock/v1/ranking/volume-surge", "HHDFS76270000", params)
        except Exception as exc:  # noqa: BLE001 - best-effort discovery; surface per-exchange errors.
            errors[exchange] = f"{exc.__class__.__name__}: {exc}"
            continue
        rows: list[Any] = []
        for key in ("output2", "output1", "output"):
            value = data.get(key)
            if isinstance(value, list) and value:
                rows = value
                break
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = _clean_overseas_symbol(
                row.get("symb") or row.get("SYMB") or row.get("rsym") or row.get("excd_symb") or ""
            )
            if symbol and _is_us_symbol(symbol):
                selected.append(symbol)
    unique = tuple(dict.fromkeys(selected))
    if max_symbols > 0:
        unique = unique[:max_symbols]
    return {"symbols": unique, "errors": errors, "ok": not errors}


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

    if market in {"NASDAQ", "NAS", "NASD"}:
        return "NAS"
    if market in {"NYSE", "NYS"}:
        return "NYS"
    if market in {"AMEX", "AMS"}:
        return "AMS"

    # No usable hint: consult the built ticker→exchange listing map so NYSE/AMEX names
    # are quoted on their real exchange. A wrong-exchange overseas quote returns price 0,
    # which is exactly why affordable NYSE names never yielded a usable tick before.
    try:
        from app.trading.shared_decision_engine import _load_us_listed_exchange_map

        mapped = _load_us_listed_exchange_map().get(str(symbol or "").upper().strip())
        if mapped == "NYSE":
            return "NYS"
        if mapped == "AMEX":
            return "AMS"
        if mapped == "NASD":
            return "NAS"
    except Exception:  # noqa: BLE001 - map is best-effort; fall back to the default.
        pass

    return _env_any("KIS_DEFAULT_US_QUOTE_EXCHANGE", default="NAS").upper()


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
        # KIS HDFSASP0 / HHDFS76200100: p*=price, v*=size, d*=change.
        # Treating dbid1/dask1 as prices admitted values such as 5/706 for a
        # $16 stock and poisoned both live features and forward labels.
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
        "bvol",
        "bid_size",
        "best_bid_size",
        "total_bid_volume",
    )
    ask_size = _first_float(
        flat,
        "vask1",
        "askv1",
        "avol",
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
    if bid <= 0 or ask <= 0 or ask < bid:
        raise RuntimeError(f"INVALID_BID_ASK_FIELDS bid={bid} ask={ask} keys={sorted(flat.keys())[:80]}")
    if last is None:
        raise RuntimeError(f"MISSING_LAST_PRICE_FIELDS keys={sorted(flat.keys())[:80]}")
    midpoint = (bid + ask) / 2.0
    spread_bps = (ask - bid) / midpoint * 10_000.0
    try:
        maximum_source_spread_bps = max(
            100.0,
            float(os.getenv("REALTIME_US_REST_MAX_SOURCE_SPREAD_BPS", "2000")),
        )
    except (TypeError, ValueError):
        maximum_source_spread_bps = 2000.0
    if spread_bps > maximum_source_spread_bps:
        raise RuntimeError(
            "IMPLAUSIBLE_BID_ASK_SPREAD "
            f"bid={bid} ask={ask} last={last} spread_bps={spread_bps:.2f}"
        )

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


_US_BRIDGE_VENUES = {
    "NAS": "NASDAQ", "BAQ": "NASDAQ",
    "NYS": "NYSE", "BAY": "NYSE",
    "AMS": "AMEX", "BAA": "AMEX",
}


def _rest_quote_metadata(exchange: str, now: datetime) -> Any:
    """이 브릿지가 만드는 이벤트의 출처 metadata.

    **이 경로는 KIS 해외 REST 시세다** (``sequence_key`` 가 ``us-kis-rest:`` 로 시작하고,
    거래량이 세션 누적값이라 체결 단위가 아니다). 그래서 ``feed_scope`` 는
    ``REST_SNAPSHOT`` 이다 — WebSocket 체결로 위장하지 않는다.

    ``source`` 문자열은 여기서 바꾸지 않는다. 기존 live-buy 판정
    (``live_feature_frame`` / ``market_data_health`` / ``realtime_trading_engine``) 이
    ``source == KIS_REALTIME_SOURCE`` 를 직접 비교하고 있어서, 그 값을 지금 바꾸면 실거래
    동작이 함께 바뀐다. metadata 는 그 사실을 **표현** 하고, 판정 전환은 별도 단계다.
    ``docs/realtime_session_gap_analysis.md`` 의 남은 작업 참조.
    """
    from app.data.market_capabilities import (
        FeedScope,
        MarketGroup,
        SessionId,
        Venue,
        default_service,
    )
    from app.data.realtime_types import FeedMetadata

    code = str(exchange or "").upper().strip()
    venue_name = _US_BRIDGE_VENUES.get(code)
    venue = Venue(venue_name) if venue_name else Venue.UNKNOWN
    service = default_service()
    session = SessionId.UNKNOWN
    for capability in service.active_capabilities(MarketGroup.US, now):
        session = capability.session
        break
    return FeedMetadata(
        market_group=MarketGroup.US,
        exchange={"NASDAQ": "NASD", "NYSE": "NYSE", "AMEX": "AMEX"}.get(venue_name or "", ""),
        venue=venue,
        session=session,
        currency="USD",
        feed_scope=FeedScope.REST_SNAPSHOT,
        tr_id="",
        subscription_key=code,
        is_consolidated=False,
        # REST 스냅샷은 실시간 신규매수 근거가 될 수 없다.
        is_tradeable=False,
        metadata_inferred=False,
    )


def _make_records(symbol: str, exchange: str, data: dict[str, float]) -> tuple[Any | None, Any]:
    now = datetime.now(timezone.utc)
    seq = f"us-kis-rest:{symbol}:{now.isoformat()}:{uuid.uuid4().hex[:8]}"
    cumulative_volume = max(0.0, float(data.get("volume") or 0.0))
    last_price = float(data["last"])
    with _US_POLL_STATE_LOCK:
        previous = dict(_US_POLL_STATE.get(symbol) or {})
        _US_POLL_STATE[symbol] = {
            "cumulative_volume": cumulative_volume,
            "last_price": last_price,
        }
    previous_volume = previous.get("cumulative_volume")
    previous_price = previous.get("last_price")
    volume_delta = 0.0
    if previous_volume is not None and cumulative_volume >= previous_volume:
        volume_delta = cumulative_volume - previous_volume
    price_changed = previous_price is not None and last_price != previous_price
    # KIS overseas REST quotes expose cumulative session volume, not the size of
    # a new trade. Do not manufacture a trade tick on every poll. The first
    # observation seeds state; later observations emit only on actual volume or
    # last-price movement.
    emit_tick = previous_volume is not None and (volume_delta > 0.0 or price_changed)

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
        "meta": _rest_quote_metadata(exchange, now),
    }

    tick = (
        _construct_dataclass(
            RealtimeTradeTick,
            {
                **common,
                "price": last_price,
                "last_price": last_price,
                "volume": volume_delta,
                "record_id": seq + ":tick",
            },
        )
        if emit_tick
        else None
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

    with closing(sqlite3.connect(STORE_PATH)) as conn:
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
    try:
        symbol_delay_seconds = max(
            0.0,
            min(2.0, float(os.getenv("REALTIME_US_REST_SYMBOL_DELAY_SEC", "0.4"))),
        )
    except (TypeError, ValueError):
        symbol_delay_seconds = 0.4

    for index, symbol in enumerate(target_symbols):
        if index > 0 and symbol_delay_seconds > 0.0:
            # KIS applies a shared per-second request budget across account,
            # quote, orderbook and ranking endpoints. Spread symbol bursts so
            # the fourth candidate is not predictably rejected.
            time.sleep(symbol_delay_seconds)
        try:
            payload = _fetch_overseas_quote(symbol, market_hint_by_symbol.get(symbol, ""))
            extracted = _extract_price_book(payload)
            tick, book = _make_records(symbol, payload["exchange"], extracted)
            if tick is not None:
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

    if ticks or books:
        from app.data.market_data_health import evaluate_market_data_health

        for symbol in target_symbols:
            evaluate_market_data_health(
                store,
                symbol,
                max_quote_age_ms=15_000,
                max_orderbook_age_ms=15_000,
            )
            store.build_latest_minute_bar(symbol)

    # Never rewrite timestamps on historical rows. A failed REST refresh must
    # remain visibly stale instead of making old data look current.
    touched: dict[str, int] = {}

    return {
        "ok": not errors,
        "symbols": target_symbols,
        "saved": saved,
        "touched": touched,
        "errors": errors,
        "target_source": "context.reasoning_paths",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
