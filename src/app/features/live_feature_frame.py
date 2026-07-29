from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.data.market_data_health import evaluate_market_data_health
from app.data.realtime_types import KIS_REALTIME_SOURCE
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.feature_provenance import FeatureProvenance
from app.features.feature_schema import FeatureSchema, LIVE_SHORT_HORIZON_SCHEMA
from app.features.news_sentiment_store import NewsSentimentStore


class FeatureFrameError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveFeatureFrame:
    symbol: str
    decision_time: datetime
    schema: FeatureSchema
    values: tuple[float, ...]
    provenance: FeatureProvenance
    mark_price: float = 0.0
    mark_source: str = "unknown"

    @property
    def feature_schema_hash(self) -> str:
        return self.schema.schema_hash

    def as_feature_dict(self) -> dict[str, float]:
        return dict(zip(self.schema.feature_names, self.values, strict=True))

    def validate(self) -> None:
        if len(self.values) != len(self.schema.feature_names):
            raise FeatureFrameError("FEATURE_COUNT_MISMATCH")
        if any(not math.isfinite(value) for value in self.values):
            raise FeatureFrameError("FEATURE_NAN_OR_INF")
        if self.provenance.source != "kis_realtime_websocket":
            raise FeatureFrameError("FEATURE_SOURCE_NOT_KIS_REALTIME")


class LiveFeatureFrameBuilder:
    def __init__(
        self,
        store: RealtimeMarketDataStore,
        *,
        schema: FeatureSchema = LIVE_SHORT_HORIZON_SCHEMA,
        max_quote_age_ms: int = int(os.getenv("LIVE_FEATURE_MAX_QUOTE_AGE_MS", "15000")),
        max_orderbook_age_ms: int = int(os.getenv("LIVE_FEATURE_MAX_ORDERBOOK_AGE_MS", "15000")),
        # US quotes/orderbooks are REST-polled once per live-trading refresh cycle
        # (tens of seconds apart), not streamed sub-second like the KRX websocket.
        # A single 15s gate marks a healthy US REST feed permanently stale and starves
        # US buys. Give US a cadence-aligned freshness window; KR keeps the tight gate.
        # This only aligns the freshness gate with the feed cadence — it does NOT relax
        # any spread/cost/liquidity/instrument risk gate.
        max_quote_age_ms_us: int = int(os.getenv("LIVE_FEATURE_MAX_QUOTE_AGE_MS_US", "90000")),
        max_orderbook_age_ms_us: int = int(os.getenv("LIVE_FEATURE_MAX_ORDERBOOK_AGE_MS_US", "90000")),
        journal_path: str | Path = "logs/live-feature-frames.jsonl",
        sentiment_store: NewsSentimentStore | None = None,
    ) -> None:
        self.store = store
        self.schema = schema
        try:
            self.sentiment_store = sentiment_store or NewsSentimentStore()
        except Exception:  # noqa: BLE001 - sentiment is optional; never block frame building.
            self.sentiment_store = None
        self.max_quote_age_ms = max_quote_age_ms
        self.max_orderbook_age_ms = max_orderbook_age_ms
        self.max_quote_age_ms_us = max_quote_age_ms_us
        self.max_orderbook_age_ms_us = max_orderbook_age_ms_us
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_us_symbol(symbol: str) -> bool:
        # KRX realtime uses 6-digit numeric codes; US symbols are alphabetic.
        s = str(symbol or "").strip().upper()
        return not (s.isdigit() and len(s) == 6)

    def build(self, symbol: str, *, decision_time: datetime | None = None) -> LiveFeatureFrame:
        explicit_decision_time = decision_time is not None
        decision_time = decision_time or datetime.now(timezone.utc)
        is_us = self._is_us_symbol(symbol)
        quote_age_ms = self.max_quote_age_ms_us if is_us else self.max_quote_age_ms
        orderbook_age_ms = self.max_orderbook_age_ms_us if is_us else self.max_orderbook_age_ms
        since = decision_time - timedelta(minutes=3)
        historical = (
            explicit_decision_time
            and decision_time < datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        ticks = self.store.recent_ticks(
            symbol,
            since,
            until=decision_time if historical else None,
        )
        orderbooks = self.store.recent_orderbooks(
            symbol,
            since,
            until=decision_time if historical else None,
        )
        orderbook = orderbooks[-1] if orderbooks else None
        if historical:
            validation_ticks = ticks
            validation_orderbook = orderbook
            if not validation_ticks:
                prior_ticks = self.store.recent_ticks(
                    symbol,
                    decision_time - timedelta(days=1),
                    until=decision_time,
                )
                validation_ticks = prior_ticks[-1:] if prior_ticks else ()
            if validation_orderbook is None:
                prior_books = self.store.recent_orderbooks(
                    symbol,
                    decision_time - timedelta(days=1),
                    until=decision_time,
                )
                validation_orderbook = prior_books[-1] if prior_books else None
            self._validate_historical_sources(
                validation_ticks,
                validation_orderbook,
                decision_time,
                quote_age_ms,
                orderbook_age_ms,
            )
        else:
            health = evaluate_market_data_health(
                self.store,
                symbol,
                max_quote_age_ms=quote_age_ms,
                max_orderbook_age_ms=orderbook_age_ms,
                now=decision_time,
            )
            if not health.ok_for_live_buy:
                raise FeatureFrameError(
                    "MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE:" + ",".join(health.reason_codes)
                )
            orderbook = self.store.latest_orderbook(symbol)
        if orderbook is None:
            raise FeatureFrameError("MISSING_SOURCE_RECORDS")
        if ticks:
            prices = [tick.price for tick in ticks]
            volumes = [max(0, tick.volume) for tick in ticks]
            source_record_ids = tuple(tick.record_id for tick in ticks)
            latest_source_received_at = ticks[-1].received_at
        elif orderbooks:
            valid_books = tuple(book for book in orderbooks if book.best_bid > 0 and book.best_ask > 0)
            prices = [(book.best_bid + book.best_ask) / 2 for book in valid_books]
            volumes = [max(0, book.total_bid_volume + book.total_ask_volume) for book in valid_books]
            source_record_ids = tuple(book.record_id for book in valid_books)
            latest_source_received_at = valid_books[-1].received_at if valid_books else orderbook.received_at
        else:
            raise FeatureFrameError("MISSING_SOURCE_RECORDS")
        if not prices:
            raise FeatureFrameError("MISSING_SOURCE_RECORDS")
        total_volume = sum(volumes)
        vwap = (
            sum(price * volume for price, volume in zip(prices, volumes, strict=True)) / total_volume
            if total_volume > 0
            else prices[-1]
        )
        vol = _stdev(_returns(prices))
        bid_depth = float(orderbook.total_bid_volume)
        ask_depth = float(orderbook.total_ask_volume)
        depth_ratio = bid_depth / max(1.0, ask_depth)
        technical = _technical_columns(prices, volumes)
        second_features = _second_level_features(
            ticks,
            orderbooks,
            decision_time,
        )
        feature_dict = {
            **second_features,
            "return_30s": _window_return(ticks, decision_time, seconds=30),
            "return_1m": _window_return(ticks, decision_time, seconds=60),
            "return_3m": _safe_return(prices[-1], prices[0]),
            "distance_from_vwap": _safe_return(prices[-1], vwap),
            "spread_bps": orderbook.spread_bps,
            "orderbook_imbalance": orderbook.imbalance,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_ratio": depth_ratio,
            "liquidity_score": min(1.0, math.log1p(total_volume) / math.log1p(1_000_000)),
            "realized_volatility_3m": vol,
            "max_drop_3m": min((_safe_return(price, prices[0]) for price in prices), default=0.0),
            "cost_to_volatility_ratio": (orderbook.spread_bps / 10_000.0) / max(vol, 1e-6),
            "principal_cushion_ratio": 1.0,
            "news_sentiment": self._news_sentiment(symbol, decision_time),
            **technical,
        }
        values = tuple(float(feature_dict[name]) for name in self.schema.feature_names)
        provenance = FeatureProvenance(
            symbol=symbol,
            decision_time=decision_time,
            tick_record_ids=source_record_ids,
            orderbook_record_id=orderbook.record_id,
            source="kis_realtime_websocket",
            max_input_age_ms=max(
                (decision_time - latest_source_received_at).total_seconds() * 1000,
                (decision_time - orderbook.received_at).total_seconds() * 1000,
            ),
        )
        frame = LiveFeatureFrame(
            symbol,
            decision_time,
            self.schema,
            values,
            provenance,
            float(prices[-1]),
            "tick" if ticks else "orderbook_mid",
        )
        frame.validate()
        self._journal(frame)
        return frame

    @staticmethod
    def _validate_historical_sources(
        ticks: tuple,
        orderbook: object | None,
        decision_time: datetime,
        quote_age_ms: int,
        orderbook_age_ms: int,
    ) -> None:
        quote = ticks[-1] if ticks else orderbook
        reasons: list[str] = []
        if quote is None:
            reasons.append("QUOTE_COUNT_ZERO")
        else:
            quote_age = (decision_time - quote.received_at).total_seconds() * 1000
            if quote_age < 0 or quote_age > quote_age_ms:
                reasons.append("QUOTE_STALE")
            if quote.source != KIS_REALTIME_SOURCE:
                reasons.append("QUOTE_SOURCE_NOT_KIS_REALTIME")
        if orderbook is None:
            reasons.append("ORDERBOOK_COUNT_ZERO")
        else:
            book_age = (decision_time - orderbook.received_at).total_seconds() * 1000
            if book_age < 0 or book_age > orderbook_age_ms:
                reasons.append("ORDERBOOK_STALE")
            if orderbook.source != KIS_REALTIME_SOURCE:
                reasons.append("ORDERBOOK_SOURCE_NOT_KIS_REALTIME")
        if reasons:
            raise FeatureFrameError(
                "MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE:" + ",".join(reasons)
            )

    def _news_sentiment(self, symbol: str, decision_time: datetime) -> float:
        """Recency-decayed news sentiment as of decision_time (0.0 = no fresh news).

        No lookahead (only news observed on/before decision_time) and best-effort:
        any failure yields neutral 0.0 so the market-data frame is never blocked.
        """
        if self.sentiment_store is None:
            return 0.0
        try:
            value = self.sentiment_store.score_as_of(symbol, decision_time)
        except Exception:  # noqa: BLE001 - never let optional sentiment break a frame.
            return 0.0
        return value if math.isfinite(value) else 0.0

    def _journal(self, frame: LiveFeatureFrame) -> None:
        payload = {
            "symbol": frame.symbol,
            "decision_time": frame.decision_time.isoformat(),
            "feature_schema_hash": frame.feature_schema_hash,
            "mark_price": frame.mark_price,
            "mark_source": frame.mark_source,
            "source_record_ids": frame.provenance.source_record_ids,
            "values": frame.as_feature_dict(),
        }
        with self.journal_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _technical_columns(prices: list[float], volumes: list[float]) -> dict[str, float]:
    """Evidence-based technical indicator columns for the live schema.

    Computed from the same realtime tick price/volume series used for the other
    columns and delegated to ``app.technical.indicators`` (single source of TA
    truth). Every value is coerced to a NEUTRAL finite default when the
    indicator is unavailable (short data), so a live frame never fails
    validation because of these additions.
    """
    from app.technical import indicators as ti

    last = prices[-1] if prices else 0.0
    ema_fast = ti.ema(prices, 12)
    ema_slow = ti.ema(prices, 26)
    ema_gap_bps = (
        (ema_fast - ema_slow) / last * 10_000.0
        if (ema_fast is not None and ema_slow is not None and last > 0)
        else 0.0
    )
    macd_res = ti.macd(prices)
    boll = ti.bollinger(prices, 20)
    donch = ti.donchian(prices, 20, prices)
    donchian_breakout = (
        (last - donch.high) / last if (donch.ok and donch.high and last > 0) else 0.0
    )
    return {
        "rsi_14": _finite_or(ti.rsi(prices, 14), 50.0),
        "macd_histogram": _finite_or(macd_res.histogram if macd_res.ok else None, 0.0),
        "bollinger_percent_b": _finite_or(boll.percent_b if boll.ok else None, 0.5),
        "ema_gap_bps": _finite_or(ema_gap_bps, 0.0),
        "donchian_breakout": _finite_or(donchian_breakout, 0.0),
        "volume_spike_ratio": _finite_or(ti.volume_spike_ratio(volumes, 20), 1.0),
    }


def _finite_or(value: float | None, default: float) -> float:
    return float(value) if (value is not None and math.isfinite(value)) else default


def _safe_return(current: float, previous: float) -> float:
    return 0.0 if previous <= 0 else current / previous - 1.0


def _window_return(ticks: tuple, decision_time: datetime, *, seconds: int) -> float:
    cutoff = decision_time - timedelta(seconds=seconds)
    visible = tuple(tick for tick in ticks if tick.exchange_timestamp >= cutoff)
    if len(visible) < 2:
        return 0.0
    return _safe_return(visible[-1].price, visible[0].price)


def _second_level_features(
    ticks: tuple,
    orderbooks: tuple,
    decision_time: datetime,
) -> dict[str, float]:
    """Compute entry-timing features from true event timestamps.

    Counts and volume are limited to their exact trailing windows. Values are
    finite even when the feed is sparse, while ``second_data_ready`` preserves
    the distinction between a neutral signal and missing microstructure data.
    """
    tick_1s = _events_in_window(ticks, decision_time, 1)
    tick_5s = _events_in_window(ticks, decision_time, 5)
    tick_10s = _events_in_window(ticks, decision_time, 10)
    books_5s = _events_in_window(orderbooks, decision_time, 5)
    returns_10s = _returns([float(tick.price) for tick in tick_10s])
    buy_volume = sum(
        max(0.0, float(tick.volume))
        for tick in tick_5s
        if str(tick.trade_direction or "").upper() in {"BUY", "B"}
    )
    sell_volume = sum(
        max(0.0, float(tick.volume))
        for tick in tick_5s
        if str(tick.trade_direction or "").upper() in {"SELL", "S"}
    )
    directed_volume = buy_volume + sell_volume
    spread_change = 0.0
    imbalance_change = 0.0
    if len(books_5s) >= 2:
        spread_change = (
            float(books_5s[-1].spread_bps) - float(books_5s[0].spread_bps)
        ) / 100.0
        imbalance_change = float(books_5s[-1].imbalance) - float(books_5s[0].imbalance)
    unique_seconds = {
        tick.exchange_timestamp.replace(microsecond=0)
        for tick in tick_10s
    }
    return {
        "return_1s": _event_window_return(tick_1s),
        "return_5s": _event_window_return(tick_5s),
        "return_10s": _event_window_return(tick_10s),
        "tick_count_1s": float(len(tick_1s)),
        "tick_count_5s": float(len(tick_5s)),
        "volume_1s_log": math.log1p(sum(max(0.0, float(tick.volume)) for tick in tick_1s)),
        "volume_5s_log": math.log1p(sum(max(0.0, float(tick.volume)) for tick in tick_5s)),
        "aggressor_imbalance_5s": (
            (buy_volume - sell_volume) / directed_volume if directed_volume > 0 else 0.0
        ),
        "realized_volatility_10s": _stdev(returns_10s),
        "spread_change_5s": spread_change,
        "orderbook_imbalance_change_5s": imbalance_change,
        "second_data_ready": 1.0 if len(tick_10s) >= 3 and len(unique_seconds) >= 2 and books_5s else 0.0,
    }


def _events_in_window(events: tuple, decision_time: datetime, seconds: int) -> tuple:
    cutoff = decision_time - timedelta(seconds=seconds)
    return tuple(
        event
        for event in events
        if cutoff <= event.exchange_timestamp <= decision_time
    )


def _event_window_return(ticks: tuple) -> float:
    if len(ticks) < 2:
        return 0.0
    return _safe_return(float(ticks[-1].price), float(ticks[0].price))


def _returns(prices: list[float]) -> list[float]:
    return [_safe_return(prices[index], prices[index - 1]) for index in range(1, len(prices))]


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
