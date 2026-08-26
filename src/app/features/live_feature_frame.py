from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from app.audit import log_path
from app.data.market_data_health import evaluate_market_data_health
from app.data.realtime_types import KIS_REALTIME_SOURCE
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.feature_provenance import FeatureProvenance
from app.features.feature_schema import FeatureSchema, LIVE_SHORT_HORIZON_SCHEMA
from app.features.news_sentiment_store import NewsSentimentStore
from app.features import session_structure


_FEATURE_JOURNAL_LOCK = threading.Lock()


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
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    #: Named quantities the builder computed but the model schema does not carry.
    #:
    #: ``values`` is positional against ``schema.feature_names``, so anything not
    #: in that list was previously computed and dropped on the floor. The GNN
    #: context adapter then re-read those names out of ``as_feature_dict()`` and
    #: got a silent default: ``box_high``, ``box_low``, ``box_mid``,
    #: ``box_previous_close`` and ``realized_volatility_3m`` were all being
    #: served as 0.0 while training supplied real values for the same slots.
    #: Keeping them here costs nothing (they are already computed) and lets a
    #: second consumer read them BY NAME instead of by a lucky schema position.
    extras: Mapping[str, float] = field(default_factory=dict)

    @property
    def feature_schema_hash(self) -> str:
        return self.schema.schema_hash

    def as_feature_dict(self) -> dict[str, float]:
        return dict(zip(self.schema.feature_names, self.values, strict=True))

    def as_context_dict(self) -> dict[str, float]:
        """Schema values plus the named extras, for non-model consumers.

        Schema values win on a name collision: the vector the live model scored
        is the authority for anything that is in it.
        """
        return {
            **{str(name): float(value) for name, value in self.extras.items()},
            **self.as_feature_dict(),
        }

    def validate(self) -> None:
        if len(self.values) != len(self.schema.feature_names):
            raise FeatureFrameError("FEATURE_COUNT_MISMATCH")
        if any(not math.isfinite(value) for value in self.values):
            raise FeatureFrameError("FEATURE_NAN_OR_INF")
        if self.provenance.source != "kis_realtime_websocket":
            raise FeatureFrameError("FEATURE_SOURCE_NOT_KIS_REALTIME")


class LiveFeatureFrameBuilder:
    @staticmethod
    def historical_dependencies(*, market: str):
        """Declare completed-bar needs to the demand-driven warmup coordinator."""
        from app.data.minute_bar_warmup import HistoricalDependency
        from app.features.strategy_graph_context import SESSION_CONTEXT_BARS

        return (
            HistoricalDependency(
                component="feature:live_feature_frame",
                timeframe_minutes=1,
                minimum_observations=64,
                preferred_observations=SESSION_CONTEXT_BARS,
                required_fields=(
                    "open", "high", "low", "close", "volume", "spread_bps",
                    "orderbook_imbalance", "liquidity_score",
                ),
                market_constraints=(str(market).upper(),),
            ),
        )

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
        journal_path: str | Path | None = None,
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
        self.journal_path = (
            Path(journal_path) if journal_path is not None
            else log_path("live-feature-frames.jsonl")
        )
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
        # ONE minute-bar fetch per frame, shared by every slow-context consumer.
        slow_bars, slow_rows = _slow_context_bars(self.store, symbol, decision_time)
        rvgi_box = _rvgi_box_columns(
            slow_bars,
            symbol,
            decision_time,
            float(prices[-1]),
        )
        # Session structure is exchange-aware and works for both KRX and US
        # minute bars. Dropping it for US made every opening/gap strategy
        # permanently context-less in the market where those low-turnover
        # strategies are most useful.
        session_diagnostics = _session_structure_diagnostics(
            self.store, symbol, decision_time
        )
        second_features = _second_level_features(
            ticks,
            orderbooks,
            decision_time,
        )
        # The entry/strategy horizon is measured in minutes, so its technical
        # context must come from completed one-minute bars.  The old mapper used
        # indicators calculated from the trailing three-minute tick window.  That
        # is fine for a websocket, but a US REST feed contributes only a handful
        # of observations and therefore emitted the synthetic neutral tuple
        # (EMA gap 0, RSI 50, volume ratio 1, volatility unavailable) for every
        # symbol.  Preserve the fast columns for the short-horizon model and carry
        # a separate causal slow context for micro/strategy election.
        slow_technical = _slow_technical_columns(
            slow_bars,
            symbol=symbol,
            orderbook=orderbook,
            price=float(prices[-1]),
        )
        # Slow indicator families, computed from COMPLETED fixed-time bars only.
        # Placed after second_features so aggressor flow can join the volume family.
        family_columns = _indicator_family_columns(
            slow_bars,
            spread_bps=float(orderbook.spread_bps),
            liquidity_score=technical.get("liquidity_score"),
            orderbook_imbalance=technical.get("orderbook_imbalance"),
            aggressor_imbalance=second_features.get("aggressor_imbalance_5s"),
        )
        feature_dict = {
            **second_features,
            **family_columns,
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
            **rvgi_box,
        }
        values = tuple(float(feature_dict[name]) for name in self.schema.feature_names)
        # Everything computed above that the model schema does not carry, kept by
        # name instead of discarded, plus the bar-derived GNN context contract.
        # The strategy-graph fields are namespaced so they can never collide with
        # a tick-window column of a similar name: ``spread_bps`` (instantaneous
        # book) and ``graph:spread_bps_scaled`` (last completed bar) are different
        # measurements, and conflating them is the class of bug this fixes.
        graph_context = _strategy_graph_context_columns(
            slow_bars,
            slow_rows,
            symbol=symbol,
            rvgi_box=rvgi_box,
        )
        schema_names = set(self.schema.feature_names)
        extras = {
            name: float(value)
            for name, value in feature_dict.items()
            if name not in schema_names
        }
        extras.update(
            {f"graph:{name}": float(value) for name, value in graph_context.items()}
        )
        extras.update(
            {f"slow_technical:{name}": float(value) for name, value in slow_technical.items()}
        )
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
            session_diagnostics,
            extras,
        )
        frame.validate()
        # Book depth is journalled as QUALITY metadata, deliberately outside
        # ``values``. The labelling pipeline uses it to reject a malformed or
        # one-sided book before that book can become a return label, which is a
        # data-integrity question, not a model input. Raw depths were removed from
        # the model vector in schema v5 because they encode instrument identity
        # (between-symbol variance ratio 0.986/0.984) -- and because the gates read
        # them out of ``values``, that removal silently made every frame fail
        # ``bid_depth > 0`` and froze row materialization. Keeping the two concerns
        # in separate fields is what stops a feature-set change from disabling a
        # quality check again.
        self._journal(
            frame,
            book_quality={
                "bid_depth": float(feature_dict.get("bid_depth", 0.0)),
                "ask_depth": float(feature_dict.get("ask_depth", 0.0)),
            },
        )
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

    def _journal(
        self,
        frame: LiveFeatureFrame,
        *,
        book_quality: dict[str, float] | None = None,
    ) -> None:
        try:
            provenance_limit = max(
                1,
                int(os.getenv("LIVE_FEATURE_JOURNAL_PROVENANCE_ID_LIMIT", "64")),
            )
        except (TypeError, ValueError):
            provenance_limit = 64
        payload = {
            "symbol": frame.symbol,
            "decision_time": frame.decision_time.isoformat(),
            "feature_schema_hash": frame.feature_schema_hash,
            "mark_price": frame.mark_price,
            "mark_source": frame.mark_source,
            # Full provenance remains in the realtime SQLite store. Persisting
            # every tick ID here made each transient journal row tens of KB.
            "source_record_ids": frame.provenance.source_record_ids[-provenance_limit:],
            "values": frame.as_feature_dict(),
            "book_quality": dict(book_quality or {}),
            "diagnostics": dict(frame.diagnostics),
        }
        try:
            maximum_bytes = max(
                16 * 1024 * 1024,
                int(float(os.getenv("LIVE_FEATURE_JOURNAL_MAX_BYTES", str(256 * 1024 * 1024)))),
            )
        except (TypeError, ValueError):
            maximum_bytes = 256 * 1024 * 1024
        with _FEATURE_JOURNAL_LOCK:
            if self.journal_path.exists() and self.journal_path.stat().st_size >= maximum_bytes:
                # Labelled rows have already been materialized. Atomic rotation
                # keeps collection continuous and bounds this transient source log.
                rotated = self.journal_path.with_suffix(self.journal_path.suffix + ".rotated")
                try:
                    if rotated.exists():
                        rotated.unlink()
                    os.replace(self.journal_path, rotated)
                    rotated.unlink(missing_ok=True)
                except OSError:
                    pass
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
    visible = _events_in_window(ticks, decision_time, seconds)
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
    classified = _classified_trade_directions(tick_5s, orderbooks)
    buy_volume = sum(
        max(0.0, float(tick.volume))
        for tick, direction in classified
        if direction == "BUY"
    )
    sell_volume = sum(
        max(0.0, float(tick.volume))
        for tick, direction in classified
        if direction == "SELL"
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
        tick.received_at.replace(microsecond=0)
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
        "source_latency_ms_10s": (
            max(float(getattr(tick, "latency_ms", 0.0) or 0.0) for tick in tick_10s)
            if tick_10s
            else 0.0
        ),
        "ingestion_span_seconds_10s": (
            max(0.0, (tick_10s[-1].received_at - tick_10s[0].received_at).total_seconds())
            if len(tick_10s) >= 2
            else 0.0
        ),
        # Overseas HDFSCNT0 is event-driven but quiet names commonly print only
        # twice in five seconds. Two distinct prints plus a contemporaneous book
        # resolve direction/return; volatility edge falls back to completed 1m
        # history when a third print is absent.
        "second_data_ready": 1.0 if len(tick_10s) >= 2 and len(unique_seconds) >= 2 and books_5s else 0.0,
    }


def _classified_trade_directions(ticks: tuple, orderbooks: tuple) -> tuple[tuple[Any, str | None], ...]:
    """Broker direction, then causal quote test, then tick rule.

    HDFSCNT0 labels only trades at/through its bid or ask. Prints inside the
    spread therefore arrived as directionless and every flow strategy saw an
    aggressor imbalance of exactly zero. The quote/tick rule is causal: only a
    book at or before the trade and only the preceding trade are consulted.
    """
    books = sorted(orderbooks, key=lambda item: item.exchange_timestamp)
    ordered_ticks = sorted(ticks, key=lambda item: item.exchange_timestamp)
    result: list[tuple[Any, str | None]] = []
    book_index = 0
    latest_book = None
    previous_price: float | None = None
    for tick in ordered_ticks:
        while (
            book_index < len(books)
            and books[book_index].exchange_timestamp <= tick.exchange_timestamp
        ):
            latest_book = books[book_index]
            book_index += 1
        raw = str(tick.trade_direction or "").upper()
        direction = "BUY" if raw in {"BUY", "B"} else "SELL" if raw in {"SELL", "S"} else None
        price = float(tick.price)
        if direction is None and latest_book is not None:
            bid = float(latest_book.best_bid)
            ask = float(latest_book.best_ask)
            mid = (bid + ask) / 2.0 if bid > 0 and ask >= bid else 0.0
            if mid > 0:
                direction = "BUY" if price > mid else "SELL" if price < mid else None
        if direction is None and previous_price is not None:
            direction = "BUY" if price > previous_price else "SELL" if price < previous_price else None
        result.append((tick, direction))
        previous_price = price
    return tuple(result)


def _slow_context_bars(store, symbol: str, decision_time: datetime):
    """One minute-bar fetch per frame, shared by every slow-context consumer.

    ``recent_minute_bars`` is a query against a multi-GB SQLite store. The frame
    runs per symbol on a ~5s sweep, so each extra call is paid 25x per sweep under
    WAL contention with the ingest writers. Adding a second independent fetch for
    the indicator families stalled the trading loop hard enough to wedge the
    server twice; the arithmetic on the bars was never the expensive part.

    Returns the completed-bar set AND the rows it was built from. ``completed_bars``
    projects each row down to OHLCV, dropping the ``spread_bps`` /
    ``orderbook_imbalance`` / ``liquidity_score`` columns that
    ``realtime_minute_bars`` persists -- the very columns the historical labelling
    path reads. Returning the rows lets the GNN context read the same persisted
    numbers instead of re-deriving a proxy from the bar range.
    """
    from app.technical.causal_bars import completed_bars

    try:
        # 240 minutes covered the trend window but stopped short of the session
        # open after roughly 13:00 KST, which would make the session block
        # unavailable for the back half of every session — precisely the hours the
        # opening-range and first-half-hour theses are evaluated against.
        reader = getattr(store, "reconciled_minute_bars", store.recent_minute_bars)
        rows = reader(
            symbol,
            decision_time - timedelta(minutes=_ctx_session_bars()),
            limit=_ctx_session_bars(),
            **({"market": "US" if not (symbol.isdigit() and len(symbol) == 6) else "KR"}
               if hasattr(store, "reconciled_minute_bars") else {}),
        )
    except Exception:  # noqa: BLE001 - slow context fails closed, never raises.
        rows = ()
    bar_set = completed_bars(
        rows,
        symbol=symbol,
        as_of=decision_time,
        timeframe_minutes=1,
        warmup_required=64,
    )
    return bar_set, tuple(rows or ())


def _ctx_session_bars() -> int:
    from app.features import strategy_graph_context as ctx

    return int(ctx.SESSION_CONTEXT_BARS)


def _strategy_graph_context_columns(
    bar_set,
    rows: tuple[Any, ...],
    *,
    symbol: str,
    rvgi_box: Mapping[str, float],
) -> dict[str, float]:
    """The GNN context contract, computed from completed one-minute bars.

    Deliberately independent of the tick window that feeds the live short-horizon
    model. The historical labelling path has minute bars and nothing else, so a
    tick-derived quantity here could never be trained -- which is exactly how v4
    ended up serving ``aggressor_imbalance_5s`` into a slot fitted on a clipped
    bar return. Every field below is computed from the same source table, over
    the same window, by the same estimator as
    :func:`app.evaluation.stored_counterfactual._label_features`.

    Returns an empty mapping when there is not enough completed history; the
    caller records that as an unavailable context rather than serving zeros.
    """
    # ``_as_utc`` is reused rather than reimplemented: the bar timestamps this
    # keys against were normalised by that exact function inside ``completed_bars``,
    # and a second near-copy is how a lookup starts missing on tz-naive rows.
    from app.technical.causal_bars import _as_utc
    from app.features import strategy_graph_context as ctx

    bars = tuple(getattr(bar_set, "bars", ()) or ())
    if len(bars) < 2:
        return {}
    current = bars[-1]
    history = bars[max(0, len(bars) - 1 - ctx.CONTEXT_HISTORY_BARS) : -1]
    if not history:
        return {}
    price = float(current.close)
    if not math.isfinite(price) or price <= 0:
        return {}

    closes = [float(bar.close) for bar in history]
    volumes = [float(bar.volume) for bar in history]
    returns = [
        closes[position] / closes[position - 1] - 1.0
        for position in range(1, len(closes))
        if closes[position - 1] > 0
    ]
    bar_return = (
        price / closes[-1] - 1.0 if closes and closes[-1] > 0 else 0.0
    )
    vwap = ctx.volume_weighted_close(closes, volumes)

    # Persisted microstructure of the bar the decision is anchored on. Keyed by
    # minute_start, matching how the labelling path keys its own lookup, so a
    # gap in the store yields "no context" rather than another bar's numbers.
    micro = {
        _as_utc(getattr(row, "minute_start", None)): row
        for row in rows
        if getattr(row, "minute_start", None) is not None
    }.get(current.as_of)
    if micro is None:
        return {}

    box_available = float(rvgi_box.get("box_available", 0.0) or 0.0)
    box_high = float(rvgi_box.get("box_high", 0.0) or 0.0)
    # The trend block reads the same window as the statistics above: the history
    # plus the anchor bar. The labelling path passes ``bars[history_start:index+1]``,
    # which is that same span, and both hand it to the one shared estimator.
    trend_window = bars[max(0, len(bars) - 1 - ctx.CONTEXT_HISTORY_BARS) :]
    return {
        **ctx.microstructure_columns(
            _optional_column(micro, "spread_bps"),
            _optional_column(micro, "orderbook_imbalance"),
            _optional_column(micro, "liquidity_score"),
        ),
        **ctx.trend_structure_columns(trend_window),
        **ctx.session_structure_columns(*ctx.session_slice(bars)),
        "return_1m_scaled": ctx.scaled_return(bar_return),
        "realized_volatility_30m": ctx.realized_volatility(returns),
        "distance_from_vwap": ctx.safe_ratio(price, vwap) - 1.0 if vwap else 0.0,
        "volume_spike_ratio": ctx.volume_spike_ratio(float(current.volume), volumes),
        "is_krx": ctx.is_krx_symbol(symbol),
        "rvgi_available": float(rvgi_box.get("rvgi_available", 0.0) or 0.0),
        "rvgi": float(rvgi_box.get("rvgi", 0.0) or 0.0),
        "rvgi_signal": float(rvgi_box.get("rvgi_signal", 0.0) or 0.0),
        "rvgi_diff": float(rvgi_box.get("rvgi_diff", 0.0) or 0.0),
        "rvgi_slope": float(rvgi_box.get("rvgi_slope", 0.0) or 0.0),
        "rvgi_bullish_cross": float(rvgi_box.get("rvgi_bullish_cross", 0.0) or 0.0),
        "box_available": box_available,
        "box_high_ratio": ctx.safe_ratio(box_high, price),
        "box_low_ratio": ctx.safe_ratio(rvgi_box.get("box_low"), price),
        "box_mid_ratio": ctx.safe_ratio(rvgi_box.get("box_mid"), price),
        "box_width_pct": float(rvgi_box.get("box_width_pct", 0.0) or 0.0),
        "box_position": float(rvgi_box.get("box_position", 0.0) or 0.0),
        # (price / box_high - 1) * 100. The live column is in true bps, the
        # contract is in percent.
        "breakout_distance_pct": (
            float(rvgi_box.get("breakout_distance_bps", 0.0) or 0.0) / 100.0
        ),
        "box_previous_close_ratio": ctx.safe_ratio(
            rvgi_box.get("box_previous_close"), price
        ),
        "box_context_available": (
            1.0
            if float(rvgi_box.get("box_context_timestamp_epoch", 0.0) or 0.0) > 0
            else 0.0
        ),
    }


def _optional_column(row: Any, name: str) -> float | None:
    """A persisted microstructure column, or ``None`` when it is absent."""
    value = getattr(row, name, None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _indicator_family_columns(
    bar_set,
    *,
    spread_bps: float | None = None,
    liquidity_score: float | None = None,
    orderbook_imbalance: float | None = None,
    aggressor_imbalance: float | None = None,
) -> dict[str, float]:
    """Compact indicator-family columns from COMPLETED bars.

    Uses the shared causal bar builder so a live value and a replay value for the
    same as-of are produced by identical code. On any failure every column falls
    back to its neutral and the ``*_available`` flags read 0 -- the model is told
    the reading is missing rather than being handed a fabricated neutral.
    """
    from app.technical.indicator_families import (
        build_families,
        compact_model_features,
    )

    try:
        bars = tuple(getattr(bar_set, "bars", ()) or ())
        bundle = build_families(
            bars,
            aggressor_imbalance=aggressor_imbalance,
            orderbook_imbalance=orderbook_imbalance,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
        )
        return compact_model_features(bars, bundle)
    except Exception:  # noqa: BLE001 - families fail closed to an all-zero mask.
        return compact_model_features((), build_families(()))


def _slow_technical_columns(
    bar_set,
    *,
    symbol: str,
    orderbook: object,
    price: float,
) -> dict[str, float]:
    """Causal minute-horizon technical inputs for strategy election.

    These stay outside the model schema: changing them must not invalidate the
    seconds-model artifact, and the raw fields are intended for rule/strategy
    consumers rather than pooled cross-symbol learning.
    """
    from app.technical.feature_builder import build_technical_feature_set

    bars = tuple(getattr(bar_set, "bars", ()) or ())
    if not bars:
        return {}
    features = build_technical_feature_set(
        bars,
        symbol=symbol,
        orderbook=orderbook,
        price=price,
        # Absolute volume normalisation is instrument-identifying and was removed
        # from the model schema.  Spread/depth remain the live execution-quality
        # evidence; do not turn a low raw share count into a universal liquidity
        # veto here.
        liquidity_score=None,
    )
    names = (
        "ema_fast", "ema_slow", "macd", "macd_signal", "macd_histogram",
        "short_return", "momentum_persistence", "adx", "plus_di", "minus_di",
        "dmi_spread", "supertrend", "supertrend_direction",
        "supertrend_distance_bps", "rsi", "bb_percent_b", "bb_bandwidth",
        "keltner_mid", "keltner_upper", "keltner_lower", "keltner_position",
        "keltner_bandwidth", "prior_keltner_squeeze_ratio", "choppiness", "vwap", "vwap_distance_bps", "vwap_slope",
        "relative_volume", "volume_spike_ratio", "donchian_high",
        "donchian_low",
        "breakout_strength", "donchian_low_distance",
        "false_breakout_risk", "atr_pct", "realized_volatility",
        "volatility_expansion",
    )
    result: dict[str, float] = {}
    for name in names:
        value = getattr(features, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[name] = float(value)
    result["bar_count"] = float(len(bars))
    # Conservative, directly observed exit-capacity evidence. This is only the
    # completed bars in the current context window, so it is a lower bound on a
    # full day's traded value rather than a forecast or a fabricated ADV.
    result["observed_trading_value"] = sum(
        max(0.0, float(bar.close)) * max(0.0, float(bar.volume))
        for bar in bars
        if math.isfinite(float(bar.close)) and math.isfinite(float(bar.volume))
    )
    return result


def _rvgi_box_columns(
    bar_set,
    symbol: str,
    decision_time: datetime,
    current_price: float,
) -> dict[str, float]:
    """Build slow descriptors from completed one-minute bars, never ticks."""
    from app.technical import indicators as ti

    unavailable = {
        "rvgi_available": 0.0,
        "rvgi": 0.0,
        "rvgi_signal": 0.0,
        "rvgi_diff": 0.0,
        "rvgi_slope": 0.0,
        "rvgi_bullish_cross": 0.0,
        "box_available": 0.0,
        "box_high": 0.0,
        "box_low": 0.0,
        "box_mid": 0.0,
        "box_width_pct": 0.0,
        "box_position": 0.0,
        "breakout_distance_bps": 0.0,
        "box_previous_close": 0.0,
        "box_context_timestamp_epoch": 0.0,
    }
    # Bars come from the single per-frame fetch (see _slow_context_bars); this
    # function no longer queries the store itself.
    bars = tuple(getattr(bar_set, "bars", ()) or ())
    if not bars:
        return unavailable
    rvgi_result = ti.rvgi(bars, 10)
    box = ti.causal_box_geometry(bars, 20)
    result = dict(unavailable)
    if rvgi_result.ok and rvgi_result.main is not None and rvgi_result.signal is not None:
        result.update(
            rvgi_available=1.0,
            rvgi=float(rvgi_result.main),
            rvgi_signal=float(rvgi_result.signal),
            rvgi_diff=float(rvgi_result.main - rvgi_result.signal),
            rvgi_slope=float(rvgi_result.slope or 0.0),
            rvgi_bullish_cross=1.0 if rvgi_result.bullish_cross else 0.0,
        )
    if box.ok and box.high is not None and box.low is not None and box.mid is not None:
        result.update(
            box_available=1.0,
            box_high=float(box.high),
            box_low=float(box.low),
            box_mid=float(box.mid),
            box_width_pct=float(box.width_pct or 0.0),
            box_position=float(box.position or 0.0),
            breakout_distance_bps=(
                (current_price / box.high - 1.0) * 10_000.0
                if current_price > 0 and box.high > 0
                else 0.0
            ),
            box_previous_close=float(bars[-2].close) if len(bars) >= 2 else 0.0,
            box_context_timestamp_epoch=(
                float(box.source_timestamp.timestamp())
                if hasattr(box.source_timestamp, "timestamp")
                else 0.0
            ),
        )
    return result


_KST = timezone(timedelta(hours=9))


#: ``(symbol, trading_day) -> completed bars from earlier trading days``. Bounded
#: because a live universe rotates and a stale entry is only wasted memory.
_PREVIOUS_SESSION_BARS_CACHE: dict[tuple[str, Any], tuple[Any, ...]] = {}
_PREVIOUS_SESSION_BARS_CACHE_MAX = 256


def _cached_previous_session_bars(
    store: RealtimeMarketDataStore,
    symbol: str,
    decision_time: datetime,
    current_day: Any,
    zone: Any,
) -> tuple[Any, ...] | None:
    """Completed bars from trading days BEFORE ``current_day``.

    Immutable for the whole trading day, so it is fetched once per
    ``(symbol, current_day)``. Returns ``None`` only when the query itself failed,
    which the caller treats as "context unavailable" and fails closed on; an empty
    tuple means the query succeeded and there is genuinely no prior history.
    """
    key = (symbol, current_day)
    cached = _PREVIOUS_SESSION_BARS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        reader = getattr(store, "reconciled_minute_bars", store.recent_minute_bars)
        raw = reader(
            symbol,
            decision_time - timedelta(days=14),
            limit=1600,
            **({"market": "US" if not (symbol.isdigit() and len(symbol) == 6) else "KR"}
               if hasattr(store, "reconciled_minute_bars") else {}),
        )
    except Exception:  # noqa: BLE001 - unavailable context must fail closed.
        return None
    previous = tuple(
        bar
        for bar in raw
        if bar.minute_start + timedelta(minutes=1) <= decision_time
        and bar.minute_start.astimezone(zone).date() < current_day
    )
    if len(_PREVIOUS_SESSION_BARS_CACHE) >= _PREVIOUS_SESSION_BARS_CACHE_MAX:
        # Drop entries for earlier trading days first; they can never be read
        # again. Only fall back to arbitrary eviction if today filled the cache.
        for stale in [
            existing
            for existing in _PREVIOUS_SESSION_BARS_CACHE
            if existing[1] != current_day
        ] or list(_PREVIOUS_SESSION_BARS_CACHE)[:1]:
            _PREVIOUS_SESSION_BARS_CACHE.pop(stale, None)
    _PREVIOUS_SESSION_BARS_CACHE[key] = previous
    return previous


def _session_structure_diagnostics(
    store: RealtimeMarketDataStore,
    symbol: str,
    decision_time: datetime,
    *,
    minimum_samples: int = 3,
) -> dict[str, Any]:
    """Measured opening context from completed bars, absent when unanswerable.

    The session clock comes from ``session_structure.regular_session`` so a US
    symbol is anchored to 09:30 America/New_York rather than to 09:00 Asia/Seoul.
    """
    session = session_structure.regular_session(symbol)
    zone = session.zone
    local_now = decision_time.astimezone(zone)
    # Before today's regular open there is no current-session opening range.
    # Returning yesterday's completed range would leak a stale gap thesis into
    # premarket trading because ``trading_day`` correctly points to the previous
    # completed session at that time.
    #
    # Checked BEFORE any query: this used to fetch fourteen days of minute bars
    # first and only then discover it had nothing to say, so every pre-open and
    # weekend cycle paid the full cost for an empty answer.
    if local_now.weekday() >= 5 or local_now.time() < session.open_time:
        return {}
    session_open = session.session_open(decision_time)
    current_day = session.trading_day(decision_time)
    # Split in two, because the two halves have completely different lifetimes.
    #
    # The prior-day bars feed ``historical_volatility`` and ``previous_close``, and
    # they cannot change until tomorrow -- yet this ran a 14-day / 1600-row query
    # on EVERY feature-frame build, per symbol, per engine cycle. Measured
    # 2026-08-21 at 50-95ms per call against an idle reader, and far worse under a
    # growing WAL: it is the query the trading engine and the KIS websocket loop
    # were both found parked in. Cached per trading day it runs once per symbol
    # per day; only today's bars are re-read.
    previous = _cached_previous_session_bars(
        store, symbol, decision_time, current_day, zone
    )
    if previous is None:
        return {}
    try:
        reader = getattr(store, "reconciled_minute_bars", store.recent_minute_bars)
        current_raw = reader(
            symbol,
            session_open - timedelta(hours=1),
            limit=1600,
            **({"market": "US" if not (symbol.isdigit() and len(symbol) == 6) else "KR"}
               if hasattr(store, "reconciled_minute_bars") else {}),
        )
    except Exception:  # noqa: BLE001 - unavailable context must fail closed.
        return {}
    current_bars = tuple(
        bar
        for bar in current_raw
        if bar.minute_start + timedelta(minutes=1) <= decision_time
        and bar.minute_start.astimezone(zone).date() == current_day
    )
    previous_close = float(previous[-1].close) if previous else 0.0
    session_open_price = (
        float(getattr(current_bars[0], "open", 0.0) or 0.0)
        if current_bars
        else 0.0
    )
    # Gap context exists as soon as the first regular-session bar completes. It
    # must not wait for the full 30-minute opening range: doing so moved a 5-minute
    # gap thesis outside its useful entry horizon.
    result: dict[str, Any] = {}
    if session_open_price > 0.0:
        result["session_open_price"] = session_open_price
    if previous_close > 0.0:
        result["previous_close_price"] = previous_close
    if session_open_price > 0.0 and previous_close > 0.0:
        gap_rate = session_open_price / previous_close - 1.0
        result["gap_rate"] = gap_rate
        result["gap_submode"] = "continuation" if gap_rate > 0.0 else "fade"

    observed = session_structure.opening_range(
        current_bars,
        session_open=session_open,
        minutes=30,
        now=decision_time,
    )
    if observed is None:
        return result
    opening_return = session_structure.first_half_hour_return_bps(
        current_bars,
        previous_close=previous_close,
        session_open=session_open,
        minutes=30,
        now=decision_time,
    )

    by_day: dict[Any, list[Any]] = {}
    for bar in previous:
        day = bar.minute_start.astimezone(zone).date()
        by_day.setdefault(day, []).append(bar)
    historical_volatility: list[float] = []
    for day, day_bars in sorted(by_day.items()):
        day_open = datetime(
            day.year,
            day.month,
            day.day,
            session.open_time.hour,
            session.open_time.minute,
            tzinfo=zone,
        )
        prior_range = session_structure.opening_range(
            day_bars,
            session_open=day_open,
            minutes=30,
            now=day_open + timedelta(days=1),
        )
        if prior_range is not None:
            historical_volatility.append(prior_range.volatility)
    volatility_percentile = session_structure.first_half_hour_volatility_percentile(
        historical_volatility,
        observed.volatility,
        minimum_samples=minimum_samples,
    )
    result.update(
        opening_range_high=observed.high,
        opening_range_low=observed.low,
        opening_range_minutes=observed.minutes,
    )
    if opening_return is not None:
        result["first_half_hour_return_bps"] = opening_return
    if volatility_percentile is not None:
        result["first_half_hour_volatility_percentile"] = volatility_percentile
    return result


def _events_in_window(events: tuple, decision_time: datetime, seconds: int) -> tuple:
    """Return events observable inside the point-in-time ingestion window.

    Exchange timestamps describe when a venue says an event happened; they do
    not describe when this process could act on it.  KIS can deliver a burst
    several seconds later, so cutting by exchange time incorrectly produced an
    empty window despite fresh records. ``received_at`` is the causal clock and
    is also the clock used by market-data health.
    """

    cutoff = decision_time - timedelta(seconds=seconds)
    return tuple(sorted(
        (
            event
            for event in events
            if cutoff <= event.received_at <= decision_time
        ),
        key=lambda event: (event.received_at, event.exchange_timestamp),
    ))


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
