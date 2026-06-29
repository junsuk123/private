from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.data.market_data_health import evaluate_market_data_health
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.feature_provenance import FeatureProvenance
from app.features.feature_schema import FeatureSchema, LIVE_SHORT_HORIZON_SCHEMA


class FeatureFrameError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveFeatureFrame:
    symbol: str
    decision_time: datetime
    schema: FeatureSchema
    values: tuple[float, ...]
    provenance: FeatureProvenance

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
        max_quote_age_ms: int = 3000,
        max_orderbook_age_ms: int = 3000,
        journal_path: str | Path = "logs/live-feature-frames.jsonl",
    ) -> None:
        self.store = store
        self.schema = schema
        self.max_quote_age_ms = max_quote_age_ms
        self.max_orderbook_age_ms = max_orderbook_age_ms
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, symbol: str, *, decision_time: datetime | None = None) -> LiveFeatureFrame:
        decision_time = decision_time or datetime.now(timezone.utc)
        health = evaluate_market_data_health(
            self.store,
            symbol,
            max_quote_age_ms=self.max_quote_age_ms,
            max_orderbook_age_ms=self.max_orderbook_age_ms,
            now=decision_time,
        )
        if not health.ok_for_live_buy:
            raise FeatureFrameError("MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE:" + ",".join(health.reason_codes))
        since = decision_time - timedelta(minutes=3)
        ticks = tuple(tick for tick in self.store.recent_ticks(symbol, since) if tick.exchange_timestamp <= decision_time)
        orderbook = self.store.latest_orderbook(symbol)
        if not ticks or orderbook is None:
            raise FeatureFrameError("MISSING_SOURCE_RECORDS")
        prices = [tick.price for tick in ticks]
        volumes = [max(0, tick.volume) for tick in ticks]
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
        feature_dict = {
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
        }
        values = tuple(float(feature_dict[name]) for name in self.schema.feature_names)
        provenance = FeatureProvenance(
            symbol=symbol,
            decision_time=decision_time,
            tick_record_ids=tuple(tick.record_id for tick in ticks),
            orderbook_record_id=orderbook.record_id,
            source="kis_realtime_websocket",
            max_input_age_ms=max(
                (decision_time - ticks[-1].received_at).total_seconds() * 1000,
                (decision_time - orderbook.received_at).total_seconds() * 1000,
            ),
        )
        frame = LiveFeatureFrame(symbol, decision_time, self.schema, values, provenance)
        frame.validate()
        self._journal(frame)
        return frame

    def _journal(self, frame: LiveFeatureFrame) -> None:
        payload = {
            "symbol": frame.symbol,
            "decision_time": frame.decision_time.isoformat(),
            "feature_schema_hash": frame.feature_schema_hash,
            "source_record_ids": frame.provenance.source_record_ids,
            "values": frame.as_feature_dict(),
        }
        with self.journal_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _safe_return(current: float, previous: float) -> float:
    return 0.0 if previous <= 0 else current / previous - 1.0


def _window_return(ticks: tuple, decision_time: datetime, *, seconds: int) -> float:
    cutoff = decision_time - timedelta(seconds=seconds)
    visible = tuple(tick for tick in ticks if tick.exchange_timestamp >= cutoff)
    if len(visible) < 2:
        return 0.0
    return _safe_return(visible[-1].price, visible[0].price)


def _returns(prices: list[float]) -> list[float]:
    return [_safe_return(prices[index], prices[index - 1]) for index in range(1, len(prices))]


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
