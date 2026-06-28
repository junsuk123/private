from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from statistics import mean, pstdev
from typing import Mapping, Sequence

from app.features.schemas import OHLCVBar


@dataclass(frozen=True)
class ShortHorizonFeatureConfig:
    return_windows_minutes: tuple[int, ...] = (1, 3, 5, 15, 30)
    realized_volatility_windows_minutes: tuple[int, ...] = (5, 30)
    volume_zscore_window: int = 20
    session_open: time = time(9, 0)
    session_close: time = time(15, 30)
    required_return_windows: tuple[str, ...] = ("ret_1m", "ret_3m", "ret_5m")


@dataclass(frozen=True)
class ShortHorizonFeatures:
    ticker: str
    timestamp: datetime
    returns_by_window: dict[str, float | None]
    realized_volatility: dict[str, float | None]
    volume_zscore: float | None
    spread_rate: float | None
    orderbook_depth_score: float | None
    liquidity_score: float | None
    market_alignment_score: float | None
    time_of_day_weight: float | None
    is_valid: bool
    missing_fields: tuple[str, ...] = field(default_factory=tuple)

    def as_feature_dict(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, value in self.returns_by_window.items():
            if value is not None:
                values[name] = value
        for name, value in self.realized_volatility.items():
            if value is not None:
                values[name] = value
        for name in (
            "volume_zscore",
            "spread_rate",
            "orderbook_depth_score",
            "liquidity_score",
            "market_alignment_score",
            "time_of_day_weight",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        return values


class ShortHorizonFeatureBuilder:
    def __init__(self, config: ShortHorizonFeatureConfig | None = None) -> None:
        self.config = config or ShortHorizonFeatureConfig()

    def build(
        self,
        minute_bars: Sequence[OHLCVBar],
        *,
        as_of: datetime | None = None,
        daily_bars: Sequence[OHLCVBar] = (),
        market_index_bars: Sequence[OHLCVBar] = (),
        orderbook: Mapping[str, float | int] | None = None,
    ) -> ShortHorizonFeatures:
        visible = _visible_bars(minute_bars, as_of)
        missing: list[str] = []
        if not visible:
            timestamp = as_of or datetime.now()
            return ShortHorizonFeatures(
                ticker="",
                timestamp=timestamp,
                returns_by_window=_empty_returns(self.config),
                realized_volatility=_empty_volatility(self.config),
                volume_zscore=None,
                spread_rate=None,
                orderbook_depth_score=None,
                liquidity_score=None,
                market_alignment_score=None,
                time_of_day_weight=_time_of_day_weight(timestamp, self.config),
                is_valid=False,
                missing_fields=("minute_bars",),
            )

        ticker = visible[-1].ticker
        timestamp = visible[-1].as_of
        returns = self._returns_by_window(visible, daily_bars, timestamp, missing)
        volatility = self._realized_volatility(visible, missing)
        volume_z = _volume_zscore(visible, self.config.volume_zscore_window)
        if volume_z is None:
            missing.append("volume_zscore")

        spread_rate, depth_score = _orderbook_features(orderbook)
        if orderbook:
            if spread_rate is None:
                missing.append("spread_rate")
            if depth_score is None:
                missing.append("orderbook_depth_score")
        else:
            missing.extend(["spread_rate", "orderbook_depth_score"])

        liquidity = _liquidity_score(volume_z, spread_rate, depth_score)
        if liquidity is None:
            missing.append("liquidity_score")

        market_alignment = _market_alignment_score(visible, market_index_bars, timestamp)
        if market_alignment is None:
            missing.append("market_alignment_score")

        time_weight = _time_of_day_weight(timestamp, self.config)
        if time_weight is None:
            missing.append("time_of_day_weight")

        missing_tuple = tuple(dict.fromkeys(missing))
        is_valid = all(returns.get(name) is not None for name in self.config.required_return_windows)
        is_valid = is_valid and not any(volatility.get(f"realized_volatility_{window}m") is None for window in (5,))
        return ShortHorizonFeatures(
            ticker=ticker,
            timestamp=timestamp,
            returns_by_window=returns,
            realized_volatility=volatility,
            volume_zscore=volume_z,
            spread_rate=spread_rate,
            orderbook_depth_score=depth_score,
            liquidity_score=liquidity,
            market_alignment_score=market_alignment,
            time_of_day_weight=time_weight,
            is_valid=is_valid,
            missing_fields=missing_tuple,
        )

    def _returns_by_window(
        self,
        bars: Sequence[OHLCVBar],
        daily_bars: Sequence[OHLCVBar],
        timestamp: datetime,
        missing: list[str],
    ) -> dict[str, float | None]:
        returns: dict[str, float | None] = {}
        for window in self.config.return_windows_minutes:
            name = f"ret_{window}m"
            returns[name] = _return_from_minutes(bars, window)
            if returns[name] is None:
                missing.append(name)

        returns["ret_1d"] = _daily_return(bars, daily_bars, timestamp)
        if returns["ret_1d"] is None:
            missing.append("ret_1d")

        for window in (10, 30):
            name = f"ret_open_{window}m"
            returns[name] = _return_from_session_open(bars, timestamp, window, self.config)
            if returns[name] is None:
                missing.append(name)

        returns["ret_preclose_30m"] = _return_from_preclose_window(bars, timestamp, 30, self.config)
        if returns["ret_preclose_30m"] is None:
            missing.append("ret_preclose_30m")
        return returns

    def _realized_volatility(self, bars: Sequence[OHLCVBar], missing: list[str]) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for window in self.config.realized_volatility_windows_minutes:
            name = f"realized_volatility_{window}m"
            values[name] = _realized_volatility(bars, window)
            if values[name] is None:
                missing.append(name)
        return values


class TickerRollingFeatureState:
    """Ticker-level ordered ring buffer for short-horizon features.

    The state stores only observed bars and never reads bars after `as_of`.
    It is intended for realtime/paper/live hot paths that update one bar at a
    time instead of repeatedly sorting a growing history.
    """

    def __init__(
        self,
        ticker: str,
        config: ShortHorizonFeatureConfig | None = None,
        *,
        max_bars: int | None = None,
    ) -> None:
        self.ticker = ticker
        self.config = config or ShortHorizonFeatureConfig()
        required = max(
            (*self.config.return_windows_minutes, *self.config.realized_volatility_windows_minutes, self.config.volume_zscore_window)
        ) + 2
        self.max_bars = max_bars or max(256, required)
        self._bars: deque[OHLCVBar] = deque(maxlen=self.max_bars)

    def update(
        self,
        bar: OHLCVBar,
        *,
        as_of: datetime | None = None,
        daily_bars: Sequence[OHLCVBar] = (),
        market_index_bars: Sequence[OHLCVBar] = (),
        market_index_state: "TickerRollingFeatureState | None" = None,
        orderbook: Mapping[str, float | int] | None = None,
    ) -> ShortHorizonFeatures:
        self.add_bar(bar)
        return self.build(
            as_of=as_of or bar.as_of,
            daily_bars=daily_bars,
            market_index_bars=market_index_bars,
            market_index_state=market_index_state,
            orderbook=orderbook,
        )

    def add_bar(self, bar: OHLCVBar) -> None:
        if bar.ticker != self.ticker:
            raise ValueError(f"bar ticker {bar.ticker!r} does not match rolling state {self.ticker!r}")
        if self._bars and bar.as_of < self._bars[-1].as_of:
            ordered = [item for item in self._bars if item.as_of != bar.as_of]
            ordered.append(bar)
            ordered.sort(key=lambda item: item.as_of)
            self._bars.clear()
            self._bars.extend(ordered[-self.max_bars :])
            return
        if self._bars and bar.as_of == self._bars[-1].as_of:
            self._bars[-1] = bar
            return
        self._bars.append(bar)

    def build(
        self,
        *,
        as_of: datetime | None = None,
        daily_bars: Sequence[OHLCVBar] = (),
        market_index_bars: Sequence[OHLCVBar] = (),
        market_index_state: "TickerRollingFeatureState | None" = None,
        orderbook: Mapping[str, float | int] | None = None,
    ) -> ShortHorizonFeatures:
        visible = self.visible_bars(as_of)
        if market_index_state is not None:
            market_index_bars = market_index_state.visible_bars(as_of or (visible[-1].as_of if visible else None))
        return _build_short_horizon_features_from_ordered(
            visible,
            self.config,
            as_of=as_of,
            daily_bars=daily_bars,
            market_index_bars=market_index_bars,
            orderbook=orderbook,
        )

    def visible_bars(self, as_of: datetime | None = None) -> tuple[OHLCVBar, ...]:
        if as_of is None:
            return tuple(self._bars)
        return tuple(bar for bar in self._bars if bar.as_of <= as_of)

    @property
    def bar_count(self) -> int:
        return len(self._bars)


def _build_short_horizon_features_from_ordered(
    visible: Sequence[OHLCVBar],
    config: ShortHorizonFeatureConfig,
    *,
    as_of: datetime | None = None,
    daily_bars: Sequence[OHLCVBar] = (),
    market_index_bars: Sequence[OHLCVBar] = (),
    orderbook: Mapping[str, float | int] | None = None,
) -> ShortHorizonFeatures:
    missing: list[str] = []
    if not visible:
        timestamp = as_of or datetime.now()
        return ShortHorizonFeatures(
            ticker="",
            timestamp=timestamp,
            returns_by_window=_empty_returns(config),
            realized_volatility=_empty_volatility(config),
            volume_zscore=None,
            spread_rate=None,
            orderbook_depth_score=None,
            liquidity_score=None,
            market_alignment_score=None,
            time_of_day_weight=_time_of_day_weight(timestamp, config),
            is_valid=False,
            missing_fields=("minute_bars",),
        )

    ticker = visible[-1].ticker
    timestamp = visible[-1].as_of
    returns: dict[str, float | None] = {}
    for window in config.return_windows_minutes:
        name = f"ret_{window}m"
        returns[name] = _return_from_minutes(visible, window)
        if returns[name] is None:
            missing.append(name)
    returns["ret_1d"] = _daily_return(visible, daily_bars, timestamp)
    if returns["ret_1d"] is None:
        missing.append("ret_1d")
    for window in (10, 30):
        name = f"ret_open_{window}m"
        returns[name] = _return_from_session_open(visible, timestamp, window, config)
        if returns[name] is None:
            missing.append(name)
    returns["ret_preclose_30m"] = _return_from_preclose_window(visible, timestamp, 30, config)
    if returns["ret_preclose_30m"] is None:
        missing.append("ret_preclose_30m")

    volatility: dict[str, float | None] = {}
    for window in config.realized_volatility_windows_minutes:
        name = f"realized_volatility_{window}m"
        volatility[name] = _realized_volatility(visible, window)
        if volatility[name] is None:
            missing.append(name)

    volume_z = _volume_zscore(visible, config.volume_zscore_window)
    if volume_z is None:
        missing.append("volume_zscore")
    spread_rate, depth_score = _orderbook_features(orderbook)
    if orderbook:
        if spread_rate is None:
            missing.append("spread_rate")
        if depth_score is None:
            missing.append("orderbook_depth_score")
    else:
        missing.extend(["spread_rate", "orderbook_depth_score"])
    liquidity = _liquidity_score(volume_z, spread_rate, depth_score)
    if liquidity is None:
        missing.append("liquidity_score")
    market_alignment = _market_alignment_score(visible, market_index_bars, timestamp)
    if market_alignment is None:
        missing.append("market_alignment_score")
    time_weight = _time_of_day_weight(timestamp, config)
    if time_weight is None:
        missing.append("time_of_day_weight")
    missing_tuple = tuple(dict.fromkeys(missing))
    is_valid = all(returns.get(name) is not None for name in config.required_return_windows)
    is_valid = is_valid and not any(volatility.get(f"realized_volatility_{window}m") is None for window in (5,))
    return ShortHorizonFeatures(
        ticker=ticker,
        timestamp=timestamp,
        returns_by_window=returns,
        realized_volatility=volatility,
        volume_zscore=volume_z,
        spread_rate=spread_rate,
        orderbook_depth_score=depth_score,
        liquidity_score=liquidity,
        market_alignment_score=market_alignment,
        time_of_day_weight=time_weight,
        is_valid=is_valid,
        missing_fields=missing_tuple,
    )


def _visible_bars(bars: Sequence[OHLCVBar], as_of: datetime | None) -> tuple[OHLCVBar, ...]:
    ordered = tuple(sorted(bars, key=lambda bar: bar.as_of))
    if as_of is not None:
        ordered = tuple(bar for bar in ordered if bar.as_of <= as_of)
    return ordered


def _empty_returns(config: ShortHorizonFeatureConfig) -> dict[str, float | None]:
    names = [f"ret_{window}m" for window in config.return_windows_minutes]
    names.extend(["ret_1d", "ret_open_10m", "ret_open_30m", "ret_preclose_30m"])
    return {name: None for name in names}


def _empty_volatility(config: ShortHorizonFeatureConfig) -> dict[str, float | None]:
    return {f"realized_volatility_{window}m": None for window in config.realized_volatility_windows_minutes}


def _return_from_minutes(bars: Sequence[OHLCVBar], minutes: int) -> float | None:
    if len(bars) <= minutes:
        return None
    previous = bars[-minutes - 1].close
    current = bars[-1].close
    return _safe_return(current, previous)


def _daily_return(bars: Sequence[OHLCVBar], daily_bars: Sequence[OHLCVBar], timestamp: datetime) -> float | None:
    visible_daily = tuple(bar for bar in sorted(daily_bars, key=lambda bar: bar.as_of) if bar.as_of <= timestamp)
    if len(visible_daily) >= 2:
        return _safe_return(visible_daily[-1].close, visible_daily[-2].close)
    if visible_daily:
        return _safe_return(bars[-1].close, visible_daily[-1].close)
    return None


def _return_from_session_open(
    bars: Sequence[OHLCVBar],
    timestamp: datetime,
    minutes: int,
    config: ShortHorizonFeatureConfig,
) -> float | None:
    open_dt = datetime.combine(timestamp.date(), config.session_open, tzinfo=timestamp.tzinfo)
    target_dt = open_dt + timedelta(minutes=minutes)
    if timestamp < target_dt:
        return None
    open_bar = _first_bar_at_or_after(bars, open_dt)
    target_bar = _last_bar_at_or_before(bars, target_dt)
    if open_bar is None or target_bar is None or target_bar.as_of < target_dt:
        return None
    return _safe_return(target_bar.close, open_bar.close)


def _return_from_preclose_window(
    bars: Sequence[OHLCVBar],
    timestamp: datetime,
    minutes: int,
    config: ShortHorizonFeatureConfig,
) -> float | None:
    close_dt = datetime.combine(timestamp.date(), config.session_close, tzinfo=timestamp.tzinfo)
    start_dt = close_dt - timedelta(minutes=minutes)
    if timestamp < start_dt:
        return None
    start_bar = _last_bar_at_or_before(bars, start_dt)
    current_bar = bars[-1]
    if start_bar is None or current_bar.as_of < start_dt:
        return None
    return _safe_return(current_bar.close, start_bar.close)


def _realized_volatility(bars: Sequence[OHLCVBar], minutes: int) -> float | None:
    if len(bars) <= minutes:
        return None
    closes = [bar.close for bar in bars[-minutes - 1 :]]
    returns = [_safe_return(closes[i], closes[i - 1]) for i in range(1, len(closes))]
    numeric = [value for value in returns if value is not None and math.isfinite(value)]
    if len(numeric) < 2:
        return None
    return pstdev(numeric)


def _volume_zscore(bars: Sequence[OHLCVBar], window: int) -> float | None:
    if len(bars) <= window:
        return None
    baseline = [bar.volume for bar in bars[-window - 1 : -1]]
    deviation = pstdev(baseline)
    if deviation == 0:
        return None
    return (bars[-1].volume - mean(baseline)) / deviation


def _orderbook_features(orderbook: Mapping[str, float | int] | None) -> tuple[float | None, float | None]:
    if not orderbook:
        return None, None
    bid = _positive_float(orderbook.get("best_bid"))
    ask = _positive_float(orderbook.get("best_ask"))
    bid_depth = _positive_float(orderbook.get("bid_depth") or orderbook.get("total_bid_size"))
    ask_depth = _positive_float(orderbook.get("ask_depth") or orderbook.get("total_ask_size"))
    spread_rate = None
    if bid is not None and ask is not None and ask >= bid:
        midpoint = (ask + bid) / 2
        spread_rate = (ask - bid) / midpoint if midpoint else None
    depth_score = None
    if bid_depth is not None and ask_depth is not None:
        total = bid_depth + ask_depth
        depth_score = min(1.0, math.log1p(total) / math.log1p(1_000_000))
    return spread_rate, depth_score


def _liquidity_score(volume_zscore: float | None, spread_rate: float | None, depth_score: float | None) -> float | None:
    if spread_rate is None or depth_score is None:
        return None
    spread_component = max(0.0, min(1.0, 1 - spread_rate / 0.01))
    volume_component = 0.5 if volume_zscore is None else max(0.0, min(1.0, 0.5 + volume_zscore / 6))
    return (spread_component * 0.45) + (depth_score * 0.35) + (volume_component * 0.20)


def _market_alignment_score(
    bars: Sequence[OHLCVBar],
    market_index_bars: Sequence[OHLCVBar],
    timestamp: datetime,
) -> float | None:
    visible_index = tuple(bar for bar in sorted(market_index_bars, key=lambda bar: bar.as_of) if bar.as_of <= timestamp)
    ticker_ret = _return_from_minutes(bars, 5)
    market_ret = _return_from_minutes(visible_index, 5)
    if ticker_ret is None or market_ret is None:
        return None
    if ticker_ret == 0 or market_ret == 0:
        return 0.5
    return 1.0 if ticker_ret * market_ret > 0 else 0.0


def _time_of_day_weight(timestamp: datetime, config: ShortHorizonFeatureConfig) -> float | None:
    open_dt = datetime.combine(timestamp.date(), config.session_open, tzinfo=timestamp.tzinfo)
    close_dt = datetime.combine(timestamp.date(), config.session_close, tzinfo=timestamp.tzinfo)
    if timestamp < open_dt or timestamp > close_dt:
        return None
    elapsed_minutes = (timestamp - open_dt).total_seconds() / 60
    total_minutes = (close_dt - open_dt).total_seconds() / 60
    remaining_minutes = total_minutes - elapsed_minutes
    if elapsed_minutes <= 30:
        return 1.0
    if remaining_minutes <= 30:
        return 0.8
    return 0.6


def _first_bar_at_or_after(bars: Sequence[OHLCVBar], timestamp: datetime) -> OHLCVBar | None:
    for bar in bars:
        if bar.as_of >= timestamp:
            return bar
    return None


def _last_bar_at_or_before(bars: Sequence[OHLCVBar], timestamp: datetime) -> OHLCVBar | None:
    candidate = None
    for bar in bars:
        if bar.as_of <= timestamp:
            candidate = bar
        else:
            break
    return candidate


def _safe_return(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    value = current / previous - 1
    return value if math.isfinite(value) else None


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number <= 0 or not math.isfinite(number):
        return None
    return number
