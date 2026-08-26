from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.quant.config import QuantConfig, default_quant_config
from app.quant.contracts import (
    DataQuality,
    QuantBar,
    QuantEvidence,
    ValidationStatus,
    aware_utc,
)

LOCAL_IMPLEMENTATION = "local_quant_engine:incremental-v1"
_GS = "gs_quant:2.1.3"


@dataclass
class _State:
    closes: deque[float]
    returns: deque[float]
    gains: deque[float]
    losses: deque[float]
    times: deque[datetime]
    ema_fast: float | None = None
    ema_slow: float | None = None
    macd_signal: float | None = None
    smoothed_average: float | None = None
    avg_gain: float | None = None
    avg_loss: float | None = None
    last_close: float | None = None
    peak: float | None = None
    max_drawdown: float = 0.0
    count: int = 0
    last_received: datetime | None = None
    last_end: datetime | None = None
    evidence: tuple[QuantEvidence, ...] = ()


class QuantEvidenceCache:
    """Bounded O(1) latest-evidence cache; no pandas and no window scans."""

    def __init__(self, max_items: int = 10_000) -> None:
        self._max_items = max(1, int(max_items))
        self._items: OrderedDict[tuple[str, str, str, str], QuantEvidence] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def put_many(self, evidence: Sequence[QuantEvidence]) -> None:
        with self._lock:
            for item in evidence:
                key = (item.market, item.symbol, item.bar_interval, item.metric)
                self._items[key] = item
                self._items.move_to_end(key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)

    def latest(self, market: str, symbol: str, interval: str) -> tuple[QuantEvidence, ...]:
        prefix = (market, symbol, interval)
        with self._lock:
            rows = tuple(value for key, value in self._items.items() if key[:3] == prefix)
            if rows:
                self._hits += 1
            else:
                self._misses += 1
            return rows

    def health(self) -> dict[str, Any]:
        with self._lock:
            requests = self._hits + self._misses
            return {
                "items": len(self._items),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / requests if requests else 0.0,
            }


class IncrementalQuantEngine:
    """Completed-bar incremental statistics for the live path.

    Each update is O(max configured window) only for the small rolling snapshot and
    never imports pandas or GS Quant. Duplicate/out-of-order bars are rejected so an
    older observation cannot rewrite current evidence.
    """

    def __init__(self, config: QuantConfig | None = None) -> None:
        self.config = config or default_quant_config()
        self.cache = QuantEvidenceCache(self.config.cache_size)
        self._states: dict[tuple[str, str, str], _State] = {}
        self._lock = threading.RLock()
        self._updates = 0
        self._invalid = 0
        self._latency_total_ms = 0.0

    def update(self, bar: QuantBar, *, as_of: datetime | None = None) -> tuple[QuantEvidence, ...]:
        started = time.perf_counter()
        decision_time = aware_utc(as_of or bar.received_at)
        received = aware_utc(bar.received_at)
        ended = aware_utc(bar.end_time)
        if received > decision_time:
            raise ValueError("lookahead blocked: bar was not received by as_of")
        if ended > decision_time:
            raise ValueError("incomplete/future bar cannot produce evidence")
        key = (bar.market, bar.symbol, bar.interval)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                capacity = max(self.config.price_window, self.config.return_window) + 1
                state = _State(
                    deque(maxlen=capacity), deque(maxlen=capacity),
                    deque(maxlen=capacity), deque(maxlen=capacity),
                    deque(maxlen=capacity),
                )
                self._states[key] = state
            if state.last_received is not None and received <= state.last_received:
                raise ValueError("duplicate or out-of-order knowledge time")
            if state.last_end is not None and ended <= state.last_end:
                raise ValueError("duplicate or out-of-order bar end")
            self._advance(state, float(bar.close), ended, received)
            evidence = self._snapshot(bar, state, decision_time)
            state.evidence = evidence
            self.cache.put_many(evidence)
            self._updates += 1
            self._invalid += sum(item.data_quality is DataQuality.INVALID for item in evidence)
            self._latency_total_ms += (time.perf_counter() - started) * 1000.0
            return evidence

    def compute_features(self, bars: Sequence[QuantBar], *, as_of: datetime | None = None) -> tuple[QuantEvidence, ...]:
        result: tuple[QuantEvidence, ...] = ()
        for bar in sorted(bars, key=lambda item: item.received_at):
            if as_of is not None and aware_utc(bar.received_at) > aware_utc(as_of):
                continue
            result = self.update(bar, as_of=as_of or bar.received_at)
        return result

    def compute_risk_metrics(self, context: Mapping[str, Any]) -> tuple[QuantEvidence, ...]:
        return self.cache.latest(str(context.get("market", "")), str(context.get("symbol", "")), str(context.get("interval", "1m")))

    def evaluate_portfolio(self, context: Mapping[str, Any]) -> tuple[QuantEvidence, ...]:
        from app.quant.portfolio import (
            beta, concentration, correlation, max_drawdown, portfolio_returns,
            portfolio_volatility,
        )

        portfolio_id = str(context.get("portfolio_id") or "").strip()
        market = str(context.get("market") or "").strip()
        interval = str(context.get("interval") or "").strip()
        if not portfolio_id or not market or not interval:
            raise ValueError("portfolio_id, market, and interval are required")
        as_of = aware_utc(context["as_of"])
        input_start = aware_utc(context["input_start"])
        input_end = aware_utc(context["input_end"])
        if input_end > as_of:
            raise ValueError("portfolio input cannot be newer than as_of")
        returns_by_symbol = context.get("returns_by_symbol")
        weights = context.get("weights")
        if not isinstance(returns_by_symbol, Mapping) or not isinstance(weights, Mapping):
            raise ValueError("real returns_by_symbol and weights are required")
        portfolio = portfolio_returns(returns_by_symbol, weights)
        benchmark = context.get("benchmark_returns")
        correlations = []
        symbols = tuple(weights)
        for index, left in enumerate(symbols):
            for right in symbols[index + 1:]:
                value = correlation(returns_by_symbol[left], returns_by_symbol[right])
                if value is not None:
                    correlations.append(value)
        values: tuple[tuple[str, float | None, str | None], ...] = (
            ("portfolio_return", portfolio[-1] if portfolio else None, "empty_aligned_returns"),
            ("portfolio_volatility", portfolio_volatility(portfolio, self.config.annualization), "insufficient_aligned_returns"),
            ("concentration", concentration(weights), None),
            ("cross_asset_correlation", sum(correlations) / len(correlations) if correlations else None, "insufficient_cross_asset_pairs"),
            ("benchmark_beta", beta(portfolio, benchmark) if isinstance(benchmark, Sequence) else None, "benchmark_returns_unavailable"),
            ("drawdown", max_drawdown(portfolio), "empty_aligned_returns"),
            # No observed risk-free series => no invented Sharpe.
            ("sharpe", None, "risk_free_rate_unavailable"),
        )
        output = []
        for metric, value, reason in values:
            available = value is not None
            output.append(QuantEvidence(
                symbol=portfolio_id, market=market, timestamp=as_of,
                bar_interval=interval, metric=metric, value=value,
                window=len(portfolio) or None, input_start=input_start, input_end=input_end,
                freshness_ms=max(0.0, (as_of - input_end).total_seconds() * 1000.0),
                data_quality=DataQuality.GOOD if available else DataQuality.DEGRADED,
                implementation=LOCAL_IMPLEMENTATION,
                method_reference=f"{_GS}:timeseries.local_portfolio_concept",
                validation_status=ValidationStatus.UNVALIDATED,
                unavailable_reason=None if available else reason,
            ))
        return tuple(output)

    def run_scenario(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"available": False, "unavailable_reason": "no_local_scenario_supplied"}

    def health(self) -> Mapping[str, Any]:
        with self._lock:
            latest = max((s.last_received for s in self._states.values() if s.last_received), default=None)
            return {
                "provider": "local_quant_engine",
                "reference_version": "2.1.3",
                "hot_path_gs_quant_import": False,
                "updates": self._updates,
                "invalid_metric_count": self._invalid,
                "calculation_latency_mean_ms": self._latency_total_ms / self._updates if self._updates else 0.0,
                "latest_evidence_at": latest.isoformat() if latest else None,
                "cache": self.cache.health(),
            }

    def _advance(self, state: _State, close: float, ended: datetime, received: datetime) -> None:
        cfg = self.config
        previous = state.last_close
        state.closes.append(close)
        state.times.append(ended)
        change = 0.0
        if previous is not None:
            state.returns.append(close / previous - 1.0)
            change = close - previous
            state.gains.append(max(change, 0.0))
            state.losses.append(max(-change, 0.0))
        alpha_fast = 2.0 / (cfg.ema_fast + 1.0)
        alpha_slow = 2.0 / (cfg.ema_slow + 1.0)
        state.ema_fast = close if state.ema_fast is None else alpha_fast * close + (1 - alpha_fast) * state.ema_fast
        state.ema_slow = close if state.ema_slow is None else alpha_slow * close + (1 - alpha_slow) * state.ema_slow
        macd = state.ema_fast - state.ema_slow
        alpha_signal = 2.0 / (cfg.macd_signal + 1.0)
        state.macd_signal = macd if state.macd_signal is None else alpha_signal * macd + (1 - alpha_signal) * state.macd_signal
        next_count = state.count + 1
        # Match GS Quant's Window(int) ramp: the first value is produced after
        # ``window`` prior observations, using the current trailing window.
        if next_count == cfg.price_window + 1:
            state.smoothed_average = sum(tuple(state.closes)[-cfg.price_window:]) / cfg.price_window
        elif next_count > cfg.price_window + 1 and state.smoothed_average is not None:
            state.smoothed_average = (state.smoothed_average * (cfg.price_window - 1) + close) / cfg.price_window
        if next_count == cfg.rsi_window + 2:
            state.avg_gain = sum(tuple(state.gains)[-cfg.rsi_window:]) / cfg.rsi_window
            state.avg_loss = sum(tuple(state.losses)[-cfg.rsi_window:]) / cfg.rsi_window
        elif next_count > cfg.rsi_window + 2 and state.avg_gain is not None and state.avg_loss is not None:
            gain, loss = state.gains[-1], state.losses[-1]
            state.avg_gain = (state.avg_gain * (cfg.rsi_window - 1) + gain) / cfg.rsi_window
            state.avg_loss = (state.avg_loss * (cfg.rsi_window - 1) + loss) / cfg.rsi_window
        state.peak = close if state.peak is None else max(state.peak, close)
        state.max_drawdown = min(state.max_drawdown, close / state.peak - 1.0)
        state.last_close, state.last_end, state.last_received = close, ended, received
        state.count += 1

    def _snapshot(self, bar: QuantBar, state: _State, as_of: datetime) -> tuple[QuantEvidence, ...]:
        cfg = self.config
        prices = list(state.closes)
        returns = list(state.returns)
        window_prices = prices[-cfg.price_window:]
        window_returns = returns[-cfg.return_window:]
        warm_price = state.count > cfg.price_window
        warm_return = len(window_returns) >= cfg.return_window
        mean = sum(window_prices) / len(window_prices)
        std = _sample_std(window_prices)
        return_mean = sum(window_returns) / len(window_returns) if window_returns else None
        return_std = _sample_std(window_returns) if len(window_returns) > 1 else None
        current = prices[-1]
        rsi_value = None
        if state.count >= cfg.rsi_window + 2 and state.avg_gain is not None and state.avg_loss is not None:
            rsi_value = 100.0 if state.avg_loss == 0 else 100.0 - 100.0 / (1.0 + state.avg_gain / state.avg_loss)
        freshness = max(0.0, (as_of - aware_utc(bar.received_at)).total_seconds() * 1000.0)
        quality = DataQuality.GOOD if freshness <= cfg.stale_after_ms else DataQuality.DEGRADED
        start = state.times[-min(len(state.times), cfg.price_window)]
        metrics: list[tuple[str, float | None, int | None, bool, str, str | None]] = [
            ("simple_return", returns[-1] if returns else None, 1, bool(returns), f"{_GS}:timeseries.statistics.returns", "warmup"),
            ("log_return", math.log(current / prices[-2]) if len(prices) > 1 else None, 1, len(prices) > 1, f"{_GS}:timeseries.statistics.log_return", "warmup"),
            ("rolling_mean", mean if warm_price else None, cfg.price_window, warm_price, f"{_GS}:timeseries.statistics.mean", "warmup"),
            ("rolling_std", std if warm_price else None, cfg.price_window, warm_price, f"{_GS}:timeseries.statistics.std", "warmup"),
            ("zscore", (current - mean) / std if warm_price and std and std > 0 else None, cfg.price_window, warm_price and bool(std), f"{_GS}:timeseries.statistics.zscores", "zero_variance_or_warmup"),
            ("moving_average", mean if warm_price else None, cfg.price_window, warm_price, f"{_GS}:timeseries.technicals.moving_average", "warmup"),
            ("exponential_moving_average", state.ema_fast, cfg.ema_fast, state.count >= cfg.ema_fast, f"{_GS}:timeseries.technicals.exponential_moving_average", "warmup"),
            ("smoothed_moving_average", state.smoothed_average, cfg.price_window, warm_price, f"{_GS}:timeseries.technicals.smoothed_moving_average", "warmup"),
            ("bollinger_lower", mean - cfg.bollinger_stddev * std if warm_price and std is not None else None, cfg.price_window, warm_price, f"{_GS}:timeseries.technicals.bollinger_bands", "warmup"),
            ("bollinger_upper", mean + cfg.bollinger_stddev * std if warm_price and std is not None else None, cfg.price_window, warm_price, f"{_GS}:timeseries.technicals.bollinger_bands", "warmup"),
            ("bollinger_width", (2 * cfg.bollinger_stddev * std / mean) if warm_price and std is not None and mean else None, cfg.price_window, warm_price and bool(mean), f"{_GS}:timeseries.technicals.bollinger_bands", "warmup_or_zero_mean"),
            ("bollinger_position", (current - (mean - cfg.bollinger_stddev * std)) / (2 * cfg.bollinger_stddev * std) if warm_price and std and std > 0 else None, cfg.price_window, warm_price and bool(std), f"{_GS}:timeseries.technicals.bollinger_bands", "zero_variance_or_warmup"),
            ("macd", state.ema_fast - state.ema_slow if state.ema_fast is not None and state.ema_slow is not None else None, cfg.ema_slow, state.count >= cfg.ema_slow, f"{_GS}:timeseries.technicals.macd", "warmup"),
            ("macd_signal", state.macd_signal, cfg.macd_signal, state.count >= cfg.ema_slow + cfg.macd_signal - 1, f"{_GS}:timeseries.technicals.macd", "warmup"),
            ("rsi", rsi_value, cfg.rsi_window, rsi_value is not None, f"{_GS}:timeseries.technicals.relative_strength_index", "warmup"),
            ("trend", (state.ema_fast / state.ema_slow - 1.0) if state.count >= cfg.ema_slow and state.ema_slow else None, cfg.ema_slow, state.count >= cfg.ema_slow, f"{_GS}:timeseries.technicals.exponential_moving_average", "warmup"),
            ("realized_volatility", return_std * math.sqrt(cfg.annualization) if warm_return and return_std is not None else None, cfg.return_window, warm_return, f"{_GS}:timeseries.statistics.std", "warmup"),
            ("max_drawdown", state.max_drawdown, None, True, f"{_GS}:timeseries.statistics.min_", None),
        ]
        result = []
        for metric, value, window, available, reference, reason in metrics:
            result.append(QuantEvidence(
                symbol=bar.symbol, market=bar.market, timestamp=as_of,
                bar_interval=bar.interval, metric=metric, value=value if available else None,
                window=window, input_start=start, input_end=aware_utc(bar.end_time),
                freshness_ms=freshness,
                data_quality=quality if available else DataQuality.DEGRADED,
                implementation=LOCAL_IMPLEMENTATION, method_reference=reference,
                validation_status=ValidationStatus.UNVALIDATED,
                unavailable_reason=None if available else reason,
            ))
        return tuple(result)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))
