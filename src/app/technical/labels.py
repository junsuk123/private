"""Supervised short-horizon labels from adjacent realtime frames.

Labels are built from the FUTURE price path relative to a decision, using only
observations that actually exist (no look-ahead beyond available data, no
synthetic/sample/hash rows). Net-profitable-after-cost labels deduct realistic
round-trip costs via :class:`TradingCostEngine`.

The builder is source-agnostic and pure: the caller supplies the realized
future path (e.g. from ``RealtimeMarketDataStore.recent_ticks`` after the
decision time). Each horizon whose data does not yet exist yields ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

# Source tags that must never be used for live/paper learning.
_SYNTHETIC_SOURCE_TOKENS = ("synthetic", "sample", "reference", "demo", "hash", "mock", "fixture")


def is_synthetic_source(source: str) -> bool:
    s = str(source or "").strip().lower()
    return any(token in s for token in _SYNTHETIC_SOURCE_TOKENS)


class _CostEngineLike(Protocol):
    def estimate(self, **kwargs) -> object: ...


@dataclass(frozen=True)
class LabelConfig:
    horizons_seconds: tuple[int, ...] = (5, 15, 30, 60, 300)
    net_label_horizon_seconds: int = 60
    take_profit_bps: float = 25.0
    stop_loss_bps: float = 25.0
    venue: str = "KRX"
    instrument_type: str = "domestic_stock"
    market: str = "KR"


@dataclass(frozen=True)
class ShortHorizonLabels:
    symbol: str
    entry_price: float
    future_return_5s: float | None
    future_return_15s: float | None
    future_return_30s: float | None
    future_return_60s: float | None
    future_return_5m: float | None
    max_favorable_excursion_bps: float | None
    max_adverse_excursion_bps: float | None
    net_profitable_after_cost_label: int | None
    hit_take_profit_before_stop_label: int | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "future_return_5s": self.future_return_5s,
            "future_return_15s": self.future_return_15s,
            "future_return_30s": self.future_return_30s,
            "future_return_60s": self.future_return_60s,
            "future_return_5m": self.future_return_5m,
            "max_favorable_excursion_bps": self.max_favorable_excursion_bps,
            "max_adverse_excursion_bps": self.max_adverse_excursion_bps,
            "net_profitable_after_cost_label": self.net_profitable_after_cost_label,
            "hit_take_profit_before_stop_label": self.hit_take_profit_before_stop_label,
            "metadata": dict(self.metadata),
        }


def _price_at_or_after(path: Sequence[tuple[float, float]], seconds: float) -> float | None:
    for offset, price in path:
        if offset >= seconds and price > 0:
            return price
    return None


class LabelBuilder:
    def __init__(self, cost_engine: _CostEngineLike | None = None, config: LabelConfig | None = None) -> None:
        self.cost_engine = cost_engine
        self.config = config or LabelConfig()

    def build(
        self,
        *,
        symbol: str,
        entry_price: float,
        future_path: Sequence[tuple[float, float]],
        source: str = "",
        source_freshness_ms: float | None = None,
        spread_bps: float | None = None,
        expected_slippage_bps: float | None = None,
        quantity: int = 1,
        average_daily_trading_value: float | None = None,
    ) -> ShortHorizonLabels | None:
        """Build labels or return ``None`` if inputs are unusable.

        ``future_path`` is a sequence of ``(seconds_after_decision, price)``
        sorted ascending, containing only realized (already-observed) points.
        """
        if is_synthetic_source(source):
            return None
        if entry_price is None or entry_price <= 0:
            return None
        path = sorted(
            ((float(o), float(p)) for o, p in future_path if p and p > 0 and o >= 0),
            key=lambda x: x[0],
        )
        cfg = self.config

        def _ret(seconds: int) -> float | None:
            px = _price_at_or_after(path, seconds)
            return (px / entry_price - 1.0) if px is not None else None

        returns = {s: _ret(s) for s in (5, 15, 30, 60, 300)}

        # MFE/MAE over the realized path up to the largest horizon.
        max_h = max(cfg.horizons_seconds) if cfg.horizons_seconds else 300
        window = [p for o, p in path if o <= max_h]
        mfe_bps = mae_bps = None
        if window:
            mfe_bps = max((p / entry_price - 1.0) for p in window) * 10_000.0
            mae_bps = min((p / entry_price - 1.0) for p in window) * 10_000.0

        # Take-profit-before-stop: walk the path in order.
        tp = entry_price * (1.0 + cfg.take_profit_bps / 10_000.0)
        stop = entry_price * (1.0 - cfg.stop_loss_bps / 10_000.0)
        hit_tp_before_stop: int | None = None
        for _, price in path:
            if price >= tp:
                hit_tp_before_stop = 1
                break
            if price <= stop:
                hit_tp_before_stop = 0
                break

        # Net-profitable-after-cost at the primary horizon.
        net_label: int | None = None
        cost_rate = None
        net_return = None
        primary_ret = returns.get(cfg.net_label_horizon_seconds)
        if primary_ret is not None:
            exit_price = entry_price * (1.0 + primary_ret)
            if self.cost_engine is not None:
                breakdown = self.cost_engine.estimate(
                    symbol=symbol,
                    market=cfg.market,
                    venue=cfg.venue,
                    instrument_type=cfg.instrument_type,
                    entry_price=entry_price,
                    expected_exit_price=exit_price,
                    quantity=max(1, int(quantity)),
                    average_daily_trading_value=average_daily_trading_value,
                )
                net_return = float(getattr(breakdown, "net_expected_return", primary_ret))
                cost_rate = float(getattr(breakdown, "total_cost_rate", 0.0))
            else:
                cost_rate = ((spread_bps or 0.0) + (expected_slippage_bps or 0.0)) / 10_000.0
                net_return = primary_ret - cost_rate
            net_label = 1 if net_return > 0 else 0

        metadata = {
            "net_label_horizon_seconds": cfg.net_label_horizon_seconds,
            "source": source,
            "source_freshness_ms": source_freshness_ms,
            "spread_bps": spread_bps,
            "expected_slippage_bps": expected_slippage_bps,
            "cost_rate_assumed": cost_rate,
            "net_return_after_cost": net_return,
            "path_points": len(path),
            "take_profit_bps": cfg.take_profit_bps,
            "stop_loss_bps": cfg.stop_loss_bps,
        }
        return ShortHorizonLabels(
            symbol=symbol,
            entry_price=entry_price,
            future_return_5s=returns[5],
            future_return_15s=returns[15],
            future_return_30s=returns[30],
            future_return_60s=returns[60],
            future_return_5m=returns[300],
            max_favorable_excursion_bps=mfe_bps,
            max_adverse_excursion_bps=mae_bps,
            net_profitable_after_cost_label=net_label,
            hit_take_profit_before_stop_label=hit_tp_before_stop,
            metadata=metadata,
        )
