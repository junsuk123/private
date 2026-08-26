from __future__ import annotations

import math
from datetime import datetime
from typing import Mapping, Sequence


def simple_returns(prices: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in prices)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("prices must be finite and positive")
    return tuple(values[index] / values[index - 1] - 1.0 for index in range(1, len(values)))


def covariance(left: Sequence[float], right: Sequence[float]) -> float | None:
    x, y = _aligned(left, right)
    if len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - 1)


def correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    cov = covariance(left, right)
    if cov is None:
        return None
    x, y = _aligned(left, right)
    vx, vy = covariance(x, x), covariance(y, y)
    if not vx or not vy or vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def beta(asset_returns: Sequence[float], benchmark_returns: Sequence[float]) -> float | None:
    cov = covariance(asset_returns, benchmark_returns)
    variance = covariance(benchmark_returns, benchmark_returns)
    if cov is None or variance is None or variance <= 0:
        return None
    return cov / variance


def portfolio_returns(
    returns_by_symbol: Mapping[str, Sequence[float]], weights: Mapping[str, float]
) -> tuple[float, ...]:
    if not returns_by_symbol:
        return ()
    unknown = set(weights) - set(returns_by_symbol)
    if unknown:
        raise ValueError(f"missing real return series for {sorted(unknown)}")
    total = sum(float(value) for value in weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("portfolio weights must sum to one")
    length = min(len(tuple(returns_by_symbol[symbol])) for symbol in weights)
    if length == 0:
        return ()
    return tuple(
        sum(float(weights[symbol]) * float(tuple(returns_by_symbol[symbol])[-length + index]) for symbol in weights)
        for index in range(length)
    )


def portfolio_volatility(returns: Sequence[float], annualization: int) -> float | None:
    values = tuple(float(value) for value in returns)
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance * annualization)


def concentration(weights: Mapping[str, float]) -> float:
    return sum(float(weight) ** 2 for weight in weights.values())


def max_drawdown(returns: Sequence[float]) -> float | None:
    wealth = peak = 1.0
    drawdown = 0.0
    seen = False
    for value in returns:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("returns must be finite")
        wealth *= 1.0 + number
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
        seen = True
    return drawdown if seen else None


def _aligned(left: Sequence[float], right: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    size = min(len(left), len(right))
    if size <= 0:
        return (), ()
    x, y = tuple(float(value) for value in left)[-size:], tuple(float(value) for value in right)[-size:]
    if any(not math.isfinite(value) for value in (*x, *y)):
        raise ValueError("series must be finite")
    return x, y
