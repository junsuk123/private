"""Execution-quality layer.

Short-horizon profitability is dominated by execution quality: a positive expected
net edge is destroyed if spread, slippage, or poor fills eat the alpha. This layer
evaluates spread burden, order-book pressure, expected slippage, and fill probability
BEFORE submission, and rejects a BUY when the expected fill would invalidate the net
edge. It also provides a limit-order price/timeout/reprice policy and consumes realized
slippage history (from :mod:`app.storage.execution_quality_store`) so a symbol with
persistently bad fills is down-scored or blocked.

Formulas (spec):
    spread_alpha_ratio        = spread_rate / max(gross_expected_return, eps)
    realized_slippage_rate    = |fill_price - decision_reference_price| / decision_reference_price
    execution_adjusted_net    = net_expected_return - expected_extra_slippage_rate
Reject when spread_alpha_ratio > max_spread_alpha_ratio, or when the execution-adjusted
net return drops below the required minimum.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


@dataclass(frozen=True)
class ExecutionQualityInput:
    symbol: str
    strategy_family: str
    decision_reference_price: float
    gross_expected_return: float
    net_expected_return: float
    required_min_net_return: float
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    order_quantity: int = 1
    side: str = "BUY"


@dataclass(frozen=True)
class ExecutionQualityAssessment:
    allowed: bool
    symbol: str
    spread_rate: float
    spread_alpha_ratio: float
    expected_slippage_rate: float
    orderbook_pressure: float          # >0 favours buyers (more bid depth), <0 favours sellers
    fill_probability: float
    execution_adjusted_net_return: float
    limit_price: float
    reject_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionQualityEngine:
    def __init__(self, store: "Any | None" = None) -> None:
        self.store = store
        self.max_spread_alpha_ratio = _env_float("EXEC_MAX_SPREAD_ALPHA_RATIO", 0.35)
        self.max_expected_slippage_rate = _env_float("EXEC_MAX_EXPECTED_SLIPPAGE_RATE", 0.004)
        self.min_fill_probability = _env_float("EXEC_MIN_FILL_PROBABILITY", 0.3)
        # A symbol whose recent realized slippage exceeds this is blocked/penalized.
        self.max_recent_realized_slippage = _env_float("EXEC_MAX_RECENT_REALIZED_SLIPPAGE", 0.006)

    def assess(self, request: ExecutionQualityInput) -> ExecutionQualityAssessment:
        bid = max(0.0, float(request.best_bid))
        ask = max(0.0, float(request.best_ask))
        ref = max(_EPSILON, float(request.decision_reference_price))
        warnings: list[str] = []

        if bid > 0 and ask > 0 and ask >= bid:
            mid = (bid + ask) / 2.0
            spread_rate = (ask - bid) / mid
        else:
            # No usable book: fall back to the decision price with an unknown-spread flag.
            mid = ref
            spread_rate = 0.0
            warnings.append("NO_ORDERBOOK")

        # Order-book pressure in [-1, 1]; positive means bid-heavy (supports a buy).
        total_depth = float(request.bid_depth) + float(request.ask_depth)
        orderbook_pressure = (
            (float(request.bid_depth) - float(request.ask_depth)) / total_depth if total_depth > 0 else 0.0
        )

        gross = abs(float(request.gross_expected_return))
        spread_alpha_ratio = spread_rate / max(gross, _EPSILON)

        # Expected slippage: half-spread plus a depth-driven impact term, floored by any
        # persistently-bad realized history for this symbol.
        expected_slippage_rate = spread_rate / 2.0
        if request.side.upper() == "BUY" and orderbook_pressure < 0:
            # Thin bid relative to ask -> a buy is likelier to walk the book up.
            expected_slippage_rate += min(0.003, abs(orderbook_pressure) * spread_rate)
        recent_realized = self._recent_realized_slippage(request.symbol, request.strategy_family)
        if recent_realized is not None:
            expected_slippage_rate = max(expected_slippage_rate, recent_realized)

        # Fill probability heuristic: a marketable-ish limit at the ask with supportive
        # pressure fills easily; a wide spread with adverse pressure fills poorly.
        fill_probability = _clamp(0.9 - spread_alpha_ratio + 0.1 * orderbook_pressure, 0.0, 1.0)

        execution_adjusted_net = float(request.net_expected_return) - expected_slippage_rate

        # Limit price: for a BUY, cap at the ask (do not chase above it); if no book, use ref.
        limit_price = ask if (request.side.upper() == "BUY" and ask > 0) else ref

        reject_reason: str | None = None
        # Check the symbol's realized-slippage history first — it is the most actionable
        # reason (a symbol that persistently fills badly should be blocked by name).
        if recent_realized is not None and recent_realized > self.max_recent_realized_slippage:
            reject_reason = "EXEC_SYMBOL_SLIPPAGE_HISTORY_BAD"
        elif spread_alpha_ratio > self.max_spread_alpha_ratio + _EPSILON:
            reject_reason = "EXEC_SPREAD_CONSUMES_ALPHA"
        elif expected_slippage_rate > self.max_expected_slippage_rate + _EPSILON:
            reject_reason = "EXEC_EXPECTED_SLIPPAGE_TOO_HIGH"
        elif execution_adjusted_net < float(request.required_min_net_return) - _EPSILON:
            reject_reason = "EXEC_ADJUSTED_NET_BELOW_MIN"
        elif fill_probability < self.min_fill_probability:
            reject_reason = "EXEC_FILL_PROBABILITY_TOO_LOW"

        return ExecutionQualityAssessment(
            allowed=reject_reason is None,
            symbol=request.symbol,
            spread_rate=spread_rate,
            spread_alpha_ratio=spread_alpha_ratio,
            expected_slippage_rate=expected_slippage_rate,
            orderbook_pressure=orderbook_pressure,
            fill_probability=fill_probability,
            execution_adjusted_net_return=execution_adjusted_net,
            limit_price=limit_price,
            reject_reason=reject_reason,
            warnings=tuple(warnings),
        )

    def record_fill(
        self,
        *,
        symbol: str,
        strategy_family: str,
        decision_reference_price: float,
        fill_price: float,
        side: str = "BUY",
        time_bucket: str | None = None,
    ) -> float:
        """Record a realized fill and return the realized slippage rate."""
        ref = max(_EPSILON, float(decision_reference_price))
        realized = abs(float(fill_price) - ref) / ref
        if self.store is not None:
            try:
                self.store.record(
                    symbol=symbol,
                    strategy_family=strategy_family,
                    realized_slippage_rate=realized,
                    side=side,
                    time_bucket=time_bucket,
                )
            except Exception:  # noqa: BLE001 - persistence is best-effort.
                logger.warning("execution-quality store record failed for %s", symbol)
        return realized

    def _recent_realized_slippage(self, symbol: str, strategy_family: str) -> float | None:
        if self.store is None:
            return None
        try:
            return self.store.recent_average(symbol=symbol, strategy_family=strategy_family)
        except Exception:  # noqa: BLE001
            return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)
