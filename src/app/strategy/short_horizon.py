from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Sequence

from app.features.indicator_engine import sma
from app.features.schemas import OHLCVBar
from app.features.short_horizon_features import ShortHorizonFeatures
from app.strategy.candidates import StrategyCandidate


@dataclass(frozen=True)
class ShortTermReversalConfig:
    enabled: bool = True
    paper_only: bool = True
    rebound_ratio: float = 0.35
    max_rebound_cap: float = 0.006
    min_shock_vol_multiple: float = 1.2
    target_net_return: float = 0.003
    max_spread_rate: float = 0.0015
    min_liquidity_score: float = 0.5
    expected_holding_minutes: int = 30
    epsilon: float = 1e-9


class ShortTermReversalEngine:
    """Creates short-horizon reversal candidates without creating orders."""

    strategy_family = "short_term_reversal"
    signal_name = "jegadeesh_1990_short_term_reversal"

    def __init__(self, config: ShortTermReversalConfig | None = None) -> None:
        self.config = config or ShortTermReversalConfig()

    def generate_candidate(
        self,
        features: ShortHorizonFeatures,
        *,
        entry_price: float,
        trading_mode: str = "paper",
    ) -> StrategyCandidate | None:
        if not self.config.enabled:
            return None
        if self.config.paper_only and trading_mode != "paper":
            return None
        if entry_price <= 0 or not features.is_valid:
            return None

        ret_5m = features.returns_by_window.get("ret_5m")
        ret_15m = features.returns_by_window.get("ret_15m")
        vol_5m = features.realized_volatility.get("realized_volatility_5m")
        vol_30m = features.realized_volatility.get("realized_volatility_30m")
        if ret_5m is None or vol_5m is None or vol_5m <= 0:
            return None
        if features.spread_rate is None or features.liquidity_score is None:
            return None
        if features.spread_rate >= self.config.max_spread_rate:
            return None
        if features.liquidity_score <= self.config.min_liquidity_score:
            return None

        shock_score = abs(ret_5m) / max(vol_5m, self.config.epsilon)
        if ret_5m >= 0 or shock_score < self.config.min_shock_vol_multiple:
            return None

        expected_rebound = min(self.config.max_rebound_cap, abs(ret_5m) * self.config.rebound_ratio)
        if expected_rebound <= 0:
            return None
        expected_exit_price = entry_price * (1 + expected_rebound)
        gross_expected_return = expected_exit_price / entry_price - 1
        confidence = self._confidence(
            shock_score=shock_score,
            ret_15m=ret_15m,
            vol_30m=vol_30m,
            liquidity_score=features.liquidity_score,
            spread_rate=features.spread_rate,
        )

        candidate_features = features.as_feature_dict()
        candidate_features.update(
            {
                "shock_score": shock_score,
                "expected_rebound": expected_rebound,
                "target_net_return": self.config.target_net_return,
                "entry_price": entry_price,
            }
        )
        return StrategyCandidate(
            ticker=features.ticker,
            strategy_family=self.strategy_family,
            signal_name=self.signal_name,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            expected_holding_minutes=self.config.expected_holding_minutes,
            gross_expected_return=gross_expected_return,
            confidence=confidence,
            features=candidate_features,
            ontology_tags=self._ontology_tags(gross_expected_return),
            reason=(
                "Recent negative short-horizon return is large relative to realized "
                "volatility with sufficient liquidity and controlled spread."
            ),
        )

    def _confidence(
        self,
        *,
        shock_score: float,
        ret_15m: float | None,
        vol_30m: float | None,
        liquidity_score: float,
        spread_rate: float,
    ) -> float:
        shock_component = min(1.0, shock_score / 3.0) * 0.45
        liquidity_component = min(1.0, liquidity_score) * 0.30
        spread_component = max(0.0, 1 - spread_rate / self.config.max_spread_rate) * 0.15
        context_component = 0.05
        if ret_15m is not None and ret_15m < 0:
            context_component += 0.03
        if vol_30m is not None and vol_30m > 0:
            context_component += 0.02
        return max(0.0, min(1.0, shock_component + liquidity_component + spread_component + context_component))

    def _ontology_tags(self, gross_expected_return: float) -> list[str]:
        tags = [
            "ShortTermOverreaction",
            "ShortTermReversalCandidate",
            "LiquiditySupportedReversal",
            "BidAskBounceRisk",
        ]
        if gross_expected_return >= self.config.target_net_return:
            tags.append("CostEfficientReversal")
        return tags


@dataclass(frozen=True)
class IntradayMomentumConfig:
    enabled: bool = True
    paper_only: bool = True
    opening_window_minutes: int = 30
    beta_r_open_to_late: float = 0.25
    target_net_return: float = 0.003
    min_open_return: float = 0.002
    min_volume_zscore: float = 0.5
    min_market_alignment_score: float = 0.5
    requires_market_alignment: bool = True
    expected_holding_minutes: int = 180
    session_open: time = time(9, 0)
    session_close: time = time(15, 30)


class IntradayMomentumEngine:
    """Creates opening-return momentum candidates without creating orders."""

    strategy_family = "intraday_momentum"
    signal_name = "gao_2018_opening_return_momentum"

    def __init__(self, config: IntradayMomentumConfig | None = None) -> None:
        self.config = config or IntradayMomentumConfig()

    def generate_candidate(
        self,
        features: ShortHorizonFeatures,
        *,
        entry_price: float,
        trading_mode: str = "paper",
    ) -> StrategyCandidate | None:
        if not self.config.enabled:
            return None
        if self.config.paper_only and trading_mode != "paper":
            return None
        if entry_price <= 0 or not features.is_valid:
            return None

        open_return_name = f"ret_open_{self.config.opening_window_minutes}m"
        r_open = features.returns_by_window.get(open_return_name)
        volume_zscore = features.volume_zscore
        market_alignment_score = features.market_alignment_score
        realized_volatility_30m = features.realized_volatility.get("realized_volatility_30m")
        time_of_day_weight = features.time_of_day_weight

        if r_open is None or volume_zscore is None or time_of_day_weight is None:
            return None
        if r_open <= self.config.min_open_return:
            return None
        if volume_zscore <= self.config.min_volume_zscore:
            return None
        if self.config.requires_market_alignment and market_alignment_score is None:
            return None
        if market_alignment_score is not None and market_alignment_score <= self.config.min_market_alignment_score:
            return None

        expected_late_return = self.config.beta_r_open_to_late * r_open
        if expected_late_return <= 0:
            return None
        expected_exit_price = entry_price * (1 + expected_late_return)
        gross_expected_return = expected_exit_price / entry_price - 1
        confidence = self._confidence(
            r_open=r_open,
            volume_zscore=volume_zscore,
            market_alignment_score=market_alignment_score,
            realized_volatility_30m=realized_volatility_30m,
            time_of_day_weight=time_of_day_weight,
        )

        candidate_features = features.as_feature_dict()
        candidate_features.update(
            {
                "opening_return_feature": open_return_name,
                "r_open": r_open,
                "beta_r_open_to_late": self.config.beta_r_open_to_late,
                "expected_late_return": expected_late_return,
                "target_net_return": self.config.target_net_return,
                "entry_price": entry_price,
                "session_open_hour": float(self.config.session_open.hour),
                "session_close_hour": float(self.config.session_close.hour),
            }
        )
        return StrategyCandidate(
            ticker=features.ticker,
            strategy_family=self.strategy_family,
            signal_name=self.signal_name,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            expected_holding_minutes=self.config.expected_holding_minutes,
            gross_expected_return=gross_expected_return,
            confidence=confidence,
            features=candidate_features,
            ontology_tags=self._ontology_tags(market_alignment_score),
            reason=(
                "Opening-window return is positive with volume confirmation and "
                "market-direction support under configurable intraday momentum beta."
            ),
        )

    def _confidence(
        self,
        *,
        r_open: float,
        volume_zscore: float,
        market_alignment_score: float | None,
        realized_volatility_30m: float | None,
        time_of_day_weight: float,
    ) -> float:
        return_component = min(1.0, r_open / max(self.config.min_open_return * 4, 1e-9)) * 0.35
        volume_component = min(1.0, max(0.0, volume_zscore / 3.0)) * 0.25
        market_component = (market_alignment_score if market_alignment_score is not None else 0.5) * 0.20
        time_component = max(0.0, min(1.0, time_of_day_weight)) * 0.10
        volatility_component = 0.10
        if realized_volatility_30m is not None:
            volatility_component = max(0.0, min(0.10, 0.10 - realized_volatility_30m))
        return max(
            0.0,
            min(
                1.0,
                return_component
                + volume_component
                + market_component
                + time_component
                + volatility_component,
            ),
        )

    def _ontology_tags(self, market_alignment_score: float | None) -> list[str]:
        tags = [
            "IntradayMomentum",
            "OpeningReturnStrength",
            "VolumeConfirmedMomentum",
            "LateDayContinuationCandidate",
        ]
        if market_alignment_score is not None and market_alignment_score > self.config.min_market_alignment_score:
            tags.append("MarketDirectionAligned")
        return tags


@dataclass(frozen=True)
class TechnicalRuleConfig:
    enabled: bool = True
    paper_only: bool = True
    ma_fast: int = 5
    ma_slow: int = 20
    range_window: int = 20
    breakout_buffer: float = 0.001
    volume_multiplier: float = 1.5
    breakout_capture_ratio: float = 0.4
    volatility_target: float = 0.006
    target_net_return: float = 0.003
    max_spread_rate: float = 0.0015
    min_liquidity_score: float = 0.5
    expected_holding_minutes: int = 60


class TechnicalRuleEngine:
    """Creates Brock et al. style technical-rule candidates without creating orders."""

    strategy_family = "technical_rule"
    signal_name = "brock_1992_technical_breakout"

    def __init__(self, config: TechnicalRuleConfig | None = None) -> None:
        self.config = config or TechnicalRuleConfig()

    def generate_candidate(
        self,
        features: ShortHorizonFeatures,
        bars: Sequence[OHLCVBar],
        *,
        entry_price: float,
        trading_mode: str = "paper",
    ) -> StrategyCandidate | None:
        if not self.config.enabled:
            return None
        if self.config.paper_only and trading_mode != "paper":
            return None
        if entry_price <= 0 or not features.is_valid:
            return None
        if features.spread_rate is None or features.spread_rate >= self.config.max_spread_rate:
            return None
        if features.liquidity_score is None or features.liquidity_score <= self.config.min_liquidity_score:
            return None

        visible = tuple(sorted((bar for bar in bars if bar.as_of <= features.timestamp), key=lambda bar: bar.as_of))
        min_bars = max(self.config.ma_slow + 1, self.config.range_window + 1)
        if len(visible) < min_bars:
            return None

        closes = [bar.close for bar in visible]
        volumes = [bar.volume for bar in visible]
        ma_signal = self._ma_crossover(closes)
        range_signal, breakout_width, rolling_high = self._range_breakout(visible)
        volume_confirmed = _volume_confirmed(volumes, self.config.range_window, self.config.volume_multiplier)
        if (ma_signal or range_signal) and not volume_confirmed:
            return None
        if not ma_signal and not range_signal:
            return None

        ma_gap = _ma_gap(closes, self.config.ma_fast, self.config.ma_slow) or 0.0
        vol_30m = features.realized_volatility.get("realized_volatility_30m") or 0.0
        signal_width = max(breakout_width or 0.0, ma_gap, vol_30m)
        expected_return = min(self.config.volatility_target, signal_width * self.config.breakout_capture_ratio)
        if expected_return <= 0:
            return None

        expected_exit_price = entry_price * (1 + expected_return)
        gross_expected_return = expected_exit_price / entry_price - 1
        candidate_features = features.as_feature_dict()
        candidate_features.update(
            {
                "ma_fast": float(self.config.ma_fast),
                "ma_slow": float(self.config.ma_slow),
                "range_window": float(self.config.range_window),
                "ma_crossover": 1.0 if ma_signal else 0.0,
                "range_breakout": 1.0 if range_signal else 0.0,
                "volume_confirmed": 1.0,
                "rolling_high": rolling_high or 0.0,
                "breakout_width": breakout_width or 0.0,
                "ma_gap": ma_gap,
                "expected_technical_return": expected_return,
                "target_net_return": self.config.target_net_return,
                "entry_price": entry_price,
            }
        )
        return StrategyCandidate(
            ticker=features.ticker,
            strategy_family=self.strategy_family,
            signal_name=self.signal_name,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            expected_holding_minutes=self.config.expected_holding_minutes,
            gross_expected_return=gross_expected_return,
            confidence=self._confidence(ma_signal, range_signal, volume_confirmed, features.liquidity_score, features.spread_rate),
            features=candidate_features,
            ontology_tags=self._ontology_tags(ma_signal, range_signal, volume_confirmed),
            reason=(
                "Technical breakout candidate based on moving-average crossover or "
                "trading-range breakout with volume confirmation."
            ),
        )

    def _ma_crossover(self, closes: Sequence[float]) -> bool:
        current_fast = sma(list(closes), self.config.ma_fast)
        current_slow = sma(list(closes), self.config.ma_slow)
        previous_fast = sma(list(closes[:-1]), self.config.ma_fast)
        previous_slow = sma(list(closes[:-1]), self.config.ma_slow)
        if None in (current_fast, current_slow, previous_fast, previous_slow):
            return False
        return bool(current_fast > current_slow and previous_fast <= previous_slow)

    def _range_breakout(self, bars: Sequence[OHLCVBar]) -> tuple[bool, float | None, float | None]:
        previous = bars[-self.config.range_window - 1 : -1]
        if len(previous) < self.config.range_window:
            return False, None, None
        rolling_high = max(bar.high for bar in previous)
        current_close = bars[-1].close
        threshold = rolling_high * (1 + self.config.breakout_buffer)
        if rolling_high <= 0 or current_close <= threshold:
            return False, None, rolling_high
        return True, current_close / rolling_high - 1, rolling_high

    def _confidence(
        self,
        ma_signal: bool,
        range_signal: bool,
        volume_confirmed: bool,
        liquidity_score: float,
        spread_rate: float,
    ) -> float:
        signal_component = (0.25 if ma_signal else 0.0) + (0.30 if range_signal else 0.0)
        volume_component = 0.20 if volume_confirmed else 0.0
        liquidity_component = max(0.0, min(1.0, liquidity_score)) * 0.15
        spread_component = max(0.0, 1 - spread_rate / self.config.max_spread_rate) * 0.10
        return max(0.0, min(1.0, signal_component + volume_component + liquidity_component + spread_component))

    def _ontology_tags(self, ma_signal: bool, range_signal: bool, volume_confirmed: bool) -> list[str]:
        tags = ["BreakoutWatch", "TechnicalBreakoutBuy"]
        if ma_signal:
            tags.append("MovingAverageBreakout")
        if range_signal:
            tags.append("TradingRangeBreakout")
        if volume_confirmed:
            tags.append("VolumeConfirmedBreakout")
        else:
            tags.append("FalseBreakoutRisk")
        return tags


def _ma_gap(closes: Sequence[float], fast: int, slow: int) -> float | None:
    fast_value = sma(list(closes), fast)
    slow_value = sma(list(closes), slow)
    if fast_value is None or slow_value in (None, 0):
        return None
    return max(0.0, fast_value / slow_value - 1)


def _volume_confirmed(volumes: Sequence[float], window: int, multiplier: float) -> bool:
    if len(volumes) <= window:
        return False
    baseline = volumes[-window - 1 : -1]
    average_volume = sum(baseline) / len(baseline)
    if average_volume <= 0:
        return False
    return volumes[-1] > average_volume * multiplier
