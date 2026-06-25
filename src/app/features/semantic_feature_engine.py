from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from app.features.schemas import RawIndicatorRecord, SemanticFeatureRecord


@dataclass(frozen=True)
class SemanticMappingConfig:
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    stochastic_overbought: float = 80.0
    stochastic_oversold: float = 20.0
    volume_spike_ratio: float = 2.0
    high_volatility_annualized: float = 0.45
    low_volatility_annualized: float = 0.15
    severe_drawdown: float = -0.20
    short_momentum_threshold: float = 0.03
    medium_momentum_threshold: float = 0.08
    bollinger_squeeze_width: float = 0.08
    bullish_candle_body: float = 0.60
    bearish_candle_body: float = 0.60
    close_near_high: float = 0.70
    close_near_low: float = -0.70


class SemanticFeatureEngine:
    def __init__(self, config: SemanticMappingConfig | None = None) -> None:
        self.config = config or SemanticMappingConfig()

    def generate(self, indicators: tuple[RawIndicatorRecord, ...]) -> tuple[SemanticFeatureRecord, ...]:
        if not indicators:
            return ()
        by_name = {record.indicator_name: record for record in indicators}
        ticker = indicators[-1].ticker
        as_of = indicators[-1].as_of
        features: list[SemanticFeatureRecord] = []

        self._add_return_features(features, ticker, as_of, by_name)
        self._add_trend_features(features, ticker, as_of, by_name)
        self._add_momentum_features(features, ticker, as_of, by_name)
        self._add_volatility_features(features, ticker, as_of, by_name)
        self._add_volume_features(features, ticker, as_of, by_name)
        self._add_candle_features(features, ticker, as_of, by_name)
        return tuple(features)

    def _add_return_features(
        self, features: list[SemanticFeatureRecord], ticker: str, as_of: datetime, by_name: dict[str, RawIndicatorRecord]
    ) -> None:
        r1 = _number(by_name.get("return_1d"))
        r5 = _number(by_name.get("return_5d"))
        r20 = _number(by_name.get("return_20d"))
        drawdown = _number(by_name.get("rolling_drawdown_20d"))
        if r1 is not None and r1 > 0:
            features.append(self._feature(ticker, as_of, "DailyReturnPositive", "price_return", min(1, 0.5 + r1 * 10), ("return_1d",), "supportsSignal", "BuyCandidate"))
        if r5 is not None and r5 >= self.config.short_momentum_threshold:
            features.append(self._feature(ticker, as_of, "ShortTermMomentumPositive", "price_return", min(1, r5 / 0.10), ("return_5d",), "supportsSignal", "BuyCandidate"))
        if r20 is not None and r20 >= self.config.medium_momentum_threshold:
            features.append(self._feature(ticker, as_of, "MediumTermTrendPositive", "price_return", min(1, r20 / 0.20), ("return_20d",), "supportsSignal", "HoldWithTrailingStop"))
        if drawdown is not None and drawdown <= self.config.severe_drawdown:
            features.append(self._feature(ticker, as_of, "DrawdownSevere", "volatility_risk", min(1, abs(drawdown) / 0.35), ("rolling_drawdown_20d",), "increasesRiskOf", "ReduceRiskCandidate"))

    def _add_trend_features(
        self, features: list[SemanticFeatureRecord], ticker: str, as_of: datetime, by_name: dict[str, RawIndicatorRecord]
    ) -> None:
        price = _last_price_proxy(by_name)
        sma20 = _number(by_name.get("sma_20"))
        sma60 = _number(by_name.get("sma_60"))
        macd_line = _number(by_name.get("macd_line"))
        macd_signal = _number(by_name.get("macd_signal"))
        macd_hist = _number(by_name.get("macd_histogram"))
        if price is not None and sma20 is not None and price > sma20:
            features.append(self._feature(ticker, as_of, "PriceAboveMA20", "trend", min(1, price / sma20 - 1 + 0.5), ("sma_20",), "supportsSignal", "BuyCandidate"))
        if sma20 is not None and sma60 is not None and sma20 > sma60:
            features.append(self._feature(ticker, as_of, "MA20AboveMA60", "trend", min(1, sma20 / sma60 - 1 + 0.55), ("sma_20", "sma_60"), "supportsSignal", "BuyCandidate"))
        if macd_line is not None and macd_signal is not None and macd_line > macd_signal:
            features.append(self._feature(ticker, as_of, "MACDBullishCross", "trend", min(1, abs(macd_line - macd_signal) + 0.5), ("macd_line", "macd_signal"), "supportsSignal", "BuyCandidate"))
        if macd_line is not None and macd_signal is not None and macd_line < macd_signal:
            features.append(self._feature(ticker, as_of, "MACDBearishCross", "trend", min(1, abs(macd_line - macd_signal) + 0.5), ("macd_line", "macd_signal"), "contradictsSignal", "AggressiveBuy"))
        if macd_hist is not None and macd_hist > 0:
            features.append(self._feature(ticker, as_of, "MomentumIncreasing", "trend", min(1, abs(macd_hist) + 0.45), ("macd_histogram",), "supportsSignal", "HoldWithTrailingStop"))

    def _add_momentum_features(
        self, features: list[SemanticFeatureRecord], ticker: str, as_of: datetime, by_name: dict[str, RawIndicatorRecord]
    ) -> None:
        rsi = _number(by_name.get("rsi_14"))
        adx = _number(by_name.get("adx_14"))
        stoch_k = _number(by_name.get("stochastic_k_14"))
        mfi = _number(by_name.get("mfi_14"))
        if rsi is not None and rsi >= self.config.rsi_overbought:
            state = "TrendOverbought" if adx is not None and adx > 30 else "RangeOverbought"
            signal = "HoldWithTrailingStop" if state == "TrendOverbought" else "WaitOrTakeProfit"
            features.append(self._feature(ticker, as_of, state, "momentum", min(1, (rsi - 50) / 50), ("rsi_14",), "supportsSignal", signal))
        if rsi is not None and rsi <= self.config.rsi_oversold:
            features.append(self._feature(ticker, as_of, "RSIOversold", "momentum", min(1, (50 - rsi) / 50), ("rsi_14",), "supportsSignal", "Watchlist"))
        if stoch_k is not None and stoch_k >= self.config.stochastic_overbought:
            features.append(self._feature(ticker, as_of, "RangeOverbought", "momentum", min(1, stoch_k / 100), ("stochastic_k_14",), "supportsSignal", "WaitOrTakeProfit"))
        if stoch_k is not None and stoch_k <= self.config.stochastic_oversold:
            features.append(self._feature(ticker, as_of, "RangeOversold", "momentum", min(1, (100 - stoch_k) / 100), ("stochastic_k_14",), "supportsSignal", "Watchlist"))
        if mfi is not None and mfi > 80:
            features.append(self._feature(ticker, as_of, "MoneyFlowBuyingPressure", "momentum", min(1, mfi / 100), ("mfi_14",), "supportsSignal", "BuyCandidate"))
        if mfi is not None and mfi < 20:
            features.append(self._feature(ticker, as_of, "MoneyFlowSellingPressure", "momentum", min(1, (100 - mfi) / 100), ("mfi_14",), "increasesRiskOf", "ReduceRiskCandidate"))

    def _add_volatility_features(
        self, features: list[SemanticFeatureRecord], ticker: str, as_of: datetime, by_name: dict[str, RawIndicatorRecord]
    ) -> None:
        volatility = _number(by_name.get("historical_volatility_20d"))
        width = _number(by_name.get("bollinger_band_width_20"))
        percent_b = _number(by_name.get("bollinger_percent_b_20"))
        if volatility is not None and volatility >= self.config.high_volatility_annualized:
            features.append(self._feature(ticker, as_of, "VolatilityHigh", "volatility_risk", min(1, volatility / 0.80), ("historical_volatility_20d",), "increasesRiskOf", "RiskAdjustedSizing"))
        if volatility is not None and volatility <= self.config.low_volatility_annualized:
            features.append(self._feature(ticker, as_of, "VolatilityLow", "volatility", min(1, (self.config.low_volatility_annualized - volatility) / self.config.low_volatility_annualized), ("historical_volatility_20d",), "supportsSignal", "BreakoutWatch"))
        if width is not None and width <= self.config.bollinger_squeeze_width:
            features.append(self._feature(ticker, as_of, "BollingerSqueeze", "volatility", min(1, (self.config.bollinger_squeeze_width - width) / self.config.bollinger_squeeze_width), ("bollinger_band_width_20",), "supportsSignal", "BreakoutWatch"))
        if percent_b is not None and percent_b >= 1:
            features.append(self._feature(ticker, as_of, "UpperBandTouch", "volatility", min(1, percent_b), ("bollinger_percent_b_20",), "supportsSignal", "WaitOrTakeProfit"))
        if percent_b is not None and percent_b <= 0:
            features.append(self._feature(ticker, as_of, "LowerBandTouch", "volatility", min(1, abs(percent_b) + 0.5), ("bollinger_percent_b_20",), "supportsSignal", "Watchlist"))

    def _add_volume_features(
        self, features: list[SemanticFeatureRecord], ticker: str, as_of: datetime, by_name: dict[str, RawIndicatorRecord]
    ) -> None:
        ratio = _number(by_name.get("volume_spike_ratio"))
        r1 = _number(by_name.get("return_1d"))
        if ratio is not None and ratio >= self.config.volume_spike_ratio:
            features.append(self._feature(ticker, as_of, "VolumeSpike", "volume_flow", min(1, ratio / 4), ("volume_spike_ratio",), "supportsSignal", "BreakoutWatch"))
            if r1 is not None and r1 > 0:
                features.append(self._feature(ticker, as_of, "VolumeBackedRise", "volume_flow", min(1, ratio / 4), ("volume_spike_ratio", "return_1d"), "supportsSignal", "BuyCandidate"))
            if r1 is not None and r1 < 0:
                features.append(self._feature(ticker, as_of, "VolumeBackedFall", "volume_flow", min(1, ratio / 4), ("volume_spike_ratio", "return_1d"), "increasesRiskOf", "SellCandidate"))

    def _add_candle_features(
        self, features: list[SemanticFeatureRecord], ticker: str, as_of: datetime, by_name: dict[str, RawIndicatorRecord]
    ) -> None:
        body = _number(by_name.get("candle_body_ratio"))
        clv = _number(by_name.get("close_location_value"))
        if clv is not None and clv >= self.config.close_near_high:
            features.append(self._feature(ticker, as_of, "CloseNearHigh", "price_action", min(1, (clv + 1) / 2), ("close_location_value",), "supportsSignal", "BuyCandidate"))
        if clv is not None and clv <= self.config.close_near_low:
            features.append(self._feature(ticker, as_of, "CloseNearLow", "price_action", min(1, abs(clv)), ("close_location_value",), "increasesRiskOf", "SellCandidate"))
        if body is not None and body >= self.config.bullish_candle_body and clv is not None and clv > 0:
            features.append(self._feature(ticker, as_of, "BullishCandle", "price_action", min(1, body), ("candle_body_ratio", "close_location_value"), "supportsSignal", "BuyCandidate"))
        if body is not None and body >= self.config.bearish_candle_body and clv is not None and clv < 0:
            features.append(self._feature(ticker, as_of, "BearishCandle", "price_action", min(1, body), ("candle_body_ratio", "close_location_value"), "increasesRiskOf", "SellCandidate"))

    def _feature(
        self,
        ticker: str,
        as_of: datetime,
        name: str,
        category: str,
        confidence: float,
        indicators: tuple[str, ...],
        relation: str,
        target_signal: str | None,
    ) -> SemanticFeatureRecord:
        node_id = hashlib.sha256(f"{ticker}:{as_of.isoformat()}:{name}".encode("utf-8")).hexdigest()[:16]
        return SemanticFeatureRecord(
            ticker=ticker,
            as_of=as_of,
            feature_name=name,
            feature_category=category,
            state="active",
            confidence=max(0.0, min(1.0, confidence)),
            supporting_indicators=indicators,
            semantic_relation=relation,  # type: ignore[arg-type]
            target_signal=target_signal,
            ontology_node_id=f"semantic:{node_id}",
        )


def _number(record: RawIndicatorRecord | None) -> float | None:
    if record is None or record.value is None or isinstance(record.value, str):
        return None
    return float(record.value)


def _last_price_proxy(by_name: dict[str, RawIndicatorRecord]) -> float | None:
    sma20 = _number(by_name.get("sma_20"))
    r1 = _number(by_name.get("return_1d"))
    if sma20 is None:
        return None
    return sma20 * (1 + (r1 or 0))
