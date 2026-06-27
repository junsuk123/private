from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Mapping, Sequence

from app.cost import TradingCostEngine
from app.features.schemas import OHLCVBar
from app.features.short_horizon_features import ShortHorizonFeatures
from app.strategy.candidates import StrategyCandidate


@dataclass(frozen=True)
class PairRelativeValueConfig:
    enabled: bool = True
    paper_only: bool = True
    formation_window_days: int = 60
    trading_window_days: int = 20
    max_pair_distance: float = 0.15
    spread_z_entry: float = -2.0
    convergence_ratio: float = 0.4
    target_net_return: float = 0.004
    min_liquidity_score: float = 0.5
    max_spread_rate: float = 0.0015
    max_market_beta_difference: float = 0.35
    expected_holding_minutes: int = 60 * 24 * 5


@dataclass(frozen=True)
class PairAssetProfile:
    ticker: str
    sector: str | None = None
    theme: str | None = None
    market_beta: float | None = None


@dataclass(frozen=True)
class PairUniverseMember:
    ticker_a: str
    ticker_b: str
    pair_distance: float
    shared_sector: str | None = None
    shared_theme: str | None = None
    market_beta_difference: float | None = None


class PairUniverseBuilder:
    def __init__(self, config: PairRelativeValueConfig | None = None) -> None:
        self.config = config or PairRelativeValueConfig()

    def build(
        self,
        price_history_by_ticker: Mapping[str, Sequence[OHLCVBar]],
        *,
        profiles: Mapping[str, PairAssetProfile] | None = None,
        as_of: object | None = None,
    ) -> tuple[PairUniverseMember, ...]:
        del as_of
        profiles = profiles or {}
        tickers = sorted(price_history_by_ticker)
        members: list[PairUniverseMember] = []
        for index, ticker_a in enumerate(tickers):
            for ticker_b in tickers[index + 1 :]:
                profile_a = profiles.get(ticker_a, PairAssetProfile(ticker=ticker_a))
                profile_b = profiles.get(ticker_b, PairAssetProfile(ticker=ticker_b))
                if not _metadata_compatible(profile_a, profile_b, self.config):
                    continue
                path_a = _normalized_path(
                    price_history_by_ticker[ticker_a],
                    self.config.formation_window_days,
                    skip_recent=self.config.trading_window_days,
                )
                path_b = _normalized_path(
                    price_history_by_ticker[ticker_b],
                    self.config.formation_window_days,
                    skip_recent=self.config.trading_window_days,
                )
                distance = _pair_distance(path_a, path_b)
                if distance is None or distance >= self.config.max_pair_distance:
                    continue
                members.append(
                    PairUniverseMember(
                        ticker_a=ticker_a,
                        ticker_b=ticker_b,
                        pair_distance=distance,
                        shared_sector=profile_a.sector if profile_a.sector == profile_b.sector else None,
                        shared_theme=profile_a.theme if profile_a.theme == profile_b.theme else None,
                        market_beta_difference=_beta_difference(profile_a, profile_b),
                    )
                )
        return tuple(sorted(members, key=lambda pair: pair.pair_distance))


class PairRelativeValueEngine:
    """Creates long-only relative-value mean-reversion candidates."""

    strategy_family = "pair_relative_value"
    signal_name = "gatev_2006_long_only_mean_reversion"

    def __init__(
        self,
        config: PairRelativeValueConfig | None = None,
        cost_engine: TradingCostEngine | None = None,
    ) -> None:
        self.config = config or PairRelativeValueConfig()
        self.cost_engine = cost_engine or TradingCostEngine()

    def generate_candidate(
        self,
        pair: PairUniverseMember,
        price_history_by_ticker: Mapping[str, Sequence[OHLCVBar]],
        features_by_ticker: Mapping[str, ShortHorizonFeatures],
        *,
        trading_mode: str = "paper",
    ) -> StrategyCandidate | None:
        if not self.config.enabled:
            return None
        if self.config.paper_only and trading_mode != "paper":
            return None
        if pair.pair_distance >= self.config.max_pair_distance:
            return None

        spread_series = _spread_series(
            price_history_by_ticker.get(pair.ticker_a, ()),
            price_history_by_ticker.get(pair.ticker_b, ()),
            self.config.formation_window_days,
        )
        if len(spread_series) < max(3, min(self.config.trading_window_days, self.config.formation_window_days)):
            return None
        spread_mean = mean(spread_series)
        spread_std = pstdev(spread_series)
        if spread_std <= 0:
            return None
        spread = spread_series[-1]
        spread_z = (spread - spread_mean) / spread_std
        if spread_z > self.config.spread_z_entry:
            return None

        underperformer = pair.ticker_a
        peer = pair.ticker_b
        features = features_by_ticker.get(underperformer)
        if features is None or not features.is_valid:
            return None
        if features.spread_rate is None or features.spread_rate >= self.config.max_spread_rate:
            return None
        if features.liquidity_score is None or features.liquidity_score <= self.config.min_liquidity_score:
            return None

        entry_price = _last_close(price_history_by_ticker.get(underperformer, ()))
        if entry_price is None or entry_price <= 0:
            return None
        expected_convergence = abs(spread_z) * spread_std * self.config.convergence_ratio
        if expected_convergence <= 0:
            return None
        expected_exit_price = entry_price * (1 + expected_convergence)
        cost = self.cost_engine.estimate(
            symbol=underperformer,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            quantity=1,
            target_net_return=self.config.target_net_return,
        )
        if not cost.tradable or cost.net_expected_return < self.config.target_net_return:
            return None

        candidate_features = features.as_feature_dict()
        candidate_features.update(
            {
                "pair_distance": pair.pair_distance,
                "spread": spread,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "spread_z": spread_z,
                "expected_convergence": expected_convergence,
                "peer_ticker": 0.0,
                "target_net_return": self.config.target_net_return,
                "net_expected_return_after_cost": cost.net_expected_return,
                "cost_to_alpha_ratio": cost.cost_to_alpha_ratio,
            }
        )
        return StrategyCandidate(
            ticker=underperformer,
            strategy_family=self.strategy_family,
            signal_name=self.signal_name,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            expected_holding_minutes=self.config.expected_holding_minutes,
            gross_expected_return=cost.gross_expected_return,
            confidence=self._confidence(pair.pair_distance, spread_z, features.liquidity_score),
            features=candidate_features,
            ontology_tags=self._ontology_tags(pair),
            reason=(
                f"{underperformer} underperformed close substitute {peer}; long-only "
                "candidate assumes partial spread mean reversion after trading costs."
            ),
        )

    def _confidence(self, pair_distance: float, spread_z: float, liquidity_score: float) -> float:
        distance_component = max(0.0, 1 - pair_distance / self.config.max_pair_distance) * 0.30
        divergence_component = min(1.0, abs(spread_z) / 4.0) * 0.45
        liquidity_component = max(0.0, min(1.0, liquidity_score)) * 0.25
        return max(0.0, min(1.0, distance_component + divergence_component + liquidity_component))

    def _ontology_tags(self, pair: PairUniverseMember) -> list[str]:
        tags = [
            "CloseSubstitutePair",
            "PairSpreadDivergence",
            "MeanReversionCandidate",
            "RelativeUndervaluation",
        ]
        if pair.shared_sector:
            tags.append("CommonSectorFactor")
        if pair.shared_theme:
            tags.append("CommonThemeFactor")
        return tags


def _metadata_compatible(
    profile_a: PairAssetProfile,
    profile_b: PairAssetProfile,
    config: PairRelativeValueConfig,
) -> bool:
    same_sector = bool(profile_a.sector and profile_a.sector == profile_b.sector)
    same_theme = bool(profile_a.theme and profile_a.theme == profile_b.theme)
    beta_difference = _beta_difference(profile_a, profile_b)
    beta_ok = beta_difference is None or beta_difference <= config.max_market_beta_difference
    return (same_sector or same_theme) and beta_ok


def _beta_difference(profile_a: PairAssetProfile, profile_b: PairAssetProfile) -> float | None:
    if profile_a.market_beta is None or profile_b.market_beta is None:
        return None
    return abs(profile_a.market_beta - profile_b.market_beta)


def _normalized_path(
    bars: Sequence[OHLCVBar],
    window: int,
    *,
    skip_recent: int = 0,
) -> tuple[float, ...]:
    ordered = tuple(sorted(bars, key=lambda bar: bar.as_of))
    if skip_recent > 0 and len(ordered) > skip_recent:
        ordered = ordered[:-skip_recent]
    visible = ordered[-window:]
    if not visible or visible[0].close <= 0:
        return ()
    first = visible[0].close
    return tuple(bar.close / first for bar in visible if bar.close > 0)


def _pair_distance(path_a: Sequence[float], path_b: Sequence[float]) -> float | None:
    length = min(len(path_a), len(path_b))
    if length < 2:
        return None
    return sum((path_a[-length + i] - path_b[-length + i]) ** 2 for i in range(length))


def _spread_series(
    bars_a: Sequence[OHLCVBar],
    bars_b: Sequence[OHLCVBar],
    window: int,
) -> tuple[float, ...]:
    path_a = _normalized_path(bars_a, window)
    path_b = _normalized_path(bars_b, window)
    length = min(len(path_a), len(path_b))
    if length < 2:
        return ()
    return tuple(path_a[-length + i] - path_b[-length + i] for i in range(length))


def _last_close(bars: Sequence[OHLCVBar]) -> float | None:
    if not bars:
        return None
    ordered = sorted(bars, key=lambda bar: bar.as_of)
    close = ordered[-1].close
    return close if math.isfinite(close) and close > 0 else None
