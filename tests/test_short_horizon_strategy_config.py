from __future__ import annotations

from datetime import datetime, timezone

from app.features.short_horizon_features import ShortHorizonFeatures
from app.trading_pipeline import (
    build_strategy_candidate_factory_from_config,
    generate_short_horizon_strategy_candidates,
    load_short_horizon_strategy_config,
)


def test_short_horizon_config_loads_conservative_execution_defaults() -> None:
    config = load_short_horizon_strategy_config()

    assert config["execution"]["live_trading_enabled"] is False
    assert config["execution"]["default_mode"] == "paper_trading"
    assert config["strategy_candidate_factory"]["paper_only"] is True
    assert config["reality_check"]["required_for_live"] is True


def test_pipeline_blocks_live_mode_when_config_disables_live_trading() -> None:
    result = generate_short_horizon_strategy_candidates(
        features_by_ticker={"005930": _features()},
        entry_prices={"005930": 10_000},
        mode="live_trading",
        config={
            "execution": {"live_trading_enabled": False, "default_mode": "paper_trading"},
            "strategy_candidate_factory": {"enabled": True, "paper_only": True},
        },
    )

    assert result.candidates == ()
    assert result.filtered_candidates == ()


def test_factory_builds_from_loaded_config() -> None:
    factory = build_strategy_candidate_factory_from_config(load_short_horizon_strategy_config())

    assert factory.config.paper_only is True
    assert factory.config.max_cost_to_alpha_ratio == 0.5
    assert factory.config.max_spread_rate == 0.0015


def _features() -> ShortHorizonFeatures:
    return ShortHorizonFeatures(
        ticker="005930",
        timestamp=datetime(2026, 1, 2, 9, 35, tzinfo=timezone.utc),
        returns_by_window={
            "ret_1m": 0.003,
            "ret_3m": 0.006,
            "ret_5m": 0.009,
            "ret_15m": 0.014,
            "ret_30m": 0.018,
            "ret_1d": 0.025,
            "ret_open_10m": 0.008,
            "ret_open_30m": 0.02,
            "ret_preclose_30m": None,
        },
        realized_volatility={
            "realized_volatility_5m": 0.002,
            "realized_volatility_30m": 0.004,
        },
        volume_zscore=2.5,
        spread_rate=0.0005,
        orderbook_depth_score=0.8,
        liquidity_score=0.85,
        market_alignment_score=0.9,
        time_of_day_weight=1.0,
        is_valid=True,
        missing_fields=(),
    )
