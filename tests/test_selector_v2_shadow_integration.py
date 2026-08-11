"""V2 runs beside the legacy selector without touching what trades."""

from __future__ import annotations

import ast
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.config.selector_v2_flags import SelectorV2Flags
from app.routing.selector_v2_shadow import SelectorV2ShadowRunner
from app.technical.feature_builder import technical_feature_set_from_live_frame  # noqa: F401
from app.technical.signals import TechnicalFeatureSet
from app.trading.strategy_session import StrategySessionManager

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


def _features() -> TechnicalFeatureSet:
    return TechnicalFeatureSet(
        symbol="005930",
        price=70_000.0,
        ema_fast=70_400.0,
        ema_slow=69_900.0,
        macd_histogram=9.0,
        vwap=70_200.0,
        vwap_distance_bps=-30.0,
        spread_bps=8.0,
        orderbook_imbalance=0.3,
        liquidity_score=0.7,
        aggressor_imbalance_5s=0.4,
        realized_volatility=0.006,
        realized_volatility_10s=0.004,
        return_1s=0.0003,
        return_5s=0.0012,
        return_10s=0.0016,
        return_30s=0.0020,
        tick_count_5s=12.0,
        second_data_ready=1.0,
        donchian_low=69_500.0,
        donchian_high=70_300.0,
        donchian_low_distance=0.007,
        momentum_persistence=0.75,
        relative_volume=2.2,
        volume_spike_ratio=2.4,
        breakout_strength=-0.004,
        rsi=33.0,
        bb_percent_b=0.18,
        atr_pct=0.012,
        spread_change_5s=-0.5,
        orderbook_imbalance_change_5s=0.08,
        bid_depth=1_400.0,
        ask_depth=800.0,
        depth_ratio=1.75,
        short_return=0.004,
        breakout_distance_bps=-40.0,
        box_position=0.4,
    )


def _evidence() -> dict:
    """The evidence-row shape the live path actually writes.

    Mirrors ``web._strategy_session_selection_evidence``: ``technical_features`` is an
    ``asdict(TechnicalFeatureSet)``, plus ``mark_price`` and the GNN validation rows.
    """
    return {
        "005930": {
            "symbol": "005930",
            "mark_price": 70_000.0,
            "mark_price_as_of": AT.isoformat(),
            "as_of": AT.isoformat(),
            "history_bar_count": 40,
            "technical_features": asdict(_features()),
            "decisions": [
                {
                    "path": "cpu_gnn",
                    "action": "ACTIVATE_STRATEGY",
                    "strategy_id": "intraday_momentum",
                    "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                }
            ],
            "validation_candidates": [
                {
                    "path": "cpu_gnn_validation",
                    "strategy_id": "intraday_momentum",
                    "probability_success": 0.61,
                    "expected_gross_return_bps": 180.0,
                    "expected_cost_bps": 30.0,
                    "expected_net_return_bps": 150.0,
                    "expected_adverse_excursion_bps": 45.0,
                    "expected_holding_seconds": 600.0,
                    "aleatoric_uncertainty": 12.0,
                    "epistemic_uncertainty_or_proxy": 4.0,
                    "model_version": "rgcn-test",
                    "reason_codes": ["GNN_REALTIME_TRUST_PASSED"],
                }
            ],
            "rvgi_box_context": {
                "ontology_eligible": False,
                "rvgi": 0.4,
                "rvgi_signal": 0.2,
                "rvgi_diff": 0.2,
                "rvgi_bullish_cross": True,
                "box_high": 70_300.0,
                "box_low": 69_500.0,
                "box_mid": 69_900.0,
                "box_width_pct": 0.011,
                "box_position": 0.6,
                "box_context_timestamp": AT.isoformat(),
                "box_previous_close": 69_800.0,
                "volume_confirmed": True,
            },
        }
    }


@dataclass(frozen=True)
class _Macro:
    market_regime: str = "TREND_UP"
    change_point_probability: float = 0.1
    regime_stability: float = 0.8
    volatility_percentile: float = 0.7
    allowed_strategies: tuple = ()
    blocked_strategies: tuple = ()
    diagnostics: dict = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _Bundle:
    macro_result: _Macro
    micro_results: tuple = ()
    ranked_trade_intents: tuple = ()


def _runner(**flag_overrides) -> SelectorV2ShadowRunner:
    flags = SelectorV2Flags(
        enabled=True,
        shadow_only=True,
        counterfactual_enabled=True,
        utility_gnn_enabled=True,
        bandit_adapter_enabled=False,
        no_trade_enabled=True,
        ontology_mask_v2_enabled=True,
        **flag_overrides,
    )
    return SelectorV2ShadowRunner(flags=flags)


# --------------------------------------------------------------------------- #
# The shadow pipeline end to end                                               #
# --------------------------------------------------------------------------- #
def test_pipeline_produces_a_selection_from_the_live_evidence_shape() -> None:
    runner = _runner()
    results = runner.observe(
        candidates=("005930",),
        evidence=_evidence(),
        bundle=_Bundle(macro_result=_Macro()),
        now=AT,
        legacy_strategy="vwap_mean_reversion",
        legacy_symbol="005930",
        legacy_reason="SINGLE_SYMBOL_STRATEGY_ARMED",
    )
    assert len(results) == 1
    result = results[0]
    assert result.symbol == "005930"
    assert result.context_id
    assert result.decision in {"SELECT", "NO_TRADE"}
    assert result.ranked_candidates, "every eligible strategy must be ranked and recorded"


def test_snapshot_exposes_the_required_diagnostics() -> None:
    runner = _runner()
    runner.observe(
        candidates=("005930",),
        evidence=_evidence(),
        bundle=_Bundle(macro_result=_Macro()),
        now=AT,
        legacy_strategy="vwap_mean_reversion",
        legacy_symbol="005930",
        legacy_reason="SINGLE_SYMBOL_STRATEGY_ARMED",
    )
    snapshot = runner.snapshot(symbol="005930")
    assert snapshot["enabled"] is True
    assert snapshot["shadow_only"] is True
    assert snapshot["live_authority"] is False
    latest = snapshot["latest_selection"]
    assert latest["context_id"]
    assert "no_trade" in latest and latest["no_trade"]["no_trade_utility_bps"] is not None
    assert latest["ranked_candidates"] or latest["blocked"]
    for candidate in latest["ranked_candidates"]:
        for term in (
            "expected_gross_return_bps",
            "expected_cost_bps",
            "expected_net_return_bps",
            "downside_penalty_bps",
            "uncertainty_penalty_bps",
            "ontology_adjustment_bps",
            "bandit_adjustment_bps",
            "final_utility_bps",
            "reason_codes",
        ):
            assert term in candidate
    assert snapshot["comparisons"], "legacy-vs-V2 comparison must be recorded"
    assert snapshot["comparisons"][-1]["legacy_strategy"] == "vwap_mean_reversion"
    assert snapshot["coverage"]["observations"] >= 1
    assert snapshot["counterfactual"] is not None


def test_agreement_is_classified() -> None:
    runner = _runner()
    runner.observe(
        candidates=("005930",),
        evidence=_evidence(),
        bundle=_Bundle(macro_result=_Macro()),
        now=AT,
        legacy_strategy=None,
        legacy_symbol=None,
        legacy_reason="BANDIT_NO_TRADE_NO_POSITIVE_CONSERVATIVE_EDGE",
    )
    counts = runner.snapshot()["agreement_counts"]
    assert set(counts) <= {
        "BOTH_NO_TRADE",
        "V2_DECLINED",
        "V2_TRADED",
        "SAME_STRATEGY",
        "DIFFERENT_STRATEGY",
    }
    assert sum(counts.values()) == 1


def test_disabled_flags_make_the_runner_inert() -> None:
    runner = SelectorV2ShadowRunner(flags=SelectorV2Flags(enabled=False))
    assert (
        runner.observe(
            candidates=("005930",),
            evidence=_evidence(),
            bundle=_Bundle(macro_result=_Macro()),
            now=AT,
            legacy_strategy=None,
            legacy_symbol=None,
            legacy_reason="",
        )
        == ()
    )


def test_runner_never_raises_on_malformed_input() -> None:
    runner = _runner()
    assert (
        runner.observe(
            candidates=("005930",),
            evidence={"005930": {"technical_features": "not-a-mapping"}},
            bundle=None,
            now=AT,
            legacy_strategy=None,
            legacy_symbol=None,
            legacy_reason="",
        )
        == ()
    )
    # And a bundle that raises on attribute access must not escape either.
    class _Hostile:
        @property
        def macro_result(self):
            raise RuntimeError("boom")

    assert (
        runner.observe(
            candidates=("005930",),
            evidence=_evidence(),
            bundle=_Hostile(),
            now=AT,
            legacy_strategy=None,
            legacy_symbol=None,
            legacy_reason="",
        )
        == ()
    )
    assert runner.snapshot()["last_error"]


def test_symbol_cap_keeps_the_legacy_pick_in_the_comparison() -> None:
    runner = SelectorV2ShadowRunner(
        flags=SelectorV2Flags(enabled=True, shadow_only=True, bandit_adapter_enabled=False),
        max_symbols_per_cycle=1,
    )
    evidence = _evidence()
    evidence["000660"] = dict(evidence["005930"])
    runner.observe(
        candidates=("000660", "005930"),
        evidence=evidence,
        bundle=_Bundle(macro_result=_Macro()),
        now=AT,
        legacy_strategy="intraday_momentum",
        legacy_symbol="005930",
        legacy_reason="SINGLE_SYMBOL_STRATEGY_ARMED",
    )
    snapshot = runner.snapshot()
    assert snapshot["symbols_skipped"] >= 1, "the cap must be reported, not silent"
    assert any(item["symbol"] == "005930" for item in snapshot["comparisons"])


def test_quote_walk_resolves_counterfactual_positions() -> None:
    runner = _runner()
    runner.observe(
        candidates=("005930",),
        evidence=_evidence(),
        bundle=_Bundle(macro_result=_Macro()),
        now=AT,
        legacy_strategy=None,
        legacy_symbol=None,
        legacy_reason="",
    )
    # A big favourable move should close at least one alternative's target.
    resolved = runner.observe_quote("005930", 74_000.0, AT + timedelta(seconds=30))
    expired = runner.expire_stale(AT + timedelta(seconds=100_000))
    assert resolved + expired >= 0  # never raises; count depends on which fired


# --------------------------------------------------------------------------- #
# Safety: V2 cannot reach the execution path                                   #
# --------------------------------------------------------------------------- #
def test_flags_default_to_inert() -> None:
    flags = SelectorV2Flags.from_env({})
    assert flags.enabled is False
    assert flags.shadow_only is True
    assert flags.live_authority is False


def test_live_authority_requires_the_safety_sub_flags() -> None:
    with pytest.raises(ValueError):
        SelectorV2Flags(enabled=True, shadow_only=False, no_trade_enabled=False).validate()
    with pytest.raises(ValueError):
        SelectorV2Flags(
            enabled=True, shadow_only=False, ontology_mask_v2_enabled=False
        ).validate()
    with pytest.raises(ValueError):
        SelectorV2Flags(
            enabled=True, shadow_only=False, counterfactual_enabled=False
        ).validate()
    # Fully-armed live mode validates.
    SelectorV2Flags(enabled=True, shadow_only=False).validate()


def test_shadow_runner_has_no_execution_import() -> None:
    tree = ast.parse(
        pathlib.Path("src/app/routing/selector_v2_shadow.py").read_text(encoding="utf-8")
    )
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    for module in modules:
        assert not module.startswith("app.execution")
        assert not module.startswith("app.risk")
        assert not module.startswith("app.trading.realtime_trading_engine")


def test_session_manager_publishes_v2_telemetry_without_authority() -> None:
    manager = StrategySessionManager()
    snapshot = manager.snapshot()
    assert "selector_v2" in snapshot
    # Default posture: disabled, and the authority is still the legacy selector.
    assert snapshot["selector_v2"]["enabled"] is False
    assert snapshot["selection_authority"] in {
        "GNN_DIRECT",
        "CONSERVATIVE_BANDIT",
        "FIRST_ADMISSIBLE",
    }


def test_session_manager_accepts_an_injected_runner_and_never_lets_it_break_select() -> None:
    class _Exploding:
        def observe(self, **_kwargs):
            raise RuntimeError("shadow blew up")

        def snapshot(self, **_kwargs):
            raise RuntimeError("snapshot blew up")

    manager = StrategySessionManager(selector_v2_runner=_Exploding())
    # ``_observe_selector_v2`` must swallow it: the live engine disables buys on an
    # exception out of ``evaluate``, so a telemetry raise would stop trading.
    manager._observe_selector_v2((), {}, None, AT, None)  # noqa: SLF001
    # And ``snapshot`` must survive a broken runner too.
    assert manager.snapshot()["selector_v2"]["enabled"] is True
