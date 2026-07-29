from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.trading.strategy_supervisor import (
    HaltLevel,
    StrategySupervisor,
    SupervisorConfig,
    SupervisorObservation,
)

NOW = datetime(2026, 7, 29, 4, 30, tzinfo=timezone.utc)


def observation(**over) -> SupervisorObservation:
    base = dict(
        symbol="005930",
        as_of=NOW,
        strategy_id="intraday_momentum",
        position_open=True,
        data_age_seconds=1.0,
        session_tradable=True,
        broker_healthy=True,
        ontology_allows_strategy=True,
        macro_blocks_buy=False,
        macro_risk_level="LOW",
        liquidity_score=0.8,
        spread_bps=10.0,
        realized_volatility=0.003,
        daily_realized_loss=0.0,
        daily_loss_limit=1500.0,
    )
    base.update(over)
    return SupervisorObservation(**base)


@pytest.fixture()
def supervisor() -> StrategySupervisor:
    return StrategySupervisor(SupervisorConfig(enabled=True))


class TestHealthyTape:
    def test_clean_observation_does_not_halt(self, supervisor):
        verdict = supervisor.evaluate(observation())
        assert verdict.level is HaltLevel.NONE
        assert not verdict.blocks_new_entries
        assert not verdict.forces_exit
        assert verdict.reason_codes == ()


class TestHardViolations:
    def test_stale_data_forces_exit(self, supervisor):
        verdict = supervisor.evaluate(observation(data_age_seconds=120.0))
        assert verdict.level is HaltLevel.HARD
        assert verdict.forces_exit
        assert any(code.startswith("DATA_STALE") for code in verdict.hard_reason_codes)

    def test_stale_data_is_ignored_while_the_session_is_closed(self, supervisor):
        """Overnight staleness must not queue a market-open liquidation."""
        verdict = supervisor.evaluate(
            observation(session_tradable=False, data_age_seconds=50_000.0)
        )
        assert verdict.level is HaltLevel.SOFT
        assert not verdict.forces_exit
        assert "SESSION_NOT_TRADABLE" in verdict.soft_reason_codes
        assert not any(code.startswith("DATA_STALE") for code in verdict.reason_codes)

    def test_unhealthy_broker_forces_exit(self, supervisor):
        verdict = supervisor.evaluate(observation(broker_healthy=False))
        assert verdict.level is HaltLevel.HARD
        assert "BROKER_UNHEALTHY" in verdict.hard_reason_codes

    def test_daily_loss_limit_breach_forces_exit(self, supervisor):
        verdict = supervisor.evaluate(
            observation(daily_realized_loss=-1600.0, daily_loss_limit=1500.0)
        )
        assert verdict.level is HaltLevel.HARD
        assert "DAILY_LOSS_LIMIT_BREACHED" in verdict.hard_reason_codes


class TestSoftViolations:
    def test_ontology_withdrawing_the_strategy_is_soft_not_a_liquidation(self, supervisor):
        """A regime rotation must not dump an open position."""
        verdict = supervisor.evaluate(observation(ontology_allows_strategy=False))
        assert verdict.level is HaltLevel.SOFT
        assert verdict.blocks_new_entries
        assert not verdict.forces_exit
        assert "ONTOLOGY_WITHDREW_STRATEGY:intraday_momentum" in verdict.soft_reason_codes

    def test_macro_block_buy_only_stops_new_entries(self, supervisor):
        verdict = supervisor.evaluate(observation(macro_blocks_buy=True))
        assert verdict.level is HaltLevel.SOFT
        assert verdict.blocks_new_entries
        assert not verdict.forces_exit
        assert "MACRO_BLOCK_BUY" in verdict.soft_reason_codes

    def test_liquidity_and_spread_degradation_are_soft(self, supervisor):
        verdict = supervisor.evaluate(observation(liquidity_score=0.05, spread_bps=200.0))
        assert verdict.level is HaltLevel.SOFT
        assert not verdict.forces_exit
        assert any(code.startswith("LIQUIDITY_DEGRADED") for code in verdict.soft_reason_codes)
        assert any(code.startswith("SPREAD_WIDENED") for code in verdict.soft_reason_codes)

    def test_elevated_volatility_is_soft(self, supervisor):
        verdict = supervisor.evaluate(observation(realized_volatility=0.5))
        assert verdict.level is HaltLevel.SOFT
        assert any(code.startswith("VOLATILITY_ELEVATED") for code in verdict.soft_reason_codes)

    def test_daily_loss_near_breach_is_soft(self, supervisor):
        verdict = supervisor.evaluate(
            observation(daily_realized_loss=-1300.0, daily_loss_limit=1500.0)
        )
        assert verdict.level is HaltLevel.SOFT
        assert any(
            code.startswith("DAILY_LOSS_BUDGET_NEAR_BREACH") for code in verdict.soft_reason_codes
        )


class TestGrading:
    def test_hard_wins_over_soft(self, supervisor):
        verdict = supervisor.evaluate(
            observation(broker_healthy=False, macro_blocks_buy=True, liquidity_score=0.01)
        )
        assert verdict.level is HaltLevel.HARD
        assert verdict.forces_exit
        # Soft findings are still reported, just not the deciding tier.
        assert "MACRO_BLOCK_BUY" in verdict.soft_reason_codes

    def test_unobserved_fields_do_not_halt(self, supervisor):
        verdict = supervisor.evaluate(
            observation(
                data_age_seconds=None,
                session_tradable=None,
                broker_healthy=None,
                ontology_allows_strategy=None,
                liquidity_score=None,
                spread_bps=None,
                realized_volatility=None,
                daily_realized_loss=None,
                daily_loss_limit=None,
            )
        )
        assert verdict.level is HaltLevel.NONE

    def test_disabled_supervisor_never_halts(self):
        supervisor = StrategySupervisor(SupervisorConfig(enabled=False))
        verdict = supervisor.evaluate(observation(session_tradable=False))
        assert verdict.level is HaltLevel.NONE
        assert verdict.reason_codes == ("SUPERVISOR_DISABLED",)

    def test_last_verdict_is_retained_per_symbol(self, supervisor):
        supervisor.evaluate(observation(macro_blocks_buy=True))
        latest = supervisor.last_verdict("005930")
        assert latest is not None and latest.level is HaltLevel.SOFT
        assert "005930" in supervisor.snapshot()["last_verdicts"]


class TestMacroPermissionTranslation:
    """The macro layer uses a coarser strategy vocabulary than the algorithms.

    Regression: comparing the fine id directly against the macro allow-list made
    every elected strategy look withdrawn, HARD-halting the engine every cycle.
    """

    def test_fine_id_matches_its_macro_family(self):
        from app.technical.strategy_algorithms import macro_strategy_permitted

        assert macro_strategy_permitted("intraday_momentum", ("momentum",), ()) is True
        assert macro_strategy_permitted("breakout_volume", ("breakout",), ()) is True
        assert macro_strategy_permitted("vwap_mean_reversion", ("vwap_reversion",), ()) is True
        assert macro_strategy_permitted("liquidity_shock_reversal", ("mean_reversion",), ()) is True

    def test_blocked_family_blocks_the_fine_id(self):
        from app.technical.strategy_algorithms import macro_strategy_permitted

        assert macro_strategy_permitted("event_momentum", (), ("momentum",)) is False
        assert macro_strategy_permitted("intraday_momentum", ("momentum",), ("momentum",)) is False

    def test_absent_lists_are_unanswerable_not_a_withdrawal(self):
        from app.technical.strategy_algorithms import macro_strategy_permitted

        assert macro_strategy_permitted("intraday_momentum", (), ()) is None
        assert macro_strategy_permitted("", ("momentum",), ()) is None

    def test_unknown_strategy_is_not_silently_allowed(self):
        from app.technical.strategy_algorithms import macro_strategy_permitted

        assert macro_strategy_permitted("made_up_strategy", ("momentum",), ()) is False

    def test_supervisor_does_not_halt_on_a_permitted_family(self, supervisor):
        from app.technical.strategy_algorithms import macro_strategy_permitted

        verdict = supervisor.evaluate(
            observation(
                strategy_id="intraday_momentum",
                ontology_allows_strategy=macro_strategy_permitted(
                    "intraday_momentum", ("momentum", "breakout"), ()
                ),
            )
        )
        assert verdict.level is HaltLevel.NONE

    def test_retained_verdict_smoke(self, supervisor):
        supervisor.evaluate(observation(macro_blocks_buy=True))
        latest = supervisor.last_verdict("005930")
        assert latest is not None and latest.level is HaltLevel.SOFT
        assert "005930" in supervisor.snapshot()["last_verdicts"]
