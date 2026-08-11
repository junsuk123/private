from __future__ import annotations

from app.technical import reason_codes as rc
from app.technical.regime import MarketRegime
from app.technical.signals import (
    BreakoutSignalProvider,
    CompositeTechnicalSignalEngine,
    MeanReversionSignalProvider,
    MomentumTrendSignalProvider,
    SignalDirection,
    TechnicalFeatureSet,
    VolatilityBandSignalProvider,
    VwapVolumeSignalProvider,
)


def trend_up_features(**over) -> TechnicalFeatureSet:
    base = dict(
        symbol="A",
        price=100.0,
        ema_fast=100.2,
        ema_slow=100.0,
        macd=0.5,
        macd_signal=0.1,
        macd_histogram=0.4,
        short_return=0.006,
        momentum_persistence=0.8,
        rsi=62,
        bb_percent_b=0.7,
        bb_bandwidth=0.03,
        vwap=99.8,
        vwap_distance_bps=15,
        vwap_slope=3.0,
        relative_volume=1.6,
        volume_spike_ratio=2.0,
        donchian_high=100.05,
        donchian_low=99.0,
        breakout_strength=0.0,
        false_breakout_risk=0.1,
        atr_pct=0.008,
        realized_volatility=0.004,
        volatility_expansion=0.9,
        liquidity_score=0.9,
        spread_bps=5.0,
        orderbook_imbalance=0.3,
        expected_slippage_bps=2.0,
    )
    base.update(over)
    return TechnicalFeatureSet(**base)


def tick_features(**over) -> TechnicalFeatureSet:
    """Trend-up bars plus a populated sub-second window.

    The owned-strategy path fires on ticks, so a bar-only fixture must fail
    closed. This helper is the fixture for the cases that should fire.
    """
    tick = dict(
        return_1s=0.0002,
        return_5s=0.0008,
        return_10s=0.0010,
        tick_count_1s=2.0,
        tick_count_5s=9.0,
        volume_1s_log=6.0,
        volume_5s_log=8.0,
        aggressor_imbalance_5s=0.35,
        realized_volatility_10s=0.0025,
        spread_change_5s=-0.4,
        orderbook_imbalance_change_5s=0.1,
        second_data_ready=1.0,
    )
    tick.update(over)
    return trend_up_features(**tick)


def _absorption_horizon_clears_cost(horizon_seconds: int, symbol: str) -> bool:
    """Is this holding clock inside its configured range AND able to pay costs?

    ``_cost_feasible_volatility_edge`` promises "the shortest completed-bar
    volatility horizon clearing costs", so the meaningful assertion is that
    promise, not whatever number today's volatility happens to produce.
    """
    from app.technical.strategy_algorithms import (
        get_algorithm,
        tick_expected_move_bps,
    )

    algorithm = get_algorithm("liquidity_shock_reversal")
    base = int(algorithm.p("absorption_horizon_seconds"))
    maximum = int(algorithm.p("max_absorption_horizon_seconds"))
    if not base <= int(horizon_seconds) <= maximum:
        return False
    floor, _ = algorithm.entry_floor_bps(symbol)
    capture = algorithm.p("absorption_capture_fraction")
    volatility = tick_features().realized_volatility

    def move(seconds: int) -> float:
        return tick_expected_move_bps(
            volatility, seconds, window_seconds=60, capture_fraction=capture
        )

    if move(int(horizon_seconds)) < floor:
        return False
    # Shortest: one second less must NOT clear the floor, unless the clock is
    # already pinned at the configured base.
    return int(horizon_seconds) == base or move(int(horizon_seconds) - 1) < floor


def _election(strategy_id: str):
    """Minimal election context that satisfies each context-dependent algorithm."""
    from app.technical.strategy_algorithms import ElectionContext

    extra = {
        "event_momentum": dict(
            event_fresh=True, event_age_seconds=120.0, event_ttl_seconds=1800.0
        ),
        "cross_sectional_relative_strength": dict(sector_rank=1, sector_candidate_count=8),
        "gap_context": dict(
            gap_rate=0.02, gap_submode="continuation", session_open_price=99.0
        ),
    }
    return ElectionContext(strategy_id, **extra.get(strategy_id, {}))


class TestProviderSchema:
    def test_all_providers_output_valid_schema(self):
        f = trend_up_features()
        for provider in (
            MomentumTrendSignalProvider(),
            BreakoutSignalProvider(),
            MeanReversionSignalProvider(),
            VwapVolumeSignalProvider(),
            VolatilityBandSignalProvider(),
        ):
            s = provider.evaluate(f)
            assert -1.0 <= s.score <= 1.0
            assert 0.0 <= s.confidence <= 1.0
            assert s.expected_edge_bps >= 0.0
            assert s.expected_horizon_seconds > 0
            assert isinstance(s.reason_codes, tuple)

    def test_unavailable_when_data_missing(self):
        empty = TechnicalFeatureSet(symbol="A")
        s = MomentumTrendSignalProvider().evaluate(empty)
        assert not s.available
        assert s.reason_codes == (rc.TECHNICAL_SIGNAL_UNAVAILABLE,)
        assert s.score == 0.0

    def test_edge_is_zero_without_volatility_proxy(self):
        f = trend_up_features(realized_volatility=None, atr_pct=None)
        s = MomentumTrendSignalProvider().evaluate(f)
        # No fabricated floor: no vol proxy -> no expected edge.
        assert s.expected_edge_bps == 0.0


class TestProviderDirection:
    def test_momentum_buy_in_uptrend(self):
        s = MomentumTrendSignalProvider().evaluate(trend_up_features())
        assert s.direction == SignalDirection.BUY
        assert s.score > 0
        assert rc.MOMENTUM_CONFIRMED in s.reason_codes

    def test_momentum_sell_in_downtrend(self):
        f = trend_up_features(ema_fast=99.8, macd=-0.5, macd_histogram=-0.4, short_return=-0.006, momentum_persistence=0.2)
        s = MomentumTrendSignalProvider().evaluate(f)
        assert s.direction == SignalDirection.SELL
        assert s.score < 0

    def test_breakout_confirmed(self):
        s = BreakoutSignalProvider().evaluate(trend_up_features(breakout_strength=0.001))
        assert s.direction == SignalDirection.BUY
        assert rc.BREAKOUT_CONFIRMED in s.reason_codes

    def test_breakout_missing_volume_flag(self):
        s = BreakoutSignalProvider().evaluate(trend_up_features(volume_spike_ratio=1.0, breakout_strength=0.001))
        assert rc.VOLUME_CONFIRMATION_MISSING in s.reason_codes

    def test_reversion_buy_when_oversold(self):
        f = trend_up_features(rsi=22, bb_percent_b=0.03)
        s = MeanReversionSignalProvider().evaluate(f)
        assert s.direction == SignalDirection.BUY
        assert rc.MEAN_REVERSION_CANDIDATE in s.reason_codes

    def test_vwap_breakdown_flag(self):
        s = VwapVolumeSignalProvider().evaluate(trend_up_features(vwap_distance_bps=-10))
        assert rc.VWAP_BREAKDOWN in s.reason_codes
        assert s.score < 0

    def test_volatility_band_reduce_on_expansion(self):
        f = trend_up_features(volatility_expansion=2.0, atr_pct=0.05, bb_bandwidth=0.12)
        s = VolatilityBandSignalProvider().evaluate(f)
        assert s.score < 0
        assert s.direction == SignalDirection.REDUCE


class TestCompositeEngine:
    def setup_method(self):
        self.engine = CompositeTechnicalSignalEngine()

    def test_buy_in_clean_uptrend(self):
        c = self.engine.evaluate(trend_up_features())
        assert c.direction == SignalDirection.BUY
        assert not c.blocks_buy
        assert c.expected_edge_bps >= 0.0
        assert c.selected_methodology
        assert rc.VWAP_CONFIRMATION_OK in c.reason_codes

    def test_blocked_in_low_liquidity(self):
        c = self.engine.evaluate(trend_up_features(liquidity_score=0.05))
        assert c.blocks_buy
        assert c.direction == SignalDirection.HOLD
        assert rc.LOW_LIQUIDITY_TECHNICAL_BLOCK in c.reason_codes

    def test_blocked_in_high_volatility(self):
        c = self.engine.evaluate(trend_up_features(realized_volatility=0.05))
        assert c.blocks_buy
        assert rc.HIGH_VOLATILITY_TECHNICAL_BLOCK in c.reason_codes

    def test_buy_requires_vwap_confirmation(self):
        # Strong momentum but price below VWAP -> no BUY (confirmation layer).
        c = self.engine.evaluate(trend_up_features(vwap_distance_bps=-12, vwap_slope=-3.0))
        assert c.direction == SignalDirection.HOLD
        assert not c.blocks_buy  # not a risk block, just unconfirmed
        assert rc.VWAP_BREAKDOWN in c.reason_codes

    def test_no_single_indicator_buys_alone(self):
        # Only RSI present (deep oversold), nothing else -> cannot form BUY.
        f = TechnicalFeatureSet(symbol="A", price=100.0, rsi=10, liquidity_score=0.9)
        c = self.engine.evaluate(f)
        assert c.direction != SignalDirection.BUY

    def test_reversion_blocked_in_downtrend(self):
        # Oversold but strong downtrend -> reversion contributor blocked.
        f = trend_up_features(
            ema_fast=99.8,
            ema_slow=100.0,
            macd=-0.5,
            macd_histogram=-0.5,
            short_return=-0.01,
            vwap_distance_bps=-40,
            rsi=22,
            bb_percent_b=0.03,
            volume_spike_ratio=1.0,
            breakout_strength=-0.02,
        )
        c = self.engine.evaluate(f)
        assert c.direction != SignalDirection.BUY
        assert rc.MEAN_REVERSION_BLOCKED_BY_DOWNTREND in c.reason_codes

    def test_composite_determinism(self):
        f = trend_up_features()
        a = self.engine.evaluate(f)
        b = self.engine.evaluate(f)
        assert a.direction == b.direction
        assert a.score == b.score
        assert a.confidence == b.confidence

    def test_exit_deterioration_codes(self):
        f = trend_up_features(vwap_distance_bps=-10, macd_histogram=-0.3, volatility_expansion=2.0)
        codes = self.engine.evaluate_exit_deterioration(f)
        assert rc.TECHNICAL_EXIT_DETERIORATION in codes
        assert rc.VWAP_BREAKDOWN in codes

    def test_no_exit_deterioration_when_healthy(self):
        codes = self.engine.evaluate_exit_deterioration(trend_up_features())
        assert codes == ()

    def test_owned_strategy_runs_its_own_algorithm(self):
        # KRX symbol on purpose: the entry floor is now the market's round-trip
        # cost, and this fixture's ~53bp edge clears KRX (~44bp) but not US
        # (~61bp). The subject here is which algorithm runs, so the market has to
        # be stated rather than inherited from a placeholder ticker.
        features = tick_features(symbol="005930", breakout_strength=0.0007)
        momentum = self.engine.evaluate_owned_strategy(features, "intraday_momentum")
        breakout = self.engine.evaluate_owned_strategy(features, "breakout_volume")
        assert momentum.selected_methodology == "intraday_momentum"
        assert breakout.selected_methodology == "breakout_volume"
        assert momentum.diagnostics["strategy_locked"] is True
        assert breakout.diagnostics["strategy_locked"] is True
        # Distinct algorithms, so distinct evidence — not one shared provider.
        assert momentum.diagnostics["algorithm"]["strategy_id"] == "intraday_momentum"
        assert breakout.diagnostics["algorithm"]["strategy_id"] == "breakout_volume"
        assert set(momentum.reason_codes) != set(breakout.reason_codes)

    def test_owned_strategy_fails_closed_without_tick_data(self):
        # Bar-only input can never fire a mechanical tick trigger.
        signal = self.engine.evaluate_owned_strategy(
            trend_up_features(),
            "intraday_momentum",
        )
        assert signal.direction == SignalDirection.HOLD
        assert "TICK_WINDOW_NOT_READY" in signal.reason_codes

    def test_owned_momentum_requires_bar_trend_agreement(self):
        signal = self.engine.evaluate_owned_strategy(
            tick_features(macd_histogram=-0.4),
            "intraday_momentum",
        )
        assert signal.direction == SignalDirection.HOLD
        assert "BAR_TREND_DISAGREES" in signal.reason_codes

    def test_owned_momentum_requires_buy_side_aggressor_flow(self):
        signal = self.engine.evaluate_owned_strategy(
            tick_features(aggressor_imbalance_5s=-0.4),
            "intraday_momentum",
        )
        assert signal.direction == SignalDirection.HOLD
        assert "AGGRESSOR_FLOW_NOT_BUY_SIDE" in signal.reason_codes

    def test_every_strategy_id_has_a_distinct_algorithm(self):
        """Regression: four ids used to resolve to one momentum provider."""
        from app.technical.strategy_algorithms import ALGORITHM_IDS, build_algorithm_registry

        registry = build_algorithm_registry()
        assert set(registry) == set(ALGORITHM_IDS)
        assert len({type(algorithm) for algorithm in registry.values()}) == len(ALGORITHM_IDS)
        # Exit geometry must also differ by thesis, not just by a bps constant.
        features = tick_features()
        bases = {
            algorithm.exit_rule(100.0, features, _election(strategy_id)).target_basis
            for strategy_id, algorithm in registry.items()
        }
        assert len(bases) >= 4

    def test_mean_reversion_does_not_fire_on_a_momentum_tape(self):
        signal = self.engine.evaluate_owned_strategy(tick_features(), "vwap_mean_reversion")
        assert signal.direction == SignalDirection.HOLD
        assert "VWAP_DISPLACEMENT_TOO_SMALL" in signal.reason_codes

    def test_vwap_reversion_requires_observed_recovery_not_missing_as_zero(self):
        base = dict(
            symbol="005930",
            vwap_distance_bps=-100.0,
            vwap=101.0,
            rsi=28.0,
            bb_percent_b=0.08,
            orderbook_imbalance_change_5s=0.1,
        )
        missing = self.engine.evaluate_owned_strategy(
            tick_features(**base, return_5s=None), "vwap_mean_reversion"
        )
        flat = self.engine.evaluate_owned_strategy(
            tick_features(**base, return_5s=0.0), "vwap_mean_reversion"
        )
        recovered = self.engine.evaluate_owned_strategy(
            tick_features(**base, return_5s=0.0008), "vwap_mean_reversion"
        )

        assert missing.direction == SignalDirection.HOLD
        assert "VWAP_RECOVERY_WINDOW_MISSING" in missing.reason_codes
        assert flat.direction == SignalDirection.HOLD
        assert "STILL_FALLING_ON_TICKS" in flat.reason_codes
        assert recovered.direction == SignalDirection.BUY
        assert "VWAP_DISPLACEMENT_STABILISED" in recovered.reason_codes

    def test_breakout_requires_positive_five_second_acceptance(self):
        base = dict(symbol="005930", breakout_strength=0.0007)
        missing = self.engine.evaluate_owned_strategy(
            tick_features(**base, return_5s=None), "breakout_volume"
        )
        flat = self.engine.evaluate_owned_strategy(
            tick_features(**base, return_5s=0.0), "breakout_volume"
        )
        accepted = self.engine.evaluate_owned_strategy(
            tick_features(**base, return_5s=0.0008), "breakout_volume"
        )

        assert "BREAKOUT_ACCEPTANCE_WINDOW_MISSING" in missing.reason_codes
        assert rc.FALSE_BREAKOUT_RISK_HIGH in flat.reason_codes
        assert accepted.direction == SignalDirection.BUY
        assert "BREAKOUT_ACCEPTED_ON_TICKS" in accepted.reason_codes

    def test_shock_reversal_needs_a_shock_and_a_contracting_spread(self):
        # 140bp shock: at the configured 40% recovery capture that is ~56bp of
        # expected edge, which is what it takes to clear a ~44bp KRX floor. The
        # old fixture used 60bp, worth ~24bp — a trade that cannot pay for its
        # own round trip (see the companion test below).
        shock = tick_features(
            symbol="005930",
            return_10s=-0.0140,
            spread_change_5s=-2.5,
            aggressor_imbalance_5s=-0.20,
            orderbook_imbalance=0.30,
        )
        fired = self.engine.evaluate_owned_strategy(shock, "liquidity_shock_reversal")
        assert fired.direction == SignalDirection.BUY
        assert "LIQUIDITY_SHOCK_STABILISED" in fired.reason_codes

        widening = self.engine.evaluate_owned_strategy(
            tick_features(
                symbol="005930",
                return_10s=-0.0140,
                spread_change_5s=3.0,
                aggressor_imbalance_5s=-0.20,
                orderbook_imbalance=0.30,
            ),
            "liquidity_shock_reversal",
        )
        assert widening.direction == SignalDirection.HOLD
        assert "SPREAD_STILL_WIDENING" in widening.reason_codes

    def test_shock_reversal_refuses_a_shock_too_small_to_pay_its_costs(self):
        # The configured minimum (40bp shock, 40% capture = 16bp) and this 60bp
        # shock (24bp) are both below the ~34bp KRX round trip, so the strategy's
        # own thresholds describe a structurally negative-expectancy trade. It
        # used to fire here, clear an 8bp floor, and die at the ProfitabilityGate.
        small = tick_features(
            symbol="005930",
            return_10s=-0.0060,
            spread_change_5s=-2.5,
            aggressor_imbalance_5s=-0.20,
            orderbook_imbalance=0.30,
        )

        signal = self.engine.evaluate_owned_strategy(small, "liquidity_shock_reversal")

        assert signal.direction == SignalDirection.HOLD
        assert "EDGE_BELOW_COST_FLOOR" in signal.reason_codes
        # The stabilisation itself was still detected; only the economics failed.
        assert "LIQUIDITY_SHOCK_STABILISED" in signal.reason_codes

    def test_absorption_horizon_is_the_shortest_that_clears_its_cost_floor(self):
        """The clock is solved from cost, and the BASE horizon is not a valid answer.

        This assertion used to be ``== 600``, the configured base. On a US name
        that base is economically infeasible: the fixture's completed-bar
        volatility carries only 44.3bps over 600s against a 61.2bps round-trip
        floor, so a position closed there could not pay for itself. Pinning the
        constant was therefore asserting the one end of the range the solver
        exists to avoid, and it silently went stale the moment the horizon became
        cost-derived.

        What is worth pinning is the contract in
        ``_cost_feasible_volatility_edge``: the SHORTEST horizon whose expected
        move clears the floor. A regression that reverts the clock to a constant,
        or that inverts the volatility scaling, breaks this; a change in the
        fixture's volatility does not.
        """
        from app.technical.strategy_algorithms import (
            get_algorithm,
            tick_expected_move_bps,
        )

        algorithm = get_algorithm("liquidity_shock_reversal")
        features = tick_features(
            symbol="SOFI",
            return_30s=0.0004,
            orderbook_imbalance=-0.40,
            spread_change_5s=-0.2,
        )
        base = int(algorithm.p("absorption_horizon_seconds"))
        maximum = int(algorithm.p("max_absorption_horizon_seconds"))
        capture = algorithm.p("absorption_capture_fraction")
        floor, floor_diagnostics = algorithm.entry_floor_bps("SOFI")
        edge, horizon = algorithm._cost_feasible_volatility_edge(
            features,
            base_horizon_seconds=base,
            maximum_horizon_seconds=maximum,
            capture_fraction=capture,
        )

        # The floor is the venue's round trip, not an arbitrary constant.
        assert floor_diagnostics["floor_basis"] == "round_trip_cost"
        assert base <= horizon <= maximum
        # It clears the floor...
        assert edge >= floor
        # ...and it is the shortest horizon that does.
        assert (
            tick_expected_move_bps(
                features.realized_volatility,
                horizon - 1,
                window_seconds=60,
                capture_fraction=capture,
            )
            < floor
        )
        # The base horizon really is infeasible here, so the solver extending past
        # it is the behaviour under test rather than an accident of the fixture.
        assert (
            tick_expected_move_bps(
                features.realized_volatility,
                base,
                window_seconds=60,
                capture_fraction=capture,
            )
            < floor
        )

    def test_a_calm_tape_cannot_be_carried_to_a_feasible_absorption_horizon(self):
        """When even the maximum horizon cannot pay, the clock stops at the cap.

        The solver clamps rather than inventing an unbounded holding time, and the
        cost floor then rejects the entry downstream — which is the correct pair of
        behaviours: no fabricated horizon, and no trade.
        """
        from app.technical.strategy_algorithms import get_algorithm

        algorithm = get_algorithm("liquidity_shock_reversal")
        calm = tick_features(
            symbol="SOFI",
            return_30s=0.0004,
            orderbook_imbalance=-0.40,
            spread_change_5s=-0.2,
            realized_volatility=0.00002,
        )
        maximum = int(algorithm.p("max_absorption_horizon_seconds"))
        floor, _ = algorithm.entry_floor_bps("SOFI")
        edge, horizon = algorithm._cost_feasible_volatility_edge(
            calm,
            base_horizon_seconds=int(algorithm.p("absorption_horizon_seconds")),
            maximum_horizon_seconds=maximum,
            capture_fraction=algorithm.p("absorption_capture_fraction"),
        )

        assert horizon == maximum
        assert edge < floor
        decision = algorithm.entry(calm, _election("liquidity_shock_reversal"))
        assert decision.triggered is False
        assert "EDGE_BELOW_COST_FLOOR" in decision.reason_codes

    def test_us_ask_heavy_absorption_branch_requires_price_recovery(self):
        absorbed = tick_features(
            symbol="SOFI",
            return_30s=0.0004,
            orderbook_imbalance=-0.40,
            spread_change_5s=-0.2,
        )

        fired = self.engine.evaluate_owned_strategy(
            absorbed,
            "liquidity_shock_reversal",
        )

        assert fired.direction == SignalDirection.BUY
        assert "ASK_HEAVY_ABSORPTION_CONFIRMED" in fired.reason_codes
        # The holding clock is DERIVED from cost, so it is asserted as a property
        # rather than as a constant -- see
        # test_absorption_horizon_is_the_shortest_that_clears_its_cost_floor.
        assert _absorption_horizon_clears_cost(fired.expected_horizon_seconds, "SOFI")

        rest_polled = self.engine.evaluate_owned_strategy(
            tick_features(
                symbol="SOFI",
                return_30s=0.0004,
                orderbook_imbalance=-0.40,
                spread_change_5s=-0.2,
                second_data_ready=False,
                tick_count_5s=0,
            ),
            "liquidity_shock_reversal",
        )
        assert rest_polled.direction == SignalDirection.BUY
        assert "ASK_HEAVY_ABSORPTION_CONFIRMED" in rest_polled.reason_codes

        still_falling = self.engine.evaluate_owned_strategy(
            tick_features(
                symbol="SOFI",
                return_30s=-0.0001,
                orderbook_imbalance=-0.40,
                spread_change_5s=-0.2,
            ),
            "liquidity_shock_reversal",
        )
        assert still_falling.direction == SignalDirection.HOLD
        assert "NO_LIQUIDITY_SHOCK_DETECTED" in still_falling.reason_codes

        flat_recovery = self.engine.evaluate_owned_strategy(
            tick_features(
                symbol="SOFI",
                return_30s=0.0001,
                orderbook_imbalance=-0.40,
                spread_change_5s=-0.2,
            ),
            "liquidity_shock_reversal",
        )
        assert flat_recovery.direction == SignalDirection.HOLD
        assert "ASK_HEAVY_ABSORPTION_CONFIRMED" not in flat_recovery.reason_codes

    def test_absorption_branch_is_not_extrapolated_to_krx(self):
        result = self.engine.evaluate_owned_strategy(
            tick_features(
                symbol="005930",
                return_30s=0.0004,
                orderbook_imbalance=-0.40,
                spread_change_5s=-0.2,
            ),
            "liquidity_shock_reversal",
        )

        assert result.direction == SignalDirection.HOLD
        assert "ASK_HEAVY_ABSORPTION_CONFIRMED" not in result.reason_codes

    def test_event_momentum_requires_event_context(self):
        from app.technical.strategy_algorithms import ElectionContext

        without = self.engine.evaluate_owned_strategy(tick_features(), "event_momentum")
        assert without.direction == SignalDirection.HOLD
        assert "EVENT_EVIDENCE_ABSENT" in without.reason_codes

        expired = self.engine.evaluate_owned_strategy(
            tick_features(),
            "event_momentum",
            ElectionContext(
                "event_momentum",
                event_fresh=True,
                event_age_seconds=4000.0,
                event_ttl_seconds=1800.0,
            ),
        )
        assert "EVENT_TTL_EXPIRED" in expired.reason_codes

    def test_gap_context_picks_exactly_one_submode(self):
        from app.technical.strategy_algorithms import ElectionContext

        # A continuation mandate can never buy a down gap.
        signal = self.engine.evaluate_owned_strategy(
            tick_features(),
            "gap_context",
            ElectionContext("gap_context", gap_rate=-0.02, gap_submode="continuation"),
        )
        assert "GAP_CONTINUATION_REQUIRES_UP_GAP" in signal.reason_codes

    def test_owned_strategy_no_longer_blocks_on_regime(self):
        """Regime/liquidity admissibility moved to the supervisor."""
        signal = self.engine.evaluate_owned_strategy(
            tick_features(liquidity_score=0.05, realized_volatility=0.05),
            "intraday_momentum",
        )
        assert rc.LOW_LIQUIDITY_TECHNICAL_BLOCK not in signal.reason_codes
        assert rc.HIGH_VOLATILITY_TECHNICAL_BLOCK not in signal.reason_codes
        assert "regime" in signal.diagnostics
