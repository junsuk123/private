from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.config.selector_v2_flags import SelectorV2Flags
from app.routing.selector_v2_promotion import (
    SelectorAuthorityState,
    SelectorPromotionConfig,
    SelectorPromotionController,
)
from app.routing.selector_v2_shadow import SelectorV2ShadowRunner


AT = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _groups(*, live: bool = False, negative: bool = False):
    groups = []
    for index in range(4):
        selected_net = -20.0 if negative else 20.0 + index
        selected = SimpleNamespace(
            strategy_id="winner",
            net_return_bps=selected_net,
            gross_return_bps=selected_net + 5.0,
            quotes_observed=3,
            evidence_source="shadow",
            regime="TREND",
        )
        alternative = SimpleNamespace(
            strategy_id="alternative",
            net_return_bps=-5.0,
            gross_return_bps=0.0,
            quotes_observed=3,
            evidence_source="shadow",
            regime="TREND",
        )
        groups.append(
            SimpleNamespace(
                context_id=f"ctx-{index}",
                symbol="TEST",
                market="US",
                opened_at=AT + timedelta(days=index // 2),
                selected_strategy="winner",
                decision="SELECT",
                outcomes={"winner": selected, "alternative": alternative},
                live_outcome_net_bps=selected_net if live else None,
                live_outcome_source="live_probe" if live else None,
                predicted_utility_bps={"winner": 10.0, "alternative": 1.0},
            )
        )
    return groups


def _config(tmp_path):
    return SelectorPromotionConfig(
        minimum_contexts=4,
        minimum_traded_contexts=4,
        minimum_distinct_days=2,
        minimum_chronological_windows=2,
        minimum_positive_windows=2,
        minimum_lower_bound_net_bps=0.0,
        additional_cost_stress_bps=1.0,
        maximum_mean_regret_bps=5.0,
        maximum_wrong_regime_trade_rate=0.2,
        minimum_top1_hit_rate=0.5,
        required_consecutive_passes=2,
        minimum_live_probe_contexts=4,
        minimum_live_probe_days=2,
        state_path=str(tmp_path / "selector-promotion.json"),
    )


def test_automatic_ladder_persists_and_requires_real_probe_outcomes(tmp_path):
    config = _config(tmp_path)
    controller = SelectorPromotionController(config)

    first = controller.evaluate(_groups(), now=AT + timedelta(days=2))
    second = controller.evaluate(_groups(), now=AT + timedelta(days=2, minutes=5))
    assert first.changed is False
    assert second.to_state is SelectorAuthorityState.LIVE_PROBE
    assert controller.order_size_fraction == 0.10

    restored = SelectorPromotionController(config)
    assert restored.state is SelectorAuthorityState.LIVE_PROBE
    blocked = restored.evaluate(_groups(), now=AT + timedelta(days=3))
    assert "SELECTOR_LIVE_SAMPLE_BELOW_4" in blocked.reason_codes

    restored.evaluate(_groups(live=True), now=AT + timedelta(days=3, minutes=5))
    promoted = restored.evaluate(_groups(live=True), now=AT + timedelta(days=3, minutes=10))
    assert promoted.to_state is SelectorAuthorityState.LIVE
    assert restored.order_size_fraction == 1.0


def test_live_authority_demotes_on_negative_conservative_edge(tmp_path):
    config = _config(tmp_path)
    controller = SelectorPromotionController(config)
    controller.evaluate(_groups(), now=AT)
    controller.evaluate(_groups(), now=AT + timedelta(minutes=5))
    assert controller.state is SelectorAuthorityState.LIVE_PROBE

    # The production demotion floor is 20 contexts. Lower only that test input by
    # providing twenty independently keyed negative contexts.
    bad = []
    for batch in range(5):
        for group in _groups(negative=True):
            group.context_id = f"bad-{batch}-{group.context_id}"
            bad.append(group)
    decision = controller.evaluate(bad, now=AT + timedelta(days=5))
    assert decision.to_state is SelectorAuthorityState.SHADOW
    assert "SELECTOR_AUTO_DEMOTION_NEGATIVE_EDGE" in decision.reason_codes


def test_auto_promote_requires_all_live_safety_components():
    flags = SelectorV2Flags(
        enabled=True,
        shadow_only=True,
        auto_promote=True,
        counterfactual_enabled=False,
    )
    try:
        flags.validate()
    except ValueError as exc:
        assert "STRATEGY_COUNTERFACTUAL_ENABLED" in str(exc)
    else:  # pragma: no cover - explicit assertion message is clearer than pytest.raises here.
        raise AssertionError("automatic promotion without counterfactual evidence must fail")


def test_single_resolved_context_is_insufficient_not_an_evaluator_fault(tmp_path):
    controller = SelectorPromotionController(_config(tmp_path))
    decision = controller.evaluate(_groups()[:1], now=AT)
    assert decision.to_state is SelectorAuthorityState.SHADOW
    assert "SELECTOR_NET_LOWER_BOUND_NOT_POSITIVE" in decision.reason_codes
    assert controller.snapshot()["evidence_rows"] == 1


def _declined_groups(count: int, *, start: datetime):
    """Contexts the selector passed on, where trading would have lost money anyway.

    ``selected_strategy=None`` is the NO_TRADE decision, and its realised outcome is
    0.0 by definition. Every alternative loses, so declining was correct: these carry no
    regret and must not be able to argue against the selector's measured edge.
    """
    groups = []
    for index in range(count):
        loser = SimpleNamespace(
            strategy_id="alternative",
            net_return_bps=-8.0,
            gross_return_bps=-3.0,
            quotes_observed=3,
            evidence_source="shadow",
            regime="RANGE",
        )
        groups.append(
            SimpleNamespace(
                context_id=f"declined-{index}",
                symbol="TEST",
                market="KR",
                opened_at=start + timedelta(minutes=index),
                selected_strategy=None,
                decision="NO_TRADE",
                outcomes={"alternative": loser},
                live_outcome_net_bps=None,
                live_outcome_source=None,
                predicted_utility_bps={"alternative": -4.0},
            )
        )
    return groups


def test_declines_do_not_dilute_the_measured_edge(tmp_path):
    """A selective selector must still be able to earn LIVE_PROBE.

    Regression for a deadlock: the edge bound was averaged over every context, and a
    declined context scores exactly 0.0. Those zeros pulled the 95% lower bound toward
    zero, so the more contexts the selector correctly passed on, the further it fell
    below ``minimum_lower_bound_net_bps`` — the rung got harder precisely because the
    selector was doing its job. Live evidence showed 3,531 declines against 0 trades.
    """
    config = _config(tmp_path)
    controller = SelectorPromotionController(config)
    evidence = [*_groups(), *_declined_groups(400, start=AT)]

    metrics = controller.evaluate(evidence, now=AT + timedelta(days=2)).metrics
    assert metrics["traded_context_count"] == 4
    assert metrics["context_count"] == 404
    # The all-context bound is swamped by the 400 zeros; the traded bound is not.
    assert metrics["lower_bound_net_bps"] < 1.0
    assert metrics["traded_lower_bound_net_bps"] > 15.0

    decision = controller.evaluate(evidence, now=AT + timedelta(days=2, minutes=5))
    assert decision.to_state is SelectorAuthorityState.LIVE_PROBE
    assert controller.order_size_fraction == 0.10


def test_losing_trades_still_demote_when_declines_would_have_masked_them(tmp_path):
    """Demotion reads the undiluted bound too, so declines cannot hide a losing run."""
    config = _config(tmp_path)
    controller = SelectorPromotionController(config)
    controller.evaluate([*_groups(), *_declined_groups(400, start=AT)], now=AT)
    controller.evaluate(
        [*_groups(), *_declined_groups(400, start=AT)], now=AT + timedelta(minutes=5)
    )
    assert controller.state is SelectorAuthorityState.LIVE_PROBE

    losing = []
    for batch in range(5):
        for group in _groups(negative=True):
            group.context_id = f"bad-{batch}-{group.context_id}"
            losing.append(group)
    # 20 losing trades against 400 declines: the all-context bound stays above the
    # demotion floor, so only the traded bound can catch this.
    decision = controller.evaluate(
        [*losing, *_declined_groups(400, start=AT)], now=AT + timedelta(days=5)
    )
    assert decision.metrics["lower_bound_net_bps"] > config.demotion_lower_bound_bps
    assert decision.to_state is SelectorAuthorityState.SHADOW
    assert "SELECTOR_AUTO_DEMOTION_NEGATIVE_EDGE" in decision.reason_codes


def test_runner_reports_effective_automatic_authority(tmp_path):
    controller = SelectorPromotionController(_config(tmp_path))
    controller.evaluate(_groups(), now=AT)
    controller.evaluate(_groups(), now=AT + timedelta(minutes=5))
    runner = SelectorV2ShadowRunner(
        flags=SelectorV2Flags(enabled=True, shadow_only=True, auto_promote=True),
        promotion=controller,
    )
    snapshot = runner.snapshot()
    assert snapshot["configured_shadow_only"] is True
    assert snapshot["shadow_only"] is False
    assert snapshot["live_authority"] is True
    assert snapshot["order_size_fraction"] == 0.10
