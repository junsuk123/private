"""Stable strategy identity contract shared by routing, models and execution."""

from __future__ import annotations


# Append-only: model output indices and persisted masks depend on this order.
STRATEGY_IDS: tuple[str, ...] = (
    "intraday_momentum",
    "breakout_volume",
    "vwap_mean_reversion",
    "liquidity_shock_reversal",
    "event_momentum",
    "cross_sectional_relative_strength",
    "gap_context",
    "rvgi_box_breakout",
    # Added for the current high-volatility, flow-driven tape. Appended, never
    # inserted: model output indices and persisted strategy masks depend on the
    # order above staying byte-stable.
    "residual_relative_strength",
    "adaptive_anchored_vwap_reversion",
    "ofi_microprice_exhaustion_reversal",
    # Opening-range breakout gated on relative volume ("stocks in play"). Added
    # from the published day-trading literature, where the unrestricted breakout
    # is unprofitable and the relative-volume restriction is what carries the
    # result — so the RVOL gate is part of the thesis, not a tuning knob.
    "opening_range_breakout",
    # Market intraday momentum: the first half-hour return predicts the last
    # half-hour return (Gao/Han/Li/Zhou, JFE 2018; confirmed in 12 of 16 developed
    # markets and in APAC). Added because it is ONE round trip per day, and this
    # account pays 20bps of statutory tax on every round trip — 72% of its KRX cost
    # is turnover-driven, so a strategy's trip count matters as much as its edge.
    "market_intraday_momentum",
    # --- SHORT-side theses -------------------------------------------------- #
    # Three, not thirteen: the long catalogue is not mirrored, because a short is
    # not a long with the sign flipped. It pays a borrow fee, its loss is unbounded
    # above, and it can be recalled. Only theses that are structurally *needed* in a
    # falling tape are added, and each is a distinct hypothesis rather than the
    # negation of a long one.
    #
    # Their presence in this tuple is NOT permission to trade them: every one ships
    # in SHADOW and must earn LIVE_PROBE from forward, borrow-aware, out-of-sample
    # outcomes (see app.trading.short_strategy_promotion).
    "market_intraday_momentum_short",
    "opening_range_breakdown",
    "residual_relative_weakness",
    # Completed one-minute reversal after a deep session-VWAP displacement.
    # Appended to preserve every existing output index while giving this thesis
    # a first-class GNN head, ontology mask, bandit arm and execution identity.
    "bar_confirmed_vwap_recovery",
    # The first thesis in this catalogue whose horizon crosses a session boundary.
    #
    # Added on cost grounds, from measurement rather than preference. A US round
    # trip through this account is 51.2bps (settlement-verified), so an entry must
    # expect ~61bps to be worth taking, and the median absolute move on the stored
    # US tape is 6.7bps over 3 minutes, 16.4 over 30 and 79 over a full session.
    # Every other long thesis here is boxed inside a horizon where that number
    # cannot be reached, which is exactly what the strategy-utility checkpoint
    # reports: EXECUTION_COST_EXCEEDS_GROSS_EDGE on every US strategy that fires.
    # The overnight gap is the first window whose move clears the round trip —
    # median 69.1bps, P(|move| > 51.2bps) = 0.62 — and it costs one round trip.
    #
    # It ships SHADOW-only. Magnitude is measured; DIRECTION is not: the stored
    # sample gives an unconditional overnight mean of +15.6bps over 55 symbol-days
    # (below cost), and the subsample carrying every price point this thesis needs
    # is 21 rows, on which that same mean flips to -49.8. Nothing in that data can
    # authorize a live order, so promotion is left where it belongs — forward,
    # out-of-sample outcomes through app.trading.short_strategy_promotion's ladder.
    "overnight_gap_carry",
)

# Compatibility alias for callers that explicitly ask for the complete catalog.
# There is only one tier: every member is a first-class strategy.
ALL_STRATEGY_IDS: tuple[str, ...] = STRATEGY_IDS

# Catalogued strategies whose thesis is short-side. Direction is a property of the
# STRATEGY here (each of these only ever opens SHORT), while the tradable ARM
# identity carries direction explicitly — see
# ``app.trading.directional.DirectionalStrategyKey``.
SHORT_STRATEGY_IDS: tuple[str, ...] = (
    "market_intraday_momentum_short",
    "opening_range_breakdown",
    "residual_relative_weakness",
)

# The long strategy each short thesis is the counterpart of. Used only for
# reporting and for the LONG-vs-SHORT-vs-NO_TRADE comparison view; posteriors are
# never shared across a pair (see DirectionalStrategyKey).
SHORT_LONG_COUNTERPART: dict[str, str] = {
    "market_intraday_momentum_short": "market_intraday_momentum",
    "opening_range_breakdown": "opening_range_breakout",
    "residual_relative_weakness": "residual_relative_strength",
}


def is_short_strategy(strategy_id: object) -> bool:
    return str(strategy_id or "") in set(SHORT_STRATEGY_IDS)

STRATEGY_INDEX: dict[str, int] = {
    strategy_id: index for index, strategy_id in enumerate(STRATEGY_IDS)
}


def is_known_strategy(strategy_id: object) -> bool:
    return str(strategy_id or "") in STRATEGY_INDEX


# --------------------------------------------------------------------------- #
# Micro-layer vocabulary bridge                                                #
# --------------------------------------------------------------------------- #
# The micro/ontology layer classifies a METHODOLOGY and reports it as a
# ``SelectedStrategy`` enum value: momentum / breakout / mean_reversion /
# vwap_reversion. Those are deliberately generic — they are what the composite
# technical engine can distinguish, and they also serve as macro permission tokens.
#
# The execution layer needs a CATALOGUED strategy id, because that is what resolves
# an algorithm, an exit geometry and a deployment authorisation. The two vocabularies
# had ZERO overlap, so every ontology-elected candidate reached
# ``_deployment_authorized`` as e.g. "momentum", ``get_algorithm("momentum")``
# returned None, and the election was rejected with
# ``STRATEGY_NOT_LIVE_AUTHORIZED:momentum``. Measured: 0 of 13 catalogued strategies
# were reachable through the ontology path, which is why ontology strategy selection
# never happened at all.
#
# Mapped by THESIS, not by backtested performance — picking the best-scoring target
# would be fitting the translation table to past results:
#   momentum        EMA fast/slow trend continuation -> intraday_momentum
#   breakout        Donchian range break + volume    -> breakout_volume
#   vwap_reversion  explicit VWAP displacement       -> vwap_mean_reversion
#   mean_reversion  RSI / Bollinger %B oversold      -> vwap_mean_reversion
#
# The last one is the loosest fit and is recorded as such: the generic oversold
# thesis reverts to a band midline, while the catalogued strategy reverts to VWAP.
# The catalogue has no live-authorised RSI-reversion strategy, and the target
# algorithm still applies its own VWAP conditions, so a mismatch fails closed rather
# than trading on the wrong thesis.
METHODOLOGY_STRATEGY_ALIASES: dict[str, str] = {
    "momentum": "intraday_momentum",
    "breakout": "breakout_volume",
    "vwap_reversion": "vwap_mean_reversion",
    "mean_reversion": "vwap_mean_reversion",
}

# Micro verdicts that are not buy theses at all; they must never resolve to a
# tradable strategy id.
NON_TRADABLE_MICRO_STRATEGIES: frozenset[str] = frozenset(
    {"hold", "sell", "reduce_risk"}
)


def resolve_strategy_id(name: object) -> str | None:
    """Catalogued strategy id for a catalogued id OR a micro methodology name.

    Returns ``None`` when the name is not tradable (``hold``/``sell``/
    ``reduce_risk``) or cannot be resolved — the caller must treat that as "no
    strategy", never as a default pick. Silently defaulting to the first catalogue
    entry is how a generic ontology gate would become an executable trade.
    """
    text = str(name or "").strip().lower()
    if not text or text in NON_TRADABLE_MICRO_STRATEGIES:
        return None
    if text in STRATEGY_INDEX:
        return text
    resolved = METHODOLOGY_STRATEGY_ALIASES.get(text)
    return resolved if resolved in STRATEGY_INDEX else None
