"""Feature flags for the V2 strategy-selection pipeline. Safe by default.

Default posture, stated as code rather than as a comment
------------------------------------------------------
``STRATEGY_SELECTOR_V2_ENABLED`` defaults to false, and when it is on
``STRATEGY_SELECTOR_V2_SHADOW_ONLY`` defaults to true. A bare process therefore cannot
grant V2 authority. The production launcher may additionally enable
``STRATEGY_SELECTOR_V2_AUTO_PROMOTE``: V2 still starts in SHADOW, but a persisted evidence
controller can earn LIVE_PROBE and then LIVE authority without an operator editing flags.
The static ``live_authority`` property below means only operator-forced authority; runtime
telemetry reports the controller's effective authority separately.

``validate`` refuses combinations that would be silently unsafe, in the same style as
``app.config.refactor_flags.RefactorFeatureFlags.validate``. A misconfiguration should stop
startup, not produce a subtly different trading posture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

__all__ = ["SelectorV2Flags"]


def _bool(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")


@dataclass(frozen=True)
class SelectorV2Flags:
    #: Compute a V2 selection at all. Off means the module is inert.
    enabled: bool = False
    #: V2's result is telemetry only. TRUE is the only safe default; see the docstring.
    shadow_only: bool = True
    #: Let measured forward evidence move SHADOW -> LIVE_PROBE -> LIVE.  The initial
    #: posture is still shadow-only; this merely replaces an operator editing an env
    #: variable with the persisted promotion controller.
    auto_promote: bool = False
    #: Open counterfactual shadow positions for the strategies V2 did not pick.
    counterfactual_enabled: bool = True
    #: Use the R-GCN utility vector where one exists (else the heuristic predictor).
    utility_gnn_enabled: bool = False
    #: Apply the bounded realized-history correction term.
    bandit_adapter_enabled: bool = True
    #: NO_TRADE as a first-class arm. Off makes the selector always pick something, which
    #: is the ``forced_selection`` failure — hence it defaults on and ``validate`` refuses
    #: to run it off in live mode.
    no_trade_enabled: bool = True
    #: Hard eligibility mask keyed on real strategy ids.
    ontology_mask_v2_enabled: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SelectorV2Flags":
        values = os.environ if env is None else env
        flags = cls(
            enabled=_bool(values, "STRATEGY_SELECTOR_V2_ENABLED", False),
            shadow_only=_bool(values, "STRATEGY_SELECTOR_V2_SHADOW_ONLY", True),
            auto_promote=_bool(values, "STRATEGY_SELECTOR_V2_AUTO_PROMOTE", False),
            counterfactual_enabled=_bool(values, "STRATEGY_COUNTERFACTUAL_ENABLED", True),
            utility_gnn_enabled=_bool(values, "STRATEGY_UTILITY_GNN_ENABLED", False),
            bandit_adapter_enabled=_bool(values, "STRATEGY_BANDIT_ADAPTER_ENABLED", True),
            no_trade_enabled=_bool(values, "STRATEGY_NO_TRADE_ENABLED", True),
            ontology_mask_v2_enabled=_bool(
                values, "STRATEGY_ONTOLOGY_MASK_V2_ENABLED", True
            ),
        )
        flags.validate()
        return flags

    def validate(self) -> None:
        # With the selector off, every sub-flag is unread, so no combination of them can be
        # unsafe and none is worth failing startup over.
        if self.enabled and (not self.shadow_only or self.auto_promote):
            # Live authority. Every safety-relevant sub-flag must be ON, because the live
            # path would otherwise select without the mask or without NO_TRADE.
            if not self.no_trade_enabled:
                raise ValueError(
                    "STRATEGY_SELECTOR_V2 live mode requires STRATEGY_NO_TRADE_ENABLED: "
                    "without it the selector must always choose a strategy"
                )
            if not self.ontology_mask_v2_enabled:
                raise ValueError(
                    "STRATEGY_SELECTOR_V2 live mode requires "
                    "STRATEGY_ONTOLOGY_MASK_V2_ENABLED: hard eligibility cannot be skipped"
                )
            if not self.counterfactual_enabled:
                raise ValueError(
                    "STRATEGY_SELECTOR_V2 live mode requires "
                    "STRATEGY_COUNTERFACTUAL_ENABLED: selector regret must stay measurable "
                    "once V2 holds authority"
                )

    @property
    def live_authority(self) -> bool:
        """Operator-forced authority from flags; excludes automatically earned state."""
        return bool(self.enabled and not self.shadow_only)

    def as_dict(self) -> dict[str, bool]:
        return {
            "STRATEGY_SELECTOR_V2_ENABLED": self.enabled,
            "STRATEGY_SELECTOR_V2_SHADOW_ONLY": self.shadow_only,
            "STRATEGY_SELECTOR_V2_AUTO_PROMOTE": self.auto_promote,
            "STRATEGY_COUNTERFACTUAL_ENABLED": self.counterfactual_enabled,
            "STRATEGY_UTILITY_GNN_ENABLED": self.utility_gnn_enabled,
            "STRATEGY_BANDIT_ADAPTER_ENABLED": self.bandit_adapter_enabled,
            "STRATEGY_NO_TRADE_ENABLED": self.no_trade_enabled,
            "STRATEGY_ONTOLOGY_MASK_V2_ENABLED": self.ontology_mask_v2_enabled,
            "live_authority": self.live_authority,
        }
