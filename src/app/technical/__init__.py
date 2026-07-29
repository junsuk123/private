"""Evidence-based short-term technical prediction layer.

This package adds deterministic technical indicators, a rule-based market
regime classifier, methodology-specific signal providers, supervised label
builders, and a conservative prediction engine.

Design contract (see docs/decision_and_risk.md):
    * Every output in this package is ADVISORY EVIDENCE ONLY.
    * Nothing here creates or submits a FinalOrder, and nothing here relaxes
      TradingCostEngine / ProfitabilityGate / PrincipalProtectionEngine /
      RiskManager / FinalTradeGate. Those remain the sole authorities.
    * Functions are pure and deterministic. Missing/short data yields NaN-safe
      ``None`` (or ``ok=False`` result objects) plus explicit diagnostics —
      never fabricated prices or future data.
"""

from __future__ import annotations

__all__ = ["indicators"]
