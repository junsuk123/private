"""Strategy validation: one evaluator, purged splits, cost stress, lifecycle ledger.

Every module here is OFF the trading hot path. Nothing in this package is imported by the
selector, the session manager or the execution layer — the only coupling is the other
direction: the selector reads a lifecycle state that this package writes.
"""

from app.strategy_validation.audit_runner import (
    AuditClassification,
    AuditReport,
    AuditThresholds,
    StrategyAudit,
    StrategyAuditRunner,
)
from app.strategy_validation.cost_stress import (
    DEFAULT_SCENARIOS,
    CostStressResult,
    CostStressScenario,
    cost_stress,
)
from app.strategy_validation.metrics import (
    StrategyMetrics,
    TradeObservation,
    compute_metrics,
    effective_sample_count,
)
from app.strategy_validation.parameter_stability import (
    ParameterStabilityResult,
    ParameterSweep,
    parameter_stability,
)
from app.strategy_validation.purged_cv import (
    PurgedSplit,
    combinatorial_splits,
    purged_kfold_splits,
)
from app.strategy_validation.regime_breakdown import (
    BucketStats,
    RegimeBreakdown,
    regime_breakdown,
)
from app.strategy_validation.registry import (
    ALLOWED_PROMOTIONS,
    LifecycleTransition,
    PromotionGates,
    StrategyValidationRecord,
    StrategyValidationRegistry,
)
from app.strategy_validation.walk_forward import (
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward,
)

__all__ = [
    "ALLOWED_PROMOTIONS",
    "AuditClassification",
    "AuditReport",
    "AuditThresholds",
    "BucketStats",
    "CostStressResult",
    "CostStressScenario",
    "DEFAULT_SCENARIOS",
    "LifecycleTransition",
    "ParameterStabilityResult",
    "ParameterSweep",
    "PromotionGates",
    "PurgedSplit",
    "RegimeBreakdown",
    "StrategyAudit",
    "StrategyAuditRunner",
    "StrategyMetrics",
    "StrategyValidationRecord",
    "StrategyValidationRegistry",
    "TradeObservation",
    "WalkForwardResult",
    "WalkForwardWindow",
    "combinatorial_splits",
    "compute_metrics",
    "cost_stress",
    "effective_sample_count",
    "parameter_stability",
    "purged_kfold_splits",
    "regime_breakdown",
    "walk_forward",
]
