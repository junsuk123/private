"""Local quantitative reference layer.

The live path is deliberately dependency-free: GS Quant is imported only by the
explicit offline validator in :mod:`app.quant.reference`.
"""

from app.quant.contracts import (
    DataQuality,
    QuantBar,
    QuantEvidence,
    QuantKnowledgeProvider,
    ValidationStatus,
)
from app.quant.engine import IncrementalQuantEngine

__all__ = [
    "DataQuality",
    "IncrementalQuantEngine",
    "QuantBar",
    "QuantEvidence",
    "QuantKnowledgeProvider",
    "ValidationStatus",
]
