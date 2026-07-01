from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.graph.theory_registry import TheoryRegistry, get_theory_registry


DEFAULT_VALIDATION_PATH = Path("data/models/theory_validation_scores.json")


@dataclass(frozen=True)
class TheoryValidationScore:
    theory_id: str
    validation_weight: float
    status: str
    oos_trade_count: int = 0
    diagnostics: dict[str, Any] | None = None


class TheoryValidationStore:
    def __init__(self, path: str | Path = DEFAULT_VALIDATION_PATH, registry: TheoryRegistry | None = None) -> None:
        self.path = Path(path)
        self.registry = registry or get_theory_registry()
        self.scores = self._load()

    def weight_for(self, theory_id: str) -> float:
        score = self.scores.get(theory_id)
        if score is not None:
            return max(0.0, min(1.0, score.validation_weight))
        return self.registry.weight_for(theory_id)

    def _load(self) -> dict[str, TheoryValidationScore]:
        if not self.path.exists():
            return {}
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        scores: dict[str, TheoryValidationScore] = {}
        for theory_id, data in (loaded.get("theories") or {}).items():
            scores[theory_id] = TheoryValidationScore(
                theory_id=theory_id,
                validation_weight=float(data.get("validation_weight", self.registry.weight_for(theory_id))),
                status=str(data.get("status", "unvalidated")),
                oos_trade_count=int(data.get("oos_trade_count", 0)),
                diagnostics=dict(data.get("diagnostics") or {}),
            )
        return scores


@lru_cache(maxsize=1)
def get_theory_validation_store() -> TheoryValidationStore:
    return TheoryValidationStore()


def reset_theory_validation_store() -> None:
    get_theory_validation_store.cache_clear()
