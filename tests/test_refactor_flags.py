from __future__ import annotations

import pytest

from app.config.refactor_flags import RefactorFeatureFlags


def test_refactor_defaults_preserve_legacy_and_disable_new_live_path() -> None:
    flags = RefactorFeatureFlags.from_env({})
    assert flags.legacy_vote_path is True
    assert flags.strategy_owned_execution is False
    assert flags.live_enabled is False


def test_refactor_live_requires_strategy_owned_execution() -> None:
    with pytest.raises(ValueError, match="strategy-owned execution"):
        RefactorFeatureFlags.from_env({"REFACTOR_LIVE_ENABLED": "true"})


def test_gnn_rerank_requires_ontology_router() -> None:
    with pytest.raises(ValueError, match="ontology router"):
        RefactorFeatureFlags.from_env({"REFACTOR_GNN_RERANK": "true"})


def test_invalid_boolean_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        RefactorFeatureFlags.from_env({"REFACTOR_GNN_SHADOW": "sometimes"})
