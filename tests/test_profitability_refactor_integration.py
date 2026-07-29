"""Integration tests for the profitability refactor acceptance criteria.

The headline case comes straight from the replay baseline (docs/validation.md):
the pre-refactor system took trades that were GROSS-positive but NET-negative after costs.
The unified ProfitabilityGate must now reject exactly those in the live buy path.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from app.schemas.domain import AccountSnapshot
from app.trading.shared_decision_engine import SharedLiveDecisionEngine


class _BuyStore:
    def __init__(self, price: float) -> None:
        now = datetime.now(timezone.utc)
        self._tick = SimpleNamespace(price=price, received_at=now, exchange_timestamp=now, sequence_key=f"buy:{price}")

    def latest_tick(self, symbol: str):
        return self._tick

    def latest_orderbook(self, symbol: str):
        return None


class _ThinEdgePredictor:
    """Model with a small positive predicted move: gross-positive, net-negative after cost."""

    def __init__(self, bps: float) -> None:
        self._bps = bps

    def predict(self, frame):
        return SimpleNamespace(
            probability_success=0.7,
            expected_net_return_bps=self._bps,
            uncertainty_score=0.2,
            approved=True,
            reason_codes=(),
            model_artifact_id="unit",
            feature_schema_hash=frame.feature_schema_hash,
            provider="trained_model",
            is_fallback=False,
        )


def _graph(support=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying")):
    class _G:
        def matching(self, subject=None, predicate=None, obj=None):
            if predicate == "supportsSignal":
                return [SimpleNamespace(object=o) for o in support]
            return []

    return _G()


def _engine_with_model(price: float, bps: float) -> SharedLiveDecisionEngine:
    engine = SharedLiveDecisionEngine(_BuyStore(price=price), predictor=_ThinEdgePredictor(bps))
    engine.feature_builder = SimpleNamespace(
        build=lambda symbol, decision_time=None: SimpleNamespace(
            feature_schema_hash="unit-schema",
            provenance=SimpleNamespace(source_record_ids=("unit-frame",)),
        )
    )
    return engine


class ProfitabilityRefactorIntegrationTest(unittest.TestCase):
    def test_gross_positive_net_negative_buy_is_rejected(self) -> None:
        # +0.3% predicted move: gross-positive but does not clear the KR cost + net floor.
        # An explicit 0.8% KR floor is set here so this exercises the gate MECHANISM
        # independent of the deployment's tuned profitability_policy.yaml (which relaxes
        # the KR floor to 0.0). The policy is read at gate construction, so the env must
        # be set BEFORE the engine is built.
        with_env = {"REALTIME_MODEL_AUXILIARY_ONLY": "true", "REALTIME_MIN_BUY_NET_RETURN_KR": "0.008"}
        old = {k: os.environ.get(k) for k in with_env}
        os.environ.update(with_env)
        try:
            engine = _engine_with_model(price=10_000.0, bps=30.0)
            account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
            result = engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=_graph())
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertFalse(result.approved, result.reason_codes)
        self.assertIn("PROFITABILITY_GATE_REJECTED", result.reason_codes)
        pd = result.diagnostics.get("profitability_decision", {})
        self.assertFalse(pd.get("allowed", True))
        # Confirm the pathology: gross > 0 but net < required.
        self.assertGreater(pd.get("gross_expected_return", 0.0), 0.0)
        self.assertLess(pd.get("net_expected_return", 1.0), pd.get("required_min_net_return", 0.0))

    def test_genuinely_profitable_buy_is_approved(self) -> None:
        # +2% predicted move clears KR cost + floor.
        engine = _engine_with_model(price=10_000.0, bps=200.0)
        account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
        old = os.environ.get("REALTIME_MODEL_AUXILIARY_ONLY")
        os.environ["REALTIME_MODEL_AUXILIARY_ONLY"] = "true"
        try:
            result = engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=_graph())
        finally:
            if old is None:
                os.environ.pop("REALTIME_MODEL_AUXILIARY_ONLY", None)
            else:
                os.environ["REALTIME_MODEL_AUXILIARY_ONLY"] = old
        self.assertTrue(result.approved, result.reason_codes)
        pd = result.diagnostics.get("profitability_decision", {})
        self.assertTrue(pd.get("allowed"))
        self.assertGreaterEqual(pd.get("net_expected_return", 0.0), pd.get("required_min_net_return", 1.0))


if __name__ == "__main__":
    unittest.main()
