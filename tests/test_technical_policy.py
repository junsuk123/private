from __future__ import annotations

from app.technical.policy import DEFAULT_POLICY_PATH, load_technical_policy


class TestPolicyLoader:
    def test_loads_repo_default(self):
        policy = load_technical_policy(DEFAULT_POLICY_PATH)
        assert policy.enabled is True
        assert policy.signal_engine.methodology_weights["vwap_volume_liquidity"] == 1.2
        assert policy.prediction.min_confidence == 0.5
        assert policy.regime.min_liquidity_score == 0.35
        assert 60 in policy.default_horizons_seconds

    def test_missing_file_falls_back_to_defaults(self):
        policy = load_technical_policy("config/does_not_exist_technical.yaml")
        assert policy.enabled is True
        assert policy.signal_engine.methodology_weights["momentum_trend_following"] == 1.0

    def test_env_override_precedence(self, monkeypatch):
        monkeypatch.setenv("TECHNICAL_REGIME_MIN_LIQUIDITY", "0.7")
        policy = load_technical_policy(DEFAULT_POLICY_PATH)
        assert policy.regime.min_liquidity_score == 0.7
