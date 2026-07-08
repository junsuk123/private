from __future__ import annotations

from app.graph.macro_micro_config import DEFAULT_CONFIG_PATH, load_macro_micro_policy


class TestConfigLoader:
    def test_loads_repo_default(self):
        p = load_macro_micro_policy(DEFAULT_CONFIG_PATH)
        assert p.enabled is True
        assert p.macro_loop_interval_seconds == 60
        assert p.micro_loop_interval_seconds == 5
        assert p.coordinator_config.max_parallel_symbols == 20
        assert p.macro_config.candidate_limit == 30
        assert p.micro_config.minimum_micro_confidence == 0.55
        # Strategy permissions from YAML.
        assert "momentum" in p.macro_config.strategy_permissions["TREND_UP"]["allow"]
        assert "new_buy" in p.macro_config.strategy_permissions["HIGH_VOLATILITY_RISK"]["block"]
        assert p.diagnostics["config_fallbacks"] == []

    def test_missing_file_uses_defaults(self):
        p = load_macro_micro_policy("config/does_not_exist_macro_micro.yaml")
        assert p.enabled is True
        assert p.coordinator_config.max_parallel_symbols == 20
        assert any("config_missing" in f for f in p.diagnostics["config_fallbacks"])

    def test_env_override_precedence(self, monkeypatch):
        monkeypatch.setenv("MACRO_CANDIDATE_LIMIT", "7")
        p = load_macro_micro_policy("config/does_not_exist_macro_micro.yaml")
        # from_env feeds the default when YAML absent -> env wins.
        assert p.macro_config.candidate_limit == 7

    def test_invalid_values_fall_back_and_are_recorded(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(
            "enabled: true\nmacro:\n  loop_interval_seconds: 5\n  candidate_limit: not_a_number\n"
            "micro:\n  max_parallel_symbols: 99999\n",
            encoding="utf-8",
        )
        p = load_macro_micro_policy(cfg)
        # macro loop clamped up to the 30s minimum; parallel clamped to 200.
        assert p.macro_loop_interval_seconds == 30
        assert p.coordinator_config.max_parallel_symbols == 200
        assert any("candidate_limit" in f for f in p.diagnostics["config_fallbacks"])
        assert any("macro_loop_interval_seconds" in f for f in p.diagnostics["config_fallbacks"])
