"""Tests for cascade.config."""

import pytest

from cascade.config import (
    AgentConfig,
    CascadeConfig,
    MemoryConfig,
    PathsConfig,
    load_config,
)
from cascade.exceptions import CascadeConfigError


class TestDefaults:
    def test_defaults_load_without_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert isinstance(cfg, CascadeConfig)
        assert cfg.agent.provider == "anthropic"
        assert cfg.agent.model == "claude-opus-4-7"
        assert cfg.memory.path == "team-memory"
        assert "src/**" in cfg.paths.allowed
        assert ".github/**" in cfg.paths.disallowed

    def test_top_level_is_frozen(self):
        cfg = CascadeConfig()
        with pytest.raises(Exception):
            cfg.test_command = "make test"


class TestLoadFromFile:
    def test_partial_overrides_use_defaults_for_rest(self, tmp_path):
        cfg_file = tmp_path / "cascade.yaml"
        cfg_file.write_text(
            "agent:\n  model: claude-sonnet-4-6\n  temperature: 0.5\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.agent.model == "claude-sonnet-4-6"
        assert cfg.agent.temperature == 0.5
        assert cfg.agent.provider == "anthropic"  # default preserved

    def test_full_override(self, tmp_path):
        cfg_file = tmp_path / "cascade.yaml"
        cfg_file.write_text(
            """
version: 1
agent:
  provider: anthropic
  model: claude-opus-4-7
  max_iterations: 2
  temperature: 0.1
memory:
  path: ./my-memory
  max_chars_per_call: 50000
paths:
  allowed:
    - src/**
  disallowed:
    - secret/**
test_command: pytest -xvs
"""
        )
        cfg = load_config(cfg_file)
        assert cfg.agent.max_iterations == 2
        assert cfg.memory.path == "./my-memory"
        assert cfg.test_command == "pytest -xvs"
        assert "secret/**" in cfg.paths.disallowed


class TestErrors:
    def test_invalid_yaml(self, tmp_path):
        cfg_file = tmp_path / "cascade.yaml"
        cfg_file.write_text("agent:\n  this is: not valid:\n  yaml: : :")
        with pytest.raises(CascadeConfigError, match="Invalid YAML"):
            load_config(cfg_file)

    def test_top_level_not_a_dict(self, tmp_path):
        cfg_file = tmp_path / "cascade.yaml"
        cfg_file.write_text("- just\n- a\n- list")
        with pytest.raises(CascadeConfigError, match="must be a YAML mapping"):
            load_config(cfg_file)

    def test_invalid_field_type(self, tmp_path):
        cfg_file = tmp_path / "cascade.yaml"
        cfg_file.write_text("agent:\n  max_iterations: not-a-number\n")
        with pytest.raises(CascadeConfigError, match="Invalid config"):
            load_config(cfg_file)

    def test_value_out_of_range(self, tmp_path):
        cfg_file = tmp_path / "cascade.yaml"
        cfg_file.write_text("agent:\n  temperature: 3.0\n")  # max 2
        with pytest.raises(CascadeConfigError):
            load_config(cfg_file)


class TestSubModels:
    def test_agent_config_validates_iterations_range(self):
        with pytest.raises(Exception):
            AgentConfig(max_iterations=10)
        with pytest.raises(Exception):
            AgentConfig(max_iterations=-1)

    def test_memory_config_chars_bounds(self):
        with pytest.raises(Exception):
            MemoryConfig(max_chars_per_call=500)  # below min
        with pytest.raises(Exception):
            MemoryConfig(max_chars_per_call=500_000)  # above max

    def test_paths_config_defaults(self):
        p = PathsConfig()
        assert "src/**" in p.allowed
        assert ".github/**" in p.disallowed
