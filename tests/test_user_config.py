"""Tests for cascade.user_config (credentials + precedence)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from cascade.exceptions import CascadeConfigError
from cascade.user_config import (
    LLMProviderConfig,
    UserConfig,
    VCSProviderConfig,
    config_dir,
    config_path,
    load_user_config,
    mask_secret,
    resolve_issue_credentials,
    resolve_llm_credentials,
    resolve_vcs_credentials,
    save_user_config,
)


# --------- Path resolution ----------


class TestConfigPath:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("CASCADE_CONFIG_HOME", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/test")
        assert config_dir() == Path("/home/test/.config/cascade")
        assert config_path() == Path("/home/test/.config/cascade/config.yaml")

    def test_cascade_home_override(self, monkeypatch):
        monkeypatch.setenv("CASCADE_CONFIG_HOME", "/tmp/cstest")
        assert config_dir() == Path("/tmp/cstest")

    def test_xdg_config_home(self, monkeypatch):
        monkeypatch.delenv("CASCADE_CONFIG_HOME", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg")
        assert config_dir() == Path("/xdg/cascade")


# --------- Load / save ----------


class TestLoad:
    def test_missing_file_returns_empty_defaults(self, tmp_path):
        cfg = load_user_config(tmp_path / "nope.yaml")
        assert isinstance(cfg, UserConfig)
        assert cfg.defaults.llm_provider == "anthropic"
        assert cfg.llm_providers == {}

    def test_load_with_providers(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            "version: 1\n"
            "defaults:\n"
            "  llm_provider: openai\n"
            "  vcs_provider: github\n"
            "llm_providers:\n"
            "  openai:\n"
            "    api_key: sk-test\n"
            "    default_model: gpt-5\n"
            "  anthropic:\n"
            "    api_key: sk-ant\n"
            "vcs_providers:\n"
            "  github:\n"
            "    token: ghp-xxx\n"
        )
        cfg = load_user_config(p)
        assert cfg.defaults.llm_provider == "openai"
        assert cfg.llm_providers["openai"].api_key == "sk-test"
        assert cfg.llm_providers["openai"].default_model == "gpt-5"
        assert cfg.llm_providers["anthropic"].api_key == "sk-ant"
        assert cfg.vcs_providers["github"].token == "ghp-xxx"

    def test_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("not:\n  valid:\n   :")
        with pytest.raises(CascadeConfigError, match="Invalid YAML"):
            load_user_config(p)

    def test_non_mapping_top_level_raises(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("- just a list")
        with pytest.raises(CascadeConfigError, match="must be a YAML mapping"):
            load_user_config(p)

    def test_unknown_field_rejected(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            "llm_providers:\n  anthropic:\n    api_key: sk\n    "
            "what_is_this: oops\n"
        )
        with pytest.raises(CascadeConfigError):
            load_user_config(p)


class TestSave:
    def test_save_round_trip(self, tmp_path):
        cfg = UserConfig(
            llm_providers={"openai": LLMProviderConfig(api_key="sk-x")},
            vcs_providers={"github": VCSProviderConfig(token="ghp-x")},
        )
        p = tmp_path / "config.yaml"
        save_user_config(cfg, p)
        loaded = load_user_config(p)
        assert loaded.llm_providers["openai"].api_key == "sk-x"
        assert loaded.vcs_providers["github"].token == "ghp-x"

    def test_save_creates_parent_dir(self, tmp_path):
        cfg = UserConfig()
        p = tmp_path / "deep" / "config.yaml"
        save_user_config(cfg, p)
        assert p.exists()

    def test_save_sets_restrictive_permissions(self, tmp_path):
        cfg = UserConfig(
            llm_providers={"anthropic": LLMProviderConfig(api_key="sk-secret")}
        )
        p = tmp_path / "config.yaml"
        save_user_config(cfg, p)
        mode = p.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got 0o{mode:o}"


# --------- Credential resolution ----------


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all relevant env vars so tests start from a known state."""
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITLAB_TOKEN",
        "BITBUCKET_TOKEN",
        "AZURE_DEVOPS_TOKEN",
        "AZURE_DEVOPS_EXT_PAT",
        "JIRA_API_TOKEN",
        "LINEAR_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


class TestResolveLLMCredentials:
    def test_uses_user_config_key(self, clean_env):
        cfg = UserConfig(
            llm_providers={"anthropic": LLMProviderConfig(api_key="from-config")}
        )
        resolved = resolve_llm_credentials(user_config=cfg)
        assert resolved.api_key == "from-config"
        assert resolved.source == "user config"
        assert resolved.provider == "anthropic"

    def test_cli_override_wins_over_config(self, clean_env):
        cfg = UserConfig(
            llm_providers={"anthropic": LLMProviderConfig(api_key="from-config")}
        )
        resolved = resolve_llm_credentials(user_config=cfg, api_key_override="cli-key")
        assert resolved.api_key == "cli-key"
        assert "CLI override" in resolved.source

    def test_env_fallback(self, clean_env, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        cfg = UserConfig()
        resolved = resolve_llm_credentials(user_config=cfg)
        assert resolved.api_key == "from-env"
        assert "env var" in resolved.source

    def test_config_wins_over_env(self, clean_env, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        cfg = UserConfig(
            llm_providers={"anthropic": LLMProviderConfig(api_key="from-config")}
        )
        resolved = resolve_llm_credentials(user_config=cfg)
        assert resolved.api_key == "from-config"

    def test_explicit_provider_overrides_default(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
        cfg = UserConfig()  # default is anthropic
        resolved = resolve_llm_credentials(user_config=cfg, provider="openai")
        assert resolved.provider == "openai"
        assert resolved.api_key == "sk-oai"

    def test_google_accepts_either_env_var(self, clean_env, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        cfg = UserConfig()
        resolved = resolve_llm_credentials(user_config=cfg, provider="google")
        assert resolved.api_key == "gemini-key"

    def test_missing_key_raises_with_actionable_message(self, clean_env):
        cfg = UserConfig()
        with pytest.raises(CascadeConfigError, match="No API key configured") as info:
            resolve_llm_credentials(user_config=cfg)
        # Actionable hints land in exc.hints, not the message
        joined = "\n".join(info.value.hints)
        assert "cascade configure llm anthropic" in joined

    def test_provider_without_key_requirement_does_not_raise(self, clean_env):
        # claude_code uses the local Claude subscription; no API key needed
        cfg = UserConfig()
        resolved = resolve_llm_credentials(user_config=cfg, provider="claude_code")
        assert resolved.api_key is None

    def test_model_override_propagates(self, clean_env, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        cfg = UserConfig()
        resolved = resolve_llm_credentials(
            user_config=cfg, model_override="claude-x"
        )
        assert resolved.model == "claude-x"

    def test_default_model_from_config(self, clean_env):
        cfg = UserConfig(
            llm_providers={
                "anthropic": LLMProviderConfig(api_key="k", default_model="claude-y")
            }
        )
        resolved = resolve_llm_credentials(user_config=cfg)
        assert resolved.model == "claude-y"


class TestResolveVCSCredentials:
    def test_uses_user_config_token(self, clean_env):
        cfg = UserConfig(
            vcs_providers={"github": VCSProviderConfig(token="ghp-cfg")}
        )
        resolved = resolve_vcs_credentials(user_config=cfg)
        assert resolved.token == "ghp-cfg"

    def test_env_fallback(self, clean_env, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp-env")
        cfg = UserConfig()
        resolved = resolve_vcs_credentials(user_config=cfg)
        assert resolved.token == "ghp-env"

    def test_gh_token_also_works(self, clean_env, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "ghp-alt")
        cfg = UserConfig()
        resolved = resolve_vcs_credentials(user_config=cfg)
        assert resolved.token == "ghp-alt"

    def test_azure_devops_requires_org(self, clean_env, monkeypatch):
        monkeypatch.setenv("AZURE_DEVOPS_TOKEN", "azp-x")
        cfg = UserConfig()
        # Token resolves, but organization may still be None -- caller's job
        # to enforce. We don't fail in the credential resolver.
        resolved = resolve_vcs_credentials(user_config=cfg, provider="azure_devops")
        assert resolved.token == "azp-x"

    def test_missing_token_raises(self, clean_env):
        cfg = UserConfig()
        with pytest.raises(CascadeConfigError, match="No token configured") as info:
            resolve_vcs_credentials(user_config=cfg)
        joined = "\n".join(info.value.hints)
        assert "cascade configure vcs" in joined


class TestResolveIssueCredentials:
    def test_uses_user_config(self, clean_env):
        from cascade.user_config import IssueSourceConfig

        cfg = UserConfig(
            issue_sources={
                "jira": IssueSourceConfig(
                    token="jt", base_url="https://x.atlassian.net", user="u@e.com"
                )
            }
        )
        resolved = resolve_issue_credentials(user_config=cfg, provider="jira")
        assert resolved.token == "jt"
        assert resolved.base_url == "https://x.atlassian.net"
        assert resolved.user == "u@e.com"

    def test_missing_raises(self, clean_env):
        cfg = UserConfig()
        with pytest.raises(CascadeConfigError, match="No token configured") as info:
            resolve_issue_credentials(user_config=cfg, provider="jira")
        joined = "\n".join(info.value.hints)
        assert "cascade configure issue jira" in joined


# --------- mask_secret ----------


class TestMaskSecret:
    def test_long_secret(self):
        assert mask_secret("sk-1234567890abcdef") == "***************cdef"

    def test_short_secret(self):
        assert mask_secret("abc") == "***"

    def test_none(self):
        assert mask_secret(None) == "(not set)"

    def test_empty(self):
        assert mask_secret("") == "(not set)"

    def test_custom_visible_chars(self):
        assert mask_secret("sk-1234567890", visible=6) == "*******567890"
