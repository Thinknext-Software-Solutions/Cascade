"""User-level configuration: credentials, defaults, provider settings.

Lives at `~/.config/cascade/config.yaml` (or $CASCADE_CONFIG_HOME if set).
Persists between projects -- this is where API keys and other personal
defaults are stored so users don't have to re-enter them per repo.

Precedence (highest wins):
    1. CLI flags (handled by individual commands)
    2. Project-level cascade.yaml (handled by config.py / CascadeConfig)
    3. User-level config (this module)
    4. Environment variables (this module reads them as fallback)

This module is intentionally separate from `config.py` (which handles
project-level config). The two compose via `resolve_credentials()`.

Security:
- Config file is created with mode 0600 (user-read/write only)
- Credentials are never logged or printed without explicit masking
- `mask_secret()` helper for display
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .exceptions import CascadeConfigError


# ----------------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------------


CONFIG_FILENAME = "config.yaml"


def config_dir() -> Path:
    """Where the user config lives.

    Honors:
        - $CASCADE_CONFIG_HOME (explicit override)
        - $XDG_CONFIG_HOME/cascade (XDG standard)
        - ~/.config/cascade (XDG default on Linux/Mac)
    """
    explicit = os.environ.get("CASCADE_CONFIG_HOME")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "cascade"
    return Path.home() / ".config" / "cascade"


def config_path() -> Path:
    """Full path to the user config file."""
    return config_dir() / CONFIG_FILENAME


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class LLMProviderConfig(BaseModel):
    """Settings for one LLM provider."""

    model_config = ConfigDict(extra="forbid")

    api_key: Optional[str] = Field(default=None, description="API key (secret)")
    default_model: Optional[str] = Field(
        default=None, description="Default model identifier for this provider"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Override the API base URL (e.g. for Azure OpenAI, "
        "local LLM gateways, or self-hosted models)",
    )


class VCSProviderConfig(BaseModel):
    """Settings for one VCS provider."""

    model_config = ConfigDict(extra="forbid")

    token: Optional[str] = Field(default=None, description="Access token (secret)")
    base_url: Optional[str] = Field(
        default=None,
        description="Override the API base URL (e.g. for self-hosted GitLab "
        "or Bitbucket Server)",
    )
    organization: Optional[str] = Field(
        default=None,
        description="Organization or account (required for Azure DevOps)",
    )


class IssueSourceConfig(BaseModel):
    """Settings for one issue tracker."""

    model_config = ConfigDict(extra="forbid")

    token: Optional[str] = Field(default=None, description="Access token (secret)")
    base_url: Optional[str] = Field(
        default=None,
        description="API base URL. Required for Jira (your-org.atlassian.net), "
        "self-hosted GitLab, Azure DevOps, etc.",
    )
    user: Optional[str] = Field(
        default=None,
        description="Username/email (required for Jira basic auth)",
    )


class DefaultsConfig(BaseModel):
    """User-level defaults that apply when not overridden by project config."""

    model_config = ConfigDict(extra="forbid")

    llm_provider: str = Field(default="anthropic", description="Default LLM provider")
    vcs_provider: str = Field(default="github", description="Default VCS provider")
    issue_provider: Optional[str] = Field(
        default=None, description="Default issue tracker (none until configured)"
    )


class UserConfig(BaseModel):
    """The full ~/.config/cascade/config.yaml structure."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    llm_providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    vcs_providers: dict[str, VCSProviderConfig] = Field(default_factory=dict)
    issue_sources: dict[str, IssueSourceConfig] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------------


def load_user_config(path: Optional[Path] = None) -> UserConfig:
    """Load the user config, returning an empty default if no file exists.

    Args:
        path: Explicit path to the config file. If None, uses config_path().

    Returns:
        A validated UserConfig.

    Raises:
        CascadeConfigError: If the file exists but is malformed.
    """
    if path is None:
        path = config_path()
    if not path.exists():
        return UserConfig()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CascadeConfigError(f"Could not read user config {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise CascadeConfigError(f"Invalid YAML in user config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CascadeConfigError(
            f"User config {path} must be a YAML mapping at the top level"
        )
    try:
        return UserConfig.model_validate(data)
    except ValidationError as exc:
        raise CascadeConfigError(f"Invalid user config at {path}:\n{exc}") from exc


def save_user_config(config: UserConfig, path: Optional[Path] = None) -> Path:
    """Persist the user config to disk with restrictive permissions.

    Writes atomically (temp-then-rename) and chmods the file to 0600
    so credentials aren't world-readable.

    Returns the path that was written.
    """
    if path is None:
        path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = config.model_dump(mode="json", exclude_none=False)
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    tmp.replace(path)
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        # On Windows or quirky filesystems chmod may not work; not fatal.
        pass
    return path


# ----------------------------------------------------------------------------
# Credential resolution (the precedence engine)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedLLMCredentials:
    """Effective LLM credentials after applying precedence rules."""

    provider: str
    api_key: Optional[str]  # may be None for providers that don't need one
    model: Optional[str]  # may be None -> use the client's default
    base_url: Optional[str]
    source: str  # description of where this came from (for diagnostics)


@dataclass(frozen=True)
class ResolvedVCSCredentials:
    provider: str
    token: Optional[str]
    base_url: Optional[str]
    organization: Optional[str]
    source: str


@dataclass(frozen=True)
class ResolvedIssueCredentials:
    provider: str
    token: Optional[str]
    base_url: Optional[str]
    user: Optional[str]
    source: str


# Env-var names per provider. Keep this list canonical.
LLM_ENV_KEY_NAMES: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "claude_code": (),  # uses local Claude Code subscription; no API key
    "copilot": (),  # uses local Copilot subscription; no API key
    "ollama": (),  # local self-hosted; no API key
}

VCS_ENV_TOKEN_NAMES: dict[str, tuple[str, ...]] = {
    "github": ("GITHUB_TOKEN", "GH_TOKEN"),
    "gitlab": ("GITLAB_TOKEN",),
    "bitbucket": ("BITBUCKET_TOKEN",),
    "azure_devops": ("AZURE_DEVOPS_TOKEN", "AZURE_DEVOPS_EXT_PAT"),
}

ISSUE_ENV_TOKEN_NAMES: dict[str, tuple[str, ...]] = {
    "github": ("GITHUB_TOKEN", "GH_TOKEN"),
    "jira": ("JIRA_API_TOKEN",),
    "azure_devops": ("AZURE_DEVOPS_TOKEN",),
    "linear": ("LINEAR_API_KEY",),
    "gitlab": ("GITLAB_TOKEN",),
}


def resolve_llm_credentials(
    *,
    user_config: UserConfig,
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> ResolvedLLMCredentials:
    """Resolve LLM credentials by applying the precedence rules.

    Args:
        user_config: The loaded UserConfig (may be empty).
        provider: Explicit provider name. Else uses user_config.defaults.llm_provider.
        model_override: CLI-supplied model override.
        api_key_override: CLI-supplied API key override (rare; mostly for CI).

    Returns:
        ResolvedLLMCredentials. `api_key` may be None for providers that
        don't require one (Claude Code, Copilot, Ollama).

    Raises:
        CascadeConfigError: If the chosen provider requires an API key and
            none was found in any source.
    """
    resolved_provider = provider or user_config.defaults.llm_provider
    provider_cfg = user_config.llm_providers.get(resolved_provider, LLMProviderConfig())

    # API key precedence: override -> user config -> env vars
    key = api_key_override or provider_cfg.api_key
    source = "CLI override" if api_key_override else (
        "user config" if provider_cfg.api_key else None
    )
    if key is None:
        for env_name in LLM_ENV_KEY_NAMES.get(resolved_provider, ()):
            env_val = os.environ.get(env_name)
            if env_val:
                key = env_val
                source = f"env var {env_name}"
                break

    needs_key = bool(LLM_ENV_KEY_NAMES.get(resolved_provider))
    if needs_key and not key:
        env_hint = " or ".join(LLM_ENV_KEY_NAMES.get(resolved_provider, ()))
        raise CascadeConfigError(
            f"No API key configured for LLM provider '{resolved_provider}'. "
            f"Run: cascade configure llm {resolved_provider} --key <KEY>  "
            f"OR set environment variable {env_hint}."
        )

    model = model_override or provider_cfg.default_model
    return ResolvedLLMCredentials(
        provider=resolved_provider,
        api_key=key,
        model=model,
        base_url=provider_cfg.base_url,
        source=source or "(no key needed for this provider)",
    )


def resolve_vcs_credentials(
    *,
    user_config: UserConfig,
    provider: Optional[str] = None,
    token_override: Optional[str] = None,
) -> ResolvedVCSCredentials:
    """Resolve VCS credentials by applying the precedence rules."""
    resolved = provider or user_config.defaults.vcs_provider
    cfg = user_config.vcs_providers.get(resolved, VCSProviderConfig())

    token = token_override or cfg.token
    source = "CLI override" if token_override else (
        "user config" if cfg.token else None
    )
    if token is None:
        for env_name in VCS_ENV_TOKEN_NAMES.get(resolved, ()):
            env_val = os.environ.get(env_name)
            if env_val:
                token = env_val
                source = f"env var {env_name}"
                break

    if not token:
        env_hint = " or ".join(VCS_ENV_TOKEN_NAMES.get(resolved, ())) or "<set in cascade configure>"
        raise CascadeConfigError(
            f"No token configured for VCS provider '{resolved}'. "
            f"Run: cascade configure vcs {resolved} --token <TOKEN>  "
            f"OR set environment variable {env_hint}."
        )

    return ResolvedVCSCredentials(
        provider=resolved,
        token=token,
        base_url=cfg.base_url,
        organization=cfg.organization,
        source=source or "user config",
    )


def resolve_issue_credentials(
    *,
    user_config: UserConfig,
    provider: str,
    token_override: Optional[str] = None,
) -> ResolvedIssueCredentials:
    """Resolve issue tracker credentials. Provider is always explicit
    here (you ask for tickets from a specific tracker)."""
    cfg = user_config.issue_sources.get(provider, IssueSourceConfig())

    token = token_override or cfg.token
    source = "CLI override" if token_override else (
        "user config" if cfg.token else None
    )
    if token is None:
        for env_name in ISSUE_ENV_TOKEN_NAMES.get(provider, ()):
            env_val = os.environ.get(env_name)
            if env_val:
                token = env_val
                source = f"env var {env_name}"
                break

    if not token:
        env_hint = " or ".join(ISSUE_ENV_TOKEN_NAMES.get(provider, ())) or "<set in cascade configure>"
        raise CascadeConfigError(
            f"No token configured for issue source '{provider}'. "
            f"Run: cascade configure issue {provider} --token <TOKEN>  "
            f"OR set environment variable {env_hint}."
        )

    return ResolvedIssueCredentials(
        provider=provider,
        token=token,
        base_url=cfg.base_url,
        user=cfg.user,
        source=source or "user config",
    )


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------


def mask_secret(value: Optional[str], *, visible: int = 4) -> str:
    """Render a secret for display, exposing only the last few characters."""
    if not value:
        return "(not set)"
    if len(value) <= visible:
        return "*" * len(value)
    return "*" * (len(value) - visible) + value[-visible:]
