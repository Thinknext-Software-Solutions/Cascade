"""Configuration loading and validation for Cascade.

Reads cascade.yaml from the repo root. Provides typed access via the
CascadeConfig pydantic model. Failures are CascadeConfigError with
clear messages so users know exactly what to fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .exceptions import CascadeConfigError


DEFAULT_CONFIG_FILENAME = "cascade.yaml"


class AgentConfig(BaseModel):
    """LLM agent settings."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="anthropic", description="LLM provider name")
    model: str = Field(
        default="claude-opus-4-7", description="Model identifier for the provider"
    )
    max_iterations: int = Field(
        default=1,
        ge=0,
        le=5,
        description="How many review-feedback iteration rounds the agent will run "
        "before giving up. 0 = no iteration (open PR and done).",
    )
    temperature: float = Field(default=0.2, ge=0, le=2)


class MemoryConfig(BaseModel):
    """Team memory layer settings."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(
        default="team-memory",
        description="Relative path (from repo root) to the team-memory directory",
    )
    max_chars_per_call: int = Field(
        default=20_000,
        ge=1_000,
        le=200_000,
        description="Cap on team-memory chars included in any single LLM call "
        "to avoid blowing the context window",
    )


class PathsConfig(BaseModel):
    """File-path allow/deny lists for what the agent may modify."""

    model_config = ConfigDict(frozen=True)

    allowed: list[str] = Field(
        default_factory=lambda: ["src/**", "tests/**", "docs/**"],
        description="Glob patterns. Agent may write to matching paths.",
    )
    disallowed: list[str] = Field(
        default_factory=lambda: [
            ".github/**",
            "migrations/**",
            "cascade.yaml",
            "team-memory/**",
        ],
        description="Glob patterns. Agent may NEVER write to these, even if "
        "they also match an allowed pattern. Deny wins over allow.",
    )


class CascadeConfig(BaseModel):
    """Top-level config loaded from cascade.yaml."""

    model_config = ConfigDict(frozen=True)

    version: int = Field(default=1, description="Config schema version")
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    language: Optional[str] = Field(
        default=None,
        description="Explicit language override. If None, Cascade auto-detects "
        "from marker files in the repo root. Supported: python, typescript, "
        "javascript, go, rust, java, ruby, csharp.",
    )
    test_command: Optional[str] = Field(
        default=None,
        description="Override the language profile's default test command. "
        "If None, Cascade uses the language profile's test_command.",
    )


def load_config(path: Optional[Path] = None) -> CascadeConfig:
    """Load cascade.yaml from the given path or the current working directory.

    Args:
        path: Explicit path to a YAML file. If None, looks for ./cascade.yaml.

    Returns:
        A validated CascadeConfig.

    Raises:
        CascadeConfigError: If the file is missing, unreadable, malformed,
            or fails validation. The error message names what's wrong.
    """
    if path is None:
        path = Path.cwd() / DEFAULT_CONFIG_FILENAME

    if not path.exists():
        # Missing config is OK -- return defaults. Users can opt in to config.
        return CascadeConfig()

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CascadeConfigError(f"Could not read {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise CascadeConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise CascadeConfigError(
            f"Top level of {path} must be a YAML mapping, got {type(data).__name__}"
        )

    try:
        return CascadeConfig.model_validate(data)
    except ValidationError as exc:
        raise CascadeConfigError(f"Invalid config in {path}:\n{exc}") from exc
