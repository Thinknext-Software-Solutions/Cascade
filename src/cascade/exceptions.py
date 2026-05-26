"""Custom exceptions for Cascade.

All Cascade-raised exceptions inherit from CascadeError so callers can
catch the whole family with one except clause.

Every CascadeError optionally carries a `hint` (one or more concrete fixes
the user can try) and a `learn_more` pointer (a CLI command or URL). The
CLI formats these into a multi-line response instead of a single bare line.

Pattern:

    raise CascadeConfigError(
        "Invalid YAML in cascade.yaml",
        hint=[
            "Check for unclosed quotes or bad indentation",
            "Run: cascade init --force  to regenerate with defaults",
        ],
        learn_more="cascade doctor",
    )
"""

from __future__ import annotations

from typing import Optional, Union


HintType = Union[str, list[str], None]


class CascadeError(Exception):
    """Base exception for all Cascade errors.

    `hint` is a single string or a list of strings, each one a concrete
    suggestion the user can try. The CLI renders them as bullet points.

    `learn_more` is a CLI command (e.g. "cascade doctor") or URL that
    points to broader documentation about the failure mode.
    """

    def __init__(
        self,
        message: str,
        *,
        hint: HintType = None,
        learn_more: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.learn_more = learn_more

    @property
    def hints(self) -> list[str]:
        """Always return hints as a list (empty if none)."""
        if self.hint is None:
            return []
        if isinstance(self.hint, str):
            return [self.hint]
        return list(self.hint)


class CascadeConfigError(CascadeError):
    """Raised when cascade.yaml or ~/.config/cascade/config.yaml is
    missing, malformed, or has invalid values."""


class CascadeMemoryError(CascadeError):
    """Raised when team-memory/ files can't be read or are malformed."""


class CascadeLLMError(CascadeError):
    """Raised when the LLM provider returns an error, times out, or returns
    output that can't be parsed into the expected schema."""


class CascadeTranscriptionError(CascadeError):
    """Raised when audio transcription fails (file not found, unsupported
    format, Whisper model error, etc.)."""


class CascadeExtractionError(CascadeError):
    """Raised when story extraction fails to produce a usable result."""


class CascadeRepoError(CascadeError):
    """Raised when git or GitHub/GitLab/Bitbucket/Azure DevOps operations fail."""
