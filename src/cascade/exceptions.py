"""Custom exceptions for Cascade.

All Cascade-raised exceptions inherit from CascadeError so callers can
catch the whole family with one except clause.
"""


class CascadeError(Exception):
    """Base exception for all Cascade errors."""


class CascadeConfigError(CascadeError):
    """Raised when cascade.yaml is missing, malformed, or has invalid values."""


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
    """Raised when git or GitHub operations fail."""
