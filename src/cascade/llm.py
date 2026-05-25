"""LLM provider abstraction.

Defines a minimal, provider-agnostic interface (`LLMClient`) so the rest of
Cascade can call LLMs without knowing which vendor or model is in use.
The Anthropic implementation is the only one shipped in v0.1; OpenAI,
Ollama, and others slot in as additional implementations later.

Design rules:
- Synchronous calls in v0.1 (async in v0.2 if needed)
- Structured output via Pydantic schemas (we don't parse free-form JSON)
- Every call returns the parsed model + raw response for debuggability
- Errors are wrapped in CascadeLLMError with provider/model context
- No retries here -- that's a higher-level concern (the orchestrator decides
  whether to retry, escalate, or fail)
- Token usage exposed via LLMUsage for cost tracking later
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import CascadeLLMError


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMUsage:
    """Token usage from a single LLM call."""

    input_tokens: int
    output_tokens: int
    model: str
    provider: str


@dataclass(frozen=True)
class LLMResponse(Generic[T]):
    """Result of a structured LLM call.

    `parsed` is the Pydantic model instance the LLM was instructed to produce.
    `raw_text` is the underlying text/JSON for debugging.
    """

    parsed: T
    raw_text: str
    usage: LLMUsage


# ----------------------------------------------------------------------------
# Abstract interface
# ----------------------------------------------------------------------------


class LLMClient(ABC):
    """Provider-agnostic LLM interface.

    Implementations must support `structured_call`: given a prompt and a
    Pydantic schema, return an instance of that schema. How they achieve
    that (tool use, JSON mode, structured outputs API) is implementation-
    specific.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def structured_call(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> LLMResponse[T]:
        """Call the LLM with a structured-output target.

        Args:
            system: System prompt setting the role/behavior.
            user: User message containing the task input.
            schema: Pydantic model class the LLM must produce.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the parsed schema instance.

        Raises:
            CascadeLLMError: API failure, timeout, or output that can't be
                parsed into `schema`. The error message includes provider,
                model, and (truncated) raw output for diagnosis.
        """


# ----------------------------------------------------------------------------
# Anthropic implementation
# ----------------------------------------------------------------------------


# Imported lazily inside AnthropicClient so test suites that don't exercise
# the Anthropic path don't need the SDK installed.
_AnthropicSDK = None


def _load_anthropic_sdk():
    global _AnthropicSDK
    if _AnthropicSDK is None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover -- exercised in integration
            raise CascadeLLMError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc
        _AnthropicSDK = anthropic
    return _AnthropicSDK


class AnthropicClient(LLMClient):
    """Anthropic Claude implementation using tool use for structured output.

    Tool use is more reliable than JSON mode for forcing schema compliance:
    we expose a single tool whose input_schema is our Pydantic schema's
    JSON schema, then force the model to call it. Whatever shows up in
    `tool_use.input` is guaranteed to match the schema (or the SDK errors).
    """

    DEFAULT_MODEL = "claude-opus-4-7"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ):
        anthropic = _load_anthropic_sdk()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise CascadeLLMError(
                "No Anthropic API key provided. Pass api_key= or set "
                "ANTHROPIC_API_KEY environment variable."
            )
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout_seconds)
        self._model = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def structured_call(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> LLMResponse[T]:
        tool_name = _schema_to_tool_name(schema)
        json_schema = _pydantic_schema_to_tool_input_schema(schema)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=[
                    {
                        "name": tool_name,
                        "description": (
                            schema.__doc__
                            or f"Return a structured {schema.__name__}."
                        ).strip(),
                        "input_schema": json_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise CascadeLLMError(
                f"Anthropic API call failed for model {self._model}: {exc}"
            ) from exc

        tool_input = _extract_tool_input(response, tool_name)

        try:
            parsed = schema.model_validate(tool_input)
        except ValidationError as exc:
            truncated = json.dumps(tool_input)[:500]
            raise CascadeLLMError(
                f"LLM output did not match schema {schema.__name__}: {exc}\n"
                f"Raw input (truncated): {truncated}"
            ) from exc

        usage = LLMUsage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            model=self._model,
            provider=self.provider_name,
        )

        return LLMResponse(
            parsed=parsed, raw_text=json.dumps(tool_input), usage=usage
        )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _schema_to_tool_name(schema: type[BaseModel]) -> str:
    """Convert PascalCase model name to snake_case tool name."""
    name = schema.__name__
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "submit_" + "".join(result)


def _pydantic_schema_to_tool_input_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic v2 schema into a JSON schema suitable for Anthropic
    tool definitions. Strips $defs/$ref into inlined form when possible."""
    return schema.model_json_schema()


def _extract_tool_input(response: Any, expected_tool_name: str) -> dict[str, Any]:
    """Pull the tool_use input dict from an Anthropic response.

    Anthropic returns content blocks; we want the one of type 'tool_use'
    with the matching name.
    """
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use":
            if getattr(block, "name", None) == expected_tool_name:
                tool_input = getattr(block, "input", None)
                if isinstance(tool_input, dict):
                    return tool_input
    raise CascadeLLMError(
        f"Anthropic response did not contain expected tool_use '{expected_tool_name}'. "
        f"Got blocks: {[getattr(b, 'type', '?') for b in getattr(response, 'content', [])]}"
    )


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "anthropic",
    "openai",
    "google",
    "claude_code",
    "ollama",
)


def build_client(
    provider: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """Construct an LLMClient by provider name.

    Args:
        provider: One of SUPPORTED_PROVIDERS.
        model: Specific model identifier, or None for the provider default.
        api_key: API key. Most providers require this; claude_code does not.
        base_url: Optional API endpoint override.

    Returns:
        An LLMClient ready to use.

    Raises:
        CascadeLLMError: Unknown provider or initialization failure.
    """
    p = provider.lower()
    if p == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    if p == "openai":
        # Lazy import to avoid pulling in the openai SDK for users who don't need it
        from .llm_openai import OpenAIClient

        if api_key is None:
            raise CascadeLLMError("OpenAI provider requires an API key")
        return OpenAIClient(api_key=api_key, model=model, base_url=base_url)
    if p == "google":
        from .llm_google import GoogleGeminiClient

        if api_key is None:
            raise CascadeLLMError("Google Gemini provider requires an API key")
        return GoogleGeminiClient(api_key=api_key, model=model, base_url=base_url)
    if p == "claude_code":
        from .llm_claude_code import ClaudeCodeClient

        # claude_code uses local subscription; no api_key
        return ClaudeCodeClient(model=model)
    if p == "ollama":
        from .llm_ollama import OllamaClient

        return OllamaClient(model=model, base_url=base_url, api_key=api_key)
    raise CascadeLLMError(
        f"Unknown LLM provider '{provider}'. Supported: "
        f"{', '.join(SUPPORTED_PROVIDERS)}."
    )


def build_client_from_credentials(creds) -> LLMClient:
    """Build an LLMClient from a ResolvedLLMCredentials object.

    Convenience wrapper that hides the dispatch on provider name. Accepts
    a `user_config.ResolvedLLMCredentials`.
    """
    return build_client(
        provider=creds.provider,
        model=creds.model,
        api_key=creds.api_key,
        base_url=creds.base_url,
    )
