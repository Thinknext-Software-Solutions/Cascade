"""Ollama / self-hosted LLM provider (OpenAI-compatible).

Ollama, vLLM, LM Studio, and other local model runners expose an
OpenAI-compatible API at /v1. This client just wraps the OpenAIClient
with a sensible default base_url pointing at the standard Ollama port.

Usage:
    cascade configure llm ollama --model llama3 --base-url http://localhost:11434/v1
    cascade configure llm ollama --set-default

Or directly:
    OllamaClient(model="llama3", base_url="http://localhost:11434/v1")

No API key is required for Ollama (we pass a dummy "ollama" string to
satisfy the OpenAI SDK's auth check).
"""

from __future__ import annotations

import logging
from typing import Optional, TypeVar

from pydantic import BaseModel

from .llm import LLMClient, LLMResponse
from .llm_openai import OpenAIClient


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaClient(LLMClient):
    """LLM client for Ollama / self-hosted models exposing the OpenAI API.

    Delegates everything to OpenAIClient with a different default base URL
    and a dummy API key.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_MODEL = "llama3.1"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 300.0,
    ):
        self._delegate = OpenAIClient(
            api_key=api_key or "ollama",  # local servers ignore this; SDK requires non-empty
            model=model or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout_seconds=timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._delegate.model

    def structured_call(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> LLMResponse[T]:
        # Re-tag the response as coming from "ollama" provider for cost tracking.
        response = self._delegate.structured_call(
            system=system,
            user=user,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # The LLMUsage from OpenAIClient says provider="openai"; rewrite it.
        from .llm import LLMUsage

        rewritten_usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.usage.model,
            provider="ollama",
        )
        return LLMResponse(
            parsed=response.parsed,
            raw_text=response.raw_text,
            usage=rewritten_usage,
        )
