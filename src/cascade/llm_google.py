"""Google Gemini LLM provider.

Uses Gemini's JSON mode with a response_schema for structured output.
Compatible with Gemini 2.x and later.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import CascadeLLMError
from .llm import LLMClient, LLMResponse, LLMUsage


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_GoogleSDK = None


def _load_google_sdk():
    global _GoogleSDK
    if _GoogleSDK is None:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeLLMError(
                "google-genai SDK not installed. Run: pip install google-genai"
            ) from exc
        _GoogleSDK = genai
    return _GoogleSDK


class GoogleGeminiClient(LLMClient):
    """Google Gemini implementation using response_schema for structured output."""

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        *,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ):
        genai = _load_google_sdk()
        # The new google-genai SDK takes api_key directly in Client()
        kwargs: dict = {"api_key": api_key}
        if base_url:
            # Vertex / custom endpoint support
            kwargs["http_options"] = {"base_url": base_url}
        self._client = genai.Client(**kwargs)
        self._model = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "google"

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
        # Gemini supports Pydantic models directly as response_schema in the
        # newer SDK; we use the model class itself for cleanest behavior.
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
        except Exception as exc:
            raise CascadeLLMError(
                f"Google API call failed for model {self._model}: {exc}"
            ) from exc

        # Gemini's response.text contains the JSON string. response.parsed (if
        # available) contains a parsed instance, but the format depends on
        # the SDK version, so we re-validate via Pydantic for safety.
        raw_text = getattr(response, "text", "") or ""
        if not raw_text:
            raise CascadeLLMError(
                f"Gemini returned empty text for {schema.__name__}"
            )

        try:
            parsed_json = json.loads(raw_text)
        except ValueError as exc:
            raise CascadeLLMError(
                f"Failed to parse Gemini response as JSON: {exc}"
            ) from exc

        try:
            parsed = schema.model_validate(parsed_json)
        except ValidationError as exc:
            truncated = raw_text[:500]
            raise CascadeLLMError(
                f"Gemini output did not match schema {schema.__name__}: {exc}\n"
                f"Raw output (truncated): {truncated}"
            ) from exc

        # Token counting from usage_metadata
        usage_meta = getattr(response, "usage_metadata", None)
        usage = LLMUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
            model=self._model,
            provider=self.provider_name,
        )

        return LLMResponse(parsed=parsed, raw_text=raw_text, usage=usage)
