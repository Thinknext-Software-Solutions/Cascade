"""OpenAI LLM provider.

Uses OpenAI's Structured Outputs (response_format with json_schema) to get
schema-conformant JSON, then validates into the requested Pydantic model.
Works for any OpenAI-compatible API endpoint (set base_url to point at
Azure OpenAI, OpenRouter, vLLM, etc.).
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

_OpenAISDK = None


def _load_openai_sdk():
    global _OpenAISDK
    if _OpenAISDK is None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise CascadeLLMError(
                "openai SDK not installed. Run: pip install openai"
            ) from exc
        _OpenAISDK = openai
    return _OpenAISDK


class OpenAIClient(LLMClient):
    """OpenAI Structured Outputs implementation.

    The OpenAI Structured Outputs feature (gpt-4o-2024-08-06 and later) guarantees
    the model returns JSON matching the supplied schema. We use it via the
    `response_format` parameter with `type: "json_schema"`.
    """

    DEFAULT_MODEL = "gpt-5"

    def __init__(
        self,
        *,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ):
        openai = _load_openai_sdk()
        kwargs: dict = {"api_key": api_key, "timeout": timeout_seconds}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._model = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "openai"

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
        # OpenAI Structured Outputs uses JSON Schema with some restrictions.
        # Pydantic's model_json_schema() produces a compatible schema for most
        # cases; the OpenAI SDK normalizes additional metadata.
        json_schema = schema.model_json_schema()
        schema_name = _to_schema_name(schema.__name__)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": json_schema,
                        "strict": False,  # strict mode rejects many valid Pydantic schemas
                    },
                },
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise CascadeLLMError(
                f"OpenAI API call failed for model {self._model}: {exc}"
            ) from exc

        try:
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                raise CascadeLLMError(
                    f"OpenAI returned empty content for {schema.__name__}"
                )
            parsed_json = json.loads(content)
        except (KeyError, IndexError, ValueError) as exc:
            raise CascadeLLMError(
                f"Failed to parse OpenAI response as JSON: {exc}"
            ) from exc

        try:
            parsed = schema.model_validate(parsed_json)
        except ValidationError as exc:
            truncated = json.dumps(parsed_json)[:500]
            raise CascadeLLMError(
                f"OpenAI output did not match schema {schema.__name__}: {exc}\n"
                f"Raw output (truncated): {truncated}"
            ) from exc

        usage_obj = getattr(response, "usage", None)
        usage = LLMUsage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
            model=self._model,
            provider=self.provider_name,
        )

        return LLMResponse(parsed=parsed, raw_text=content, usage=usage)


def _to_schema_name(class_name: str) -> str:
    """Convert PascalCase to snake_case (OpenAI's preferred schema name format)."""
    result = []
    for i, ch in enumerate(class_name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)
