"""Tests for cascade.llm_google (mocked SDK)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from cascade.exceptions import CascadeLLMError
from cascade.llm import LLMResponse


class Sample(BaseModel):
    answer: str = Field(..., min_length=1)
    confidence: int = Field(..., ge=0, le=100)


def _fake_gemini_response(text: str, *, prompt_tokens=10, output_tokens=20):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
        ),
    )


@pytest.fixture
def fake_genai_sdk():
    sdk = MagicMock()
    sdk.Client = MagicMock()
    with patch("cascade.llm_google._GoogleSDK", sdk), patch(
        "cascade.llm_google._load_google_sdk", return_value=sdk
    ):
        yield sdk


class TestGoogleGeminiClient:
    def test_initializes_with_api_key(self, fake_genai_sdk):
        from cascade.llm_google import GoogleGeminiClient

        GoogleGeminiClient(api_key="g-x")
        fake_genai_sdk.Client.assert_called_once()
        kwargs = fake_genai_sdk.Client.call_args.kwargs
        assert kwargs["api_key"] == "g-x"

    def test_default_model(self, fake_genai_sdk):
        from cascade.llm_google import GoogleGeminiClient

        c = GoogleGeminiClient(api_key="g-x")
        assert c.model == GoogleGeminiClient.DEFAULT_MODEL
        assert c.provider_name == "google"

    def test_structured_call_returns_parsed(self, fake_genai_sdk):
        from cascade.llm_google import GoogleGeminiClient

        instance = fake_genai_sdk.Client.return_value
        instance.models.generate_content.return_value = _fake_gemini_response(
            json.dumps({"answer": "yes", "confidence": 80}),
            prompt_tokens=100,
            output_tokens=50,
        )
        c = GoogleGeminiClient(api_key="g-x", model="gemini-test")
        out = c.structured_call(system="be helpful", user="please", schema=Sample)
        assert isinstance(out, LLMResponse)
        assert isinstance(out.parsed, Sample)
        assert out.parsed.answer == "yes"
        assert out.usage.input_tokens == 100
        assert out.usage.output_tokens == 50
        assert out.usage.provider == "google"

    def test_structured_call_passes_response_schema(self, fake_genai_sdk):
        from cascade.llm_google import GoogleGeminiClient

        instance = fake_genai_sdk.Client.return_value
        instance.models.generate_content.return_value = _fake_gemini_response(
            json.dumps({"answer": "ok", "confidence": 50})
        )
        c = GoogleGeminiClient(api_key="g-x")
        c.structured_call(system="s", user="u", schema=Sample)
        kwargs = instance.models.generate_content.call_args.kwargs
        assert kwargs["config"]["response_mime_type"] == "application/json"
        assert kwargs["config"]["response_schema"] is Sample
        assert kwargs["config"]["system_instruction"] == "s"

    def test_api_error_wrapped(self, fake_genai_sdk):
        from cascade.llm_google import GoogleGeminiClient

        instance = fake_genai_sdk.Client.return_value
        instance.models.generate_content.side_effect = RuntimeError("quota exceeded")
        c = GoogleGeminiClient(api_key="g-x")
        with pytest.raises(CascadeLLMError, match="Google API call failed"):
            c.structured_call(system="s", user="u", schema=Sample)

    def test_empty_text_raises(self, fake_genai_sdk):
        from cascade.llm_google import GoogleGeminiClient

        instance = fake_genai_sdk.Client.return_value
        instance.models.generate_content.return_value = _fake_gemini_response("")
        c = GoogleGeminiClient(api_key="g-x")
        with pytest.raises(CascadeLLMError, match="empty text"):
            c.structured_call(system="s", user="u", schema=Sample)

    def test_invalid_json_wrapped(self, fake_genai_sdk):
        from cascade.llm_google import GoogleGeminiClient

        instance = fake_genai_sdk.Client.return_value
        instance.models.generate_content.return_value = _fake_gemini_response(
            "not valid json at all"
        )
        c = GoogleGeminiClient(api_key="g-x")
        with pytest.raises(CascadeLLMError, match="Failed to parse Gemini"):
            c.structured_call(system="s", user="u", schema=Sample)


class TestFactoryDispatch:
    def test_google_dispatched_correctly(self, fake_genai_sdk):
        from cascade.llm import build_client

        client = build_client("google", api_key="g-x", model="gemini-test")
        assert client.provider_name == "google"
        assert client.model == "gemini-test"

    def test_google_requires_api_key(self):
        from cascade.llm import build_client

        with pytest.raises(CascadeLLMError, match="requires an API key"):
            build_client("google")
