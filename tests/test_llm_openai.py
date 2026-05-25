"""Tests for cascade.llm_openai (mocked SDK)."""

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


def _fake_chat_response(json_string: str, input_tokens=10, output_tokens=20):
    """Build a SimpleNamespace mimicking an OpenAI chat completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json_string),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens, completion_tokens=output_tokens
        ),
    )


@pytest.fixture
def fake_openai_sdk():
    sdk = MagicMock()
    sdk.OpenAI = MagicMock()
    with patch("cascade.llm_openai._OpenAISDK", sdk), patch(
        "cascade.llm_openai._load_openai_sdk", return_value=sdk
    ):
        yield sdk


class TestOpenAIClient:
    def test_initializes_with_api_key(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        OpenAIClient(api_key="sk-x")
        fake_openai_sdk.OpenAI.assert_called_once()
        kwargs = fake_openai_sdk.OpenAI.call_args.kwargs
        assert kwargs["api_key"] == "sk-x"

    def test_passes_base_url_when_set(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        OpenAIClient(api_key="sk-x", base_url="https://my.gateway/v1")
        kwargs = fake_openai_sdk.OpenAI.call_args.kwargs
        assert kwargs["base_url"] == "https://my.gateway/v1"

    def test_default_model(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        c = OpenAIClient(api_key="sk-x")
        assert c.model == OpenAIClient.DEFAULT_MODEL
        assert c.provider_name == "openai"

    def test_structured_call_returns_parsed(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        instance = fake_openai_sdk.OpenAI.return_value
        instance.chat.completions.create.return_value = _fake_chat_response(
            json.dumps({"answer": "yes", "confidence": 80}),
            input_tokens=100,
            output_tokens=50,
        )
        c = OpenAIClient(api_key="sk-x", model="gpt-test")
        out = c.structured_call(system="be helpful", user="please", schema=Sample)
        assert isinstance(out, LLMResponse)
        assert isinstance(out.parsed, Sample)
        assert out.parsed.answer == "yes"
        assert out.usage.input_tokens == 100
        assert out.usage.output_tokens == 50
        assert out.usage.provider == "openai"

    def test_structured_call_passes_json_schema(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        instance = fake_openai_sdk.OpenAI.return_value
        instance.chat.completions.create.return_value = _fake_chat_response(
            json.dumps({"answer": "ok", "confidence": 50})
        )
        c = OpenAIClient(api_key="sk-x")
        c.structured_call(system="s", user="u", schema=Sample)
        kwargs = instance.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "sample"
        assert "properties" in kwargs["response_format"]["json_schema"]["schema"]

    def test_api_error_wrapped(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        instance = fake_openai_sdk.OpenAI.return_value
        instance.chat.completions.create.side_effect = RuntimeError("rate limit")
        c = OpenAIClient(api_key="sk-x")
        with pytest.raises(CascadeLLMError, match="OpenAI API call failed"):
            c.structured_call(system="s", user="u", schema=Sample)

    def test_empty_content_raises(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        instance = fake_openai_sdk.OpenAI.return_value
        instance.chat.completions.create.return_value = _fake_chat_response("")
        c = OpenAIClient(api_key="sk-x")
        with pytest.raises(CascadeLLMError, match="empty content"):
            c.structured_call(system="s", user="u", schema=Sample)

    def test_invalid_json_wrapped(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        instance = fake_openai_sdk.OpenAI.return_value
        instance.chat.completions.create.return_value = _fake_chat_response(
            "not valid json"
        )
        c = OpenAIClient(api_key="sk-x")
        with pytest.raises(CascadeLLMError, match="Failed to parse OpenAI"):
            c.structured_call(system="s", user="u", schema=Sample)

    def test_schema_validation_failure_wrapped(self, fake_openai_sdk):
        from cascade.llm_openai import OpenAIClient

        instance = fake_openai_sdk.OpenAI.return_value
        instance.chat.completions.create.return_value = _fake_chat_response(
            json.dumps({"answer": "ok", "confidence": 999})  # out of range
        )
        c = OpenAIClient(api_key="sk-x")
        with pytest.raises(CascadeLLMError, match="did not match schema"):
            c.structured_call(system="s", user="u", schema=Sample)


class TestFactoryDispatch:
    def test_openai_dispatched_correctly(self, fake_openai_sdk):
        from cascade.llm import build_client

        client = build_client("openai", api_key="sk-x", model="gpt-test")
        assert client.provider_name == "openai"
        assert client.model == "gpt-test"

    def test_openai_requires_api_key(self):
        from cascade.llm import build_client

        with pytest.raises(CascadeLLMError, match="requires an API key"):
            build_client("openai")

    def test_unknown_provider_lists_supported(self):
        from cascade.llm import build_client

        with pytest.raises(CascadeLLMError, match="Supported:"):
            build_client("cohere")
