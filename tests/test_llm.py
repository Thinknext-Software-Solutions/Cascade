"""Tests for cascade.llm.

We test the LLMClient interface and the Anthropic-specific helpers WITHOUT
making real API calls. Integration tests that actually hit Anthropic live
in tests/integration/ and are gated behind an env var.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from cascade.exceptions import CascadeLLMError
from cascade.llm import (
    AnthropicClient,
    LLMResponse,
    LLMUsage,
    _extract_tool_input,
    _pydantic_schema_to_tool_input_schema,
    _schema_to_tool_name,
    build_client,
)


class SampleOutput(BaseModel):
    """A sample tool output schema for tests."""

    answer: str = Field(..., min_length=1)
    confidence: int = Field(..., ge=0, le=100)


def _fake_anthropic_response(tool_name: str, tool_input: dict, input_tokens=10, output_tokens=20):
    """Build a SimpleNamespace mimicking the shape of an Anthropic SDK response."""
    block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage)


# --------- Helpers ----------


class TestHelpers:
    def test_schema_to_tool_name_snake_cases(self):
        class FooBar(BaseModel):
            x: int

        assert _schema_to_tool_name(FooBar) == "submit_foo_bar"

    def test_schema_to_tool_name_handles_single_word(self):
        class Story(BaseModel):
            x: int

        assert _schema_to_tool_name(Story) == "submit_story"

    def test_pydantic_schema_conversion_includes_required_fields(self):
        js = _pydantic_schema_to_tool_input_schema(SampleOutput)
        assert "properties" in js
        assert "answer" in js["properties"]
        assert "confidence" in js["properties"]

    def test_extract_tool_input_returns_dict(self):
        resp = _fake_anthropic_response("submit_x", {"a": 1})
        out = _extract_tool_input(resp, "submit_x")
        assert out == {"a": 1}

    def test_extract_tool_input_raises_on_wrong_name(self):
        resp = _fake_anthropic_response("submit_other", {"a": 1})
        with pytest.raises(CascadeLLMError, match="did not contain expected tool_use"):
            _extract_tool_input(resp, "submit_x")

    def test_extract_tool_input_raises_when_no_tool_use(self):
        resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
        with pytest.raises(CascadeLLMError, match="did not contain"):
            _extract_tool_input(resp, "submit_x")


# --------- AnthropicClient ----------


@pytest.fixture
def fake_sdk():
    """Patch the lazy-loaded anthropic SDK with a MagicMock and yield it."""
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic = MagicMock()
    with patch("cascade.llm._AnthropicSDK", mock_anthropic):
        with patch("cascade.llm._load_anthropic_sdk", return_value=mock_anthropic):
            yield mock_anthropic


class TestAnthropicClient:
    def test_requires_api_key(self, fake_sdk, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(CascadeLLMError, match="No Anthropic API key"):
            AnthropicClient()

    def test_uses_explicit_api_key(self, fake_sdk):
        AnthropicClient(api_key="sk-test")
        fake_sdk.Anthropic.assert_called_once()
        kwargs = fake_sdk.Anthropic.call_args.kwargs
        assert kwargs["api_key"] == "sk-test"

    def test_uses_env_api_key(self, fake_sdk, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        AnthropicClient()
        kwargs = fake_sdk.Anthropic.call_args.kwargs
        assert kwargs["api_key"] == "sk-from-env"

    def test_default_model(self, fake_sdk):
        c = AnthropicClient(api_key="sk-x")
        assert c.model == AnthropicClient.DEFAULT_MODEL
        assert c.provider_name == "anthropic"

    def test_structured_call_returns_parsed_response(self, fake_sdk):
        instance = fake_sdk.Anthropic.return_value
        instance.messages.create.return_value = _fake_anthropic_response(
            "submit_sample_output",
            {"answer": "yes", "confidence": 80},
            input_tokens=100,
            output_tokens=50,
        )
        c = AnthropicClient(api_key="sk-x", model="claude-test")
        out = c.structured_call(
            system="be helpful",
            user="please respond",
            schema=SampleOutput,
        )
        assert isinstance(out, LLMResponse)
        assert isinstance(out.parsed, SampleOutput)
        assert out.parsed.answer == "yes"
        assert out.parsed.confidence == 80
        assert out.usage.input_tokens == 100
        assert out.usage.output_tokens == 50
        assert out.usage.provider == "anthropic"
        assert out.usage.model == "claude-test"

    def test_structured_call_passes_correct_tool_definition(self, fake_sdk):
        instance = fake_sdk.Anthropic.return_value
        instance.messages.create.return_value = _fake_anthropic_response(
            "submit_sample_output", {"answer": "ok", "confidence": 50}
        )
        c = AnthropicClient(api_key="sk-x")
        c.structured_call(
            system="sys",
            user="usr",
            schema=SampleOutput,
            max_tokens=1234,
            temperature=0.7,
        )
        kwargs = instance.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 1234
        assert kwargs["temperature"] == 0.7
        assert kwargs["system"] == "sys"
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_sample_output"}
        assert kwargs["tools"][0]["name"] == "submit_sample_output"
        assert "properties" in kwargs["tools"][0]["input_schema"]

    def test_api_error_wrapped_in_cascade_error(self, fake_sdk):
        instance = fake_sdk.Anthropic.return_value
        instance.messages.create.side_effect = RuntimeError("network died")
        c = AnthropicClient(api_key="sk-x")
        with pytest.raises(CascadeLLMError, match="Anthropic API call failed"):
            c.structured_call(system="s", user="u", schema=SampleOutput)

    def test_invalid_schema_response_wrapped(self, fake_sdk):
        instance = fake_sdk.Anthropic.return_value
        # Confidence out of range -> Pydantic ValidationError
        instance.messages.create.return_value = _fake_anthropic_response(
            "submit_sample_output", {"answer": "ok", "confidence": 999}
        )
        c = AnthropicClient(api_key="sk-x")
        with pytest.raises(CascadeLLMError, match="did not match schema"):
            c.structured_call(system="s", user="u", schema=SampleOutput)


# --------- Factory ----------


class TestBuildClient:
    def test_unknown_provider_raises(self):
        with pytest.raises(CascadeLLMError, match="Unknown LLM provider"):
            build_client("cohere")

    def test_anthropic_factory(self, fake_sdk, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
        c = build_client("anthropic", model="claude-test")
        assert isinstance(c, AnthropicClient)
        assert c.model == "claude-test"


# --------- LLMUsage shape ----------


def test_llm_usage_is_frozen():
    u = LLMUsage(input_tokens=1, output_tokens=2, model="m", provider="p")
    with pytest.raises(Exception):
        u.input_tokens = 99
