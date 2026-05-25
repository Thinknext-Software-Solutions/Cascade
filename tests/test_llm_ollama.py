"""Tests for cascade.llm_ollama."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from cascade.llm import LLMResponse
from cascade.exceptions import CascadeLLMError


class Sample(BaseModel):
    answer: str = Field(..., min_length=1)
    confidence: int = Field(..., ge=0, le=100)


@pytest.fixture
def fake_openai_sdk():
    sdk = MagicMock()
    sdk.OpenAI = MagicMock()
    with patch("cascade.llm_openai._OpenAISDK", sdk), patch(
        "cascade.llm_openai._load_openai_sdk", return_value=sdk
    ):
        yield sdk


def _fake_chat_response(json_string: str, input_tokens=10, output_tokens=20):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=json_string))
        ],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


class TestOllamaClient:
    def test_defaults_for_base_url_and_model(self, fake_openai_sdk):
        from cascade.llm_ollama import OllamaClient

        OllamaClient()
        kwargs = fake_openai_sdk.OpenAI.call_args.kwargs
        assert kwargs["base_url"] == OllamaClient.DEFAULT_BASE_URL
        assert kwargs["api_key"] == "ollama"

    def test_custom_base_url(self, fake_openai_sdk):
        from cascade.llm_ollama import OllamaClient

        OllamaClient(model="qwen2.5", base_url="http://gpu-host:8000/v1")
        kwargs = fake_openai_sdk.OpenAI.call_args.kwargs
        assert kwargs["base_url"] == "http://gpu-host:8000/v1"

    def test_provider_name_is_ollama(self, fake_openai_sdk):
        from cascade.llm_ollama import OllamaClient

        c = OllamaClient(model="phi3")
        assert c.provider_name == "ollama"
        assert c.model == "phi3"

    def test_structured_call_returns_parsed(self, fake_openai_sdk):
        from cascade.llm_ollama import OllamaClient

        instance = fake_openai_sdk.OpenAI.return_value
        instance.chat.completions.create.return_value = _fake_chat_response(
            json.dumps({"answer": "ok", "confidence": 75})
        )
        c = OllamaClient(model="llama3")
        out = c.structured_call(system="s", user="u", schema=Sample)
        assert isinstance(out, LLMResponse)
        assert out.parsed.confidence == 75
        # Usage should be re-tagged as ollama, not openai
        assert out.usage.provider == "ollama"


class TestFactoryDispatch:
    def test_ollama_factory(self, fake_openai_sdk):
        from cascade.llm import SUPPORTED_PROVIDERS, build_client

        assert "ollama" in SUPPORTED_PROVIDERS
        client = build_client(
            "ollama", model="llama3", base_url="http://localhost:11434/v1"
        )
        assert client.provider_name == "ollama"
        assert client.model == "llama3"
