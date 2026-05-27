"""Tests for cascade.llm_claude_code.

Exercises the SDK-message-handling code in ``ClaudeCodeClient`` with a
mocked async query so the tests do not require the bundled Claude Code
CLI or any network access. The focus is the diagnostics path added in
#3: ``_collect_response`` must capture ``ResultMessage`` fields and the
most recent ``AssistantMessage.stop_reason`` so ``structured_call`` can
raise a message that names the actual upstream failure mode.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from cascade.exceptions import CascadeLLMError


class Sample(BaseModel):
    answer: str = Field(..., min_length=1)


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def _assistant(text: str | None = None, stop_reason: str | None = None) -> SimpleNamespace:
    """Mock an AssistantMessage: has `content` (list of text blocks) and `stop_reason`."""
    content = [_text_block(text)] if text is not None else []
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _result(
    *,
    is_error: bool = False,
    subtype: str = "success",
    api_error_status: int | None = None,
    errors: Any = None,
    result: str | None = None,
) -> SimpleNamespace:
    """Mock a ResultMessage: has `is_error` AND `subtype` together, no `content`."""
    return SimpleNamespace(
        is_error=is_error,
        subtype=subtype,
        api_error_status=api_error_status,
        errors=errors,
        result=result,
        content=None,
        stop_reason=None,
    )


def _make_client_with_messages(messages: list[Any]):
    """Build a ClaudeCodeClient whose SDK query yields the given messages."""
    from cascade.llm_claude_code import ClaudeCodeClient

    async def _fake_query(*, prompt, options):
        for msg in messages:
            yield msg

    fake_options_cls = lambda **kw: SimpleNamespace(**kw)  # noqa: E731
    with patch(
        "cascade.llm_claude_code._load_claude_agent_sdk",
        return_value=(_fake_query, fake_options_cls),
    ):
        return ClaudeCodeClient(model="test-model")


def test_happy_path_returns_parsed_schema() -> None:
    client = _make_client_with_messages(
        [
            _assistant(text='```json\n{"answer": "yes"}\n```', stop_reason="end_turn"),
            _result(is_error=False, subtype="success"),
        ]
    )
    response = client.structured_call(system="s", user="u", schema=Sample)
    assert response.parsed.answer == "yes"


def test_result_with_is_error_surfaces_http_status_and_subtype() -> None:
    """A 429 must produce a clear error with the HTTP code visible."""
    client = _make_client_with_messages(
        [
            _result(
                is_error=True,
                subtype="success",
                api_error_status=429,
                result="Rate limit exceeded",
            ),
        ]
    )
    with pytest.raises(CascadeLLMError) as ei:
        client.structured_call(system="s", user="u", schema=Sample)
    msg = str(ei.value)
    # Retry-policy marker must remain so cascade.retry treats this as transient.
    assert "SDK call failed" in msg
    assert "HTTP 429" in msg
    assert "subtype=success" in msg


def test_result_with_500_status_is_surfaced() -> None:
    client = _make_client_with_messages(
        [
            _result(
                is_error=True,
                subtype="success",
                api_error_status=500,
                result="Internal error",
            ),
        ]
    )
    with pytest.raises(CascadeLLMError, match="HTTP 500"):
        client.structured_call(system="s", user="u", schema=Sample)


def test_result_with_errors_field_is_included_in_message() -> None:
    client = _make_client_with_messages(
        [
            _result(
                is_error=True,
                subtype="error_during_execution",
                errors=["tool foo failed"],
            ),
        ]
    )
    with pytest.raises(CascadeLLMError) as ei:
        client.structured_call(system="s", user="u", schema=Sample)
    assert "errors=['tool foo failed']" in str(ei.value)


def test_max_tokens_stop_reason_raises_non_transient_error() -> None:
    """Truncation by max_tokens must NOT carry the retry marker.

    A retry of the same prompt would produce the same truncation, so
    cascade.retry treats this error as non-transient and surfaces it
    immediately. The substring "SDK call failed" is the retry marker
    and MUST NOT appear in this message.
    """
    client = _make_client_with_messages(
        [
            _assistant(text="partial output...", stop_reason="max_tokens"),
            _result(is_error=False, subtype="success"),
        ]
    )
    with pytest.raises(CascadeLLMError) as ei:
        client.structured_call(system="s", user="u", schema=Sample)
    msg = str(ei.value)
    assert "truncated at max_tokens" in msg
    assert "SDK call failed" not in msg


def test_result_message_is_inspected_even_without_content() -> None:
    """A ResultMessage has no `content`; the old code's early `continue`
    on empty content would have skipped it. Regression guard."""
    client = _make_client_with_messages(
        [
            _result(is_error=True, subtype="success", api_error_status=529),
        ]
    )
    with pytest.raises(CascadeLLMError, match="HTTP 529"):
        client.structured_call(system="s", user="u", schema=Sample)


def test_final_result_message_wins_over_earlier_ones() -> None:
    """Multi-turn sessions can emit several results; the LAST is authoritative."""
    client = _make_client_with_messages(
        [
            _result(is_error=False, subtype="success"),
            _assistant(text='```json\n{"answer": "hi"}\n```', stop_reason="end_turn"),
            # Final session verdict says the API call ultimately failed.
            _result(is_error=True, subtype="success", api_error_status=503),
        ]
    )
    with pytest.raises(CascadeLLMError, match="HTTP 503"):
        client.structured_call(system="s", user="u", schema=Sample)


def test_assistant_message_without_stop_reason_does_not_clobber_existing() -> None:
    """Some assistant deltas have stop_reason=None; later assistant msg sets it."""
    client = _make_client_with_messages(
        [
            _assistant(text="first chunk ", stop_reason=None),
            _assistant(text='```json\n{"answer": "ok"}\n```', stop_reason="end_turn"),
            _result(is_error=False, subtype="success"),
        ]
    )
    response = client.structured_call(system="s", user="u", schema=Sample)
    assert response.parsed.answer == "ok"
