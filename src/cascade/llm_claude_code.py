"""Claude Code LLM provider -- uses the user's existing Claude subscription.

For developers who have Claude Code installed locally, this provider routes
LLM calls through the Claude Agent SDK rather than requiring a separate
Anthropic API key. Removes the API-key adoption barrier entirely.

How structured output works: the Agent SDK doesn't have a built-in
"structured output" mode the way the direct API does. We compose a prompt
that asks for JSON matching the schema, then parse and validate. If the
output is malformed, we surface a clear error (this is a known limitation
documented in the LLM section of the README).

For high-stakes structured output where reliability matters, the direct
Anthropic provider is still preferred. This provider is best for users
who want zero-setup adoption and accept slightly less reliable structured
parsing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import CascadeLLMError
from .llm import LLMClient, LLMResponse, LLMUsage


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_ClaudeAgentSDK = None


def _load_claude_agent_sdk():
    global _ClaudeAgentSDK
    if _ClaudeAgentSDK is None:
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeLLMError(
                "claude-agent-sdk not installed. Install Claude Code "
                "(https://claude.com/claude-code) and run: "
                "pip install claude-agent-sdk"
            ) from exc
        _ClaudeAgentSDK = (query, ClaudeAgentOptions)
    return _ClaudeAgentSDK


class ClaudeCodeClient(LLMClient):
    """LLM client that uses the local Claude Code subscription via the
    Claude Agent SDK. No API key required."""

    DEFAULT_MODEL = "claude-opus-4-7"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        timeout_seconds: float = 300.0,
    ):
        # Load lazily so users without the SDK can still use other providers
        self._query, self._options_cls = _load_claude_agent_sdk()
        self._model = model or self.DEFAULT_MODEL
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "claude_code"

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
        json_schema = schema.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)

        prompt = (
            f"{user}\n\n"
            f"---\n\n"
            f"Respond with a single JSON object matching this exact schema. "
            f"Wrap the JSON in a ```json code fence. Do not include any "
            f"explanation outside the fence.\n\n"
            f"Schema:\n```json\n{schema_str}\n```"
        )

        options = self._options_cls(
            system_prompt=system,
            model=self._model,
            permission_mode="bypassPermissions",  # we're not running tools
        )

        try:
            # The SDK is async; we run it via asyncio.run
            collected_text = self._collect_response(prompt, options)
        except Exception as exc:
            raise CascadeLLMError(
                f"Claude Code SDK call failed for model {self._model}: {exc}"
            ) from exc

        json_text = _extract_json(collected_text)
        if json_text is None:
            raise CascadeLLMError(
                "Claude Code response did not contain a JSON code fence. "
                f"First 300 chars: {collected_text[:300]!r}"
            )

        try:
            parsed_json = json.loads(json_text)
        except ValueError as exc:
            raise CascadeLLMError(
                f"Failed to parse Claude Code JSON output: {exc}\n"
                f"JSON text (truncated): {json_text[:500]}"
            ) from exc

        try:
            parsed = schema.model_validate(parsed_json)
        except ValidationError as exc:
            raise CascadeLLMError(
                f"Claude Code output did not match schema {schema.__name__}: {exc}"
            ) from exc

        # The SDK doesn't always surface token counts the same way; we
        # default to 0 if not available.
        return LLMResponse(
            parsed=parsed,
            raw_text=json_text,
            usage=LLMUsage(
                input_tokens=0,
                output_tokens=0,
                model=self._model,
                provider=self.provider_name,
            ),
        )

    def _collect_response(self, prompt: str, options) -> str:
        """Run the async query and collect text from message stream."""
        import asyncio

        async def _run() -> str:
            chunks: list[str] = []
            async for message in self._query(prompt=prompt, options=options):
                # The SDK yields message objects with different shapes; we look for
                # text content blocks on AssistantMessage instances.
                content = getattr(message, "content", None)
                if not content:
                    continue
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(text)
            return "".join(chunks)

        return asyncio.run(asyncio.wait_for(_run(), timeout=self._timeout_seconds))


def _extract_json(text: str) -> Optional[str]:
    """Pull the contents of the first ```json ... ``` fence, or the first
    bare JSON-looking object if no fence is present."""
    fence = re.search(r"```(?:json)?\s*\n?(.+?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # Fallback: try to find the first { ... } that looks like JSON
    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace:
        return brace.group(1).strip()
    return None
