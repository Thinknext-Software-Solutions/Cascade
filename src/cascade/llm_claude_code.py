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

Error diagnostics: the SDK yields a final ``ResultMessage`` whose
``is_error`` field is ``True`` when the upstream Anthropic API rejected
the call (rate limited, overloaded, bad request). The accompanying
``api_error_status`` field carries the HTTP code (429/500/529/...).
``_collect_response`` captures these so that ``structured_call`` can
raise a message that names the actual failure mode instead of the
opaque "SDK call failed". Without this, a 429 looks identical to a
500 looks identical to a malformed response, and the retry policy in
``cascade.retry`` has nothing useful to key on.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

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
            collected_text, diag = self._collect_response(prompt, options)
        except Exception as exc:
            raise CascadeLLMError(
                f"Claude Code SDK call failed for model {self._model}: {exc}"
            ) from exc

        # The SDK completed normally but the upstream Anthropic API
        # returned an error (rate-limited, overloaded, bad request).
        # Surface the structured fields so users and the retry policy
        # can see the actual HTTP status instead of a generic wrapper.
        if diag.is_error:
            raise CascadeLLMError(_format_result_error(self._model, diag))

        # NB: we deliberately do NOT raise on stop_reason == "max_tokens"
        # here. If the model happened to emit a complete, valid JSON
        # object before hitting the budget, parsing will succeed and
        # we should return it. Only escalate truncation to an error
        # when the downstream parse actually fails -- handled below in
        # the parse/validate branches via _truncation_hint().

        json_text = _extract_json(collected_text)
        if json_text is None:
            hint = _truncation_hint(self._model, max_tokens, diag)
            if hint is not None:
                raise CascadeLLMError(hint)
            raise CascadeLLMError(
                "Claude Code response did not contain a JSON code fence. "
                f"First 300 chars: {collected_text[:300]!r}"
            )

        try:
            parsed_json = json.loads(json_text)
        except ValueError as exc:
            hint = _truncation_hint(self._model, max_tokens, diag)
            if hint is not None:
                raise CascadeLLMError(hint) from exc
            raise CascadeLLMError(
                f"Failed to parse Claude Code JSON output: {exc}\n"
                f"JSON text (truncated): {json_text[:500]}"
            ) from exc

        try:
            parsed = schema.model_validate(parsed_json)
        except ValidationError as exc:
            hint = _truncation_hint(self._model, max_tokens, diag)
            if hint is not None:
                raise CascadeLLMError(hint) from exc
            raise CascadeLLMError(
                f"Claude Code output did not match schema {schema.__name__}: {exc}"
            ) from exc

        # The SDK doesn't always surface token counts the same way; we
        # default to 0 if not available. Cost will be $0 either way
        # because claude_code is covered by the user's subscription.
        return LLMResponse(
            parsed=parsed,
            raw_text=json_text,
            usage=LLMUsage.build(
                input_tokens=0,
                output_tokens=0,
                model=self._model,
                provider=self.provider_name,
            ),
        )

    def _collect_response(
        self, prompt: str, options
    ) -> tuple[str, "ResponseDiagnostics"]:
        """Run the async query, collect text, and capture diagnostics.

        Returns the concatenated assistant text plus a ``ResponseDiagnostics``
        carrying the final ``ResultMessage`` fields (``is_error``,
        ``api_error_status``, etc.) and the most recent assistant
        ``stop_reason``. The caller decides what counts as success.
        """
        import asyncio

        async def _run() -> tuple[str, ResponseDiagnostics]:
            chunks: list[str] = []
            diag = ResponseDiagnostics()
            async for message in self._query(prompt=prompt, options=options):
                # AssistantMessage: collect text + record stop_reason. Use
                # duck typing (getattr) so the SDK message hierarchy can
                # change shape without breaking us; if a future SDK adds a
                # new message type with a `content` list of text blocks,
                # we still pick it up.
                stop = getattr(message, "stop_reason", None)
                if stop is not None:
                    diag.stop_reason = stop
                content = getattr(message, "content", None)
                if content:
                    for block in content:
                        text = getattr(block, "text", None)
                        if text:
                            chunks.append(text)
                # ResultMessage: capture the final API status. We
                # identify it by the presence of `is_error` AND `subtype`
                # together (AssistantMessage has neither). Always read
                # the LAST one we see, since multi-turn sessions may emit
                # several and the final one is the session's verdict.
                is_error = getattr(message, "is_error", None)
                subtype = getattr(message, "subtype", None)
                if is_error is not None and subtype is not None:
                    diag.is_error = bool(is_error)
                    diag.subtype = subtype
                    diag.api_error_status = getattr(
                        message, "api_error_status", None
                    )
                    diag.errors = getattr(message, "errors", None)
                    diag.result_text = getattr(message, "result", None)
            return "".join(chunks), diag

        return asyncio.run(asyncio.wait_for(_run(), timeout=self._timeout_seconds))


@dataclass
class ResponseDiagnostics:
    """Final-state fields lifted off the SDK message stream.

    Defaults assume a successful, non-truncated stream so call sites can
    construct an instance and only update what they see. Fields mirror
    the SDK's ``ResultMessage`` and ``AssistantMessage`` names so the
    mapping is one-to-one and grep-friendly.
    """

    is_error: bool = False
    subtype: Optional[str] = None
    api_error_status: Optional[int] = None
    errors: Optional[Any] = None
    result_text: Optional[str] = None
    stop_reason: Optional[str] = None


def _format_result_error(model: str, diag: "ResponseDiagnostics") -> str:
    """Build a Cascade-#2-compatible error message from a failed ResultMessage.

    The message always begins with ``"Claude Code SDK call failed for
    model <X>"`` so that ``cascade.retry._is_transient`` continues to
    recognize this as a retryable upstream-provider failure. The
    structured tail names the HTTP status (when available), the
    ``subtype`` the CLI reported, and any inline error text -- so a
    user reading the log immediately knows whether it was a 429
    (rate limit), a 500/529 (overloaded), or something else.
    """
    parts = [f"Claude Code SDK call failed for model {model}"]
    if diag.api_error_status is not None:
        parts.append(f"(HTTP {diag.api_error_status})")
    if diag.subtype is not None:
        parts.append(f"subtype={diag.subtype}")
    if diag.errors:
        parts.append(f"errors={diag.errors}")
    if diag.result_text:
        parts.append(f"result={diag.result_text!r}")
    return " ".join(parts)


def _truncation_hint(
    model: str, max_tokens: int, diag: "ResponseDiagnostics"
) -> Optional[str]:
    """Return a max-tokens truncation error message, or None if not applicable.

    Called from the JSON-parse / schema-validate failure branches as a
    diagnostic upgrade: when parsing fails AND the SDK told us the model
    hit its output budget, the truncation is almost certainly the cause,
    so we report that instead of the raw parser error. The message
    explicitly does NOT contain the ``"SDK call failed"`` marker so
    ``cascade.retry`` treats it as non-transient -- retrying the same
    prompt would produce the same truncation.
    """
    if diag.stop_reason != "max_tokens":
        return None
    return (
        f"Claude Code response was truncated at max_tokens "
        f"({max_tokens} requested) and the partial output did not parse. "
        f"Either raise max_output_tokens or split the work into smaller "
        f"chunks. Model: {model}."
    )


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
