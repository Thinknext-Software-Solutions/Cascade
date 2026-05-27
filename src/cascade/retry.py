"""Transient-failure retry helper for LLM provider calls.

The provider clients (``llm.py``, ``llm_claude_code.py``, etc.) deliberately
do not retry: the module docstring in ``llm.py`` declares retries an
orchestrator-level concern so each call site can choose its own policy.
This module is the orchestrator-level policy used by ``planner`` and
``coder``: wrap a single ``LLMClient.structured_call`` in an exponential
backoff with jitter so a transient network blip or stream interruption
does not abort an entire ``cascade build`` and force the user to re-run
from the top.

Scope: only ``CascadeLLMError`` instances whose message indicates a
provider-call failure are retried. Output-parse and schema-validation
failures (also ``CascadeLLMError``) are passed through immediately
because re-running the same prompt is unlikely to fix a deterministic
schema mismatch and would burn LLM time for no benefit. Non-LLM
exceptions are never caught here.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

from .exceptions import CascadeLLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retry policy. Three attempts with exponential backoff
# (~2s, ~4s, ~8s) plus jitter so a thundering herd of retries does not
# all hit the API at the same instant. Tuned against the failure mode
# we see in practice: a few seconds of API instability that clears
# on its own.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_BACKOFF_S = 2.0

# Substrings in CascadeLLMError messages that indicate the call to the
# upstream provider failed (network/timeout/stream interruption), as
# opposed to an output we received but could not parse. Retrying makes
# sense only for the former.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "SDK call failed",
    "API call failed",
)


def _is_transient(exc: CascadeLLMError) -> bool:
    """Return True iff this LLM error looks like a transient upstream blip."""
    message = str(exc)
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _backoff_seconds(attempt: int, base: float) -> float:
    """Exponential backoff with jitter for the given 0-indexed attempt."""
    # attempt=0 -> ~base, attempt=1 -> ~2*base, attempt=2 -> ~4*base.
    # Jitter is half the base on top of each interval.
    return base * (2 ** attempt) + random.uniform(0, base / 2)


def call_with_retry(
    fn: Callable[[], T],
    *,
    description: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_backoff_s: float = DEFAULT_BASE_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Invoke ``fn`` with exponential-backoff retry on transient LLM errors.

    Args:
        fn: A zero-argument callable that performs one LLM call.
        description: Human-readable label for log lines (e.g. "planner",
            "coder"). Appears in the structured log so users can see which
            step is retrying.
        max_attempts: Total attempts including the first. Default 3.
        base_backoff_s: First backoff interval in seconds. Subsequent
            intervals double. Default 2.0.
        sleep: Sleep function (overridable for tests so retry logic does
            not actually pause).

    Returns:
        Whatever ``fn`` returns on a successful attempt.

    Raises:
        CascadeLLMError: The last error if every attempt failed transiently,
            or the first error if it was non-transient.
        Exception: Any other exception type from ``fn`` is passed through
            without retry on the first attempt.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(max_attempts):
        try:
            return fn()
        except CascadeLLMError as exc:
            if not _is_transient(exc):
                # Non-transient: re-raise immediately so the user sees
                # the real error (schema mismatch, malformed output, etc.)
                # without waiting for the backoff schedule.
                raise
            if attempt == max_attempts - 1:
                logger.warning(
                    "retry.exhausted",
                    extra={
                        "step": description,
                        "attempts": max_attempts,
                        "error": str(exc),
                    },
                )
                raise
            delay = _backoff_seconds(attempt, base_backoff_s)
            logger.warning(
                "retry.transient",
                extra={
                    "step": description,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "sleep_s": round(delay, 2),
                    "error": str(exc),
                },
            )
            sleep(delay)

    # Unreachable: the loop above either returns or raises on every iteration.
    raise RuntimeError("call_with_retry exited the loop without returning")
