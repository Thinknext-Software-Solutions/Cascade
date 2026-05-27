"""Tests for cascade.retry."""

from __future__ import annotations

import pytest

from cascade.exceptions import CascadeLLMError
from cascade.retry import call_with_retry


class _Counter:
    """Tiny helper: count invocations and optionally raise the first N times."""

    def __init__(self, *, raise_n_times: int = 0, exc: Exception | None = None):
        self.calls = 0
        self._raise_n_times = raise_n_times
        self._exc = exc or CascadeLLMError(
            "Claude Code SDK call failed for model X: boom"
        )

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self._raise_n_times:
            raise self._exc
        return "ok"


def _no_sleep(_seconds: float) -> None:  # don't actually sleep in tests
    pass


def test_succeeds_on_first_attempt() -> None:
    counter = _Counter(raise_n_times=0)
    result = call_with_retry(counter, description="test", sleep=_no_sleep)
    assert result == "ok"
    assert counter.calls == 1


def test_retries_transient_then_succeeds() -> None:
    counter = _Counter(raise_n_times=2)
    result = call_with_retry(
        counter, description="test", max_attempts=3, sleep=_no_sleep
    )
    assert result == "ok"
    assert counter.calls == 3


def test_retries_until_attempts_exhausted_then_raises_last_error() -> None:
    counter = _Counter(raise_n_times=5)
    with pytest.raises(CascadeLLMError, match="SDK call failed"):
        call_with_retry(
            counter, description="test", max_attempts=3, sleep=_no_sleep
        )
    assert counter.calls == 3


def test_non_transient_llm_error_is_not_retried() -> None:
    """Schema/parse failures bubble through immediately."""
    counter = _Counter(
        raise_n_times=5,
        exc=CascadeLLMError("LLM output did not match schema Plan: ..."),
    )
    with pytest.raises(CascadeLLMError, match="did not match schema"):
        call_with_retry(
            counter, description="test", max_attempts=3, sleep=_no_sleep
        )
    # Non-transient errors must not retry; one call only.
    assert counter.calls == 1


def test_non_llm_exception_is_not_caught() -> None:
    """ValueError (or anything not CascadeLLMError) is not the retry policy's concern."""

    def fn() -> str:
        raise ValueError("not an LLM error")

    with pytest.raises(ValueError, match="not an LLM error"):
        call_with_retry(fn, description="test", sleep=_no_sleep)


def test_max_attempts_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        call_with_retry(lambda: "ok", description="test", max_attempts=0)


def test_anthropic_api_failure_marker_is_recognized_as_transient() -> None:
    """The Anthropic provider phrases its errors as 'API call failed'."""
    counter = _Counter(
        raise_n_times=1,
        exc=CascadeLLMError(
            "Anthropic API call failed for model claude-opus-4-7: ConnectionError"
        ),
    )
    result = call_with_retry(
        counter, description="test", max_attempts=2, sleep=_no_sleep
    )
    assert result == "ok"
    assert counter.calls == 2


def test_sleep_is_called_between_attempts() -> None:
    sleep_calls: list[float] = []
    counter = _Counter(raise_n_times=2)
    call_with_retry(
        counter,
        description="test",
        max_attempts=3,
        sleep=sleep_calls.append,
    )
    # Two failures -> two sleeps before the third (successful) attempt.
    assert len(sleep_calls) == 2
    # Exponential backoff: second sleep is roughly double the first
    # (plus jitter), so always strictly greater.
    assert sleep_calls[1] > sleep_calls[0]


def test_no_sleep_after_final_failure() -> None:
    """Once we've decided to give up, we should not sleep again."""
    sleep_calls: list[float] = []
    counter = _Counter(raise_n_times=5)
    with pytest.raises(CascadeLLMError):
        call_with_retry(
            counter,
            description="test",
            max_attempts=3,
            sleep=sleep_calls.append,
        )
    # 3 attempts -> 2 sleeps (between attempts 1->2 and 2->3); the last
    # failure does not sleep before raising.
    assert len(sleep_calls) == 2
