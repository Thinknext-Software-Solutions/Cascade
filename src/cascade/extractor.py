"""Story extractor: transcript -> StoryBatch.

The first real AI stage in the Cascade pipeline. Takes a meeting transcript
plus team memory and produces a batch of candidate user stories for human
review.

Architecture:
- Pure function-style: takes inputs, returns outputs, no global state
- Logs structured events at key checkpoints for observability
- Wraps all errors in CascadeExtractionError for orchestrator clarity
- Token usage surfaced for cost tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import __version__
from .exceptions import CascadeExtractionError, CascadeLLMError
from .llm import LLMClient, LLMUsage
from .memory import TeamMemory
from .prompts import EXTRACTOR_SYSTEM_PROMPT, build_extractor_user_prompt
from .schemas import MeetingTranscript, StoryBatch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionResult:
    """Result of running the extractor on a single transcript."""

    batch: StoryBatch
    usage: LLMUsage


def extract_stories(
    *,
    transcript: MeetingTranscript,
    llm: LLMClient,
    memory: Optional[TeamMemory] = None,
    memory_char_budget: int = 20_000,
    max_output_tokens: int = 8192,
    temperature: float = 0.2,
) -> ExtractionResult:
    """Extract user stories from a meeting transcript.

    Args:
        transcript: The parsed meeting transcript.
        llm: LLM client to call.
        memory: Optional TeamMemory. If None, an empty memory is used.
        memory_char_budget: Max characters of team memory to include.
        max_output_tokens: Cap on the LLM's response size.
        temperature: Sampling temperature.

    Returns:
        ExtractionResult containing the StoryBatch (with .stories possibly
        empty if nothing extractable was found) and token usage.

    Raises:
        CascadeExtractionError: If the LLM call fails or returns output
            that violates the StoryBatch contract beyond what schema
            validation already enforces.
    """
    transcript_text = transcript.as_text()
    if not transcript_text.strip():
        raise CascadeExtractionError(
            f"Transcript {transcript.meeting_id} has no speaker turns to extract from"
        )

    memory_context = ""
    if memory is not None:
        memory_context = memory.as_llm_context(max_chars=memory_char_budget)

    meeting_date = transcript.captured_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
    user_prompt = build_extractor_user_prompt(
        transcript_text=transcript_text,
        team_memory_context=memory_context,
        meeting_id=transcript.meeting_id,
        meeting_date=meeting_date,
    )

    logger.info(
        "extractor.start",
        extra={
            "meeting_id": transcript.meeting_id,
            "transcript_chars": len(transcript_text),
            "memory_chars": len(memory_context),
            "provider": llm.provider_name,
            "model": llm.model,
        },
    )

    try:
        response = llm.structured_call(
            system=EXTRACTOR_SYSTEM_PROMPT,
            user=user_prompt,
            schema=StoryBatch,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
    except CascadeLLMError as exc:
        raise CascadeExtractionError(
            f"LLM call failed for meeting {transcript.meeting_id}: {exc}"
        ) from exc

    batch = response.parsed

    # Cross-field validation: the LLM should have used the meeting_id we asked for.
    # If not, override with the correct value (defensive) and log.
    if batch.meeting_id != transcript.meeting_id:
        logger.warning(
            "extractor.meeting_id_mismatch",
            extra={
                "asked": transcript.meeting_id,
                "returned": batch.meeting_id,
            },
        )
        batch = batch.model_copy(update={"meeting_id": transcript.meeting_id})

    # Stamp the extractor identity. The LLM might have echoed back placeholders.
    batch = batch.model_copy(
        update={
            "extractor_model": llm.model,
            "extractor_version": __version__,
            "extracted_at": datetime.now(timezone.utc),
        }
    )

    logger.info(
        "extractor.done",
        extra={
            "meeting_id": transcript.meeting_id,
            "story_count": len(batch.stories),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )

    return ExtractionResult(batch=batch, usage=response.usage)
