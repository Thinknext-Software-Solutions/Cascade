"""Pydantic data models for Cascade.

These are the typed contracts between pipeline stages. Every stage produces
and/or consumes one or more of these. They double as the structured-output
targets for LLM calls (via tool use), so the LLM is forced to return data
that conforms.

Design rules:
- Every field is typed and validated
- IDs are stable across runs (so users can reference stories by ID)
- Timestamps in ISO 8601 UTC
- No business logic here -- pure data
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------


class StorySize(str, Enum):
    """Rough size estimate for a story."""

    XS = "XS"  # < 1 hour
    S = "S"  # 1-4 hours
    M = "M"  # 4-16 hours
    L = "L"  # 1-3 days
    XL = "XL"  # 3+ days; should probably be split


class StoryStatus(str, Enum):
    """Lifecycle status of a story extracted from a meeting."""

    EXTRACTED = "extracted"  # came out of the extractor, not yet reviewed
    APPROVED = "approved"  # human said "yes, build this"
    REJECTED = "rejected"  # human said "no, skip this"
    EDITED = "edited"  # human accepted with edits
    IN_PROGRESS = "in_progress"  # build stage is running
    SHIPPED = "shipped"  # PR opened and merged
    FAILED = "failed"  # build pipeline failed


# ----------------------------------------------------------------------------
# Transcript primitives
# ----------------------------------------------------------------------------


class SpeakerTurn(BaseModel):
    """A single contiguous speaker turn in a meeting transcript."""

    model_config = ConfigDict(frozen=True)

    speaker: str = Field(..., description="Speaker label, e.g. 'Speaker A' or a real name")
    text: str = Field(..., min_length=1, description="What they said, verbatim")
    start_seconds: float = Field(..., ge=0, description="Turn start time from beginning of audio")
    end_seconds: float = Field(..., gt=0, description="Turn end time")

    @field_validator("end_seconds")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_seconds")
        if start is not None and v <= start:
            raise ValueError(f"end_seconds ({v}) must be > start_seconds ({start})")
        return v


class MeetingTranscript(BaseModel):
    """A full meeting transcript with metadata."""

    meeting_id: str = Field(..., description="Stable ID, e.g. 'meeting-2026-09-12-1430'")
    source_file: str = Field(..., description="Path to the source audio/video/text file")
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the transcript was generated",
    )
    duration_seconds: float = Field(..., ge=0, description="Total meeting length")
    turns: list[SpeakerTurn] = Field(default_factory=list, description="Ordered speaker turns")
    speakers: list[str] = Field(default_factory=list, description="Unique speakers detected")

    def as_text(self) -> str:
        """Render as a plain-text transcript suitable for LLM input."""
        lines = []
        for t in self.turns:
            lines.append(f"[{t.start_seconds:.1f}s] {t.speaker}: {t.text}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Story primitives
# ----------------------------------------------------------------------------


class AcceptanceCriterion(BaseModel):
    """A single acceptance criterion in 'Given/When/Then' style."""

    given: str = Field(..., min_length=1, description="The starting context")
    when: str = Field(..., min_length=1, description="The action taken")
    then: str = Field(..., min_length=1, description="The expected outcome")

    def as_text(self) -> str:
        return f"Given {self.given}, when {self.when}, then {self.then}."


class Story(BaseModel):
    """A user story extracted from a meeting transcript.

    The story IS the contract between human review and code generation.
    If the story is wrong, the code will be wrong. So we capture everything
    needed for a downstream Coder to produce a correct PR.
    """

    id: str = Field(..., description="Stable ID, e.g. 'story-2026-09-12-001'")
    title: str = Field(..., min_length=4, max_length=140, description="Short imperative title")
    description: str = Field(
        ...,
        min_length=10,
        description="'As a X, I want Y so that Z' format preferred",
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=list,
        min_length=1,
        description="At least one acceptance criterion required",
    )
    size: StorySize = Field(default=StorySize.M, description="Rough effort estimate")
    confidence: int = Field(
        default=70,
        ge=0,
        le=100,
        description="Extractor's confidence in this story (0-100). Low confidence "
        "stories should be flagged for human attention.",
    )
    source_meeting_id: str = Field(..., description="ID of the meeting this came from")
    source_turn_indices: list[int] = Field(
        default_factory=list,
        description="Indices into MeetingTranscript.turns that informed this story",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Extractor's notes for the human reviewer (e.g. ambiguities)",
    )
    status: StoryStatus = Field(default=StoryStatus.EXTRACTED)

    def as_text(self) -> str:
        """Render as a plain-text story suitable for review display."""
        lines = [f"{self.title} [{self.size.value}, confidence {self.confidence}]", ""]
        lines.append(self.description)
        lines.append("")
        lines.append("Acceptance criteria:")
        for i, ac in enumerate(self.acceptance_criteria, 1):
            lines.append(f"  {i}. {ac.as_text()}")
        if self.notes:
            lines.append("")
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class StoryBatch(BaseModel):
    """A batch of stories extracted from a single meeting.

    Saved to disk as YAML for human review.
    """

    meeting_id: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extractor_model: str = Field(..., description="LLM model that extracted these")
    extractor_version: str = Field(..., description="Cascade version that extracted these")
    stories: list[Story] = Field(default_factory=list)

    def approved(self) -> list[Story]:
        """Return only approved (or edited-and-accepted) stories."""
        keep = {StoryStatus.APPROVED, StoryStatus.EDITED}
        return [s for s in self.stories if s.status in keep]
