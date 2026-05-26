"""Pydantic models for the Studio HTTP API.

These are response shapes for the JSON endpoints. They deliberately differ
from the core cascade.schemas types: API models drop internal fields,
flatten nested structures, and add UI-relevant counts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from ...schemas import StoryStatus


class ProjectSummary(BaseModel):
    """One row in the project list."""

    id: str = Field(..., description="Stable hash of the absolute path")
    name: str = Field(..., description="Directory name")
    path: str = Field(..., description="Absolute path")
    language: Optional[str] = Field(default=None, description="Detected language")
    transcript_count: int = Field(default=0, ge=0)
    story_batch_count: int = Field(default=0, ge=0)


class ProjectDetail(BaseModel):
    """Project page payload: summary + list of story batches."""

    project: ProjectSummary
    batches: list["StoryBatchSummary"]


class StoryBatchSummary(BaseModel):
    """One row in the story batches list for a project."""

    id: str = Field(..., description="Filename without .yaml extension")
    file_name: str = Field(..., description="Filename on disk")
    meeting_id: str
    extracted_at: datetime
    extractor_model: str
    story_count: int = Field(ge=0)
    approved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    edited_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)


class AcceptanceCriterionView(BaseModel):
    given: str
    when: str
    then: str


class StoryView(BaseModel):
    """One story in the review board UI."""

    id: str
    title: str
    description: str
    acceptance_criteria: list[AcceptanceCriterionView]
    size: str  # XS/S/M/L/XL
    confidence: int = Field(ge=0, le=100)
    status: str  # StoryStatus value
    notes: Optional[str] = None
    source_meeting_id: str


class StoryBatchDetail(BaseModel):
    """Story review board payload: batch summary + every story in it."""

    batch: StoryBatchSummary
    stories: list[StoryView]


class StoryDecisionRequest(BaseModel):
    """Payload for POST /decisions: one decision for one story."""

    story_id: str
    decision: str = Field(..., description="One of: approve, reject, reset")

    def status(self) -> StoryStatus:
        d = self.decision.lower()
        if d == "approve":
            return StoryStatus.APPROVED
        if d == "reject":
            return StoryStatus.REJECTED
        if d == "reset":
            return StoryStatus.EXTRACTED
        raise ValueError(
            f"decision must be 'approve', 'reject', or 'reset', got {self.decision!r}"
        )


class StoryDecisionResponse(BaseModel):
    """Response after applying one decision."""

    story_id: str
    new_status: str  # StoryStatus value
    batch: StoryBatchSummary  # so the frontend can refresh the counts


# Required for forward references on Pydantic v2
ProjectDetail.model_rebuild()
