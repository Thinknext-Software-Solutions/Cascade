"""Schemas for the plan/code/test/PR pipeline stages.

Kept separate from `schemas.py` (which holds extraction-stage schemas) so
each pipeline stage has its own focused module. All of these are also LLM
structured-output targets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------------------------
# Plan stage
# ----------------------------------------------------------------------------


class FileAction(str, Enum):
    """What the agent intends to do with a file."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


class FilePlan(BaseModel):
    """Planned change to a single file."""

    path: str = Field(
        ...,
        min_length=1,
        description="Repo-relative path, e.g. 'src/api/users.py'",
    )
    action: FileAction = Field(..., description="create/modify/delete")
    intent: str = Field(
        ...,
        min_length=10,
        description="What this file change should accomplish, in plain English. "
        "Not the code -- the intent. The coder stage produces the code.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Other repo-relative paths this change interacts with "
        "(e.g. an API endpoint references the model class it serializes).",
    )


class Plan(BaseModel):
    """Implementation plan for an approved story."""

    story_id: str = Field(..., description="Story this plan is for")
    summary: str = Field(
        ...,
        min_length=10,
        description="One-paragraph description of the overall approach",
    )
    files: list[FilePlan] = Field(
        default_factory=list,
        min_length=1,
        description="Ordered list of file-level steps. At least one required.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Risks the planner identifies (e.g. 'this touches the auth path; "
        "needs extra review'). Surfaced to the human reviewer.",
    )
    out_of_scope: list[str] = Field(
        default_factory=list,
        description="Things the planner deliberately excludes (e.g. 'not "
        "refactoring the existing pagination'). Prevents scope creep.",
    )


# ----------------------------------------------------------------------------
# Code stage
# ----------------------------------------------------------------------------


class FileChange(BaseModel):
    """Generated content for a single file."""

    path: str = Field(..., min_length=1, description="Repo-relative path")
    action: FileAction = Field(...)
    content: Optional[str] = Field(
        default=None,
        description="Full file contents for create/modify actions. None for "
        "delete.",
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Why this file change matters for the story. Surfaced in "
        "the PR description.",
    )


class CodeChange(BaseModel):
    """Set of generated file changes for one plan."""

    story_id: str
    plan_summary: str = Field(
        ...,
        description="Copied from the originating Plan.summary for traceability",
    )
    files: list[FileChange] = Field(
        default_factory=list,
        min_length=1,
        description="At least one file change required",
    )


# ----------------------------------------------------------------------------
# Test stage
# ----------------------------------------------------------------------------


class TestResult(BaseModel):
    """Outcome of running tests after applying a CodeChange."""

    # Tell pytest not to collect this as a test class. The "Test" prefix
    # is part of the domain vocabulary (test execution result), not a hint.
    __test__ = False

    model_config = ConfigDict(frozen=True)

    passed: bool
    exit_code: int
    duration_seconds: float = Field(..., ge=0)
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    command: str = Field(..., description="The command that was executed")
    summary: str = Field(
        default="",
        description="One-line summary, e.g. 'passed: 42, failed: 0' or "
        "'failed: 3 tests in 2 files'",
    )


# ----------------------------------------------------------------------------
# PR stage
# ----------------------------------------------------------------------------


class PullRequestRef(BaseModel):
    """Reference to an opened pull request."""

    number: int = Field(..., gt=0)
    url: str = Field(..., min_length=8)
    branch: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
