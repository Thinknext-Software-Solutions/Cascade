"""Shared pytest fixtures for the Cascade test suite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cascade.schemas import (
    AcceptanceCriterion,
    MeetingTranscript,
    SpeakerTurn,
    Story,
    StoryBatch,
    StorySize,
    StoryStatus,
)


@pytest.fixture
def sample_speaker_turns() -> list[SpeakerTurn]:
    """A short multi-speaker exchange with a clear decision."""
    return [
        SpeakerTurn(
            speaker="Alice",
            text="The biggest user pain right now is the inability to undo deletes. "
            "Three customers logged tickets last week.",
            start_seconds=0.0,
            end_seconds=5.4,
        ),
        SpeakerTurn(
            speaker="Bob",
            text="Agreed. Let's add an undo within 30 seconds of a delete, "
            "shown as a toast notification with an Undo button.",
            start_seconds=5.5,
            end_seconds=12.1,
        ),
        SpeakerTurn(
            speaker="Alice",
            text="Scope it to user-initiated single-record deletes for now. "
            "Bulk delete can come later.",
            start_seconds=12.2,
            end_seconds=17.8,
        ),
        SpeakerTurn(
            speaker="Bob",
            text="Works. I'll write that up. Also, totally separate topic, "
            "we should think about a referral program someday.",
            start_seconds=18.0,
            end_seconds=23.5,
        ),
        SpeakerTurn(
            speaker="Alice",
            text="Yeah, parking that for now -- no decision yet on referrals.",
            start_seconds=23.6,
            end_seconds=27.0,
        ),
    ]


@pytest.fixture
def sample_transcript(sample_speaker_turns) -> MeetingTranscript:
    """A complete transcript suitable for end-to-end extractor tests."""
    return MeetingTranscript(
        meeting_id="meeting-2026-09-12-1430",
        source_file="recordings/team-sync.mp3",
        captured_at=datetime(2026, 9, 12, 14, 30, tzinfo=timezone.utc),
        duration_seconds=27.0,
        turns=sample_speaker_turns,
        speakers=["Alice", "Bob"],
    )


@pytest.fixture
def empty_transcript() -> MeetingTranscript:
    """A transcript with no turns -- triggers the 'nothing to extract' path."""
    return MeetingTranscript(
        meeting_id="meeting-empty",
        source_file="recordings/empty.mp3",
        duration_seconds=10.0,
        turns=[],
        speakers=[],
    )


@pytest.fixture
def sample_story() -> Story:
    """A canonical story instance for tests that need one."""
    return Story(
        id="story-2026-09-12-001",
        title="Add 30-second undo for single-record deletes",
        description=(
            "As a user, I want to undo a record deletion within 30 seconds "
            "so that I can recover from accidental deletions."
        ),
        acceptance_criteria=[
            AcceptanceCriterion(
                given="a user just deleted a single record",
                when="they click the Undo button in the toast within 30s",
                then="the record is restored and the toast disappears",
            ),
            AcceptanceCriterion(
                given="a user just deleted a single record",
                when="30 seconds pass without clicking Undo",
                then="the deletion is permanent and the toast disappears",
            ),
        ],
        size=StorySize.S,
        confidence=92,
        source_meeting_id="meeting-2026-09-12-1430",
        source_turn_indices=[0, 1, 2],
        notes=None,
        status=StoryStatus.EXTRACTED,
    )


@pytest.fixture
def sample_story_batch(sample_story) -> StoryBatch:
    """A StoryBatch containing exactly one sample story."""
    return StoryBatch(
        meeting_id="meeting-2026-09-12-1430",
        extractor_model="claude-opus-4-7",
        extractor_version="0.0.1",
        stories=[sample_story],
    )


@pytest.fixture
def populated_memory_dir(tmp_path: Path) -> Path:
    """A team-memory directory with substantive (non-template) content."""
    mem = tmp_path / "team-memory"
    mem.mkdir()
    (mem / "conventions.md").write_text(
        "# Conventions\n\n"
        "We use snake_case for Python functions and PascalCase for classes. "
        "All API responses are JSON. Database table names are singular nouns. "
        "We avoid global mutable state. Type hints required on all public functions."
    )
    (mem / "decisions.md").write_text(
        "# Decisions\n\n"
        "## [2026-01-15] PostgreSQL chosen over MongoDB\n"
        "Chose Postgres because we want schema enforcement and the relational "
        "guarantees. All new persistence uses Postgres via SQLAlchemy."
    )
    (mem / "glossary.md").write_text(
        "# Glossary\n\n"
        "**Workspace**: a user's top-level container. Each user can have many "
        "workspaces. NOT a folder or UI panel."
    )
    return mem
