"""Tests for cascade.schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cascade.schemas import (
    AcceptanceCriterion,
    MeetingTranscript,
    SpeakerTurn,
    Story,
    StoryBatch,
    StorySize,
    StoryStatus,
)


# --------- SpeakerTurn ----------


class TestSpeakerTurn:
    def test_valid_turn(self):
        t = SpeakerTurn(speaker="Speaker A", text="Hello world", start_seconds=0, end_seconds=2)
        assert t.text == "Hello world"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker="A", text="", start_seconds=0, end_seconds=1)

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError, match="must be > start_seconds"):
            SpeakerTurn(speaker="A", text="hi", start_seconds=5, end_seconds=3)

    def test_end_equal_start_rejected(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker="A", text="hi", start_seconds=5, end_seconds=5)

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker="A", text="hi", start_seconds=-1, end_seconds=1)

    def test_turn_is_frozen(self):
        t = SpeakerTurn(speaker="A", text="hi", start_seconds=0, end_seconds=1)
        with pytest.raises(ValidationError):
            t.text = "modified"


# --------- MeetingTranscript ----------


class TestMeetingTranscript:
    def test_minimal_transcript(self):
        m = MeetingTranscript(
            meeting_id="m1",
            source_file="m.mp3",
            duration_seconds=60,
        )
        assert m.turns == []
        assert m.speakers == []

    def test_as_text_renders_turns(self):
        m = MeetingTranscript(
            meeting_id="m1",
            source_file="m.mp3",
            duration_seconds=60,
            turns=[
                SpeakerTurn(speaker="A", text="Hi", start_seconds=0, end_seconds=1),
                SpeakerTurn(speaker="B", text="Hello", start_seconds=1.5, end_seconds=2.5),
            ],
        )
        text = m.as_text()
        assert "A: Hi" in text
        assert "B: Hello" in text
        # ordering preserved
        assert text.index("A: Hi") < text.index("B: Hello")

    def test_captured_at_defaults_to_now(self):
        m = MeetingTranscript(meeting_id="m1", source_file="m.mp3", duration_seconds=60)
        assert m.captured_at.tzinfo == timezone.utc
        # within a reasonable window
        delta = abs((datetime.now(timezone.utc) - m.captured_at).total_seconds())
        assert delta < 5


# --------- AcceptanceCriterion ----------


class TestAcceptanceCriterion:
    def test_valid(self):
        ac = AcceptanceCriterion(
            given="a logged-in user",
            when="they click logout",
            then="they're redirected to /login",
        )
        assert "redirected to /login" in ac.as_text()

    def test_empty_field_rejected(self):
        with pytest.raises(ValidationError):
            AcceptanceCriterion(given="", when="they act", then="something happens")


# --------- Story ----------


class TestStory:
    def _make_story(self, **overrides):
        defaults = dict(
            id="story-001",
            title="Add user logout endpoint",
            description="As a user, I want a logout button so that I can end my session.",
            acceptance_criteria=[
                AcceptanceCriterion(
                    given="logged in", when="POST /logout", then="session ended"
                )
            ],
            source_meeting_id="meeting-001",
        )
        defaults.update(overrides)
        return Story(**defaults)

    def test_minimal_valid(self):
        s = self._make_story()
        assert s.size == StorySize.M
        assert s.confidence == 70
        assert s.status == StoryStatus.EXTRACTED

    def test_title_length_bounds(self):
        with pytest.raises(ValidationError):
            self._make_story(title="bad")  # too short
        with pytest.raises(ValidationError):
            self._make_story(title="x" * 200)  # too long

    def test_acceptance_criteria_required(self):
        with pytest.raises(ValidationError):
            self._make_story(acceptance_criteria=[])

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            self._make_story(confidence=-1)
        with pytest.raises(ValidationError):
            self._make_story(confidence=101)

    def test_as_text_includes_all_parts(self):
        s = self._make_story(
            notes="ambiguous about whether logout is from all devices",
        )
        text = s.as_text()
        assert s.title in text
        assert s.description in text
        assert "session ended" in text
        assert "Notes:" in text


# --------- StoryBatch ----------


class TestStoryBatch:
    def test_approved_filters_correctly(self):
        ac = [AcceptanceCriterion(given="g", when="w", then="t")]
        stories = [
            Story(
                id=f"s-{i}",
                title=f"Story {i} title goes here",
                description="As a user, I want X so that Y.",
                acceptance_criteria=ac,
                source_meeting_id="m1",
                status=status,
            )
            for i, status in enumerate(
                [
                    StoryStatus.EXTRACTED,
                    StoryStatus.APPROVED,
                    StoryStatus.REJECTED,
                    StoryStatus.EDITED,
                ]
            )
        ]
        batch = StoryBatch(
            meeting_id="m1",
            extractor_model="claude-opus-4-7",
            extractor_version="0.0.1",
            stories=stories,
        )
        approved = batch.approved()
        assert len(approved) == 2
        statuses = {s.status for s in approved}
        assert statuses == {StoryStatus.APPROVED, StoryStatus.EDITED}
