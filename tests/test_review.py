"""Tests for cascade.review (interactive review loop).

We test through a scripted ReviewInterface that records calls and returns
canned actions. No real stdin/stdout interaction in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from cascade.io import read_story_batch, write_story_batch
from cascade.review import ReviewAction, ReviewInterface, ReviewSummary, review_batch
from cascade.schemas import (
    AcceptanceCriterion,
    Story,
    StoryBatch,
    StorySize,
    StoryStatus,
)


@dataclass
class ScriptedInterface(ReviewInterface):
    """Fake review interface that returns scripted actions."""

    actions: list[ReviewAction] = field(default_factory=list)
    edit_replacements: list[Story] = field(default_factory=list)
    shown_stories: list[Story] = field(default_factory=list)
    summary_seen: Optional[ReviewSummary] = None

    def show_intro(self, batch: StoryBatch) -> None:
        pass

    def show_story(self, index: int, total: int, story: Story) -> None:
        self.shown_stories.append(story)

    def prompt_action(self, story: Story) -> ReviewAction:
        if not self.actions:
            return ReviewAction.SKIP
        return self.actions.pop(0)

    def edit_story(self, story: Story) -> Story:
        return self.edit_replacements.pop(0) if self.edit_replacements else story

    def show_summary(self, summary: ReviewSummary) -> None:
        self.summary_seen = summary


def _make_story(idx: int, status: StoryStatus = StoryStatus.EXTRACTED) -> Story:
    return Story(
        id=f"story-{idx:03d}",
        title=f"Story {idx} title goes here",
        description=f"As a user, I want feature {idx} so that benefit {idx}.",
        acceptance_criteria=[
            AcceptanceCriterion(given="g", when="w", then=f"outcome {idx}")
        ],
        size=StorySize.S,
        confidence=80,
        source_meeting_id="m-1",
        status=status,
    )


def _make_batch(n: int = 3) -> StoryBatch:
    return StoryBatch(
        meeting_id="m-1",
        extractor_model="claude-test",
        extractor_version="0.0.1",
        stories=[_make_story(i) for i in range(1, n + 1)],
    )


@pytest.fixture
def batch_file(tmp_path) -> Path:
    p = tmp_path / "batch.yaml"
    write_story_batch(_make_batch(3), p)
    return p


# --------- Happy paths for each action ----------


class TestReviewLoop:
    def test_accept_all(self, batch_file):
        iface = ScriptedInterface(
            actions=[ReviewAction.ACCEPT, ReviewAction.ACCEPT, ReviewAction.ACCEPT]
        )
        result, summary = review_batch(batch_file, iface)
        assert summary.accepted == 3
        assert summary.rejected == 0
        assert summary.quit_early is False
        for s in result.stories:
            assert s.status == StoryStatus.APPROVED

    def test_reject_all(self, batch_file):
        iface = ScriptedInterface(
            actions=[ReviewAction.REJECT] * 3
        )
        result, summary = review_batch(batch_file, iface)
        assert summary.rejected == 3
        for s in result.stories:
            assert s.status == StoryStatus.REJECTED

    def test_skip_does_not_change_status(self, batch_file):
        iface = ScriptedInterface(actions=[ReviewAction.SKIP] * 3)
        result, summary = review_batch(batch_file, iface)
        assert summary.skipped == 3
        for s in result.stories:
            assert s.status == StoryStatus.EXTRACTED

    def test_quit_stops_loop(self, batch_file):
        iface = ScriptedInterface(
            actions=[ReviewAction.ACCEPT, ReviewAction.QUIT, ReviewAction.ACCEPT]
        )
        result, summary = review_batch(batch_file, iface)
        assert summary.accepted == 1
        assert summary.quit_early is True
        # Stories 2 and 3 should still be EXTRACTED
        assert result.stories[1].status == StoryStatus.EXTRACTED
        assert result.stories[2].status == StoryStatus.EXTRACTED

    def test_edit_uses_replacement_story(self, batch_file):
        replacement = _make_story(99)
        iface = ScriptedInterface(
            actions=[ReviewAction.EDIT, ReviewAction.ACCEPT, ReviewAction.ACCEPT],
            edit_replacements=[replacement],
        )
        result, summary = review_batch(batch_file, iface)
        assert summary.edited == 1
        assert summary.accepted == 2
        assert result.stories[0].id == "story-099"
        assert result.stories[0].status == StoryStatus.EDITED

    def test_mixed_actions(self, batch_file):
        iface = ScriptedInterface(
            actions=[ReviewAction.ACCEPT, ReviewAction.REJECT, ReviewAction.SKIP]
        )
        result, summary = review_batch(batch_file, iface)
        assert (summary.accepted, summary.rejected, summary.skipped) == (1, 1, 1)
        assert result.stories[0].status == StoryStatus.APPROVED
        assert result.stories[1].status == StoryStatus.REJECTED
        assert result.stories[2].status == StoryStatus.EXTRACTED


# --------- Persistence ----------


class TestPersistence:
    def test_decisions_persisted_after_each_action(self, batch_file):
        iface = ScriptedInterface(
            actions=[ReviewAction.ACCEPT, ReviewAction.QUIT]
        )
        review_batch(batch_file, iface)
        reloaded = read_story_batch(batch_file)
        assert reloaded.stories[0].status == StoryStatus.APPROVED


# --------- Skipping already-decided stories ----------


class TestSkipsAlreadyDecided:
    def test_does_not_re_prompt_approved_stories(self, tmp_path):
        batch = StoryBatch(
            meeting_id="m-1",
            extractor_model="m",
            extractor_version="0",
            stories=[
                _make_story(1, status=StoryStatus.APPROVED),
                _make_story(2, status=StoryStatus.EXTRACTED),
                _make_story(3, status=StoryStatus.REJECTED),
            ],
        )
        path = tmp_path / "b.yaml"
        write_story_batch(batch, path)
        iface = ScriptedInterface(actions=[ReviewAction.ACCEPT])
        result, summary = review_batch(path, iface)
        # Only the EXTRACTED one should have been shown
        assert len(iface.shown_stories) == 1
        assert iface.shown_stories[0].id == "story-002"
        assert summary.accepted == 1
        # Previously-decided statuses preserved
        assert result.stories[0].status == StoryStatus.APPROVED
        assert result.stories[2].status == StoryStatus.REJECTED


# --------- Summary callback ----------


class TestSummaryShown:
    def test_summary_passed_to_interface(self, batch_file):
        iface = ScriptedInterface(actions=[ReviewAction.ACCEPT] * 3)
        review_batch(batch_file, iface)
        assert iface.summary_seen is not None
        assert iface.summary_seen.accepted == 3


# --------- CLIReviewInterface unit tests ----------


class TestCLIReviewInterface:
    def test_prompt_action_parses_chars(self):
        from cascade.review import CLIReviewInterface

        ui = CLIReviewInterface(
            echo=lambda _msg: None,
            read_line=lambda _prompt: "a",
        )
        story = _make_story(1)
        assert ui.prompt_action(story) == ReviewAction.ACCEPT

    def test_prompt_action_retries_on_invalid(self):
        from cascade.review import CLIReviewInterface

        responses = iter(["x", "?", "r"])
        ui = CLIReviewInterface(
            echo=lambda _msg: None,
            read_line=lambda _prompt: next(responses),
        )
        story = _make_story(1)
        assert ui.prompt_action(story) == ReviewAction.REJECT
