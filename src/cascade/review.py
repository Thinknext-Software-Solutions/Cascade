"""Interactive review of extracted stories.

Walks the user through each story in a batch with accept/edit/reject/skip
prompts. Updates are written back to the source YAML on each decision so
work isn't lost if the session is interrupted.

The actual decision-prompting is wrapped behind a small interface
(`ReviewInterface`) so tests can drive the review loop without standard
input.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import yaml

from .exceptions import CascadeError
from .io import write_story_batch
from .schemas import Story, StoryBatch, StoryStatus


logger = logging.getLogger(__name__)


class ReviewAction(str, Enum):
    """One of the choices a reviewer makes per story."""

    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"
    SKIP = "skip"
    QUIT = "quit"


@dataclass(frozen=True)
class ReviewSummary:
    """Result of a complete review session."""

    accepted: int
    edited: int
    rejected: int
    skipped: int
    quit_early: bool


class ReviewInterface:
    """Minimal interface for prompting + displaying.

    Default (CLI) implementation prints to stdout and reads stdin. Tests
    pass a recording double that scripts decisions.
    """

    def show_intro(self, batch: StoryBatch) -> None: ...
    def show_story(self, index: int, total: int, story: Story) -> None: ...
    def prompt_action(self, story: Story) -> ReviewAction: ...
    def edit_story(self, story: Story) -> Story: ...
    def show_summary(self, summary: ReviewSummary) -> None: ...


# ----------------------------------------------------------------------------
# Default CLI implementation
# ----------------------------------------------------------------------------


class CLIReviewInterface(ReviewInterface):
    """Default reviewer UI. Uses click for prompts; falls back to input()
    so the module is independent of click for testability."""

    _ACTION_MAP: dict[str, ReviewAction] = {
        "a": ReviewAction.ACCEPT,
        "e": ReviewAction.EDIT,
        "r": ReviewAction.REJECT,
        "s": ReviewAction.SKIP,
        "q": ReviewAction.QUIT,
    }

    def __init__(
        self,
        *,
        echo: Callable[[str], None] = print,
        read_line: Callable[[str], str] = input,
        editor_env_var: str = "EDITOR",
    ):
        self._echo = echo
        self._read_line = read_line
        self._editor_env_var = editor_env_var

    def show_intro(self, batch: StoryBatch) -> None:
        self._echo("")
        self._echo(f"Reviewing {len(batch.stories)} story/stories from {batch.meeting_id}")
        self._echo(f"Extracted by {batch.extractor_model} at {batch.extracted_at.isoformat()}")
        self._echo("")
        self._echo("For each story: [a]ccept  [e]dit  [r]eject  [s]kip  [q]uit")
        self._echo("Decisions are saved as you go; safe to quit any time.")
        self._echo("")

    def show_story(self, index: int, total: int, story: Story) -> None:
        bar = "-" * 60
        self._echo(bar)
        self._echo(f"Story {index} of {total}  [size {story.size.value}, confidence {story.confidence}]")
        self._echo(bar)
        self._echo(f"Title:       {story.title}")
        self._echo("")
        self._echo("Description:")
        for line in story.description.splitlines():
            self._echo(f"  {line}")
        self._echo("")
        self._echo("Acceptance criteria:")
        for i, ac in enumerate(story.acceptance_criteria, 1):
            self._echo(f"  {i}. {ac.as_text()}")
        if story.notes:
            self._echo("")
            self._echo(f"Notes: {story.notes}")
        self._echo("")

    def prompt_action(self, story: Story) -> ReviewAction:
        while True:
            raw = self._read_line("[a/e/r/s/q] > ").strip().lower()
            if raw in self._ACTION_MAP:
                return self._ACTION_MAP[raw]
            self._echo("Please enter one of: a, e, r, s, q")

    def edit_story(self, story: Story) -> Story:
        """Open the story as YAML in $EDITOR and return the parsed result."""
        text = yaml.safe_dump(story.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        edited_text = _open_in_editor(text, suffix=".yaml", editor_env_var=self._editor_env_var)
        try:
            data = yaml.safe_load(edited_text)
            if not isinstance(data, dict):
                raise CascadeError("Edited story YAML must be a mapping")
            return Story.model_validate(data)
        except Exception as exc:
            self._echo(f"Could not parse edited YAML: {exc}")
            self._echo("Keeping original story unchanged.")
            return story

    def show_summary(self, summary: ReviewSummary) -> None:
        self._echo("")
        self._echo("=" * 60)
        self._echo("Review summary")
        self._echo("=" * 60)
        self._echo(f"  accepted: {summary.accepted}")
        self._echo(f"  edited:   {summary.edited}")
        self._echo(f"  rejected: {summary.rejected}")
        self._echo(f"  skipped:  {summary.skipped}")
        if summary.quit_early:
            self._echo("  (quit before reaching every story)")
        self._echo("")


# ----------------------------------------------------------------------------
# The review loop
# ----------------------------------------------------------------------------


def review_batch(
    batch_path: Path,
    interface: Optional[ReviewInterface] = None,
) -> tuple[StoryBatch, ReviewSummary]:
    """Walk the user through reviewing each story in the batch.

    Updates are persisted to disk after each decision so partial progress
    is preserved across interruptions.

    Args:
        batch_path: Path to the StoryBatch YAML.
        interface: Optional ReviewInterface; defaults to CLIReviewInterface.

    Returns:
        (updated_batch, summary) tuple.

    Raises:
        CascadeError: If the batch file is missing or malformed.
    """
    if interface is None:
        interface = CLIReviewInterface()

    from .io import read_story_batch

    batch = read_story_batch(batch_path)
    interface.show_intro(batch)

    accepted = edited = rejected = skipped = 0
    quit_early = False
    total = len(batch.stories)
    updated_stories: list[Story] = list(batch.stories)

    for i, story in enumerate(batch.stories, 1):
        # Skip stories already decided on a previous review pass.
        if story.status in (StoryStatus.APPROVED, StoryStatus.REJECTED, StoryStatus.EDITED):
            continue

        interface.show_story(i, total, story)
        action = interface.prompt_action(story)

        new_story = story
        if action == ReviewAction.ACCEPT:
            new_story = story.model_copy(update={"status": StoryStatus.APPROVED})
            accepted += 1
        elif action == ReviewAction.EDIT:
            new_story = interface.edit_story(story).model_copy(
                update={"status": StoryStatus.EDITED}
            )
            edited += 1
        elif action == ReviewAction.REJECT:
            new_story = story.model_copy(update={"status": StoryStatus.REJECTED})
            rejected += 1
        elif action == ReviewAction.SKIP:
            skipped += 1
        elif action == ReviewAction.QUIT:
            quit_early = True
            break

        # Persist after every non-skip decision so we don't lose work
        if action in (ReviewAction.ACCEPT, ReviewAction.EDIT, ReviewAction.REJECT):
            updated_stories[i - 1] = new_story
            interim_batch = batch.model_copy(update={"stories": updated_stories})
            write_story_batch(interim_batch, batch_path)

    final_batch = batch.model_copy(update={"stories": updated_stories})
    summary = ReviewSummary(
        accepted=accepted,
        edited=edited,
        rejected=rejected,
        skipped=skipped,
        quit_early=quit_early,
    )
    interface.show_summary(summary)
    return final_batch, summary


# ----------------------------------------------------------------------------
# $EDITOR helper
# ----------------------------------------------------------------------------


def _open_in_editor(initial_text: str, *, suffix: str, editor_env_var: str) -> str:
    """Write initial_text to a temp file, open $EDITOR on it, return the
    file's contents after the editor exits."""
    editor = os.environ.get(editor_env_var) or os.environ.get("VISUAL") or "vi"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(initial_text)
        tmp_path = f.name
    try:
        # Editors may want a real terminal; subprocess inherits stdin/stdout/stderr
        subprocess.run([editor, tmp_path], check=False)
        with open(tmp_path, encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
