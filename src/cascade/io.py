"""Serialization helpers for Cascade artifacts.

Story batches and transcripts are persisted to disk as YAML (human-readable,
editable in $EDITOR). This module is the single place that knows how to
read and write them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .exceptions import CascadeError
from .schemas import MeetingTranscript, StoryBatch


def write_story_batch(batch: StoryBatch, path: Path) -> None:
    """Write a StoryBatch as YAML to the given path.

    Parent directories are created if missing. The file is written
    atomically (write-temp-then-rename) so a crash mid-write leaves
    the previous version intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = batch.model_dump(mode="json")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    tmp.replace(path)


def read_story_batch(path: Path) -> StoryBatch:
    """Read a StoryBatch from a YAML file."""
    if not path.exists():
        raise CascadeError(f"Story batch file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CascadeError(f"Invalid story batch file (top level not a mapping): {path}")
    return StoryBatch.model_validate(raw)


def write_transcript(transcript: MeetingTranscript, path: Path) -> None:
    """Write a transcript as YAML to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = transcript.model_dump(mode="json")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    tmp.replace(path)


def read_transcript(path: Path) -> MeetingTranscript:
    """Read a transcript from a YAML file."""
    if not path.exists():
        raise CascadeError(f"Transcript file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CascadeError(f"Invalid transcript file (top level not a mapping): {path}")
    return MeetingTranscript.model_validate(raw)
