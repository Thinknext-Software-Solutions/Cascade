"""Tests for cascade.io (serialization round-trips)."""

from __future__ import annotations

import pytest

from cascade.exceptions import CascadeError
from cascade.io import (
    read_story_batch,
    read_transcript,
    write_story_batch,
    write_transcript,
)


class TestStoryBatchRoundTrip:
    def test_write_then_read_preserves_data(self, tmp_path, sample_story_batch):
        path = tmp_path / "stories" / "batch.yaml"
        write_story_batch(sample_story_batch, path)
        loaded = read_story_batch(path)
        assert loaded.meeting_id == sample_story_batch.meeting_id
        assert len(loaded.stories) == len(sample_story_batch.stories)
        assert loaded.stories[0].title == sample_story_batch.stories[0].title
        assert loaded.stories[0].confidence == sample_story_batch.stories[0].confidence
        assert (
            loaded.stories[0].acceptance_criteria[0].then
            == sample_story_batch.stories[0].acceptance_criteria[0].then
        )

    def test_write_creates_parent_directories(self, tmp_path, sample_story_batch):
        path = tmp_path / "deep" / "nested" / "stories" / "batch.yaml"
        assert not path.parent.exists()
        write_story_batch(sample_story_batch, path)
        assert path.exists()

    def test_write_is_atomic_no_temp_file_remains(self, tmp_path, sample_story_batch):
        path = tmp_path / "batch.yaml"
        write_story_batch(sample_story_batch, path)
        # The temp file with .tmp suffix should be gone after replace()
        assert not (tmp_path / "batch.yaml.tmp").exists()


class TestStoryBatchErrors:
    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(CascadeError, match="not found"):
            read_story_batch(tmp_path / "missing.yaml")

    def test_read_non_mapping_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- not\n- a\n- mapping")
        with pytest.raises(CascadeError, match="not a mapping"):
            read_story_batch(path)


class TestTranscriptRoundTrip:
    def test_write_then_read_preserves_data(self, tmp_path, sample_transcript):
        path = tmp_path / "transcripts" / "t.yaml"
        write_transcript(sample_transcript, path)
        loaded = read_transcript(path)
        assert loaded.meeting_id == sample_transcript.meeting_id
        assert loaded.duration_seconds == sample_transcript.duration_seconds
        assert len(loaded.turns) == len(sample_transcript.turns)
        assert loaded.turns[0].text == sample_transcript.turns[0].text
