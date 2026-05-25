"""Tests for cascade.transcribe.

The Whisper backends are mocked -- we don't actually transcribe audio in
unit tests (too slow, requires model downloads). We test:
- Validation paths (missing files, unsupported formats)
- Backend selection logic
- Segment-to-turn conversion (the part most likely to have bugs)
- Diarization assignment math
"""

from __future__ import annotations

import pytest

from cascade.exceptions import CascadeTranscriptionError
from cascade.schemas import SpeakerTurn
from cascade.transcribe import (
    SUPPORTED_BACKENDS,
    SUPPORTED_EXTENSIONS,
    _RawSegment,
    _SpeakerInterval,
    _apply_speakers,
    _overlap_score,
    _segments_to_turns,
    _validate_source,
    transcribe_file,
)


# --------- Validation ----------


class TestValidateSource:
    def test_missing_file(self, tmp_path):
        with pytest.raises(CascadeTranscriptionError, match="not found"):
            _validate_source(tmp_path / "missing.mp3")

    def test_directory_rejected(self, tmp_path):
        with pytest.raises(CascadeTranscriptionError, match="not a file"):
            _validate_source(tmp_path)

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("not audio")
        with pytest.raises(CascadeTranscriptionError, match="Unsupported file extension"):
            _validate_source(f)

    def test_supported_extension(self, tmp_path):
        f = tmp_path / "x.mp3"
        f.write_bytes(b"fake mp3")
        # No exception means valid
        _validate_source(f)


def test_supported_extensions_includes_common_formats():
    for ext in (".mp3", ".wav", ".m4a", ".mp4", ".mov"):
        assert ext in SUPPORTED_EXTENSIONS


def test_supported_backends_listed():
    assert "auto" in SUPPORTED_BACKENDS
    assert "faster-whisper" in SUPPORTED_BACKENDS
    assert "local-whisper" in SUPPORTED_BACKENDS
    assert "openai-api" in SUPPORTED_BACKENDS


# --------- transcribe_file: entry-point behavior with mocked backends ----------


class TestTranscribeFileEntryPoint:
    def test_unknown_backend_raises(self, tmp_path):
        f = tmp_path / "x.mp3"
        f.write_bytes(b"fake")
        with pytest.raises(CascadeTranscriptionError, match="Unknown transcription backend"):
            transcribe_file(f, backend="invented-backend")

    def test_openai_api_backend_requires_key(self, tmp_path):
        f = tmp_path / "x.mp3"
        f.write_bytes(b"fake")
        with pytest.raises(CascadeTranscriptionError, match="requires openai_api_key"):
            transcribe_file(f, backend="openai-api")

    def test_returns_transcript_when_backend_yields_segments(self, tmp_path, monkeypatch):
        """Substitute a backend function via monkeypatch."""
        from cascade import transcribe as tm

        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake")

        def fake_local(source, model_name, language):
            return (
                [
                    _RawSegment(text="Hello world", start=0.0, end=2.0),
                    _RawSegment(text="And another thing", start=2.1, end=4.5),
                ],
                4.5,
                "en",
            )

        monkeypatch.setattr(tm, "_run_local_whisper", fake_local)
        result = transcribe_file(
            f, backend="local-whisper", enable_diarization=False
        )
        assert result.backend == "local-whisper"
        assert result.diarization_used is False
        assert len(result.transcript.turns) == 1  # merged into one Speaker turn
        assert "Hello world" in result.transcript.turns[0].text
        assert "another thing" in result.transcript.turns[0].text


# --------- Segment -> turn conversion ----------


class TestSegmentsToTurns:
    def test_empty_input(self):
        assert list(_segments_to_turns([])) == []

    def test_merges_consecutive_same_speaker(self):
        segs = [
            _RawSegment(text="Hi", start=0, end=1, speaker="Alice"),
            _RawSegment(text="there", start=1, end=2, speaker="Alice"),
        ]
        turns = list(_segments_to_turns(segs))
        assert len(turns) == 1
        assert turns[0].speaker == "Alice"
        assert turns[0].text == "Hi there"
        assert turns[0].start_seconds == 0
        assert turns[0].end_seconds == 2

    def test_splits_on_speaker_change(self):
        segs = [
            _RawSegment(text="Hi", start=0, end=1, speaker="Alice"),
            _RawSegment(text="Hello", start=1, end=2, speaker="Bob"),
            _RawSegment(text="Yeah", start=2, end=3, speaker="Alice"),
        ]
        turns = list(_segments_to_turns(segs))
        assert len(turns) == 3
        assert turns[0].speaker == "Alice"
        assert turns[1].speaker == "Bob"
        assert turns[2].speaker == "Alice"

    def test_skips_empty_text(self):
        segs = [
            _RawSegment(text="   ", start=0, end=1, speaker="Alice"),
            _RawSegment(text="Real text", start=1, end=2, speaker="Alice"),
        ]
        turns = list(_segments_to_turns(segs))
        assert len(turns) == 1
        assert turns[0].text == "Real text"

    def test_defaults_to_generic_speaker_label_when_missing(self):
        segs = [_RawSegment(text="anonymous", start=0, end=1)]
        turns = list(_segments_to_turns(segs))
        assert turns[0].speaker == "Speaker"

    def test_yields_speaker_turn_instances(self):
        segs = [_RawSegment(text="hi", start=0, end=1, speaker="Alice")]
        turns = list(_segments_to_turns(segs))
        assert isinstance(turns[0], SpeakerTurn)

    def test_zero_duration_turn_extended(self):
        segs = [_RawSegment(text="hi", start=5.0, end=5.0, speaker="Alice")]
        turns = list(_segments_to_turns(segs))
        assert turns[0].end_seconds > turns[0].start_seconds


# --------- Diarization assignment math ----------


class TestApplySpeakers:
    def test_no_intervals_returns_segments_unchanged(self):
        segs = [_RawSegment(text="hi", start=0, end=2)]
        out = _apply_speakers(segs, [])
        assert out == segs

    def test_assigns_speaker_by_overlap(self):
        segs = [
            _RawSegment(text="alice talks", start=0, end=3),
            _RawSegment(text="bob talks", start=3.1, end=6),
        ]
        intervals = [
            _SpeakerInterval(speaker="Speaker A", start=0, end=3.05),
            _SpeakerInterval(speaker="Speaker B", start=3.05, end=6),
        ]
        out = _apply_speakers(segs, intervals)
        assert out[0].speaker == "Speaker A"
        assert out[1].speaker == "Speaker B"

    def test_overlap_score_prefers_midpoint_containment(self):
        iv_contains = _SpeakerInterval(speaker="A", start=0, end=10)
        iv_partial = _SpeakerInterval(speaker="B", start=5, end=15)
        # segment is 4..6, midpoint=5
        # contains=10s overlap=2s, score=2+1=3
        # partial=10s overlap=1s, score=1+1=2 (contains midpoint 5)
        s_contains = _overlap_score(iv_contains, 4, 6, 5)
        s_partial = _overlap_score(iv_partial, 4, 6, 5)
        assert s_contains > s_partial
