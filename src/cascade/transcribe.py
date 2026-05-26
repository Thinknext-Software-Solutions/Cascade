"""Audio/video transcription with optional speaker diarization.

Three backends, picked by config or auto-detected:
  - local-whisper: openai-whisper running on the user's machine (default,
    no API calls, but heavy install and slow on CPU)
  - faster-whisper: CTranslate2-based, much faster than openai-whisper
    with the same quality (recommended if installed)
  - openai-api: OpenAI's hosted Whisper API (fast, no local model needed,
    but sends audio to OpenAI)

Diarization (assigning speaker labels) is OPTIONAL and requires pyannote.
If pyannote isn't installed, all turns are labeled "Speaker" with no
attempt at speaker separation. Cascade still works fine -- speaker labels
are nice-to-have, not load-bearing for story extraction.

Design rules:
- Backends are loaded lazily so users only install what they use
- File-size and format validation up front (clear errors, no surprises)
- All errors wrapped in CascadeTranscriptionError with context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .exceptions import CascadeTranscriptionError
from .schemas import MeetingTranscript, SpeakerTurn


logger = logging.getLogger(__name__)


SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm",
)
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mov", ".avi", ".mkv",
)
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    SUPPORTED_AUDIO_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
)

SUPPORTED_BACKENDS: tuple[str, ...] = (
    "auto",
    "local-whisper",
    "faster-whisper",
    "openai-api",
)


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of a transcription run."""

    transcript: MeetingTranscript
    backend: str  # which backend produced this
    diarization_used: bool


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def transcribe_file(
    source: Path,
    *,
    backend: str = "auto",
    model: str = "base",
    language: Optional[str] = None,
    enable_diarization: bool = True,
    openai_api_key: Optional[str] = None,
    diarization_hf_token: Optional[str] = None,
    meeting_id: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe an audio or video file into a structured MeetingTranscript.

    Args:
        source: Path to the audio/video file.
        backend: 'auto' (try faster-whisper, then openai-whisper, then API),
            or an explicit backend name.
        model: Whisper model size for local backends ('tiny', 'base', 'small',
            'medium', 'large'). Ignored for the openai-api backend.
        language: ISO 639-1 code (e.g. 'en') to force a language; None to
            auto-detect.
        enable_diarization: If True and pyannote is installed, label speakers.
            If False, all turns get a single 'Speaker' label.
        openai_api_key: Required if backend is 'openai-api'.
        diarization_hf_token: HuggingFace token for the pyannote model.
        meeting_id: Stable ID for the transcript. Defaults to a slug derived
            from the file name + timestamp.

    Returns:
        TranscriptionResult with the parsed transcript and backend metadata.

    Raises:
        CascadeTranscriptionError: File missing/unsupported, no backend
            available, or transcription failure.
    """
    _validate_source(source)

    chosen_backend = backend
    if chosen_backend == "auto":
        chosen_backend = _pick_backend(prefer_api=openai_api_key is not None)

    logger.info(
        "transcribe.start",
        extra={
            "source": str(source),
            "backend": chosen_backend,
            "model": model,
            "diarization": enable_diarization,
        },
    )

    if chosen_backend == "faster-whisper":
        segments, duration, detected_lang = _run_faster_whisper(source, model, language)
    elif chosen_backend == "local-whisper":
        segments, duration, detected_lang = _run_local_whisper(source, model, language)
    elif chosen_backend == "openai-api":
        if not openai_api_key:
            raise CascadeTranscriptionError(
                "openai-api backend requires openai_api_key"
            )
        segments, duration, detected_lang = _run_openai_api(
            source, openai_api_key, language
        )
    else:
        raise CascadeTranscriptionError(
            f"Unknown transcription backend '{chosen_backend}'. "
            f"Supported: {', '.join(SUPPORTED_BACKENDS)}"
        )

    diarization_used = False
    if enable_diarization:
        try:
            speaker_assignments = _diarize(source, diarization_hf_token)
            segments = _apply_speakers(segments, speaker_assignments)
            diarization_used = True
        except CascadeTranscriptionError as exc:
            logger.warning(
                "transcribe.diarization_skipped", extra={"reason": str(exc)}
            )

    turns = list(_segments_to_turns(segments))
    speakers = sorted({t.speaker for t in turns}) if turns else []
    meeting_id_final = meeting_id or _default_meeting_id(source)

    transcript = MeetingTranscript(
        meeting_id=meeting_id_final,
        source_file=str(source),
        captured_at=datetime.now(timezone.utc),
        duration_seconds=duration,
        turns=turns,
        speakers=speakers,
    )

    logger.info(
        "transcribe.done",
        extra={
            "meeting_id": meeting_id_final,
            "turns": len(turns),
            "speakers": len(speakers),
            "duration": duration,
            "language": detected_lang,
        },
    )

    return TranscriptionResult(
        transcript=transcript,
        backend=chosen_backend,
        diarization_used=diarization_used,
    )


# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------


def _validate_source(source: Path) -> None:
    if not source.exists():
        raise CascadeTranscriptionError(f"Source file not found: {source}")
    if not source.is_file():
        raise CascadeTranscriptionError(f"Source is not a file: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise CascadeTranscriptionError(
            f"Unsupported file extension '{source.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def _default_meeting_id(source: Path) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"meeting-{source.stem}-{ts}"


def _pick_backend(*, prefer_api: bool) -> str:
    """Pick the best available backend in priority order."""
    if prefer_api:
        return "openai-api"
    # Try faster-whisper first (much faster than openai-whisper on CPU)
    try:
        import faster_whisper  # noqa: F401
        return "faster-whisper"
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401  # this is openai-whisper
        return "local-whisper"
    except ImportError:
        pass
    raise CascadeTranscriptionError(
        "No transcription backend available. Install the ingest extra:\n"
        "  pip install 'cascade-agent[ingest]'   (recommended; bundles whisper + diarization)\n"
        "Or install a backend directly:\n"
        "  pip install faster-whisper            (fastest CPU option)\n"
        "  pip install openai-whisper            (well-tested)\n"
        "Or set backend='openai-api' with an OpenAI API key."
    )


# ----------------------------------------------------------------------------
# Backend implementations (lazy-loaded SDKs)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _RawSegment:
    text: str
    start: float
    end: float
    speaker: Optional[str] = None


def _run_faster_whisper(
    source: Path, model_name: str, language: Optional[str]
) -> tuple[list[_RawSegment], float, str]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise CascadeTranscriptionError(
            "faster-whisper not installed. Run: pip install faster-whisper"
        ) from exc

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(
            str(source), language=language, beam_size=5
        )
        segments = [
            _RawSegment(text=s.text.strip(), start=float(s.start), end=float(s.end))
            for s in segments_iter
        ]
    except Exception as exc:
        raise CascadeTranscriptionError(
            f"faster-whisper transcription failed: {exc}"
        ) from exc

    duration = float(getattr(info, "duration", 0.0))
    detected = getattr(info, "language", "unknown")
    return segments, duration, detected


def _run_local_whisper(
    source: Path, model_name: str, language: Optional[str]
) -> tuple[list[_RawSegment], float, str]:
    try:
        import whisper  # type: ignore  # openai-whisper
    except ImportError as exc:  # pragma: no cover
        raise CascadeTranscriptionError(
            "openai-whisper not installed. Run: pip install openai-whisper"
        ) from exc

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(str(source), language=language)
    except Exception as exc:
        raise CascadeTranscriptionError(
            f"local Whisper transcription failed: {exc}"
        ) from exc

    segments = [
        _RawSegment(text=s["text"].strip(), start=float(s["start"]), end=float(s["end"]))
        for s in result.get("segments", [])
    ]
    # whisper doesn't return total duration; derive from last segment
    duration = segments[-1].end if segments else 0.0
    detected = result.get("language", "unknown")
    return segments, duration, detected


def _run_openai_api(
    source: Path, api_key: str, language: Optional[str]
) -> tuple[list[_RawSegment], float, str]:
    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise CascadeTranscriptionError(
            "openai SDK not installed. Run: pip install openai"
        ) from exc

    try:
        client = openai.OpenAI(api_key=api_key)
        with source.open("rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                language=language,
            )
    except Exception as exc:
        raise CascadeTranscriptionError(
            f"OpenAI Whisper API call failed: {exc}"
        ) from exc

    raw_segments = getattr(result, "segments", []) or []
    segments = [
        _RawSegment(
            text=s.get("text", "").strip() if isinstance(s, dict) else s.text.strip(),
            start=float(s.get("start", 0.0) if isinstance(s, dict) else s.start),
            end=float(s.get("end", 0.0) if isinstance(s, dict) else s.end),
        )
        for s in raw_segments
    ]
    duration = float(getattr(result, "duration", segments[-1].end if segments else 0.0))
    detected = getattr(result, "language", "unknown")
    return segments, duration, detected


# ----------------------------------------------------------------------------
# Diarization (optional)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _SpeakerInterval:
    speaker: str  # "Speaker A", "Speaker B", ...
    start: float
    end: float


def _diarize(
    source: Path, hf_token: Optional[str]
) -> list[_SpeakerInterval]:
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise CascadeTranscriptionError(
            "pyannote.audio not installed; speaker labels disabled. "
            "Install: pip install pyannote.audio (and get a HuggingFace token)."
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(str(source))
    except Exception as exc:
        raise CascadeTranscriptionError(
            f"pyannote diarization failed: {exc}"
        ) from exc

    # Re-label speakers as "Speaker A", "Speaker B" ... in order of first appearance
    label_map: dict[str, str] = {}
    intervals: list[_SpeakerInterval] = []
    next_label_idx = 0
    for turn, _, raw_speaker in diarization.itertracks(yield_label=True):
        if raw_speaker not in label_map:
            label_map[raw_speaker] = f"Speaker {chr(ord('A') + next_label_idx)}"
            next_label_idx += 1
        intervals.append(
            _SpeakerInterval(
                speaker=label_map[raw_speaker],
                start=float(turn.start),
                end=float(turn.end),
            )
        )
    return intervals


def _apply_speakers(
    segments: list[_RawSegment], speaker_intervals: list[_SpeakerInterval]
) -> list[_RawSegment]:
    """For each transcription segment, find the diarization interval whose
    midpoint best overlaps and use that speaker label."""
    if not speaker_intervals:
        return segments
    relabeled: list[_RawSegment] = []
    for seg in segments:
        midpoint = (seg.start + seg.end) / 2
        best = max(
            speaker_intervals,
            key=lambda iv: _overlap_score(iv, seg.start, seg.end, midpoint),
            default=None,
        )
        speaker = best.speaker if best else "Speaker"
        relabeled.append(_RawSegment(text=seg.text, start=seg.start, end=seg.end, speaker=speaker))
    return relabeled


def _overlap_score(
    interval: _SpeakerInterval, seg_start: float, seg_end: float, midpoint: float
) -> float:
    """Higher = better match between this speaker interval and a segment."""
    overlap_start = max(interval.start, seg_start)
    overlap_end = min(interval.end, seg_end)
    overlap = max(0.0, overlap_end - overlap_start)
    # Prefer intervals that contain the midpoint and have more total overlap
    contains_midpoint = 1.0 if interval.start <= midpoint <= interval.end else 0.0
    return overlap + contains_midpoint


# ----------------------------------------------------------------------------
# Segment -> SpeakerTurn conversion
# ----------------------------------------------------------------------------


def _segments_to_turns(segments: Iterable[_RawSegment]) -> Iterable[SpeakerTurn]:
    """Convert raw segments into SpeakerTurn objects, merging consecutive
    same-speaker segments to keep the transcript readable."""
    current_speaker: Optional[str] = None
    current_text: list[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for seg in segments:
        if not seg.text.strip():
            continue
        speaker = seg.speaker or "Speaker"
        if speaker == current_speaker and current_end is not None:
            # extend the running turn
            current_text.append(seg.text)
            current_end = seg.end
        else:
            # flush previous
            if current_speaker is not None and current_start is not None and current_end is not None:
                yield SpeakerTurn(
                    speaker=current_speaker,
                    text=" ".join(current_text).strip(),
                    start_seconds=current_start,
                    end_seconds=current_end,
                )
            current_speaker = speaker
            current_text = [seg.text]
            current_start = seg.start
            current_end = seg.end

    if current_speaker is not None and current_start is not None and current_end is not None:
        # Guard against zero-duration turns (Whisper sometimes emits these)
        if current_end <= current_start:
            current_end = current_start + 0.1
        yield SpeakerTurn(
            speaker=current_speaker,
            text=" ".join(current_text).strip(),
            start_seconds=current_start,
            end_seconds=current_end,
        )
