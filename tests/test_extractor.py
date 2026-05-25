"""Tests for cascade.extractor.

The extractor depends on an LLM, so we use a fake LLMClient that returns
a canned StoryBatch. This isolates the extractor's logic (prompt assembly,
metadata stamping, error wrapping) from any real API calls.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cascade.exceptions import CascadeExtractionError, CascadeLLMError
from cascade.extractor import extract_stories
from cascade.llm import LLMClient, LLMResponse, LLMUsage
from cascade.memory import TeamMemory
from cascade.schemas import AcceptanceCriterion, Story, StoryBatch, StorySize


class FakeLLMClient(LLMClient):
    """LLMClient that returns a canned response and records inputs.

    Tests configure `canned_batch` and inspect `last_system`/`last_user`
    to verify the extractor built the right prompts.
    """

    def __init__(
        self,
        *,
        canned_batch: StoryBatch | None = None,
        raise_error: Exception | None = None,
        model_name: str = "fake-model",
    ):
        self._canned = canned_batch
        self._raise = raise_error
        self._model = model_name
        self.calls = 0
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_schema: type | None = None

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def structured_call(self, *, system, user, schema, max_tokens=8192, temperature=0.2):
        self.calls += 1
        self.last_system = system
        self.last_user = user
        self.last_schema = schema
        if self._raise is not None:
            raise self._raise
        assert self._canned is not None, "FakeLLMClient needs canned_batch or raise_error"
        usage = LLMUsage(
            input_tokens=100, output_tokens=200, model=self._model, provider="fake"
        )
        return LLMResponse(parsed=self._canned, raw_text="fake", usage=usage)


def _empty_batch(meeting_id: str = "meeting-x") -> StoryBatch:
    return StoryBatch(
        meeting_id=meeting_id,
        extractor_model="placeholder",
        extractor_version="placeholder",
        stories=[],
    )


def _one_story_batch(meeting_id: str = "meeting-x") -> StoryBatch:
    return StoryBatch(
        meeting_id=meeting_id,
        extractor_model="placeholder",
        extractor_version="placeholder",
        stories=[
            Story(
                id="story-2026-09-12-001",
                title="Add 30-second undo for record deletes",
                description="As a user, I want to undo a delete so that recovery is possible.",
                acceptance_criteria=[
                    AcceptanceCriterion(
                        given="just deleted", when="click Undo within 30s", then="restored"
                    )
                ],
                size=StorySize.S,
                confidence=92,
                source_meeting_id=meeting_id,
                source_turn_indices=[0, 1, 2],
            )
        ],
    )


class TestHappyPath:
    def test_returns_extraction_result(self, sample_transcript):
        llm = FakeLLMClient(canned_batch=_one_story_batch(sample_transcript.meeting_id))
        result = extract_stories(transcript=sample_transcript, llm=llm)
        assert len(result.batch.stories) == 1
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 200
        assert result.usage.provider == "fake"

    def test_stamps_extractor_metadata(self, sample_transcript):
        llm = FakeLLMClient(
            canned_batch=_one_story_batch(sample_transcript.meeting_id),
            model_name="my-specific-model",
        )
        result = extract_stories(transcript=sample_transcript, llm=llm)
        assert result.batch.extractor_model == "my-specific-model"
        assert result.batch.extractor_version  # any non-empty version
        delta = abs(
            (datetime.now(timezone.utc) - result.batch.extracted_at).total_seconds()
        )
        assert delta < 5

    def test_overrides_mismatched_meeting_id(self, sample_transcript):
        # LLM mistakenly returned a different meeting_id; extractor should fix it.
        wrong = _one_story_batch("meeting-bogus")
        llm = FakeLLMClient(canned_batch=wrong)
        result = extract_stories(transcript=sample_transcript, llm=llm)
        assert result.batch.meeting_id == sample_transcript.meeting_id

    def test_calls_llm_once(self, sample_transcript):
        llm = FakeLLMClient(canned_batch=_empty_batch(sample_transcript.meeting_id))
        extract_stories(transcript=sample_transcript, llm=llm)
        assert llm.calls == 1

    def test_passes_storybatch_schema(self, sample_transcript):
        llm = FakeLLMClient(canned_batch=_empty_batch(sample_transcript.meeting_id))
        extract_stories(transcript=sample_transcript, llm=llm)
        assert llm.last_schema is StoryBatch


class TestPromptAssembly:
    def test_transcript_text_in_user_prompt(self, sample_transcript):
        llm = FakeLLMClient(canned_batch=_empty_batch(sample_transcript.meeting_id))
        extract_stories(transcript=sample_transcript, llm=llm)
        assert "Alice:" in llm.last_user
        assert "undo" in llm.last_user.lower()

    def test_meeting_id_in_user_prompt(self, sample_transcript):
        llm = FakeLLMClient(canned_batch=_empty_batch(sample_transcript.meeting_id))
        extract_stories(transcript=sample_transcript, llm=llm)
        assert sample_transcript.meeting_id in llm.last_user

    def test_memory_included_when_provided(self, sample_transcript, populated_memory_dir):
        mem = TeamMemory.load(populated_memory_dir)
        llm = FakeLLMClient(canned_batch=_empty_batch(sample_transcript.meeting_id))
        extract_stories(transcript=sample_transcript, llm=llm, memory=mem)
        assert "snake_case" in llm.last_user
        assert "PostgreSQL" in llm.last_user

    def test_memory_omitted_when_none(self, sample_transcript):
        llm = FakeLLMClient(canned_batch=_empty_batch(sample_transcript.meeting_id))
        extract_stories(transcript=sample_transcript, llm=llm, memory=None)
        # When no memory, prompt should indicate that and not include made-up content
        assert "No team memory provided" in llm.last_user

    def test_system_prompt_has_role_definition(self, sample_transcript):
        llm = FakeLLMClient(canned_batch=_empty_batch(sample_transcript.meeting_id))
        extract_stories(transcript=sample_transcript, llm=llm)
        assert "product manager" in llm.last_system.lower()
        assert "confidence" in llm.last_system.lower()


class TestErrors:
    def test_empty_transcript_raises(self, empty_transcript):
        llm = FakeLLMClient(canned_batch=_empty_batch())
        with pytest.raises(CascadeExtractionError, match="no speaker turns"):
            extract_stories(transcript=empty_transcript, llm=llm)
        assert llm.calls == 0  # didn't even bother calling the LLM

    def test_llm_error_wrapped(self, sample_transcript):
        llm = FakeLLMClient(raise_error=CascadeLLMError("api down"))
        with pytest.raises(CascadeExtractionError, match="LLM call failed"):
            extract_stories(transcript=sample_transcript, llm=llm)
