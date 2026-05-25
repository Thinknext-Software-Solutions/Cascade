"""Tests for cascade.planner."""

from __future__ import annotations

import pytest

from cascade.exceptions import CascadeError, CascadeLLMError
from cascade.languages import PYTHON
from cascade.llm import LLMClient, LLMResponse, LLMUsage
from cascade.plan_schemas import FileAction, FilePlan, Plan
from cascade.planner import plan_story
from cascade.schemas import AcceptanceCriterion, Story, StorySize, StoryStatus


class FakeLLMClient(LLMClient):
    def __init__(self, *, canned: Plan | None = None, raise_error: Exception | None = None):
        self._canned = canned
        self._raise = raise_error
        self.last_user: str | None = None
        self.last_system: str | None = None

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def structured_call(self, *, system, user, schema, max_tokens=8192, temperature=0.2):
        self.last_system = system
        self.last_user = user
        if self._raise:
            raise self._raise
        assert self._canned is not None
        return LLMResponse(
            parsed=self._canned,
            raw_text="fake",
            usage=LLMUsage(input_tokens=10, output_tokens=20, model="fake-model", provider="fake"),
        )


def _story(story_id="s-001") -> Story:
    return Story(
        id=story_id,
        title="Add undo for record deletes",
        description="As a user, I want undo so that mistakes are reversible.",
        acceptance_criteria=[
            AcceptanceCriterion(given="just deleted", when="click Undo", then="restored")
        ],
        size=StorySize.S,
        confidence=92,
        source_meeting_id="m-1",
        status=StoryStatus.APPROVED,
    )


def _plan(story_id="s-001") -> Plan:
    return Plan(
        story_id=story_id,
        summary="Add Undo toast component and wire delete handler",
        files=[
            FilePlan(
                path="src/ui/UndoToast.py",
                action=FileAction.CREATE,
                intent="A new component that shows an Undo button for 30s after a delete",
            )
        ],
    )


class TestHappyPath:
    def test_returns_plan_result(self):
        llm = FakeLLMClient(canned=_plan())
        result = plan_story(story=_story(), llm=llm, language=PYTHON)
        assert result.plan.story_id == "s-001"
        assert len(result.plan.files) == 1
        assert result.usage.input_tokens == 10

    def test_overrides_mismatched_story_id(self):
        llm = FakeLLMClient(canned=_plan(story_id="wrong"))
        result = plan_story(story=_story("s-001"), llm=llm, language=PYTHON)
        assert result.plan.story_id == "s-001"

    def test_prompt_includes_language_guidance(self):
        llm = FakeLLMClient(canned=_plan())
        plan_story(story=_story(), llm=llm, language=PYTHON)
        assert "Python" in llm.last_user
        assert "type hints" in llm.last_user.lower()
        assert "tests/" in llm.last_user

    def test_prompt_includes_story(self):
        llm = FakeLLMClient(canned=_plan())
        plan_story(story=_story(), llm=llm, language=PYTHON)
        assert "Add undo" in llm.last_user

    def test_system_prompt_has_role(self):
        llm = FakeLLMClient(canned=_plan())
        plan_story(story=_story(), llm=llm, language=PYTHON)
        assert "architect" in llm.last_system.lower()


class TestErrors:
    def test_llm_error_wrapped(self):
        llm = FakeLLMClient(raise_error=CascadeLLMError("api down"))
        with pytest.raises(CascadeError, match="Planner LLM call failed"):
            plan_story(story=_story(), llm=llm, language=PYTHON)
