"""Tests for cascade.demo (cascade try)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cascade.demo import TOY_STORY, create_demo_workspace, run_demo
from cascade.exceptions import CascadeError
from cascade.languages import GO, PYTHON
from cascade.llm import LLMClient, LLMResponse, LLMUsage
from cascade.plan_schemas import CodeChange, FileAction, FileChange, FilePlan, Plan
from cascade.schemas import StorySize, StoryStatus


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not installed"
)


class FakeLLM(LLMClient):
    """Returns a canned Plan then a canned CodeChange."""

    def __init__(self, *, plan: Plan, change: CodeChange):
        self._plan = plan
        self._change = change

    @property
    def provider_name(self):
        return "fake"

    @property
    def model(self):
        return "fake-model"

    def structured_call(self, *, system, user, schema, max_tokens=8192, temperature=0.2):
        if schema.__name__ == "Plan":
            parsed = self._plan
        elif schema.__name__ == "CodeChange":
            parsed = self._change
        else:
            raise AssertionError(f"Unexpected schema: {schema.__name__}")
        return LLMResponse(
            parsed=parsed,
            raw_text="fake",
            usage=LLMUsage(input_tokens=1, output_tokens=1, model="fake", provider="fake"),
        )


def _hello_plan_and_change():
    plan = Plan(
        story_id=TOY_STORY.id,
        summary="Add a hello() function with a passing pytest test",
        files=[
            FilePlan(
                path="src/cascade_demo/__init__.py",
                action=FileAction.MODIFY,
                intent="Export a hello() function that returns 'world'",
            ),
            FilePlan(
                path="tests/test_hello.py",
                action=FileAction.CREATE,
                intent="A pytest test that calls hello() and asserts it returns 'world'",
            ),
        ],
    )
    change = CodeChange(
        story_id=TOY_STORY.id,
        plan_summary=plan.summary,
        files=[
            FileChange(
                path="src/cascade_demo/__init__.py",
                action=FileAction.MODIFY,
                content="def hello() -> str:\n    return 'world'\n",
                reason="The hello() function the story asked for",
            ),
            FileChange(
                path="tests/test_hello.py",
                action=FileAction.CREATE,
                content=(
                    "from cascade_demo import hello\n\n"
                    "def test_hello_returns_world():\n"
                    "    assert hello() == 'world'\n"
                ),
                reason="Verifies the acceptance criterion",
            ),
        ],
    )
    return plan, change


class TestToyStory:
    def test_toy_story_is_approved(self):
        assert TOY_STORY.status == StoryStatus.APPROVED
        assert TOY_STORY.size == StorySize.XS
        assert TOY_STORY.confidence == 100


class TestCreateDemoWorkspace:
    def test_creates_python_project_structure(self, tmp_path):
        ws = tmp_path / "demo"
        result = create_demo_workspace(PYTHON, ws)
        assert result == ws
        assert (ws / "pyproject.toml").exists()
        assert (ws / "src" / "cascade_demo" / "__init__.py").exists()
        assert (ws / "tests" / "__init__.py").exists()
        assert (ws / ".git").exists()

    def test_unsupported_language_raises(self, tmp_path):
        with pytest.raises(CascadeError, match="only supports Python"):
            create_demo_workspace(GO, tmp_path / "go-demo")


class TestRunDemo:
    def test_end_to_end_with_fake_llm_passing(self, tmp_path):
        plan, change = _hello_plan_and_change()
        llm = FakeLLM(plan=plan, change=change)

        events: list[tuple[str, str]] = []

        def collect(stage: str, msg: str) -> None:
            events.append((stage, msg))

        result = run_demo(llm=llm, on_stage=collect, keep_workspace=True)
        # Cleanup since we asked to keep
        try:
            assert result.plan.summary == plan.summary
            assert len(result.code_change.files) == 2
            # Test execution result depends on pytest being available + the
            # generated code actually running. With the FakeLLM producing
            # valid hello(), tests should pass.
            assert result.test_result.passed is True
            assert result.success is True
            # We notified at every stage
            stages = {stage for stage, _ in events}
            assert {"workspace", "plan", "code", "apply", "test"}.issubset(stages)
        finally:
            shutil.rmtree(result.workspace, ignore_errors=True)

    def test_workspace_cleanup_by_default(self, tmp_path):
        plan, change = _hello_plan_and_change()
        llm = FakeLLM(plan=plan, change=change)
        result = run_demo(llm=llm, keep_workspace=False)
        # The workspace path returned but should be gone
        assert not result.workspace.exists()
