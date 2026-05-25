"""Tests for cascade.coder."""

from __future__ import annotations

import pytest

from cascade.coder import generate_code, read_existing_files
from cascade.exceptions import CascadeError, CascadeLLMError
from cascade.languages import PYTHON
from cascade.llm import LLMClient, LLMResponse, LLMUsage
from cascade.plan_schemas import CodeChange, FileAction, FileChange, FilePlan, Plan
from cascade.schemas import AcceptanceCriterion, Story, StorySize, StoryStatus


class FakeLLM(LLMClient):
    def __init__(self, *, canned=None, raise_error=None):
        self._canned = canned
        self._raise = raise_error
        self.last_user: str | None = None
        self.last_system: str | None = None

    @property
    def provider_name(self): return "fake"
    @property
    def model(self): return "fake-model"

    def structured_call(self, *, system, user, schema, max_tokens=8192, temperature=0.2):
        self.last_system = system
        self.last_user = user
        if self._raise:
            raise self._raise
        return LLMResponse(
            parsed=self._canned,
            raw_text="fake",
            usage=LLMUsage(input_tokens=5, output_tokens=10, model="fake", provider="fake"),
        )


def _story() -> Story:
    return Story(
        id="s-001",
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


def _plan_create_only() -> Plan:
    return Plan(
        story_id="s-001",
        summary="Add an Undo component",
        files=[
            FilePlan(
                path="src/ui/undo.py",
                action=FileAction.CREATE,
                intent="A new component for showing an Undo toast",
            ),
            FilePlan(
                path="tests/test_undo.py",
                action=FileAction.CREATE,
                intent="Tests for the new Undo component",
            ),
        ],
    )


def _change_for(plan: Plan) -> CodeChange:
    files = []
    for fp in plan.files:
        if fp.action == FileAction.DELETE:
            files.append(
                FileChange(
                    path=fp.path,
                    action=fp.action,
                    content=None,
                    reason="removing obsolete file as planned",
                )
            )
        else:
            files.append(
                FileChange(
                    path=fp.path,
                    action=fp.action,
                    content="def f(): pass\n",
                    reason="implements the planned intent",
                )
            )
    return CodeChange(story_id=plan.story_id, plan_summary=plan.summary, files=files)


class TestHappyPath:
    def test_generates_code_for_simple_plan(self, tmp_path):
        plan = _plan_create_only()
        llm = FakeLLM(canned=_change_for(plan))
        result = generate_code(
            story=_story(), plan=plan, llm=llm, language=PYTHON, repo_root=tmp_path
        )
        assert len(result.change.files) == 2
        for fc in result.change.files:
            assert fc.content is not None
            assert "def" in fc.content

    def test_prompt_includes_plan_summary_and_files(self, tmp_path):
        plan = _plan_create_only()
        llm = FakeLLM(canned=_change_for(plan))
        generate_code(
            story=_story(), plan=plan, llm=llm, language=PYTHON, repo_root=tmp_path
        )
        assert plan.summary in llm.last_user
        assert "src/ui/undo.py" in llm.last_user
        assert "tests/test_undo.py" in llm.last_user

    def test_prompt_includes_existing_contents_for_modify(self, tmp_path):
        plan = Plan(
            story_id="s-001",
            summary="Tweak the existing module",
            files=[
                FilePlan(
                    path="src/existing.py",
                    action=FileAction.MODIFY,
                    intent="Add a parameter to the foo() function",
                ),
            ],
        )
        (tmp_path / "src").mkdir()
        existing_content = "def foo():\n    return 42\n"
        (tmp_path / "src" / "existing.py").write_text(existing_content)
        llm = FakeLLM(canned=_change_for(plan))
        generate_code(
            story=_story(), plan=plan, llm=llm, language=PYTHON, repo_root=tmp_path
        )
        assert "def foo():" in llm.last_user
        assert "return 42" in llm.last_user


class TestValidationAgainstPlan:
    def test_rejects_file_not_in_plan(self, tmp_path):
        plan = _plan_create_only()
        change = _change_for(plan)
        # Add a file the LLM hallucinated -- not in the plan
        change.files.append(
            FileChange(
                path="src/sneaky.py",
                action=FileAction.CREATE,
                content="x = 1",
                reason="hallucinated",
            )
        )
        llm = FakeLLM(canned=change)
        with pytest.raises(CascadeError, match="wasn't in the plan"):
            generate_code(
                story=_story(), plan=plan, llm=llm, language=PYTHON, repo_root=tmp_path
            )

    def test_rejects_action_mismatch(self, tmp_path):
        plan = _plan_create_only()
        change = _change_for(plan)
        # Swap the action on one file
        change = change.model_copy(
            update={
                "files": [
                    FileChange(
                        path=change.files[0].path,
                        action=FileAction.DELETE,  # plan said CREATE
                        content=None,
                        reason="wrong action",
                    ),
                    *change.files[1:],
                ]
            }
        )
        llm = FakeLLM(canned=change)
        with pytest.raises(CascadeError, match="does not match plan action"):
            generate_code(
                story=_story(), plan=plan, llm=llm, language=PYTHON, repo_root=tmp_path
            )

    def test_rejects_empty_content_on_create(self, tmp_path):
        plan = _plan_create_only()
        change = CodeChange(
            story_id="s-001",
            plan_summary=plan.summary,
            files=[
                FileChange(
                    path=plan.files[0].path,
                    action=FileAction.CREATE,
                    content="   \n",  # whitespace-only (passes Pydantic, fails coder check)
                    reason="placeholder content meant to fail",
                ),
                FileChange(
                    path=plan.files[1].path,
                    action=FileAction.CREATE,
                    content="def x(): pass",
                    reason="the real implementation lives here",
                ),
            ],
        )
        llm = FakeLLM(canned=change)
        with pytest.raises(CascadeError, match="empty content"):
            generate_code(
                story=_story(), plan=plan, llm=llm, language=PYTHON, repo_root=tmp_path
            )


class TestErrors:
    def test_empty_plan_raises_early(self):
        # An empty Plan would actually fail Pydantic validation, but we
        # still defensively check in generate_code. Build a plan-shaped
        # object via model_construct (skips validators).
        empty_plan = Plan.model_construct(story_id="s-001", summary="x", files=[])
        with pytest.raises(CascadeError, match="zero files"):
            generate_code(
                story=_story(), plan=empty_plan, llm=FakeLLM(), language=PYTHON, repo_root=__import__("pathlib").Path(".")
            )

    def test_llm_error_wrapped(self, tmp_path):
        plan = _plan_create_only()
        llm = FakeLLM(raise_error=CascadeLLMError("api down"))
        with pytest.raises(CascadeError, match="Coder LLM call failed"):
            generate_code(
                story=_story(), plan=plan, llm=llm, language=PYTHON, repo_root=tmp_path
            )


class TestReadExistingFiles:
    def test_reads_modify_targets(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("contents-a")
        plan = Plan(
            story_id="s-001",
            summary="touch a.py",
            files=[
                FilePlan(
                    path="src/a.py",
                    action=FileAction.MODIFY,
                    intent="add a thing to a",
                )
            ],
        )
        out = read_existing_files(tmp_path, plan)
        assert out == {"src/a.py": "contents-a"}

    def test_skips_missing_files(self, tmp_path):
        plan = Plan(
            story_id="s-001",
            summary="touch a.py",
            files=[
                FilePlan(
                    path="src/missing.py",
                    action=FileAction.MODIFY,
                    intent="modify a missing file (planner overclaimed)",
                )
            ],
        )
        out = read_existing_files(tmp_path, plan)
        assert out == {}

    def test_ignores_create_actions(self, tmp_path):
        plan = Plan(
            story_id="s-001",
            summary="create a new",
            files=[
                FilePlan(
                    path="src/new.py",
                    action=FileAction.CREATE,
                    intent="brand-new file with the new feature",
                )
            ],
        )
        out = read_existing_files(tmp_path, plan)
        assert out == {}
