"""Tests for cascade.plan_schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cascade.plan_schemas import (
    CodeChange,
    FileAction,
    FileChange,
    FilePlan,
    Plan,
    PullRequestRef,
    TestResult,
)


class TestFilePlan:
    def test_minimal_valid(self):
        fp = FilePlan(
            path="src/api/users.py",
            action=FileAction.CREATE,
            intent="Add a paginated GET endpoint for users",
        )
        assert fp.references == []

    def test_short_intent_rejected(self):
        with pytest.raises(ValidationError):
            FilePlan(path="x.py", action=FileAction.CREATE, intent="x")

    def test_action_must_be_known(self):
        with pytest.raises(ValidationError):
            FilePlan(path="x.py", action="invent", intent="long enough intent")


class TestPlan:
    def _fp(self):
        return FilePlan(
            path="src/x.py",
            action=FileAction.CREATE,
            intent="implement the new endpoint with tests",
        )

    def test_requires_at_least_one_file(self):
        with pytest.raises(ValidationError):
            Plan(story_id="s-1", summary="a non-trivial summary", files=[])

    def test_minimal_valid(self):
        plan = Plan(
            story_id="s-1",
            summary="Add an endpoint and a test",
            files=[self._fp()],
        )
        assert plan.risks == []
        assert plan.out_of_scope == []


class TestFileChange:
    def test_create_with_content(self):
        fc = FileChange(
            path="src/x.py",
            action=FileAction.CREATE,
            content="def f(): pass\n",
            reason="primary entry point for the new feature",
        )
        assert fc.content.startswith("def ")

    def test_delete_with_null_content(self):
        fc = FileChange(
            path="src/dead.py",
            action=FileAction.DELETE,
            content=None,
            reason="superseded by the new module",
        )
        assert fc.content is None


class TestCodeChange:
    def test_minimal_valid(self):
        change = CodeChange(
            story_id="s-1",
            plan_summary="some summary",
            files=[
                FileChange(
                    path="x.py",
                    action=FileAction.CREATE,
                    content="x = 1\n",
                    reason="the new module",
                )
            ],
        )
        assert change.story_id == "s-1"


class TestTestResult:
    def test_frozen(self):
        r = TestResult(
            passed=True,
            exit_code=0,
            duration_seconds=1.5,
            command="pytest",
        )
        with pytest.raises(Exception):
            r.passed = False


class TestPullRequestRef:
    def test_minimal_valid(self):
        pr = PullRequestRef(
            number=42,
            url="https://github.com/x/y/pull/42",
            branch="cascade/story-1",
            title="Add X",
        )
        assert pr.number == 42

    def test_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            PullRequestRef(
                number=0,
                url="https://github.com/x/y/pull/0",
                branch="b",
                title="t",
            )
