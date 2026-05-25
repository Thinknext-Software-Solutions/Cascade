"""Tests for cascade.pipeline (end-to-end orchestration).

Uses a real local git repo plus fake LLM/GitHub clients to exercise the
full plan -> code -> apply -> install -> test -> commit -> push -> PR
flow without hitting any remote service.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cascade.exceptions import CascadeError, CascadeRepoError
from cascade.languages import LanguageProfile
from cascade.llm import LLMClient, LLMResponse, LLMUsage
from cascade.pipeline import build_story, get_github_token_from_env
from cascade.plan_schemas import (
    CodeChange,
    FileAction,
    FileChange,
    FilePlan,
    Plan,
    PullRequestRef,
)
from cascade.schemas import AcceptanceCriterion, Story, StorySize, StoryStatus


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not installed"
)


# --------- Test doubles ----------


class FakeLLM(LLMClient):
    """Returns a canned Plan first, then a canned CodeChange on second call."""

    def __init__(self, *, plan: Plan, change: CodeChange):
        self._plan = plan
        self._change = change
        self.calls = 0

    @property
    def provider_name(self): return "fake"
    @property
    def model(self): return "fake-model"

    def structured_call(self, *, system, user, schema, max_tokens=8192, temperature=0.2):
        self.calls += 1
        if schema.__name__ == "Plan":
            parsed = self._plan
        elif schema.__name__ == "CodeChange":
            parsed = self._change
        else:
            raise AssertionError(f"FakeLLM got unexpected schema: {schema.__name__}")
        return LLMResponse(
            parsed=parsed,
            raw_text="fake",
            usage=LLMUsage(input_tokens=1, output_tokens=1, model="fake", provider="fake"),
        )


class FakeGitHubClient:
    def __init__(self):
        self.calls: list[dict] = []

    def open_pull_request(self, *, owner, repo, head, base, title, body):
        self.calls.append(
            dict(owner=owner, repo=repo, head=head, base=base, title=title, body=body)
        )
        return PullRequestRef(
            number=99,
            url=f"https://github.com/{owner}/{repo}/pull/99",
            branch=head,
            title=title,
        )


# --------- Fixtures ----------


@pytest.fixture
def git_repo_with_origin(tmp_path):
    """Bare remote + working clone with main branch.

    Lets push operations succeed (origin exists and accepts pushes).
    """
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)

    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=work, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@e.com"], cwd=work, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=work, check=True
    )
    subprocess.run(
        ["git", "remote", "add", "origin", f"https://github.com/test/cascade-test.git"],
        cwd=work,
        check=True,
    )
    # Override the URL to push to our local bare remote
    subprocess.run(
        ["git", "remote", "set-url", "origin", str(remote)], cwd=work, check=True
    )
    (work / "README.md").write_text("# initial")
    subprocess.run(["git", "add", "README.md"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=work, check=True)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=work, check=True)
    return work


@pytest.fixture
def fake_language(tmp_path):
    """A language profile whose test_command always passes."""
    return LanguageProfile(
        name="fake",
        display_name="Fake",
        file_extensions=(".py",),
        source_dir_default="src",
        test_dir_default="tests",
        test_file_glob="test_*.py",
        test_command=("true",),  # always exits 0
        install_command=None,
        type_check_command=None,
        detection_files=(),
    )


def _approved_story() -> Story:
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


def _basic_plan_and_change():
    plan = Plan(
        story_id="s-001",
        summary="Add a new module implementing the Undo logic",
        files=[
            FilePlan(
                path="src/undo.py",
                action=FileAction.CREATE,
                intent="The new Undo logic with a 30-second window",
            )
        ],
    )
    change = CodeChange(
        story_id="s-001",
        plan_summary=plan.summary,
        files=[
            FileChange(
                path="src/undo.py",
                action=FileAction.CREATE,
                content="def undo(record_id):\n    pass\n",
                reason="the new Undo entry point",
            )
        ],
    )
    return plan, change


# --------- End-to-end happy path ----------


class TestBuildStory:
    def test_full_pipeline_with_pr(self, git_repo_with_origin, fake_language):
        story = _approved_story()
        plan, change = _basic_plan_and_change()
        llm = FakeLLM(plan=plan, change=change)
        gh = FakeGitHubClient()

        # The origin URL needs to look like a GitHub repo for detect_github_repo
        subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                "https://github.com/test/cascade-test.git",
            ],
            cwd=git_repo_with_origin,
            check=True,
        )
        # But for the push to succeed we need a real remote. Add a second
        # remote pointing at the bare repo, and have the pipeline use 'origin'.
        # Trick: rewrite the URL just for our test to a bare repo URL.
        # Simpler approach: create a separate bare repo and use file:// URL
        # that still parses as github (it won't, so we need to mock detect_github_repo)
        # -- but the test below uses no_pr=True to avoid that complexity.
        # For the full-PR test, we mock detect_github_repo via monkeypatch.

        from unittest.mock import patch

        with patch(
            "cascade.pipeline.detect_github_repo", return_value=("test", "cascade-test")
        ), patch("cascade.pipeline.push_branch"):  # skip the actual push
            result = build_story(
                story=story,
                repo_root=git_repo_with_origin,
                llm=llm,
                language=fake_language,
                github_client=gh,
            )

        assert result.commit_sha is not None
        assert result.test_result.passed is True
        assert result.pull_request is not None
        assert result.pull_request.number == 99
        assert "test/cascade-test" in result.pull_request.url
        assert gh.calls[0]["head"].startswith("cascade/s-001/")
        assert gh.calls[0]["base"] == "main"
        assert "Add undo" in gh.calls[0]["title"]
        assert "Add undo" in gh.calls[0]["body"]

    def test_no_pr_stops_after_commit(self, git_repo_with_origin, fake_language):
        story = _approved_story()
        plan, change = _basic_plan_and_change()
        llm = FakeLLM(plan=plan, change=change)
        result = build_story(
            story=story,
            repo_root=git_repo_with_origin,
            llm=llm,
            language=fake_language,
            github_client=None,
            push_and_open_pr=False,
        )
        assert result.commit_sha is not None
        assert result.pull_request is None

    def test_file_actually_created(self, git_repo_with_origin, fake_language):
        story = _approved_story()
        plan, change = _basic_plan_and_change()
        llm = FakeLLM(plan=plan, change=change)
        build_story(
            story=story,
            repo_root=git_repo_with_origin,
            llm=llm,
            language=fake_language,
            github_client=None,
            push_and_open_pr=False,
        )
        assert (git_repo_with_origin / "src" / "undo.py").exists()
        assert "def undo" in (git_repo_with_origin / "src" / "undo.py").read_text()

    def test_dirty_working_tree_aborts(self, git_repo_with_origin, fake_language):
        (git_repo_with_origin / "dirty.txt").write_text("uncommitted")
        story = _approved_story()
        plan, change = _basic_plan_and_change()
        llm = FakeLLM(plan=plan, change=change)
        with pytest.raises(CascadeRepoError, match="uncommitted"):
            build_story(
                story=story,
                repo_root=git_repo_with_origin,
                llm=llm,
                language=fake_language,
                github_client=None,
                push_and_open_pr=False,
            )

    def test_push_requires_github_client(self, git_repo_with_origin, fake_language):
        story = _approved_story()
        plan, change = _basic_plan_and_change()
        llm = FakeLLM(plan=plan, change=change)
        with pytest.raises(CascadeError, match="github_client is required"):
            build_story(
                story=story,
                repo_root=git_repo_with_origin,
                llm=llm,
                language=fake_language,
                github_client=None,
                push_and_open_pr=True,
            )


# --------- get_github_token_from_env ----------


class TestGithubTokenLookup:
    def test_uses_github_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "abc")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert get_github_token_from_env() == "abc"

    def test_falls_back_to_gh_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "xyz")
        assert get_github_token_from_env() == "xyz"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        with pytest.raises(CascadeError, match="No GitHub token"):
            get_github_token_from_env()
