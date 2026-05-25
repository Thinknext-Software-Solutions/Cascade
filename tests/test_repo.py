"""Tests for cascade.repo (local git + GitHub PR).

Real git operations are exercised in a tmp_path repo. GitHub PR creation
uses a fake GitHubClient (we don't hit the real API).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cascade.exceptions import CascadeRepoError
from cascade.plan_schemas import (
    CodeChange,
    FileAction,
    FileChange,
    PullRequestRef,
    TestResult,
)
from cascade.repo import (
    _parse_github_url,
    apply_code_change,
    build_pr_body,
    current_branch,
    ensure_clean_working_tree,
    safe_branch_name,
    stage_and_commit,
)
from cascade.schemas import AcceptanceCriterion, Story, StorySize, StoryStatus


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not installed"
)


@pytest.fixture
def git_repo(tmp_path):
    """A fresh local git repo with one initial commit."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "README.md").write_text("# initial\n")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True
    )
    return tmp_path


# --------- Pure helpers ----------


class TestSafeBranchName:
    def test_basic(self):
        n = safe_branch_name("story-001", "Add user logout endpoint")
        assert n == "cascade/story-001/add-user-logout-endpoint"

    def test_strips_unsafe_chars(self):
        n = safe_branch_name("s-1", "Add `quotes` and *stars*!")
        assert " " not in n
        assert "`" not in n
        assert "*" not in n

    def test_caps_slug_length(self):
        n = safe_branch_name("s-1", "x" * 200)
        slug_part = n.split("/")[-1]
        assert len(slug_part) <= 50

    def test_empty_title_falls_back(self):
        n = safe_branch_name("s-1", "")
        assert n.endswith("/story")

    def test_custom_prefix(self):
        n = safe_branch_name("s-1", "Add X", prefix="cscd")
        assert n.startswith("cscd/")


class TestParseGithubUrl:
    def test_https(self):
        owner, repo = _parse_github_url("https://github.com/foo/bar.git")
        assert (owner, repo) == ("foo", "bar")

    def test_https_no_dot_git(self):
        owner, repo = _parse_github_url("https://github.com/foo/bar")
        assert (owner, repo) == ("foo", "bar")

    def test_ssh(self):
        owner, repo = _parse_github_url("git@github.com:foo/bar.git")
        assert (owner, repo) == ("foo", "bar")

    def test_ssh_no_dot_git(self):
        owner, repo = _parse_github_url("git@github.com:foo/bar")
        assert (owner, repo) == ("foo", "bar")

    def test_invalid_raises(self):
        with pytest.raises(CascadeRepoError):
            _parse_github_url("https://gitlab.com/foo/bar")


# --------- Local git operations against a real tmp repo ----------


class TestGitOps:
    def test_current_branch(self, git_repo):
        assert current_branch(git_repo) == "main"

    def test_ensure_clean_passes_when_clean(self, git_repo):
        ensure_clean_working_tree(git_repo)

    def test_ensure_clean_raises_when_dirty(self, git_repo):
        (git_repo / "dirty.txt").write_text("uncommitted")
        with pytest.raises(CascadeRepoError, match="uncommitted changes"):
            ensure_clean_working_tree(git_repo)

    def test_apply_create_writes_file(self, git_repo):
        change = CodeChange(
            story_id="s-1",
            plan_summary="add a new file",
            files=[
                FileChange(
                    path="src/new.py",
                    action=FileAction.CREATE,
                    content="x = 1\n",
                    reason="new module",
                )
            ],
        )
        touched = apply_code_change(git_repo, change)
        assert (git_repo / "src" / "new.py").read_text() == "x = 1\n"
        assert touched[0] == git_repo / "src" / "new.py"

    def test_apply_modify_overwrites(self, git_repo):
        (git_repo / "x.py").write_text("old")
        change = CodeChange(
            story_id="s-1",
            plan_summary="modify existing file",
            files=[
                FileChange(
                    path="x.py",
                    action=FileAction.MODIFY,
                    content="new",
                    reason="updated to support the new feature",
                )
            ],
        )
        apply_code_change(git_repo, change)
        assert (git_repo / "x.py").read_text() == "new"

    def test_apply_delete_removes(self, git_repo):
        (git_repo / "dead.py").write_text("bye")
        change = CodeChange(
            story_id="s-1",
            plan_summary="delete the obsolete module",
            files=[
                FileChange(
                    path="dead.py",
                    action=FileAction.DELETE,
                    content=None,
                    reason="obsolete after the refactor",
                )
            ],
        )
        apply_code_change(git_repo, change)
        assert not (git_repo / "dead.py").exists()

    def test_stage_and_commit_creates_commit(self, git_repo):
        new_file = git_repo / "added.py"
        new_file.write_text("x = 1")
        sha = stage_and_commit(git_repo, [new_file], "feat: add x")
        assert sha is not None
        log = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=git_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert log == "feat: add x"

    def test_stage_and_commit_returns_none_when_nothing_changed(self, git_repo):
        sha = stage_and_commit(git_repo, [], "nothing to do")
        assert sha is None


# --------- PR body rendering ----------


class TestBuildPrBody:
    def _story(self):
        return Story(
            id="story-1",
            title="Add undo for record deletes",
            description="As a user, I want undo so that mistakes are reversible.",
            acceptance_criteria=[
                AcceptanceCriterion(
                    given="just deleted", when="click Undo", then="restored"
                )
            ],
            size=StorySize.S,
            confidence=92,
            source_meeting_id="meeting-1",
            status=StoryStatus.APPROVED,
        )

    def _change(self):
        return CodeChange(
            story_id="story-1",
            plan_summary="Add an Undo component",
            files=[
                FileChange(
                    path="src/ui/undo.py",
                    action=FileAction.CREATE,
                    content="x = 1",
                    reason="the new Undo component",
                )
            ],
        )

    def _test_result(self, passed=True):
        return TestResult(
            passed=passed,
            exit_code=0 if passed else 1,
            duration_seconds=1.2,
            command="pytest",
            summary="42 passed in 0.18s" if passed else "3 failed",
        )

    def test_includes_title_and_description(self):
        body = build_pr_body(
            story=self._story(),
            change=self._change(),
            test_result=self._test_result(),
        )
        assert "Add undo" in body
        assert "As a user" in body

    def test_lists_acceptance_criteria(self):
        body = build_pr_body(
            story=self._story(),
            change=self._change(),
            test_result=self._test_result(),
        )
        assert "restored" in body

    def test_lists_file_changes(self):
        body = build_pr_body(
            story=self._story(),
            change=self._change(),
            test_result=self._test_result(),
        )
        assert "src/ui/undo.py" in body
        assert "the new Undo component" in body

    def test_includes_test_result(self):
        body = build_pr_body(
            story=self._story(),
            change=self._change(),
            test_result=self._test_result(passed=True),
        )
        assert "PASSED" in body
        assert "pytest" in body

    def test_includes_cascade_attribution(self):
        body = build_pr_body(
            story=self._story(),
            change=self._change(),
            test_result=self._test_result(),
        )
        assert "Cascade" in body
        assert "story-1" in body
        assert "meeting-1" in body


# --------- Fake GitHubClient pattern (no real API calls) ----------


class FakeGitHubClient:
    def __init__(self, *, returned_number=42):
        self.calls: list[dict] = []
        self._returned = returned_number

    def open_pull_request(self, *, owner, repo, head, base, title, body):
        self.calls.append(
            dict(owner=owner, repo=repo, head=head, base=base, title=title, body=body)
        )
        return PullRequestRef(
            number=self._returned,
            url=f"https://github.com/{owner}/{repo}/pull/{self._returned}",
            branch=head,
            title=title,
        )


def test_fake_github_client_satisfies_protocol():
    client = FakeGitHubClient()
    pr = client.open_pull_request(
        owner="o", repo="r", head="h", base="main", title="t", body="b"
    )
    assert pr.number == 42
    assert pr.url.endswith("/pull/42")
    assert client.calls[0]["title"] == "t"
