"""Tests for cascade.doctor (health check)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cascade.doctor import (
    CheckResult,
    CheckStatus,
    check_git_repo,
    check_language,
    check_project_config,
    check_python_version,
    check_team_memory,
    check_test_command,
    run_doctor,
    summarize,
)


# --------- Python version check ----------


def test_python_version_ok_on_supported_python():
    r = check_python_version()
    # We require 3.11+ and these tests run on >= 3.12
    assert r.status == CheckStatus.OK
    assert "Python" in r.name


# --------- Git repo check ----------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    return tmp_path


class TestGitRepo:
    def test_not_a_repo(self, tmp_path):
        r = check_git_repo(tmp_path)
        assert r.status == CheckStatus.FAIL
        assert "not a git repository" in r.message

    def test_valid_repo(self, git_repo):
        r = check_git_repo(git_repo)
        assert r.status == CheckStatus.OK
        assert "main" in r.message


# --------- cascade.yaml check ----------


class TestProjectConfig:
    def test_missing_is_warn(self, tmp_path):
        r = check_project_config(tmp_path)
        assert r.status == CheckStatus.WARN
        assert "not found" in r.message

    def test_valid_yaml(self, tmp_path):
        (tmp_path / "cascade.yaml").write_text(
            "version: 1\nagent:\n  provider: anthropic\n"
        )
        r = check_project_config(tmp_path)
        assert r.status == CheckStatus.OK

    def test_invalid_yaml(self, tmp_path):
        (tmp_path / "cascade.yaml").write_text("not:\n  valid:\n   :")
        r = check_project_config(tmp_path)
        assert r.status == CheckStatus.FAIL


# --------- Team memory check ----------


class TestTeamMemory:
    def test_missing_dir(self, tmp_path):
        r = check_team_memory(tmp_path)
        assert r.status == CheckStatus.WARN
        assert "not found" in r.message

    def test_template_only_is_warn(self, tmp_path):
        mem = tmp_path / "team-memory"
        mem.mkdir()
        for name in ("conventions.md", "decisions.md", "constraints.md", "glossary.md", "prior-work.md"):
            (mem / name).write_text("# Title\n\n> Template only\n")
        r = check_team_memory(tmp_path)
        assert r.status == CheckStatus.WARN

    def test_populated_files(self, tmp_path):
        mem = tmp_path / "team-memory"
        mem.mkdir()
        # Three files with substantive content
        for i, name in enumerate(("conventions.md", "decisions.md", "constraints.md")):
            (mem / name).write_text(
                f"# {name}\n\nWe use snake_case. " * 30
            )
        # Two files with templates
        for name in ("glossary.md", "prior-work.md"):
            (mem / name).write_text("# Template only\n")
        r = check_team_memory(tmp_path)
        assert r.status == CheckStatus.OK
        assert "3/5" in r.message


# --------- Language check ----------


class TestLanguage:
    def test_undetected_warns(self, tmp_path):
        r = check_language(tmp_path)
        assert r.status == CheckStatus.WARN

    def test_python_detected(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        r = check_language(tmp_path)
        assert r.status == CheckStatus.OK
        assert "Python" in r.message


# --------- Test command check ----------


class TestTestRunner:
    def test_python_pytest_found(self, tmp_path):
        # We're running in an env with pytest installed
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        r = check_test_command(tmp_path)
        # Should be OK because pytest is on PATH in this test env
        assert r.status in {CheckStatus.OK, CheckStatus.WARN}


# --------- summarize ----------


class TestSummarize:
    def test_counts_correctly(self):
        results = [
            CheckResult(name="a", status=CheckStatus.OK, message=""),
            CheckResult(name="b", status=CheckStatus.OK, message=""),
            CheckResult(name="c", status=CheckStatus.WARN, message=""),
            CheckResult(name="d", status=CheckStatus.FAIL, message=""),
            CheckResult(name="e", status=CheckStatus.SKIP, message=""),
        ]
        ok, warn, fail, skip = summarize(results)
        assert (ok, warn, fail, skip) == (2, 1, 1, 1)


# --------- end to end ----------


class TestRunDoctor:
    def test_returns_results_for_every_check(self, git_repo):
        results = run_doctor(git_repo)
        # We expect at least 10 checks
        assert len(results) >= 10
        for r in results:
            assert r.name
            assert r.status in CheckStatus
