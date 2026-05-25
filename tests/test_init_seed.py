"""Tests for cascade.init_seed (smart team-memory seeding)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cascade.init_seed import (
    find_existing_adr_paths,
    seed_conventions,
    seed_constraints,
    seed_decisions,
    seed_glossary,
    seed_prior_work,
    seed_team_memory,
)
from cascade.languages import GO, PYTHON, TYPESCRIPT


class TestSeedConventions:
    def test_no_language_provides_generic_template(self):
        body = seed_conventions(None)
        assert "# Conventions" in body
        assert "Cascade-suggested" in body
        assert "Describe your team" in body

    def test_python_gets_python_specifics(self):
        body = seed_conventions(PYTHON)
        assert "snake_case" in body
        assert "pytest" in body
        assert "type hints" in body.lower()

    def test_typescript_gets_typescript_specifics(self):
        body = seed_conventions(TYPESCRIPT)
        assert "TypeScript" in body
        assert "Vitest" in body
        assert "named exports" in body.lower()

    def test_go_gets_go_specifics(self):
        body = seed_conventions(GO)
        assert "gofmt" in body
        assert "_test.go" in body


class TestSeedDecisions:
    def test_includes_template(self, tmp_path):
        body = seed_decisions(tmp_path)
        assert "Decision template" in body
        assert "Context" in body
        assert "Decision" in body

    def test_finds_existing_adr_directory(self, tmp_path):
        (tmp_path / "docs" / "adr").mkdir(parents=True)
        body = seed_decisions(tmp_path)
        assert "docs/adr" in body
        assert "existing architecture documentation" in body.lower()

    def test_finds_architecture_md(self, tmp_path):
        (tmp_path / "ARCHITECTURE.md").write_text("# arch")
        body = seed_decisions(tmp_path)
        assert "ARCHITECTURE.md" in body

    def test_no_adr_dirs_no_existing_section(self, tmp_path):
        body = seed_decisions(tmp_path)
        # Template still present but no "Existing architecture" section
        assert "existing architecture documentation" not in body.lower()


class TestFindAdr:
    def test_finds_multiple(self, tmp_path):
        (tmp_path / "docs" / "adr").mkdir(parents=True)
        (tmp_path / "ARCHITECTURE.md").write_text("# arch")
        found = find_existing_adr_paths(tmp_path)
        assert "docs/adr" in found
        assert "ARCHITECTURE.md" in found

    def test_none_when_absent(self, tmp_path):
        assert find_existing_adr_paths(tmp_path) == []


class TestSeedPriorWork:
    def test_no_git_log_uses_template(self, tmp_path):
        body = seed_prior_work(tmp_path)
        assert "# Prior Work" in body
        assert "Format" in body

    def test_with_git_commits(self, tmp_path):
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@e.com"], cwd=tmp_path, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "f.txt").write_text("a")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "add pagination to users endpoint"],
            cwd=tmp_path,
            check=True,
        )
        body = seed_prior_work(tmp_path)
        assert "add pagination" in body.lower()


class TestStaticSeeds:
    def test_glossary_has_example(self):
        body = seed_glossary()
        assert "# Glossary" in body
        assert "Workspace" in body

    def test_constraints_has_categories(self):
        body = seed_constraints()
        assert "Performance" in body
        assert "Security" in body
        assert "Deployment" in body


class TestSeedTeamMemory:
    def test_returns_all_five_files(self, tmp_path):
        out = seed_team_memory(tmp_path)
        expected = {"conventions.md", "decisions.md", "constraints.md", "glossary.md", "prior-work.md"}
        assert set(out.keys()) == expected

    def test_seeds_python_when_detected(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        out = seed_team_memory(tmp_path)
        assert "snake_case" in out["conventions.md"]

    def test_seeds_without_language(self, tmp_path):
        # No marker files; conventions falls back to generic
        out = seed_team_memory(tmp_path)
        assert "Describe your team" in out["conventions.md"]
