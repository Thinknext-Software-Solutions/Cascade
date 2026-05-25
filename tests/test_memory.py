"""Tests for cascade.memory."""

import pytest

from cascade.exceptions import CascadeMemoryError
from cascade.memory import KNOWN_MEMORY_FILES, MemoryFile, TeamMemory


# --------- MemoryFile.is_empty heuristic ----------


class TestMemoryFileIsEmpty:
    def _file(self, content):
        from pathlib import Path

        return MemoryFile(name="x.md", content=content, path=Path("/tmp/x.md"))

    def test_only_headers(self):
        f = self._file("# Title\n## Subtitle\n### Sub-sub")
        assert f.is_empty

    def test_only_template_blockquotes(self):
        f = self._file(
            "# Conventions\n\n> Tell your team's AI sessions what your team uses.\n"
        )
        assert f.is_empty

    def test_only_template_italic_instructions(self):
        f = self._file(
            "# Conventions\n\n*This is your team's file -- fill it in with real specifics.*\n"
        )
        assert f.is_empty

    def test_substantive_content_not_empty(self):
        f = self._file(
            "# Conventions\n\n"
            "We use snake_case for Python functions, PascalCase for classes. "
            "All API responses are JSON. Database table names are singular. "
            "We avoid global state aggressively."
        )
        assert not f.is_empty

    def test_short_substantive_is_empty(self):
        # Less than 50 chars of real content
        f = self._file("# Title\n\nWe use snake_case.")
        assert f.is_empty


# --------- TeamMemory.load ----------


class TestLoad:
    def test_no_directory_returns_empty(self, tmp_path):
        mem = TeamMemory.load(tmp_path / "nonexistent")
        assert mem.files == []
        assert not mem.is_meaningfully_populated

    def test_load_existing_files(self, tmp_path):
        mem_dir = tmp_path / "team-memory"
        mem_dir.mkdir()
        (mem_dir / "conventions.md").write_text(
            "We use snake_case for Python. All API responses are JSON. "
            "Tables singular. No global state."
        )
        (mem_dir / "decisions.md").write_text(
            "# Decisions\n\n> Tell us your decisions"
        )  # template-y, will be empty
        mem = TeamMemory.load(mem_dir)
        assert len(mem.files) == 2
        assert len(mem.non_empty_files) == 1
        assert mem.non_empty_files[0].name == "conventions.md"

    def test_load_skips_unknown_files(self, tmp_path):
        mem_dir = tmp_path / "team-memory"
        mem_dir.mkdir()
        (mem_dir / "conventions.md").write_text("real content x" * 20)
        (mem_dir / "random.md").write_text("ignored")
        mem = TeamMemory.load(mem_dir)
        names = {f.name for f in mem.files}
        assert names == {"conventions.md"}

    def test_load_preserves_order(self, tmp_path):
        mem_dir = tmp_path / "team-memory"
        mem_dir.mkdir()
        for name in KNOWN_MEMORY_FILES:
            (mem_dir / name).write_text(f"real content {name} " * 10)
        mem = TeamMemory.load(mem_dir)
        assert [f.name for f in mem.files] == list(KNOWN_MEMORY_FILES)

    def test_load_path_that_is_a_file_raises(self, tmp_path):
        f = tmp_path / "not-a-dir.txt"
        f.write_text("hi")
        with pytest.raises(CascadeMemoryError, match="not a directory"):
            TeamMemory.load(f)


# --------- as_llm_context formatting ----------


class TestAsLLMContext:
    def _make_mem(self, **files):
        from pathlib import Path

        memfiles = [
            MemoryFile(name=name, content=content, path=Path(f"/tmp/{name}"))
            for name, content in files.items()
        ]
        return TeamMemory(files=memfiles, root=Path("/tmp"))

    def test_empty_returns_empty_string(self, tmp_path):
        mem = TeamMemory.load(tmp_path / "nonexistent")
        assert mem.as_llm_context() == ""

    def test_only_template_files_returns_empty(self):
        mem = self._make_mem(
            **{"conventions.md": "# Conventions\n\n> Tell us your conventions"}
        )
        assert mem.as_llm_context() == ""

    def test_includes_substantive_files_with_headers(self):
        mem = self._make_mem(
            **{
                "conventions.md": "snake_case for Python everywhere. " * 20,
                "decisions.md": "Postgres over MongoDB chosen. " * 20,
            }
        )
        out = mem.as_llm_context()
        assert "# Team memory" in out
        assert "## conventions.md" in out
        assert "## decisions.md" in out
        assert "snake_case" in out
        assert "Postgres" in out

    def test_respects_max_chars_budget(self):
        # Two big files; budget should truncate them
        mem = self._make_mem(
            **{
                "conventions.md": "X" * 50_000,
                "decisions.md": "Y" * 50_000,
            }
        )
        out = mem.as_llm_context(max_chars=2_000)
        assert len(out) <= 2_000
        assert "truncated" in out.lower()

    def test_minimum_per_file_budget_enforced(self):
        # Even a tight overall budget should give each file at least a chunk
        mem = self._make_mem(
            **{
                "conventions.md": "real conventions content " * 100,
            }
        )
        out = mem.as_llm_context(max_chars=600)
        assert "conventions" in out
