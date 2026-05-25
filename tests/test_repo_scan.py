"""Tests for cascade.repo_scan."""

from __future__ import annotations

from cascade.languages import PYTHON, TYPESCRIPT
from cascade.repo_scan import DEFAULT_IGNORE_DIRS, scan_repo


def test_scan_includes_python_source_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
    summary = scan_repo(tmp_path, PYTHON)
    paths = {f.path for f in summary.files}
    assert "src/app.py" in paths
    assert "pyproject.toml" in paths


def test_scan_skips_ignored_dirs(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("data")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("data")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    summary = scan_repo(tmp_path, PYTHON)
    paths = {f.path for f in summary.files}
    assert all(not p.startswith(".git/") for p in paths)
    assert all(not p.startswith("node_modules/") for p in paths)
    assert "src/app.py" in paths


def test_scan_filters_by_language_extensions(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.ts").write_text("const x = 1")
    summary_py = scan_repo(tmp_path, PYTHON)
    summary_ts = scan_repo(tmp_path, TYPESCRIPT)
    py_paths = {f.path for f in summary_py.files}
    ts_paths = {f.path for f in summary_ts.files}
    assert "a.py" in py_paths
    assert "b.ts" not in py_paths  # wrong language
    assert "b.ts" in ts_paths
    assert "a.py" not in ts_paths


def test_scan_includes_common_config_files(tmp_path):
    (tmp_path / "README.md").write_text("# x")
    (tmp_path / "config.yaml").write_text("a: b")
    summary = scan_repo(tmp_path, PYTHON)
    paths = {f.path for f in summary.files}
    assert "README.md" in paths
    assert "config.yaml" in paths


def test_scan_caps_at_max_files(tmp_path):
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text("x")
    summary = scan_repo(tmp_path, PYTHON, max_files=10)
    assert len(summary.files) == 10
    assert summary.truncated is True


def test_scan_as_text_lists_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1")
    summary = scan_repo(tmp_path, PYTHON)
    text = summary.as_text()
    assert "Python" in text
    assert "src/a.py" in text


def test_default_ignore_dirs_include_common_ones():
    assert ".git" in DEFAULT_IGNORE_DIRS
    assert "node_modules" in DEFAULT_IGNORE_DIRS
    assert "__pycache__" in DEFAULT_IGNORE_DIRS
    assert "target" in DEFAULT_IGNORE_DIRS  # rust


def test_scan_handles_nonexistent_root(tmp_path):
    summary = scan_repo(tmp_path / "missing", PYTHON)
    assert summary.files == ()
    assert summary.truncated is False
