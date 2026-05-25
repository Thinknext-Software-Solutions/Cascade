"""Lightweight repo introspection: produce a compact tree summary for LLM prompts.

The planner and coder both need *some* awareness of the repo structure
without dumping every file's contents. This module provides a budgeted
summary: directory tree first, then small files inlined where useful.

Design:
- Respects .gitignore-style ignore patterns (built-in defaults; we don't
  parse .gitignore in v0.1 -- a deliberate simplification)
- Caps total characters and files to keep prompts reasonable
- Deterministic ordering for reproducible LLM inputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .languages import LanguageProfile


DEFAULT_IGNORE_DIRS: tuple[str, ...] = (
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    "target",  # rust
    ".gradle",
    ".idea",
    ".vscode",
    ".pkgs",
    ".wrangler",
    "coverage",
    "htmlcov",
)


DEFAULT_IGNORE_FILE_SUFFIXES: tuple[str, ...] = (
    ".pyc",
    ".pyo",
    ".so",
    ".o",
    ".class",
    ".jar",
    ".log",
    ".lock",
)


@dataclass(frozen=True)
class FileEntry:
    """A single file in the repo summary."""

    path: str  # repo-relative POSIX path
    size_bytes: int


@dataclass(frozen=True)
class RepoSummary:
    """A compact, LLM-friendly summary of a repo's structure."""

    root: Path
    language: LanguageProfile
    files: tuple[FileEntry, ...]
    truncated: bool  # True if we hit the file cap

    def as_text(self) -> str:
        """Render as a tree-like text listing."""
        lines = [f"# Repo structure ({self.language.display_name})", ""]
        for f in self.files:
            lines.append(f"  {f.path} ({f.size_bytes} bytes)")
        if self.truncated:
            lines.append("")
            lines.append("  ... (output truncated)")
        return "\n".join(lines)


def scan_repo(
    root: Path,
    language: LanguageProfile,
    *,
    max_files: int = 200,
    ignore_dirs: Iterable[str] = DEFAULT_IGNORE_DIRS,
    ignore_file_suffixes: Iterable[str] = DEFAULT_IGNORE_FILE_SUFFIXES,
) -> RepoSummary:
    """Walk the repo and produce a RepoSummary of relevant files.

    'Relevant' = source/test files matching the language profile, plus
    common config files (pyproject.toml, package.json, etc.).

    Args:
        root: Repository root.
        language: Language profile -- used to filter by file extension.
        max_files: Cap on the number of files included.
        ignore_dirs: Directory names to skip entirely.
        ignore_file_suffixes: File suffixes to skip.

    Returns:
        A RepoSummary. `truncated=True` if max_files was hit.
    """
    ignored_dirs = set(ignore_dirs)
    ignored_suffixes = set(ignore_file_suffixes)

    relevant_suffixes = set(language.file_extensions) | {
        ".md",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
    }
    config_filenames = set(language.detection_files) | {
        "README.md",
        "Makefile",
        "Dockerfile",
    }

    collected: list[FileEntry] = []
    truncated = False

    for path in sorted(_walk(root, ignored_dirs)):
        if len(collected) >= max_files:
            truncated = True
            break
        rel = path.relative_to(root)
        rel_str = rel.as_posix()
        if any(rel_str.endswith(suf) for suf in ignored_suffixes):
            continue

        is_relevant = (
            path.suffix in relevant_suffixes or path.name in config_filenames
        )
        if not is_relevant:
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue
        collected.append(FileEntry(path=rel_str, size_bytes=size))

    return RepoSummary(
        root=root,
        language=language,
        files=tuple(collected),
        truncated=truncated,
    )


def _walk(root: Path, ignored_dirs: set[str]) -> Iterable[Path]:
    """Recursively yield files under root, skipping ignored_dirs."""
    if not root.is_dir():
        return
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in ignored_dirs:
                    continue
                stack.append(entry)
            elif entry.is_file():
                yield entry
