"""Team memory layer.

Loads, validates, and formats the contents of the team-memory/ directory
for inclusion in LLM calls. This is the substrate that makes Cascade
"team-aware" -- conventions, decisions, glossary, prior work, and constraints
that every AI stage reads as grounding context.

Design notes:
- Files are plain markdown so humans can edit them freely
- Loading is permissive (missing files are tolerated, but at least one
  must exist or we warn)
- Formatting truncates politely if total chars would exceed budget
- v0.1: simple full-file inclusion. v0.2+: smart retrieval via embeddings
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .exceptions import CascadeMemoryError


# The 5 known memory files. Order matters: it's the order they're presented
# to the LLM, which roughly corresponds to importance for most decisions.
KNOWN_MEMORY_FILES: tuple[str, ...] = (
    "conventions.md",
    "decisions.md",
    "constraints.md",
    "glossary.md",
    "prior-work.md",
)


@dataclass(frozen=True)
class MemoryFile:
    """A single loaded memory file."""

    name: str  # e.g. "conventions.md"
    content: str  # the raw markdown
    path: Path  # filesystem path it was loaded from

    @property
    def is_empty(self) -> bool:
        """True if the file only contains the starter-template scaffolding.

        We detect this heuristically: a file is considered 'empty' if it has
        no non-comment, non-header substantive content. This matters because
        Cascade should not feed essentially-empty memory to the LLM (wastes
        context window and may confuse the model).
        """
        substantive_lines = [
            line.strip()
            for line in self.content.splitlines()
            if line.strip()
            and not line.strip().startswith("#")
            and not line.strip().startswith(">")
            and not line.strip().startswith("```")
            and "*This is your team's file" not in line
            and "*Add your team's" not in line
            and "*Add summaries" not in line
        ]
        # Less than 50 chars of real content = effectively empty
        return sum(len(s) for s in substantive_lines) < 50


class TeamMemory:
    """Loaded team memory for a repo.

    Use TeamMemory.load(...) rather than the constructor directly.
    """

    def __init__(self, files: list[MemoryFile], root: Path):
        self._files = files
        self._root = root

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "TeamMemory":
        """Load all known team-memory files from the given root.

        Args:
            root: Path to the team-memory directory. If None, uses
                  ./team-memory in the current working directory.

        Returns:
            A TeamMemory instance. May be empty if no files were found.

        Raises:
            CascadeMemoryError: If the directory exists but is unreadable.
        """
        if root is None:
            root = Path.cwd() / "team-memory"

        if not root.exists():
            # No team memory directory at all -- legal, but the user should
            # know it's improving their AI output by creating one.
            return cls(files=[], root=root)

        if not root.is_dir():
            raise CascadeMemoryError(f"{root} exists but is not a directory")

        loaded: list[MemoryFile] = []
        for name in KNOWN_MEMORY_FILES:
            file_path = root / name
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise CascadeMemoryError(
                    f"Could not read team-memory file {file_path}: {exc}"
                ) from exc
            loaded.append(MemoryFile(name=name, content=content, path=file_path))

        return cls(files=loaded, root=root)

    @property
    def files(self) -> list[MemoryFile]:
        return list(self._files)

    @property
    def non_empty_files(self) -> list[MemoryFile]:
        """Files that contain substantive content (not just templates)."""
        return [f for f in self._files if not f.is_empty]

    @property
    def is_meaningfully_populated(self) -> bool:
        """True if there's enough team memory to actually help the AI."""
        return len(self.non_empty_files) >= 1

    def as_llm_context(self, max_chars: int = 20_000) -> str:
        """Format the loaded memory for inclusion in an LLM prompt.

        Args:
            max_chars: Maximum total characters to include. If memory exceeds
                this, each file is truncated proportionally with a clear
                truncation marker.

        Returns:
            A single string ready to drop into a prompt template. Empty
            string if no non-empty files exist.
        """
        usable = self.non_empty_files
        if not usable:
            return ""

        # Headline for the section
        header = (
            "# Team memory (read carefully -- this captures what the team has "
            "decided, prefers, and uses)\n\n"
        )

        # Compute per-file budget. Reserve some for headers/separators.
        overhead_per_file = 100  # rough estimate for "## filename\n\n" and separators
        budget_for_content = max_chars - len(header) - overhead_per_file * len(usable)
        per_file_budget = max(500, budget_for_content // len(usable))

        sections = [header]
        for f in usable:
            section_header = f"## {f.name}\n\n"
            content = f.content
            if len(content) > per_file_budget:
                truncate_to = per_file_budget - 100
                content = (
                    content[:truncate_to]
                    + f"\n\n[...truncated, {len(f.content) - truncate_to} chars omitted...]"
                )
            sections.append(section_header + content + "\n\n---\n\n")

        result = "".join(sections)
        # Final safety clamp -- never exceed max_chars even if math was off
        if len(result) > max_chars:
            result = result[: max_chars - 30] + "\n\n[...truncated...]"
        return result
