"""Project discovery: find Cascade-enabled repos in a workspace.

A "project" is any directory containing a cascade.yaml file. Studio walks
the configured workspace root to surface them in the dashboard.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ...config import load_config
from ...exceptions import CascadeError
from ...languages import detect_language


logger = logging.getLogger(__name__)


# How deep to walk looking for cascade.yaml files. Most monorepos use
# the root, so 3 levels is plenty. Past 3 we'd start picking up vendor
# directories and other noise.
DEFAULT_MAX_DEPTH = 3

# Directory names to skip during discovery (vendored deps, build output,
# version control internals, etc.).
SKIP_DIRS: tuple[str, ...] = (
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "target",
    ".gradle",
    ".idea",
    ".vscode",
    "vendor",
)


@dataclass(frozen=True)
class DiscoveredProject:
    """A Cascade-enabled project found during discovery."""

    id: str  # stable across runs (hash of absolute path)
    name: str  # directory name
    absolute_path: Path
    language: Optional[str]  # detected language, or None if unknown
    transcript_count: int
    story_batch_count: int


def project_id_for_path(path: Path) -> str:
    """Deterministic short ID derived from the absolute path.

    Same project always gets the same ID across runs, so frontend
    bookmarks and URLs stay stable.
    """
    h = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return h[:12]


def discover_projects(
    workspace_root: Path,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[DiscoveredProject]:
    """Walk workspace_root looking for cascade.yaml files.

    Each directory containing a cascade.yaml becomes a DiscoveredProject.
    Results are sorted by name for stable UI ordering.

    Args:
        workspace_root: Top directory to scan.
        max_depth: Recursion depth limit (0 = only check workspace_root itself).

    Returns:
        A list of DiscoveredProject, sorted by project name.

    Raises:
        CascadeError: If workspace_root doesn't exist or isn't a directory.
    """
    if not workspace_root.exists():
        raise CascadeError(
            f"Workspace root does not exist: {workspace_root}",
            hint=[
                "Pass an existing directory via --workspace flag",
                "Or set STUDIO_WORKSPACE_ROOT environment variable",
                "Or run cascade ui from inside a directory containing your projects",
            ],
        )
    if not workspace_root.is_dir():
        raise CascadeError(f"Workspace root is not a directory: {workspace_root}")

    found: list[DiscoveredProject] = []
    seen_paths: set[Path] = set()
    for cascade_yaml in _walk_for_cascade_yaml(workspace_root, max_depth):
        project_root = cascade_yaml.parent.resolve()
        if project_root in seen_paths:
            continue
        seen_paths.add(project_root)
        try:
            found.append(_build_project(project_root))
        except Exception as exc:  # don't let one bad project break discovery
            logger.warning(
                "discovery.skip", extra={"path": str(project_root), "reason": str(exc)}
            )

    found.sort(key=lambda p: p.name.lower())
    return found


def get_project(project_id: str, workspace_root: Path) -> Optional[DiscoveredProject]:
    """Look up a single project by its stable ID.

    Returns None if no project with that ID is found in the workspace.
    """
    for p in discover_projects(workspace_root):
        if p.id == project_id:
            return p
    return None


def _walk_for_cascade_yaml(
    root: Path, max_depth: int
) -> Iterable[Path]:
    """Yield every cascade.yaml file at or under root, up to max_depth."""
    # If root itself has cascade.yaml, yield it
    candidate = root / "cascade.yaml"
    if candidate.is_file():
        yield candidate
        # If a project is found at root, we still might want to scan deeper
        # (monorepo case), so we continue.

    if max_depth <= 0:
        return

    try:
        entries = list(root.iterdir())
    except (OSError, PermissionError):
        return

    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name in SKIP_DIRS or entry.name.startswith("."):
            continue
        yield from _walk_for_cascade_yaml(entry, max_depth - 1)


def _build_project(project_root: Path) -> DiscoveredProject:
    """Construct a DiscoveredProject from a directory that contains
    cascade.yaml."""
    # Best-effort language detection; ignore errors
    try:
        profile = detect_language(project_root)
        language_name = profile.name if profile else None
    except Exception:
        language_name = None

    transcripts_dir = project_root / "transcripts"
    transcript_count = (
        len(list(transcripts_dir.glob("*.yaml"))) if transcripts_dir.is_dir() else 0
    )

    stories_dir = project_root / "stories"
    story_batch_count = (
        len(list(stories_dir.glob("*.yaml"))) if stories_dir.is_dir() else 0
    )

    return DiscoveredProject(
        id=project_id_for_path(project_root),
        name=project_root.name,
        absolute_path=project_root,
        language=language_name,
        transcript_count=transcript_count,
        story_batch_count=story_batch_count,
    )
