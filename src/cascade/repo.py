"""Local git operations and GitHub PR creation.

Wraps the two side-effectful subsystems Cascade needs at the end of the
pipeline:
- Local git: create branch, write files, commit, push
- GitHub: open a PR via the REST API

Both are deliberately small and explicit. We do NOT use a heavyweight
git library; we shell out to the `git` CLI because every user has it and
the operations we need are simple.

Security model:
- Never run a git command with user-supplied content in the argv except
  for branch names, commit messages, and file paths -- all of which we
  control or sanitize.
- GitHub credentials come from a token (env var or constructor arg),
  never embedded in URLs.
"""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Protocol

from .exceptions import CascadeRepoError
from .plan_schemas import CodeChange, FileAction, PullRequestRef, TestResult
from .schemas import Story


logger = logging.getLogger(__name__)


# Branch names must be safe for git and for URLs. The slug component
# is restricted to [a-zA-Z0-9._-] -- note '/' is NOT allowed here even
# though it is legal in git refs, because safe_branch_name builds the
# multi-segment path explicitly; a slash inside the slug would create
# an extra path component (and has produced trailing empty segments in
# the past, which git rejects with "refusing to lock ref ... ends in /").
_SAFE_BRANCH_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")
_REPEATED_DASHES = re.compile(r"-+")
_REPEATED_DOTS = re.compile(r"\.+")


# ----------------------------------------------------------------------------
# Local git
# ----------------------------------------------------------------------------


def safe_branch_name(story_id: str, title: str, prefix: str = "cascade") -> str:
    """Build a git-safe branch name from a story.

    Format: ``<prefix>/<story_id>/<slug-from-title>``

    The slug is normalized so the full ref satisfies ``git
    check-ref-format``: no '/' or whitespace inside the slug, no
    consecutive '..' (collapsed), and no leading or trailing '.' / '-'
    (stripped). A title that normalizes to an empty string falls back
    to the literal ``story``.
    """
    slug = title.lower().strip()
    slug = _SAFE_BRANCH_CHARS.sub("-", slug)
    # git refs reject '..' (used by revision syntax) and leading/trailing
    # '.' (reserved). Collapse runs of dashes too, to avoid the
    # cosmetically-ugly '--' you get when adjacent punctuation in the
    # title each substitutes to a dash (e.g. ': ' -> '--').
    slug = _REPEATED_DASHES.sub("-", slug)
    slug = _REPEATED_DOTS.sub(".", slug)
    slug = slug.strip("-.")[:50].rstrip("-.") or "story"
    return f"{prefix}/{story_id}/{slug}"


def _run_git(
    repo_root: Path, *args: str, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess:
    """Run a git command in the given repo root."""
    if not shutil.which("git"):
        raise CascadeRepoError("git is not installed or not on PATH")
    argv = ["git", *args]
    logger.debug("git", extra={"argv": shlex.join(argv), "cwd": str(repo_root)})
    result = subprocess.run(
        argv,
        cwd=repo_root,
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise CascadeRepoError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def current_branch(repo_root: Path) -> str:
    """Return the currently checked-out branch name."""
    res = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    return res.stdout.strip()


def ensure_clean_working_tree(repo_root: Path) -> None:
    """Raise if the working tree has uncommitted changes."""
    res = _run_git(repo_root, "status", "--porcelain")
    _check_clean(res)


def _check_clean(res) -> None:
    if res.stdout.strip():
        raise CascadeRepoError(
            "Working tree has uncommitted changes",
            hint=[
                "Commit your work first: git add -A && git commit -m '...'",
                "Or stash it temporarily: git stash",
                "Or discard it (DESTRUCTIVE): git restore .",
            ],
            learn_more=(
                "Cascade refuses to run on a dirty tree because generated "
                "code could be silently mixed with your in-progress work."
            ),
        )


def create_branch(repo_root: Path, branch: str, base: str = "main") -> None:
    """Create and check out a new branch from `base`."""
    _run_git(repo_root, "fetch", "origin", base, check=False)
    _run_git(repo_root, "checkout", "-b", branch, f"origin/{base}", check=False)
    # Fallback for repos without a configured upstream
    if current_branch(repo_root) != branch:
        _run_git(repo_root, "checkout", "-b", branch)


def apply_code_change(repo_root: Path, change: CodeChange) -> list[Path]:
    """Apply a CodeChange to the working tree.

    Creates/overwrites files for create+modify actions, deletes files for
    delete actions. Returns the list of paths actually touched.
    """
    touched: list[Path] = []
    for fc in change.files:
        target = repo_root / fc.path
        if fc.action == FileAction.DELETE:
            if target.exists():
                target.unlink()
                touched.append(target)
            continue
        # create or modify
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fc.content or "", encoding="utf-8")
        touched.append(target)
    return touched


def stage_and_commit(
    repo_root: Path, paths: list[Path], message: str
) -> Optional[str]:
    """Stage the listed paths and commit.

    Returns the new commit SHA, or None if there was nothing to commit
    (e.g. all 'changes' were no-ops).
    """
    if not paths:
        return None
    rel = [p.relative_to(repo_root).as_posix() for p in paths]
    _run_git(repo_root, "add", "--", *rel)

    # Only commit if something is actually staged
    status = _run_git(repo_root, "status", "--porcelain")
    if not status.stdout.strip():
        return None

    _run_git(repo_root, "commit", "-m", message)
    sha = _run_git(repo_root, "rev-parse", "HEAD").stdout.strip()
    return sha


def push_branch(repo_root: Path, branch: str, remote: str = "origin") -> None:
    """Push the branch to the remote, setting upstream tracking."""
    _run_git(repo_root, "push", "--set-upstream", remote, branch)


def detect_github_repo(repo_root: Path) -> tuple[str, str]:
    """Parse the origin remote URL into (owner, repo).

    Supports both HTTPS and SSH forms:
        https://github.com/owner/repo(.git)
        git@github.com:owner/repo(.git)
    """
    res = _run_git(repo_root, "remote", "get-url", "origin")
    url = res.stdout.strip()
    return _parse_github_url(url)


def _parse_github_url(url: str) -> tuple[str, str]:
    https = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if https:
        return https.group(1), https.group(2)
    ssh = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if ssh:
        return ssh.group(1), ssh.group(2)
    raise CascadeRepoError(
        f"Could not parse GitHub repo from origin URL: {url!r}"
    )


# ----------------------------------------------------------------------------
# GitHub PR creation
# ----------------------------------------------------------------------------


class GitHubClient(Protocol):
    """Minimal interface we need from a GitHub client.

    Implemented for real by PyGithubClient (lazy-imports PyGithub).
    Tests pass a fake that records calls.
    """

    def open_pull_request(
        self,
        *,
        owner: str,
        repo: str,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestRef: ...


class PyGithubClient:
    """Default GitHubClient implementation using the PyGithub library."""

    def __init__(self, *, token: str):
        try:
            from github import Github
        except ImportError as exc:  # pragma: no cover
            raise CascadeRepoError(
                "PyGithub not installed. Run: pip install PyGithub"
            ) from exc
        self._gh = Github(token)

    def open_pull_request(
        self,
        *,
        owner: str,
        repo: str,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestRef:
        try:
            repo_obj = self._gh.get_repo(f"{owner}/{repo}")
            pr = repo_obj.create_pull(title=title, body=body, head=head, base=base)
        except Exception as exc:
            raise CascadeRepoError(
                f"Failed to open PR for {owner}/{repo}: {exc}"
            ) from exc
        return PullRequestRef(
            number=pr.number,
            url=pr.html_url,
            branch=head,
            title=title,
        )


def build_pr_body(
    *, story: Story, change: CodeChange, test_result: TestResult
) -> str:
    """Render the PR description from the story, code change, and test result."""
    lines: list[str] = []
    lines.append(f"## {story.title}")
    lines.append("")
    lines.append(story.description)
    lines.append("")
    lines.append("### Acceptance criteria")
    for ac in story.acceptance_criteria:
        lines.append(f"- {ac.as_text()}")
    lines.append("")
    lines.append("### Changes")
    for fc in change.files:
        lines.append(f"- **{fc.action.value}** `{fc.path}` -- {fc.reason}")
    lines.append("")
    lines.append("### Tests")
    lines.append(f"- Command: `{test_result.command}`")
    lines.append(
        f"- Result: {'PASSED' if test_result.passed else 'FAILED'} "
        f"(exit {test_result.exit_code}, {test_result.duration_seconds:.1f}s)"
    )
    if test_result.summary:
        lines.append(f"- Summary: {test_result.summary}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"_Generated by [Cascade](https://github.com/Thinknext-Software-Solutions/Cascade) "
        f"from story `{story.id}` (source meeting `{story.source_meeting_id}`). "
        "Review carefully before merge._"
    )
    return "\n".join(lines)
