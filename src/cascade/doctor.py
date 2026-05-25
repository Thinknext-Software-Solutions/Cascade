"""cascade doctor: end-to-end health check for a Cascade installation.

Runs ~12 checks across environment, project config, providers, git, and
test runner. Each check reports green / yellow / red plus an actionable
suggestion when it fails. Designed to be the first thing a user runs
after `pip install cascade-agent` so they know everything works before
their first real build.

Inspired by `gh auth status`, `rustup show`, and `homebrew doctor` -- a
diagnostic conversation, not a stack trace.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from .config import load_config
from .exceptions import CascadeError
from .languages import detect_language, resolve_language
from .memory import TeamMemory
from .user_config import (
    LLM_ENV_KEY_NAMES,
    VCS_ENV_TOKEN_NAMES,
    load_user_config,
)


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one diagnostic check."""

    name: str
    status: CheckStatus
    message: str
    suggestion: Optional[str] = None  # actionable hint when status != OK


# ----------------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------------


def check_python_version() -> CheckResult:
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 11:
        return CheckResult(
            name="Python version",
            status=CheckStatus.OK,
            message=f"{major}.{minor} (>=3.11 required)",
        )
    return CheckResult(
        name="Python version",
        status=CheckStatus.FAIL,
        message=f"{major}.{minor} -- Cascade requires Python 3.11+",
        suggestion="Install Python 3.11 or later. On macOS: brew install python@3.12",
    )


def check_optional_extras() -> CheckResult:
    """Report which optional [extras] are installed."""
    extras = {
        "openai": "openai",
        "google": "google.genai",
        "claude_code": "claude_agent_sdk",
        "gitlab": "gitlab",
        "jira": "atlassian",
        "studio": "fastapi",
    }
    installed = []
    missing = []
    for extra, module in extras.items():
        try:
            importlib.import_module(module)
            installed.append(extra)
        except ImportError:
            missing.append(extra)
    if not missing:
        return CheckResult(
            name="Optional extras",
            status=CheckStatus.OK,
            message=f"all installed ({', '.join(installed)})",
        )
    return CheckResult(
        name="Optional extras",
        status=CheckStatus.WARN,
        message=f"installed: {', '.join(installed) or 'none'} -- missing: {', '.join(missing)}",
        suggestion=f"To enable everything: pip install cascade-agent[all]",
    )


def check_git_installed() -> CheckResult:
    if shutil.which("git"):
        try:
            v = subprocess.run(
                ["git", "--version"], capture_output=True, text=True, check=True
            ).stdout.strip()
            return CheckResult(name="git", status=CheckStatus.OK, message=v)
        except (subprocess.CalledProcessError, OSError):
            pass
    return CheckResult(
        name="git",
        status=CheckStatus.FAIL,
        message="not on PATH",
        suggestion="Install git from https://git-scm.com/downloads",
    )


def check_git_repo(repo_root: Path) -> CheckResult:
    if not (repo_root / ".git").exists():
        return CheckResult(
            name="Git repository",
            status=CheckStatus.FAIL,
            message=f"{repo_root} is not a git repository",
            suggestion="cd into a git repo, or run: git init",
        )
    # `git rev-parse HEAD` fails if there are no commits yet; use
    # `symbolic-ref --short HEAD` which works on a fresh repo too.
    try:
        branch = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        return CheckResult(
            name="Git repository",
            status=CheckStatus.WARN,
            message=f"could not determine current branch: {exc}",
        )
    return CheckResult(
        name="Git repository",
        status=CheckStatus.OK,
        message=f"on branch '{branch}'",
    )


def check_git_remote(repo_root: Path) -> CheckResult:
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return CheckResult(
            name="Git origin",
            status=CheckStatus.OK,
            message=url,
        )
    except subprocess.CalledProcessError:
        return CheckResult(
            name="Git origin",
            status=CheckStatus.WARN,
            message="no 'origin' remote configured",
            suggestion="Add one: git remote add origin <url>. Cascade can build locally without it but can't push or open PRs.",
        )


def check_project_config(repo_root: Path) -> CheckResult:
    cfg_path = repo_root / "cascade.yaml"
    if not cfg_path.exists():
        return CheckResult(
            name="cascade.yaml",
            status=CheckStatus.WARN,
            message="not found (defaults will be used)",
            suggestion="Run: cascade init  to create one.",
        )
    try:
        load_config(cfg_path)
        return CheckResult(
            name="cascade.yaml",
            status=CheckStatus.OK,
            message=f"valid ({cfg_path})",
        )
    except CascadeError as exc:
        return CheckResult(
            name="cascade.yaml",
            status=CheckStatus.FAIL,
            message=f"invalid: {exc}",
            suggestion="Edit cascade.yaml or run: cascade init --force  to reset it.",
        )


def check_team_memory(repo_root: Path) -> CheckResult:
    cfg = load_config(repo_root / "cascade.yaml") if (repo_root / "cascade.yaml").exists() else None
    memory_path = repo_root / (cfg.memory.path if cfg else "team-memory")
    if not memory_path.exists():
        return CheckResult(
            name="Team memory",
            status=CheckStatus.WARN,
            message=f"{memory_path} not found",
            suggestion="Run: cascade init  to create starter files.",
        )
    try:
        memory = TeamMemory.load(memory_path)
    except CascadeError as exc:
        return CheckResult(
            name="Team memory",
            status=CheckStatus.FAIL,
            message=str(exc),
        )
    non_empty = len(memory.non_empty_files)
    if non_empty == 0:
        return CheckResult(
            name="Team memory",
            status=CheckStatus.WARN,
            message=f"{memory_path} exists but all files are template-only",
            suggestion="Edit team-memory/*.md with your team's real conventions, decisions, and constraints. The more accurate this is, the better Cascade's output.",
        )
    if non_empty < 3:
        return CheckResult(
            name="Team memory",
            status=CheckStatus.WARN,
            message=f"{non_empty}/5 files have substantive content",
            suggestion="Fill in more of the team-memory/*.md files for higher-quality output.",
        )
    return CheckResult(
        name="Team memory",
        status=CheckStatus.OK,
        message=f"{non_empty}/5 files populated",
    )


def check_language(repo_root: Path) -> CheckResult:
    cfg = load_config(repo_root / "cascade.yaml") if (repo_root / "cascade.yaml").exists() else None
    configured = cfg.language if cfg else None
    try:
        profile = resolve_language(repo_root, configured_name=configured)
    except CascadeError as exc:
        return CheckResult(
            name="Language",
            status=CheckStatus.WARN,
            message=str(exc),
            suggestion="Set 'language: <name>' in cascade.yaml. Supported: python, typescript, javascript, go, rust, java, ruby, csharp.",
        )
    source = "configured" if configured else "auto-detected"
    return CheckResult(
        name="Language",
        status=CheckStatus.OK,
        message=f"{profile.display_name} ({source})",
    )


def check_llm_provider() -> CheckResult:
    """Verify the default LLM provider has credentials available."""
    user_cfg = load_user_config()
    provider = user_cfg.defaults.llm_provider
    needs_key = bool(LLM_ENV_KEY_NAMES.get(provider))

    # Try to resolve credentials without making a network call
    from .user_config import resolve_llm_credentials

    try:
        creds = resolve_llm_credentials(user_config=user_cfg, provider=provider)
    except CascadeError as exc:
        return CheckResult(
            name=f"LLM provider ({provider})",
            status=CheckStatus.FAIL,
            message=f"{exc}",
            suggestion=f"Run: cascade configure llm {provider} --key <KEY>  (or set the env var)",
        )

    if not needs_key:
        return CheckResult(
            name=f"LLM provider ({provider})",
            status=CheckStatus.OK,
            message=f"configured (no API key required for {provider})",
        )
    return CheckResult(
        name=f"LLM provider ({provider})",
        status=CheckStatus.OK,
        message=f"key found via {creds.source}",
    )


def check_vcs_provider() -> CheckResult:
    """Verify the default VCS provider has a token."""
    user_cfg = load_user_config()
    provider = user_cfg.defaults.vcs_provider
    from .user_config import resolve_vcs_credentials

    try:
        creds = resolve_vcs_credentials(user_config=user_cfg, provider=provider)
        return CheckResult(
            name=f"VCS provider ({provider})",
            status=CheckStatus.OK,
            message=f"token found via {creds.source}",
        )
    except CascadeError as exc:
        return CheckResult(
            name=f"VCS provider ({provider})",
            status=CheckStatus.WARN,
            message=str(exc),
            suggestion=f"Run: cascade configure vcs {provider} --token <TOKEN>  (only needed when you want to open PRs)",
        )


def check_test_command(repo_root: Path) -> CheckResult:
    """Verify the language's test command exists on PATH."""
    cfg = load_config(repo_root / "cascade.yaml") if (repo_root / "cascade.yaml").exists() else None
    configured = cfg.language if cfg else None
    try:
        profile = resolve_language(repo_root, configured_name=configured)
    except CascadeError:
        return CheckResult(
            name="Test runner",
            status=CheckStatus.SKIP,
            message="(language not detected; skipping)",
        )
    exe = profile.test_command[0]
    if shutil.which(exe):
        return CheckResult(
            name="Test runner",
            status=CheckStatus.OK,
            message=f"{' '.join(profile.test_command)} (found {exe})",
        )
    return CheckResult(
        name="Test runner",
        status=CheckStatus.WARN,
        message=f"'{exe}' not on PATH",
        suggestion=f"Install {exe} (Cascade runs '{' '.join(profile.test_command)}' to validate generated code).",
    )


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


def run_doctor(repo_root: Optional[Path] = None) -> list[CheckResult]:
    """Run all diagnostic checks and return the results in order."""
    repo_root = (repo_root or Path.cwd()).resolve()

    checks: list[Callable[[], CheckResult]] = [
        check_python_version,
        check_optional_extras,
        check_git_installed,
        lambda: check_git_repo(repo_root),
        lambda: check_git_remote(repo_root),
        lambda: check_project_config(repo_root),
        lambda: check_team_memory(repo_root),
        lambda: check_language(repo_root),
        check_llm_provider,
        check_vcs_provider,
        lambda: check_test_command(repo_root),
    ]

    return [check() for check in checks]


def summarize(results: list[CheckResult]) -> tuple[int, int, int, int]:
    """Count results by status. Returns (ok, warn, fail, skip)."""
    counts = {CheckStatus.OK: 0, CheckStatus.WARN: 0, CheckStatus.FAIL: 0, CheckStatus.SKIP: 0}
    for r in results:
        counts[r.status] += 1
    return counts[CheckStatus.OK], counts[CheckStatus.WARN], counts[CheckStatus.FAIL], counts[CheckStatus.SKIP]
