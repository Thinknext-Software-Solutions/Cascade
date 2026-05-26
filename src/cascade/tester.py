"""Tester stage: run language-appropriate tests via subprocess.

Wraps the language profile's test_command in a subprocess call, captures
output, and returns a structured TestResult. Optionally runs the install
command first if dependencies aren't yet present.

Security model:
- We only run commands defined by the LanguageProfile (or an explicit
  override from CascadeConfig.test_command). We do NOT shell-interpolate
  LLM output into the command, so prompt-injection cannot escalate.
- Commands are run with shell=False and argv as a list, never as a string.
- Working directory is explicit; we never rely on the parent's cwd.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

from .exceptions import CascadeError
from .languages import LanguageProfile
from .plan_schemas import TestResult


logger = logging.getLogger(__name__)


def run_tests(
    repo_root: Path,
    language: LanguageProfile,
    *,
    override_command: Optional[Sequence[str]] = None,
    timeout_seconds: float = 600.0,
    env: Optional[dict[str, str]] = None,
) -> TestResult:
    """Run the test suite for the resolved language.

    Args:
        repo_root: Repository working directory.
        language: Language profile whose test_command to use.
        override_command: Explicit argv to use instead of the profile default.
            Comes from CascadeConfig.test_command when set.
        timeout_seconds: Subprocess timeout. Kills the run on overrun and
            returns a TestResult with passed=False.
        env: Optional environment variable overrides.

    Returns:
        A TestResult.

    Raises:
        CascadeError: If the test executable can't be found at all
            (the user hasn't installed the language's tooling).
    """
    argv = list(override_command) if override_command else list(language.test_command)
    if not argv:
        raise CascadeError(
            f"No test command configured for language {language.name}"
        )

    if not _executable_available(argv[0]):
        install_hint = {
            "pytest": "pip install pytest",
            "npx": "Install Node.js: https://nodejs.org",
            "go": "Install Go: https://go.dev/dl",
            "cargo": "Install Rust: https://rustup.rs",
            "mvn": "Install Maven: https://maven.apache.org",
            "bundle": "Install Ruby + Bundler: gem install bundler",
            "dotnet": "Install .NET SDK: https://dotnet.microsoft.com/download",
        }.get(argv[0], f"Install '{argv[0]}' first")
        raise CascadeError(
            f"Test executable '{argv[0]}' not found on PATH",
            hint=[
                install_hint,
                "Or override the test command in cascade.yaml:\ntest_command: <your-command>",
                "Or run cascade build with --no-pr first to verify code generation without running tests",
            ],
        )

    logger.info(
        "tester.start",
        extra={
            "command": shlex.join(argv),
            "cwd": str(repo_root),
            "timeout": timeout_seconds,
        },
    )

    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        logger.warning("tester.timeout", extra={"duration": duration})
        return TestResult(
            passed=False,
            exit_code=-1,
            duration_seconds=duration,
            stdout=stdout,
            stderr=stderr + f"\n\n[tester] timed out after {timeout_seconds}s",
            command=shlex.join(argv),
            summary=f"timed out after {timeout_seconds:.0f}s",
        )
    except OSError as exc:
        raise CascadeError(
            f"Failed to run test command {shlex.join(argv)}: {exc}"
        ) from exc

    duration = time.monotonic() - started
    passed = completed.returncode == 0

    result = TestResult(
        passed=passed,
        exit_code=completed.returncode,
        duration_seconds=duration,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=shlex.join(argv),
        summary=_summarize(completed.stdout, completed.stderr, passed),
    )

    logger.info(
        "tester.done",
        extra={
            "passed": passed,
            "exit_code": completed.returncode,
            "duration": duration,
        },
    )

    return result


def install_dependencies(
    repo_root: Path,
    language: LanguageProfile,
    *,
    timeout_seconds: float = 600.0,
) -> TestResult:
    """Run the language's install command (e.g. pip install, npm install).

    Returns a TestResult for uniform handling -- a non-zero exit code here
    means the build can't proceed.

    Skipped silently (returning a passing result) if the language profile
    has no install_command defined.
    """
    if language.install_command is None:
        return TestResult(
            passed=True,
            exit_code=0,
            duration_seconds=0,
            command="(no install command for this language)",
            summary="skipped",
        )

    argv = list(language.install_command)
    if not _executable_available(argv[0]):
        raise CascadeError(
            f"Install executable '{argv[0]}' not found on PATH."
        )

    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            exit_code=-1,
            duration_seconds=time.monotonic() - started,
            command=shlex.join(argv),
            summary=f"install timed out after {timeout_seconds:.0f}s",
        )

    duration = time.monotonic() - started
    return TestResult(
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        duration_seconds=duration,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=shlex.join(argv),
        summary=("install ok" if completed.returncode == 0 else "install failed"),
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _executable_available(name: str) -> bool:
    """Check whether an executable exists on PATH."""
    return shutil.which(name) is not None


def _summarize(stdout: str, stderr: str, passed: bool) -> str:
    """Heuristic one-line summary from test output.

    We don't try to parse every framework's output format. We surface:
    - "passed" or "failed" headline
    - The first informative line from stdout that contains pass/fail counts
    """
    headline = "passed" if passed else "failed"
    text = (stdout or "") + "\n" + (stderr or "")
    # Look for lines that smell like a summary
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped:
            continue
        # pytest: "==== 42 passed in 0.18s ===="
        # jest:   "Tests:       1 failed, 41 passed, 42 total"
        # go:     "PASS" / "FAIL"
        # cargo:  "test result: ok. 12 passed; 0 failed"
        if any(
            sig in lower
            for sig in (
                "passed",
                "failed",
                "test result",
                "tests:",
                "ok.",
                "fail",
            )
        ):
            return f"{headline}: {stripped[:140]}"
    return headline
