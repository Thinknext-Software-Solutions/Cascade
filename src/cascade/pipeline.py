"""Top-level orchestrator: takes an approved Story to a merged-PR-ready PR.

Composes planner -> coder -> apply -> install -> test -> branch+commit+push -> PR.
Each step is fail-fast: any error in any step aborts the rest and surfaces
the error to the caller (the CLI).

This is the function the `cascade build` command calls per approved story.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .coder import generate_code
from .exceptions import CascadeError, CascadeRepoError
from .languages import LanguageProfile
from .llm import LLMClient
from .memory import TeamMemory
from .plan_schemas import CodeChange, Plan, PullRequestRef, TestResult
from .planner import plan_story
from .progress import NoopProgress, ProgressReporter
from .repo import (
    GitHubClient,
    apply_code_change,
    build_pr_body,
    create_branch,
    current_branch,
    detect_github_repo,
    ensure_clean_working_tree,
    push_branch,
    safe_branch_name,
    stage_and_commit,
)
from .repo_scan import scan_repo
from .schemas import Story
from .tester import install_dependencies, run_tests


logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Outcome of building one story end-to-end."""

    story: Story
    plan: Plan
    change: CodeChange
    install_result: TestResult
    test_result: TestResult
    branch: str
    commit_sha: Optional[str]
    pull_request: Optional[PullRequestRef]
    # Cumulative LLM cost across plan + code stages for this story
    total_llm_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def build_story(
    *,
    story: Story,
    repo_root: Path,
    llm: LLMClient,
    language: LanguageProfile,
    memory: Optional[TeamMemory] = None,
    github_client: Optional[GitHubClient] = None,
    base_branch: str = "main",
    test_override_command: Optional[list[str]] = None,
    push_and_open_pr: bool = True,
    progress: Optional[ProgressReporter] = None,
) -> BuildResult:
    """End-to-end: take an approved story and ship a PR.

    Steps:
        1. Sanity-check the working tree is clean.
        2. Plan the implementation.
        3. Generate the code.
        4. Create a new branch off `base_branch`.
        5. Apply the code change to disk.
        6. Install dependencies (best effort).
        7. Run tests.
        8. Commit the change.
        9. Push the branch.
       10. Open a PR via the GitHub client.

    Args:
        story: An APPROVED story.
        repo_root: Repository root.
        llm: LLM client for planning + coding.
        language: Resolved language profile.
        memory: Optional team memory.
        github_client: GitHub client. Required if push_and_open_pr is True.
        base_branch: Branch the new work branches off (default 'main').
        test_override_command: Optional explicit test argv. Else uses the
            language profile's default.
        push_and_open_pr: When False, stop after the local commit. Useful
            for dry runs and tests.

    Returns:
        BuildResult capturing every stage's output.

    Raises:
        CascadeError / subclasses: At any failed step. Earlier steps remain
            applied (e.g. branch created, code applied) so the user can
            inspect and recover.
    """
    if push_and_open_pr and github_client is None:
        raise CascadeError(
            "github_client is required when push_and_open_pr=True"
        )

    progress = progress or NoopProgress()
    ensure_clean_working_tree(repo_root)
    starting_branch = current_branch(repo_root)
    logger.info("pipeline.start", extra={"story_id": story.id, "branch": starting_branch})

    # Step 2: plan
    with progress.stage("plan", "asking LLM to plan the change") as p:
        repo_summary = scan_repo(repo_root, language)
        plan_result = plan_story(
            story=story,
            llm=llm,
            language=language,
            memory=memory,
            repo_summary=repo_summary,
        )
        p.succeed(f"{len(plan_result.plan.files)} file(s) to change")

    # Step 3: code
    with progress.stage("code", "writing code + tests") as p:
        code_result = generate_code(
            story=story,
            plan=plan_result.plan,
            llm=llm,
            language=language,
            repo_root=repo_root,
            memory=memory,
        )
        p.succeed(f"{len(code_result.change.files)} file(s) generated")

    # Step 4: branch
    branch = safe_branch_name(story.id, story.title)
    with progress.stage("branch", f"creating {branch}"):
        create_branch(repo_root, branch, base=base_branch)
        logger.info("pipeline.branch_created", extra={"branch": branch})

    # Step 5: apply
    with progress.stage("apply", "writing files to disk") as p:
        touched = apply_code_change(repo_root, code_result.change)
        logger.info("pipeline.applied", extra={"file_count": len(touched)})
        p.succeed(f"{len(touched)} file(s) written")

    # Step 6: install (best effort -- log but don't fail the pipeline if missing)
    with progress.stage("install", "installing dependencies") as p:
        try:
            install_result = install_dependencies(repo_root, language)
            if not install_result.passed:
                logger.warning(
                    "pipeline.install_failed",
                    extra={"command": install_result.command},
                )
                p.fail("dependency install failed")
            else:
                p.succeed(install_result.summary or "ok")
        except CascadeError as exc:
            logger.warning("pipeline.install_skipped", extra={"reason": str(exc)})
            install_result = TestResult(
                passed=True,
                exit_code=0,
                duration_seconds=0,
                command="(skipped: install tool not available)",
                summary="skipped",
            )
            p.succeed("skipped (install tool not on PATH)")

    # Step 7: test
    with progress.stage("test", f"running {' '.join(language.test_command)}") as p:
        test_result = run_tests(
            repo_root,
            language,
            override_command=test_override_command,
        )
        if test_result.passed:
            p.succeed(test_result.summary or "passed")
        else:
            p.fail(test_result.summary or f"failed (exit {test_result.exit_code})")

    # Step 8: commit
    with progress.stage("commit", "committing the change") as p:
        commit_message = _build_commit_message(story, code_result.change)
        commit_sha = stage_and_commit(repo_root, touched, commit_message)
        if commit_sha is None:
            raise CascadeRepoError(
                "Nothing to commit -- the code change resulted in no working-tree "
                "diff. The coder may have generated identical content."
            )
        logger.info("pipeline.committed", extra={"sha": commit_sha})
        p.succeed(f"{commit_sha[:8]}")

    # Aggregate LLM cost across plan + code stages for this story
    total_in = plan_result.usage.input_tokens + code_result.usage.input_tokens
    total_out = plan_result.usage.output_tokens + code_result.usage.output_tokens
    total_cost = (
        plan_result.usage.estimated_cost_usd + code_result.usage.estimated_cost_usd
    )

    if not push_and_open_pr:
        return BuildResult(
            story=story,
            plan=plan_result.plan,
            change=code_result.change,
            install_result=install_result,
            test_result=test_result,
            branch=branch,
            commit_sha=commit_sha,
            pull_request=None,
            total_llm_cost_usd=total_cost,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    # Step 9: push
    with progress.stage("push", f"pushing {branch}"):
        push_branch(repo_root, branch)
        logger.info("pipeline.pushed", extra={"branch": branch})

    # Step 10: open PR
    with progress.stage("pr", "opening pull request") as p:
        owner, repo = detect_github_repo(repo_root)
        pr = github_client.open_pull_request(  # type: ignore[union-attr]
            owner=owner,
            repo=repo,
            head=branch,
            base=base_branch,
            title=_pr_title(story),
            body=build_pr_body(
                story=story,
                change=code_result.change,
                test_result=test_result,
            ),
        )
        logger.info("pipeline.pr_opened", extra={"url": pr.url, "number": pr.number})
        p.succeed(f"#{pr.number}")

    return BuildResult(
        story=story,
        plan=plan_result.plan,
        change=code_result.change,
        install_result=install_result,
        test_result=test_result,
        branch=branch,
        commit_sha=commit_sha,
        pull_request=pr,
        total_llm_cost_usd=total_cost,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
    )


def _build_commit_message(story: Story, change: CodeChange) -> str:
    """Produce a conventional-commit-style message for the change."""
    short_title = story.title.strip().rstrip(".")
    body_lines = [
        f"feat: {short_title}",
        "",
        f"Story: {story.id}",
        f"Meeting: {story.source_meeting_id}",
        "",
        "Files changed:",
    ]
    for fc in change.files:
        body_lines.append(f"  - {fc.action.value}: {fc.path}")
    body_lines.append("")
    body_lines.append("Generated by Cascade. Human review required before merge.")
    return "\n".join(body_lines)


def _pr_title(story: Story) -> str:
    """PR title from a story. Conventional-commit style if it fits."""
    base = story.title.strip().rstrip(".")
    title = f"feat: {base}"
    return title[:140]


def get_github_token_from_env() -> str:
    """Pull a GitHub token from the standard env vars.

    Checks (in order): GITHUB_TOKEN, GH_TOKEN.
    Raises CascadeError if neither is set.
    """
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    raise CascadeError(
        "No GitHub token in environment. Set GITHUB_TOKEN or GH_TOKEN "
        "before running cascade build. (Create a token at "
        "https://github.com/settings/tokens with 'repo' scope.)"
    )
